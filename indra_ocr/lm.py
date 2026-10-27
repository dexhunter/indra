"""Beam search with a token n-gram language model, for lines the recognizer is
unsure of.

The recognizer reads greedily: one best class per step.  On a line whose
confidence is under the gate, a CTC prefix beam search keeps several readings
and the language model breaks close calls between them:
score = log P_ctc + alpha * log P_lm + beta * tokens.  The recognizer's own
probabilities still decide, so a clearly written odd word survives.

lm.npz holds the model as a backoff table (no pickle): per history length n,
sorted keys of the (history, token) pairs seen in training with their
log-probabilities, and sorted history keys with their backoff weights.  A key
packs tokens ``bits`` bits each (0 = line start, t + 1 = token t): 11 unless
the file says more (a larger vocab), with order x bits at most TAG_SHIFT.
"""

from __future__ import annotations

import math

import numpy as np

BITS = 11  # per token, unless lm.npz holds "bits"
TAG_SHIFT = 61  # the history length goes in the top 3 bits
BOS = -1


def _pack(tokens, bits: int = BITS) -> int:
    k = 0
    for t in tokens:
        k = (k << bits) | (0 if t == BOS else t + 1)
    return k


class NgramLM:
    def __init__(self, path):
        z = np.load(path)
        self.order = int(z["order"])
        self.vocab_size = int(z["vocab_size"])
        self.bits = int(z["bits"]) if "bits" in z else BITS
        if self.order * self.bits > TAG_SHIFT:
            raise ValueError(f"order {self.order} x {self.bits} bits do not fit a key")
        self.k = [z[f"k{n}"] for n in range(self.order)]
        self.v = [z[f"v{n}"] for n in range(self.order)]
        self.hk = [z[f"hk{n}"] for n in range(self.order)]
        self.hb = [z[f"hb{n}"] for n in range(self.order)]
        self.cache: dict = {}
        self.ikcache: dict = {}
        # All history lengths in one sorted table, the length in the top bits
        # (keys use at most order x bits), so a step needs one search per table.
        tags = [np.uint64(n) << np.uint64(TAG_SHIFT) for n in range(self.order)]
        for name, keys, vals in (("p", self.k, self.v), ("h", self.hk, self.hb)):
            allk = np.concatenate([k | t for k, t in zip(keys, tags)])
            allv = np.concatenate(vals).astype(np.float64)
            o = np.argsort(allk)
            setattr(self, f"{name}_keys", allk[o])
            setattr(self, f"{name}_vals", allv[o])
        n = np.arange(self.order, dtype=np.uint64)
        self._tags = (n << np.uint64(TAG_SHIFT))[None, :]
        self._masks = ((np.uint64(1) << (np.uint64(self.bits) * n)) - np.uint64(1))[
            None, :
        ]

    @property
    def max_logp(self) -> float:
        """An upper bound of every value logp and logp_keys return: the largest
        log-probability plus every positive backoff weight a back-off can add."""
        if not hasattr(self, "_max_logp"):
            top = max(float(self.p_vals.max()), -math.log(self.vocab_size))
            bo = sum(max(0.0, float(h.max())) for h in self.hb if len(h))
            self._max_logp = top + bo
        return self._max_logp

    @staticmethod
    def _find(keys: np.ndarray, key: int) -> int:
        i = int(np.searchsorted(keys, np.uint64(key)))
        return i if i < len(keys) and int(keys[i]) == key else -1

    def logp(self, hist: tuple, t: int) -> float:
        ck = (hist, t)
        v = self.cache.get(ck)
        if v is not None:
            return v
        top = min(len(hist), self.order - 1)
        hs = [tuple(hist[len(hist) - n :]) if n else () for n in range(top + 1)]
        m = 0  # the longest history the model has seen
        for n in range(1, top + 1):
            if self._find(self.hk[n], _pack(hs[n], self.bits)) < 0:
                break
            m = n
        acc, v = 0.0, None
        for n in range(m, -1, -1):
            i = self._find(self.k[n], _pack(hs[n] + (t,), self.bits))
            if i >= 0:
                v = acc + float(self.v[n][i])
                break
            acc += float(self.hb[n][self._find(self.hk[n], _pack(hs[n], self.bits))])
        if v is None:
            v = acc - math.log(self.vocab_size)
        if len(self.cache) < 1_000_000:
            self.cache[ck] = v
        return v

    def logp_keys(self, keys: list[int]) -> list[float]:
        """logp for many (history, token) pairs, each given as one key: the
        last order - 1 history tokens and then the token, packed (``_pack``).
        Shorter histories are the key's low bits, so every history length
        is one vectorized search over all the pairs."""
        out = [self.ikcache.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        if not miss:
            return out
        N, P = self.order, len(miss)
        full = np.array([keys[i] for i in miss], np.uint64)[:, None]
        b = np.uint64(self.bits)
        ct = full & np.uint64((1 << self.bits) - 1)
        hq = (full >> b) & self._masks
        pq = (hq << b) | ct | self._tags
        hq = hq | self._tags
        found = []
        for table, vals, q in (
            (self.h_keys, self.h_vals, hq),
            (self.p_keys, self.p_vals, pq),
        ):
            j = np.minimum(np.searchsorted(table, q), len(table) - 1)
            hit = table[j] == q
            found.append((hit, np.where(hit, vals[j], 0.0)))
        (h_ok, hb), (p_ok, pv) = found
        h_ok[:, 0] = True
        # m: the longest history length the model has seen, per pair.  The
        # value comes from the longest seen history n <= m that saw the token,
        # plus the backoff weights of the longer seen histories (n, m]; with
        # no such n, all backoffs and a uniform floor.
        m = np.cumprod(h_ok[:, 1:], axis=1).sum(1) if N > 1 else np.zeros(P, int)
        valid = np.arange(N)[None, :] <= m[:, None]
        hitm = valid & p_ok
        has = hitm.any(1)
        chosen = N - 1 - np.argmax(hitm[:, ::-1], axis=1)
        suffix = np.zeros((P, N + 1))
        suffix[:, :N] = np.cumsum((hb * valid)[:, ::-1], axis=1)[:, ::-1]
        rows = np.arange(P)
        res = np.where(
            has,
            suffix[rows, chosen + 1] + pv[rows, chosen],
            suffix[:, 0] - math.log(self.vocab_size),
        )
        for r, i in enumerate(miss):
            out[i] = float(res[r])
            if len(self.ikcache) < 1_000_000:
                self.ikcache[keys[i]] = out[i]
        return out


def _lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if type(a) is float and type(b) is float:
        if a > b:
            if a != math.inf:
                return a + math.log(1.0 + math.exp(b - a))
        elif b != math.inf:
            return b + math.log(math.exp(a - b) + 1.0)
        return math.nan
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(
    blank: np.ndarray,
    topv: np.ndarray,
    topi: np.ndarray,
    lm: NgramLM,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam: int = 8,
    blank_skip: float = 0.999,
    cand_floor: float = 1e-3,
) -> tuple[list[int], list[int]]:
    """CTC prefix beam search over a line's steps.  blank: [T] blank
    log-probabilities; topv / topi: [T, k] the k best classes per step (blank
    included) and their log-probabilities; a class under ``cand_floor``
    probability at a step is not tried there.  Returns (token ids, the step
    where each token was first emitted)."""
    ninf = -math.inf
    bits = lm.bits
    hmask = (1 << (bits * (lm.order - 1))) - 1
    # Ordered (prefix, record) pairs. Records hold blank/non-blank log p,
    # LM score, steps, packed history and a cached CTC total.
    beams = [((), [0.0, ninf, 0.0, (), 0, 0.0])]
    log_skip = math.log(blank_skip)
    log_floor = math.log(cand_floor)
    lse = _lse

    def rank(item):
        e = item[1]
        return -(e[5] + e[2])

    steps = range(len(blank))
    # Ordinary numeric matrices can use Python scalars throughout the search.
    native_rows = (
        type(blank) is np.ndarray and blank.dtype == np.float64
        and type(lm) is NgramLM
        and type(topi) is np.ndarray and type(topv) is np.ndarray
        and topi.ndim == topv.ndim == 2
        and topi.dtype == np.int64 and topv.dtype == np.float64
        and topi.shape[0] >= len(steps) and topv.shape[0] >= len(steps)
    )
    if native_rows:
        topi, topv = topi.tolist(), topv.tolist()
        if blank.ndim == 1:
            blank = blank.tolist()

        def lse(a, b, exp=math.exp, log=math.log, inf=math.inf, ninf=ninf):
            if a == ninf:
                return b
            if b == ninf:
                return a
            if a > b:
                if a != inf:
                    return a + log(1.0 + exp(b - a))
            elif b != inf:
                return b + log(exp(a - b) + 1.0)
            return math.nan
    for t in steps:
        b0 = float(blank[t])
        if b0 > log_skip:
            for _, e in beams:
                e[0] = e[5] + b0
                e[1] = ninf
                e[5] = e[0]
            continue
        nxt: dict = {}

        if native_rows:
            cands = [
                (c, v) for c, v in zip(topi[t], topv[t])
                if c != 0 and v >= log_floor
            ]
        else:
            cands = [
                (int(c), float(v))
                for c, v in zip(topi[t], topv[t])
                if c != 0 and v >= log_floor
            ]
        # Every beam's own entries first (the blank, and a repeat that collapses):
        # their scores only grow below, so with a full beam the lowest of them is
        # a floor; a new prefix that cannot reach it never enters the beam, and
        # its LM score is not looked up (the result is the same).
        for prefix, (b, nb, s, st, hk, tot) in beams:
            last = prefix[-1] if prefix else None
            repeat = ninf
            for c, lp in cands:
                if c == last:
                    repeat = lse(repeat, nb + lp)
            new_b = tot + b0
            nxt[prefix] = [new_b, repeat, s, st, hk, new_b if repeat == ninf else lse(new_b, repeat)]
        floor = (
            min(e[5] + e[2] for e in nxt.values())
            if len(beams) >= beam
            else ninf
        )
        reach = beta + alpha * lm.max_logp
        cutoff = floor - 1e-9  # float slack
        ext = []  # (extended prefix, log p, LM score, steps, history)
        keys = []
        for prefix, (b, nb, s, st, hk, tot) in beams:
            last = prefix[-1] if prefix else None
            new_steps = None
            for c, lp in cands:
                # after a blank, the same class is a new token
                new_from = b + lp if c == last else tot + lp
                if new_from == ninf:
                    continue
                q = prefix + (c,)
                old = nxt.get(q)
                if old is not None:  # an existing beam: its LM score stays
                    old[1] = lse(old[1], new_from)
                    old[5] = lse(old[0], old[1])
                elif new_from + s + reach >= cutoff:
                    if new_steps is None:
                        new_steps = st + (t,)
                    key = (hk << bits) | (c + 1)
                    ext.append((q, new_from, s, new_steps, key & hmask))
                    keys.append(key)
        lps = lm.logp_keys(keys) if ext else []
        for (prefix, new_from, s, st, hk), lmp in zip(ext, lps):
            score = s + alpha * lmp + beta
            old = nxt.get(prefix)
            if old is None:
                nxt[prefix] = [ninf, new_from, score, st, hk, new_from]
            else:
                old[0] = lse(old[0], ninf)
                old[1] = lse(old[1], new_from)
                old[5] = lse(old[0], old[1])
        beams = sorted(nxt.items(), key=rank)[:beam]
    best = max(beams, key=lambda kv: kv[1][5] + kv[1][2])
    return list(best[0]), list(best[1][3])
