from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .utils import l2_topk
from .base import ANNIndex


@dataclass
class LSHParams:
    k: int = 10
    L: int = 20
    w: float = 4.0
    seed: int = 1


PRIME = (1 << 32) - 5


class EuclideanLSH(ANNIndex):
    def __init__(self, params: LSHParams):
        self.p = params
        self.A = np.zeros(1)
        self.B = np.zeros(1)
        self.tables: List[Dict[Tuple[int, ...], List[int]]] = []

    def build(self, base: np.ndarray) -> None:
        rng = np.random.default_rng(self.p.seed)
        n, d = base.shape
        self.A = rng.standard_normal((self.p.L, self.p.k, d), dtype=np.float32)
        self.B = rng.uniform(0.0, self.p.w, size=(self.p.L, self.p.k)).astype(np.float32)
        self.tables = [dict() for _ in range(self.p.L)]
        for i in range(n):
            x = base[i]
            for t in range(self.p.L):
                key = self._hash(t, x)
                self.tables[t].setdefault(key, []).append(i)

    def _hash(self, t: int, x: np.ndarray) -> Tuple[int, ...]:
        vals = (self.A[t] @ x + self.B[t]) / self.p.w
        h = np.floor(vals).astype(np.uint32)
        # key = 0
        # for v in h:
        #     v %= PRIME
        #     if v < 0:
        #         v += PRIME
        #     key += v
        #     key %= PRIME
        # return key
        return tuple(int(v) for v in h)

    def query(
        self,
        base: np.ndarray,
        query: np.ndarray,
        topK: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        cand = set()
        for t in range(self.p.L):
            key = self._hash(t, query)
            bucket = self.tables[t].get(key)
            if bucket:
                cand.update(bucket)
        idx = np.fromiter(cand, dtype=np.int64) if cand else np.array([], dtype=np.int64)
        return l2_topk(base, query, idx, topK)
