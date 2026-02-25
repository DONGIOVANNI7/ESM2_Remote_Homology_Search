from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple
import itertools
import numpy as np

from .base import ANNIndex
from .utils import l2_topk


@dataclass
class HypercubeParams:
    k: int = 14
    M: int = 2000
    w: float = 4.0
    probes: int = 10
    seed: int = 1


class HypercubeIndex(ANNIndex):
    def __init__(self, params: HypercubeParams):
        self.p = params
        self.R = None
        self.B = None
        self.buckets: Dict[int, List[int]] = {}

    def build(self, base: np.ndarray) -> None:
        rng = np.random.default_rng(self.p.seed)
        n, d = base.shape
        self.R = rng.standard_normal((self.p.k, d), dtype=np.float32)
        self.B = rng.uniform(0.0, self.p.w, size=self.p.k).astype(np.float32)
        self.buckets = {}
        for i in range(n):
            code = self._code(base[i])
            self.buckets.setdefault(code, []).append(i)

    def _code(self, x: np.ndarray) -> int:
        bits = (((self.R @ x + self.B) / self.p.w) >= 0).astype(np.uint8)
        code = 0
        for i, b in enumerate(bits):
            if b:
                code |= (1 << i)
        return int(code)

    def _probe_codes(self, code: int) -> Iterable[int]:
        yield code
        yielded = 1
        for dist in range(1, self.p.k + 1):
            for flips in itertools.combinations(range(self.p.k), dist):
                mask = 0
                for bit in flips:
                    mask |= (1 << bit)
                yield code ^ mask
                yielded += 1
                if yielded >= self.p.probes:
                    return

    def query(
        self,
        base: np.ndarray,
        query: np.ndarray,
        topK: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        qcode = self._code(query)
        cand: List[int] = []
        for c in self._probe_codes(qcode):
            bucket = self.buckets.get(c)
            if bucket:
                cand.extend(bucket)
                if len(cand) >= self.p.M:
                    break
        idx = np.array(cand[: self.p.M], dtype=np.int64)
        return l2_topk(base, query, idx, topK)
