from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .base import ANNIndex
from .kmeans import kmeans_lloyd
from .utils import l2_topk, sample_rows


@dataclass
class IVFParams:
    nlist: int = 2048
    nprobe: int = 10
    iters: int = 25
    train_size: int = 50000
    seed: int = 1


class IVFFlat(ANNIndex):
    def __init__(self, params: IVFParams):
        self.p = params
        self.centroids = None
        self.lists: List[np.ndarray] = []

    def build(self, base: np.ndarray) -> None:
        train = sample_rows(base, self.p.train_size, seed=self.p.seed).astype(np.float32)
        C, _ = kmeans_lloyd(train, self.p.nlist, iters=self.p.iters, seed=self.p.seed)
        self.centroids = C
        x_norm = np.einsum("ij,ij->i", base, base)[:, None]
        c_norm = np.einsum("ij,ij->i", C, C)[None, :]
        dist = x_norm + c_norm - 2.0 * (base @ C.T)
        assign = np.argmin(dist, axis=1).astype(np.int64)
        self.lists = [np.where(assign == j)[0].astype(np.int64) for j in range(self.p.nlist)]

    def _nearest_lists(self, q: np.ndarray) -> np.ndarray:
        diff = self.centroids - q
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        if self.p.nprobe >= dist_sq.size:
            order = np.argsort(dist_sq)
        else:
            order = np.argpartition(dist_sq, self.p.nprobe)[: self.p.nprobe]
            order = order[np.argsort(dist_sq[order])]
        return order.astype(np.int64)

    def query(
        self,
        base: np.ndarray,
        query: np.ndarray,
        topK: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        lists = self._nearest_lists(query)
        cand = np.concatenate([self.lists[i] for i in lists])
        return l2_topk(base, query, cand, topK)
