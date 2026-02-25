from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .kmeans import kmeans_lloyd
from .utils import l2_topk, sample_rows
from .base import ANNIndex


@dataclass
class IVFPQParams:
    nlist: int = 2048
    nprobe: int = 10
    m: int = 8
    ks: int = 256
    iters: int = 25
    train_size: int = 50000
    seed: int = 1
    refine: bool = False


class IVFPQ(ANNIndex):
    def __init__(self, params: IVFPQParams):
        self.p = params
        self.centroids = None
        self.lists: List[np.ndarray] = []
        self.codes: List[np.ndarray] = []
        self.codebooks = None
        self.d = None

    def build(self, base: np.ndarray) -> None:
        n, d = base.shape
        self.d = d
        if d % self.p.m != 0:
            raise ValueError(f"d={d} must be divisible by m={self.p.m}")
        dsub = d // self.p.m

        train = sample_rows(base, self.p.train_size, seed=self.p.seed).astype(np.float32)
        C, _ = kmeans_lloyd(train, self.p.nlist, iters=self.p.iters, seed=self.p.seed)
        self.centroids = C

        x_norm = np.einsum("ij,ij->i", base, base)[:, None]
        c_norm = np.einsum("ij,ij->i", C, C)[None, :]
        dist = x_norm + c_norm - 2.0 * (base @ C.T)
        assign = np.argmin(dist, axis=1).astype(np.int64)
        self.lists = [np.where(assign == j)[0].astype(np.int64) for j in range(self.p.nlist)]

        cb = np.empty((self.p.m, self.p.ks, dsub), dtype=np.float32)
        for si in range(self.p.m):
            sub = train[:, si*dsub:(si+1)*dsub]
            ks_eff = min(self.p.ks, sub.shape[0])
            cbi, _ = kmeans_lloyd(sub, ks_eff, iters=max(10, self.p.iters//2), seed=self.p.seed + 17*si)
            if ks_eff < self.p.ks:
                pad = np.tile(cbi[-1:], (self.p.ks - ks_eff, 1))
                cbi = np.vstack([cbi, pad])
            cb[si] = cbi[: self.p.ks]
        self.codebooks = cb

        self.codes = []
        for j in range(self.p.nlist):
            idx = self.lists[j]
            if idx.size == 0:
                self.codes.append(np.zeros((0, self.p.m), dtype=np.uint8))
                continue
            V = base[idx].astype(np.float32)
            self.codes.append(self._encode(V))

    def _encode(self, V: np.ndarray) -> np.ndarray:
        m = self.p.m
        dsub = self.d // m
        codes = np.empty((V.shape[0], m), dtype=np.uint8)
        for si in range(m):
            sub = V[:, si*dsub:(si+1)*dsub]
            cb = self.codebooks[si]
            x_norm = np.einsum("ij,ij->i", sub, sub)[:, None]
            c_norm = np.einsum("ij,ij->i", cb, cb)[None, :]
            dist = x_norm + c_norm - 2.0 * (sub @ cb.T)
            codes[:, si] = np.argmin(dist, axis=1).astype(np.uint8)
        return codes

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
        m = self.p.m
        dsub = self.d // m
        lists = self._nearest_lists(query)

        tables = []
        for si in range(m):
            qsub = query[si*dsub:(si+1)*dsub].astype(np.float32)
            cb = self.codebooks[si]
            diff = cb - qsub
            tables.append(np.einsum("ij,ij->i", diff, diff).astype(np.float32))

        cand_idx = []
        cand_approx = []
        for li in lists:
            idx = self.lists[li]
            if idx.size == 0:
                continue
            codes = self.codes[li]
            dist_sq = np.zeros((codes.shape[0],), dtype=np.float32)
            for si in range(m):
                dist_sq += tables[si][codes[:, si]]
            cand_idx.append(idx)
            cand_approx.append(dist_sq)

        if not cand_idx:
            return l2_topk(base, query, np.array([], dtype=np.int64), topK)

        idx_all = np.concatenate(cand_idx)
        approx_all = np.concatenate(cand_approx)
        pool = max(topK * 20, topK)
        if pool >= approx_all.size:
            order = np.argsort(approx_all)
        else:
            order = np.argpartition(approx_all, pool)[:pool]
            order = order[np.argsort(approx_all[order])]
        pool_idx = idx_all[order]

        if self.p.refine:
            return l2_topk(base, query, pool_idx, topK)
        else:
            top = pool_idx[:topK]
            d = np.sqrt(approx_all[order][:topK]).astype(np.float32)
            return top, d
