from __future__ import annotations
from typing import Tuple
import numpy as np


def l2_topk(
    base: np.ndarray,
    q: np.ndarray,
    idx: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        idx = np.arange(base.shape[0], dtype=np.int64)
    sub = base[idx]
    diff = sub - q
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    if k >= dist_sq.size:
        order = np.argsort(dist_sq)
    else:
        order = np.argpartition(dist_sq, k)[:k]
        order = order[np.argsort(dist_sq[order])]
    return idx[order], np.sqrt(dist_sq[order]).astype(np.float32)


def sample_rows(
    X: np.ndarray,
    n: int,
    seed: int = 1
) -> np.ndarray:
    if n >= X.shape[0]:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]
