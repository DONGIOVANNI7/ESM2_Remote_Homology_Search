from __future__ import annotations
from typing import Tuple
import numpy as np


def kmeans_lloyd(
    X: np.ndarray,
    k: int,
    iters: int = 25,
    seed: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k <= 0 or k > n:
        raise ValueError("k must be in 1..n")

    centroids = np.empty((k, d), dtype=np.float32)
    centroids[0] = X[rng.integers(0, n)]
    dist_sq = np.einsum("ij,ij->i", X - centroids[0], X - centroids[0])
    for i in range(1, k):
        probs = dist_sq / (dist_sq.sum() + 1e-12)
        centroids[i] = X[rng.choice(n, p=probs)]
        newd = np.einsum("ij,ij->i", X - centroids[i], X - centroids[i])
        dist_sq = np.minimum(dist_sq, newd)

    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        x_norm = np.einsum("ij,ij->i", X, X)[:, None]
        c_norm = np.einsum("ij,ij->i", centroids, centroids)[None, :]
        dist = x_norm + c_norm - 2.0 * (X @ centroids.T)
        new_assign = np.argmin(dist, axis=1)
        if np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)
            else:
                centroids[j] = X[rng.integers(0, n)]
    return centroids.astype(np.float32), assign.astype(np.int64)
