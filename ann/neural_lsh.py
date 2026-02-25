from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .kmeans import kmeans_lloyd
from .utils import l2_topk, sample_rows
from .base import ANNIndex


class MLP(nn.Module):
    def __init__(self, d_in: int, n_out: int, layers: int = 3, hidden: int = 256):
        super().__init__()
        mods: List[nn.Module] = []
        in_f = d_in
        for _ in range(max(layers - 1, 0)):
            mods += [nn.Linear(in_f, hidden), nn.ReLU()]
            in_f = hidden
        mods += [nn.Linear(in_f, n_out)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


@dataclass
class NeuralParams:
    m: int = 2048
    T: int = 5
    layers: int = 3
    hidden: int = 256
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    train_size: int = 50000
    seed: int = 1


class NeuralLSH(ANNIndex):
    def __init__(self, params: NeuralParams, device: str = "auto"):
        self.p = params
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
        self.centroids = None
        self.model = None
        self.inverted: List[np.ndarray] = []

    def build(self, base: np.ndarray) -> None:
        n, d = base.shape
        train = sample_rows(base, self.p.train_size, seed=self.p.seed).astype(np.float32)
        C, _ = kmeans_lloyd(train, self.p.m, iters=20, seed=self.p.seed)
        self.centroids = C

        x_norm = np.einsum("ij,ij->i", base, base)[:, None]
        c_norm = np.einsum("ij,ij->i", C, C)[None, :]
        dist = x_norm + c_norm - 2.0 * (base @ C.T)
        assign = np.argmin(dist, axis=1).astype(np.int64)
        self.inverted = [np.where(assign == j)[0].astype(np.int64) for j in range(self.p.m)]

        tr = sample_rows(base, self.p.train_size, seed=self.p.seed).astype(np.float32)
        x_norm = np.einsum("ij,ij->i", tr, tr)[:, None]
        c_norm = np.einsum("ij,ij->i", C, C)[None, :]
        dist = x_norm + c_norm - 2.0 * (tr @ C.T)
        y = np.argmin(dist, axis=1).astype(np.int64)

        torch.manual_seed(self.p.seed)
        np.random.seed(self.p.seed)
        model = MLP(d_in=d, n_out=self.p.m, layers=self.p.layers, hidden=self.p.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.p.lr)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(tr), torch.from_numpy(y)), batch_size=self.p.batch_size, shuffle=True)

        model.train()
        for _ in range(self.p.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        self.model = model

    def _top_bins(self, q: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        T = min(self.p.T, probs.size)
        top = np.argpartition(-probs, T-1)[:T]
        return top[np.argsort(-probs[top])].astype(np.int64)

    def query(
        self,
        base: np.ndarray,
        query: np.ndarray,
        topK: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        bins = self._top_bins(query)
        cand = []
        for b in bins:
            if 0 <= b < len(self.inverted):
                cand.append(self.inverted[b])
        idx = np.unique(np.concatenate(cand)).astype(np.int64) if cand else np.array([], dtype=np.int64)
        return l2_topk(base, query, idx, topK)
