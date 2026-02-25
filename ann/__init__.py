# ANN package
from .hypercube import HypercubeIndex, HypercubeParams
from .ivf_flat import IVFFlat, IVFParams
from .ivf_pq import IVFPQ, IVFPQParams
from .lsh import EuclideanLSH, LSHParams
from .neural_lsh import NeuralLSH, NeuralParams
from .base import ANNIndex, ANNStatistics


ANN_METHODS = {
    "lsh": "Euclidean LSH",
    "hypercube": "Hypercube",
    "ivfflat": "IVF-Flat",
    "ivfpq": "IVF-PQ",
    "neural": "Neural LSH"
}
