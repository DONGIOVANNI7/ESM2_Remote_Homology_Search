from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ANNStatistics:
    qps: int = 0
    recall: float = 0


class ANNIndex(ABC):
    @abstractmethod
    def build(self, base: np.ndarray):
        pass

    @abstractmethod
    def query(
        self,
        base: np.ndarray,
        query: np.ndarray,
        topK: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
