from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class PakAnomalyStrategy(ABC):
    @abstractmethod
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any]
    ) -> tuple[np.ndarray, list[str]]:
        pass
