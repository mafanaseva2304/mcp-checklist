from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class TechModeStrategy(ABC):
    @abstractmethod
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any] = None
    ) -> tuple[np.ndarray, list[str]]:
        pass
