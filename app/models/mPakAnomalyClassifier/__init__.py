import pickle
import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from app.models import Model
from core.model_core_sync import unpickle
from app.models.mPakAnomalyClassifier.factory import make_strategy 


class PakAnomalyClassifier(Model):
    def __init__(self):
        self.unit = None
        self.params: Dict[str, Any] = {}
        self.artifacts: Dict[str, Any] = {}

    def load(self, pickle_model: bytes):
        state = unpickle(pickle_model)
        if not isinstance(state, dict) or 'unit' not in state:
            raise ValueError("Неверный формат для PakAnomalyClassifier.load")
        self.unit = state['unit']
        self.params = state.get('params', {})
        self.artifacts = state.get('artifacts', {})

    def save(self):
        return pickle.dumps({
            'unit': self.unit,
            'params': self.params,
            'artifacts': self.artifacts
        }, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, data: Union[pd.DataFrame, np.ndarray])-> tuple[np.ndarray, list[str]]:
        if self.unit is None:
            raise RuntimeError("PakAnomalyClassifier не инициализирован")
        
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError("data должен быть либо pd.DataFrame, либо np.ndarray")

        strategy = make_strategy(self.unit)

        labels, comments = strategy.predict(data, self.params, self.artifacts)
        return labels, comments
    
    def train(self, data: pd.DataFrame, y: pd.Series):
        """
            data - Данные для обучения
            y - Лабораторные анализы для обучения
            Запуск обучения модели
        """
        pass

    def test(self, data: pd.DataFrame, y: pd.Series):
        """
            data - Данные для прогноза
            y - Лабораторные анализы для тестирования
            Запуск тестировния модели
            Return: r2_score
        """
        pass
    