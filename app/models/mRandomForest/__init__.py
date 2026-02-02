import pickle
from app.models import Model
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from core.model_core_sync import unpickle


class mRandomForest(Model):
    def __init__(self):
        """ 
                        Инициализация модели
                """
        self._model = RandomForestRegressor(warm_start=True, random_state=42)
        self.intercept_ = 0

    def predict(self, data: DataFrame):
        """ 
                        data - Данные для прогноза
                        Запуск прогнозирования модели
                        Return: List(value)
                """
        return self._model.predict(data) + self.intercept_

    def train(self, data: DataFrame, y: Series):
        """ 
                        data - Данные для обучения
                        y - Лабораторные анализы для обучения
                        Запуск обучения модели
                """
        self._model.fit(data, y)
        return self._model.score(data, y)

    def hot_train(self, data: DataFrame, y: Series):
        """ 
                        data - Данные для дообучения
                        y - Лабораторные анализы для дообучения
                        Запуск дообучения модели
                """
        self._model.n_estimators += 1
        self._model.fit(data, y)
        return self._model.score(data, y)

    def test(self, data: DataFrame, y: Series):
        """ 
                        data - Данные для прогноза
                        y - Лабораторные анализы для тестирования
                        Запуск тестировния модели
                        Return: r2_score
                """
        return self._model.score(data, y)

    def save(self):
        """ 
                       retrun pikle_model
                """
        return pickle.dumps(self._model)

    def load(self, pikle_model):
        """ 
                        pikle_model - Загрузка модели из pikle_load_model
                """
        self._model = unpickle(pikle_model)
        self.intercept_ = 0
