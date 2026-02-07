import pickle
from models import Model
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline


class mPipeline(Model):
    def __init__(self):
        """
        Инициализация модели
        """
        self._model = Pipeline(())
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
        self._model = pickle.loads(pikle_model)
        self.intercept_ = 0
