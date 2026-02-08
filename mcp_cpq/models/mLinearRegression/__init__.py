from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from models import Model


class mLinearRegression(Model):
    def __init__(self):
        """Инициализация модели"""
        self._model = LinearRegression()

    def predict(self, data: DataFrame):
        """data - Данные для прогноза
        Запуск прогнозирования модели
        Return: List(value)
        """
        predict = self._model.predict(data) + self.intercept_
        return predict
        # pass

    def train(self, data: DataFrame, y: Series):
        """data - Данные для обучения
        y - Лабораторные анализы для обучения
        Запуск обучения модели
        """
        self._model.fit(data, y)
        return self._model.score(data, y)
        # pass

    def test(self, data: DataFrame, y: Series):
        """data - Данные для прогноза
        y - Лабораторные анализы для тестирования
        Запуск тестировния модели
        Return: r2_score
        """
        return self._model.score(data, y)
        # pass

    def save(self):
        """Сохранение модели
        Return: Intercept, Coefficients
        """
        return self._model
        # pass

    def load(self, intercept: float, coefs: ndarray):
        """intercept - Свободный член модели
        coefs - Коэффициенты при X модели
        Загрузка модели
        """
        self.intercept_ = intercept
        self.coef_ = coefs
        self._model.intercept_ = 0
        self._model.coef_ = coefs
        # pass
