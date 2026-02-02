from app.models import Model
from numpy import ndarray
from pandas import DataFrame, Series
from statsmodels.tsa.ar_model import AR


class mAutoRegression(Model):
    def __init__(self, data: ndarray, dates: ndarray):
        """ 
                        Инициализация модели
                """
        self._model = AR(data, dates=dates)
        self.train()
        # return self.predict(data)

    def predict(self, start, end, dynamic: bool):
        """ 
                        data - Данные для прогноза
                        Запуск прогнозирования модели
                        Return: List(value)
                """
        return self._model.predict(start=start, end=end, dynamic=dynamic)
        # pass

    def train(self):
        """ 
                        data - Данные для обучения
                        y - Лабораторные анализы для обучения
                        Запуск обучения модели
                """
        self._model = self._model.fit()
        # return self._model.score(data, y)
        # pass

    def test(self, data: DataFrame, y: Series):
        """ 
                        data - Данные для прогноза
                        y - Лабораторные анализы для тестирования
                        Запуск тестировния модели
                        Return: r2_score
                """
        return self._model.score(data, y)
        # pass

    def save(self):
        """ 
                        Сохранение модели
                        Return: Intercept, Coefficients
                """
        pass

    def load(self, intercept: float, coefs: ndarray):
        """ 
                        intercept - Свободный член модели
                        coefs - Коэффициенты при X модели
                        Загрузка модели
                """
        pass
