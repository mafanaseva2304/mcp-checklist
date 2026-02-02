from numpy import ndarray, array, full, divide
from pandas import DataFrame, Series
from typing import Union
from app.models import Model


class DNP_model(Model):

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe [columns:[Н-бутан, Сумма пентанов]], ndarray [[Н-бутан, Сумма пентанов]]
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        if not self.__dict__.get('intercept_'):
            self.intercept_ = 0.0
        if type(data) == DataFrame:
            input_two_dimen_array = data.to_numpy()
        elif type(data) == ndarray:
            input_two_dimen_array = data

        amounts_substance = array(
            [full(input_two_dimen_array[:, 0].shape, 2.1 / 58.12), input_two_dimen_array[:, 0] / 58.12,
             input_two_dimen_array[:, 1] / 72.15])

        substance_content = array(
            [
                [0.55, 0.41, 0.2],
                [0.6, 0.45, 0.21]
            ])

        amounts_substance_mol = divide(amounts_substance, amounts_substance.sum(axis=0)) * 100

        delta1 = (amounts_substance_mol.transpose() * substance_content[0] / 100).sum(axis=1) - 0.1
        delta2 = (amounts_substance_mol.transpose() * substance_content[1] / 100).sum(axis=1) - 0.5

        rizb = ((0.1 + (0.5 - 0.1) * delta1 / (delta1 - delta2)) - 0.1) + self.intercept_
        return rizb.tolist()

    def train(self, data: DataFrame, y: Series):
        """ data - Данные для обучения
            y - Лабораторные анализы для обучения
            Запуск обучения модели
        """
        pass

    def test(self, data: DataFrame, y: Series):
        """ data - Данные для прогноза
            y - Лабораторные анализы для тестирования
            Запуск тестировния модели
            Return: r2_score
        """
        pass

    def save(self):
        """ Сохранение модели
            Return: Intercept, Coefficients
        """
        return self.intercept_

    def load(self, intercept: float):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        self.intercept_ = intercept
