from numpy import ndarray, nansum, power
from pandas import DataFrame, Series
from typing import Union
from models import Model


class Smes_cp_model(Model):

    def predict(self, data: Union[DataFrame, ndarray]):
        """data - Данные для прогноза
        dataframe [flow_rate:Series, quality:Series], ndarray [[flow_rate:ndarray, quality:ndarray]]
        Запуск прогнозирования качества смеси
        Return: Quality value
        """
        return_list = list()
        if not self.__dict__.get("intercept_"):
            self.intercept_ = 0.0
        if type(data) == DataFrame:
            input_array = data.to_numpy().astype("float64")
        elif type(data) == ndarray:
            input_array = data.astype("float64")

        ind_half_array = int(data.shape[1] / 2)
        return_list = (
            power(
                (
                    nansum(
                        power(input_array[:, ind_half_array:] + 273, (1 / 0.05))
                        * input_array[:, :ind_half_array],
                        axis=1,
                    )
                    / nansum(input_array[:, :ind_half_array], axis=1)
                ),
                0.05,
            )
            + self.intercept_
            - 273
        ).tolist()
        return return_list

    def train(self, data: DataFrame, y: Series):
        """data - Данные для обучения
        y - Лабораторные анализы для обучения
        Запуск обучения модели
        """
        pass

    def test(self, data: DataFrame, y: Series):
        """data - Данные для прогноза
        y - Лабораторные анализы для тестирования
        Запуск тестировния модели
        Return: r2_score
        """
        pass

    def save(self):
        """Сохранение модели
        Return: Intercept, Coefficients
        """
        return self.intercept_

    def load(self, intercept: float):
        """intercept - Свободный член модели
        coefs - Коэффициенты при X модели
        Загрузка модели
        """
        self.intercept_ = intercept
