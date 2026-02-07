from numpy import ndarray, nansum
from pandas import DataFrame, Series
from typing import Union
from models import Model


class Smes_model(Model):

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
            ind_half_df = int((len(data.columns) - 1) / 2 + 1)
            d_q = data.iloc[:, :ind_half_df].to_numpy().astype("float64")
            d_q[d_q < 1] = 0
            d_fr = data.iloc[:, ind_half_df:].to_numpy().astype("float64")
            return_list = (
                (nansum(d_q * d_fr, axis=1) / nansum(d_q, axis=1)) + self.intercept_
            ).tolist()
        elif type(data) == ndarray:
            ind_half_array = int(data.shape[1] / 2)
            data[:, :ind_half_array][data[:, :ind_half_array] < 1] = 0
            return_list = (
                (
                    nansum(data[:, ind_half_array:] * data[:, :ind_half_array], axis=1)
                    / nansum(data[:, :ind_half_array], axis=1)
                )
                + self.intercept_
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
