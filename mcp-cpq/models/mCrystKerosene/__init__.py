from numpy import ndarray, array, full, divide
import numpy as np
from pandas import DataFrame, Series
from typing import Union
from models import Model


class CrystKerosene(Model):

    def predict(self, data: Union[DataFrame, ndarray]):
        """data - Данные качества смеси для прогноза
        dataframe [columns:[IBP, EBP, D15]], ndarray [[IBP, EBP, D15]]
        Запуск прогнозирования качества смеси
        Return: list(Quality value)
        """
        if not self.__dict__.get("intercept_"):
            self.intercept_ = 0.0
        if type(data) == DataFrame:
            input_array = data.to_numpy()
        elif type(data) == ndarray:
            input_array = data

        # SG = 283 / (2 * input_array[:, 2] + 263)
        T_b = (input_array[:, 0] + input_array[:, 1]) / 2 + 273.15
        # MW = 0.00016607 * math.pow(T_b, 2.1962)/math.pow(input_array[:, 2], 1.0164)
        MW = (
            42.965
            * np.exp(
                0.0002097 * T_b
                - 7.78712 * input_array[:, 2]
                + 0.00208476 * T_b * input_array[:, 2]
            )
            * np.power(T_b, 1.26007)
            * np.power(input_array[:, 2], 4.98308)
        )

        i = 0.3773 * np.power(input_array[:, 2], 0.9182) / np.power(T_b, 0.02269)
        n = np.sqrt((1 + 2 * i) / (1 - i))
        m = MW * (n - 1.475)

        # if MW <= 200:
        X_p = 3.7387 - 4.0829 * input_array[:, 2] + 0.014772 * m
        X_n = -1.5027 + 2.10152 * input_array[:, 2] - 0.02388 * m
        X_a = 1 - X_p - X_n

        # Condition of nonnegative X_p, X_n, X_a
        cond_arr = np.vstack((X_p, X_n, X_a)).T
        cond_arr[cond_arr < 0] = 0
        cond_arr = cond_arr / cond_arr.sum(axis=1, keepdims=True)
        X_p, X_n, X_a = [col.ravel() for col in cond_arr.T]

        M_p = np.power(346.8907104 - 49.67709886 * np.log(1070 - T_b), 1.5)
        M_n = np.power(310.6962930 - 44.66279589 * np.log(1028 - T_b), 1.5)
        M_a = np.power(307.5487316 - 44.50378282 * np.log(1015 - T_b), 1.5)

        T_p = 397 - np.exp(6.5096 - 0.14187 * np.power(M_p, 0.470))
        T_n = 370 - np.exp(6.52504 - 0.04945 * np.power(M_n, 2 / 3))
        T_a = 375 - np.exp(6.53599 - 0.04912 * np.power(M_a, 2 / 3))

        T_fk = X_p * T_p + X_n * T_n + X_a * T_a
        t_fk = T_fk - 273.15 + +self.intercept_
        return t_fk.tolist()

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
