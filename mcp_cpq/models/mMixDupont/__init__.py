from itertools import combinations
from typing import Union

from numpy import array, isnan, nansum, ndarray, prod, sum
from pandas import DataFrame, Series

from models import Model


class Mix_Dupont(Model):
    # тип Dupont

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
            ind_half_df = int(len(data.columns) / 2)
            nl_mass_df = data.iloc[:, : self.k].to_numpy().astype("float64")
            lin_mass_df = (
                data.iloc[:, self.k : ind_half_df].to_numpy().astype("float64")
            )
            nl_quality_df = (
                data.iloc[:, ind_half_df : (ind_half_df + self.k)]
                .to_numpy()
                .astype("float64")
            )
            lin_quality_df = (
                data.iloc[:, (ind_half_df + self.k) :].to_numpy().astype("float64")
            )
        elif type(data) == ndarray:
            ind_half_array = int(data.shape[1] / 2)
            nl_mass_df = data[:, : self.k]
            lin_mass_df = data[:, self.k : ind_half_array]
            nl_quality_df = data[:, ind_half_array : (ind_half_array + self.k)]
            lin_quality_df = data[:, (ind_half_array + self.k) :]

        X_check = []
        for i, _ in enumerate(nl_mass_df):
            x = list(combinations(nl_mass_df[i, :], 2))
            for j, k in enumerate(x):
                x[j] = prod(k)
            X_check.append(x)
        X_check = array(X_check)

        Ox_lin_check = nansum(lin_quality_df * lin_mass_df, axis=1) / sum(
            [nansum(lin_mass_df, axis=1), nansum(nl_mass_df, axis=1)], axis=0
        )
        Ox_lin_check[isnan(Ox_lin_check)] = 0
        Ox_nlin_check = nansum(nl_quality_df * nl_mass_df, axis=1) / sum(
            [nansum(lin_mass_df, axis=1), nansum(nl_mass_df, axis=1)], axis=0
        )
        Ox_nlin_check[isnan(Ox_nlin_check)] = 0
        P = (
            nansum(X_check * self.bonus, axis=1)
            / sum([nansum(lin_mass_df, axis=1), nansum(nl_mass_df, axis=1)], axis=0)
            ** 2
        )
        P[isnan(P)] = 0
        return_list = ((Ox_lin_check + Ox_nlin_check + P) + self.intercept_).tolist()
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

    def load(self, bonus: list, num_nlin: int, intercept: float):
        """type_model - Тип смешения
        type_benz - Тип продукта
        type_quality - Тип показателя качества
        Загрузка модели
        """
        self.bonus = bonus
        self.k = num_nlin
        self.intercept_ = intercept
