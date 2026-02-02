from numpy import ndarray, nansum, sum, isnan, column_stack
from pandas import DataFrame, Series
from typing import Union
from app.models import Model


class Mix_Ethyl(Model):
    # тип Ethyl

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe [flow_rate:Series, quality:Series], ndarray [[flow_rate:ndarray, quality:ndarray]]
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        return_list = list()
        if not self.__dict__.get('intercept_'):
            self.intercept_ = 0.0

        if type(data) == DataFrame:
            ind_half_df = int(len(data.columns) / 5)
            nl_mass_df = data.iloc[:, :self.k].to_numpy().astype('float64')
            lin_mass_df = data.iloc[:, self.k:ind_half_df].to_numpy().astype('float64')
            l = sum([nansum(lin_mass_df, axis=1), nansum(nl_mass_df, axis=1)], axis=0).reshape(-1, 1)
            nl_mass_df = (nl_mass_df / l)
            lin_mass_df = (lin_mass_df / l)
            nl_moch_df = data.iloc[:, ind_half_df:(ind_half_df + self.k)].to_numpy().astype('float64')
            lin_moch_df = data.iloc[:, (ind_half_df + self.k):(2 * ind_half_df)].to_numpy().astype('float64')
            nl_ioch_df = data.iloc[:, (2 * ind_half_df):(2 * ind_half_df + self.k)].to_numpy().astype('float64')
            lin_ioch_df = data.iloc[:, (2 * ind_half_df + self.k):(3 * ind_half_df)].to_numpy().astype('float64')
            nl_olef_df = data.iloc[:, (3 * ind_half_df):(3 * ind_half_df + self.k)].to_numpy().astype('float64')
            lin_olef_df = data.iloc[:, (3 * ind_half_df + self.k):(4 * ind_half_df)].to_numpy().astype('float64')
            nl_arom_df = data.iloc[:, (4 * ind_half_df):(4 * ind_half_df + self.k)].to_numpy().astype('float64')
            lin_arom_df = data.iloc[:, (4 * ind_half_df + self.k):].to_numpy().astype('float64')
        elif type(data) == ndarray:
            ind_half_array = int(data.shape[1] / 5)
            nl_mass_df = data[:, :self.k]
            lin_mass_df = data[:, self.k:ind_half_array]
            l = sum([nansum(lin_mass_df, axis=1), nansum(nl_mass_df, axis=1)], axis=0).reshape(-1, 1)
            nl_mass_df = (nl_mass_df / l)
            lin_mass_df = (lin_mass_df / l)
            nl_moch_df = data[:, ind_half_array:(ind_half_array + self.k)]
            lin_moch_df = data[:, (ind_half_array + self.k):(2 * ind_half_array)]
            nl_ioch_df = data[:, (2 * ind_half_array):(2 * ind_half_array + self.k)]
            lin_ioch_df = data[:, (2 * ind_half_array + self.k):(3 * ind_half_array)]
            nl_olef_df = data[:, (3 * ind_half_array):(3 * ind_half_array + self.k)]
            lin_olef_df = data[:, (3 * ind_half_array + self.k):(4 * ind_half_array)]
            nl_arom_df = data[:, (4 * ind_half_array):(4 * ind_half_array + self.k)]
            lin_arom_df = data[:, (4 * ind_half_array + self.k):]

        if self.type_quality == 1:
            k_1 = (nansum(nl_mass_df * nl_moch_df * (nl_ioch_df - nl_moch_df), axis=1) -
                   (nansum(nl_mass_df * nl_moch_df, axis=1) * nansum(nl_mass_df * (nl_ioch_df - nl_moch_df), axis=1))/ nansum(nl_mass_df, axis=1))
            k_2 = (nansum(nl_mass_df * (nl_olef_df ** 2), axis=1) -
                   (nansum(nl_mass_df * nl_olef_df, axis=1) ** 2) / nansum(nl_mass_df, axis=1))
            k_3 = (nansum(nl_mass_df * (nl_arom_df ** 2), axis=1) -
                   (nansum(nl_mass_df * nl_arom_df, axis=1) ** 2) / nansum(nl_mass_df, axis=1)) / 100
            X_check = column_stack((k_1, k_2, k_3))
            X_check[isnan(X_check)] = 0

            Ox_lin_check = nansum(lin_moch_df * lin_mass_df, axis=1)
            Ox_lin_check[isnan(Ox_lin_check)] = 0
            Ox_nlin_check = nansum(nl_moch_df * nl_mass_df, axis=1)
            Ox_nlin_check[isnan(Ox_nlin_check)] = 0
            P = nansum(X_check * self.bonus, axis=1)
            P[isnan(P)] = 0

        elif self.type_quality == 2:
            k_1 = (nansum(nl_mass_df * nl_ioch_df * (nl_ioch_df - nl_moch_df), axis=1) -
                   (nansum(nl_mass_df * nl_ioch_df, axis=1) * nansum(nl_mass_df * (nl_ioch_df - nl_moch_df), axis=1))/ nansum(nl_mass_df, axis=1))
            k_2 = (nansum(nl_mass_df * (nl_olef_df ** 2), axis=1) -
                   (nansum(nl_mass_df * nl_olef_df, axis=1) ** 2) / nansum(nl_mass_df, axis=1))
            k_3 = (nansum(nl_mass_df * (nl_arom_df ** 2), axis=1) -
                   (nansum(nl_mass_df * nl_arom_df, axis=1) ** 2) / nansum(nl_mass_df, axis=1))
            X_check = column_stack((k_1, k_2, k_3))
            X_check[isnan(X_check)] = 0

            Ox_lin_check = nansum(lin_ioch_df * lin_mass_df, axis=1)
            Ox_lin_check[isnan(Ox_lin_check)] = 0
            Ox_nlin_check = nansum(nl_ioch_df * nl_mass_df, axis=1)
            Ox_nlin_check[isnan(Ox_nlin_check)] = 0
            P = nansum(X_check * self.bonus, axis=1)
            P[isnan(P)] = 0
        return_list = ((Ox_lin_check + Ox_nlin_check + P) + self.intercept_).tolist()
        return return_list

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

    def load(self, bonus: list, num_nlin: int, type_quality: int, intercept: float):
        """ type_model - Тип смешения
            type_benz - Тип продукта
            type_quality - Тип показателя качества
            Загрузка модели
        """
        self.bonus = bonus
        self.k = num_nlin
        self.intercept_ = intercept
        self.type_quality = type_quality
