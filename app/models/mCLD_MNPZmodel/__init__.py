from numpy import ndarray, nansum
from pandas import DataFrame, Series
from typing import Union
from app.models import Model


class CLD_MNPZmodel(Model):
    """
    CLD_MNPZ- Т помутн, °С (МНПЗ)
    """
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
            ind_half_df = int((len(data.columns) - 1) / 2 + 1)
            d_fr  = data.iloc[:, :ind_half_df].to_numpy().astype('float64')
            d_q = data.iloc[:, ind_half_df:].to_numpy().astype('float64')
            d_q_transformed = (0.0026415*(d_q/5*9+32+460))**20
            d_fr[d_fr < 1] = 0
            weight_sum = (nansum(d_q_transformed * d_fr, axis=1) / nansum(d_fr, axis=1))
            return_list = ((weight_sum**(1/20)/0.0026415-32-460)*5/9  + self.intercept_)
        elif type(data) == ndarray:
            ind_half_array = int(data.shape[1] / 2)
            d_fr = data[:, :ind_half_array].astype('float64')
            d_q = data[:, ind_half_array:].astype('float64')
            d_q_transformed = (0.0026415*(d_q/5*9+32+460))**20
            d_fr[d_fr < 1] = 0
            weight_sum = (nansum(d_q_transformed * d_fr, axis=1) / nansum(d_fr, axis=1))
            return_list = ((weight_sum**(1/20)/0.0026415-32-460)*5/9 + self.intercept_)
        return return_list.tolist()

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