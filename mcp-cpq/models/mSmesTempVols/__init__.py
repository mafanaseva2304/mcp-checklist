from numpy import ndarray, array, delete, interp, arange, argmin, abs
from pandas import DataFrame, Series
from typing import Union
from models import Model
from models.mMixture import mMixture


class Smes_temp_vols_model(Model):

    def predict(self, data: DataFrame):
        """data - Данные для прогноза смеси температур кипения
        dataframe [flow_rate:Series, quality:Series]
        Запуск прогнозирования качества смеси
        Return: Quality value
        """
        return_list = list()
        d_q = data.iloc[:, : self.count_flows_].to_numpy()
        d_fr = data.iloc[:, self.count_flows_ :].to_numpy()
        if d_fr.shape[1] % self.count_volumes_ != 0:
            raise ValueError("Несовпадение данных температур потоков с заданым числом")
        if d_q.shape[1] - (d_fr.shape[1] // self.count_volumes_) != 0:
            raise ValueError("Несовпадение данных количества потоков")
        mix = mMixture(intercept=self.intercepts_)
        volumes = array(self.volumes_)
        out_temps = array(list(self.intercepts_.keys()))
        for i_q, dq_i in enumerate(d_q):
            mix_flows = dq_i
            mix_temps = d_fr[i_q].reshape((self.count_flows_, self.count_volumes_))

            item_for_delete = []
            for i_en, fl_i in enumerate(mix_flows):
                if fl_i < 1:
                    item_for_delete.append(i_en)
            if item_for_delete:
                mix_flows = delete(mix_flows, item_for_delete, axis=0)
                mix_temps = delete(mix_temps, item_for_delete, axis=0)
            if len(mix_flows) < 1:
                if len(out_temps) > 1:
                    return_list.append({i_none: None for i_none in self.intercepts_})
                else:
                    return_list.append(None)
            elif len(mix_flows) == 1:
                result = {
                    i: j
                    for i, j in zip(
                        range(1, 100),
                        interp(range(1, 100), volumes, mix_temps[-1].astype("float64")),
                    )
                }
                percent_out = dict()
                for temp in out_temps:
                    percent_out[temp] = (
                        list(result.keys())[
                            argmin(abs(array(list(result.values())) - temp))
                        ]
                        + self.intercepts_[temp]
                    )
                if len(out_temps) > 1:
                    return_list.append(dict(zip(out_temps, percent_out[out_temps])))
                else:
                    return_list.append(percent_out[out_temps[0]] + self.intercept_)
            else:
                mix.train_coef(mix_temps, volumes)
                smes_mix = mix.predict(mix_temps, mix_flows, arange(1, 100))
                percent_out = dict()
                for temp in out_temps:
                    percent_out[temp] = (
                        list(smes_mix.keys())[
                            argmin(abs(array(list(smes_mix.values())) - temp))
                        ]
                        + self.intercepts_[temp]
                    )
                if len(out_temps) > 1:
                    return_list.append(percent_out)
                else:
                    return_list.append(percent_out[out_temps[0]] + self.intercept_)

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

    def load(self, intercepts: dict, count_flows: int, volumes: list):
        """Загрузка модели
        intercepts - Свободные члены модели
        intercepts = {10: 0,  50:0, 95: 0} (проценты выхода соответствуют прогнозируемым параметрам температур)
        count_flows - Количество входящих в модель потоков смешения
        volumes - проценты выхода входящих потоков смешения
        volumes = [1, 50, 98]
        intercept - свободный коэффициент модели
        """
        self.intercepts_ = intercepts
        self.count_flows_ = count_flows
        self.count_volumes_ = len(volumes)
        self.volumes_ = volumes
        for k, v in intercepts.items():
            self.intercept_ = v
