from models import Model
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import optimize


class mMixture(Model):
    def __init__(self, coef=None, intercept: dict = None):
        """Инициализация составной модели,
        Args
        models (Dict[int, str]): словарь из процентов выхода и соответствующих им
                                назнаниях моделей, входящих в состав общей модели
        self._bias (Dict[str, float]): словарь с названиями моделей и bias, обновляется при train
        """
        # self._models = models
        if intercept is None:
            intercept = dict()
        self._intercept = intercept
        # for i, name in self._models.items():
        #     self._bias[name] = None
        self._coef = coef

    def a1(self, vol):
        return (
            -0.0000000000380144 * np.power(vol, 6.0)
            + 0.0000000056371828 * np.power(vol, 5.0)
            + 0.00000018190875 * np.power(vol, 4.0)
            - 0.000069095194 * np.power(vol, 3.0)
            + 0.0038943365 * np.power(vol, 2.0)
            - 0.068398325 * vol
            + 0.9177
        )

    def b1(self, vol):
        return (
            0.000000000004134309 * np.power(vol, 6.0)
            - 0.0000000001210607 * np.power(vol, 5.0)
            - 0.00000014705044 * np.power(vol, 4.0)
            + 0.000019454037 * np.power(vol, 3.0)
            - 0.00090298931 * np.power(vol, 2.0)
            + 0.014332337 * vol
            + 1.019
        )

    def a(self, vol):
        return (
            0.85875
            - 0.1945 * np.abs(vol - 4.5)
            + 0.1945 * np.abs(vol - 5.5)
            + 0.1076 * np.abs(vol - 19.5)
            - 0.1076 * np.abs(vol - 20.5)
            + 0.07455 * np.abs(vol - 39.5)
            - 0.07455 * np.abs(vol - 40.5)
            - 0.01075 * np.abs(vol - 59.5)
            + 0.01075 * np.abs(vol - 60.5)
            + 0.03925 * np.abs(vol - 79.5)
            - 0.03925 * np.abs(vol - 80.5)
            - 0.0741 * np.abs(vol - 92.5)
            + 0.0741 * np.abs(vol - 93.5)
        )

    def b(self, vol):
        return (
            1.02042
            + 0.04855 * np.abs(vol - 4.5)
            - 0.04855 * np.abs(vol - 5.5)
            - 0.0262 * np.abs(vol - 19.5)
            + 0.0262 * np.abs(vol - 20.5)
            - 0.01375 * np.abs(vol - 39.5)
            + 0.01375 * np.abs(vol - 40.5)
            + 0.00275 * np.abs(vol - 59.5)
            - 0.00275 * np.abs(vol - 60.5)
            - 0.0064 * np.abs(vol - 79.5)
            + 0.0064 * np.abs(vol - 80.5)
            + 0.0134 * np.abs(vol - 92.5)
            - 0.0134 * np.abs(vol - 93.5)
        )

    def d86_fun(self, temp_tbp, volume):
        return (
            np.exp(np.log((temp_tbp + 273.15) / self.a(volume)) / self.b(volume))
            - 273.15
        )

    def tbp_fun(self, temp_d86, volume):
        return self.a(volume) * np.power((temp_d86 + 273.15), self.b(volume)) - 273.15

    def tbp_fun1(self, temp_d86, volume):
        return self.a1(volume) * np.power((temp_d86 + 273.15), self.b1(volume)) - 273.15

    def integfun(self, b, c, t, t50):
        return (
            (
                b
                * np.sqrt(2.0)
                * np.tan((np.sin((t - t50) / c) * np.sqrt(2.0) * 0.5) / b)
                + ((t - t50) / c)
            )
            / np.pi
            + 1.0
        ) / 2.0

    def diffun_s(self, b, s):
        return (
            (
                (1 + np.power(np.tan(0.5 * np.sin(s) * np.sqrt(2.0) / b), 2))
                * np.cos(s)
                + 1
            )
            / np.pi
            / 2.0
        )

    # def integfun_s(b, s):
    #     return ((b * np.power(2, 0.5) * np.tan((np.sin(s) * np.power(2, 0.5)*
    # 0.5) / b) + (s)) / np.pi + 1.0) / 2.0

    def diffprov(self, t50, c, b, t, vol):
        return np.power(self.integfun(b, c, t, t50) * 100 - vol, 2)

    def fun(self, coef, temps, volumes):
        res = []
        for t, v in zip(temps, volumes):
            if v <= 50:
                res.append(self.diffprov(coef[0], coef[1], coef[2], t, v))
            if v >= 50:
                res.append(self.diffprov(coef[0], coef[1], coef[3], t, v))
        return res

    def get_initial_guess(self, tbp, volumes):
        volumes_lst = list(volumes)
        ind_50 = volumes_lst.index(50)
        ind_max = volumes_lst.index(max(volumes))
        ind_min = volumes_lst.index(min(volumes))
        t_50 = tbp[ind_50]
        t_diff = (tbp[ind_max] - tbp[ind_min]) / 2
        return [t_50, t_diff, 1, 1]

    def get_coef(self, tbp, volumes):
        solution = optimize.least_squares(
            self.fun,
            self.get_initial_guess(tbp, volumes),
            jac="3-point",
            bounds=([0, 0, 1, 1], [np.inf, np.inf, 10, 10]),
            args=(tbp, volumes),
            loss="soft_l1",
        )
        return solution.x

    def train_coef(self, temps, volumes):
        """
        Метод для получение коэффициентов из данных компонентов и объемов
        Формат данных:
        volumes = np.array([1, 50, 98])
        temps = np.array([
            [30.049025, 51.229911, 72.410797],
            [32.840950, 46.128136, 59.415321],
            [30.309742, 48.567392, 66.825043],
            [28.614759, 49.664161, 70.713562],
        ])
        Столбцы в temps соответстуют объему выкипания. Строки соответствуют количеству
        компонентов. Количество столбцов в volumes и temps должно соответствовать


        """
        tbp_temps = self.tbp_fun(temps, volumes)
        self._coef = [self.get_coef(tbp, volumes) for tbp in tbp_temps]

    def approx_temp(self, temp, tbp_dist, volumes):
        bps = {}
        for dv in volumes:
            i = np.argmin(np.abs(tbp_dist - dv))
            bps[dv] = self.d86_fun(temp[i], dv)
        return bps

    def get_diff_curve(self, temps, active_flow_ids):
        dfs_ss = []
        for i, cf in enumerate(self._coef):
            t50, c, b, t = cf[0], cf[1], cf[2], cf[3]
            if i in active_flow_ids:
                dfs_s = np.zeros(temps.shape)
                s = (temps - t50) / c
                diff_b = (temps < t50) & (temps > t50 - c * np.pi)
                diff_t = (temps >= t50) & (temps < t50 + c * np.pi)
                dfs_s[diff_b] = self.diffun_s(b, s[diff_b]) * 100
                dfs_s[diff_t] = self.diffun_s(t, s[diff_t]) * 100
                dfs_s = dfs_s / max(dfs_s)
                dfs_s = 50 * dfs_s / np.sum(dfs_s)
                dfs_ss.append(dfs_s)
        return dfs_ss

    def get_integral_curve(self, dfs_ss, flows):
        flow_share = np.divide(flows, np.sum(flows))
        dfs_ss_share = dfs_ss * flow_share[:, np.newaxis]
        dfs_all = np.sum(dfs_ss_share, axis=0)
        tbp = np.cumsum(dfs_all + np.array([0, *dfs_all[:-1]]))
        return tbp

    def predict(self, data: np.array, flows: List[float], volumes: List[int]):
        """Запуск прогнозирования модели (по температурам D86 рассчитывает смешение фракционного состава нефтесырья)
        Args
        data (pd.DataFrame): температуры D86 в потоках фракционного состава
        flow (List[float]):  расходы потоков
        vols (List[int]): точный процент выхода, который соответствует [Tnk, T10, T50, Tkk]

        Формат данных:
        flows = np.array([11.1, 22.2, 33.3, 44.4])
        volumes = np.array([1, 50, 98])
        data = np.array([
            [30.049025, 51.229911, 72.410797],
            [32.840950, 46.128136, 59.415321],
            [30.309742, 48.567392, 66.825043],
            [28.614759, 49.664161, 70.713562],
        ])

        """
        # if data.shape[1] <= 4 or data.shape[1] % 2 != 0:
        #     raise Exception('Data shape error')
        # elif len(flow) != data.shape[1] / 2:
        #     raise Exception('Flow length error')
        # elif len(vols) != data.shape[1] / 2:
        #     raise Exception('Vols length error')

        ##
        ##
        flows[flows < 1] = 0
        active_flow_ids = [i for i, f in enumerate(flows) if f > 0]
        mn = np.inf
        mx = -np.inf
        for i, cf in enumerate(self._coef):
            t50 = cf[0]
            c = cf[1]
            if i in active_flow_ids:
                if t50 - c * np.pi < mn:
                    mn = t50 - c * np.pi
                if t50 + c * np.pi > mx:
                    mx = t50 + c * np.pi

        # mn = self.limits[0][0] if mn < self.limits[0][1] else mn
        # mx = self.limits[1][0] if mx > self.limits[1][1] else mx

        t_all = np.linspace(mn, mx, int(mx - mn))
        dfs_ss = self.get_diff_curve(t_all, active_flow_ids)

        if len(active_flow_ids) == 1:
            flow_id = active_flow_ids[0]
            d86 = dict(zip(volumes, data[flow_id]))
        else:
            tbp = self.get_integral_curve(dfs_ss, flows[active_flow_ids])
            d86 = self.approx_temp(t_all, tbp, volumes)
        for k, v in d86.items():
            d86[k] = v + self._intercept.get(k, 0)
        return d86

    def train(
        self,
        d_threshold: Dict[int, float],
        d_mult: Dict[int, float],
        d_lab: Dict[int, float],
        d_pred: Dict[int, float],
    ):
        for k, pred in d_pred.items():
            lab = d_lab.get(k)
            if lab is None:
                continue
            threshold = d_threshold.get(k, 0)
            mult = d_mult.get(k, 0)
            delta = lab - pred
            if abs(delta) > threshold:
                self._intercept[k] = delta * mult
        # """Переобучение модели (в процессе обновляется словарь с bias)
        # Args
        # bias_porog (Dict[str, int]): словарь со значениями порогами bias для каждой модели
        # bias_coef (Dict[str, int]): словарь со значениями коэффициентами bias для каждой модели
        # last_lab (Dict[str, int]): словарь с последними значениями LAB для каждой модели
        # last_va (Dict[str, int]): словарь с последними значениями VA для каждой модели
        # """
        # for i, name in self._models.items():
        #     self._bias[name] = (last_lab[name] - last_va[name]) * bias_coef[name]
        #     if abs(last_lab[name] - last_va[name]) >= bias_porog[name]:
        #         delt = (last_lab[name] - last_va[name]) * bias_coef[name]
        #         self._bias[name] = delt
        # return self._bias

    def test(self):
        pass
