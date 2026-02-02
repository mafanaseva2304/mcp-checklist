from math import pi
import numpy as np
from app.models import Model
from app.models.mMixture import mMixture
from pandas import DataFrame, Series
from typing import Union


class SmesTempUILN(Model):
    def __init__(self, alg_smes):
        """ Инициализация модели
        """
        self._alg_smes = alg_smes
        self.intercept_ = 0.0

    def get_delay_time(self, time, flow_rate, vp):
        """
        time - индекс, для которого необходимо рассчитать параметр
        array - numpy массив потока [м3/ч]
        vp - объем трубы [м3]

        return: индекс элемента в массиве, соответсвующий показателю качества time
        """

        # Замена Nan на 0:
        flow_rate = np.nan_to_num(flow_rate)

        if flow_rate.sum() == 0:
            print('Один из потоков неактивен')
            return time
            # raise Exception('Один из потоков неактивен')
        else:
            m = 1
            while flow_rate[time:time + m].sum() < vp:
                m += 1
                if m == 300:
                    break
            delay_time = time + m
            if delay_time > 300:
                delay_time = 300
            return delay_time

    def predict(self, data: np.ndarray, flows: np.ndarray):
        """Запуск прогнозирования модели (по температурам D86 рассчитывает смешение фракционного состава нефтесырья)
        Args
        data (np.ndarray): температуры D86 в потоках фракционного состава
        flow (np.ndarray):  расходы потоков

        Формат данных:
        flows = np.array([11.1, 22.2, 33.3])  С-100, С-250, АВТ-6
        volumes = np.array([1, 50, 98])
        data = np.array([
            [30.049025, 51.229911, 72.410797], АВТ-6
            [32.840950, 46.128136, 59.415321], С-250
            [30.309742, 48.567392, 66.825043], С-100,
        ])

        """
        l1 = 812
        l2 = 303
        l3 = 440
        l4 = 1676
        d1 = (219 - 8 * 2) * 10 ** (-3)
        d2 = (159 - 6 * 2) * 10 ** (-3)

        mix = mMixture(intercept={1: 0, 10: 0, 50: 0, 90: 0, 98: 0})

        favt6, fs250, fs100 = flows[0], flows[1], flows[2]
        sum_flow_rates = flows.sum(axis=1)

        volumes = np.array([1, 10, 50, 90, 98])

        v_b3b4 = l1 * (pi * d1 ** 2) / 4
        b3_time = self.get_delay_time(0, sum_flow_rates/60, v_b3b4)

        f_s100s250 = fs100 + fs250
        v_b2b3 = l2 * (pi * d2 ** 2) / 4
        b2_time = self.get_delay_time(b3_time, f_s100s250/60, v_b2b3)

        v_a1b2 = l4 * (pi * d2 ** 2) / 4
        a1_time = self.get_delay_time(b2_time, fs250/60, v_a1b2)

        s100s250_mix_temps = np.zeros([2, 5])
        s100s250_mix_temps[0, :] = data[b2_time, 2, :]  # С-100
        s100s250_mix_temps[1, :] = data[a1_time, 1, :]  # С-250

        s1250s100_flows = np.zeros(2)
        s1250s100_flows[1] = fs100[b2_time]  # С-100
        s1250s100_flows[0] = fs250[a1_time]  # С-250

        s1250s100_active = s1250s100_flows > 2
        s1250s100_flows = s1250s100_flows[s1250s100_active]
        s100s250_mix_temps = s100s250_mix_temps[s1250s100_active]
        # for i_en, fl_i in enumerate(s1250s100_flows):
        #     if fl_i < 2:
        #         s1250s100_flows = np.delete(s1250s100_flows,(i_en), axis = 0)
        #         s100s250_mix_temps = np.delete(s100s250_mix_temps,(i_en), axis = 0)
        if len(s1250s100_flows) < 1:
            print('Нет начальных активных потоков (С-100, С-250)')
            b2_mix = {1: 0, 10: 0, 50: 0, 90: 0, 98: 0}
            f_s100s250 = f_s100s250 - f_s100s250
        else:
            mix.train_coef(s100s250_mix_temps, volumes)
            # b2_mix = (q1[b2_time] * f1[b2_time] + q2[a1_time] * f2[a1_time]) / (f1[b2_time] + f2[a1_time])
            b2_mix = mix.predict(s100s250_mix_temps, s1250s100_flows, volumes)

        v_c1b3 = l3 * (pi * d1 ** 2) / 4
        c1_time = self.get_delay_time(b3_time, favt6/60, v_c1b3)

        b2avt6_temps = np.zeros([2, 5])
        b2avt6_temps[0, :] = np.asarray(list(b2_mix.values()))  # Смесь С-100 и С-250
        b2avt6_temps[1, :] = data[c1_time, 0, :]  # АВТ-6

        b2avt6_flows = np.zeros(2)
        b2avt6_flows[0] = f_s100s250[b2_time]  # Смесь С-100 и С-250
        b2avt6_flows[1] = favt6[c1_time]  # АВТ-6

        b2avt6_active = b2avt6_flows > 2
        b2avt6_flows = b2avt6_flows[b2avt6_active]
        b2avt6_temps = b2avt6_temps[b2avt6_active]
        # for i_en, fl_i in enumerate(b2avt6_flows):
        #     if fl_i < 2:
        #         b2avt6_flows = np.delete(b2avt6_flows,(i_en), axis = 0)
        #         b2avt6_temps = np.delete(b2avt6_temps,(i_en), axis = 0)
        if len(b2avt6_flows) < 1:
            raise Exception('Нет активных потоков')

        mix.train_coef(b2avt6_temps, volumes)
        # b4_mix = (q3[c1_time] * f3[c1_time] + b2_mix * f1[b2_time]) / (f3[c1_time] + f1[b2_time])
        b4_mix = mix.predict(b2avt6_temps, b2avt6_flows, volumes)

        return b4_mix

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
        return self._model

    def load(self, intercept: float, coefs: np.ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesTempGO(Model):
    def __init__(self, alg_smes):
        """ Инициализация модели
        """
        self._alg_smes = alg_smes
        self.intercept_ = 0.0

    def get_delay_time(self, time, flow_rate, vp):
        """
        time - индекс, для которого необходимо рассчитать параметр
        array - numpy массив потока [м3/ч]
        vp - объем трубы [м3]

        return: индекс элемента в массиве, соответсвующий показателю качества time
        """

        # Замена Nan на 0:
        flow_rate = np.nan_to_num(flow_rate)

        if flow_rate.sum() == 0:
            print('Один из потоков неактивен')
            return time
            # raise Exception('Один из потоков неактивен')
        else:
            m = 1
            while flow_rate[time:time + m].sum() < vp:
                m += 1
                if m == 300:
                    break
            delay_time = time + m
            if delay_time > 300:
                delay_time = 300
            return delay_time

    def predict(self, data: np.ndarray, flows: np.ndarray):
        """Запуск прогнозирования модели (по температурам D86 рассчитывает смешение фракционного состава нефтесырья)
        Args
        data (np.ndarray): температуры D86 в потоках фракционного состава
        flow (np.ndarray):  расходы потоков

        Формат данных:
        flows = np.array([11.1, 22.2])  85-120, 120-180
        volumes = np.array([1, 50, 98])
        data = np.array([
            [30.049025, 51.229911, 72.410797], 85-120
            [32.840950, 46.128136, 59.415321], 120-180
        ])

        """
        l1 = 1620
        l2 = 1600
        d1 = (89 - 3.5 * 2) * 10 ** (-3)
        d2 = (89 - 3.5 * 2) * 10 ** (-3)
        volumes = np.array([1, 10, 50, 90, 98])
        mix = mMixture(intercept={1: 0, 10: 0, 50: 0, 90: 0, 98: 0})

        f9946, f9947, f35_11 = flows[0], flows[1], flows[2]
        # f35_11 - 35-11-1000:FR401.F (поток 85-120 и 120-180 на уч-ке 1600 м.)

        v_joint = l2 * (pi * d1 ** 2) / 4
        joint_time = self.get_delay_time(0, f35_11/60, v_joint)  # время смешения потока 85-120 с 120-180
        v_120_180 = (l1 - l2) * (pi * d2 ** 2) / 4  # объем трубы 120-180 на участке в 60 метров
        f_120_180 = f9946 + f9947  # поток фракции 120-180 до смешения
        time_120_180 = self.get_delay_time(joint_time, f_120_180/60, v_120_180)

        mix_temps = np.zeros([2, 5])
        flows = np.zeros(2)
        mix_temps[0, :] = data[joint_time, 0, :]  # 85-120
        mix_temps[1, :] = data[time_120_180, 1, :]  # 120-180
        flows[0] = f35_11[joint_time]  # 85-120
        flows[1] = f_120_180[time_120_180]  # 120-180

        flows_active = flows > 2
        flows = flows[flows_active]
        mix_temps = mix_temps[flows_active]
        # for i_en, fl_i in enumerate(flows):
        #     if fl_i < 2:
        #         flows = np.delete(flows,(i_en), axis = 0)
        #         mix_temps = np.delete(mix_temps,(i_en), axis = 0)
        if len(flows) < 1:
            raise Exception('Нет активных потоков')

        mix.train_coef(mix_temps, volumes)
        mix_quality = mix.predict(mix_temps, flows, volumes)
        return mix_quality

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
        return self._model

    def load(self, intercept: float, coefs: np.ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesTempS250(Model):
    def __init__(self, alg_smes):
        """ Инициализация модели
        """
        self._alg_smes = alg_smes
        self.intercept_ = 0.0

    def get_delay_time(self, time, flow_rate, vp):
        """
        time - индекс, для которого необходимо рассчитать параметр
        array - numpy массив потока [м3/ч]
        vp - объем трубы [м3]

        return: индекс элемента в массиве, соответсвующий показателю качества time
        """

        # Замена Nan на 0:
        flow_rate = np.nan_to_num(flow_rate)

        if flow_rate.sum() == 0:
            print('Один из потоков неактивен')
            return time
            # raise Exception('Один из потоков неактивен')
        else:
            m = 1
            while flow_rate[time:time + m].sum() < vp:
                m += 1
                if m == 300:
                    break
            delay_time = time + m
            if delay_time > 300:
                delay_time = 300
            return delay_time

    def predict(self, data: np.ndarray, flows: np.ndarray):
        """Запуск прогнозирования модели (по температурам D86 рассчитывает смешение фракционного состава нефтесырья)
        Args
        data (np.ndarray): температуры D86 в потоках фракционного состава
        flow (np.ndarray):  расходы потоков

        Формат данных:
        flows = np.array([11.1, 22.2, 33.3])  85-120, 120-180, C-100
        volumes = np.array([1, 50, 98])
        data = np.array([
            [30.049025, 51.229911, 72.410797], 85-120
            [32.840950, 46.128136, 59.415321], 120-180
            [12.840950, 13.128136, 14.415321] C-100
        ])

        """
        l1 = 2020  # 120-180
        l2 = 2000  # 85-120
        d1 = (273 - 8 * 2) * 10 ** (-3)  # АВТ-6

        volumes = np.array([1, 10, 50, 90, 98])
        mix = mMixture(intercept={1: 0, 10: 0, 50: 0, 90: 0, 98: 0})

        f1, f2, f3, f4, f5 = flows[0], flows[1], flows[2], flows[3], flows[4]
        # FQIR9946, FQIR9947, FQIR0025, FI2514, FI2515

        v_joint = l2 * (pi * d1 ** 2) / 4
        joint_time = self.get_delay_time(0, f4/60, v_joint)  # время смешения потока 85-120 с 120-180

        v_120_180 = (l1 - l2) * (pi * d1 ** 2) / 4  # объем трубы 120-180 на участке в 60 метров
        f_120_180 = f1 + f2  # поток фракции 120-180 до смешения
        time_120_180 = self.get_delay_time(joint_time, f_120_180/60, v_120_180)

        mix_temps = np.zeros([3, 5])
        flows = np.zeros(3)
        mix_temps[0, :] = data[time_120_180, 0, :]  # 120-180
        mix_temps[1, :] = data[joint_time, 1, :]  # 85-120
        mix_temps[2, :] = data[0, 2, :]  # С-100
        flows[0] = f_120_180[time_120_180]  # 120-180
        flows[1] = f3[joint_time]  # 120-180
        flows[2] = f5[0]  # С-100

        for i_en, fl_i in enumerate(flows):
            if fl_i < 2:
                flows = np.delete(flows,(i_en), axis = 0)
                mix_temps = np.delete(mix_temps,(i_en), axis = 0)
        if len(flows) < 1:
            raise Exception('Нет активных потоков')

        mix.train_coef(mix_temps, volumes)
        mix_quality = mix.predict(mix_temps, flows, volumes)

        return mix_quality

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
        return self._model

    def load(self, intercept: float, coefs: np.ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass

class SmesTypeModelsPredict():
    def predict(self, data_by_predict: np.ndarray):
        """Запуск прогнозирования смесевой модели, хранящейся в классе SmesTypeModelsPredict

        Args:
        data_by_predict (np.ndarray): данные по всем предикторам модели

        Формат данных:
        data_by_predict = np.array([
            [...],      Date
            [...],      FQIR0025/FQIR0039: [Tnk, T10, T50, T90, Tkk]
            [...],      FQIR9947/FIC2540: [Tnk, T10, T50, T90, Tkk]
            [...],      FI2515/FI1162: [Tnk, T10, T50, T90, Tkk]    (No columns if 'SmesTempGO')
            [...],      Flows
        ])

        """
        if 'SmesTempUILN' in self.model_.__str__():
            temps_np = np.zeros([len(data_by_predict), 3, 5])
            temps_np[:, 0, :] = data_by_predict[:, 0:5] # 'FQIR0039_Tnk': 'FQIR0039_Tkk'
            temps_np[:, 1, :] = data_by_predict[:, 5:10] # 'FIC2540_Tnk': 'FIC2540_Tkk'
            temps_np[:, 2, :] = data_by_predict[:, 10:15] # 'FI1162_Tnk': 'FI1162_Tkk'
            flows_np = data_by_predict[:, 15:].T # 'AVT6:FQI0039_V.F_Model':'KUPN:U100.FIC1162.PV_Model'
            pred = self.model_.predict(temps_np, flows_np) 
                        
        elif 'SmesTempGO' in self.model_.__str__():
            temps_np = np.zeros([len(data_by_predict), 2, 5])
            temps_np[:, 0, :] = data_by_predict[:, 0:5] # 'FQIR0025_Tnk' : 'FQIR0025_Tkk'
            temps_np[:, 1, :] = data_by_predict[:, 5:10] # 'FQIR9947_Tnk' : 'FQIR9947_Tkk'
            flows_np = data_by_predict[:, 10:].T # 'AVT6:FQIR9946.F_Model':'35-11-1000:FR401.F_Model'
            pred = self.model_.predict(temps_np, flows_np) 

        elif 'SmesTempS250' in self.model_.__str__():
            temps_np = np.zeros([len(data_by_predict), 3, 5])
            temps_np[:, 0, :] = data_by_predict[:, 0:5] # 'FQIR0025_Tnk' : 'FQIR0025_Tkk'
            temps_np[:, 1, :] = data_by_predict[:, 5:10] # 'FQIR9947_Tnk': 'FQIR9947_Tkk'
            temps_np[:, 2, :] = data_by_predict[:, 10:15] # 'FI2515_Tnk': 'FI2515_Tkk'
            flows_np = data_by_predict[:, 15:].T # 'AVT6:FQIR9946.F_Model':'FI2515_Flow'
            pred = self.model_.predict(temps_np, flows_np) 
        for key in pred.keys():
            pred[key] = pred.setdefault(key, 0) + self.intercept_
        return pred

    def save(self):
        """ Сохранение модели
            Return: Intercept, Coefficients
        """
        return self.model_

    def load(self, smes_class: Union[SmesTempGO,SmesTempUILN], intercept: float):
        """ 
            smes_class - объект смесевой модели
            intercept - bias модели
        """
        self.model_ = smes_class
        self.model_.intercept_ = 0
        self.intercept_ = intercept
