from numpy import ndarray, ones_like, nan_to_num, isfinite
from pandas import DataFrame, Series
from typing import Union

from math import pi
from app.models import Model


def test_flow(fl):
    if fl < 2:
        return 0
    else:
        return fl


class SmesUILN(Model):
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
        flow_rate = nan_to_num(flow_rate)

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

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe, ndarray
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        q1 = data[:, 0]
        q2 = data[:, 1]
        q3 = data[:, 2]
        f1 = data[:, 3]
        f2 = data[:, 4]
        f3 = data[:, 5]
        # l1 = data[0, 6]
        # l2 = data[1, 6]
        # l3 = data[2, 6]
        # l4 = data[3, 6]
        # d1 = data[4, 6]
        # d2 = data[5, 6]

        l1 = 812
        l2 = 303
        l3 = 440
        l4 = 1676
        d1 = (219 - 8 * 2) * 10 ** (-3)
        d2 = (159 - 6 * 2) * 10 ** (-3)

        sum_flow_rates = f1 + f2 + f3

        v_b3b4 = l1 * (pi * d1 ** 2) / 4
        b3_time = self.get_delay_time(0, sum_flow_rates/60, v_b3b4)

        f_s100s250 = f1 + f2
        v_b2b3 = l2 * (pi * d2 ** 2) / 4
        b2_time = self.get_delay_time(b3_time, f_s100s250/60, v_b2b3)

        v_a1b2 = l4 * (pi * d2 ** 2) / 4
        a1_time = self.get_delay_time(b2_time, f2/60, v_a1b2)

        f1[b2_time] = test_flow(f1[b2_time])
        f2[a1_time] = test_flow(f2[a1_time])
        if f1[b2_time] + f2[a1_time] > 0:
            b2_mix = (q1[b2_time] * f1[b2_time] + q2[a1_time] * f2[a1_time]) / (f1[b2_time] + f2[a1_time])
        else:
            b2_mix = 0

        v_c1b3 = l3 * (pi * d1 ** 2) / 4
        c1_time = self.get_delay_time(b3_time, f3/60, v_c1b3)

        f3[c1_time] = test_flow(f3[c1_time])
        if f3[c1_time] + f1[b2_time] > 0:
            b4_mix = (q3[c1_time] * f3[c1_time] + b2_mix * f1[b2_time]) / (f3[c1_time] + f1[b2_time])
        else:
            raise Exception('Нет активных потоков')

        return b4_mix + self.intercept_

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

    def load(self, intercept: float, coefs: ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesGO(Model):
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
        flow_rate = nan_to_num(flow_rate)

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

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe, ndarray
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        q1 = data[:, 0]  # ВАК1, 120-180
        q2 = data[:, 1]  # ВАК2, 85-120
        f1 = data[:, 2]  # FQIR9946
        f2 = data[:, 3]  # FQIR9947
        f3 = data[:, 4]  # 35-11-1000:FR401.F (поток 85-120 и 120-180 на уч-ке 1600 м.)
        # l1 = data[0, 6] # Длина 120-180 = 1620
        # l2 = data[1, 6] # Длина 85-120 = 1600
        # d1 = data[4, 6] # Диаметр 120-180 = 89*3.5
        # d2 = data[5, 6] # Диаметр 85-210 = 89*3.5

        l1 = 1620
        l2 = 1600
        d1 = (89 - 3.5 * 2) * 10 ** (-3)
        d2 = (89 - 3.5 * 2) * 10 ** (-3)

        v_joint = l2 * (pi * d1 ** 2) / 4
        joint_time = self.get_delay_time(0, f3/60, v_joint)  # время смешения потока 85-120 с 120-180

        v_120_180 = (l1 - l2) * (pi * d2 ** 2) / 4  # объем трубы 120-180 на участке в 60 метров
        f_120_180 = f1 + f2  # поток фракции 120-180 до смешения
        time_120_180 = self.get_delay_time(joint_time, f_120_180/60, v_120_180)

        f_120_180[time_120_180] = test_flow(f_120_180[time_120_180])
        f3[joint_time] = test_flow(f3[joint_time])
        if f_120_180[time_120_180] + f3[joint_time] > 0:
            mix = (q1[time_120_180] * f_120_180[time_120_180] + q2[joint_time] * f3[joint_time]) / (
                    f_120_180[time_120_180] + f3[joint_time])
        else:
            raise Exception('Нет активных потоков')

        return mix + self.intercept_

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

    def load(self, intercept: float, coefs: ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesS250(Model):
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
        flow_rate = nan_to_num(flow_rate)

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

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe, ndarray
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        q1 = data[:, 0]  # FQIR9947 Качество 120-180
        q2 = data[:, 1]  # FQIR0025 Качество 85-120
        q3 = data[:, 2]  # FI2515_Dens Качество С-100

        f1 = data[:, 3]  # FQIR9946 120-180
        f2 = data[:, 4]  # FQIR9947 120-180
        f3 = data[:, 5]  # FQIR0025 Расход 85-120
        f4 = data[:, 6]  # FI2514 Расход АВТ-6
        f5 = data[:, 6]  # FI2515 Расход С-100

        # •	0025 85-120 АВТ-6 – путь 2000м
        # •	9947+9946 120-180 АВТ-6 – путь 2000м

        l1 = 2020  # 120-180
        l2 = 2000  # 85-120
        d1 = (273 - 8 * 2) * 10 ** (-3)  # АВТ-6

        v_joint = l2 * (pi * d1 ** 2) / 4
        joint_time = self.get_delay_time(0, f4/60, v_joint)  # время смешения потока 85-120 с 120-180

        v_120_180 = (l1 - l2) * (pi * d1 ** 2) / 4  # объем трубы 120-180 на участке в 60 метров
        f_120_180 = f1 + f2  # поток фракции 120-180 до смешения
        time_120_180 = self.get_delay_time(joint_time, f_120_180/60, v_120_180)

        f_120_180[time_120_180] = test_flow(f_120_180[time_120_180])
        f3[joint_time] = test_flow(f3[joint_time])
        f5[0] = test_flow(f5[0])
        if f_120_180[time_120_180] + f3[joint_time] + f5[0] > 0:
            mix = (q1[time_120_180] * f_120_180[time_120_180] + q2[joint_time] * f3[joint_time] + q3[0] * f5[0]) / \
                  (f_120_180[time_120_180] + f3[joint_time] + f5[0])
        else:
            raise Exception('Нет активных потоков')

        return mix + self.intercept_

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

    def load(self, intercept: float, coefs: ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesGFU2С2(Model):
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
        flow_rate = nan_to_num(flow_rate)

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

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe, ndarray
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        q1 = data[:, 0]  # АВТ-6
        # q2 = data[:, 1] # С-200
        q2 = ones_like(q1) * 4  # С-200, данные взяты по лимс
        q3 = data[:, 2]  # С-100
        q4 = data[:, 3]  # ЛЧ-35-11-1000 Блок ГО
        q5 = data[:, 4]  # ЛЧ-35-11-1000 Блок Риформинга

        f1 = data[:, 5]  # AVT6:FQIR0019.F
        f2 = data[:, 6]  # AVT6:FQIR0020.F
        f3 = data[:, 7]  # C-200 KUPN:U200.FIC2017.PV
        f4 = data[:, 8]  # C-100 KUPN:U100.FIC1154.PV

        const_go = 0.5  # !!! ЗАМЕНИТЬ CONST
        const_rif = 1 - const_go
        f5 = const_go * data[:, 9]   # 35-11-1000:FI50.F (суммарный расход) Блок ГО
        f6 = const_rif * data[:, 9]   # ЛЧ-35-11-1000 Блок Риформинга

        favt6 = f1 + f2

        lavt6 = 544
        ls100 = 630
        ls200 = 630
        l3511 = 1366

        davt6 = (100 - 8 * 2) * 10 ** (-3)
        ds100 = (100 - 8 * 2) * 10 ** (-3)
        ds200 = (100 - 8 * 2) * 10 ** (-3)
        d3511 = (89 - 3.5 * 2) * 10 ** (-3)

        v_avt6 = lavt6 * (pi * davt6 ** 2) / 4
        time_avt6 = self.get_delay_time(0, favt6/60, v_avt6)

        v_s100 = ls100 * (pi * ds100 ** 2) / 4
        time_s100 = self.get_delay_time(0, f4/60, v_s100)

        v_s200 = ls200 * (pi * ds200 ** 2) / 4
        time_s200 = self.get_delay_time(0, f3/60, v_s200)

        v_3511 = l3511 * (pi * d3511 ** 2) / 4
        time_3511_go = self.get_delay_time(0, f5/60, v_3511)
        time_3511_rif = self.get_delay_time(0, f6/60, v_3511)

        favt6[time_avt6] = test_flow(favt6[time_avt6])
        f3[time_s200] = test_flow(f3[time_s200])
        f4[time_s100] = test_flow(f4[time_s100])
        f5[time_3511_go] = test_flow(f5[time_3511_go])
        f6[time_3511_rif] = test_flow(f6[time_3511_rif])

        favt6[time_avt6] = test_flow(favt6[time_avt6])
        f3[time_s200] = test_flow(f3[time_s200])
        f4[time_s100] = test_flow(f4[time_s100])
        f5[time_3511_go] = test_flow(f5[time_3511_go])
        f6[time_3511_rif] = test_flow(f6[time_3511_rif])
        if not isfinite(q1[time_avt6]):
            favt6[time_avt6] = 0
            q1[time_avt6] = 0

        if not isfinite(q2[time_s200]):
            f3[time_s200] = 0
            q2[time_s200] = 0

        if not isfinite(q3[time_s100]):
            f4[time_s100] = 0
            q3[time_s100] = 0

        if not isfinite(q4[time_3511_go]):
            f5[time_3511_go] = 0
            q4[time_3511_go] = 0

        if not isfinite(q5[time_3511_rif]):
            f6[time_3511_rif] = 0
            q5[time_3511_rif] = 0


        if favt6[time_avt6] + f3[time_s200] + f4[time_s100] + f5[time_3511_go] + f6[time_3511_rif] > 0:
            mix = (q1[time_avt6] * favt6[time_avt6] + q2[time_s200] * f3[time_s200] + q3[time_s100] * f4[time_s100] +
                   q4[time_3511_go] * f5[time_3511_go] + q5[time_3511_rif] * f6[
                       time_3511_rif]) / \
                  (favt6[time_avt6] + f3[time_s200] + f4[time_s100] + f5[time_3511_go] + f6[time_3511_rif])
        else:
            raise Exception('Нет активных потоков')

        return mix + self.intercept_

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

    def load(self, intercept: float, coefs: ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass


class SmesGFU2С5(Model):
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
        flow_rate = nan_to_num(flow_rate)

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

    def predict(self, data: Union[DataFrame, ndarray]):
        """ data - Данные для прогноза
            dataframe, ndarray
            Запуск прогнозирования качества смеси
            Return: Quality value
        """
        q1 = data[:, 0]  # АВТ-6
        q2 = data[:, 1]  # С-200
        q3 = data[:, 2]  # С-100
        q4 = data[:, 3]  # ЛЧ-35-11-1000 Блок ГО
        q5 = data[:, 4]  # ЛЧ-35-11-1000 Блок Риформинга

        f1 = data[:, 5]  # AVT6:FQIR0019.F
        f2 = data[:, 6]  # AVT6:FQIR0020.F
        f3 = data[:, 7]  # C-200 KUPN:U200.FIC2017.PV
        f4 = data[:, 8]  # C-100 KUPN:U100.FIC1154.PV

        const_go = 0.5  # !!! ЗАМЕНИТЬ CONST
        const_rif = 1 - const_go
        f5 = const_go * data[:, 9]  # 35-11-1000:FI50.F (суммарный расход) Блок ГО
        f6 = const_rif * data[:, 9]  # ЛЧ-35-11-1000 Блок Риформинга

        favt6 = f1 + f2

        lavt6 = 544
        ls100 = 630
        ls200 = 630
        l3511 = 1366

        davt6 = (100 - 8 * 2) * 10 ** (-3)
        ds100 = (100 - 8 * 2) * 10 ** (-3)
        ds200 = (100 - 8 * 2) * 10 ** (-3)
        d3511 = (89 - 3.5 * 2) * 10 ** (-3)

        v_avt6 = lavt6 * (pi * davt6 ** 2) / 4
        time_avt6 = self.get_delay_time(0, favt6/60, v_avt6)

        v_s100 = ls100 * (pi * ds100 ** 2) / 4
        time_s100 = self.get_delay_time(0, f4/60, v_s100)

        v_s200 = ls200 * (pi * ds200 ** 2) / 4
        time_s200 = self.get_delay_time(0, f3/60, v_s200)

        v_3511 = l3511 * (pi * d3511 ** 2) / 4
        time_3511_go = self.get_delay_time(0, f5/60, v_3511)
        time_3511_rif = self.get_delay_time(0, f6/60, v_3511)

        favt6[time_avt6] = test_flow(favt6[time_avt6])
        f3[time_s200] = test_flow(f3[time_s200])
        f4[time_s100] = test_flow(f4[time_s100])
        f5[time_3511_go] = test_flow(f5[time_3511_go])
        f6[time_3511_rif] = test_flow(f6[time_3511_rif])

        if not isfinite(q1[time_avt6]):
            favt6[time_avt6] = 0
            q1[time_avt6] = 0

        if not isfinite(q2[time_s200]):
            f3[time_s200] = 0
            q2[time_s200] = 0

        if not isfinite(q3[time_s100]):
            f4[time_s100] = 0
            q3[time_s100] = 0

        if not isfinite(q4[time_3511_go]):
            f5[time_3511_go] = 0
            q4[time_3511_go] = 0

        if not isfinite(q5[time_3511_rif]):
            f6[time_3511_rif] = 0
            q5[time_3511_rif] = 0

        if favt6[time_avt6] + f3[time_s200] + f4[time_s100] + f5[time_3511_go] + f6[time_3511_rif] > 0:
            mix = (q1[time_avt6] * favt6[time_avt6] + q2[time_s200] * f3[time_s200] +
                   q3[time_s100] * f4[time_s100] + q4[time_3511_go] * f5[time_3511_go] + q5[time_3511_rif] * f6[
                       time_3511_rif]) / \
                  (favt6[time_avt6] + f3[time_s200] + f4[time_s100] + f5[time_3511_go] + f6[time_3511_rif])
        else:
            raise Exception('Нет активных потоков')

        return mix + self.intercept_

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

    def load(self, intercept: float, coefs: ndarray):
        """ intercept - Свободный член модели
            coefs - Коэффициенты при X модели
            Загрузка модели
        """
        pass