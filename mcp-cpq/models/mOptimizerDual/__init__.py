from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from logging import getLogger

# from cobyqa import minimize as minimize_cobyqa
from timers import timeit

# from src.models.mSmesModel import Smes_model
np.random.seed(seed=42)
np.random.RandomState(42)
import random

random.seed(42)
logger = getLogger(__name__)


class Flows:

    def load(
        self,
        min_quality_product: np.array = np.array([None]),
        max_quality_product: np.array = np.array([None]),
        min_flow_percent: np.array = np.array([None]),
        max_flow_percent: np.array = np.array([None]),
        base_quality_flow: np.array = np.array([None]),
        current_flow: np.array = np.array([None]),
        component_status: np.array = np.array([None]),
        qp_status: np.array = np.array([None]),
        total_flow: float = 1,
        target_product: float = 100,
        tank_volume: float = 0,
        tu_mode: float = 0,
        min_flow_percent_t: np.array = np.array([None]),
        max_flow_percent_t: np.array = np.array([None]),
        min_flow_percent_u: np.array = np.array([None]),
        max_flow_percent_u: np.array = np.array([None]),
        #  min_flow_volume: np.array = np.array([0, 0, 10, 2, 10, 10]),
        #  max_flow_volume: np.array = np.array([10, 0, 50, 50, 30, 15]),
        #  min_flow_volume_t: np.array =  np.array([None]),
        #  max_flow_volume_t: np.array =  np.array([None]),
        #  min_flow_volume_u: np.array = np.array([None]),
        #  max_flow_volume_u: np.array = np.array([None]),
        #  total_flow_min: np.array = np.array([None]),
        #  total_flow_max: np.array = np.array([None]),
        tu_component_status: np.array = np.array([None]),
        t_flow_status: np.array = np.array([None]),
        u_flow_status: np.array = np.array([None]),
        comp_quality_t: np.array = np.array([None]),
        comp_quality_u: np.array = np.array([None]),
        current_procent_flow_t: np.array = np.array([None]),
        current_procent_flow_u: np.array = np.array([None]),
        tank_quality: np.array = np.array([None]),
        target_flow: np.array = np.array([None]),
        target_flow_t: np.array = np.array([None]),
        target_flow_u: np.array = np.array([None]),
        price_flow: np.array = np.array([None]),
        price_flow_t: np.array = np.array([None]),
        price_flow_u: np.array = np.array([None]),
        or_cost: np.array = np.array([None]),
        pc_cost: np.array = np.array([None]),
        pd_option: np.array = np.array([None]),
    ):
        """
        _flow - компонент , например Алкилбензин,Бензин кат. крекинга г/о уст. ГО БКК и т.д.
        _product - по итогу смешанный продукт
        Функция заполнения внутренних параметров

            :param min_quality_product: минимальное значений показателей качества товарного продукта, [0.72,0,98]
            :param max_quality_product: максимальное значений показателей качества товарного продукта,[0.775,9,98.4]
            :param or_cost: стоимость нарушения спецификации, [1, 1, 1]
            :param pc_cost: стоимость отдачи по качеству, [1, 1, 1]
            :param price_flow: цена i-го компонента, [2, 1, 1, 1, 1, 1]
            :param price_flow_t: цена j-го компонента из РП
            :param price_flow_u: цена j-го компонента с установок
            :param target_flow: требуемое количество i-го компонента, [ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]
            :param target_flow_t: требуемое количество j-го компонента из РП
            :param target_flow_u: требуемое количество j-го компонента с установок
            :param min_flow_percent: минимальное требуемое количество i-го компонента в %, [0, 10, 10, 2, 10, 10]
            :param max_flow_percent: максимальное требуемое количество i-го компонента в %, [10, 30, 50, 50, 30, 15]

            :param min_flow_percent_t: минимальное требуемое количество j-го компонента из РП в %
            :param max_flow_percent_t: максимальное требуемое количество j-го компонента из РП в %
            :param min_flow_percent_u: минимальное требуемое количество j-го компонента с установок в %
            :param max_flow_percent_u: максимальное требуемое количество j-го компонента с установок в %

            :param base_quality_flow: текущее значение качества для i-го компонента, [[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],[96.2,90.4,118.9,92,91.4,115]]
            :param comp_quality_t: текущее значение качества для j-го компонента из РП
            :param comp_quality_u: текущее значение качества для j-го компонента с установок
            :param current_flow: текущее количество i-го компонента в %, [5, 17, 27, 22, 17, 12]
            :param current_procent_flow_t: текущее количество j-го компонента из РП в %
            :param current_procent_flow_u: текущее количество j-го компонента с установок в %
            :param component_status: статус i-го компонента, [1, 1, 1, 1, 1, 1]
            :param tank_quality: качество продукта в товарном резервуаре
            :param tank_volume: объем продукта в товарном резервуаре в м3

            :param total_flow: суммарный расход компонентов в м3/ч

            :param qp_status: статус показателей качества, [1, 1, 1]

            :param target_product: расход итогового продукта, 1500 (100)

            :param tu_component_status: компоненты, подаваемые по поточной схеме, [0, 0, 0, 0, 0, 0]

            :param pd_option: вариант расчета отклонения t-го показателя качества от целевого по отдаче по качеству, [0, 1, 0]
                pd_option == 0: (Q_t - minQ_t)
                pd_option == 1: (maxQ_t - Q_t)

        """

        self.min_quality_product = min_quality_product.astype("float64")

        if len(min_quality_product) == len(max_quality_product):
            self.max_quality_product = max_quality_product.astype("float64")
        else:
            raise ShapeError("quality parameters", "max_quality_product")

        self.min_flow_percent = min_flow_percent.astype("float64")

        if len(min_flow_percent) == len(max_flow_percent):
            self.max_flow_percent = max_flow_percent.astype("float64")
        else:
            raise ShapeError("components", "max_flow_percent")

        # self.min_flow_volume = min_flow_volume.astype('float64')

        # if len(min_flow_volume) == len(max_flow_volume):
        #     self.max_flow_volume = max_flow_volume.astype('float64')
        # else:
        #     raise ShapeError('components', 'min_flow_volume')
        self.tu_mode = tu_mode
        if base_quality_flow.shape[0] != len(min_quality_product):
            raise ShapeError("quality parameters", "base_quality_flow")
        elif base_quality_flow.shape[1] != len(min_flow_percent):
            raise ShapeError("components", "base_quality_flow")
        else:
            self.base_quality_flow = base_quality_flow.astype("float64")

        if len(min_flow_percent) == len(current_flow):
            self.current_flow = current_flow.astype("float64")
        else:
            raise ShapeError("components", "current_flow")

        if len(min_flow_percent) == len(component_status):
            self.component_status = component_status.astype("float64")
        else:
            raise ShapeError("components", "component_status")

        if len(min_quality_product) == len(qp_status):
            self.qp_status = qp_status.astype("float64")
        else:
            raise ShapeError("quality parameters", "qp_status")

        # self.min_flow_volume_t = min_flow_volume_t.astype('float64')
        # self.max_flow_volume_t = max_flow_volume_t.astype('float64')
        # self.min_flow_volume_u = min_flow_volume_u.astype('float64')
        # self.max_flow_volume_u = max_flow_volume_u.astype('float64')
        # self.total_flow_min = total_flow_min.astype('float64')
        # self.total_flow_max = total_flow_max.astype('float64')
        self.init_min_quality_product = min_quality_product.astype("float64")
        self.init_max_quality_product = max_quality_product.astype("float64")

        # отсекаем ограничения качества по статусу
        self.min_quality_product = min_quality_product[self.qp_status == 1].astype(
            "float64"
        )
        self.max_quality_product = max_quality_product[self.qp_status == 1].astype(
            "float64"
        )

        self.total_flow = total_flow
        self.tank_volume = tank_volume
        self.tank_quality = tank_quality

        self.target_product = target_product
        self.target_flow = target_flow
        self.price_flow = price_flow

        self.tu_component_status = tu_component_status

        if self.tu_mode == 1:
            self.target_flow_t = target_flow_t
            self.target_flow_u = target_flow_u
            self.tu_component_status = tu_component_status
            self.comp_quality_t = comp_quality_t.astype("float64")
            self.comp_quality_u = comp_quality_u.astype("float64")
            self.current_procent_flow_t = current_procent_flow_t.astype("float64")
            self.current_procent_flow_u = current_procent_flow_u.astype("float64")
            self.min_flow_percent_t = min_flow_percent_t.astype("float64")
            self.max_flow_percent_t = max_flow_percent_t.astype("float64")
            self.min_flow_percent_u = min_flow_percent_u.astype("float64")
            self.max_flow_percent_u = max_flow_percent_u.astype("float64")
            self.tu_componect_current_status = (component_status * tu_component_status)[
                np.where(tu_component_status == 1)
            ]
            self.t_flow_status = t_flow_status * self.tu_componect_current_status
            self.u_flow_status = u_flow_status * self.tu_componect_current_status

            self.comp_quality_t = self.comp_quality_t[
                :, np.where(self.t_flow_status == 1)
            ].reshape((self.comp_quality_t.shape[0], int(self.t_flow_status.sum())))
            self.comp_quality_u = self.comp_quality_u[
                :, np.where(self.u_flow_status == 1)
            ].reshape((self.comp_quality_u.shape[0], int(self.u_flow_status.sum())))

            self.price_flow_u = price_flow_t.astype("float64")
            # self.price_flow_u[[4, 5, 6]] = price_flow_u*np.take(tu_component_status, [4, 5, 6])

            self.price_flow_t = price_flow_u.astype("float64")
            # self.price_flow_t[[4, 5, 6]] = price_flow_t*np.take(tu_component_status, [4, 5, 6])
        # else:
        #     self.target_flow_t = np.array([None])
        #     self.target_flow_u = np.array([None])
        #     self.tu_component_status = np.array([None])
        #     self.comp_quality_t = np.array([None])
        #     self.comp_quality_u = np.array([None])
        #     self.current_procent_flow_t = np.array([None])
        #     self.current_procent_flow_u = np.array([None])
        #     self.min_flow_percent_t = np.array([None])
        #     self.max_flow_percent_t = np.array([None])
        #     self.min_flow_percent_u = np.array([None])
        #     self.max_flow_percent_u = np.array([None])
        #     self.t_flow_status=np.array([None])
        #     self.u_flow_status=np.array([None])
        #     self.tu_componect_current_status = np.array([None])

        #     self.price_flow_u = np.array([None])

        #     self.price_flow_t = np.array([None])

        self.pd_option = pd_option
        self.or_cost = or_cost
        self.pc_cost = pc_cost

        # self.t_flow_status = self.tu_component_status.copy()
        # self.t_flow_status[[4,5,6]] = t_flow_status
        # self.u_flow_status = self.tu_component_status.copy()
        # self.u_flow_status[[4,5,6]] = u_flow_status

        # self.comp_quality_t = comp_quality_t * t_flow_status #[:, t_flow_status==1].astype('float64')

        # self.comp_quality_u = comp_quality_u *u_flow_status #[:, u_flow_status==1].astype('float64')

        # self.comp_quality_tu = comp_quality_t.astype('float64')

        # self.current_procent_flow_t = self.current_flow.astype('float64').copy()
        # self.current_procent_flow_t[np.where(t_flow_status==1)] = current_procent_flow_t[t_flow_status==1]

        # self.current_procent_flow_u = self.current_flow.astype('float64').copy()
        # self.current_procent_flow_u[np.where(u_flow_status==1)] = current_procent_flow_u[u_flow_status==1]

        # self.min_flow_percent_t = self.min_flow_percent.copy()
        # self.min_flow_percent_t[np.where(t_flow_status==1)] = min_flow_percent_t[t_flow_status==1]

        # self.max_flow_percent_t = self.min_flow_percent.copy()
        # self.max_flow_percent_t[np.where(t_flow_status==1)] = max_flow_percent_t[t_flow_status==1]

        # self.min_flow_percent_u = self.min_flow_percent.copy()
        # self.min_flow_percent_u[np.where(u_flow_status==1)] = min_flow_percent_u[u_flow_status==1]

        # self.max_flow_percent_u = self.max_flow_percent.copy()
        # self.max_flow_percent_u[np.where(u_flow_status==1)] = max_flow_percent_u[u_flow_status==1]

        # self.min_flow_percent_u = min_flow_percent_u.astype('float64')
        # self.max_flow_percent_u = max_flow_percent_u.astype('float64')
        # self.max_flow_percent_t = max_flow_percent_t.astype('float64')
        # self.min_flow_percent_t = min_flow_percent_t.astype('float64')

        # self.tu_componect_current_status = (component_status*tu_component_status)[np.where(tu_component_status == 1)]
        # self.tu_component_status_only = np.take(tu_component_status, [5, 6, 7]) # ай-ай-ай

    # def build_status_all_x(self, all_flows, t_flows, u_flows):
    #     if self.tu_mode == 0:
    #         x = all_flows.copy()
    #         return x
    #     elif self.tu_mode == 1:
    #         x = []
    #         i_tu = 0
    #         i_all_flows = 0
    #         for status, tu_status in zip(self.component_status, self.tu_component_status):
    #             if tu_status == 0:
    #                 x.append(all_flows[i_all_flows])
    #                 i_all_flows += 1
    #             elif tu_status == 1:
    #                 x.append(t_flows[i_tu])
    #                 x.append(u_flows[i_tu])
    #                 i_all_flows += 1
    #                 i_tu += 1
    #     return np.array(x)

    def build_x(self, all_flows, t_flows, u_flows):
        if self.tu_mode == 0:
            x = all_flows.copy()
            return x[self.component_status == 1]
        elif self.tu_mode == 1:
            x = []
            i_tu = 0
            i_all_flows = 0
            for status, tu_status in zip(
                self.component_status, self.tu_component_status
            ):
                if status == 0:
                    i_all_flows += 1
                    if tu_status == 1:
                        i_tu += 1
                    continue
                elif status == 1:
                    if tu_status == 0:
                        x.append(all_flows[i_all_flows])
                        i_all_flows += 1
                    elif tu_status == 1:
                        if self.t_flow_status[i_tu] == 1:
                            x.append(t_flows[i_tu])
                        if self.u_flow_status[i_tu] == 1:
                            x.append(u_flows[i_tu])
                        i_all_flows += 1
                        i_tu += 1

                # elif status == 1 and tu_status==0:
                #     x.append(all_flows[idx])

                # elif status == 1 and tu_status==1:

                #     if self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 1:
                #         x.append(t_flows[idx])
                #         x.append(u_flows[idx])

                #     elif self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 0:
                #         x.append(t_flows[idx])

                #     elif self.u_flow_status[idx] == 1 and self.t_flow_status[idx] == 0:
                #         x.append(u_flows[idx])

        return np.array(x)

    def extract_tu_flows(self, flows: np.array):
        procent_flow_t = []
        procent_flow_u = []
        i = 0
        i_tu = 0
        for status, tu_status in zip(self.component_status, self.tu_component_status):
            if status == 0:
                if tu_status == 1:
                    i_tu += 1
                continue
            elif status == 1:
                if tu_status == 0:
                    i += 1
                    continue
                elif tu_status == 1:
                    if self.t_flow_status[i_tu] == 1 and self.u_flow_status[i_tu] == 1:
                        procent_flow_t.append(flows[i])
                        procent_flow_u.append(flows[i + 1])
                        i += 2

                    elif (
                        self.t_flow_status[i_tu] == 1 and self.u_flow_status[i_tu] == 0
                    ):
                        procent_flow_t.append(flows[i])
                        i += 1
                    elif (
                        self.t_flow_status[i_tu] == 0 and self.u_flow_status[i_tu] == 1
                    ):
                        procent_flow_u.append(flows[i])
                        i += 1
                i_tu += 1
        # changed_idx = 0
        # for idx, (status, tu_status) in enumerate(zip(self.component_status, self.tu_component_status)):

        #     idx = idx + changed_idx

        #     if status == 0 or (status == 1 and tu_status == 0):
        #         continue

        #     elif status == 1 and tu_status == 1:

        #         if self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 1:
        #             procent_flow_t.append(flows[idx])
        #             procent_flow_u.append(flows[idx+1])
        #             changed_idx = idx + 1

        #         elif self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 0:
        #             procent_flow_t.append(flows[idx])

        #         elif self.t_flow_status[idx] == 0 and self.u_flow_status[idx] == 1:
        #             procent_flow_u.append(flows[idx])

        return np.array(procent_flow_t), np.array(procent_flow_u)

    def tu_comp_q(self, flows: np.array):
        procent_flow_t, procent_flow_u = self.extract_tu_flows(flows)
        comp_q = self.base_quality_flow[
            :, np.where(self.component_status * self.tu_component_status == 1)
        ].reshape(
            (
                self.base_quality_flow.shape[0],
                int((self.component_status * self.tu_component_status == 1).sum()),
            )
        )

        # print('comp_q, t_flow_status, u_flow_status')
        # print(self.t_flow_status,self.u_flow_status)
        # print(comp_q.shape)

        if (
            self.t_flow_status.sum() > 0.0
            and self.u_flow_status.sum() > 0.0
            and (self.t_flow_status * self.u_flow_status).sum() > 0.0
        ):
            # print('smes_tu', procent_flow_t,procent_flow_u,self.comp_quality_t, self.comp_quality_u,self.t_flow_status,self.u_flow_status)

            comp_q = (
                self.comp_quality_t * procent_flow_t
                + self.comp_quality_u * procent_flow_u
            ) / (procent_flow_t + procent_flow_u)
        if self.t_flow_status.sum() == 0.0 and self.u_flow_status.sum() > 0.0:
            ij = 0
            # print('error2')
            for i, cs in enumerate(self.u_flow_status):
                if cs == 1:
                    comp_q[i] = self.comp_quality_u[ij]
                    ij += 1

        if self.u_flow_status.sum() == 0.0 and self.t_flow_status.sum() > 0.0:
            ij = 0
            # print('error2')
            for i, cs in enumerate(self.t_flow_status):
                if cs == 1:
                    comp_q[i] = self.comp_quality_t[ij]
                    ij += 1

        # print(comp_q.shape)
        # print(np.where(self.component_status*self.tu_component_status == 1))
        # print(self.base_quality_flow.shape)
        for i, comp in enumerate(
            np.where(self.component_status * self.tu_component_status == 1)[0]
        ):
            # print(comp_q[:,i])
            # print
            self.base_quality_flow[:, comp] = comp_q[:, i]

        # procent_flow_t, procent_flow_u = self.extract_tu_flows(flows)
        # tmp_quality_t = self.comp_quality_t.copy()
        # tmp_quality_u = self.comp_quality_u.copy()
        # mask_t = self.t_flow_status[self.tu_component_status==1]==1
        # mask_u = self.u_flow_status[self.tu_component_status==1]==1

        # tmp_quality_t[:, mask_t] = self.comp_quality_t[:, mask_t] * procent_flow_t
        # tmp_quality_u[:, mask_u] = self.comp_quality_u[:, mask_u] * procent_flow_u

        # tmp_procent_flow_u = mask_u.astype('float')
        # tmp_procent_flow_t = mask_t.astype('float')
        # tmp_procent_flow_u[tmp_procent_flow_u==1] = procent_flow_u
        # tmp_procent_flow_t[tmp_procent_flow_t==1] = procent_flow_t

        # comp_q = (tmp_quality_t  + tmp_quality_u )/ (tmp_procent_flow_t + tmp_procent_flow_u)

        # for i, comp in enumerate(np.where(self.component_status*self.tu_component_status == 1)[0]):
        #     self.base_quality_flow[:,comp] = comp_q[:,i]

    def get_all_component_status(self, flows: np.array):
        if self.tu_mode == 0:
            all_flows = self.component_status * (1 - self.tu_component_status)
            all_flows[np.where(self.component_status == 1)] = flows
        elif self.tu_mode == 1:
            all_flows = np.ones(self.component_status.shape)
            i_flows = 0
            i_all_flows = 0
            i_tu = 0
            for status, tu_status in zip(
                self.component_status, self.tu_component_status
            ):
                if status == 0:
                    i_all_flows += 1
                    if tu_status == 1:
                        i_tu += 1
                    continue
                elif status == 1:
                    if tu_status == 0:
                        all_flows[i_all_flows] = flows[i_flows]
                        i_all_flows += 1
                        i_flows += 1
                    elif tu_status == 1:
                        if (
                            self.t_flow_status[i_tu] == 1
                            and self.u_flow_status[i_tu] == 1
                        ):
                            # all_flows[i_all_flows] = flows[i_flows] * flows[i_flows+1]
                            i_flows += 2
                        elif (
                            self.t_flow_status[i_tu] == 1
                            and self.u_flow_status[i_tu] == 0
                        ) or (
                            self.t_flow_status[i_tu] == 0
                            and self.u_flow_status[i_tu] == 1
                        ):
                            # all_flows[i_all_flows] = flows[i_flows]
                            i_flows += 1
                        i_all_flows += 1
                        i_tu += 1

        return np.array(all_flows)

    def get_all_flows(self, flows: np.array):
        if self.tu_mode == 0:
            all_flows = self.current_flow * (1 - self.component_status)
            all_flows[np.where(self.component_status == 1)] = flows
        elif self.tu_mode == 1:
            all_flows = self.current_flow * (1 - self.component_status)
            i_flows = 0
            i_all_flows = 0
            i_tu = 0
            for status, tu_status in zip(
                self.component_status, self.tu_component_status
            ):
                if status == 0:
                    i_all_flows += 1
                    if tu_status == 1:
                        i_tu += 1
                    continue
                elif status == 1:
                    if tu_status == 0:
                        all_flows[i_all_flows] = flows[i_flows]
                        i_all_flows += 1
                        i_flows += 1
                    elif tu_status == 1:
                        if (
                            self.t_flow_status[i_tu] == 1
                            and self.u_flow_status[i_tu] == 1
                        ):
                            all_flows[i_all_flows] = flows[i_flows] + flows[i_flows + 1]
                            i_flows += 2
                        elif (
                            self.t_flow_status[i_tu] == 1
                            and self.u_flow_status[i_tu] == 0
                        ) or (
                            self.t_flow_status[i_tu] == 0
                            and self.u_flow_status[i_tu] == 1
                        ):
                            all_flows[i_all_flows] = flows[i_flows]
                            i_flows += 1
                        i_all_flows += 1
                        i_tu += 1
        # if tu_mode == 0:
        #     all_flows = self.current_flow * (1 - self.component_status)
        #     all_flows[np.where(self.component_status == 1)] = flows

        # elif tu_mode == 1:
        #     all_flows = []
        #     changed_idx = 0
        #     for idx, (status, tu_status) in enumerate(zip(self.component_status, self.tu_component_status)):

        #         idx = idx + changed_idx

        #         if status == 0:
        #             idx = 1
        #             continue

        #         elif status == 1 and tu_status == 0:
        #             all_flows.append(flows[idx])

        #         elif status == 1 and tu_status==1:

        #             if self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 1:
        #                 all_flows.append(flows[idx] + flows[idx+1])
        #                 changed_idx = idx+1

        #             elif (self.t_flow_status[idx] == 1 and self.u_flow_status[idx] == 0):
        #                 all_flows.append(flows[idx])

        #             elif (self.t_flow_status[idx] == 0 and self.u_flow_status[idx] == 1):
        #                 all_flows.append(flows[idx])
        return np.array(all_flows)


class ShapeError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
            self.arg = args[1]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"ShapeError, the number of {self.message} in {self.arg} must match"
        else:
            return "ShapeError has been raised"


class OptimizerMixingDual:

    def load(
        self,
        func_type: str,
        dt: float = 1,
        models=[],
        flows: list = [],
        tpc_mode: int = 0,
        tu_mode: int = 1,
        vars: dict = {},
    ) -> None:
        """
        :param func_type: тип функции для оптимизации, 'variance', ['variance', 'price', 'off_spec', 'giveaway']
        :param models: список моделей для расчета качества
        :param dt: период расчета в мин
        :param tpc_mode: режим управления смешением (0 - в смесителе, 1 - в резервуаре)
        :param tu_mode: режим оптимизации по характеру смешения (0 - по одному потоку, 1 - по двум потокам)
        :param flows: объекты смесителей
        :param vars: дополнительные переменные, относящиеся к нескольким смесителям одновременно,
                     min_flow_volume: минимальное допустимое количество i-го компонента из всех смесителей,
                     max_flow_volume: максимальное допустимое количество i-го компонента из всех смесителей
        """

        self.models = models
        self.tu_mode = tu_mode
        self.tpc_mode = tpc_mode
        self.SM_FLOWS = []
        self.dt = dt
        self.min_flow_volume = vars.get("min_flow_volume", None)
        self.max_flow_volume = vars.get("max_flow_volume", None)

        self.min_flow_volume_t = vars.get("min_flow_volume_t", None)
        self.max_flow_volume_t = vars.get("max_flow_volume_t", None)
        self.min_flow_volume_u = vars.get("min_flow_volume_u", None)
        self.max_flow_volume_u = vars.get("max_flow_volume_u", None)

        if len(flows) > 1:
            if len(self.min_flow_volume) == len(self.max_flow_volume):
                self.min_flow_volume = self.min_flow_volume.astype("float64")
                self.max_flow_volume = self.max_flow_volume.astype("float64")
            else:
                raise ShapeError("flow_volume", "min_max_flow_volume")

            if tu_mode == 1:
                if len(self.min_flow_volume_t) == len(self.max_flow_volume_t):
                    self.min_flow_volume_t = self.min_flow_volume_t.astype("float64")
                    self.max_flow_volume_t = self.max_flow_volume_t.astype("float64")
                else:
                    raise ShapeError("flow_volume", "min_max_flow_volume_t")
                if len(self.min_flow_volume_u) == len(self.max_flow_volume_u):
                    self.min_flow_volume_u = self.min_flow_volume_u.astype("float64")
                    self.max_flow_volume_u = self.max_flow_volume_u.astype("float64")
                else:
                    raise ShapeError("flow_volume", "min_max_flow_volume_u")

        # проверка типа целевой функции
        if set(func_type.split("__")) <= set(
            ["variance", "price", "off_spec", "giveaway"]
        ):
            self.func_type = func_type.split("__")
        else:
            raise TargetFuncError()

        for flow in flows:

            if self.tpc_mode == 0:
                flow.total_flow = 1
                flow.dt = 1
                flow.tank_volume = 0
                flow.tank_quality = 0
            else:
                if len(flow.tank_quality) == len(flow.init_min_quality_product):
                    flow.tank_quality = flow.tank_quality.astype("float64")
                else:
                    raise ShapeError("quality parameters", "tank_quality")

            # Оптимизация по регулировке свойств в k-ом смесителе
            if "variance" in self.func_type:
                if len(flow.min_flow_percent) == len(flow.target_flow):
                    flow.target_flow = flow.target_flow.astype("float64")
                else:
                    raise ShapeError("components", "target_flow")
                if flow.tu_mode == 1:
                    flow.target_flow_t = flow.target_flow_t.astype("float64")
                    flow.target_flow_u = flow.target_flow_u.astype("float64")
                flow.target_x_flow = flow.build_x(
                    flow.target_flow, flow.target_flow_t, flow.target_flow_u
                )

            # Оптимизация по минимальной цене в k-ом смесителе
            if "price" in self.func_type:
                if len(flow.min_flow_percent) == len(flow.price_flow):
                    flow.price_flow = flow.price_flow.astype("float64")
                else:
                    raise ShapeError("components", "price_flow")
                if flow.tu_mode == 1:
                    flow.price_flow_t = flow.price_flow_t.astype("float64")
                    flow.price_flow_u = flow.price_flow_u.astype("float64")
                flow.price_x_flow = flow.build_x(
                    flow.price_flow, flow.price_flow_t, flow.price_flow_u
                )

            # Оптимизация по минимальной отдаче по качеству в k-ом смесителе
            if "off_spec" in self.func_type:
                if len(flow.init_min_quality_product) == len(flow.or_cost):
                    flow.or_cost = flow.or_cost.astype("float64")
                else:
                    raise ShapeError("quality parameters", "or_cost")

            # Оптмизация по минимальному отклонению в k-ом смесителе
            if "giveaway" in self.func_type:
                if len(flow.init_min_quality_product) == len(flow.pc_cost):
                    flow.pc_cost = flow.pc_cost.astype("float64")
                else:
                    raise ShapeError("quality parameters", "pc_cost")

                if len(flow.init_min_quality_product) == len(flow.pd_option):
                    flow.pd_option = flow.pd_option.astype("float64")
                else:
                    raise ShapeError("quality parameters", "pd_option")

            self.SM_FLOWS.append(flow)

    def build_real_tu(self, t_flow_status, u_flow_status, flow_t, flow_u, bath=0.0):
        return_tu_flow = []
        # print(result_flow_t, result_flow_u,self.t_flow_status, self.u_flow_status )
        cs = 0
        for j in t_flow_status:
            if j == 0:
                return_tu_flow.append(bath)
            else:
                return_tu_flow.append(flow_t[cs])
                cs += 1
        cs = 0
        for j in u_flow_status:
            if j == 0:
                return_tu_flow.append(bath)
            else:
                return_tu_flow.append(flow_u[cs])
                cs += 1
        return return_tu_flow

    def constraint_quality_product(
        self, s_flows: np.array, i=0, result_status=False
    ) -> list:
        """
        Для соблюдения ограничений по качеству продуктов
            :param flows: список расходов, подобранный оптимизатором
        """

        # if len(flows) > self.limit_len_separated_flows:
        #     # print('constraint_quality_product -> len(flows) > self.limit_len_separated_flows:')
        #     separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        # else:
        #     separated_flows = [flows]

        result_qualities = []
        # for i,s_flows in enumerate(separated_flows):
        # print(i, s_flows)
        all_flows = (
            self.SM_FLOWS[i].get_all_flows(s_flows)
            * self.SM_FLOWS[i].total_flow
            / 100.0
        )

        # пересчет качества компонентов с учетом поточной схемы
        # проверка на TPC нужна?

        # if self.tu_mode == 1:
        if self.SM_FLOWS[i].tu_mode == 1:
            self.SM_FLOWS[i].tu_comp_q(s_flows)
        flow_quality = list()
        for index, model in enumerate(self.models):
            flow_quality.append(
                model.predict(
                    np.concatenate(
                        [
                            np.array([all_flows]),
                            np.array([self.SM_FLOWS[i].base_quality_flow[index]]),
                        ],
                        axis=1,
                    )
                )[0]
            )
        flow_quality = np.array(flow_quality)
        # print(flow_quality,  self.SM_FLOWS[i].total_flow, self.dt,  self.SM_FLOWS[i].tank_quality,  self.SM_FLOWS[i].tank_volume)
        result_quality = (
            flow_quality * self.SM_FLOWS[i].total_flow * self.dt / 60.0
            + self.SM_FLOWS[i].tank_quality * self.SM_FLOWS[i].tank_volume
        ) / (
            self.SM_FLOWS[i].total_flow * self.dt / 60.0 + self.SM_FLOWS[i].tank_volume
        )
        result_qualities.append(result_quality[self.SM_FLOWS[i].qp_status == 1])

        if result_status:
            print("result", i + 1)
            print("flows", all_flows)
            print("flow_quality\n", flow_quality)
            print("result_quality\n", result_quality)
        return np.hstack(result_qualities)

    def constraint_quality_product_sm2(
        self, flows: np.array, result_status=False
    ) -> list:
        """
        Для соблюдения ограничений по качеству продуктов в 2 смесителях одновременно
            :param flows: список расходов, подобранный оптимизатором
        """

        if len(flows) > self.limit_len_separated_flows:
            # print('constraint_quality_product_sm2 -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]

        result_qualities = []
        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                # print(i, s_flows)
                all_flows = (
                    self.SM_FLOWS[i].get_all_flows(s_flows)
                    * self.SM_FLOWS[i].total_flow
                    / 100.0
                )

                # пересчет качества компонентов с учетом поточной схемы
                # проверка на TPC нужна?

                # if self.tu_mode == 1:
                if self.SM_FLOWS[i].tu_mode == 1:
                    self.SM_FLOWS[i].tu_comp_q(s_flows)
                flow_quality = list()
                for index, model in enumerate(self.models):
                    flow_quality.append(
                        model.predict(
                            np.concatenate(
                                [
                                    np.array([all_flows]),
                                    np.array(
                                        [self.SM_FLOWS[i].base_quality_flow[index]]
                                    ),
                                ],
                                axis=1,
                            )
                        )[0]
                    )
                flow_quality = np.array(flow_quality)
                # print(flow_quality,  self.SM_FLOWS[i].total_flow, self.dt,  self.SM_FLOWS[i].tank_quality,  self.SM_FLOWS[i].tank_volume)
                result_quality = (
                    flow_quality * self.SM_FLOWS[i].total_flow * self.dt / 60.0
                    + self.SM_FLOWS[i].tank_quality * self.SM_FLOWS[i].tank_volume
                ) / (
                    self.SM_FLOWS[i].total_flow * self.dt / 60.0
                    + self.SM_FLOWS[i].tank_volume
                )
                result_qualities.append(result_quality[self.SM_FLOWS[i].qp_status == 1])

                if result_status:
                    print("result", i)
                    print("flows", all_flows)
                    print("flow_quality\n", flow_quality)
                    print("result_quality\n", result_quality)
        return np.hstack(result_qualities)

    def constraint_flow_limit(self, flows: np.array):
        """
        Для соблюдения ограничений по количсетву компонентов
            :param flows: список расходов, подобранный оптимизатором
        """
        # print(flows)
        return flows

    def constraint_target_product(self, flows: np.array):
        """
        Для соблюдения ограничения по количеству суммарного продукта, %
            :param flows: список расходов, подобранный оптимизатором
        """

        if len(flows) > self.limit_len_separated_flows:
            # print('constraint_target_product -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]

        sum_flows = []
        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                all_flows = self.SM_FLOWS[i].get_all_flows(s_flows)
                sum_flows.append(all_flows.sum())
        return sum_flows

    def constraint_total_flow_limit(self, flows: np.array):
        """
        Для соблюдения ограничения по количеству вещества из всех смесителей, м3/ч
            :param flows: список расходов, подобранный оптимизатором
        """

        if len(flows) > self.limit_len_separated_flows:
            # print('constraint_total_flow_limit -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
            # print(separated_flows)
        else:
            separated_flows = [flows]

        sum_flows = []

        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                all_flows = self.SM_FLOWS[i].get_all_flows(s_flows)
                procent_flow_t, procent_flow_u = self.SM_FLOWS[i].extract_tu_flows(
                    s_flows
                )
                tu_flows = self.build_real_tu(
                    self.SM_FLOWS[i].t_flow_status,
                    self.SM_FLOWS[i].u_flow_status,
                    procent_flow_t,
                    procent_flow_u,
                )
                all_tu_flows = np.hstack([all_flows, tu_flows])
                # print(all_tu_flows)
                sum_flows.append(all_tu_flows * (self.SM_FLOWS[i].total_flow) / 100)

        return np.vstack(sum_flows).sum(axis=0)[np.where(self.initial_status_flow)]

    def constraint_total_flow_sum(self, flows: np.array):
        """
        Для соблюдения ограничения по суммарному количеству вещества из каждого смесителя, м3/ч
            :param flows: список расходов, подобранный оптимизатором
        """

        if len(flows) > self.limit_len_separated_flows:
            # print('constraint_total_flow_sum -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
            # print(separated_flows)
        else:
            separated_flows = [flows]

        sum_flows = []
        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                all_flows = self.SM_FLOWS[i].get_all_flows(s_flows)
                sum_flows.append(sum(all_flows * (self.SM_FLOWS[i].total_flow) / 100))
        # print(sum_flows)
        return sum_flows

    @timeit
    def optimize(self) -> np.array:
        """
        Для оптимизации, отборов в границах по качеству и количеству
        """
        for objective_func_idx, _ in enumerate(self.func_type):

            self.target_func = self.choose_target_func(objective_func_idx)
            # flow = self.SM_FLOWS[0]

            # self.SM_FLOWS[0]

            if self.tu_mode == 0:
                self.min_flow_percent_constraint = np.vstack(
                    [flow.min_flow_percent.copy() for flow in self.SM_FLOWS]
                )
                self.max_flow_percent_constraint = np.vstack(
                    [flow.max_flow_percent.copy() for flow in self.SM_FLOWS]
                )
                # self.min_flow_volume_constraint = np.vstack([flow.min_flow_volume.copy() for flow in self.SM_FLOWS])
                # self.max_flow_volume_constraint = np.vstack([flow.max_flow_volume.copy() for flow in self.SM_FLOWS])

            elif self.tu_mode == 1:
                self.min_flow_percent_constraint = [
                    flow.build_x(
                        flow.min_flow_percent.copy(),
                        flow.min_flow_percent_t.copy(),
                        flow.min_flow_percent_u.copy(),
                    )
                    for flow in self.SM_FLOWS
                ]
                self.max_flow_percent_constraint = [
                    flow.build_x(
                        flow.max_flow_percent.copy(),
                        flow.max_flow_percent_t.copy(),
                        flow.max_flow_percent_u.copy(),
                    )
                    for flow in self.SM_FLOWS
                ]
                # print(self.min_flow_percent_constraint)
            self.limit_len_separated_flows = self.min_flow_percent_constraint[0].shape[
                0
            ]
            # цель по компонентному составу - сумма всех компонентов 100%
            constraint_target_product = NonlinearConstraint(
                self.constraint_target_product,
                [flow.target_product for flow in self.SM_FLOWS],
                [flow.target_product for flow in self.SM_FLOWS],
            )

            # ограничения по качеству товарного продукта
            constraint_quality_product = NonlinearConstraint(
                self.constraint_quality_product_sm2,
                np.hstack([flow.min_quality_product.copy() for flow in self.SM_FLOWS]),
                np.hstack([flow.max_quality_product.copy() for flow in self.SM_FLOWS]),
            )

            # исходные значения количество i-го компонента в %
            # initial_result_product = np.vstack([flow.current_flow.copy() for flow in self.SM_FLOWS])

            # исходные значения количество j-го компонента из РП и установки в %
            # initial_tu_product = np.vstack([np.hstack([flow.current_procent_flow_t.copy(), flow.current_procent_flow_u.copy()]) for flow in self.SM_FLOWS])

            # initial_tu_product = np.vstack([np.hstack([flow.current_procent_flow_t.copy(), flow.current_procent_flow_u.copy()]) for flow in self.SM_FLOWS])

            initial_status_quality_product = np.vstack(
                [np.ones(flow.min_quality_product.shape) for flow in self.SM_FLOWS]
            )

            initial_tu_status = np.vstack(
                [
                    np.hstack(
                        [
                            np.array([i_t, i_u])
                            for i_t, i_u in zip(flow.t_flow_status, flow.u_flow_status)
                        ]
                    )
                    for flow in self.SM_FLOWS
                ]
            )
            # initial_tu_status = np.vstack([np.hstack([flow.t_flow_status, flow.u_flow_status]) for flow in self.SM_FLOWS])

            # initial_status_flow = np.vstack([np.hstack([ flow.component_status*(1 - flow.tu_component_status), initial_tu_status[idx]]) for idx, flow in enumerate(self.SM_FLOWS)])

            total_tu_status = np.vstack(
                [
                    np.hstack([flow.t_flow_status, flow.u_flow_status])
                    for flow in self.SM_FLOWS
                ]
            )
            self.initial_status_flow = np.vstack(
                [
                    np.hstack(
                        [
                            flow.component_status * (1 - flow.tu_component_status),
                            total_tu_status[idx],
                        ]
                    )
                    for idx, flow in enumerate(self.SM_FLOWS)
                ]
            ).any(axis=0)
            # initial_true_status_flow = np.vstack([np.ones(len(initial_status_flow[idx])) for idx, _ in enumerate(self.SM_FLOWS)])

            # self.min_flow_volume_constraint =  self.SM_FLOWS[0].build_x(self.min_flow_volume.copy(), self.min_flow_volume_t.copy(), self.min_flow_volume_u.copy())
            # self.max_flow_volume_constraint = self.SM_FLOWS[0].build_x(self.max_flow_volume.copy(), self.max_flow_volume_t.copy(), self.max_flow_volume_u.copy())

            # ограничения по компонентному составу
            constraint_flow_limit = NonlinearConstraint(
                self.constraint_flow_limit,
                np.hstack(self.min_flow_percent_constraint.copy()),
                np.hstack(self.max_flow_percent_constraint.copy()),
            )

            # ограничения
            opt_cond = [
                constraint_target_product,
                constraint_quality_product,
                constraint_flow_limit,
            ]

            if len(self.SM_FLOWS) > 1:

                if self.tu_mode == 0:
                    self.min_flow_volume_constraint = self.min_flow_volume.copy()
                    self.max_flow_volume_constraint = self.max_flow_volume.copy()
                elif self.tu_mode == 1:
                    self.min_flow_volume_constraint = np.hstack(
                        [
                            self.min_flow_volume.copy(),
                            self.min_flow_volume_t.copy(),
                            self.min_flow_volume_u.copy(),
                        ]
                    )[np.where(self.initial_status_flow)]
                    self.max_flow_volume_constraint = np.hstack(
                        [
                            self.max_flow_volume.copy(),
                            self.max_flow_volume_t.copy(),
                            self.max_flow_volume_u.copy(),
                        ]
                    )[np.where(self.initial_status_flow)]

                constraint_total_flow_limit = NonlinearConstraint(
                    self.constraint_total_flow_limit,
                    self.min_flow_volume_constraint.copy(),
                    self.max_flow_volume_constraint.copy(),
                )

                constraint_total_flow_sum = NonlinearConstraint(
                    self.constraint_total_flow_sum,
                    np.hstack([flow.total_flow for flow in self.SM_FLOWS]),
                    np.hstack([flow.total_flow for flow in self.SM_FLOWS]),
                )

                opt_cond.extend(
                    [constraint_total_flow_limit, constraint_total_flow_sum]
                )

            res_x = np.hstack(
                [
                    flow.build_x(
                        flow.current_flow.copy(),
                        flow.current_procent_flow_t.copy(),
                        flow.current_procent_flow_u.copy(),
                    )
                    for flow in self.SM_FLOWS
                ]
            )

            # # проверка на ограничение по качеству, если не проходим, то убираем из оптмизации целевую по спецификации
            # current_quality = self.constraint_quality_product(res_x)
            # separated_current_quality = current_quality.reshape(len(self.SM_FLOWS), len(self.SM_FLOWS[0].min_quality_product))
            # for i, scq in enumerate(separated_current_quality):
            #     if len(scq) > 0:
            #         if (scq <= self.SM_FLOWS[i].max_quality_product).all() and (scq >= self.SM_FLOWS[i].min_quality_product).all():
            #             self.func_type.pop(0)
            #             break

            options = {"disp": False}
            if len(self.SM_FLOWS) > 1:
                options["maxiter"] = 250
                options["radius_final"] = 0.00001
                options["final_tr_radius"] = 0.00001
                # options['debug']=True
                options["feasibility_tol"] = 0.00001
                # options['scale']=True
            res = minimize(
                self.target_func,
                res_x,
                method="COBYQA",
                constraints=opt_cond,
                options=options,
            )

            print(res)

            if len(res.x) > self.limit_len_separated_flows:
                # print('result -> len(flows) > self.limit_len_separated_flows:')
                separated_res = np.array_split(res.x, [self.limit_len_separated_flows])
            else:
                separated_res = [res.x]

            if not res.success:
                logger.warning("Оптимум не может быть достигут")
                result = []
                for i, res_sm in enumerate(separated_res):
                    if len(res_sm) > 0:
                        if not (
                            (
                                self.constraint_quality_product(res_sm, i)
                                > self.SM_FLOWS[i].min_quality_product
                            ).all()
                            and (
                                self.constraint_quality_product(res_sm, i)
                                < self.SM_FLOWS[i].max_quality_product
                            ).all()
                        ):
                            logger.warning(
                                f"Не выполняются ограничения (строгое неравенство) по качеству продукта {i+1}-го смесителя\n"
                                + f"quality_product: {self.constraint_quality_product(res_sm,i,result_status=True)}\n"
                                + f"min_quality: {self.SM_FLOWS[i].min_quality_product}\n"
                                + f"min_quality_product: {self.constraint_quality_product(res_sm,i) > self.SM_FLOWS[i].min_quality_product}\n"
                                + f"max_quality_product: {self.constraint_quality_product(res_sm,i) < self.SM_FLOWS[i].max_quality_product}"
                                + f"max_quality: {self.SM_FLOWS[i].max_quality_product}"
                            )
                        if not (
                            (
                                self.constraint_flow_limit(res_sm)
                                > self.min_flow_percent_constraint[i]
                            ).all()
                            and (
                                self.constraint_flow_limit(res_sm)
                                < self.max_flow_percent_constraint[i]
                            ).all()
                        ):
                            logger.warning(
                                f"Не выполняются ограничения (строгое неравенство) по расходам компонентов {i+1}-го смесителя\n"
                                + f"flow: {self.constraint_flow_limit(res_sm)}\n"
                                + f"min_percent: {self.min_flow_percent_constraint[i]}\n"
                                + f"min_flow_percent: {self.constraint_flow_limit(res_sm) > self.min_flow_percent_constraint[i]}\n"
                                + f"max_flow_percent: {self.constraint_flow_limit(res_sm) < self.max_flow_percent_constraint[i]}\n"
                                + f"max_percent: {self.max_flow_percent_constraint[i]}"
                            )

                        # return_tu_flow = []
                        result_flow_t, result_flow_u = self.SM_FLOWS[
                            i
                        ].extract_tu_flows(res_sm)
                        # print(result_flow_t, result_flow_u, self.SM_FLOWS[i].t_flow_status, self.SM_FLOWS[i].u_flow_status )
                        # self.func_post_status(np.ones(len(initial_status_flow[i])), self.constraint_total_flow_limit(res_sm) <  self.max_flow_volume_constraint, initial_status_flow[i])
                        return_tu_flow = self.build_real_tu(
                            self.SM_FLOWS[i].t_flow_status,
                            self.SM_FLOWS[i].u_flow_status,
                            result_flow_t,
                            result_flow_u,
                        )
                        result_flow_t_min, result_flow_u_min = self.SM_FLOWS[
                            i
                        ].extract_tu_flows(
                            self.constraint_flow_limit(res_sm)
                            > self.min_flow_percent_constraint[i]
                        )
                        return_tu_flow_min = self.build_real_tu(
                            self.SM_FLOWS[i].t_flow_status,
                            self.SM_FLOWS[i].u_flow_status,
                            result_flow_t_min,
                            result_flow_u_min,
                            True,
                        )
                        result_flow_t_max, result_flow_u_max = self.SM_FLOWS[
                            i
                        ].extract_tu_flows(
                            self.constraint_flow_limit(res_sm)
                            < self.max_flow_percent_constraint[i]
                        )
                        return_tu_flow_max = self.build_real_tu(
                            self.SM_FLOWS[i].t_flow_status,
                            self.SM_FLOWS[i].u_flow_status,
                            result_flow_t_max,
                            result_flow_u_max,
                            True,
                        )
                        result_component_flow_min = self.SM_FLOWS[
                            i
                        ].get_all_component_status(
                            self.constraint_flow_limit(res_sm)
                            > self.min_flow_percent_constraint[i]
                        )
                        result_component_flow_max = self.SM_FLOWS[
                            i
                        ].get_all_component_status(
                            self.constraint_flow_limit(res_sm)
                            < self.max_flow_percent_constraint[i]
                        )
                        result_status_flow_min = np.hstack(
                            [result_component_flow_min, return_tu_flow_min]
                        )
                        result_status_flow_max = np.hstack(
                            [result_component_flow_max, return_tu_flow_max]
                        )
                        # result_u = self.SM_FLOWS[i].u_flow_status.astype('float').copy()
                        # result_t = self.SM_FLOWS[i].t_flow_status.astype('float').copy()
                        # result_u[result_u==1.0] = result_flow_u
                        # result_t[result_t==1.0] = result_flow_t
                        # return_tu_flow = np.hstack([result_t, result_u])

                        result_status_quality_min = self.func_post_status(
                            initial_status_quality_product[i].copy(),
                            self.constraint_quality_product(res_sm, i)
                            > self.SM_FLOWS[i].min_quality_product,
                            self.SM_FLOWS[i].qp_status,
                        )
                        result_status_quality_max = self.func_post_status(
                            initial_status_quality_product[i].copy(),
                            self.constraint_quality_product(res_sm, i)
                            < self.SM_FLOWS[i].max_quality_product,
                            self.SM_FLOWS[i].qp_status,
                        )
                        # result_status_flow_min = self.func_post_status(initial_true_status_flow[i].copy(), self.constraint_flow_limit(res_sm) > self.min_flow_percent_constraint[i], initial_status_flow[i])
                        # result_status_flow_max = self.func_post_status(initial_true_status_flow[i].copy(), self.constraint_flow_limit(res_sm) < self.max_flow_percent_constraint[i], initial_status_flow[i])
                        # print(result_status_flow_max, "\n",initial_true_status_flow[i],"\n",self.constraint_flow_limit(res_sm), "\n",self.max_flow_percent_constraint[i], "\n",
                        #       self.constraint_flow_limit(res_sm) < self.max_flow_percent_constraint[i],"\n",initial_status_flow[i])

                        result.append(
                            {
                                "status": objective_func_idx,
                                "result_flow_component": self.SM_FLOWS[i].get_all_flows(
                                    flows=res_sm
                                ),
                                # 'result_flow_t_component': self.SM_FLOWS[i].extract_tu_flows(res_sm)[0],
                                # 'result_flow_u_component': self.SM_FLOWS[i].extract_tu_flows(res_sm)[1],
                                "result_flow_tu_component": return_tu_flow,
                                "result_status_quality_min": result_status_quality_min,
                                "result_status_quality_max": result_status_quality_max,
                                "result_status_flow_min": result_status_flow_min,
                                "result_status_flow_max": result_status_flow_max,
                            }
                        )

                if len(self.SM_FLOWS) > 1:
                    result_total_volume = self.constraint_total_flow_limit(res.x)

                    # print('result_total_volume',result_total_volume)
                    # print('min_flow_volume_constraint', self.min_flow_volume_constraint)
                    # print('max_flow_volume_constraint', self.max_flow_volume_constraint)
                    result_status_flow_volume_min = self.func_post_status(
                        np.ones(len(self.initial_status_flow)),
                        result_total_volume > self.min_flow_volume_constraint,
                        self.initial_status_flow,
                    )
                    result_status_flow_volume_max = self.func_post_status(
                        np.ones(len(self.initial_status_flow)),
                        result_total_volume < self.max_flow_volume_constraint,
                        self.initial_status_flow,
                    )

                    if not (
                        (result_total_volume > self.min_flow_volume_constraint).all()
                        and (
                            result_total_volume < self.max_flow_volume_constraint
                        ).all()
                    ):
                        logger.warning(
                            f"Не выполняются ограничения (строгое неравенство) по суммарным расходам компонентов\n"
                            + f"flow: {result_total_volume}\n"
                            + f"min_percent: {self.min_flow_volume_constraint}\n"
                            + f"min_flow_percent: {result_total_volume > self.min_flow_volume_constraint}\n"
                            + f"max_flow_percent: {result_total_volume < self.max_flow_volume_constraint}\n"
                            + f"max_percent: {self.max_flow_volume_constraint}"
                        )
                    rtv = np.zeros(len(self.initial_status_flow))
                    rtv[np.where(self.initial_status_flow)] = result_total_volume
                    result[0]["result_total_volume"] = rtv
                    result[0][
                        "result_status_flow_volume_min"
                    ] = result_status_flow_volume_min
                    result[0][
                        "result_status_flow_volume_max"
                    ] = result_status_flow_volume_max
                else:
                    result_total_volume = []
                    result[0]["result_total_volume"] = []
                    result[0]["result_status_flow_volume_min"] = []
                    result[0]["result_status_flow_volume_max"] = []

                # return result
                # print(result)
                return np.hstack(
                    [np.hstack([r for r in res.values()]) for res in result]
                )

            prev_opt_cond = NonlinearConstraint(self.target_func, -np.inf, res.fun)
            opt_cond.append(prev_opt_cond)
            res_x = res.x

        result = []
        for i, res_sm in enumerate(separated_res):
            if len(res_sm) > 0:
                logger.warning(
                    f"Выполняются ограничения (строгое неравенство) по качеству продукта {i+1}-го смесителя\n"
                    + f"quality_product: {self.constraint_quality_product(res_sm,i,result_status=True)}\n"
                    + f"min_quality: {self.SM_FLOWS[i].min_quality_product}\n"
                    + f"min_quality_product: {self.constraint_quality_product(res_sm,i) > self.SM_FLOWS[i].min_quality_product}\n"
                    + f"max_quality_product: {self.constraint_quality_product(res_sm,i) < self.SM_FLOWS[i].max_quality_product}\n"
                    + f"max_quality: {self.SM_FLOWS[i].max_quality_product}"
                )
                logger.warning(
                    f"Выполняются ограничения (строгое неравенство) по расходам компонентов {i+1}-го смесителя\n"
                    + f"flow: {self.constraint_flow_limit(res_sm)}\n"
                    + f"min_percent: {self.min_flow_percent_constraint[i]}\n"
                    + f"min_flow_percent: {self.constraint_flow_limit(res_sm) > self.min_flow_percent_constraint[i]}\n"
                    + f"max_flow_percent: {self.constraint_flow_limit(res_sm) < self.max_flow_percent_constraint[i]}\n"
                    + f"max_percent: {self.max_flow_percent_constraint[i]}"
                )

                result_status_quality_min = self.func_post_status(
                    initial_status_quality_product[i].copy(),
                    self.constraint_quality_product(res_sm, i)
                    > self.SM_FLOWS[i].min_quality_product,
                    self.SM_FLOWS[i].qp_status,
                )
                result_status_quality_max = self.func_post_status(
                    initial_status_quality_product[i].copy(),
                    self.constraint_quality_product(res_sm, i)
                    < self.SM_FLOWS[i].max_quality_product,
                    self.SM_FLOWS[i].qp_status,
                )
                # all_flows = self.SM_FLOWS[i].get_all_flows(flows = res_x)*self.SM_FLOWS[i].total_flow/100
                # flow_quality = list()
                # for index, model in enumerate(self.models):
                #     print(np.concatenate([np.array([all_flows]), np.array([self.SM_FLOWS[i].base_quality_flow[index]])], axis=1))
                #     flow_quality.append(model.predict(np.concatenate([np.array([all_flows]), np.array([self.SM_FLOWS[i].base_quality_flow[index]])], axis=1))[0])
                # flow_quality = np.array(flow_quality)
                # print('flow_quality',flow_quality.round(2))

                result_flow_t_min, result_flow_u_min = self.SM_FLOWS[
                    i
                ].extract_tu_flows(
                    self.constraint_flow_limit(res_sm)
                    > self.min_flow_percent_constraint[i]
                )
                return_tu_flow_min = self.build_real_tu(
                    self.SM_FLOWS[i].t_flow_status,
                    self.SM_FLOWS[i].u_flow_status,
                    result_flow_t_min,
                    result_flow_u_min,
                    True,
                )
                result_flow_t_max, result_flow_u_max = self.SM_FLOWS[
                    i
                ].extract_tu_flows(
                    self.constraint_flow_limit(res_sm)
                    < self.max_flow_percent_constraint[i]
                )
                return_tu_flow_max = self.build_real_tu(
                    self.SM_FLOWS[i].t_flow_status,
                    self.SM_FLOWS[i].u_flow_status,
                    result_flow_t_max,
                    result_flow_u_max,
                    True,
                )
                result_component_flow_min = self.SM_FLOWS[i].get_all_component_status(
                    self.constraint_flow_limit(res_sm)
                    > self.min_flow_percent_constraint[i]
                )
                result_component_flow_max = self.SM_FLOWS[i].get_all_component_status(
                    self.constraint_flow_limit(res_sm)
                    < self.max_flow_percent_constraint[i]
                )
                result_status_flow_min = np.hstack(
                    [result_component_flow_min, return_tu_flow_min]
                )
                result_status_flow_max = np.hstack(
                    [result_component_flow_max, return_tu_flow_max]
                )
                # result_status_flow_min = self.func_post_status(initial_true_status_flow[i].copy(), self.constraint_flow_limit(res_sm) > self.min_flow_percent_constraint[i], initial_status_flow[i])
                # result_status_flow_max = self.func_post_status(initial_true_status_flow[i].copy(), self.constraint_flow_limit(res_sm) < self.max_flow_percent_constraint[i], initial_status_flow[i])
                result_flow_t, result_flow_u = self.SM_FLOWS[i].extract_tu_flows(res_sm)
                return_tu_flow = self.build_real_tu(
                    self.SM_FLOWS[i].t_flow_status,
                    self.SM_FLOWS[i].u_flow_status,
                    result_flow_t,
                    result_flow_u,
                )

                result.append(
                    {
                        "status": objective_func_idx + 1,
                        "result_flow_component": self.SM_FLOWS[i].get_all_flows(
                            flows=res_sm
                        ),
                        # 'result_flow_t_component': self.SM_FLOWS[i].extract_tu_flows(res_sm)[0],
                        # 'result_flow_u_component': self.SM_FLOWS[i].extract_tu_flows(res_sm)[1],
                        "result_flow_tu_component": return_tu_flow,
                        "result_status_quality_min": result_status_quality_min,
                        "result_status_quality_max": result_status_quality_max,
                        "result_status_flow_min": result_status_flow_min,
                        "result_status_flow_max": result_status_flow_max,
                    }
                )

        if len(self.SM_FLOWS) > 1:
            result_total_volume = self.constraint_total_flow_limit(res.x)
            result_status_flow_volume_min = self.func_post_status(
                np.ones(len(self.initial_status_flow)),
                result_total_volume > self.min_flow_volume_constraint,
                self.initial_status_flow,
            )
            result_status_flow_volume_max = self.func_post_status(
                np.ones(len(self.initial_status_flow)),
                result_total_volume < self.max_flow_volume_constraint,
                self.initial_status_flow,
            )
            logger.warning(
                f"Выполняются ограничения (строгое неравенство) по суммарным расходам компонентов\n"
                + f"flow: {result_total_volume}\n"
                + f"min_percent: {self.min_flow_volume_constraint}\n"
                + f"min_flow_percent: {result_total_volume > self.min_flow_volume_constraint}\n"
                + f"max_flow_percent: {result_total_volume < self.max_flow_volume_constraint}\n"
                + f"max_percent: {self.max_flow_volume_constraint}"
            )
            rtv = np.zeros(len(self.initial_status_flow))
            rtv[np.where(self.initial_status_flow)] = result_total_volume
            result[0]["result_total_volume"] = rtv
            result[0]["result_status_flow_volume_min"] = result_status_flow_volume_min
            result[0]["result_status_flow_volume_max"] = result_status_flow_volume_max
        else:
            result_total_volume = []
            result[0]["result_total_volume"] = []
            result[0]["result_status_flow_volume_min"] = []
            result[0]["result_status_flow_volume_max"] = []

        # return result
        # print(result)
        return np.hstack([np.hstack([r for r in res.values()]) for res in result])

    def choose_target_func(self, i):
        """
        Возвращает нужный тип целевой функции
            :param flows: список расходов, подобранный оптимизатором
        """
        if self.func_type[i] == "variance":
            return self.func_min_variance
        if self.func_type[i] == "price":
            return self.func_min_price
        if self.func_type[i] == "off_spec":
            return self.func_min_off_spec
        if self.func_type[i] == "giveaway":
            return self.func_min_giveaway

    def func_post_status(self, return_status, cur_status, initial_status):
        """
        Для приведения конечных статусов
            :param return_status: список итоговых статусов
            :param cur_status: список активных статусов
            :param initial_status: список начальных статусов

        """
        # print('return_status', 'cur_status', 'initial_status')
        # print(return_status, cur_status, initial_status)
        ics = 0
        for i, bs in enumerate(initial_status):
            if bs:
                return_status[i] = int(cur_status[ics])
                ics += 1
        # print(return_status)
        return return_status

    def func_min_variance(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отколнения от текущей целевой рецептуры
            :param flows: список расходов, подобранный оптимизатором
        """
        if len(flows) > self.limit_len_separated_flows:
            # print('func_min_variance -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]
        sum_sm = []

        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                sum_sm.append(np.sum((s_flows - self.SM_FLOWS[i].target_x_flow) ** 2))

        return sum(sum_sm)

    def func_min_price(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация цены смеси на основе цен компонентов
            :param flows: список расходов, подобранный оптимизатором
        """
        if len(flows) > self.limit_len_separated_flows:
            # print('func_min_price -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]
        sum_sm = []
        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                sum_sm.append(np.sum(s_flows * self.SM_FLOWS[i].price_x_flow))

        return sum(sum_sm)

    def func_min_off_spec(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отклонения свойства от кондиционности
            :param flows: список расходов, подобранный оптимизатором
        """
        # print(flows)

        if len(flows) > self.limit_len_separated_flows:
            # print('func_min_off_spec -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]
        sum_sm = []

        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                curr_Q = self.constraint_quality_product(s_flows, i)
                pd_off = []
                for t, Q_t in enumerate(curr_Q):
                    if Q_t > self.SM_FLOWS[i].max_quality_product[t]:
                        pd_off.append(Q_t - self.SM_FLOWS[i].max_quality_product[t])
                    elif Q_t < self.SM_FLOWS[i].min_quality_product[t]:
                        pd_off.append(self.SM_FLOWS[i].min_quality_product[t] - Q_t)
                    else:
                        pd_off.append(0)

                sum_sm.append(
                    np.sum(
                        np.array(pd_off)
                        * self.SM_FLOWS[i].or_cost[self.SM_FLOWS[i].qp_status == 1]
                    )
                )
        return sum(sum_sm)

    def func_min_giveaway(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отклонения свойств от верхнего или нижнего пределов спецификаций
            :param flows: список расходов, подобранный оптимизатором
        """

        if len(flows) > self.limit_len_separated_flows:
            # print('func_min_giveaway -> len(flows) > self.limit_len_separated_flows:')
            separated_flows = np.array_split(flows, [self.limit_len_separated_flows])
        else:
            separated_flows = [flows]
        sum_sm = []
        for i, s_flows in enumerate(separated_flows):
            if len(s_flows) > 0:
                curr_Q = self.constraint_quality_product(s_flows, i)
                pd_g = []
                for t, Q_t in enumerate(curr_Q):
                    if (
                        Q_t <= self.SM_FLOWS[i].max_quality_product[t]
                        and Q_t >= self.SM_FLOWS[i].min_quality_product[t]
                    ):
                        if self.SM_FLOWS[i].pd_option[t] == 0:
                            pd_g.append(Q_t - self.SM_FLOWS[i].min_quality_product[t])
                        elif self.SM_FLOWS[i].pd_option[t] == 1:
                            pd_g.append(self.SM_FLOWS[i].max_quality_product[t] - Q_t)
                    else:
                        pd_g.append(0)
                sum_sm.append(
                    np.sum(
                        np.array(pd_g)
                        * self.SM_FLOWS[i].pc_cost[self.SM_FLOWS[i].qp_status == 1]
                    )
                )
        return sum(sum_sm)


class TargetFuncError(Exception):
    def __init__(self, *args):
        pass

    def __str__(self):
        return "TargetFuncError, an incorrect target function has been introduced"


# if __name__ == '__main__':


# SM_1 = Flows(  min_quality_product=np.array([0.72, 10, 98]),
#                max_quality_product=np.array([0.775, 9, 98.4]),
#                min_flow_percent=np.array([0, 0, 10, 2, 10, 10]),
#                max_flow_percent=np.array([10, 0, 50, 50, 30, 15]),
#                current_flow = np.array([5, 29, 27, 22, 17, 12]),
#                component_status = np.array([1, 0, 1, 1, 1, 1]),
#                qp_status = np.array([1,0,1]),
#                base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                            [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                or_cost=np.array([1, 1, 1]),
#                min_flow_percent_t = np.array([0]),
#                max_flow_percent_t = np.array([10]),
#                min_flow_percent_u = np.array([0]),
#                max_flow_percent_u = np.array([10]),
#                tu_component_status=np.array([1,0,0,0,0,0]),
#                comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                current_procent_flow_t = np.array([2]),
#                current_procent_flow_u = np.array([3]),
#                tank_quality=np.array([0.6, 0, 96]),
#                 min_flow_volume =  np.array([0, 0, 0, 0, 0, 0]),
#                 max_flow_volume = np.array([1000, 1000, 2000, 1000, 1000, 1000]),
#                 min_flow_volume_t = np.array([0]),
#                 max_flow_volume_t = np.array([10]),
#                 min_flow_volume_u = np.array([0]),
#                 max_flow_volume_u = np.array([10]),
#                 total_flow_min = np.array([7000]),
#                 total_flow_max = np.array([7000]),
#                 total_flow =7000)

# SM_2 = Flows(
#                min_quality_product=np.array([0.72, 10, 98]),
#                max_quality_product=np.array([0.775, 9, 98.4]),
#                min_flow_percent=np.array([0, 0, 10, 2, 10, 10]),
#                max_flow_percent=np.array([10, 0, 50, 50, 30, 15]),
#                current_flow = np.array([5, 29, 27, 22, 17, 12]),
#                component_status = np.array([1, 0, 1, 1, 1, 1]),
#                qp_status = np.array([1,0,1]),
#                base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                            [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                or_cost=np.array([1, 1, 1]),
#                min_flow_percent_t = np.array([0]),
#                max_flow_percent_t = np.array([10]),
#                min_flow_percent_u = np.array([0]),
#                max_flow_percent_u = np.array([10]),
#                tank_quality=np.array([0.6, 0, 96]),
#                tu_component_status=np.array([1,0,0,0,0,0]),
#                comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                current_procent_flow_t = np.array([2]),
#                current_procent_flow_u = np.array([3]),
#                min_flow_volume =  np.array([0, 0, 0, 0, 0, 0]),
#                max_flow_volume = np.array([5000, 2000, 000, 1000, 1000, 1000]),
#                min_flow_volume_t = np.array([0]),
#                max_flow_volume_t = np.array([10]),
#                min_flow_volume_u = np.array([0]),
#                max_flow_volume_u = np.array([10]),
#                total_flow_min = np.array([7000]),
#                total_flow_max = np.array([7000]),
#                total_flow =7000)


# optimizer = OptimizerMixing(models = [Smes_model(), Smes_model(), Smes_model()], dt = 60, flows = [SM_1, SM_2], tu_mode = 1, tpc_mode=1, func_type='off_spec',
#                       vars = {'min_flow_volume': np.array([0, 0, 0, 0, 0, 0]), 'max_flow_volume': np.array([5000, 5000, 1000, 2000, 3000, 4000])})

# result_SM_1, result_SM_2 = optimizer.optimize()

# print('stop')
#     optimizer = OptimizerMixing([Smes_model(), Smes_model(), Smes_model()])
#     print('Target function - off_spec')
#     optimizer.load(#func_type='off_spec',
#    min_quality_product=np.array([0.72, 10, 98]),
#    max_quality_product=np.array([0.775, 9, 98.4]),
#    min_flow_percent=np.array([0, 0, 10, 2, 10, 10]),
#    max_flow_percent=np.array([10, 0, 50, 50, 30, 15]),
#    current_flow = np.array([5, 29, 27, 22, 17, 12]),
#    component_status = np.array([1, 0, 1, 1, 1, 1]),
#    qp_status = np.array([1,0,1]),
#    base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#    or_cost=np.array([1, 1, 1]),
#    #tu_mode=1,
#    min_flow_percent_t = np.array([0]),
#    max_flow_percent_t = np.array([10]),
#    min_flow_percent_u = np.array([0]),
#    max_flow_percent_u = np.array([10]),
#    tu_component_status=np.array([1,0,0,0,0,0]),
#    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#    current_procent_flow_t = np.array([2]),
#    current_procent_flow_u = np.array([3]),
#     min_flow_volume =  np.array([0, 0, 10, 2, 10, 10]),
#     max_flow_volume = np.array([10, 0, 50, 50, 30, 15]),
#     min_flow_volume_t = np.array([0]),
#     max_flow_volume_t = np.array([10]),
#     min_flow_volume_u = np.array([0]),
#     max_flow_volume_u = np.array([10]),
#     flow_to_tank_min = np.array([0]),
#     flow_to_tank_max = np.array([0]))


#     print(optimizer.optimize())
#     print()
# print('Target function - price')
# optimizer.load(func_type='price',
#                min_quality_product=np.array([0.72, 0, 98]),
#                max_quality_product=np.array([0.775, 9, 98.4]),
#                min_flow=np.array([0, 10, 10, 2, 10, 10]),
#                max_flow=np.array([10, 30, 50, 50, 30, 15]),
#                current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                component_status = np.array([1, 1, 1, 1, 1, 1]),
#                qp_status = np.array([1,1,1]),
#                base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                            [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                min_flow_t = np.array([0]),
#                max_flow_t = np.array([10]),
#                min_flow_u = np.array([0]),
#                max_flow_u = np.array([10]),
#                tu_component_status=np.array([1,0,0,0,0,0]),
#                comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                current_procent_flow_t = np.array([2]),
#                current_procent_flow_u = np.array([3]),
#                price_flow=np.array([np.NaN, 1, 1, 1, 1, 1]),
#                price_flow_t = np.array([2]),
#                price_flow_u = np.array([2]))
# print(optimizer.optimize())
#     print()
# print('Target function - variance, tpc_mode = 1') # сходится
# optimizer.load(func_type='variance',
#                min_quality_product=np.array([0.72, 0, 98]),
#                max_quality_product=np.array([0.775, 9, 98.4]),
#                min_flow=np.array([0, 10, 10, 2, 10, 10]),
#                max_flow=np.array([10, 30, 50, 50, 30, 15]),
#                current_flow = np.array([5, 30, 27, 22, 19, 10]),
#                component_status = np.array([1, 0, 1, 1, 1, 1]),
#                qp_status = np.array([1,1,1]),
#                base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                            [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                target_flow=np.array([ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]),
#                target_flow_t=np.array([ 2.4]),
#                target_flow_u=np.array([ 3.4]),
#                total_flow=1,
#                dt=60,
#                tank_volume=1,
#                tpc_mode=1,
#                tank_quality=np.array([0.7, 0, 96]),
#                min_flow_t = np.array([0]),
#                max_flow_t = np.array([10]),
#                min_flow_u = np.array([0]),
#                max_flow_u = np.array([10]),
#                tu_component_status=np.array([1,0,0,0,0,0]),
#                comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                current_procent_flow_t = np.array([2]),
#                current_procent_flow_u = np.array([3]),)
# print(optimizer.optimize())
#     print()
#     print('Target function - giveaway')
#     optimizer.load(func_type='giveaway',
#                    min_quality_product=np.array([0.72, 0, 98]),
#                    max_quality_product=np.array([0.775, 9, 98.4]),
#                    min_flow=np.array([0, 10, 10, 2, 10, 10]),
#                    max_flow=np.array([10, 30, 50, 50, 30, 15]),
#                    current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                    component_status = np.array([1, 1, 1, 1, 1, 1]),
#                    base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                                [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                    pc_cost=np.array([1, 1, 1]),
#                    pd_option=np.array([0, 1, 0]),
#                    min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__giveaway, tpc_mode=1') # не сходится
#     optimizer.load(func_type = 'off_spec__giveaway',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     or_cost = np.array([1, 1, 1]),
#                     pc_cost = np.array([1, 1, 1]),
#                     pd_option = np.array([0, 1, 0]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),
#                    total_flow=1,
#                    dt=60,
#                    tank_volume=1,
#                    tpc_mode=1,
#                    tank_quality=np.array([0.6, 0, 96]))
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__giveaway')
#     optimizer.load(func_type = 'off_spec__giveaway',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     or_cost = np.array([1, 1, 1]),
#                     pc_cost = np.array([1, 1, 1]),
#                     pd_option = np.array([0, 1, 0]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__price')
#     optimizer.load(func_type = 'off_spec__price',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     price_flow = np.array([2, 1, 1, 1, 1, 1]),
#                     price_flow_t = np.array([2]),
#                     price_flow_u = np.array([2]),
#                     or_cost = np.array([1, 1, 1]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__price__giveaway')
#     optimizer.load(func_type = 'off_spec__price__giveaway',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     price_flow = np.array([2, 1, 1, 1, 1, 1]),
#                     price_flow_t = np.array([2]),
#                     price_flow_u = np.array([2]),
#                     or_cost = np.array([1, 1, 1]),
#                     pc_cost = np.array([1, 1, 1]),
#                     pd_option = np.array([0, 1, 0]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__giveaway__price')
#     optimizer.load(func_type = 'off_spec__giveaway__price',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     price_flow = np.array([2, 1, 1, 1, 1, 1]),
#                     price_flow_t = np.array([2]),
#                     price_flow_u = np.array([2]),
#                     or_cost = np.array([1, 1, 1]),
#                     pc_cost = np.array([1, 1, 1]),
#                     pd_option = np.array([0, 1, 0]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__variance')
#     optimizer.load(func_type = 'off_spec__variance',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     or_cost = np.array([1, 1, 1]),
#                     target_flow = np.array([ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]),
#                     target_flow_t=np.array([ 2.4]),
#                     target_flow_u=np.array([ 3.4]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),)
#     print(optimizer.optimize())
#     print()
#     print('Target function - off_spec__variance, tpc_mode=1') # не сходится
#     optimizer.load(func_type = 'off_spec__variance',
#                     min_quality_product = np.array([0.72,0,98]),
#                     max_quality_product = np.array([0.775,9,98.4]),
#                     min_flow = np.array([0, 10, 10, 2, 10, 10]),
#                     max_flow = np.array([10, 30, 50, 50, 30, 15]),
#                     current_flow = np.array([5, 17, 27, 22, 17, 12]),
#                     component_status = np.array([1, 1, 1, 1, 1, 1]),
#                     base_quality_flow = np.array([[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],
#                                                   [96.2,90.4,118.9,92,91.4,115]]),
#                     or_cost = np.array([1, 1, 1]),
#                     target_flow = np.array([ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]),
#                     target_flow_t=np.array([ 2.4]),
#                     target_flow_u=np.array([ 3.4]),
#                     min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]),
#                    total_flow=1,
#                    dt=60,
#                    tank_volume=1,
#                    tpc_mode=1,
#                    tank_quality=np.array([0.6, 0, 96]))
#     print(optimizer.optimize())
#     print()
