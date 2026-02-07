from numpy import ndarray, array, where
from pandas import DataFrame, Series
import numpy as np

from models.mMixDupont import Mix_Dupont
from models.mMixEthyl import Mix_Ethyl
from models.mMixBonus import Mix_Bonus
from models.mSmesModel import Smes_model
from models import Model
from models.mOptimizerDual import OptimizerMixingDual, Flows


class OptimizerModel:

    def load(self, count_components: int, list_quality: list):
        self.qp_count = len(list_quality)  # count_quality_product
        self.comp_count = count_components
        # dict_quality_class = {
        #     'Smes_model': Smes_model,
        #     'Mix_Bonus': Mix_Bonus,
        #     'Mix_Ethyl': Mix_Ethyl,
        #     'Mix_Dupont': Mix_Dupont
        # }

        self.smes_models = [model for model in list_quality]

    def taking_data(self, cur_data: ndarray):
        self.target_name = cur_data[0]
        self.tpc_mode = cur_data[1]
        self.total_flow = cur_data[2]
        self.dt = cur_data[3]
        self.tank_volume = cur_data[4]

        i = 5

        self.min_quality_product = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.max_quality_product = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.tank_quality = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.min_flow = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        self.max_flow = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        self.base_quality_flow = []
        for _ in range(self.qp_count):
            self.base_quality_flow.append(cur_data[i : i + self.comp_count])
            i += self.comp_count
        self.base_quality_flow = array(self.base_quality_flow)

        self.current_procent_flow = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        self.component_status = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        for ij, _ in enumerate(self.component_status):
            if self.min_flow[ij] == self.max_flow[ij] == 0:
                self.component_status[ij] = 0
            if self.component_status[ij] == 0:
                self.current_procent_flow[ij] = 0

        self.target_flow = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        self.price_flow = array(cur_data[i : i + self.comp_count])
        i += self.comp_count

        self.or_cost = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.pc_cost = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.pd_option = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.tu_comp_count = int(cur_data[i])
        i += 1

        self.tu_component_status = array(cur_data[i : i + self.comp_count])
        i += self.comp_count
        # for ij, cs in enumerate(self.component_status):
        #     if cs == 0:
        #         self.tu_component_status[ij] = 0

        # if sum(self.tu_component_status) != self.tu_comp_count:
        #     self.tu_comp_count = sum(self.tu_component_status)

        self.tu_mode = 1 if sum(self.tu_component_status) > 0 else 0

        self.comp_quality_t = []
        for _ in range(self.qp_count):
            self.comp_quality_t.append(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count
        self.comp_quality_t = array(self.comp_quality_t)

        self.comp_quality_u = []
        for _ in range(self.qp_count):
            self.comp_quality_u.append(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count
        self.comp_quality_u = array(self.comp_quality_u)

        self.current_procent_flow_t = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.current_procent_flow_u = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.min_flow_t = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.max_flow_t = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.min_flow_u = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.max_flow_u = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.target_flow_t = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.target_flow_u = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.price_flow_t = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.price_flow_u = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.qp_status = array(cur_data[i : i + self.qp_count])
        i += self.qp_count

        self.t_flow_status = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        self.u_flow_status = array(cur_data[i : i + self.tu_comp_count])
        i += self.tu_comp_count

        # for ij, cs in enumerate(self.target_flow_t):
        #     if cs == 0:
        #         self.t_flow_status[ij] = 0
        #         self.u_flow_status[ij] = 1
        #         self.current_procent_flow_t[ij] = 0
        #         self.current_procent_flow_u[ij] = 100

        # for ij, cs in enumerate(self.target_flow_u):
        #     if cs == 0:
        #         self.t_flow_status[ij] = 1
        #         self.u_flow_status[ij] = 0
        #         self.current_procent_flow_t[ij] = 100
        #         self.current_procent_flow_u[ij] = 0

        for ij in range(self.tu_comp_count):
            if self.min_flow_t[ij] == self.max_flow_t[ij] == 0:
                self.current_procent_flow_t[ij] = 0
                # self.t_flow_status[ij] = 0

            if self.min_flow_u[ij] == self.max_flow_u[ij] == 0:
                self.current_procent_flow_u[ij] = 0
                # self.u_flow_status[ij] = 0

        for ij, cs in enumerate(self.t_flow_status):
            if cs == 0:
                self.target_flow_t[ij] = 0
                self.current_procent_flow_t[ij] = 0

        for ij, cs in enumerate(self.u_flow_status):
            if cs == 0:
                self.target_flow_u[ij] = 0
                self.current_procent_flow_u[ij] = 0

        # print(self.t_flow_status,self.u_flow_status, self.target_flow_t, self.target_flow_u, self.current_procent_flow_t, self.current_procent_flow_u)

        # for ij, cs in enumerate(where(self.tu_component_status == 1)[0]):
        #     if self.min_flow_t[ij] == self.max_flow_t[ij] == self.min_flow_u[ij] == self.max_flow_u[ij] == 0:
        #         self.tu_component_status[cs] = 0
        #     if self.t_flow_status[ij] == self.u_flow_status[ij] == 0:
        #         self.tu_component_status[cs] = 0

        if self.target_name > 5:
            # Суммарные ограничения в 2 смесителях
            self.min_flow_volume = array(cur_data[i : i + self.comp_count])
            i += self.comp_count
            self.min_flow_volume_t = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count
            self.min_flow_volume_u = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count
            self.max_flow_volume = array(cur_data[i : i + self.comp_count])
            i += self.comp_count
            self.max_flow_volume_t = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count
            self.max_flow_volume_u = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            # Дополнительные параметры для 2-го смесителя
            self.total_flow_sm2 = cur_data[i]
            i += 1

            self.min_quality_product_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.max_quality_product_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.min_flow_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            self.max_flow_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            self.current_procent_flow_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            self.component_status_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            for ij, _ in enumerate(self.component_status_sm2):
                if self.min_flow_sm2[ij] == self.max_flow_sm2[ij] == 0:
                    self.component_status_sm2[ij] = 0
                if self.component_status_sm2[ij] == 0:
                    self.current_procent_flow_sm2[ij] = 0

            self.target_flow_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            self.price_flow_sm2 = array(cur_data[i : i + self.comp_count])
            i += self.comp_count

            self.or_cost_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.pc_cost_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.pd_option_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.tank_quality_sm2 = array(cur_data[i : i + self.qp_count])
            i += self.qp_count

            self.tank_volume_sm2 = cur_data[i]
            i += 1

            self.current_procent_flow_t_sm2 = array(
                cur_data[i : i + self.tu_comp_count]
            )
            i += self.tu_comp_count

            self.current_procent_flow_u_sm2 = array(
                cur_data[i : i + self.tu_comp_count]
            )
            i += self.tu_comp_count

            self.min_flow_t_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.max_flow_t_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.min_flow_u_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.max_flow_u_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.target_flow_t_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.target_flow_u_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.price_flow_t_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.price_flow_u_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.t_flow_status_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            self.u_flow_status_sm2 = array(cur_data[i : i + self.tu_comp_count])
            i += self.tu_comp_count

            for ij in range(self.tu_comp_count):
                if self.min_flow_t_sm2[ij] == self.max_flow_t_sm2[ij] == 0:
                    self.current_procent_flow_t_sm2[ij] = 0
                    # self.t_flow_status_sm2[ij] = 0

                if self.min_flow_u_sm2[ij] == self.max_flow_u_sm2[ij] == 0:
                    self.current_procent_flow_u_sm2[ij] = 0
                    # self.u_flow_status_sm2[ij] = 0

            for ij, cs in enumerate(self.t_flow_status_sm2):
                if cs == 0:
                    self.target_flow_t_sm2[ij] = 0
                    self.current_procent_flow_t_sm2[ij] = 0

            for ij, cs in enumerate(self.u_flow_status_sm2):
                if cs == 0:
                    self.target_flow_u_sm2[ij] = 0
                    self.current_procent_flow_u_sm2[ij] = 0

    def objective_functions(self, func_index):
        match func_index:
            case 1:
                return "off_spec__giveaway"
            case 2:
                return "off_spec__price"
            case 3:
                return "off_spec__price__giveaway"
            case 4:
                return "off_spec__giveaway__price"
            case 5:
                return "off_spec__variance"
            case 6:
                return "off_spec__giveaway"
            case 7:
                return "off_spec__price"
            case 8:
                return "off_spec__price__giveaway"
            case 9:
                return "off_spec__giveaway__price"
            case 10:
                return "off_spec__variance"

    def predict(self, data: ndarray):

        self.taking_data(data[0])

        # data format:
        # data[0]: 1-5 [ 1: 'off_spec__giveaway', 2: 'off_spec__price', 3: 'off_spec__price__giveaway', 4: 'off_spec__giveaway__price', 5: 'off_spec__variance']
        # data[1]: 0-1 tpc_mode: режим управления смешением [0: 'в смесителе', 1: 'в резервуаре']
        # data[2]: float total_flow: суммарный расход компонентов в м3/ч + 1
        # data[3]: float dt: период расчета в мин
        # data[4]: float tank_volume: объем продукта в товарном резервуаре в м3

        # Дальше данные идут по порядку

        # data:param min_quality_product: минимальное значений показателей качества товарного продукта, [0.72,0,98] + 1
        # data:param max_quality_product: максимальное значений показателей качества товарного продукта,[0.775,9,98.4] + 1
        # data:param tank_quality: качество продукта в товарном резервуаре, [0.7, 0, 96] +1
        # data:param min_flow: минимальное требуемое количество i-го компонента в %, [0, 10, 10, 2, 10, 10] + 1
        # data:param max_flow: максимальное требуемое количество i-го компонента в %, [10, 30, 50, 50, 30, 15] + 1
        # data:param base_quality_flow: текущее значение качества для i-го компонента, [print[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],[96.2,90.4,118.9,92,91.4,115]]
        # data:param current_procent_flow: текущее количество i-го компонента в %, [5, 17, 27, 22, 17, 12] + 1
        # data:param component_status: статус i-го компонента 0-1, [1, 1, 1, 1, 1, 1] + 1
        # data:param target_flow: требуемое количество i-го компонента в %, [ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5] + 1
        # data:param price_flow: цена i-го компонента, [2, 1, 1, 1, 1, 1] + 1
        # data:param or_cost: стоимость нарушения спецификации, [1, 1, 1] + 1
        # data:param pc_cost: стоимость отдачи по качеству, [1, 1, 1] + 1
        # data:param pd_option: вариант расчета отклонения t-го показателя качества от целевого по отдаче по качеству, [0, 1, 0]
        # data:param tu_comp_count: число компонентов, рассчитываемых по поточной схеме
        # data:param tu_component_status: статусы компонентов, рассчитываемых по поточной схеме (по числу всех компонентов)
        # data:param comp_quality_t: текущее значение качества для j-го компонента из РП
        # data:param comp_quality_u: текущее значение качества для j-го компонента с установок
        # data:param current_procent_flow_t: текущее количество j-го компонента из РП в %
        # data:param current_procent_flow_u: текущее количество j-го компонента с установок в %
        # data:param min_flow_t: минимальное требуемое количество j-го компонента из РП в %
        # data:param max_flow_t: максимальное требуемое количество j-го компонента из РП в %
        # data:param min_flow_u: минимальное требуемое количество j-го компонента с установок в %
        # data:param max_flow_u: максимальное требуемое количество j-го компонента с установок в %
        # data:param target_flow_t: требуемое количество j-го компонента из РП
        # data:param target_flow_u: требуемое количество j-го компонента с установок
        # data:param price_flow_t: цена j-го компонента из РП
        # data:param price_flow_u: цена j-го компонента с установок
        # data:param qp_status: статус показателей качества
        # data:param t_flow_status: статусы потоков компонентов, рассчитываемых по поточной схеме (статус резервуара))
        # data:param u_flow_status: статусы потоков компонентов, рассчитываемых по поточной схеме (статус установки))
        flows_list = []
        flows_dict = {}
        SM_1 = Flows()
        SM_1.load(
            min_quality_product=self.min_quality_product,
            max_quality_product=self.max_quality_product,
            min_flow_percent=self.min_flow,
            max_flow_percent=self.max_flow,
            base_quality_flow=self.base_quality_flow,
            current_flow=self.current_procent_flow,
            component_status=self.component_status,
            qp_status=self.qp_status,
            or_cost=self.or_cost,
            pc_cost=self.pc_cost,
            pd_option=self.pd_option,
            tu_mode=self.tu_mode,
            min_flow_percent_t=self.min_flow_t,
            max_flow_percent_t=self.max_flow_t,
            min_flow_percent_u=self.min_flow_u,
            max_flow_percent_u=self.max_flow_u,
            tu_component_status=self.tu_component_status,
            t_flow_status=self.t_flow_status,
            u_flow_status=self.u_flow_status,
            price_flow=self.price_flow,
            price_flow_t=self.price_flow_t,
            price_flow_u=self.price_flow_u,
            comp_quality_t=self.comp_quality_t,
            comp_quality_u=self.comp_quality_u,
            current_procent_flow_t=self.current_procent_flow_t,
            current_procent_flow_u=self.current_procent_flow_u,
            total_flow=self.total_flow,
            tank_volume=self.tank_volume,
            tank_quality=self.tank_quality,
            target_flow=self.target_flow,
            target_flow_t=self.target_flow_t,
            target_flow_u=self.target_flow_u,
        )
        flows_list.append(SM_1)
        if self.target_name > 5:
            SM_2 = Flows()
            SM_2.load(
                min_quality_product=self.min_quality_product_sm2,
                max_quality_product=self.max_quality_product_sm2,
                min_flow_percent=self.min_flow_sm2,
                max_flow_percent=self.max_flow_sm2,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow_sm2,
                component_status=self.component_status_sm2,
                qp_status=self.qp_status,
                or_cost=self.or_cost_sm2,
                pc_cost=self.pc_cost_sm2,
                pd_option=self.pd_option_sm2,
                tu_mode=self.tu_mode,
                min_flow_percent_t=self.min_flow_t_sm2,
                max_flow_percent_t=self.max_flow_t_sm2,
                min_flow_percent_u=self.min_flow_u_sm2,
                max_flow_percent_u=self.max_flow_u_sm2,
                tu_component_status=self.tu_component_status,
                t_flow_status=self.t_flow_status_sm2,
                u_flow_status=self.u_flow_status_sm2,
                price_flow=self.price_flow_sm2,
                price_flow_t=self.price_flow_t_sm2,
                price_flow_u=self.price_flow_u_sm2,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t_sm2,
                current_procent_flow_u=self.current_procent_flow_u_sm2,
                total_flow=self.total_flow_sm2,
                tank_volume=self.tank_volume_sm2,
                tank_quality=self.tank_quality_sm2,
                target_flow=self.target_flow_sm2,
                target_flow_t=self.target_flow_t_sm2,
                target_flow_u=self.target_flow_u_sm2,
            )
            flows_list.append(SM_2)
            flows_dict = {
                "min_flow_volume": self.min_flow_volume,  # - ограничения по объемноу количеству (общее на все смесители)
                "max_flow_volume": self.max_flow_volume,
                "min_flow_volume_t": self.min_flow_volume_t,
                "min_flow_volume_u": self.min_flow_volume_u,
                "max_flow_volume_t": self.max_flow_volume_t,
                "max_flow_volume_u": self.max_flow_volume_u,
            }

        optimizermix = OptimizerMixingDual()
        optimizermix.load(
            models=self.smes_models,
            dt=self.dt,
            flows=flows_list,  # - вставляем объекты смесителей
            tu_mode=self.tu_mode,
            tpc_mode=self.tpc_mode,
            func_type=self.objective_functions(self.target_name),
            vars=flows_dict,
        )

        result_SM = optimizermix.optimize()

        return result_SM

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


# #
# if __name__ == '__main__':


#     obj = OptimizerModel()

#     data = [[
# 6,
#  ]]


#     obj.load(9, [Smes_model()]*13)

#     res = obj.predict(data)


#     print(res)
#     print(len(res))
# #     # print(obj.__dict__)
