from numpy import ndarray, array
from pandas import DataFrame, Series

from models.mMixDupont import Mix_Dupont
from models.mMixEthyl import Mix_Ethyl
from models.mMixBonus import Mix_Bonus
from models.mSmesModel import Smes_model
from models.mOptimizer import OptimizerMixing
from models import Model


class OptimizerModel(Model):

    def load(self, count_components: int, list_quality: list):
        self.qp_count = len(list_quality)  # count_quality_product
        self.comp_count = count_components
        # dict_quality_class = {
        #     'Smes_model': Smes_model,
        #     'Mix_Bonus': Mix_Bonus,
        #     'Mix_Ethyl': Mix_Ethyl,
        #     'Mix_Dupont': Mix_Dupont
        # }
        self.optimizer = OptimizerMixing([model for model in list_quality])

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

    def predict(self, data: ndarray):

        self.taking_data(data[0])

        # data format:
        # data[0]: 1-5 [ 1: 'off_spec__giveaway', 2: 'off_spec__price', 3: 'off_spec__price__giveaway', 4: 'off_spec__giveaway__price', 5: 'off_spec__variance']
        # data[1]: 0-1 tpc_mode: режим управления смешением [0: 'в смесителе', 1: 'в резервуаре']
        # data[2]: float total_flow: суммарный расход компонентов в м3/ч
        # data[3]: float dt: период расчета в мин
        # data[4]: float tank_volume: объем продукта в товарном резервуаре в м3

        # Дальше данные идут по порядку

        # data:param min_quality_product: минимальное значений показателей качества товарного продукта, [0.72,0,98]
        # data:param max_quality_product: максимальное значений показателей качества товарного продукта,[0.775,9,98.4]
        # data:param tank_quality: качество продукта в товарном резервуаре, [0.7, 0, 96]
        # data:param min_flow: минимальное требуемое количество i-го компонента в %, [0, 10, 10, 2, 10, 10]
        # data:param max_flow: максимальное требуемое количество i-го компонента в %, [10, 30, 50, 50, 30, 15]
        # data:param base_quality_flow: текущее значение качества для i-го компонента, [print[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],[96.2,90.4,118.9,92,91.4,115]]
        # data:param current_procent_flow: текущее количество i-го компонента в %, [5, 17, 27, 22, 17, 12]
        # data:param component_status: статус i-го компонента 0-1, [1, 1, 1, 1, 1, 1]
        # data:param target_flow: требуемое количество i-го компонента в %, [ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]
        # data:param price_flow: цена i-го компонента, [2, 1, 1, 1, 1, 1]
        # data:param or_cost: стоимость нарушения спецификации, [1, 1, 1]
        # data:param pc_cost: стоимость отдачи по качеству, [1, 1, 1]
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

        if (
            self.target_name == 1
        ):  # Target function - off_spec__giveaway (Регулировка свойств - Минимальная отдача по качеству)

            self.optimizer.load(
                func_type="off_spec__giveaway",
                min_quality_product=self.min_quality_product,
                max_quality_product=self.max_quality_product,
                min_flow=self.min_flow,
                max_flow=self.max_flow,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow,
                component_status=self.component_status,
                qp_status=self.qp_status,
                or_cost=self.or_cost,
                pc_cost=self.pc_cost,
                pd_option=self.pd_option,
                min_flow_t=self.min_flow_t,
                max_flow_t=self.max_flow_t,
                min_flow_u=self.min_flow_u,
                max_flow_u=self.max_flow_u,
                tu_component_status=self.tu_component_status,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t,
                current_procent_flow_u=self.current_procent_flow_u,
                total_flow=self.total_flow,
                dt=self.dt,
                tank_volume=self.tank_volume,
                tpc_mode=self.tpc_mode,
                tank_quality=self.tank_quality,
            )

        elif (
            self.target_name == 2
        ):  # Target function - off_spec__price (Регулировка свойств - Минимальная цена)

            self.optimizer.load(
                func_type="off_spec__price",
                min_quality_product=self.min_quality_product,
                max_quality_product=self.max_quality_product,
                min_flow=self.min_flow,
                max_flow=self.max_flow,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow,
                component_status=self.component_status,
                qp_status=self.qp_status,
                or_cost=self.or_cost,
                price_flow=self.price_flow,
                price_flow_t=self.price_flow_t,
                price_flow_u=self.price_flow_u,
                min_flow_t=self.min_flow_t,
                max_flow_t=self.max_flow_t,
                min_flow_u=self.min_flow_u,
                max_flow_u=self.max_flow_u,
                tu_component_status=self.tu_component_status,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t,
                current_procent_flow_u=self.current_procent_flow_u,
                total_flow=self.total_flow,
                dt=self.dt,
                tank_volume=self.tank_volume,
                tpc_mode=self.tpc_mode,
                tank_quality=self.tank_quality,
            )

        elif (
            self.target_name == 3
        ):  # Target function - off_spec__price__giveaway (Регулировка свойств - Минимальная цена - Минимальная отдача по качеству)

            self.optimizer.load(
                func_type="off_spec__price__giveaway",
                min_quality_product=self.min_quality_product,
                max_quality_product=self.max_quality_product,
                min_flow=self.min_flow,
                max_flow=self.max_flow,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow,
                component_status=self.component_status,
                qp_status=self.qp_status,
                or_cost=self.or_cost,
                pc_cost=self.pc_cost,
                pd_option=self.pd_option,
                price_flow=self.price_flow,
                price_flow_t=self.price_flow_t,
                price_flow_u=self.price_flow_u,
                min_flow_t=self.min_flow_t,
                max_flow_t=self.max_flow_t,
                min_flow_u=self.min_flow_u,
                max_flow_u=self.max_flow_u,
                tu_component_status=self.tu_component_status,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t,
                current_procent_flow_u=self.current_procent_flow_u,
                total_flow=self.total_flow,
                dt=self.dt,
                tank_volume=self.tank_volume,
                tpc_mode=self.tpc_mode,
                tank_quality=self.tank_quality,
            )

        elif (
            self.target_name == 4
        ):  # Target function - off_spec__giveaway__price (Регулировка свойств - Минимальная отдача по качеству - Минимальная цена)

            self.optimizer.load(
                func_type="off_spec__giveaway__price",
                min_quality_product=self.min_quality_product,
                max_quality_product=self.max_quality_product,
                min_flow=self.min_flow,
                max_flow=self.max_flow,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow,
                component_status=self.component_status,
                qp_status=self.qp_status,
                or_cost=self.or_cost,
                pc_cost=self.pc_cost,
                pd_option=self.pd_option,
                price_flow=self.price_flow,
                price_flow_t=self.price_flow_t,
                price_flow_u=self.price_flow_u,
                min_flow_t=self.min_flow_t,
                max_flow_t=self.max_flow_t,
                min_flow_u=self.min_flow_u,
                max_flow_u=self.max_flow_u,
                tu_component_status=self.tu_component_status,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t,
                current_procent_flow_u=self.current_procent_flow_u,
                total_flow=self.total_flow,
                dt=self.dt,
                tank_volume=self.tank_volume,
                tpc_mode=self.tpc_mode,
                tank_quality=self.tank_quality,
            )

        elif (
            self.target_name == 5
        ):  # Target function - off_spec__variance (Регулировка свойств - Минимальное отклонение)

            self.optimizer.load(
                func_type="off_spec__variance",
                min_quality_product=self.min_quality_product,
                max_quality_product=self.max_quality_product,
                min_flow=self.min_flow,
                max_flow=self.max_flow,
                base_quality_flow=self.base_quality_flow,
                current_flow=self.current_procent_flow,
                component_status=self.component_status,
                qp_status=self.qp_status,
                or_cost=self.or_cost,
                target_flow=self.target_flow,
                target_flow_t=self.target_flow_t,
                target_flow_u=self.target_flow_u,
                min_flow_t=self.min_flow_t,
                max_flow_t=self.max_flow_t,
                min_flow_u=self.min_flow_u,
                max_flow_u=self.max_flow_u,
                tu_component_status=self.tu_component_status,
                comp_quality_t=self.comp_quality_t,
                comp_quality_u=self.comp_quality_u,
                current_procent_flow_t=self.current_procent_flow_t,
                current_procent_flow_u=self.current_procent_flow_u,
                total_flow=self.total_flow,
                dt=self.dt,
                tank_volume=self.tank_volume,
                tpc_mode=self.tpc_mode,
                tank_quality=self.tank_quality,
            )

        return self.optimizer.optimize()

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


#
# if __name__ == '__main__':
#     obj = OptimizerModel()
#     # data = [[1,  # num of func name
#     #          1500,  # target product
#     #          0.72,  # min quality
#     #          0,
#     #          98,
#     #          0.775,  # max quality
#     #          9,
#     #          98.4,
#     #          0,  # min flow
#     #          10,
#     #          10,
#     #          2,
#     #          10,
#     #          10,
#     #          10,  # max flow
#     #          30,
#     #          50,
#     #          50,
#     #          30,
#     #          15,
#     #          0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746,  # base quality flow
#     #          9.9, 7.6, 1, 4, 6.6, 8,
#     #          96.2, 90.4, 118.9, 92, 91.4, 115,
#     #          5.8, 17.4, 37.8,  2.9, 23.6, 12.5,
#     #          # 87, 261, 567, 43.5, 354, 187.5,  # target flow (next - current_flow)
#     #          2, 1, 1, 1, 1, 1,  # target price
#     #          1, 1, 1,  # or cost
#     #          1, 1, 1,  # pc cost
#     #          0, 1, 0  # pd_option
#     #          ]]
#
#     data = [[
#         2
#         , 0
#         , 1
#         , 60
#         , 1
#         , 725
#         , 0
#         , 95
#         , 85
#         , 0
#         , 0
#         , 0
#         , 35
#         , 15
#         , 40
#         , 75
#         , 0
#         , 0
#         , 780
#         , 10
#         , 98.2
#         , 92
#         , 35
#         , 10
#         , 18
#         , 100
#         , 50
#         , 70
#         , 125
#         , 15
#         , 215
#         , 780
#         , 10
#         , 95.2
#         , 86
#         , 35
#         , 1
#         , 18
#         , 100
#         , 50
#         , 70
#         , 125
#         , 15
#         , 215
#         , 1
#         , 0
#         , 0
#         , 0
#         , 5
#         , 10
#         , 2
#         , 0
#         , 0
#         , 10
#         , 1
#         , 1
#         , 1
#         , 30
#         , 60
#         , 35
#         , 1
#         , 1
#         , 753.9365912
#         , 753.9724478
#         , 753.8633231
#         , 753.9571186
#         , 753.7831911
#         , 753.6889731
#         , 754.3453425
#         , 753.6375377
#         , 754.3173264
#         , 6.554528576
#         , 6.385558147
#         , 6.469949172
#         , 6.004240535
#         , 6.762741472
#         , 6.383011389
#         , 6.69982376
#         , 6.857174948
#         , 6.335006837
#         , 96.79451071
#         , 97.00455109
#         , 96.48300146
#         , 96.40629417
#         , 96.27811131
#         , 96.70320345
#         , 96.23215355
#         , 96.83682586
#         , 96.10345304
#         , 86.96935018
#         , 87.19386727
#         , 86.54163915
#         , 86.62485319
#         , 87.03630723
#         , 87.49025565
#         , 86.66819136
#         , 86.85859168
#         , 86.79012747
#         , 19.27540858
#         , 19.22460187
#         , 19.34236068
#         , 18.84022083
#         , 19.32614122
#         , 18.77479069
#         , 19.35539824
#         , 18.93314334
#         , 19.19770602
#         , 1.848538027
#         , 2.35045299
#         , 2.341150099
#         , 2.05655365
#         , 2.499866921
#         , 2.458479774
#         , 1.84343588
#         , 1.770369025
#         , 1.622634929
#         , 10.7456109
#         , 10.45973414
#         , 10.61660091
#         , 10.37323118
#         , 10.55817391
#         , 10.58082039
#         , 10.58203923
#         , 10.85986895
#         , 10.67247834
#         , 68.85482098
#         , 68.5603552
#         , 69.48782165
#         , 69.23135484
#         , 69.42394583
#         , 68.60820712
#         , 69.00628669
#         , 69.20607611
#         , 69.44193661
#         , 33.80595593
#         , 34.48725416
#         , 34.3332852
#         , 34.49759291
#         , 33.89233251
#         , 34.31419815
#         , 34.01550376
#         , 33.95260553
#         , 33.73912853
#         , 56.88105187
#         , 56.24130017
#         , 56.91541706
#         , 56.08607204
#         , 56.28317033
#         , 56.80942581
#         , 56.5509856
#         , 56.68555885
#         , 56.7811311
#         , 101.9404026
#         , 101.29571
#         , 101.1780825
#         , 101.7828819
#         , 101.284618
#         , 101.7776119
#         , 101.0304472
#         , 101.1830755
#         , 101.0156181
#         , 9.068495828
#         , 9.442284609
#         , 8.802394846
#         , 9.196383364
#         , 9.128315866
#         , 9.226295212
#         , 9.052724192
#         , 9.384174884
#         , 9.203201814
#         , 108.7998609
#         , 109.2925464
#         , 108.6390789
#         , 108.7834658
#         , 108.9529094
#         , 108.6124484
#         , 109.4949011
#         , 108.7675265
#         , 108.6979254
#         , 7
#         , 0
#         , 0
#         , 0
#         , 17
#         , 47
#         , 29
#         , 0
#         , 0
#         , 1
#         , 0
#         , 0
#         , 0
#         , 1
#         , 1
#         , 1
#         , 0
#         , 0
#         , 8
#         , 0
#         , 0
#         , 0
#         , 16
#         , 48
#         , 28
#         , 0
#         , 0
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 0
#         , 1
#         , 0
#         , 0
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 0
#         , 1
#         , 1
#         , 3
#         , 0
#         , 0
#         , 0
#         , 0
#         , 1
#         , 1
#         , 1
#         , 0
#         , 0
#         , 753.9365912
#         , 753.9724478
#         , 753.8633231
#         , 6.554528576
#         , 6.385558147
#         , 6.469949172
#         , 96.79451071
#         , 97.00455109
#         , 96.48300146
#         , 86.96935018
#         , 87.19386727
#         , 86.54163915
#         , 19.27540858
#         , 19.22460187
#         , 19.34236068
#         , 1.848538027
#         , 2.35045299
#         , 2.341150099
#         , 10.7456109
#         , 10.45973414
#         , 10.61660091
#         , 68.85482098
#         , 68.5603552
#         , 69.48782165
#         , 33.80595593
#         , 34.48725416
#         , 34.3332852
#         , 56.88105187
#         , 56.24130017
#         , 56.91541706
#         , 101.9404026
#         , 101.29571
#         , 101.1780825
#         , 9.068495828
#         , 9.442284609
#         , 8.802394846
#         , 108.7998609
#         , 109.2925464
#         , 108.6390789
#         , 753.9365912
#         , 753.9724478
#         , 753.8633231
#         , 6.554528576
#         , 6.385558147
#         , 6.469949172
#         , 96.79451071
#         , 97.00455109
#         , 96.48300146
#         , 86.96935018
#         , 87.19386727
#         , 86.54163915
#         , 19.27540858
#         , 19.22460187
#         , 19.34236068
#         , 1.848538027
#         , 2.35045299
#         , 2.341150099
#         , 10.7456109
#         , 10.45973414
#         , 10.61660091
#         , 68.85482098
#         , 68.5603552
#         , 69.48782165
#         , 33.80595593
#         , 34.48725416
#         , 34.3332852
#         , 56.88105187
#         , 56.24130017
#         , 56.91541706
#         , 101.9404026
#         , 101.29571
#         , 101.1780825
#         , 9.068495828
#         , 9.442284609
#         , 8.802394846
#         , 108.7998609
#         , 109.2925464
#         , 108.6390789
#         , 50
#         , 50
#         , 50
#         , 50
#         , 50
#         , 50
#         , 1
#         , 1
#         , 1
#         , 80
#         , 80
#         , 80
#         , 1
#         , 1
#         , 1
#         , 80
#         , 80
#         , 80
#         , 10
#         , 10
#         , 10
#         , 90
#         , 90
#         , 90
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#         , 1
#
#     ]]
#     obj.load(9, [Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model(), Smes_model()])
#     # obj.load(6, ['Smes_model', 'Smes_model', 'Smes_model'])
#
#     # print(obj.__dict__)
#     res = obj.predict(data)
#     print(res)
#     # print(obj.__dict__)
