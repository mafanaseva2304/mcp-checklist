from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from logging import getLogger
from logging import getLogger

from app.models.mSmesModel import Smes_model

logger = getLogger(__name__)
logger = getLogger(__name__)


class OptimizerMixing:
    def __init__(self, models) -> None:
        """
            :param models: список моделей для расчета качества
        """
        self.models = models

    def optimize(self) -> np.array:
        """
        Для оптимизации, отборов в границах по качеству и количеству
        """
        constraint_target_product = NonlinearConstraint(self.constraint_target_product,
                                                        self.target_product,
                                                        self.target_product)
        constraint_quality_product = NonlinearConstraint(self.constraint_quality_product,
                                                         self.min_quality_product,
                                                         self.max_quality_product)

        initial_result_product = self.current_flow.copy()
        initial_tu_product = np.hstack([self.current_procent_flow_t.copy(), self.current_procent_flow_u.copy()])
        initial_status_quality_product = np.ones(self.min_quality_product.shape)
        initial_status_flow = np.hstack(
            [self.component_status * (1 - self.tu_component_status), np.ones(initial_tu_product.shape)])
        initial_true_status_flow = np.ones(initial_status_flow.shape)

        if self.tu_mode == 0:
            self.min_flow_constraint = self.min_flow.copy()
            self.max_flow_constraint = self.max_flow.copy()
        elif self.tu_mode == 1:
            self.min_flow_constraint = self.build_x(self.min_flow, self.min_flow_t, self.min_flow_u)
            self.max_flow_constraint = self.build_x(self.max_flow, self.max_flow_t, self.max_flow_u)
        constraint_flow_limit = NonlinearConstraint(self.constraint_flow_limit, self.min_flow_constraint,
                                                    self.max_flow_constraint)

        opt_cond = [constraint_target_product, constraint_quality_product, constraint_flow_limit]
        res_x = self.build_x(self.current_flow, self.current_procent_flow_t, self.current_procent_flow_u)

        current_quality = self.constraint_quality_product(res_x)
        if (current_quality <= self.max_quality_product).all() and (current_quality >= self.min_quality_product).all():
            self.func_type.pop(0)

        for i, _ in enumerate(self.func_type):
            self.target_func = self.choose_target_func(i)
            res = minimize(self.target_func,
                           res_x,
                           method='SLSQP',
                           constraints=opt_cond,
                           options={'disp': True}, )

            if not res.success:
                logger.warning("Оптимум не может быть достигут")
                if not ((self.constraint_quality_product(res.x) > self.min_quality_product).all() and (
                        self.constraint_quality_product(res.x) < self.max_quality_product).all()):
                    logger.warning("Не выполняются ограничения (строгое неравенство) по качеству продукта\n" +
                                   f"min_quality_product: {self.constraint_quality_product(res.x) > self.min_quality_product}\n" +
                                   f"max_quality_product: {self.constraint_quality_product(res.x) < self.max_quality_product}")
                if not ((self.constraint_flow_limit(res.x) > self.min_flow_constraint).all() and (
                        self.constraint_flow_limit(res.x) < self.max_flow_constraint).all()):
                    logger.warning("Не выполняются ограничения (строгое неравенство) по расходам компонентов\n" +
                                   f"min_flow: {self.constraint_flow_limit(res.x) > self.min_flow_constraint}\n" +
                                   f"max_flow: {self.constraint_flow_limit(res.x) < self.max_flow_constraint}")
                if i == 0:
                    result_status_quality_min = self.func_post_status(initial_status_quality_product.copy(),
                                                                      self.constraint_quality_product(
                                                                          res.x) > self.min_quality_product,
                                                                      self.qp_status)
                    result_status_quality_max = self.func_post_status(initial_status_quality_product.copy(),
                                                                      self.constraint_quality_product(
                                                                          res.x) < self.max_quality_product,
                                                                      self.qp_status)
                    result_status_flow_min = self.func_post_status(initial_true_status_flow.copy(),
                                                                   self.constraint_flow_limit(
                                                                       res.x) > self.min_flow_constraint,
                                                                   initial_status_flow)
                    result_status_flow_max = self.func_post_status(initial_true_status_flow.copy(),
                                                                   self.constraint_flow_limit(
                                                                       res.x) < self.max_flow_constraint,
                                                                   initial_status_flow)
                    return np.hstack([i, initial_result_product, initial_tu_product,
                                      result_status_quality_min,
                                      result_status_quality_max,
                                      result_status_flow_min,
                                      result_status_flow_max
                                      ])
                    # return 0, None
                result_status_quality_min = self.func_post_status(initial_status_quality_product.copy(),
                                                                  self.constraint_quality_product(
                                                                      res.x) > self.min_quality_product, self.qp_status)
                result_status_quality_max = self.func_post_status(initial_status_quality_product.copy(),
                                                                  self.constraint_quality_product(
                                                                      res.x) < self.max_quality_product, self.qp_status)
                result_status_flow_min = self.func_post_status(initial_true_status_flow.copy(),
                                                               self.constraint_flow_limit(
                                                                   res.x) > self.min_flow_constraint,
                                                               initial_status_flow)
                result_status_flow_max = self.func_post_status(initial_true_status_flow.copy(),
                                                               self.constraint_flow_limit(
                                                                   res.x) < self.max_flow_constraint,
                                                               initial_status_flow)
                return np.hstack([i, np.concatenate((self.get_all_flows(res_x), *self.extract_tu_flows(res_x))),
                                  result_status_quality_min,
                                  result_status_quality_max,
                                  result_status_flow_min,
                                  result_status_flow_max
                                  ])

            prev_opt_cond = NonlinearConstraint(self.target_func, -np.inf, res.fun)
            opt_cond.append(prev_opt_cond)
            res_x = res.x
        # logger.warning("Выполняются ограничения (строгое неравенство) по качеству продукта\n"+
        #                             f"quality_product: {self.constraint_quality_product(res.x).round(2)}\n")
        # logger.warning("Выполняются ограничения (строгое неравенство) по расходам компонентов\n"+
        #                            f"flow: {self.constraint_flow_limit(res.x)}\n")

        result_status_quality_min = self.func_post_status(initial_status_quality_product.copy(),
                                                          self.constraint_quality_product(
                                                              res.x) > self.min_quality_product, self.qp_status)
        result_status_quality_max = self.func_post_status(initial_status_quality_product.copy(),
                                                          self.constraint_quality_product(
                                                              res.x) < self.max_quality_product, self.qp_status)
        all_flows = self.get_all_flows(res_x) * self.total_flow / 100
        flow_quality = list()
        for index, model in enumerate(self.models):
            # print(np.concatenate([np.array([all_flows]), np.array([self.base_quality_flow[index]])], axis=1))
            flow_quality.append(model.predict(
                np.concatenate([np.array([all_flows]), np.array([self.base_quality_flow[index]])], axis=1))[0])
        flow_quality = np.array(flow_quality)
        # print('flow_quality',flow_quality.round(2))
        result_status_flow_min = self.func_post_status(initial_true_status_flow.copy(),
                                                       self.constraint_flow_limit(res.x) > self.min_flow_constraint,
                                                       initial_status_flow)
        result_status_flow_max = self.func_post_status(initial_true_status_flow.copy(),
                                                       self.constraint_flow_limit(res.x) < self.max_flow_constraint,
                                                       initial_status_flow)
        return np.hstack([i + 1, np.concatenate((self.get_all_flows(res_x), *self.extract_tu_flows(res_x))),
                          result_status_quality_min,
                          result_status_quality_max,
                          result_status_flow_min,
                          result_status_flow_max
                          ])

    def choose_target_func(self, i):
        """
        Возвращает нужный тип целевой функции
            :param flows: список расходов, подобранный оптимизатором
        """
        if self.func_type[i] == 'variance':
            return self.func_min_variance
        if self.func_type[i] == 'price':
            return self.func_min_price
        if self.func_type[i] == 'off_spec':
            return self.func_min_off_spec
        if self.func_type[i] == 'giveaway':
            return self.func_min_giveaway

    def func_post_status(self, return_status, cur_status, initial_status):
        """
        Для приведения конечных статусов
            :param return_status: список итоговых статусов
            :param cur_status: список активных статусов
            :param initial_status: список начальных статусов

        """
        # print(return_status, cur_status, initial_status)
        ics = 0
        for i, bs in enumerate(initial_status):
            if bs:
                return_status[i] = int(cur_status[ics])
                ics += 1
        return return_status

    def func_min_variance(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отколнения от текущей целевой рецептуры
            :param flows: список расходов, подобранный оптимизатором
        """
        return np.sum((flows - self.target_x_flow) ** 2)

    def func_min_price(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация цены смеси на основе цен компонентов
            :param flows: список расходов, подобранный оптимизатором
        """
        return np.sum(flows * self.price_x_flow)

    def func_min_off_spec(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отклонения свойства от кондиционности
            :param flows: список расходов, подобранный оптимизатором
        """
        curr_Q = self.constraint_quality_product(flows)

        pd_off = []
        for t, Q_t in enumerate(curr_Q):
            if Q_t > self.max_quality_product[t]:
                pd_off.append(Q_t - self.max_quality_product[t])
            elif Q_t < self.min_quality_product[t]:
                pd_off.append(self.min_quality_product[t] - Q_t)
            else:
                pd_off.append(0)
        return np.sum(np.array(pd_off) * self.or_cost[self.qp_status == 1])

    def func_min_giveaway(self, flows: np.array):
        """
        Для расчета целевой функции, минимизация отклонения свойств от верхнего или нижнего пределов спецификаций
            :param flows: список расходов, подобранный оптимизатором
        """
        curr_Q = self.constraint_quality_product(flows)
        pd_g = []
        for t, Q_t in enumerate(curr_Q):
            if Q_t <= self.max_quality_product[t] and Q_t >= self.min_quality_product[t]:
                if self.pd_option[t] == 0:
                    pd_g.append(Q_t - self.min_quality_product[t])
                elif self.pd_option[t] == 1:
                    pd_g.append(self.max_quality_product[t] - Q_t)
            else:
                pd_g.append(0)
        return np.sum(np.array(pd_g) * self.pc_cost[self.qp_status == 1])

    def build_x(self, all_flows, t_flows, u_flows):
        if self.tu_mode == 0:
            x = all_flows.copy()
            return x[self.component_status == 1]
        elif self.tu_mode == 1:
            x = []
            i_tu = 0
            i_all_flows = 0
            for status, tu_status in zip(self.component_status, self.tu_component_status):
                if status == 0:
                    i_all_flows += 1
                    continue
                elif status == 1:
                    if tu_status == 0:
                        x.append(all_flows[i_all_flows])
                        i_all_flows += 1
                    elif tu_status == 1:
                        x.append(t_flows[i_tu])
                        x.append(u_flows[i_tu])
                        i_all_flows += 1
                        i_tu += 1
        return np.array(x)

    def extract_tu_flows(self, flows: np.array):
        procent_flow_t = []
        procent_flow_u = []
        i = 0
        for status, tu_status in zip(self.component_status, self.tu_component_status):
            if status == 0:
                continue
            elif status == 1:
                if tu_status == 0:
                    i += 1
                    continue
                elif tu_status == 1:
                    procent_flow_t.append(flows[i])
                    procent_flow_u.append(flows[i + 1])
                    i += 2
        return np.array(procent_flow_t), np.array(procent_flow_u)

    def tu_comp_q(self, flows: np.array):
        procent_flow_t, procent_flow_u = self.extract_tu_flows(flows)
        comp_q = (self.comp_quality_t * procent_flow_t + self.comp_quality_u * procent_flow_u) / (
                    procent_flow_t + procent_flow_u)
        for i, comp in enumerate(np.where(self.tu_component_status == 1)[0]):
            self.base_quality_flow[:, comp] = comp_q[:, i]

    def get_all_flows(self, flows: np.array):
        if self.tu_mode == 0:
            all_flows = self.current_flow * (1 - self.component_status)
            all_flows[np.where(self.component_status == 1)] = flows
        elif self.tu_mode == 1:
            all_flows = self.current_flow * (1 - self.component_status)
            i_flows = 0
            i_all_flows = 0
            for status, tu_status in zip(self.component_status, self.tu_component_status):
                if status == 0:
                    i_all_flows += 1
                    continue
                elif status == 1:
                    if tu_status == 0:
                        all_flows[i_all_flows] = flows[i_flows]
                        i_all_flows += 1
                        i_flows += 1
                    elif tu_status == 1:
                        all_flows[i_all_flows] = flows[i_flows] + flows[i_flows + 1]
                        i_all_flows += 1
                        i_flows += 2
        return all_flows

    def constraint_quality_product(self, flows: np.array) -> list:
        """
        Для соблюдения ограничений по качеству продуктов
            :param flows: список расходов, подобранный оптимизатором
        """
        all_flows = self.get_all_flows(flows) * self.total_flow / 100.0
        # пересчет качества компонентов с учетом поточной схемы
        if self.tu_mode == 1:
            self.tu_comp_q(flows)
        flow_quality = list()
        for index, model in enumerate(self.models):
            flow_quality.append(model.predict(
                np.concatenate([np.array([all_flows]), np.array([self.base_quality_flow[index]])], axis=1))[0])
        flow_quality = np.array(flow_quality)
        result_quality = (flow_quality * self.total_flow * self.dt / 60.0 + self.tank_quality * self.tank_volume) / (
                    self.total_flow * self.dt / 60.0 + self.tank_volume)
        return result_quality[self.qp_status == 1]

    def constraint_flow_limit(self, flows: np.array):
        """
        Для соблюдения ограничений по количсетву компонентов
            :param flows: список расходов, подобранный оптимизатором
        """
        return flows

    def constraint_target_product(self, flows: np.array):
        """
        Для соблюдения ограничения по количеству суммарного продукта
            :param flows: список расходов, подобранный оптимизатором
        """
        all_flows = self.get_all_flows(flows)
        return all_flows.sum()

    def load(self,
             func_type: str,
             min_quality_product: np.array,
             max_quality_product: np.array,
             min_flow: np.array,
             max_flow: np.array,
             base_quality_flow: np.array,
             current_flow: np.array,
             component_status: np.array,
             qp_status: np.array,
             total_flow: float = 1,
             dt: float = 1,
             target_product: float = 100,
             tank_volume: float = 0,
             tpc_mode: int = 0,
             tu_mode: int = 1,
             min_flow_t: np.array = np.array([None]),
             max_flow_t: np.array = np.array([None]),
             min_flow_u: np.array = np.array([None]),
             max_flow_u: np.array = np.array([None]),
             tu_component_status: np.array = np.array([None]),
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
             pd_option: np.array = np.array([None])
             ):
        """
        _flow - компонент , например Алкилбензин,Бензин кат. крекинга г/о уст. ГО БКК и т.д.
        _product - по итогу смешанный продукт
        Функция заполнения внутренних параметров
            :param func_type: тип функции для оптимизации, 'variance', ['variance', 'price', 'off_spec', 'giveaway']
            :param min_quality_product: минимальное значений показателей качества товарного продукта, [0.72,0,98]
            :param max_quality_product: максимальное значений показателей качества товарного продукта,[0.775,9,98.4]
            :param min_flow: минимальное требуемое количество i-го компонента в %, [0, 10, 10, 2, 10, 10]
            :param max_flow: максимальное требуемое количество i-го компонента в %, [10, 30, 50, 50, 30, 15]
            :param base_quality_flow: текущее значение качества для i-го компонента, [[0.7005,0.742,0.8739,0.585,0.6542,0.746],[9.9,7.6,1,4,6.6,8],[96.2,90.4,118.9,92,91.4,115]]
            :param current_flow: текущее количество i-го компонента в %, [5, 17, 27, 22, 17, 12]
            :param component_status: статус i-го компонента, [1, 1, 1, 1, 1, 1]
            :param qp_status: статус показателей качества, [1, 1, 1]
            :param total_flow: суммарный расход компонентов в м3/ч
            :param dt: период расчета в мин
            :param target_product: расход итогового продукта, 1500 (100)
            :param tank_volume: объем продукта в товарном резервуаре в м3
            :param tpc_mode: режим управления смешением (0 - в смесителе, 1 - в резервуаре)
            :param tu_mode: режим оптимизации по характеру смешения (0 - по одному потоку, 1 - по двум потокам)
            :param min_flow_t: минимальное требуемое количество j-го компонента из РП в %
            :param max_flow_t: максимальное требуемое количество j-го компонента из РП в %
            :param min_flow_u: минимальное требуемое количество j-го компонента с установок в %
            :param max_flow_u: максимальное требуемое количество j-го компонента с установок в %
            :param tu_component_status: компоненты, подаваемые по поточной схеме, [0, 0, 0, 0, 0, 0]
            :param comp_quality_t: текущее значение качества для j-го компонента из РП
            :param comp_quality_u: текущее значение качества для j-го компонента с установок
            :param current_procent_flow_t: текущее количество j-го компонента из РП в %
            :param current_procent_flow_u: текущее количество j-го компонента с установок в %
            :param tank_quality: качество продукта в товарном резервуаре
            :param target_flow: требуемое количество i-го компонента, [ 5.8, 17.4, 37.8,  2.9, 23.6, 12.5]
            :param target_flow_t: требуемое количество j-го компонента из РП
            :param target_flow_u: требуемое количество j-го компонента с установок
            :param price_flow: цена i-го компонента, [2, 1, 1, 1, 1, 1]
            :param price_flow_t: цена j-го компонента из РП
            :param price_flow_u: цена j-го компонента с установок
            :param or_cost: стоимость нарушения спецификации, [1, 1, 1]
            :param pc_cost: стоимость отдачи по качеству, [1, 1, 1]
            :param pd_option: вариант расчета отклонения t-го показателя качества от целевого по отдаче по качеству, [0, 1, 0]
                pd_option == 0: (Q_t - minQ_t)
                pd_option == 1: (maxQ_t - Q_t)
        """

        if set(func_type.split('__')) <= set(['variance', 'price', 'off_spec', 'giveaway']):
            self.func_type = func_type.split('__')
        else:
            raise TargetFuncError()

        self.min_quality_product = min_quality_product.astype('float64')

        if len(min_quality_product) == len(max_quality_product):
            self.max_quality_product = max_quality_product.astype('float64')
        else:
            raise ShapeError('quality parameters', 'max_quality_product')

        self.min_flow = min_flow.astype('float64')

        if len(min_flow) == len(max_flow):
            self.max_flow = max_flow.astype('float64')
        else:
            raise ShapeError('components', 'max_flow')

        if base_quality_flow.shape[0] != len(min_quality_product):
            raise ShapeError('quality parameters', 'base_quality_flow')
        elif base_quality_flow.shape[1] != len(min_flow):
            raise ShapeError('components', 'base_quality_flow')
        else:
            self.base_quality_flow = base_quality_flow.astype('float64')

        self.target_product = target_product

        if len(min_flow) == len(current_flow):
            self.current_flow = current_flow.astype('float64')
        else:
            raise ShapeError('components', 'current_flow')

        if len(min_flow) == len(component_status):
            self.component_status = component_status.astype('float64')
        else:
            raise ShapeError('components', 'component_status')

        if len(min_quality_product) == len(qp_status):
            self.qp_status = qp_status.astype('float64')
        else:
            raise ShapeError('quality parameters', 'qp_status')

        self.min_quality_product = min_quality_product[self.qp_status == 1].astype('float64')
        self.max_quality_product = max_quality_product[self.qp_status == 1].astype('float64')

        self.total_flow = 1
        self.dt = 1
        self.tank_volume = 0
        self.tank_quality = 0

        if tpc_mode == 1:
            self.total_flow = total_flow
            self.dt = dt
            self.tank_volume = tank_volume
            if len(tank_quality) == len(min_quality_product):
                self.tank_quality = tank_quality.astype('float64')
            else:
                raise ShapeError('quality parameters', 'tank_quality')

        self.tu_mode = tu_mode

        if self.tu_mode == 1:
            self.tu_component_status = tu_component_status
            self.comp_quality_t = comp_quality_t.astype('float64')
            self.comp_quality_u = comp_quality_u.astype('float64')
            self.current_procent_flow_t = current_procent_flow_t.astype('float64')
            self.current_procent_flow_u = current_procent_flow_u.astype('float64')
            self.min_flow_t = min_flow_t.astype('float64')
            self.max_flow_t = max_flow_t.astype('float64')
            self.min_flow_u = min_flow_u.astype('float64')
            self.max_flow_u = max_flow_u.astype('float64')

        if 'variance' in self.func_type:
            if len(min_flow) == len(target_flow):
                self.target_flow = target_flow.astype('float64')
            else:
                raise ShapeError('components', 'target_flow')
            if self.tu_mode == 1:
                self.target_flow_t = target_flow_t.astype('float64')
                self.target_flow_u = target_flow_u.astype('float64')
            self.target_x_flow = self.build_x(self.target_flow, self.target_flow_t, self.target_flow_u)

        if 'price' in self.func_type:
            if len(min_flow) == len(price_flow):
                self.price_flow = price_flow.astype('float64')
            else:
                raise ShapeError('components', 'price_flow')
            if self.tu_mode == 1:
                self.price_flow_t = price_flow_t.astype('float64')
                self.price_flow_u = price_flow_u.astype('float64')
            self.price_x_flow = self.build_x(self.price_flow, self.price_flow_t, self.price_flow_u)

        if 'off_spec' in self.func_type:
            if len(min_quality_product) == len(or_cost):
                self.or_cost = or_cost.astype('float64')
            else:
                raise ShapeError('quality parameters', 'or_cost')

        if 'giveaway' in self.func_type:
            if len(min_quality_product) == len(pc_cost):
                self.pc_cost = pc_cost.astype('float64')
            else:
                raise ShapeError('quality parameters', 'pc_cost')

            if len(min_quality_product) == len(pd_option):
                self.pd_option = pd_option.astype('float64')
            else:
                raise ShapeError('quality parameters', 'pd_option')


class ShapeError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
            self.arg = args[1]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f'ShapeError, the number of {self.message} in {self.arg} must match'
        else:
            return 'ShapeError has been raised'


class TargetFuncError(Exception):
    def __init__(self, *args):
        pass

    def __str__(self):
        return 'TargetFuncError, an incorrect target function has been introduced'

# if __name__ == '__main__':
#     optimizer = OptimizerMixing([Smes_model(), Smes_model(), Smes_model()])
#     print('Target function - off_spec')
#     optimizer.load(func_type='off_spec',
#                    min_quality_product=np.array([0.72, 10, 98]),
#                    max_quality_product=np.array([0.775, 9, 98.4]),
#                    min_flow=np.array([0, 0, 10, 2, 10, 10]),
#                    max_flow=np.array([10, 0, 50, 50, 30, 15]),
#                    current_flow = np.array([5, 29, 27, 22, 17, 12]),
#                    component_status = np.array([1, 0, 1, 1, 1, 1]),
#                    qp_status = np.array([1,0,1]),
#                    base_quality_flow=np.array([[0.7005, 0.742, 0.8739, 0.585, 0.6542, 0.746], [9.9, 7.6, 1, 4, 6.6, 8],
#                                                [96.2, 90.4, 118.9, 92, 91.4, 115]]),
#                    or_cost=np.array([1, 1, 1]),
#                    tu_mode=1,
#                    min_flow_t = np.array([0]),
#                    max_flow_t = np.array([10]),
#                    min_flow_u = np.array([0]),
#                    max_flow_u = np.array([10]),
#                    tu_component_status=np.array([1,0,0,0,0,0]),
#                    comp_quality_t = np.array([[0.7005], [9.9], [96.2]]),
#                    comp_quality_u = np.array([[0.7005], [9.9], [96.2]]),
#                    current_procent_flow_t = np.array([2]),
#                    current_procent_flow_u = np.array([3]))
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