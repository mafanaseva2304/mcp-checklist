from abc import ABCMeta, abstractmethod
from pandas import DataFrame, Series


class Model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, data: DataFrame):
        pass

    @abstractmethod
    def train(self, data: DataFrame, y: Series):
        pass

    @abstractmethod
    def test(self, data: DataFrame, y: Series):
        pass
