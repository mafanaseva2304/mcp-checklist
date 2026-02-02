from app.models import Model
from pandas import DataFrame, Series, concat
from math import isnan
import numpy as np
import pandas as pd
import warnings
import numexpr as ne

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
dt = 3

class GetReservQuality(Model):

    def predict(self, data_by_predict: np.ndarray):
        """ Запуск прогнозирования модели

        Args:
        data_by_predict - Данные для прогноза

        Format:
         data_by_predict = np.array([
            [2.0, ...], Iput volume
            [1.3, ...], Output volume
            [579, ...], Flow Quality 
            [579, ...], Tank quality data 
            [None, ...],Level water data
            [15, ...],  Level data
            [579, ...], L1 quality
            [579, ...], L2 quality
            [579, ...], L3 quality
        ])
            
        Return: 
            quality: np.ndarray, L1: np.ndarray, L2: np.ndarray, L3: np.ndarray
        """
        Volumes = data_by_predict[:, :2].astype('float')
        Quality_data = data_by_predict[:, 2].astype('float')
        Tank_quality_data = data_by_predict[:, 3].astype('float')
        Level_data = data_by_predict[:, 4:6].astype('float')
        Quality_L = data_by_predict[:, 6:].astype('float')
        
        Quality_L[np.isnan(Quality_L)] = np.broadcast_to(Quality_data, Quality_L.T.shape).T[np.isnan(Quality_L)]
        Tank_quality_data[np.isnan(Tank_quality_data)] = Quality_data[np.isnan(Tank_quality_data)]
        L1, L2, L3 = Quality_L[:, 0], Quality_L[:, 1], Quality_L[:, 2]
        quality = np.zeros([data_by_predict.shape[0]])
        if len(Volumes) == 1:
            quality[Volumes[:, 0]<1] = Tank_quality_data[Volumes[:, 0]<1]

        if Level_data[Volumes[:, 0]>=1, :].shape[0] != 0:

            tank_level = np.apply_along_axis(self.calcDeltaVolumeTank, axis=1, arr=Level_data[Volumes[:, 0]>=1, :], reserv=True)

            if self.tank_type_name_ == "РВС":
                data_layers_volume = self.getRVSRerservVolume(tank_level, DataFrame({'mirror_area': self.mirror_area_, 'line_length': self.line_length_}))
            elif self.tank_type_name_ == "РШ":
                data_layers_volume = self.getRSHRerservVolume(tank_level, DataFrame({'height': self.height_}))
            else:
                data_layers_volume = self.getRGSRerservVolume(tank_level, DataFrame({'height': self.height_, 'line_length': self.line_length_}))
            data_layers_volume[data_layers_volume < 0] = 0

            qual_L1, qual_L2, qual_L3, qual = self.getQualityByZones(
                data_layers_volume[:, data_layers_volume[0]>1],
                Volumes[Volumes[:, 0] >= 1, :][data_layers_volume[0] > 1, :],
                Quality_data[Volumes[:, 0] >= 1][data_layers_volume[0] > 1],
                Quality_L[Volumes[:, 0] >= 1, :][data_layers_volume[0] > 1, :])

            quality = np.full(data_by_predict.shape[0], np.nan)
            quality[np.where(Volumes[:, 0] >= 1)[0][data_layers_volume[0] > 1]] = qual
            
            L1[np.where(Volumes[:, 0] >= 1)[0][data_layers_volume[0] > 1]] = qual_L1
            L2[np.where(Volumes[:, 0] >= 1)[0][data_layers_volume[0] > 1]] = qual_L2
            L3[np.where(Volumes[:, 0] >= 1)[0][data_layers_volume[0] > 1]] = qual_L3
        if len(Volumes) > 1:
            quality = pd.Series(quality).ffill()
            quality[Volumes[:, 0]<1] = None
            quality = quality.ffill()
            if quality.isnull().any() == True:
                quality[np.isnan(quality)] = Tank_quality_data[np.isnan(quality)]

            L1 = pd.Series(L1).ffill()
            L1[Volumes[:, 0] < 1] = None
            L1 = L1.ffill()
            if L1.isnull().any() == True:
                L1[np.isnan(L1)] = Quality_L[:, 0][np.isnan(L1)]
            L2 = pd.Series(L2).ffill()
            L2[Volumes[:, 0] < 1] = None
            L2 = L2.ffill()
            if L2.isnull().any() == True:
                L2[np.isnan(L2)] = Quality_L[:, 1][np.isnan(L2)]
            L3 = pd.Series(L3).ffill()
            L3[Volumes[:, 0] < 1] = None
            L3 = L3.ffill()
            if L3.isnull().any() == True:
                L3[np.isnan(L3)] = Quality_L[:, 2][np.isnan(L3)]

        return quality.to_numpy(), L1.to_numpy(), L2.to_numpy(), L3.to_numpy()
    
    def calcDeltaVolumeTank(self, Level_data, reserv=None) -> float:
        level_water_data, level_data = Level_data[0], Level_data[1]

        if isnan(level_data):
            return 0
        level_water = 0
        if isnan(level_water_data) == False:
            if level_water_data:
                level_water = self.convertUnitMeasurement(level_water_data, self.level_water_data_unit_)

        level_product = self.convertUnitMeasurement(level_data, self.level_data_unit_)
        x = level_product - level_water
        if reserv:
            return x
        return float(ne.evaluate(str(self.calc_formula_)))
    
    def convertUnitMeasurement(self, x, unit):
        if x < 0:
            return 0.0
        if "см" == str(unit):
            x = x / 100
        elif "мм" in str(unit):
            x = x / 1000
        return x
    
    def getRVSRerservVolume(self, level_data, tank_zone_info):
        data_layers_volume = np.zeros([6, len(level_data)]) # Columns : ['L1', 'L2', 'L3', 'Scont1', 'Scont2', 'Scont3']
        data_layers_volume[0] = tank_zone_info.loc[0, "mirror_area"] * level_data
        data_layers_volume[1] = tank_zone_info.loc[1, "mirror_area"] * level_data - data_layers_volume[0]
        data_layers_volume[2] = (
            tank_zone_info.loc[2, "mirror_area"] * level_data - data_layers_volume[0] - data_layers_volume[1]
        )
        data_layers_volume[3] = tank_zone_info.loc[0, "line_length"] * level_data
        data_layers_volume[4] = tank_zone_info.loc[1, "line_length"] * level_data
        data_layers_volume[5] = tank_zone_info.loc[2, "line_length"] * level_data
        return data_layers_volume
    
    def getRSHRerservVolume(self, level_data, tank_zone_info):
        data_layers_volume = np.zeros([6, len(level_data)]) # Columns : ['L1', 'L2', 'L3', 'Scont1', 'Scont2', 'Scont3']

        data_layers_volume[0] = (
            np.pi * np.power(level_data, 2) * ((tank_zone_info.loc[0, "height"] / 2) - (level_data / 3))
        )
        data_layers_volume[1] = (
            np.pi * np.power(level_data, 2) * (tank_zone_info.loc[1, "height"] / 2 - level_data / 3)
            - data_layers_volume[0]
        )
        data_layers_volume[2] = (
            np.pi * np.power(level_data, 2) * (tank_zone_info.loc[2, "height"] / 2 - level_data / 3)
            - data_layers_volume[0]
            - data_layers_volume[1]
        )
        data_layers_volume[3] = np.pi * level_data * tank_zone_info.loc[0, "height"]
        data_layers_volume[4] = np.pi * level_data * tank_zone_info.loc[1, "height"]
        data_layers_volume[5] = np.pi * level_data * tank_zone_info.loc[2, "height"]
        return data_layers_volume

    def getRGSRerservVolume(self, level_data, tank_zone_info):
        data_layers_volume = np.zeros([6, len(level_data)]) # Columns : ['L1', 'L2', 'L3', 'Scont1', 'Scont2', 'Scont3']
        
        first_group_ind = (level_data > tank_zone_info.loc[0, "height"]) & (level_data > tank_zone_info.loc[1, "height"])
        sec_group_ind = (level_data > tank_zone_info.loc[0, "height"]) & ~(level_data > tank_zone_info.loc[1, "height"])
        third_group_ind = ~(level_data > tank_zone_info.loc[0, "height"])
        

        data_layers_volume[0, first_group_ind] = tank_zone_info.loc[0, "line_length"] * ((np.pi * np.power(tank_zone_info.loc[0, "height"],2))/4)
        data_layers_volume[1, first_group_ind] = tank_zone_info.loc[1, "line_length"] * ((np.pi * np.power(tank_zone_info.loc[1, "height"],2))/4) - data_layers_volume[0, first_group_ind] 
        
        data_layers_volume[0, sec_group_ind] = tank_zone_info.loc[0, "line_length"] * ((np.pi * np.power(tank_zone_info.loc[0, "height"],2))/4)
        data_layers_volume[1, sec_group_ind] = (
        tank_zone_info.loc[1, "line_length"]
        * (
            np.power(tank_zone_info.loc[1, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data[sec_group_ind] / tank_zone_info.loc[1, "height"])
            - (tank_zone_info.loc[1, "height"] / 2 - level_data[sec_group_ind])
            * np.sqrt(tank_zone_info.loc[1, "height"] * level_data[sec_group_ind] - np.power(level_data[sec_group_ind], 2))
        )
        - data_layers_volume[0, sec_group_ind]
        )

        data_layers_volume[0, third_group_ind] = tank_zone_info.loc[0, "line_length"] * (
            np.power(tank_zone_info.loc[0, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[0, "height"])
            - (tank_zone_info.loc[0, "height"] / 2 - level_data[third_group_ind])
            * np.sqrt(tank_zone_info.loc[0, "height"] * level_data[third_group_ind] - np.power(level_data[third_group_ind], 2))
            )
        data_layers_volume[1, third_group_ind] = (
            tank_zone_info.loc[1, "line_length"]
            * (
                np.power(tank_zone_info.loc[1, "height"] / 2, 2)
                * np.arccos(1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[1, "height"])
                - (tank_zone_info.loc[1, "height"] / 2 - level_data[third_group_ind])
                * np.sqrt(tank_zone_info.loc[1, "height"] * level_data[third_group_ind] - np.power(level_data[third_group_ind], 2))
            )
            - data_layers_volume[0, third_group_ind]
            )

        data_layers_volume[2] = (
            tank_zone_info.loc[2, "line_length"]
            * (
                np.power(tank_zone_info.loc[2, "height"] / 2, 2)
                * np.arccos(1 - 2 * level_data / tank_zone_info.loc[2, "height"])
                - (tank_zone_info.loc[2, "height"] / 2 - level_data)
                * np.sqrt(tank_zone_info.loc[2, "height"] * level_data - np.power(level_data, 2))
            )
            - data_layers_volume[0]
            - data_layers_volume[1]
            )


        data_layers_volume[3, first_group_ind] = (2 * np.pi * (np.power(tank_zone_info.loc[0, "height"],2)))/4 + tank_zone_info.loc[0, "line_length"] * np.pi * tank_zone_info.loc[0, "height"]
        data_layers_volume[4, first_group_ind] = (2 * np.pi * (np.power(tank_zone_info.loc[1, "height"],2)))/4 + tank_zone_info.loc[1, "line_length"] * np.pi * tank_zone_info.loc[1, "height"]
            

        data_layers_volume[3, sec_group_ind] = (2 * np.pi * (np.power(tank_zone_info.loc[0, "height"],2)))/4 + tank_zone_info.loc[0, "line_length"] * np.pi * tank_zone_info.loc[0, "height"]
        data_layers_volume[4, sec_group_ind] = 2 * (
            np.power(tank_zone_info.loc[1, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data[sec_group_ind] / tank_zone_info.loc[1, "height"])
            - (tank_zone_info.loc[1, "height"] / 2 - level_data[sec_group_ind])
            * np.sqrt((tank_zone_info.loc[1, "height"]) * level_data[sec_group_ind] - np.power(level_data[sec_group_ind], 2))
        ) + tank_zone_info.loc[1, "line_length"] * tank_zone_info.loc[1, "height"] * np.arccos(
            1 - 2 * level_data[sec_group_ind] / tank_zone_info.loc[1, "height"]
        )

        data_layers_volume[3, third_group_ind] = 2 * (
            np.power(tank_zone_info.loc[0, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[0, "height"])
            - (tank_zone_info.loc[0, "height"] / 2 - level_data[third_group_ind])
            * np.sqrt((tank_zone_info.loc[0, "height"]) * level_data[third_group_ind] - np.power(level_data[third_group_ind], 2))
        ) + tank_zone_info.loc[0, "line_length"] * tank_zone_info.loc[0, "height"] * np.arccos(
            1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[0, "height"]
        )
        data_layers_volume[4, third_group_ind] = 2 * (
            np.power(tank_zone_info.loc[1, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[1, "height"])
            - (tank_zone_info.loc[1, "height"] / 2 - level_data[third_group_ind])
            * np.sqrt((tank_zone_info.loc[1, "height"] ) * level_data[third_group_ind] - np.power(level_data[third_group_ind], 2))
        ) + tank_zone_info.loc[1, "line_length"] * tank_zone_info.loc[1, "height"] * np.arccos(
            1 - 2 * level_data[third_group_ind] / tank_zone_info.loc[1, "height"]
        )

        data_layers_volume[5] = 2 * (
            np.power(tank_zone_info.loc[2, "height"] / 2, 2)
            * np.arccos(1 - 2 * level_data / tank_zone_info.loc[2, "height"])
            - (tank_zone_info.loc[2, "height"] / 2 - level_data)
            * np.sqrt((tank_zone_info.loc[2, "height"]) * level_data - np.power(level_data, 2))
        ) + tank_zone_info.loc[2, "line_length"] * tank_zone_info.loc[2, "height"] * np.arccos(
            1 - 2 * level_data / tank_zone_info.loc[2, "height"]
        )
        return data_layers_volume
    
    def getQualityByZones(self, data_layers_volume, Volumes, Quality_data, Quality_L):
        data_quality = np.zeros([Volumes.shape[0], 4]) # Columns : ['VFnxt2', 'VFnxt3', 'VFprvs1', 'VFprvs2']
        Quality_ind = np.zeros([Volumes.shape[0], 4]) # Columns : ['L1', 'L2', 'L3', 'quality']

        Flow_coefficient = np.full(Volumes.shape[0], self.flow_coefficient_)
        Flow_coefficient[np.any(Volumes == 0, axis = 1)] = 0

        VFlow = (Volumes[:, 0] - Volumes[:, 1] * Flow_coefficient) / data_layers_volume[0,:] * dt / 60

        data_quality[VFlow>1, 0] = 0 
        data_quality[VFlow>1, 1] = 0
        data_quality[VFlow>1, 2] = 0 
        data_quality[VFlow>1, 3] = 0 
        VFlow[VFlow>1] = 1

        data_quality[VFlow<=1, 0] = (self.mass_coefficient_ * data_layers_volume[3, VFlow<=1]) / data_layers_volume[0, VFlow<=1]
        data_quality[VFlow<=1, 1] = (self.mass_coefficient_ * data_layers_volume[4, VFlow<=1]) / data_layers_volume[1, VFlow<=1]
        data_quality[VFlow<=1, 2] = (self.mass_coefficient_ * data_layers_volume[3, VFlow<=1]) / data_layers_volume[1, VFlow<=1]
        data_quality[VFlow<=1, 3] = (self.mass_coefficient_ * data_layers_volume[4, VFlow<=1]) / data_layers_volume[2, VFlow<=1]

        for ind in range(data_quality.shape[0]):
            Quality_ind[ind, 0] = (
                Quality_data[ind] * VFlow[ind]
                + Quality_L[ind, 1] * data_quality[ind, 0]
                + Quality_L[ind, 0] * (1 - data_quality[ind, 0] - VFlow[ind])
            )

            Quality_ind[ind, 1] = (
                Quality_L[ind, 0] * data_quality[ind, 2]
                + Quality_L[ind, 2] * data_quality[ind, 1]
                + Quality_L[ind, 1] * (1 - data_quality[ind, 2] - data_quality[ind, 1])
            )

            Quality_ind[ind, 2] = Quality_L[ind, 1] * data_quality[ind, 3] + Quality_L[ind, 2] * (
                1 - data_quality[ind, 3]
            )
            Quality_ind[ind, 3] = Quality_data[ind] * Flow_coefficient[ind] + Quality_L[ind, 0] * (
                1 - Flow_coefficient[ind]
            )

            if ind!= data_quality.shape[0] - 1:
                Quality_L[ind+1] = Quality_ind[ind, :3]
                # Quality_data[ind+1] = Quality_ind[ind, 3]

        return Quality_ind[:, 0], Quality_ind[:, 1], Quality_ind[:, 2], Quality_ind[:, 3]

    def train(self, data: DataFrame, y: Series):
        """ data - Данные для обучения
            y - Лабораторные анализы для обучения
            Запуск обучения модели
        """
        self._model.fit(data, y)
        return self._model.score(data, y)

    def test(self, data: DataFrame, y: Series):
        """ data - Данные для прогноза
            y - Лабораторные анализы для тестирования
            Запуск тестировния модели

        """
        return self._model.score(data, y)

    def save(self):
        """ Сохранение модели
            Return: Intercept, Coefficients
        """
        return self._model

    def load(self, object_id: int, object_name: str, tag_prefix: str, tank_type_name: str, height: np.array, line_length: np.array, 
             mirror_area: np.array, flow_coefficient: float, mass_coefficient: float,calc_formula: str, level_data_unit: str, 
             level_water_data_unit: str, intercept: float = 0):
        """ 
            intercept - bias модели
        """
        self.intercept_ = intercept
        self.object_id_ = object_id
        self.object_name_ = object_name
        self.tag_prefix_ = tag_prefix
        self.tank_type_name_ = tank_type_name
        self.height_ = height
        self.line_length_ = line_length
        self.mirror_area_ = mirror_area
        self.flow_coefficient_ = flow_coefficient
        self.mass_coefficient_ = mass_coefficient
        self.calc_formula_ = calc_formula
        self.level_data_unit_ = level_data_unit
        self.level_water_data_unit_ = level_water_data_unit

