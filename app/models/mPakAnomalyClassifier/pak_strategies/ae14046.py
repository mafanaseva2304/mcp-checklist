import numpy as np
from typing import Dict, Any
from app.models.mPakAnomalyClassifier.pak_strategies.base import PakAnomalyStrategy


class AE14046Strategy(PakAnomalyStrategy):
    def predict(self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any] = None) -> tuple[np.ndarray, list[str]]:
        """
        0 – no anomaly
        1 - bad_value (pak is nan)
        2 - anomaly (≥ MIN_PERSIST подряд |resid| > RES_TH)

        data: np.ndarray (n, m)
            n - количество точек (буфер)
            m - количество признаков; m в порядке cfg['ALL_COLS']
        params: CFG
        rerurn: np.ndarray (n,) метки классов
        """

        cfg  = params
        tags = cfg["TAGS"]
        col  = {t: i for i, t in enumerate(cfg["ALL_COLS"])}

        pak = data[:, col[tags["PAK"]]].astype(float)           # (n,)
        regime = data[:, col[tags["REGIME"]]].astype(np.int8)   # (n,)

        # регрессионный прогноз
        pred = np.full(pak.shape[0], cfg["BIAS"], dtype=float)
        for key, w in cfg["COEFFS"].items():
            x = data[:, col[tags[key]]].astype(float)           # (n,)
            pred += w * x
        resid = pak - pred                                      # (n,)
        exceed  = (~np.isnan(resid)) & (np.abs(resid) > cfg["RES_TH"])

        # BUFER
        out = np.zeros(pak.shape[0], dtype=np.int8)
        comments = []
        consec_count = {1: 0, 2: 0}                             # счётчик превышений [0,1]
        stable_set = (1, 2)

        for i in range(pak.shape[0]):
            reg = int(regime[i])

            # (nan) -> bad_flag = [1]
            if np.isnan(pak[i]):
                out[i] = 1
                comments.append(f"плохое значение: pak is NaN")
                if reg in stable_set:
                    consec_count[reg] = 0
                continue

            # нестабильный режим -> всё обнулить = 0
            if reg not in stable_set:
                consec_count[1] = consec_count[2] = 0
                comments.append(f"нестабильный режим установки: reg={reg}")
                continue

            # текущая ошибка превышает RES_TH
            if exceed[i]:
                consec_count[reg] += 1
            else:
                consec_count[reg] = 0

            # если > MIN_PERSIST = аномалия по долгому отклонению
            if consec_count[reg] >= cfg["MIN_PERSIST"]:
                out[i] = 2
                comments.append(f"аномалия: |residual| > {cfg['RES_TH']} подряд ≥ {cfg['MIN_PERSIST']}")
            else:
                comments.append("нет аномалии")

        return out.astype(np.int8), comments