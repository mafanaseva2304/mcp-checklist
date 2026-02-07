import numpy as np
from typing import Dict, Any
from models.mPakAnomalyClassifier.utils_anomaly import gbm_predict
from models.mPakAnomalyClassifier.pak_strategies.base import PakAnomalyStrategy


class AT5001AStrategy(PakAnomalyStrategy):
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any]
    ) -> tuple[np.ndarray, list[str]]:
        """
        0 – no anomaly
        1 - bad_value (pak is nan)
        2 - anomaly (Rule A or Rule B)

        data: np.ndarray (n, m)
            n - количество точек (буфер)
            m - количество признаков; m в порядке cfg['ALL_COLS']
        params:   CFG
        artifacts: joblib.load('pak_models.pkl')
                    {models, feats, thr}

        rerurn: np.ndarray (n,) метки классов
        """

        cfg = params
        tags = cfg["TAGS"]
        col = {t: i for i, t in enumerate(cfg["ALL_COLS"])}

        idx_pak = col[tags["PAK"]]
        idx_reg = col[tags["REGIME"]]
        feat_idx = np.array([col[f] for f in cfg["FEATS"]], int)

        pak = data[:, idx_pak].astype(float)
        regime = data[:, idx_reg].astype(int)

        pred = np.full_like(pak, np.nan)
        for reg in (1, 2):
            mask = regime == reg
            if mask.any():
                pred[mask] = gbm_predict(
                    artifacts["models"][reg], data[mask][:, feat_idx]
                )

        out = np.zeros(pak.size, np.int8)
        comments = []
        cnt = {1: 0, 2: 0}  # счётчики «подряд» по режимам
        thr = artifacts["thr"]
        mp = cfg["MIN_PERSIST"]

        for i, reg in enumerate(regime):
            if np.isnan(pak[i]):  # bad Pak
                out[i] = 1
                cnt[reg] = 0
                comments.append(f"плохое значение: pak is NaN")
                continue

            # нестабильный режим -> всё сбрасываем
            if reg not in cnt:
                cnt[1] = cnt[2] = 0
                comments.append(f"нестабильный режим установки: reg={reg}")
                continue

            if np.isnan(pred[i]):  # пропуски признаков
                out[i] = 1
                cnt[reg] = 0
                comments.append(f"пропущены признаки/модель: pred is NaN")
                continue

            # длительное превышение → метка 2
            cnt[reg] = cnt[reg] + 1 if abs(pak[i] - pred[i]) > thr[reg] else 0
            if cnt[reg] >= mp:
                out[i] = 2
                comments.append(f"аномалия: |pak-pred| > {thr[reg]} подряд ≥ {mp}")
            else:
                comments.append("нет аномалии")

        return out.astype(np.int8), comments
