import numpy as np
from typing import Dict, Any
from models.mPakAnomalyClassifier.utils_anomaly import gbm_predict
from models.mPakAnomalyClassifier.pak_strategies.base import PakAnomalyStrategy


class QR116BStrategy(PakAnomalyStrategy):
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
        artifacts: joblib.load('pak_gfu_models.pkl')
                    {model_med, model_lo, model_hi}

        rerurn: np.ndarray (n,) метки классов
        """

        cfg = params
        col = {c: i for i, c in enumerate(cfg["ALL_COLS"])}
        tags = cfg["TAGS"]

        idx_pak = col[tags["PAK"]]
        idx_reg = col[tags["REGIME"]]
        feat_idx = np.array([col[f] for f in cfg["FEATS"]], int)

        pak = data[:, idx_pak].astype(float)
        regime = data[:, idx_reg].astype(int)

        feats_all = data[:, feat_idx]
        y_lo = gbm_predict(artifacts["model_lo"], feats_all)
        y_hi = gbm_predict(artifacts["model_hi"], feats_all)

        spec = cfg["SPEC_TOL"]
        mp = cfg["MIN_PERSIST"]
        out = np.zeros(pak.size, np.int8)
        run_hi = run_lo = 0
        comments = []

        for i in range(pak.size):
            # (nan) -> bad_flag = [1]
            if np.isnan(pak[i]):
                out[i] = 1
                run_hi = run_lo = 0
                comments.append(f"плохое значение: pak is NaN")
                continue

            # нестабильный режим -> всё обнулить = 0
            if regime[i] != 1:
                run_hi = run_lo = 0
                comments.append(f"нестабильный режим установки: reg={regime[i]}")
                continue

            if np.isnan(feats_all[i]).any():
                out[i] = 1
                run_hi = run_lo = 0
                comments.append(f"пропущены признаки")
                continue

            over_hi = pak[i] > y_hi[i] + spec
            under_lo = pak[i] < y_lo[i] - spec
            ruleA = over_hi or under_lo

            run_hi = run_hi + 1 if over_hi else 0
            run_lo = run_lo + 1 if under_lo else 0
            ruleB = (run_hi >= mp) or (run_lo >= mp)

            if ruleA or ruleB:
                out[i] = 2
                run_hi = run_lo = 0
                comments.append(f"аномалия")
            else:
                comments.append(f"нет аномалии")

        return out.astype(np.int8), comments
