import numpy as np
from .base import TechModeStrategy
from models.mTechModeClassifier.utils_mode import hysteresis_mode75, mad_z, mad
from typing import Dict, Any


class L600Strategy(TechModeStrategy):
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
            1 – лёгкое сырьё
            2 – тяжёлое сырьё
            0 – установка остановлена (shutdown / offline)
           -1 – alarm / outlier
           -2 – переходная зона / неопределённо

        data: np.ndarray (n, m)
        n - количество точек (буфер)
        m - количество признаков; m в порядке cfg['ALL_COLS']

        rerurn: np.ndarray (n,) метки классов
        list (n) комментарии к меткам
        """

        cfg: dict = params
        tags: dict = cfg["TAGS"]
        BASE_STATS: dict = cfg.get("BASE_STATS", {})

        T_LIGHT, T_HEAVY = cfg["T_LIGHT"], cfg["T_HEAVY"]
        QMIN1, QMIN2 = cfg["QMIN1"], cfg["QMIN2"]
        QMIN_TOTAL = cfg["QMIN_TOTAL"]
        BUF, Z_TH = cfg["BUF"], cfg["Z_TH"]

        col = {t: i for i, t in enumerate(cfg["ALL_COLS"])}

        flow_idx = [col[t] for t in tags["FLOW_LIST"]]
        q_calc = data[:, flow_idx].sum(axis=1)

        q_tot = data[:, col[tags["FLOW_TOTAL"]]].copy()
        miss_q = np.isnan(q_tot)
        q_tot[miss_q] = q_calc[miss_q]

        shutdown = q_tot < QMIN_TOTAL  # bool (n,)

        tkk_eff = data[:, col[tags["TKK_BLEND"]]].copy()  # AI5543-100
        mask_bad = np.isnan(tkk_eff)

        # solo-coker условие
        f_coker = data[:, col[tags["F_COKER"]]]
        f_light = data[:, col[tags["F_LIGHT"]]]
        f_r1 = data[:, col[tags["F_REFORM_1"]]]
        f_r2 = data[:, col[tags["F_REFORM_2"]]]

        solo_coker = (
            (f_light < QMIN2)  # прямогонный=0
            & (f_coker > QMIN1)  # кокс-нафта
            & ((f_r1 + f_r2) < QMIN1)  # ветки FI3501+FIC3032 ≈ 0
        )

        take_coker = mask_bad & solo_coker
        tkk_eff[take_coker] = data[:, col[tags["TKK_COKER"]]][take_coker]

        # легкое/тяжелое
        n = data.shape[0]
        regime_raw = np.full(n, np.nan)  # переход

        regime_raw[tkk_eff < T_LIGHT] = 1  # лёгкое
        regime_raw[tkk_eff >= T_HEAVY] = 2  # тяжёлое
        regime_raw[shutdown] = 0  # offline приоритетен

        # BUFER
        regime = hysteresis_mode75(regime_raw, BUF)
        # np.nan -> -2 : transition / неопределённо
        regime = np.where(np.isnan(regime), -2, regime)
        regime[shutdown] = 0

        # выбросы mad-z
        proc_tags = tags["PROC_TAGS"]
        mask_out = np.zeros(n, dtype=bool)

        for tag in proc_tags:
            col_idx = col[tag]
            arr = data[:, col_idx]
            stat = BASE_STATS.get(tag)
            if stat and stat["mad"]:
                median, mad_val = stat["median"], stat["mad"]
            else:
                median, mad_val = np.nanmedian(arr), mad(arr)
            z = mad_z(arr, median, mad_val)
            mask_out |= np.abs(z) > Z_TH

        outlier = mask_out & np.isin(regime, [1, 2])

        final_regime = regime.copy()
        final_regime[outlier] = -1
        final_regime = np.where(np.isnan(final_regime), -2, final_regime)

        # commets
        comments = []
        for i in range(len(final_regime)):
            if final_regime[i] == 0:
                comments.append(f"offline: Q_total < {QMIN_TOTAL}")
            elif final_regime[i] == 1:
                comments.append(f"легкое сырье: TKK < {T_LIGHT}")
            elif final_regime[i] == 2:
                comments.append(f"тяжелое сырье: TKK >= {T_HEAVY}")
            elif final_regime[i] == -1:
                comments.append(f"выброс: превышен MAD-Z")
            elif final_regime[i] == -2:
                comments.append(f"неопределено: нет стабильного режима")
            else:
                comments.append(f"нет классификации")

        return final_regime.astype(np.int8), comments
