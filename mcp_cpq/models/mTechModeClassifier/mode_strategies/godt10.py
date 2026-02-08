from typing import Any, Dict

import numpy as np

from models.mTechModeClassifier.utils_mode import hysteresis_mode75, mad, mad_z

from .base import TechModeStrategy


class GODT10Strategy(TechModeStrategy):
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
            1 – summer
            2 – winter
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
        tags = cfg["TAGS"]
        groups = cfg["GROUPS"]
        base_stats = cfg.get("BASE_STATS", {})

        QMIN = cfg["QMIN"]  # установка жива
        T_SUM_MAX = cfg["T_SUMMER_MAX"]  # верх летнего
        T_WIN_MIN = cfg["T_WINTER_MIN"]  # нижний порог зимнего
        DT_MIN = cfg["DT_MIN"]  # ΔT для зимы
        BUF = cfg["BUF"]  # буфер точек
        Z_TH = cfg["Z_TH"]
        MIN_GR_ALARM = cfg["MIN_GROUPS_ALARM"]

        # индексы столбцов
        col = {t: i for i, t in enumerate(cfg["ALL_COLS"])}

        feed = data[:, col[tags["FEED"]]]
        shutdown = feed < QMIN  # bool (n,)

        t_in = data[:, col[tags["T_IN"]]]
        t_out = data[:, col[tags["T_OUT"]]]
        dT = t_out - t_in  # shape (n,)

        n = data.shape[0]
        regime_raw = np.full(n, np.nan, dtype=float)

        # summer
        summer = t_in < T_SUM_MAX
        regime_raw[summer] = 1

        # winter
        winter = (t_in >= T_WIN_MIN) & (dT >= DT_MIN)
        regime_raw[winter] = 2

        # shutdown
        regime_raw[shutdown] = 0

        regime = hysteresis_mode75(regime_raw, BUF)

        # неопределённое = -2
        regime = np.where(np.isnan(regime), -2, regime)
        regime[shutdown] = 0

        # mad z
        # матрица масок: (all_tags_in_groups, n)
        all_keys = {k for lst in groups.values() for k in lst}
        mask_gt = np.zeros((len(all_keys), n), dtype=bool)
        key2row = {k: i for i, k in enumerate(all_keys)}

        for key in all_keys:
            if key == "dT":
                arr = dT
            else:
                arr = data[:, col[tags[key]]]
            st = base_stats.get(key, None)
            if st and st["mad"]:
                med, mad_val = st["median"], st["mad"]
            else:
                med, mad_val = np.nanmedian(arr), mad(arr)
            z = mad_z(arr, med, mad_val)
            mask_gt[key2row[key]] = np.abs(z) > Z_TH

        # групповая логика и финальный режим
        group_hit = {
            g: mask_gt[[key2row[k] for k in lst]].any(axis=0)
            for g, lst in groups.items()
        }

        alarm = sum(group_hit.values()) >= MIN_GR_ALARM  # bool (n,)
        outlier = alarm & np.isin(regime, [1, 2])

        final = regime.copy()
        final[outlier] = -1
        final = np.where(np.isnan(final), -2, final)

        # commets
        comments = []
        for i in range(len(final)):
            if final[i] == 0:
                comments.append(f"offline: FEED < {QMIN}")
            elif final[i] == 1:
                comments.append(f"лето: T_IN < {T_SUM_MAX}")
            elif final[i] == 2:
                comments.append(f"зима: T_IN >= {T_WIN_MIN} и dT >= {DT_MIN}")
            elif final[i] == -1:
                comments.append(f"выброс: превышен MAD-Z по группам")
            elif final[i] == -2:
                comments.append(f"неопределено: нет стабильного режима")
            else:
                comments.append(f"нет классификации")

        return final.astype(np.int8), comments
