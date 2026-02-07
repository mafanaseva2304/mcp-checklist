import numpy as np
from .base import TechModeStrategy
from models.mTechModeClassifier.utils_mode import (
    hysteresis_mode75,
    mad_z,
    mad,
    rolling_mad,
)
from typing import Dict, Any


class GFU2Strategy(TechModeStrategy):
    def predict(
        self, data: np.ndarray, params: Dict[str, Any], artifacts: Dict[str, Any] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
            1 – stable
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

        QMIN_TOTAL = cfg["QMIN_TOTAL"]  # т/ч
        BUF = cfg["BUF"]  # точек (≈ 2 ч)
        DISP_TH = cfg["DISP_TH"]  # «дрожание» < 3
        Z_TH = cfg["Z_TH"]  # MAD-Z порог
        MIN_TAGS_OUTLIER = cfg["MIN_TAGS_OUTLIER"]

        FEED_TAGS = tags["FEED_TAGS"]
        ANCHOR_TAGS = tags["ANCHOR_TAGS"]
        GROUP_RULES = tags["GROUP_RULES"]
        PROC_TAGS = tags["PROC_TAGS"]  # = объединение групп

        col_index = {tag: i for i, tag in enumerate(cfg["ALL_COLS"])}

        # offline
        feed_idx = [col_index[t] for t in FEED_TAGS]
        q_tot = data[:, feed_idx].sum(axis=1)
        regime_raw = np.where(q_tot < QMIN_TOTAL, 0, np.nan)  # 0 = offline

        # стабильность по ANCHOR_TAGS
        disp = np.zeros_like(q_tot, dtype=float)
        for t in ANCHOR_TAGS:
            idx = col_index[t]
            disp = np.maximum(disp, rolling_mad(data[:, idx], BUF))

        stable_mask = (disp < DISP_TH) & np.isnan(regime_raw)
        regime_raw[stable_mask] = 1  # stable = 1

        # BUFER
        regime = hysteresis_mode75(regime_raw, BUF)
        regime = np.where(np.isnan(regime), -2, regime)  # NaN -> transition
        regime[q_tot < QMIN_TOTAL] = 0

        # outlier внутри стабильного
        n = data.shape[0]
        mask_gt = np.zeros((len(PROC_TAGS), n), dtype=bool)

        for j, tag in enumerate(PROC_TAGS):
            idx = col_index[tag]
            stats = BASE_STATS.get(tag, None)
            if stats:
                median, mad_val = stats["median"], stats["mad"]
            else:
                median = np.nanmedian(data[:, idx])
                mad_val = mad(data[:, idx])
            z = mad_z(data[:, idx], median, mad_val)
            mask_gt[j] = np.abs(z) > Z_TH

        global_outlier = mask_gt.sum(axis=0) >= MIN_TAGS_OUTLIER
        group_out = np.ones(n, dtype=bool)
        for g_tags in GROUP_RULES.values():
            g_idx = [PROC_TAGS.index(t) for t in g_tags]
            group_out &= mask_gt[g_idx].any(axis=0)

        outlier = (global_outlier | group_out) & (regime == 1)
        final_regime = regime.copy()
        final_regime[outlier] = -1
        final_regime = np.where(np.isnan(final_regime), -2, final_regime)

        # commets
        comments = []
        for i in range(len(final_regime)):
            if final_regime[i] == 0:
                comments.append(f"offline: Q_total < {QMIN_TOTAL}")
            elif final_regime[i] == 1:
                tags_ok = ", ".join(ANCHOR_TAGS)
                comments.append(
                    f"стабильный режим: дисперсия < {DISP_TH} по [{tags_ok}]"
                )
            elif final_regime[i] == -1:
                reason = []
                if global_outlier[i]:
                    reason.append(
                        f"глобальный выброс: >= {MIN_TAGS_OUTLIER} тегов превышают MAD-Z"
                    )
                if group_out[i]:
                    hit_groups = [
                        name
                        for name, g_tags in GROUP_RULES.items()
                        if any(mask_gt[PROC_TAGS.index(t)][i] for t in g_tags)
                    ]
                    if hit_groups:
                        reason.append(f"выброс по группам: {', '.join(hit_groups)}")
                if not reason:
                    reason.append("выброс")
                comments.append("; ".join(reason))
            elif final_regime[i] == -2:
                comments.append(f"неопределено: нет стабильного режима")
            else:
                comments.append(f"нет классификации")

        return final_regime.astype(np.int8), comments
