# repositories/model_repository.py
from typing import Any, Dict, List, Optional

from db.postgre import AsyncPostgre
from settings.settings import QuerySettings


class AsyncQueryRepository:
    """Репозиторий для работы с моделями в БД"""

    def __init__(self, db: AsyncPostgre, settings: QuerySettings):
        self._db = db
        self._params: Dict[str, Any] = settings.model_dump()

    async def find_models_without_status_tag(
        self,
    ) -> List[Dict[str, Any]]:

        query = """
            WITH models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                  AND disabled = FALSE
                  AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                  AND name NOT LIKE %(layer1_filter)s
                  AND name NOT LIKE %(layer2_filter)s
                  AND name NOT LIKE %(layer3_filter)s
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id = %(status_type_id)s
            ),
            aggr AS (
                SELECT 
                    models.id, 
                    models.name, 
                    statuses.id AS status_id
                FROM models
                LEFT JOIN statuses ON models.id = statuses.parent_id
            )
            SELECT id, name, status_id
            FROM aggr
            WHERE status_id IS NULL
            ORDER BY id
        """

        return await self._db.fetch_all(query, self._params)

    async def find_models_without_view_diag(self) -> List[Dict[str, Any]]:

        query = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = FALSE
            AND (attributes->>'view_diag')::BOOLEAN IS NULL
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_train_config(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                id, 
                name,
                ((attributes->'train_config')->>'ol_mse_mult')::REAL AS ol_mse_mult,
                ((attributes->'train_config')->>'ol_bias_mult')::REAL AS ol_bias_mult,
                ((attributes->'train_config')->>'ol_window')::INTEGER AS ol_window,
                ((attributes->'train_config')->>'cs_mult')::REAL AS cs_mult,
                ((attributes->'train_config')->>'cs_window')::INTEGER AS cs_window
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = FALSE
            AND (attributes->>'view_diag')::BOOLEAN = TRUE
            AND parent_id IS NOT NULL
            AND (
                ((attributes->'train_config')->>'ol_mse_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'ol_bias_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'ol_window')::INTEGER IS NULL
                OR ((attributes->'train_config')->>'cs_mult')::REAL IS NULL
                OR ((attributes->'train_config')->>'cs_window')::INTEGER IS NULL
            )
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_tech_limits(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                id, 
                name,
                (attributes->>'tech_min')::REAL AS tech_min,
                (attributes->>'tech_max')::REAL AS tech_max
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = FALSE
            AND (
                (attributes->>'tech_min')::REAL IS NULL
                AND (attributes->>'view_diag')::BOOLEAN = TRUE
                OR (attributes->>'tech_max')::REAL IS NULL
            )
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_status_values(self) -> List[Dict[str, Any]]:

        query = """
            WITH models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND disabled = FALSE
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id = %(status_type_id)s
            ),
            aggr AS (
                SELECT models.id, models.name, statuses.id AS status_id
                FROM models
                JOIN statuses ON models.id = statuses.parent_id
            ),
            status_values AS (
                SELECT 
                    aggr.id, 
                    aggr.name, 
                    aggr.status_id, 
                    MAX(uv.date_value) AS status_date 
                FROM aggr
                LEFT JOIN result.union_values uv 
                    ON aggr.status_id = uv.tag_id
                    AND uv.source_type_id = %(status_type_id)s
                GROUP BY aggr.id, aggr.name, aggr.status_id
            )
            SELECT id, name, status_id, status_date
            FROM status_values 
            WHERE status_date IS NULL
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_status_last_values(self) -> List[Dict[str, Any]]:

        query = """
            WITH models AS (
                SELECT id, name
                FROM dictionary.union_tags
                WHERE source_type_id = %(model_type_id)s
                AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND disabled = FALSE
            ),
            statuses AS (
                SELECT id, name, parent_id
                FROM dictionary.union_tags
                WHERE source_type_id = %(status_type_id)s
            ),
            aggr AS (
                SELECT models.id, models.name, statuses.id AS status_id
                FROM models
                JOIN statuses ON models.id = statuses.parent_id
            ),
            status_last_values AS (
                SELECT 
                    aggr.id, 
                    aggr.name, 
                    aggr.status_id, 
                    ulv.date_value AS status_date 
                FROM aggr
                LEFT JOIN result.union_last_values ulv 
                    ON aggr.status_id = ulv.tag_id
                    AND ulv.source_type_id = %(status_type_id)s
            )
            SELECT id, name, status_id, status_date
            FROM status_last_values 
            WHERE status_date IS NULL
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_recent_values(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                ut.id, 
                ut.name,
                ulv.date_value AS last_value_date
            FROM dictionary.union_tags ut
            JOIN result.union_last_values ulv ON ut.id = ulv.tag_id
            WHERE ut.source_type_id = %(model_type_id)s
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
            AND ut.disabled = FALSE
            AND ulv.date_value < NOW() - INTERVAL '1 hour'
            ORDER BY ut.id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_parent(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                id, 
                name,
                parent_id
            FROM dictionary.union_tags ut
            WHERE ut.source_type_id = %(model_type_id)s
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
            AND ut.parent_id IS NULL
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_lab_mse(self) -> List[Dict[str, Any]]:

        query = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND disabled = FALSE
            AND (attributes->>'lab_mse')::REAL IS NULL
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_ipk_metrics(self) -> List[Dict[str, Any]]:

        query = """
            WITH metrics AS (
                SELECT 
                    ut.id,
                    ut.name,
                    ulv.json_value ? 'corr' AS corr,
                    ulv.json_value ? 'det' AS det,
                    ulv.json_value ? 'mae' AS mae,
                    ulv.json_value ? 'Ir' AS Ir,
                    ulv.json_value ? 'Icoef_det' AS Icoef_det,
                    ulv.json_value ? 'Imae' AS Imae,
                    ulv.json_value ? 'ipk' AS ipk,
                    ulv.json_value ? 'sko_la' AS sko_la,
                    ut.attributes->'ipk'->>'W_r' AS W_r,
                    ut.attributes->'ipk'->>'W_r2' AS W_r2,
                    ut.attributes->'ipk'->>'W_mae' AS W_mae,
                    ut.attributes->'ipk'->>'ipk_window' AS ipk_window,
                    status_tag.id AS status_tag_id,
                    status_ulv.real_value AS status_value
                FROM dictionary.union_tags ut
                JOIN result.union_last_values ulv 
                    ON ut.id = ulv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                LEFT JOIN dictionary.union_tags status_tag 
                    ON ut.id = status_tag.parent_id
                    AND status_tag.source_type_id = %(status_type_id)s
                LEFT JOIN result.union_last_values status_ulv 
                    ON status_tag.id = status_ulv.tag_id
                    AND status_ulv.source_type_id = %(status_type_id)s
                WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND ut.disabled = FALSE
                AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
                AND ut.parent_id IS NOT NULL
                AND ut.name NOT LIKE %(layer1_filter)s
                AND ut.name NOT LIKE %(layer2_filter)s
                AND ut.name NOT LIKE %(layer3_filter)s
            )
            SELECT 
                id,
                name,
                corr,
                det,
                mae,
                Ir,
                Icoef_det,
                Imae,
                ipk,
                sko_la,
                W_r,
                W_r2,
                W_mae,
                ipk_window,
                CASE 
                    WHEN status_value IS NULL THEN NULL
                    WHEN status_value = 4 THEN TRUE
                    ELSE FALSE
                END AS in_test
            FROM metrics
            WHERE (
                NOT corr OR NOT det OR NOT mae
                OR NOT Ir OR NOT Icoef_det OR NOT Imae
                OR NOT ipk OR NOT sko_la
                OR W_r IS NULL OR W_r2 IS NULL OR W_mae IS NULL OR ipk_window IS NULL
            )
            AND status_value != 4
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_ipk_config(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                id, 
                name,
                ((attributes->'ipk')->>'W_r')::REAL AS w_r,
                ((attributes->'ipk')->>'W_r2')::REAL AS w_r2,
                ((attributes->'ipk')->>'W_mae')::REAL AS w_mae,
                ((attributes->'ipk')->>'ipk_window')::INTEGER AS ipk_window
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (attributes->>'view_diag')::BOOLEAN = TRUE
            AND disabled = FALSE
            AND (
                ((attributes->'ipk')->>'W_r')::REAL IS NULL
                OR ((attributes->'ipk')->>'W_r2')::REAL IS NULL
                OR ((attributes->'ipk')->>'W_mae')::REAL IS NULL
                OR ((attributes->'ipk')->>'ipk_window')::INTEGER IS NULL
            )
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_without_evaluation_date(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                ut.id, 
                ut.name,
                (ulv.json_value->>'ecaluation_date')::TIMESTAMP WITH TIME ZONE AS ecaluation_date
            FROM dictionary.union_tags ut
            JOIN result.union_last_values ulv 
                ON ut.id = ulv.tag_id
                AND ut.source_type_id = %(model_type_id)s
            WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ulv.json_value->>'ecaluation_date')::TIMESTAMP WITH TIME ZONE IS NULL
            AND ut.name NOT LIKE %(layer1_filter)s
            AND ut.name NOT LIKE %(layer2_filter)s
            AND ut.name NOT LIKE %(layer3_filter)s
            ORDER BY ut.id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_with_limited_values(self) -> List[Dict[str, Any]]:

        query = """
            WITH models AS (
                SELECT 
                    ut.id, 
                    ut.name,
                    (ut.attributes->>'tech_min')::REAL AS tech_min,
                    (ut.attributes->>'tech_max')::REAL AS tech_max,
                    ulv.real_value AS last_value
                FROM dictionary.union_tags ut
                JOIN result.union_last_values ulv 
                    ON ut.id = ulv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
                AND ut.disabled = FALSE
                AND (
                    (ut.attributes->>'tech_min')::REAL IS NOT NULL
                    OR (ut.attributes->>'tech_max')::REAL IS NOT NULL
                )
            )
            SELECT 
                id,
                name,
                tech_min,
                tech_max,
                last_value,
                COALESCE(last_value <= tech_min, FALSE) AS min_bound,
                COALESCE(last_value >= tech_max, FALSE) AS max_bound
            FROM models
            WHERE last_value <= tech_min OR last_value >= tech_max
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_with_frozen_values(self) -> List[Dict[str, Any]]:

        query = """
            WITH distinct_values AS (
                SELECT DISTINCT ut.id, ut.name, uv.real_value 
                FROM dictionary.union_tags ut
                JOIN result.union_values uv 
                    ON ut.id = uv.tag_id
                    AND ut.source_type_id = %(model_type_id)s
                WHERE NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
                AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
                AND ut.disabled = FALSE
                AND uv.date_value >= NOW() - INTERVAL '1 hour'
            ),
            count_values AS (
                SELECT id, name, COUNT(real_value) AS unique_values_count
                FROM distinct_values
                GROUP BY id, name
            )
            SELECT id, name, unique_values_count
            FROM count_values 
            WHERE unique_values_count <= 1
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_with_different_units(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                ut.id, 
                ut.name,
                ut.parent_id,
                ut.unit_id AS unit_id,
                parent_ut.unit_id AS parent_unit_id
            FROM dictionary.union_tags ut
            JOIN dictionary.union_tags parent_ut ON ut.parent_id = parent_ut.id
            WHERE ut.source_type_id = %(model_type_id)s
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
            AND ut.unit_id != parent_ut.unit_id
            AND ut.name NOT LIKE %(layer1_filter)s
            AND ut.name NOT LIKE %(layer2_filter)s
            AND ut.name NOT LIKE %(layer3_filter)s
            ORDER BY ut.id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_disabled_models(self) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                id, 
                name,
                disabled
            FROM dictionary.union_tags ut
            WHERE ut.source_type_id = %(model_type_id)s
            AND NOT ((ut.attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (ut.attributes->>'view_diag')::BOOLEAN = TRUE
            AND ut.disabled != FALSE
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_models_for_predictor_check(self) -> List[Dict[str, Any]]:

        query = """
            SELECT id, name, (attributes->>'type_model')::INTEGER AS type_model
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = FALSE
            AND (attributes->>'type_model')::INTEGER IS NOT NULL
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (attributes->>'view_diag')::BOOLEAN = TRUE
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_latest_clusters(self, model_id: int) -> List[Dict[str, Any]]:

        query = """
            WITH clusters AS (
                SELECT 
                    id_model AS model_id,
                    id AS cluster_id,
                    id_cluster AS cluster,
                    RANK() OVER(
                        PARTITION BY id_model, id_cluster 
                        ORDER BY date DESC
                    ) AS cluster_order
                FROM model.model_cluster_config
                WHERE id_model = %(model_id)s
            ) 
            SELECT model_id, cluster_id, cluster
            FROM clusters
            WHERE cluster_order = 1
        """
        return await self._db.fetch_all(query, {"model_id": model_id})

    async def find_outdated_predictors(
        self, model_id: int, cluster_id: int, non_continuous_types: List[int]
    ) -> List[Dict[str, Any]]:

        query = """
            SELECT 
                mc.id_tag AS tag_id,
                mc.id_model AS model_id,
                model.name AS model_name, 
                ut.name AS tag_name, 
                ut.source_type_id, 
                ulv.date_value AS last_date
            FROM model.model_config mc
            JOIN dictionary.union_tags ut ON mc.id_tag = ut.id
            JOIN dictionary.union_tags model ON mc.id_model = model.id
            LEFT JOIN result.union_last_values ulv ON mc.id_tag = ulv.tag_id
            WHERE mc.id_model = %(model_id)s
            AND mc.id_cluster = %(cluster_id)s
            AND NOT ut.source_type_id = ANY(%(non_continuous_types)s)
            AND (ulv.date_value < NOW() - INTERVAL '1 hour' OR ulv.date_value IS NULL)
        """
        params = {
            "model_id": model_id,
            "cluster_id": cluster_id,
            "non_continuous_types": non_continuous_types,
        }
        return await self._db.fetch_all(query, params)

    # async def find_predictors_without_recent_values(self) -> List[Dict[str, Any]]: # TODO проверить, если работает, то можно оставить его. Но нужно в params добавить non_continuous_types: List[int] = [2, 3, 16, 27]

    #     query = """
    #         WITH models AS (
    #             SELECT id, name
    #             FROM dictionary.union_tags
    #             WHERE source_type_id = %(model_type_id)s
    #             AND disabled = FALSE
    #             AND (attributes->>'type_model')::INTEGER IS NOT NULL
    #             AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
    #             AND (attributes->>'view_diag')::BOOLEAN = TRUE
    #             AND name NOT LIKE %(layer1_filter)s
    #             AND name NOT LIKE %(layer2_filter)s
    #             AND name NOT LIKE %(layer3_filter)s
    #         ),
    #         ranked_clusters AS (
    #             SELECT
    #                 id_model AS model_id,
    #                 id AS cluster_id,
    #                 id_cluster AS cluster,
    #                 RANK() OVER(
    #                     PARTITION BY id_model, id_cluster
    #                     ORDER BY date DESC
    #                 ) AS cluster_order
    #             FROM model.model_cluster_config
    #             WHERE id_model IN (SELECT id FROM models)
    #         ),
    #         latest_clusters AS (
    #             SELECT model_id, cluster_id, cluster
    #             FROM ranked_clusters
    #             WHERE cluster_order = 1
    #         ),
    #         predictors AS (
    #             SELECT
    #                 mc.id_tag AS tag_id,
    #                 mc.id_model AS model_id,
    #                 model.name AS model_name,
    #                 ut.name AS tag_name,
    #                 ut.source_type_id,
    #                 ulv.date_value AS last_date,
    #                 lc.cluster_id
    #             FROM latest_clusters lc
    #             JOIN model.model_config mc
    #                 ON lc.model_id = mc.id_model
    #                 AND lc.cluster_id = mc.id_cluster
    #             JOIN dictionary.union_tags ut ON mc.id_tag = ut.id
    #             JOIN dictionary.union_tags model ON mc.id_model = model.id
    #             LEFT JOIN result.union_last_values ulv ON mc.id_tag = ulv.tag_id
    #             WHERE NOT ut.source_type_id = ANY(%(non_continuous_types)s)
    #         )
    #         SELECT
    #             tag_id,
    #             model_id,
    #             model_name,
    #             tag_name,
    #             source_type_id,
    #             last_date,
    #             cluster_id
    #         FROM predictors
    #         WHERE last_date < NOW() - INTERVAL '1 hour'
    #         OR last_date IS NULL
    #         ORDER BY model_id, tag_id
    #     """
    #     return await self._db.fetch_all(query, self._params)

    async def find_linear_models(self) -> List[Dict[str, Any]]:
        """
        Находит линейные модели.

        Returns:
            Список словарей с полями: id, name
        """
        query = """
            SELECT id, name
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = FALSE
            AND (attributes->>'type_model')::INTEGER = %(linear_type_model)s
            AND (attributes->>'view_diag')::BOOLEAN = TRUE
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)

    async def find_latest_clusters(self, model_id: int) -> List[Dict[str, Any]]:
        """
        Находит последние кластеры для модели.

        Returns:
            Список словарей с полями: model_id, cluster_id, cluster
        """
        query = """
            WITH clusters AS (
                SELECT 
                    id_model AS model_id,
                    id AS cluster_id,
                    id_cluster AS cluster,
                    RANK() OVER(
                        PARTITION BY id_model, id_cluster 
                        ORDER BY date DESC
                    ) AS cluster_order
                FROM model.model_cluster_config
                WHERE id_model = %(model_id)s
            ) 
            SELECT model_id, cluster_id, cluster
            FROM clusters
            WHERE cluster_order = 1
        """
        return await self._db.fetch_all(query, {"model_id": model_id})

    async def find_latest_binary(self, model_id: int, cluster: int) -> Optional[bytes]:
        """
        Находит последний бинарник модели для кластера.

        Returns:
            Бинарные данные или None
        """
        query = """
            WITH binaries AS (
                SELECT 
                    bin_data,
                    ROW_NUMBER() OVER (
                        PARTITION BY id_model
                        ORDER BY date DESC
                    ) AS binary_order
                FROM model.model_binary
                WHERE id_model = %(model_id)s
                AND cluster_id = %(cluster)s
            )
            SELECT bin_data 
            FROM binaries
            WHERE binary_order = 1
            LIMIT 1
        """
        result = await self._db.fetch_one(
            query, {"model_id": model_id, "cluster": cluster}
        )
        return result["bin_data"] if result else None

    async def count_predictors_in_config(self, model_id: int, cluster_id: int) -> int:
        """
        Считает количество предикторов в конфигурации.

        Returns:
            Количество предикторов
        """
        query = """
            SELECT COUNT(*) AS cnt
            FROM model.model_config
            WHERE id_model = %(model_id)s
            AND id_cluster = %(cluster_id)s
        """
        result = await self._db.fetch_one(
            query, {"model_id": model_id, "cluster_id": cluster_id}
        )
        return int(result["cnt"]) if result else 0

    async def find_models_with_type(self) -> List[Dict[str, Any]]:
        """
        Находит модели с указанным типом модели.

        Returns:
            Список словарей с полями: id, name, type_model
        """
        query = """
            SELECT 
                id, 
                name, 
                (attributes->>'type_model')::INTEGER AS type_model
            FROM dictionary.union_tags
            WHERE source_type_id = %(model_type_id)s
            AND disabled = FALSE
            AND (attributes->>'type_model')::INTEGER IS NOT NULL
            AND NOT ((attributes->>'type_model')::INTEGER = ANY(%(non_tested_model_types)s))
            AND (attributes->>'view_diag')::BOOLEAN = TRUE
            AND name NOT LIKE %(layer1_filter)s
            AND name NOT LIKE %(layer2_filter)s
            AND name NOT LIKE %(layer3_filter)s
            ORDER BY id
        """
        return await self._db.fetch_all(query, self._params)
