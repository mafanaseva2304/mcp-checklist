import pickle
from datetime import datetime
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List

from models.mCloudPoint import Smes_cp_model
from models.mCrystKerosene import CrystKerosene
from models.mDNPModel import DNP_model
from models.mFlashPoint import Smes_fp_model
from models.mGradientBoosting import mGradientBoosting
from models.mLinearRegression import mLinearRegression
from models.mMixBonus import Mix_Bonus
from models.mMixDupont import Mix_Dupont
from models.mMixEthyl import Mix_Ethyl
from models.mOptimizerModel import OptimizerModel
from models.mPipeline import mPipeline
from models.mRandomForest import mRandomForest
from models.mReservQuality import GetReservQuality
from models.mSmesModel import Smes_model
from models.mSmesTemp import Smes_temp_model
from models.mSmesTempVols import Smes_temp_vols_model
from pydantic import BaseModel, Field
from repositories.queries import AsyncQueryRepository
from settings.settings import ServiceSettings

MODEL_BINARIES: Dict[int, List[type]] = {
    0: [
        mLinearRegression,
        Smes_temp_model,
        Smes_temp_vols_model,
        Smes_fp_model,
        DNP_model,
        Smes_cp_model,
        CrystKerosene,
        mRandomForest,
        mGradientBoosting,
        mPipeline,
        Mix_Bonus,
        Mix_Dupont,
        Mix_Ethyl,
        Smes_model,
    ],
    1: [
        Smes_model,
    ],
    10: [
        GetReservQuality,
    ],
    11: [
        OptimizerModel,
    ],
}


class LimitedResult(BaseModel):
    count_rows: int = Field(description="Сколько всего строк вернул запрос")
    first_n: List[Dict[str, Any]] = Field(description="Первые N строк (по лимиту)")
    all_rows_included: bool = Field(description="Все ли строки попали в first_n")


def formed_result(limit: int):
    """
    Декоратор для async-функций, которые возвращают List[Dict].
    Обрезает результат по лимиту и заворачивает в LimitedResult.
    """

    def decorator(func: Callable[..., Awaitable[List[Dict[str, Any]]]]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> LimitedResult:
            rows = await func(*args, **kwargs)
            count = len(rows)
            return LimitedResult(
                count_rows=count,
                first_n=rows[:limit],
                all_rows_included=count <= limit,
            )

        return wrapper

    return decorator


class ModelsHealthCheck:
    def __init__(self, repository: AsyncQueryRepository, settings: ServiceSettings):
        self._repo = repository
        # self._limit = limit #TODO переделать
        self._non_continuous_types = settings.non_continuous_types

    @formed_result(10)
    async def d(self) -> LimitedResult:
        pass

    @formed_result(10)
    async def find_models_without_status_tag(self) -> LimitedResult:
        return await self._repo.find_models_without_status_tag()

    @formed_result(10)
    async def find_models_without_status_values(self) -> LimitedResult:
        return await self._repo.find_models_without_status_values()

    @formed_result(10)
    async def find_models_without_status_last_values(self) -> LimitedResult:
        return await self._repo.find_models_without_status_last_values()

    @formed_result(10)
    async def find_models_without_lab_mse(self) -> LimitedResult:
        return await self._repo.find_models_without_lab_mse()

    @formed_result(10)
    async def find_models_without_view_diag(self) -> LimitedResult:
        return await self._repo.find_models_without_view_diag()

    @formed_result(10)
    async def find_models_without_ipk_config(self) -> LimitedResult:
        return await self._repo.find_models_without_ipk_config()

    @formed_result(10)
    async def find_models_without_train_config(self) -> LimitedResult:
        return await self._repo.find_models_without_train_config()

    @formed_result(10)
    async def find_models_without_recent_values(self) -> LimitedResult:
        return await self._repo.find_models_without_recent_values()

    @formed_result(10)
    async def find_disabled_models(self) -> LimitedResult:
        return await self._repo.find_disabled_models()

    @formed_result(10)
    async def find_models_without_parent(self) -> LimitedResult:
        return await self._repo.find_models_without_parent()

    @formed_result(10)
    async def find_models_with_different_units(self) -> LimitedResult:
        return await self._repo.find_models_with_different_units()

    @formed_result(10)
    async def find_models_without_tech_limits(self) -> LimitedResult:
        return await self._repo.find_models_without_tech_limits()

    @formed_result(10)
    async def find_models_with_limited_values(self) -> LimitedResult:
        return await self._repo.find_models_with_limited_values()

    @formed_result(10)
    async def find_models_with_frozen_values(self) -> LimitedResult:
        return await self._repo.find_models_with_frozen_values()

    @formed_result(10)
    async def find_models_without_evaluation_date(self) -> LimitedResult:
        return await self._repo.find_models_without_evaluation_date()

    @formed_result(10)
    async def find_models_without_ipk_metrics(self) -> LimitedResult:
        return await self._repo.find_models_without_ipk_metrics()

    @formed_result(10)
    async def check_binary_type_compatibility(self) -> LimitedResult:
        """
        Проверяет соответствие бинарников и типов моделей.
        Возвращает LimitedResult по списку проблем.
        """
        models = await self._repo.find_models_with_type()
        incorrect_binaries: List[Dict[str, Any]] = []

        for model in models:
            model_id = model["id"]
            name = model["name"]
            type_model = model["type_model"]

            clusters = await self._repo.find_latest_clusters(model_id)

            for cluster_info in clusters:
                cluster = cluster_info["cluster"]

                try:
                    bin_data = await self._repo.find_latest_binary(model_id, cluster)
                    if not bin_data:
                        # нет бинарника для этого кластера — пропускаем
                        continue

                    binary = pickle.loads(bin_data)

                except Exception as e:
                    incorrect_binaries.append(
                        {
                            "model_id": model_id,
                            "name": name,
                            "cluster": cluster,
                            "type_model": type_model,
                            "binary_class": None,
                            "error": type(e).__name__,
                        }
                    )
                    continue

                # Неизвестный type_model
                model_type_binaries = MODEL_BINARIES.get(type_model)
                if model_type_binaries is None:
                    incorrect_binaries.append(
                        {
                            "model_id": model_id,
                            "name": name,
                            "cluster": cluster,
                            "type_model": type_model,
                            "binary_class": type(binary).__name__,
                            "error": f"Неизвестный тип модели {type_model}",
                        }
                    )
                    continue

                # Проверка, что класс бинарника разрешён для этого type_model
                if not any(isinstance(binary, cls) for cls in model_type_binaries):
                    incorrect_binaries.append(
                        {
                            "model_id": model_id,
                            "name": name,
                            "cluster": cluster,
                            "type_model": type_model,
                            "binary_class": type(binary).__name__,
                            "error": None,
                        }
                    )

        return incorrect_binaries

    @formed_result(10)
    async def check_predictors_without_recent_values(self) -> LimitedResult:
        """
        Проверяет предикторы без значений за последний час.
        Возвращает LimitedResult по списку «устаревших» предикторов.
        """
        models = await self._repo.find_models_for_predictor_check()
        all_outdated: List[Dict[str, Any]] = []

        for model in models:
            model_id = model["id"]

            clusters = await self._repo.find_latest_clusters(model_id)

            for cluster in clusters:
                cluster_id = cluster["cluster_id"]

                predictors = await self._repo.find_outdated_predictors(
                    model_id=model_id,
                    cluster_id=cluster_id,
                    non_continuous_types=self._non_continuous_types,
                )

                # фильтруем только устаревшие
                for pred in predictors:
                    if not pred.get("outdated"):
                        continue

                    last_date = pred.get("last_date")
                    all_outdated.append(
                        {
                            "tag_id": pred["tag_id"],
                            "model_id": pred["model_id"],
                            "model_name": pred["model_name"],
                            "tag_name": pred["tag_name"],
                            "source_type_id": pred["source_type_id"],
                            "last_date": (
                                last_date.isoformat()
                                if isinstance(last_date, datetime)
                                else None
                            ),
                            "cluster_id": cluster_id,
                        }
                    )
        return all_outdated

    @formed_result(10)
    async def check_predictor_count_compatibility(self) -> LimitedResult:
        """
        Проверяет соответствие количества предикторов в бинарнике и конфиге
        для линейных моделей (mLinearRegression).
        """
        linear_models = await self._repo.find_linear_models()
        diff_pred_count: List[Dict[str, Any]] = []

        for model in linear_models:
            model_id = model["id"]
            name = model["name"]

            clusters = await self._repo.find_latest_clusters(model_id)

            for cluster_info in clusters:
                cluster = cluster_info["cluster"]
                cluster_id = cluster_info["cluster_id"]

                try:
                    bin_data = await self._repo.find_latest_binary(model_id, cluster)
                    if not bin_data:
                        # нет бинарника — пропускаем
                        continue

                    binary = pickle.loads(bin_data)

                except Exception:
                    # любые проблемы с бинарником просто пропускаем (как в исходном коде)
                    continue

                # интересуют только линейные модели
                if not isinstance(binary, mLinearRegression):
                    continue

                # количество предикторов в конфигурации
                config_pred_count = await self._repo.count_predictors_in_config(
                    model_id=model_id,
                    cluster_id=cluster_id,
                )

                # количество предикторов в бинарнике
                binary_pred_count = int(len(np.ravel(binary.coef_).tolist()))

                if config_pred_count != binary_pred_count:
                    diff_pred_count.append(
                        {
                            "model_id": model_id,
                            "cluster": cluster,
                            "name": name,
                            "config_pred_count": config_pred_count,
                            "binary_pred_count": binary_pred_count,
                        }
                    )

        return diff_pred_count
