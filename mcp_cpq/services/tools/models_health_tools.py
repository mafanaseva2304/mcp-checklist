from typing import Any, Dict

from fastmcp import FastMCP
from services.models_health import ModelsHealthCheck


class ModelsHealthTools:
    """
    Регистрирует методы ModelsHealthCheck как MCP tools на переданном mcp.
    Не создаёт FastMCP, только использует переданный интерфейс MCPRegistrar.
    """

    def __init__(self, mcp: FastMCP, service: ModelsHealthCheck):
        self._mcp = mcp
        self._service = service

    def register(self) -> None:
        mcp = self._mcp
        svc = self._service

        @mcp.tool(description="Проверка на соответствие бинарников и типов моделей")
        async def check_binary_type_compatibility() -> Dict[str, Any]:
            return (await svc.check_binary_type_compatibility()).model_dump()

        @mcp.tool(
            description="Проверка соответствия количества предикторов в бинарнике и конфигурации"
        )
        async def check_predictor_count_compatibility() -> Dict[str, Any]:
            return (await svc.check_predictor_count_compatibility()).model_dump()

        @mcp.tool(
            description="Проверка предикторов, у которых нет значений за последний час"
        )
        async def check_predictors_without_recent_values() -> Dict[str, Any]:
            return (await svc.check_predictors_without_recent_values()).model_dump()

        @mcp.tool(description="Поиск отключенных моделей через поле disabled")
        async def find_disabled_models() -> Dict[str, Any]:
            return (await svc.find_disabled_models()).model_dump()

        @mcp.tool(description="Поиск моделей с разными единицами измерения с их ЛА")
        async def find_models_with_different_units() -> Dict[str, Any]:
            return (await svc.find_models_with_different_units()).model_dump()

        @mcp.tool(
            description="Поиск моделей с неизменяющимися значениями за последний час"
        )
        async def find_models_with_frozen_values() -> Dict[str, Any]:
            return (await svc.find_models_with_frozen_values()).model_dump()

        @mcp.tool(
            description="Поиск моделей, у которых последнее значение упирается в пределы tech_min или tech_max"
        )
        async def find_models_with_limited_values() -> Dict[str, Any]:
            return (await svc.find_models_with_limited_values()).model_dump()

        @mcp.tool(description="Поиск моделей без даты добавления модели")
        async def find_models_without_evaluation_date() -> Dict[str, Any]:
            return (await svc.find_models_without_evaluation_date()).model_dump()

        @mcp.tool(description="Поиск моделей без указанных полей конфигурации ИПК")
        async def find_models_without_ipk_config() -> Dict[str, Any]:
            return (await svc.find_models_without_ipk_config()).model_dump()

        @mcp.tool(description="Поиск моделей без метрик ИПК")
        async def find_models_without_ipk_metrics() -> Dict[str, Any]:
            return (await svc.find_models_without_ipk_metrics()).model_dump()

        @mcp.tool(description="Поиск моделей без указанного поля lab_mse")
        async def find_models_without_lab_mse() -> Dict[str, Any]:
            return (await svc.find_models_without_lab_mse()).model_dump()

        @mcp.tool(description="Поиск моделей без привязки к ЛА")
        async def find_models_without_parent() -> Dict[str, Any]:
            return (await svc.find_models_without_parent()).model_dump()

        @mcp.tool(description="Поиск моделей без значений за последний час")
        async def find_models_without_recent_values() -> Dict[str, Any]:
            return (await svc.find_models_without_recent_values()).model_dump()

        @mcp.tool(
            description="Поиск моделей без значений статуса в таблице union_last_values"
        )
        async def find_models_without_status_last_values() -> Dict[str, Any]:
            return (await svc.find_models_without_status_last_values()).model_dump()

        @mcp.tool(description="Поиск моделей без привязанного тега статуса")
        async def find_models_without_status_tag() -> Dict[str, Any]:
            return (await svc.find_models_without_status_tag()).model_dump()

        @mcp.tool(
            description="Поиск моделей без значений статуса в таблице union_values"
        )
        async def find_models_without_status_values() -> Dict[str, Any]:
            return (await svc.find_models_without_status_values()).model_dump()

        @mcp.tool(description="Поиск моделей без пределов tech_min и tech_max")
        async def find_models_without_tech_limits() -> Dict[str, Any]:
            return (await svc.find_models_without_tech_limits()).model_dump()

        @mcp.tool(description="Поиск моделей без полей конфигурации дообучения")
        async def find_models_without_train_config() -> Dict[str, Any]:
            return (await svc.find_models_without_train_config()).model_dump()

        @mcp.tool(description="Поиск моделей без указанного поля view_diag")
        async def find_models_without_view_diag() -> Dict[str, Any]:
            return (await svc.find_models_without_view_diag()).model_dump()
