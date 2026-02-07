from settings.settings import Settings
from db.postgre import AsyncPostgre
from repositories.queries import AsyncQueryRepository
from services.models_health import ModelsHealthCheck
from services.tools.models_health_tools import ModelsHealthTools

from fastmcp import FastMCP

mcp = FastMCP("ModelQualityChecker")

settings = Settings()

db = AsyncPostgre(dsn= settings.pg.dsn)
repository = AsyncQueryRepository(db=db, settings= settings.qr)
models_check = ModelsHealthCheck(repository= repository, settings= settings.service)

tools = ModelsHealthTools(mcp=mcp, service=models_check)
tools.register()



