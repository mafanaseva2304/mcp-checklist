import asyncio
import warnings

from core import mcp, settings

warnings.filterwarnings("ignore")

from core import models_check


async def main():
    await mcp.run_async(
        transport=settings.service.transport,
        host=settings.service.host,
        port=settings.service.port,
    )


if __name__ == "__main__":
    asyncio.run(main())
