from typing import Any, Dict, List, Optional

from psycopg.errors import IdleSessionTimeout
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout


class AsyncPostgre:
    """Асинхронный адаптер для работы с PostgreSQL через psycopg3"""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: AsyncConnectionPool | None = None

    async def _ensure_open(self):
        if self._pool is None or self._pool.closed:
            self._pool = AsyncConnectionPool(
                self._dsn,
                min_size=2,
                max_size=10,
                open=True,
            )
            await self._pool.wait()

    async def close(
        self,
    ):  # TODO не убивать весь пул, сделать что бы открывалость только одно новое соединение взамен мертвого
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]],
        fetch: str,
        limit: int | None = None,
    ):
        await self._ensure_open()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params or {})
                    if fetch == "all":
                        return await cur.fetchall()
                    if fetch == "one":
                        return await cur.fetchone()
                    if fetch == "many":
                        return await cur.fetchmany(limit or 10)
        except IdleSessionTimeout:
            await self.close()
            await self._ensure_open()
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params or {})
                    if fetch == "all":
                        return await cur.fetchall()
                    if fetch == "one":
                        return await cur.fetchone()
                    if fetch == "many":
                        return await cur.fetchmany(limit or 10)

    async def fetch_all(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return await self._execute(query, params, fetch="all")

    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._execute(query, params, fetch="one")

    async def fetch_limited(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._execute(query, params, fetch="many", limit=limit)
