from typing import List, Dict, Any, Optional
import psycopg
from psycopg.rows import dict_row


class AsyncPostgre:
    """Асинхронный адаптер для работы с PostgreSQL через psycopg3"""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[psycopg.AsyncConnectionPool] = None

    async def connect(self):
        """Инициализирует пул соединений"""
        self._pool = psycopg.AsyncConnectionPool(
            self._dsn,
            min_size=2,
            max_size=10,
            open=True,
        )

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def fetch_all(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params or {})
                return await cur.fetchall()

    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params or {})
                return await cur.fetchone()
    
    async def fetch_limited(
    self,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    limit: int = 10
    ) -> List[Dict[str, Any]]:

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params or {})
                return await cur.fetchmany(limit)