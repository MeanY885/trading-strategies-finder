"""
ASYNC DATABASE
===============
Async PostgreSQL layer using asyncpg for non-blocking database operations.
Used by WebSocket handlers and elite validation to prevent event loop blocking.
"""
import asyncio
import asyncpg
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime


class AsyncDatabase:
    """
    Async database wrapper using asyncpg connection pool.
    Provides non-blocking database operations for async contexts.
    """
    _pool: Optional[asyncpg.Pool] = None
    _pool_lock: asyncio.Lock = None

    @classmethod
    async def init_pool(cls, database_url: str = None):
        """Initialize the connection pool."""
        # Lazy init the lock (safe because GIL protects attribute assignment)
        if cls._pool_lock is None:
            cls._pool_lock = asyncio.Lock()

        async with cls._pool_lock:
            if cls._pool is not None:
                return  # Already initialized

            url = database_url or os.environ.get('DATABASE_URL')
            if not url:
                raise ValueError("DATABASE_URL environment variable is required")

            cls._pool = await asyncpg.create_pool(
                url,
                min_size=5,
                max_size=40,
                command_timeout=30,  # 30s query timeout
                statement_cache_size=100
            )

    @classmethod
    async def close_pool(cls):
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    def _row_to_dict(cls, row: asyncpg.Record, parse_equity_curve: bool = True) -> Dict:
        """Convert an asyncpg Record to a dictionary with parsed JSON fields."""
        d = dict(row)

        # Parse JSON fields
        if d.get('params'):
            try:
                d['params'] = json.loads(d['params'])
            except:
                d['params'] = {}

        if d.get('found_by'):
            try:
                d['found_by'] = json.loads(d['found_by'])
            except:
                d['found_by'] = []

        if parse_equity_curve and d.get('equity_curve'):
            try:
                d['equity_curve'] = json.loads(d['equity_curve'])
            except:
                d['equity_curve'] = []
        elif not parse_equity_curve:
            d['equity_curve'] = []

        # Convert datetime to string if needed
        if d.get('created_at') and hasattr(d['created_at'], 'isoformat'):
            d['created_at'] = d['created_at'].isoformat()

        return d

    # =========================================================================
    # Elite Validator Methods
    # =========================================================================

    @classmethod
    async def get_top_pending_per_market(cls, top_n: int = 3) -> List[Dict]:
        """
        Get top N pending strategies per (symbol, timeframe) combination.
        Uses window functions to rank by composite_score within each market.
        """
        async with cls._pool.acquire() as conn:
            rows = await conn.fetch('''
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, timeframe
                            ORDER BY composite_score DESC
                        ) as rank
                    FROM strategies
                    WHERE elite_status IS NULL OR elite_status = 'pending'
                )
                SELECT * FROM ranked
                WHERE rank <= $1
                ORDER BY composite_score DESC
            ''', top_n)
            return [cls._row_to_dict(row) for row in rows]

    @classmethod
    async def update_elite_status(cls, strategy_id: int, elite_status: str,
                                   periods_passed: int, periods_total: int,
                                   validation_data: str = None,
                                   elite_score: float = 0) -> bool:
        """Update elite validation status for a strategy."""
        async with cls._pool.acquire() as conn:
            result = await conn.execute('''
                UPDATE strategies
                SET elite_status = $1,
                    elite_validated_at = $2,
                    elite_periods_passed = $3,
                    elite_periods_total = $4,
                    elite_validation_data = $5,
                    elite_score = $6
                WHERE id = $7
            ''', elite_status, datetime.now().isoformat(), periods_passed,
                 periods_total, validation_data, elite_score, strategy_id)
            return result == 'UPDATE 1'

    @classmethod
    async def get_priority_list(cls) -> List[Dict]:
        """Get all priority items ordered by position."""
        async with cls._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM priority_items
                ORDER BY position ASC
            ''')
            return [dict(row) for row in rows]

    @classmethod
    async def get_pending_validation_count(cls) -> int:
        """Get count of strategies pending validation."""
        async with cls._pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT COUNT(*) as count FROM strategies
                WHERE elite_status IS NULL OR elite_status = 'pending'
            ''')
            return row['count']

    @classmethod
    async def get_elite_strategies_filtered(cls, status_filter: str = None,
                                             limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get elite strategies with SQL-level filtering."""
        async with cls._pool.acquire() as conn:
            if status_filter:
                rows = await conn.fetch('''
                    SELECT * FROM strategies
                    WHERE elite_status = $1
                    ORDER BY elite_score DESC
                    LIMIT $2 OFFSET $3
                ''', status_filter, limit, offset)
            else:
                rows = await conn.fetch('''
                    SELECT * FROM strategies
                    WHERE elite_status IS NOT NULL
                    ORDER BY elite_score DESC
                    LIMIT $1 OFFSET $2
                ''', limit, offset)
            return [cls._row_to_dict(row, parse_equity_curve=False) for row in rows]

    # =========================================================================
    # WebSocket Handler Methods
    # =========================================================================

    @classmethod
    async def get_strategies_paginated(cls, limit: int = 500, offset: int = 0,
                                        symbol: str = None, timeframe: str = None,
                                        sort_by: str = 'id', sort_order: str = 'DESC') -> Dict:
        """Get strategies with pagination support."""
        async with cls._pool.acquire() as conn:
            where_clauses = []
            params = []
            param_idx = 1

            if symbol:
                where_clauses.append(f'symbol = ${param_idx}')
                params.append(symbol)
                param_idx += 1

            if timeframe:
                where_clauses.append(f'timeframe = ${param_idx}')
                params.append(timeframe)
                param_idx += 1

            where_sql = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''

            valid_sort_columns = {'id', 'composite_score', 'win_rate', 'profit_factor',
                                  'total_pnl', 'created_at', 'elite_score'}
            if sort_by not in valid_sort_columns:
                sort_by = 'id'

            sort_direction = 'DESC' if sort_order.upper() == 'DESC' else 'ASC'

            # Get total count
            count_row = await conn.fetchrow(
                f'SELECT COUNT(*) as count FROM strategies{where_sql}',
                *params
            )
            total = count_row['count']

            # Get paginated results
            query = f'''
                SELECT * FROM strategies{where_sql}
                ORDER BY {sort_by} {sort_direction}
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            '''
            rows = await conn.fetch(query, *params, limit, offset)

            strategies = [cls._row_to_dict(row) for row in rows]

            return {
                'strategies': strategies,
                'total': total,
                'limit': limit,
                'offset': offset
            }

    @classmethod
    async def get_elite_counts(cls) -> Dict[str, int]:
        """Get elite status counts using SQL aggregation."""
        async with cls._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT
                    COALESCE(elite_status, 'pending') as status,
                    COUNT(*) as count
                FROM strategies
                GROUP BY COALESCE(elite_status, 'pending')
            ''')

            # Simplified status model: validated, pending, untestable, skipped
            counts = {
                'validated': 0,
                'pending': 0,
                'untestable': 0,
                'skipped': 0
            }

            for row in rows:
                status = row['status']
                count = row['count']
                if status in counts:
                    counts[status] = count
                elif status in ['elite', 'partial', 'failed']:
                    # Legacy statuses - count as validated
                    counts['validated'] += count
                elif status is None:
                    counts['pending'] += count

            return counts

    @classmethod
    async def get_elite_strategies_optimized(cls, top_n_per_market: int = 10) -> List[Dict]:
        """Get top N validated strategies per pair/timeframe."""
        async with cls._pool.acquire() as conn:
            rows = await conn.fetch('''
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, timeframe
                            ORDER BY elite_score DESC
                        ) as rank
                    FROM strategies
                    WHERE elite_status IN ('validated', 'elite', 'partial', 'failed')
                      AND elite_score > 0
                )
                SELECT * FROM ranked
                WHERE rank <= $1
                ORDER BY elite_score DESC
            ''', top_n_per_market)
            return [cls._row_to_dict(row) for row in rows]

    @classmethod
    async def get_db_stats_optimized(cls) -> Dict[str, Any]:
        """Get database statistics using SQL aggregation."""
        async with cls._pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT
                    COUNT(*) as total_strategies,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT timeframe) as unique_timeframes,
                    SUM(CASE WHEN elite_status IN ('validated', 'elite', 'partial', 'failed') THEN 1 ELSE 0 END) as validated_count
                FROM strategies
            ''')

            return {
                'total_strategies': row['total_strategies'] or 0,
                'unique_symbols': row['unique_symbols'] or 0,
                'unique_timeframes': row['unique_timeframes'] or 0,
                'validated_count': row['validated_count'] or 0
            }

    @classmethod
    async def get_all_priority_lists(cls) -> Dict:
        """Get all priority lists in a single database connection."""
        async with cls._pool.acquire() as conn:
            pairs_count = await conn.fetchrow('SELECT COUNT(*) as count FROM priority_pairs')
            populated = pairs_count['count'] > 0

            pairs = await conn.fetch(
                'SELECT id, position, value, label, enabled FROM priority_pairs ORDER BY position ASC'
            )
            periods = await conn.fetch(
                'SELECT id, position, value, label, months, enabled FROM priority_periods ORDER BY position ASC'
            )
            timeframes = await conn.fetch(
                'SELECT id, position, value, label, minutes, enabled FROM priority_timeframes ORDER BY position ASC'
            )
            granularities = await conn.fetch(
                'SELECT id, position, value, label, n_trials, enabled FROM priority_granularities ORDER BY position ASC'
            )

            return {
                'pairs': [dict(row) for row in pairs],
                'periods': [dict(row) for row in periods],
                'timeframes': [dict(row) for row in timeframes],
                'granularities': [dict(row) for row in granularities],
                'populated': populated
            }

    @classmethod
    async def reset_all_elite_validation(cls) -> int:
        """Reset all elite validation data. Returns count of reset strategies."""
        async with cls._pool.acquire() as conn:
            # Get count first
            count_row = await conn.fetchrow('SELECT COUNT(*) as count FROM strategies')
            total_count = count_row['count']

            # Reset all elite validation data
            await conn.execute('''
                UPDATE strategies
                SET elite_status = 'pending',
                    elite_score = 0,
                    elite_periods_passed = 0,
                    elite_periods_total = 0,
                    elite_validation_data = NULL
            ''')

            return total_count
