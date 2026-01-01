"""
STRATEGY DATABASE
==================
PostgreSQL persistence layer for storing winning strategies across sessions.
Allows loading historical results and comparing performance over time.

Now uses connection pooling to reduce overhead and support concurrent operations.
"""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import register_adapter, AsIs
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import os
import numpy as np
import threading
from logging_config import log

# Register numpy type adapters for psycopg2
# Without this, numpy.float64 gets serialized as "np.float64(...)" which crashes PostgreSQL
def _adapt_numpy_float64(numpy_float64):
    return AsIs(float(numpy_float64))

def _adapt_numpy_float32(numpy_float32):
    return AsIs(float(numpy_float32))

def _adapt_numpy_int64(numpy_int64):
    return AsIs(int(numpy_int64))

def _adapt_numpy_int32(numpy_int32):
    return AsIs(int(numpy_int32))

def _adapt_numpy_bool(numpy_bool):
    return AsIs(bool(numpy_bool))

register_adapter(np.float64, _adapt_numpy_float64)
register_adapter(np.float32, _adapt_numpy_float32)
register_adapter(np.int64, _adapt_numpy_int64)
register_adapter(np.int32, _adapt_numpy_int32)
register_adapter(np.bool_, _adapt_numpy_bool)


class StrategyDatabase:
    """
    PostgreSQL database for persisting strategy results.

    Schema:
    - strategies: Core strategy results with metrics
    - strategy_trades: Individual trade records
    - optimization_runs: Track optimization sessions

    Uses ThreadedConnectionPool for efficient connection reuse.
    """

    # Class-level connection pool (shared across instances)
    _pool: Optional[pool.ThreadedConnectionPool] = None
    _pool_lock = threading.Lock()

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Initialize connection pool if not already done
        self._init_pool()

        self._init_database()
        self._init_priority_table()

        # Auto-clean duplicates on startup
        self._auto_deduplicate()

    def _init_pool(self):
        """Initialize the connection pool (thread-safe)."""
        if StrategyDatabase._pool is None:
            with StrategyDatabase._pool_lock:
                if StrategyDatabase._pool is None:
                    try:
                        StrategyDatabase._pool = pool.ThreadedConnectionPool(
                            minconn=5,
                            maxconn=40,
                            dsn=self.database_url,
                            options="-c statement_timeout=30000"  # 30s query timeout
                        )
                    except Exception as e:
                        log(f"[DB Pool] Failed to create pool: {e}", level='ERROR')
                        raise

    def _get_connection(self):
        """
        Get a database connection from the pool.
        Returns a connection that should be returned via _return_connection().
        """
        if StrategyDatabase._pool is None:
            self._init_pool()
        return StrategyDatabase._pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if StrategyDatabase._pool is not None and conn is not None:
            StrategyDatabase._pool.putconn(conn)

    @contextmanager
    def get_connection(self):
        """
        Context manager for safe database connection handling.
        Automatically returns connection to pool and handles rollback on error.

        Usage:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(...)
                conn.commit()
        """
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                strategy_category TEXT,
                params TEXT,

                -- Core metrics
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                total_pnl REAL,
                max_drawdown REAL,

                -- Advanced metrics
                equity_r_squared REAL,
                recovery_factor REAL,
                sharpe_ratio REAL,
                composite_score REAL,

                -- Risk parameters (exact-match)
                tp_percent REAL,
                sl_percent REAL,

                -- Indicator parameters (Phase 2 tuning - JSON)
                indicator_params TEXT,
                tuning_improved INTEGER DEFAULT 0,
                tuning_score_before REAL,
                tuning_score_after REAL,
                tuning_improvement_pct REAL,

                -- Validation results
                val_pnl REAL,
                val_profit_factor REAL,
                val_win_rate REAL,

                -- Metadata
                found_by TEXT,
                data_source TEXT,
                symbol TEXT,
                timeframe TEXT,
                data_start TEXT,
                data_end TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                optimization_run_id INTEGER,

                -- Equity curve (JSON)
                equity_curve TEXT,

                -- Bidirectional trading fields
                trade_mode TEXT DEFAULT 'single',
                long_trades INTEGER DEFAULT 0,
                long_wins INTEGER DEFAULT 0,
                long_pnl REAL DEFAULT 0,
                short_trades INTEGER DEFAULT 0,
                short_wins INTEGER DEFAULT 0,
                short_pnl REAL DEFAULT 0,
                flip_count INTEGER DEFAULT 0,

                -- Elite strategy validation fields
                elite_status TEXT DEFAULT 'pending',
                elite_validated_at TEXT,
                elite_periods_passed INTEGER DEFAULT 0,
                elite_periods_total INTEGER DEFAULT 0,
                elite_validation_data TEXT,
                elite_score REAL DEFAULT 0,

                -- Dual Pool Architecture fields
                pool TEXT DEFAULT 'tp_sl',
                exit_type TEXT DEFAULT 'fixed_tp_sl',
                exit_indicator TEXT,
                trailing_atr_mult REAL,
                strategy_style TEXT DEFAULT 'unknown',

                -- Trend-following metrics
                avg_trade_duration_hours REAL DEFAULT 0,
                mfe_capture_ratio REAL DEFAULT 0,
                avg_winner_pct REAL DEFAULT 0,
                avg_loser_pct REAL DEFAULT 0,
                risk_reward_ratio REAL DEFAULT 0,
                trend_following_score REAL DEFAULT 0,

                -- VectorBT metrics
                total_pnl_percent REAL DEFAULT 0,
                avg_trade REAL DEFAULT 0,
                buy_hold_return REAL DEFAULT 0,
                vs_buy_hold REAL DEFAULT 0,
                consistency_score REAL DEFAULT 0,

                -- Trade details for debugger
                trades_list TEXT
            )
        ''')

        # Optimization runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id SERIAL PRIMARY KEY,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                symbol TEXT,
                timeframe TEXT,
                data_source TEXT,
                data_rows INTEGER,
                capital REAL,
                risk_percent REAL,
                strategies_tested INTEGER,
                profitable_found INTEGER,
                status TEXT DEFAULT 'running'
            )
        ''')

        # Completed optimizations tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed_optimizations (
                id SERIAL PRIMARY KEY,
                pair TEXT NOT NULL,
                period_label TEXT NOT NULL,
                timeframe_label TEXT NOT NULL,
                granularity_label TEXT NOT NULL,
                strategies_found INTEGER DEFAULT 0,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'binance',
                duration_seconds INTEGER DEFAULT NULL,
                UNIQUE(pair, period_label, timeframe_label, granularity_label)
            )
        ''')

        # Add duration_seconds column if it doesn't exist (for existing databases)
        cursor.execute('''
            ALTER TABLE completed_optimizations
            ADD COLUMN IF NOT EXISTS duration_seconds INTEGER DEFAULT NULL
        ''')

        # Add VectorBT columns if they don't exist (for existing databases)
        vectorbt_columns = [
            "ALTER TABLE strategies ADD COLUMN IF NOT EXISTS total_pnl_percent REAL DEFAULT 0",
            "ALTER TABLE strategies ADD COLUMN IF NOT EXISTS avg_trade REAL DEFAULT 0",
            "ALTER TABLE strategies ADD COLUMN IF NOT EXISTS buy_hold_return REAL DEFAULT 0",
            "ALTER TABLE strategies ADD COLUMN IF NOT EXISTS vs_buy_hold REAL DEFAULT 0",
            "ALTER TABLE strategies ADD COLUMN IF NOT EXISTS consistency_score REAL DEFAULT 0",
        ]
        for stmt in vectorbt_columns:
            cursor.execute(stmt)

        # Add trades_list column for debugger (for existing databases)
        cursor.execute("ALTER TABLE strategies ADD COLUMN IF NOT EXISTS trades_list TEXT")

        # Optimization checkpoints table for crash recovery
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_checkpoints (
                run_id VARCHAR(255) PRIMARY KEY,
                checkpoint_data JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')

        # Create indexes for fast querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_completed_combo ON completed_optimizations(pair, period_label, timeframe_label, granularity_label)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_win_rate ON strategies(win_rate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_profit_factor ON strategies(profit_factor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_composite_score ON strategies(composite_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON strategies(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON strategies(timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_status ON strategies(elite_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_score ON strategies(elite_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_combo ON strategies(symbol, timeframe, elite_status, elite_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_status_score ON strategies(elite_status, elite_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON strategies(symbol, timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON strategies(created_at DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_validation ON strategies(elite_status, composite_score DESC)')

        conn.commit()
        self._return_connection(conn)

        log("Strategy database initialized (PostgreSQL)")

    def _auto_deduplicate(self):
        """Automatically remove duplicates on startup."""
        removed = self.remove_duplicates()
        if removed > 0:
            log(f"Auto-cleaned {removed} duplicate strategies")

    def start_optimization_run(self, symbol: str = None, timeframe: str = None,
                               data_source: str = None, data_rows: int = 0,
                               capital: float = 1000.0, risk_percent: float = 2.0) -> int:
        """Start a new optimization run and return its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO optimization_runs
            (symbol, timeframe, data_source, data_rows, capital, risk_percent, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'running')
            RETURNING id
        ''', (symbol, timeframe, data_source, data_rows, capital, risk_percent))

        run_id = cursor.fetchone()[0]
        conn.commit()
        self._return_connection(conn)

        log(f"Started optimization run #{run_id}")
        return run_id

    def complete_optimization_run(self, run_id: int, strategies_tested: int,
                                  profitable_found: int):
        """Mark an optimization run as complete."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE optimization_runs
            SET completed_at = %s, strategies_tested = %s, profitable_found = %s, status = 'completed'
            WHERE id = %s
        ''', (datetime.now().isoformat(), strategies_tested, profitable_found, run_id))

        conn.commit()
        self._return_connection(conn)

        log(f"Completed optimization run #{run_id}: {profitable_found} profitable strategies")

    def checkpoint_optimization_progress(self, run_id: str, checkpoint_data: dict) -> bool:
        """
        Save optimization checkpoint for crash recovery.

        Args:
            run_id: Unique identifier for this optimization run
            checkpoint_data: Dict with keys: strategies_tested, results_so_far, last_strategy, last_direction

        Returns:
            True if checkpoint saved successfully
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO optimization_checkpoints (run_id, checkpoint_data, created_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (run_id) DO UPDATE SET
                            checkpoint_data = EXCLUDED.checkpoint_data,
                            updated_at = NOW()
                    """, (run_id, json.dumps(checkpoint_data)))
                    conn.commit()
            return True
        except Exception as e:
            log(f"[DB] Checkpoint save failed: {e}", level='WARNING')
            return False

    def load_optimization_checkpoint(self, run_id: str) -> Optional[dict]:
        """Load checkpoint data for resuming optimization."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT checkpoint_data FROM optimization_checkpoints
                        WHERE run_id = %s
                    """, (run_id,))
                    row = cur.fetchone()
                    if row:
                        return json.loads(row[0])
        except Exception as e:
            log(f"[DB] Checkpoint load failed: {e}", level='WARNING')
        return None

    def save_strategy(self, result: Any, run_id: int = None,
                      symbol: str = None, timeframe: str = None,
                      data_source: str = None, data_start: str = None,
                      data_end: str = None,
                      indicator_params: Dict = None,
                      tuning_info: Dict = None) -> Optional[int]:
        """
        Save a strategy result to the database.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract params
        params = result.params if hasattr(result, 'params') else {}
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

        # Check for duplicate
        strategy_name = getattr(result, 'strategy_name', 'unknown')
        total_trades = getattr(result, 'total_trades', 0)
        win_rate = getattr(result, 'win_rate', 0)
        total_pnl = getattr(result, 'total_pnl', 0)
        profit_factor = getattr(result, 'profit_factor', 0)

        cursor.execute('''
            SELECT id FROM strategies
            WHERE strategy_name = %s
              AND symbol = %s
              AND timeframe = %s
              AND ABS(tp_percent - %s) < 0.01
              AND ABS(sl_percent - %s) < 0.01
              AND total_trades = %s
              AND ABS(win_rate - %s) < 0.01
              AND ABS(total_pnl - %s) < 0.01
              AND ABS(profit_factor - %s) < 0.01
            LIMIT 1
        ''', (strategy_name, symbol, timeframe, tp_percent, sl_percent,
              total_trades, win_rate, total_pnl, profit_factor))

        existing = cursor.fetchone()
        if existing:
            self._return_connection(conn)
            log(f"Skipping duplicate strategy: {strategy_name} (TP={tp_percent}%, SL={sl_percent}%)", level='WARNING')
            return None

        # Convert found_by list to JSON
        found_by = json.dumps(result.found_by) if hasattr(result, 'found_by') else '[]'

        # Convert equity curve to JSON
        equity_curve = json.dumps(result.equity_curve) if hasattr(result, 'equity_curve') else '[]'

        # Convert trades_list to JSON for debugger
        trades_list = json.dumps(result.trades_list) if hasattr(result, 'trades_list') and result.trades_list else None

        # Convert indicator_params to JSON
        indicator_params_json = json.dumps(indicator_params) if indicator_params else None

        # Extract tuning info
        tuning_improved = 0
        tuning_score_before = None
        tuning_score_after = None
        tuning_improvement_pct = None
        if tuning_info:
            tuning_improved = 1 if tuning_info.get('improved', False) else 0
            tuning_score_before = tuning_info.get('before_score')
            tuning_score_after = tuning_info.get('after_score')
            tuning_improvement_pct = tuning_info.get('improvement_pct')

        # Determine trade mode from direction
        direction = getattr(result, 'direction', 'long')
        trade_mode = 'bidirectional' if direction == 'both' else direction

        cursor.execute('''
            INSERT INTO strategies
            (strategy_name, strategy_category, params, total_trades, win_rate,
             profit_factor, total_pnl, max_drawdown, equity_r_squared, recovery_factor,
             sharpe_ratio, composite_score, tp_percent, sl_percent,
             indicator_params, tuning_improved, tuning_score_before, tuning_score_after, tuning_improvement_pct,
             val_pnl, val_profit_factor, val_win_rate, found_by, data_source, symbol,
             timeframe, data_start, data_end, optimization_run_id, equity_curve,
             trade_mode, long_trades, long_wins, long_pnl, short_trades, short_wins, short_pnl, flip_count,
             total_pnl_percent, avg_trade, buy_hold_return, vs_buy_hold, consistency_score, trades_list)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (
            getattr(result, 'strategy_name', 'unknown'),
            getattr(result, 'strategy_category', 'unknown'),
            json.dumps(params),
            getattr(result, 'total_trades', 0),
            getattr(result, 'win_rate', 0),
            getattr(result, 'profit_factor', 0),
            getattr(result, 'total_pnl', 0),
            getattr(result, 'max_drawdown', 0),
            getattr(result, 'equity_r_squared', 0),
            getattr(result, 'recovery_factor', 0),
            getattr(result, 'sharpe_ratio', 0),
            getattr(result, 'composite_score', getattr(result, 'profit_factor', 0) * 10),
            tp_percent,
            sl_percent,
            indicator_params_json,
            tuning_improved,
            tuning_score_before,
            tuning_score_after,
            tuning_improvement_pct,
            getattr(result, 'val_pnl', 0),
            getattr(result, 'val_profit_factor', 0),
            getattr(result, 'val_win_rate', 0),
            found_by,
            data_source,
            symbol,
            timeframe,
            data_start,
            data_end,
            run_id,
            equity_curve,
            trade_mode,
            getattr(result, 'long_trades', 0),
            getattr(result, 'long_wins', 0),
            getattr(result, 'long_pnl', 0.0),
            getattr(result, 'short_trades', 0),
            getattr(result, 'short_wins', 0),
            getattr(result, 'short_pnl', 0.0),
            getattr(result, 'flip_count', 0),
            getattr(result, 'total_pnl_percent', 0.0),
            getattr(result, 'avg_trade', 0.0),
            getattr(result, 'buy_hold_return', 0.0),
            getattr(result, 'vs_buy_hold', 0.0),
            getattr(result, 'consistency_score', 0.0),
            trades_list
        ))

        strategy_id = cursor.fetchone()[0]
        conn.commit()
        self._return_connection(conn)

        return strategy_id

    def save_strategies_batch(self, results: List[Any], run_id: int = None,
                              symbol: str = None, timeframe: str = None,
                              data_source: str = None, data_start: str = None,
                              data_end: str = None,
                              skip_duplicates: bool = True) -> Dict[str, int]:
        """
        Save multiple strategy results efficiently using batch operations.

        Uses a single transaction with executemany() for dramatic performance improvement.
        100 strategies = 1 commit instead of 100 commits.

        Args:
            results: List of strategy results to save
            run_id: Optional optimization run ID
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m')
            data_source: Data source identifier
            data_start: Start date of data
            data_end: End date of data
            skip_duplicates: If True, skip duplicate strategies (default True)

        Returns:
            Dict with 'saved', 'skipped', and 'errors' counts
        """
        if not results:
            return {'saved': 0, 'skipped': 0, 'errors': 0}

        saved = 0
        skipped = 0
        errors = 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Build set of existing strategy keys for batch duplicate detection
                existing_keys = set()

                if skip_duplicates and results:
                    # Pre-extract keys from all results for batch lookup
                    # Key format: (strategy_name, symbol, timeframe, tp_percent_rounded, sl_percent_rounded,
                    #              total_trades, win_rate_rounded, total_pnl_rounded, profit_factor_rounded)
                    strategy_names = set()
                    for result in results:
                        strategy_name = getattr(result, 'strategy_name', 'unknown')
                        strategy_names.add(strategy_name)

                    # Single query to fetch all potentially matching strategies
                    # We fetch more columns than needed and filter in Python for exact matching
                    if strategy_names:
                        cursor.execute('''
                            SELECT strategy_name, symbol, timeframe, tp_percent, sl_percent,
                                   total_trades, win_rate, total_pnl, profit_factor
                            FROM strategies
                            WHERE strategy_name IN %s
                              AND symbol = %s
                              AND timeframe = %s
                        ''', (tuple(strategy_names), symbol, timeframe))

                        # Build set of existing keys with rounded values for comparison
                        for row in cursor.fetchall():
                            # Round to 2 decimal places for comparison (matches ABS() < 0.01 logic)
                            key = (
                                row[0],  # strategy_name
                                row[1],  # symbol
                                row[2],  # timeframe
                                round(row[3], 2),  # tp_percent
                                round(row[4], 2),  # sl_percent
                                row[5],  # total_trades (int, no rounding)
                                round(row[6], 2),  # win_rate
                                round(row[7], 2),  # total_pnl
                                round(row[8], 2),  # profit_factor
                            )
                            existing_keys.add(key)

                # Prepare batch data
                batch_data = []

                for result in results:
                    try:
                        # Extract params
                        params = result.params if hasattr(result, 'params') else {}
                        tp_percent = params.get('tp_percent', 1.0)
                        sl_percent = params.get('sl_percent', 3.0)

                        strategy_name = getattr(result, 'strategy_name', 'unknown')
                        total_trades = getattr(result, 'total_trades', 0)
                        win_rate = getattr(result, 'win_rate', 0)
                        total_pnl = getattr(result, 'total_pnl', 0)
                        profit_factor = getattr(result, 'profit_factor', 0)

                        # Check for duplicate using pre-fetched set (O(1) lookup)
                        if skip_duplicates:
                            result_key = (
                                strategy_name,
                                symbol,
                                timeframe,
                                round(tp_percent, 2),
                                round(sl_percent, 2),
                                total_trades,
                                round(win_rate, 2),
                                round(total_pnl, 2),
                                round(profit_factor, 2),
                            )
                            if result_key in existing_keys:
                                skipped += 1
                                continue

                        # Convert found_by list to JSON
                        found_by = json.dumps(result.found_by) if hasattr(result, 'found_by') else '[]'

                        # Convert equity curve to JSON
                        equity_curve = json.dumps(result.equity_curve) if hasattr(result, 'equity_curve') else '[]'

                        # Convert trades_list to JSON for debugger
                        trades_list = json.dumps(result.trades_list) if hasattr(result, 'trades_list') and result.trades_list else None

                        # Determine trade mode from direction
                        direction = getattr(result, 'direction', 'long')
                        trade_mode = 'bidirectional' if direction == 'both' else direction

                        # Build tuple for batch insert
                        row_data = (
                            strategy_name,
                            getattr(result, 'strategy_category', 'unknown'),
                            json.dumps(params),
                            total_trades,
                            win_rate,
                            profit_factor,
                            total_pnl,
                            getattr(result, 'max_drawdown', 0),
                            getattr(result, 'equity_r_squared', 0),
                            getattr(result, 'recovery_factor', 0),
                            getattr(result, 'sharpe_ratio', 0),
                            getattr(result, 'composite_score', profit_factor * 10),
                            tp_percent,
                            sl_percent,
                            getattr(result, 'val_pnl', 0),
                            getattr(result, 'val_profit_factor', 0),
                            getattr(result, 'val_win_rate', 0),
                            found_by,
                            data_source,
                            symbol,
                            timeframe,
                            data_start,
                            data_end,
                            run_id,
                            equity_curve,
                            trade_mode,
                            getattr(result, 'long_trades', 0),
                            getattr(result, 'long_wins', 0),
                            getattr(result, 'long_pnl', 0.0),
                            getattr(result, 'short_trades', 0),
                            getattr(result, 'short_wins', 0),
                            getattr(result, 'short_pnl', 0.0),
                            getattr(result, 'flip_count', 0),
                            getattr(result, 'total_pnl_percent', 0.0),
                            getattr(result, 'avg_trade', 0.0),
                            getattr(result, 'buy_hold_return', 0.0),
                            getattr(result, 'vs_buy_hold', 0.0),
                            getattr(result, 'consistency_score', 0.0),
                            trades_list
                        )
                        batch_data.append(row_data)

                    except Exception as e:
                        log(f"Error preparing strategy {getattr(result, 'strategy_name', 'unknown')}: {e}", level='ERROR')
                        errors += 1

                # Execute batch insert if we have data
                if batch_data:
                    insert_sql = '''
                        INSERT INTO strategies
                        (strategy_name, strategy_category, params, total_trades, win_rate,
                         profit_factor, total_pnl, max_drawdown, equity_r_squared, recovery_factor,
                         sharpe_ratio, composite_score, tp_percent, sl_percent,
                         val_pnl, val_profit_factor, val_win_rate, found_by, data_source, symbol,
                         timeframe, data_start, data_end, optimization_run_id, equity_curve,
                         trade_mode, long_trades, long_wins, long_pnl, short_trades, short_wins, short_pnl, flip_count,
                         total_pnl_percent, avg_trade, buy_hold_return, vs_buy_hold, consistency_score, trades_list)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    '''

                    cursor.executemany(insert_sql, batch_data)
                    saved = len(batch_data)

                # Single commit for all inserts
                conn.commit()

                log(f"Batch saved {saved} strategies ({skipped} duplicates skipped, {errors} errors)")

            except Exception as e:
                conn.rollback()
                log(f"Batch insert failed, rolling back: {e}", level='ERROR')
                raise

        return {'saved': saved, 'skipped': skipped, 'errors': errors}

    def get_top_strategies(self, limit: int = 10, symbol: str = None,
                           timeframe: str = None, min_trades: int = 3,
                           min_win_rate: float = 0.0) -> List[Dict]:
        """Get top strategies by composite score."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = '''
            SELECT * FROM strategies
            WHERE total_trades >= %s AND win_rate >= %s
        '''
        params = [min_trades, min_win_rate]

        if symbol:
            query += ' AND symbol = %s'
            params.append(symbol)

        if timeframe:
            query += ' AND timeframe = %s'
            params.append(timeframe)

        query += ' ORDER BY composite_score DESC LIMIT %s'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_best_by_win_rate(self, limit: int = 10, min_trades: int = 5) -> List[Dict]:
        """Get strategies with highest win rate."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT * FROM strategies
            WHERE total_trades >= %s
            ORDER BY win_rate DESC
            LIMIT %s
        ''', (min_trades, limit))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_best_by_profit_factor(self, limit: int = 10, min_trades: int = 5) -> List[Dict]:
        """Get strategies with highest profit factor."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT * FROM strategies
            WHERE total_trades >= %s AND profit_factor > 0
            ORDER BY profit_factor DESC
            LIMIT %s
        ''', (min_trades, limit))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def search_strategies(self, strategy_name: str = None, category: str = None,
                          min_win_rate: float = None, min_pnl: float = None,
                          symbol: str = None, timeframe: str = None) -> List[Dict]:
        """Search strategies with various filters."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = 'SELECT * FROM strategies WHERE 1=1'
        params = []

        if strategy_name:
            query += ' AND strategy_name LIKE %s'
            params.append(f'%{strategy_name}%')

        if category:
            query += ' AND strategy_category LIKE %s'
            params.append(f'%{category}%')

        if min_win_rate is not None:
            query += ' AND win_rate >= %s'
            params.append(min_win_rate)

        if min_pnl is not None:
            query += ' AND total_pnl >= %s'
            params.append(min_pnl)

        if symbol:
            query += ' AND symbol = %s'
            params.append(symbol)

        if timeframe:
            query += ' AND timeframe = %s'
            params.append(timeframe)

        query += ' ORDER BY composite_score DESC LIMIT 100'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict]:
        """Get a single strategy by ID."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT * FROM strategies WHERE id = %s', (strategy_id,))
        row = cursor.fetchone()
        self._return_connection(conn)

        return self._row_to_dict(row) if row else None

    def get_optimization_run_by_id(self, run_id: int) -> Optional[Dict]:
        """Get optimization run by ID."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM optimization_runs WHERE id = %s', (run_id,))
        row = cursor.fetchone()
        self._return_connection(conn)
        return dict(row) if row else None

    def get_optimization_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent optimization runs."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT * FROM optimization_runs
            ORDER BY started_at DESC
            LIMIT %s
        ''', (limit,))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get overall database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM strategies')
        total_strategies = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM optimization_runs')
        total_runs = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM strategies WHERE total_pnl > 0')
        profitable_strategies = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(win_rate) FROM strategies WHERE total_trades >= 5')
        avg_win_rate = cursor.fetchone()[0] or 0

        cursor.execute('SELECT MAX(composite_score) FROM strategies')
        best_score = cursor.fetchone()[0] or 0

        cursor.execute('SELECT DISTINCT symbol FROM strategies WHERE symbol IS NOT NULL')
        symbols = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT timeframe FROM strategies WHERE timeframe IS NOT NULL')
        timeframes = [row[0] for row in cursor.fetchall()]

        # Elite validation counts
        cursor.execute("SELECT COUNT(*) FROM strategies WHERE elite_status = 'validated'")
        elite_validated = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies WHERE elite_status IS NULL OR elite_status = 'pending'")
        elite_pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies WHERE elite_status = 'untestable'")
        elite_untestable = cursor.fetchone()[0]

        self._return_connection(conn)

        return {
            'total_strategies': total_strategies,
            'total_optimization_runs': total_runs,
            'profitable_strategies': profitable_strategies,
            'avg_win_rate': round(avg_win_rate, 2),
            'best_composite_score': round(best_score, 4),
            'symbols_tested': symbols,
            'timeframes_tested': timeframes,
            'elite_validated': elite_validated,
            'elite_pending': elite_pending,
            'elite_untestable': elite_untestable
        }

    def get_filter_options(self) -> Dict:
        """Get distinct symbols, timeframes, periods, and date range for filter dropdowns."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT symbol FROM strategies WHERE symbol IS NOT NULL ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT timeframe FROM strategies WHERE timeframe IS NOT NULL ORDER BY timeframe")
        timeframes = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT historical_period FROM strategies WHERE historical_period IS NOT NULL ORDER BY historical_period")
        periods = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM strategies")
        date_row = cursor.fetchone()
        date_range = {
            "min": str(date_row[0]) if date_row[0] else None,
            "max": str(date_row[1]) if date_row[1] else None
        }

        self._return_connection(conn)
        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "periods": periods,
            "date_range": date_range
        }

    def _row_to_dict(self, row: Dict, parse_equity_curve: bool = True) -> Dict:
        """Convert a database row to a dictionary with parsed JSON fields."""
        d = dict(row)

        # Parse JSON fields
        if d.get('params'):
            try:
                d['params'] = json.loads(d['params'])
            except json.JSONDecodeError:
                d['params'] = {}

        if d.get('found_by'):
            try:
                d['found_by'] = json.loads(d['found_by'])
            except json.JSONDecodeError:
                d['found_by'] = []

        if parse_equity_curve and d.get('equity_curve'):
            try:
                d['equity_curve'] = json.loads(d['equity_curve'])
            except json.JSONDecodeError:
                d['equity_curve'] = []
        elif not parse_equity_curve:
            d['equity_curve'] = []

        # Parse trades_list JSON for debugger
        if d.get('trades_list'):
            try:
                d['trades_list'] = json.loads(d['trades_list'])
            except json.JSONDecodeError:
                d['trades_list'] = []
        else:
            d['trades_list'] = []

        # Convert datetime to string if needed
        if d.get('created_at') and hasattr(d['created_at'], 'isoformat'):
            d['created_at'] = d['created_at'].isoformat()

        return d

    def update_elite_status(self, strategy_id: int, elite_status: str,
                             periods_passed: int, periods_total: int,
                             validation_data: str = None,
                             elite_score: float = 0) -> bool:
        """Update elite validation status for a strategy."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE strategies
            SET elite_status = %s,
                elite_validated_at = %s,
                elite_periods_passed = %s,
                elite_periods_total = %s,
                elite_validation_data = %s,
                elite_score = %s
            WHERE id = %s
        ''', (elite_status, datetime.now().isoformat(), periods_passed,
              periods_total, validation_data, elite_score, strategy_id))

        updated = cursor.rowcount > 0
        conn.commit()
        self._return_connection(conn)

        return updated

    def update_trades_list(self, strategy_id: int, trades_list: List[Dict]) -> bool:
        """
        Update trades_list for a strategy (used when regenerating trades for debugger).

        Args:
            strategy_id: Strategy ID
            trades_list: List of trade dictionaries

        Returns:
            True if updated successfully
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            trades_json = json.dumps(trades_list) if trades_list else None
            cursor.execute('''
                UPDATE strategies
                SET trades_list = %s
                WHERE id = %s
            ''', (trades_json, strategy_id))

            updated = cursor.rowcount > 0
            conn.commit()
            return updated
        except Exception as e:
            conn.rollback()
            log(f"[DB] Error updating trades_list for strategy {strategy_id}: {e}", level='ERROR')
            return False
        finally:
            self._return_connection(conn)

    def update_elite_status_batch(self, updates: List[Dict]) -> Dict[str, int]:
        """
        Batch update elite validation status for multiple strategies.

        Uses a single transaction for all updates - much faster than individual calls.

        Args:
            updates: List of dicts with keys:
                - strategy_id: int (required)
                - elite_status: str (required)
                - periods_passed: int (required)
                - periods_total: int (required)
                - validation_data: str (optional)
                - elite_score: float (optional, default 0)

        Returns:
            Dict with 'updated' and 'errors' counts
        """
        if not updates:
            return {'updated': 0, 'errors': 0}

        updated = 0
        errors = 0
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Prepare batch data
                batch_data = []

                for update in updates:
                    try:
                        row_data = (
                            update['elite_status'],
                            now,
                            update['periods_passed'],
                            update['periods_total'],
                            update.get('validation_data'),
                            update.get('elite_score', 0),
                            update['strategy_id']
                        )
                        batch_data.append(row_data)
                    except KeyError as e:
                        log(f"Missing required field in elite status update: {e}", level='ERROR')
                        errors += 1

                # Execute batch update
                if batch_data:
                    update_sql = '''
                        UPDATE strategies
                        SET elite_status = %s,
                            elite_validated_at = %s,
                            elite_periods_passed = %s,
                            elite_periods_total = %s,
                            elite_validation_data = %s,
                            elite_score = %s
                        WHERE id = %s
                    '''

                    cursor.executemany(update_sql, batch_data)
                    updated = len(batch_data)

                conn.commit()
                log(f"Batch updated {updated} elite statuses ({errors} errors)")

            except Exception as e:
                conn.rollback()
                log(f"Batch elite status update failed, rolling back: {e}", level='ERROR')
                raise

        return {'updated': updated, 'errors': errors}

    def delete_strategies_batch(self, strategy_ids: List[int]) -> int:
        """
        Delete multiple strategies in a single transaction.

        Args:
            strategy_ids: List of strategy IDs to delete

        Returns:
            Number of strategies deleted
        """
        if not strategy_ids:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Use ANY() for efficient batch delete
                cursor.execute(
                    'DELETE FROM strategies WHERE id = ANY(%s)',
                    (strategy_ids,)
                )
                deleted = cursor.rowcount
                conn.commit()

                log(f"Batch deleted {deleted} strategies")
                return deleted

            except Exception as e:
                conn.rollback()
                log(f"Batch delete failed, rolling back: {e}", level='ERROR')
                raise

    def record_completed_optimizations_batch(self, completions: List[Dict]) -> int:
        """
        Record multiple completed optimizations in a single transaction.

        Args:
            completions: List of dicts with keys:
                - pair: str
                - period_label: str
                - timeframe_label: str
                - granularity_label: str
                - strategies_found: int (optional, default 0)
                - source: str (optional, default 'binance')
                - duration_seconds: int (optional)

        Returns:
            Number of records upserted
        """
        if not completions:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                batch_data = []
                for comp in completions:
                    batch_data.append((
                        comp['pair'],
                        comp['period_label'],
                        comp['timeframe_label'],
                        comp['granularity_label'],
                        comp.get('strategies_found', 0),
                        comp.get('source', 'binance'),
                        comp.get('duration_seconds')
                    ))

                # Use executemany with ON CONFLICT for upsert
                upsert_sql = '''
                    INSERT INTO completed_optimizations
                    (pair, period_label, timeframe_label, granularity_label, strategies_found, source, duration_seconds, completed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (pair, period_label, timeframe_label, granularity_label)
                    DO UPDATE SET strategies_found = EXCLUDED.strategies_found,
                                  completed_at = CURRENT_TIMESTAMP,
                                  duration_seconds = EXCLUDED.duration_seconds
                '''

                cursor.executemany(upsert_sql, batch_data)
                upserted = len(batch_data)
                conn.commit()

                log(f"Batch recorded {upserted} completed optimizations")
                return upserted

            except Exception as e:
                conn.rollback()
                log(f"Batch completed optimizations record failed, rolling back: {e}", level='ERROR')
                raise

    def get_elite_strategies_optimized(self, top_n_per_market: int = 10) -> List[Dict]:
        """Get top N validated strategies per pair/timeframe."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, timeframe
                        ORDER BY elite_score DESC
                    ) as rank
                FROM strategies
                WHERE elite_status IS NOT NULL
                  AND elite_status != 'pending'
                  AND elite_score > 0
            )
            SELECT * FROM ranked
            WHERE rank <= %s
            ORDER BY elite_score DESC
        ''', (top_n_per_market,))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_strategies_pending_validation(self, limit: int = 100) -> List[Dict]:
        """Get strategies that need elite validation."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT * FROM strategies
            WHERE elite_status IS NULL OR elite_status = 'pending'
            ORDER BY composite_score DESC
            LIMIT %s
        ''', (limit,))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_top_pending_per_market(self, top_n: int = 3) -> List[Dict]:
        """
        Get top N pending strategies per (symbol, timeframe) combination.
        Uses window functions to rank by composite_score within each market.
        This reduces validation queue to only the best candidates per market.
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
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
            WHERE rank <= %s
            ORDER BY composite_score DESC
        ''', (top_n,))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies from the database."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT * FROM strategies ORDER BY id DESC')
        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row) for row in rows]

    def delete_strategy(self, strategy_id: int) -> bool:
        """Delete a strategy by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM strategies WHERE id = %s', (strategy_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        self._return_connection(conn)

        return deleted

    def remove_duplicates(self) -> int:
        """Remove duplicate strategies, keeping the most recent."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM strategies
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM strategies
                GROUP BY strategy_name, symbol, timeframe,
                         ROUND(tp_percent::numeric, 1), ROUND(sl_percent::numeric, 1),
                         total_trades, ROUND(win_rate::numeric, 1),
                         ROUND(total_pnl::numeric, 1), ROUND(profit_factor::numeric, 2)
            )
        ''')

        deleted = cursor.rowcount
        conn.commit()
        self._return_connection(conn)

        log(f"Removed {deleted} duplicate strategies from database")
        return deleted

    def clear_all(self) -> int:
        """Clear entire database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM strategies')
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM strategies')
        cursor.execute('DELETE FROM optimization_runs')
        cursor.execute('DELETE FROM completed_optimizations')
        cursor.execute('DELETE FROM priority_items')
        cursor.execute('DELETE FROM priority_pairs')
        cursor.execute('DELETE FROM priority_periods')
        cursor.execute('DELETE FROM priority_timeframes')
        cursor.execute('DELETE FROM priority_granularities')
        cursor.execute('DELETE FROM priority_settings')

        conn.commit()
        self._return_connection(conn)

        log(f"Cleared entire database: {count} strategies + all tracking data", level='WARNING')
        return count

    # =========================================================================
    # COMPLETED OPTIMIZATIONS TRACKING
    # =========================================================================

    def record_completed_optimization(self, pair: str, period_label: str,
                                       timeframe_label: str, granularity_label: str,
                                       strategies_found: int = 0, source: str = 'binance',
                                       duration_seconds: int = None):
        """Record that an optimization combination has been completed."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO completed_optimizations
            (pair, period_label, timeframe_label, granularity_label, strategies_found, completed_at, source, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
            ON CONFLICT (pair, period_label, timeframe_label, granularity_label)
            DO UPDATE SET strategies_found = EXCLUDED.strategies_found,
                          completed_at = CURRENT_TIMESTAMP,
                          duration_seconds = EXCLUDED.duration_seconds
        ''', (pair, period_label, timeframe_label, granularity_label, strategies_found, source, duration_seconds))

        conn.commit()
        self._return_connection(conn)

    def get_average_duration(self, pair: str = None, period_label: str = None,
                             timeframe_label: str = None, granularity_label: str = None) -> int:
        """Get average duration in seconds for a specific combination or similar combinations.

        Returns average duration, falling back to broader matches if exact match not found:
        1. Exact match (all 4 params)
        2. Same granularity and timeframe (most similar workload)
        3. Same granularity only
        4. Global average
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Try exact match first
        if all([pair, period_label, timeframe_label, granularity_label]):
            cursor.execute('''
                SELECT AVG(duration_seconds) FROM completed_optimizations
                WHERE pair = %s AND period_label = %s AND timeframe_label = %s
                AND granularity_label = %s AND duration_seconds IS NOT NULL
            ''', (pair, period_label, timeframe_label, granularity_label))
            result = cursor.fetchone()[0]
            if result:
                self._return_connection(conn)
                return int(result)

        # Fallback: same granularity and timeframe (similar workload)
        if granularity_label and timeframe_label:
            cursor.execute('''
                SELECT AVG(duration_seconds) FROM completed_optimizations
                WHERE granularity_label = %s AND timeframe_label = %s
                AND duration_seconds IS NOT NULL
            ''', (granularity_label, timeframe_label))
            result = cursor.fetchone()[0]
            if result:
                self._return_connection(conn)
                return int(result)

        # Fallback: same granularity only (n_trials is main factor)
        if granularity_label:
            cursor.execute('''
                SELECT AVG(duration_seconds) FROM completed_optimizations
                WHERE granularity_label = %s AND duration_seconds IS NOT NULL
            ''', (granularity_label,))
            result = cursor.fetchone()[0]
            if result:
                self._return_connection(conn)
                return int(result)

        # Final fallback: global average
        cursor.execute('''
            SELECT AVG(duration_seconds) FROM completed_optimizations
            WHERE duration_seconds IS NOT NULL
        ''')
        result = cursor.fetchone()[0]
        self._return_connection(conn)
        return int(result) if result else None

    def is_optimization_completed(self, pair: str, period_label: str,
                                   timeframe_label: str, granularity_label: str) -> bool:
        """Check if a specific combination has been completed."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 1 FROM completed_optimizations
            WHERE pair = %s AND period_label = %s AND timeframe_label = %s AND granularity_label = %s
        ''', (pair, period_label, timeframe_label, granularity_label))

        result = cursor.fetchone() is not None
        self._return_connection(conn)
        return result

    def get_completed_optimizations(self, with_timestamps: bool = False) -> dict:
        """Get completed optimization combinations."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT pair, period_label, timeframe_label, granularity_label, completed_at
            FROM completed_optimizations
        ''')

        if with_timestamps:
            completed = {
                (row[0], row[1], row[2], row[3]): str(row[4]) if row[4] else None
                for row in cursor.fetchall()
            }
        else:
            completed = {(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()}

        self._return_connection(conn)
        return completed

    def get_completed_optimizations_count(self) -> int:
        """Get count of completed optimizations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM completed_optimizations')
        count = cursor.fetchone()[0]
        self._return_connection(conn)
        return count

    def get_finest_completed_granularity(self, pair: str, period_label: str,
                                          timeframe_label: str) -> int:
        """
        Get the finest (highest n_trials) granularity already completed for a combo.

        Returns the highest n_trials value among completed granularities for this
        (pair, period, timeframe), or 0 if none completed.

        This enables smart deduplication: skip coarser granularities if a finer
        one has already been run (since finer includes all coarser values).
        """
        # Granularity label -> n_trials mapping
        GRANULARITY_TRIALS = {
            "0.1%": 10000,
            "0.2%": 2500,
            "0.5%": 400,
            "0.7%": 200,
            "1.0%": 100,
        }

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT granularity_label FROM completed_optimizations
            WHERE pair = %s AND period_label = %s AND timeframe_label = %s
        ''', (pair, period_label, timeframe_label))

        completed_granularities = [row[0] for row in cursor.fetchall()]
        self._return_connection(conn)

        if not completed_granularities:
            return 0

        # Find the highest n_trials among completed
        max_trials = 0
        for gran_label in completed_granularities:
            trials = GRANULARITY_TRIALS.get(gran_label, 0)
            if trials > max_trials:
                max_trials = trials

        return max_trials

    def clear_completed_optimizations(self, pair: str = None, granularity_label: str = None):
        """Clear completed optimization records."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if pair and granularity_label:
            cursor.execute('DELETE FROM completed_optimizations WHERE pair = %s AND granularity_label = %s',
                          (pair, granularity_label))
        elif pair:
            cursor.execute('DELETE FROM completed_optimizations WHERE pair = %s', (pair,))
        elif granularity_label:
            cursor.execute('DELETE FROM completed_optimizations WHERE granularity_label = %s', (granularity_label,))
        else:
            cursor.execute('DELETE FROM completed_optimizations')

        deleted = cursor.rowcount
        conn.commit()
        self._return_connection(conn)
        return deleted

    # =========================================================================
    # PRIORITY QUEUE MANAGEMENT
    # =========================================================================

    def _init_priority_table(self):
        """Create priority tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_items (
                id SERIAL PRIMARY KEY,
                position INTEGER NOT NULL,
                pair TEXT NOT NULL,
                period_label TEXT NOT NULL,
                period_months REAL NOT NULL,
                timeframe_label TEXT NOT NULL,
                timeframe_minutes INTEGER NOT NULL,
                granularity_label TEXT NOT NULL,
                granularity_trials INTEGER NOT NULL,
                source TEXT DEFAULT 'binance',
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pair, period_label, timeframe_label, granularity_label)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority_position ON priority_items(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority_enabled ON priority_items(enabled)')

        self._init_priority_lists_tables(cursor)

        conn.commit()
        self._return_connection(conn)

    def _init_priority_lists_tables(self, cursor):
        """Create separate priority tables for pairs, periods, and timeframes."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_pairs (
                id SERIAL PRIMARY KEY,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_periods (
                id SERIAL PRIMARY KEY,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                months REAL NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_timeframes (
                id SERIAL PRIMARY KEY,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_granularities (
                id SERIAL PRIMARY KEY,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                n_trials INTEGER NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pairs_position ON priority_pairs(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_periods_position ON priority_periods(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframes_position ON priority_timeframes(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_granularities_position ON priority_granularities(position)')

    def get_priority_list(self) -> List[Dict]:
        """Get all priority items ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT * FROM priority_items
            ORDER BY position ASC
        ''')

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [dict(row) for row in rows]

    def add_priority_item(self, pair: str, period_label: str, period_months: float,
                          timeframe_label: str, timeframe_minutes: int,
                          granularity_label: str, granularity_trials: int,
                          source: str = 'binance') -> Optional[int]:
        """Add a new priority item at the end of the list."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COALESCE(MAX(position), 0) + 1 FROM priority_items')
        next_position = cursor.fetchone()[0]

        try:
            cursor.execute('''
                INSERT INTO priority_items
                (position, pair, period_label, period_months, timeframe_label,
                 timeframe_minutes, granularity_label, granularity_trials, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (next_position, pair, period_label, period_months,
                  timeframe_label, timeframe_minutes, granularity_label,
                  granularity_trials, source))

            item_id = cursor.fetchone()[0]
            conn.commit()
            return item_id
        except psycopg2.IntegrityError:
            conn.rollback()
            return None
        finally:
            self._return_connection(conn)

    def delete_priority_item(self, item_id: int) -> bool:
        """Delete a priority item and reorder remaining items."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM priority_items WHERE id = %s', (item_id,))
        deleted = cursor.rowcount > 0

        if deleted:
            cursor.execute('''
                WITH numbered AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY position) as new_pos
                    FROM priority_items
                )
                UPDATE priority_items
                SET position = numbered.new_pos
                FROM numbered
                WHERE priority_items.id = numbered.id
            ''')

        conn.commit()
        self._return_connection(conn)
        return deleted

    def reorder_priority_items(self, id_order: List[int]) -> bool:
        """Update positions based on new order of IDs."""
        conn = self._get_connection()
        cursor = conn.cursor()

        for position, item_id in enumerate(id_order, start=1):
            cursor.execute('''
                UPDATE priority_items
                SET position = %s, updated_at = %s
                WHERE id = %s
            ''', (position, datetime.now().isoformat(), item_id))

        conn.commit()
        self._return_connection(conn)
        return True

    def toggle_priority_item(self, item_id: int) -> Optional[bool]:
        """Toggle enabled status. Returns new status or None if not found."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT enabled FROM priority_items WHERE id = %s', (item_id,))
        row = cursor.fetchone()
        if not row:
            self._return_connection(conn)
            return None

        new_status = 0 if row[0] else 1
        cursor.execute('UPDATE priority_items SET enabled = %s, updated_at = %s WHERE id = %s',
                       (new_status, datetime.now().isoformat(), item_id))

        conn.commit()
        self._return_connection(conn)
        return bool(new_status)

    def clear_priority_items(self) -> int:
        """Clear all priority items."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM priority_items')
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM priority_items')

        conn.commit()
        self._return_connection(conn)
        return count

    def get_enabled_priority_combinations(self) -> List[Dict]:
        """Get enabled priority items formatted for optimization."""
        items = self.get_priority_list()
        return [
            {
                'source': item['source'],
                'pair': item['pair'],
                'period': {
                    'label': item['period_label'],
                    'months': item['period_months']
                },
                'timeframe': {
                    'label': item['timeframe_label'],
                    'minutes': item['timeframe_minutes']
                },
                'granularity': {
                    'label': item['granularity_label'],
                    'n_trials': item['granularity_trials']
                }
            }
            for item in items if item['enabled']
        ]

    # =========================================================================
    # NEW 3-LIST PRIORITY SYSTEM
    # =========================================================================

    def get_all_priority_lists(self) -> Dict:
        """Get all priority lists in a single database connection."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT COUNT(*) FROM priority_pairs')
        pairs_count = cursor.fetchone()['count']
        populated = pairs_count > 0

        # Select only needed columns (exclude created_at which isn't JSON serializable)
        cursor.execute('SELECT id, position, value, label, enabled FROM priority_pairs ORDER BY position ASC')
        pairs = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT id, position, value, label, months, enabled FROM priority_periods ORDER BY position ASC')
        periods = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT id, position, value, label, minutes, enabled FROM priority_timeframes ORDER BY position ASC')
        timeframes = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT id, position, value, label, n_trials, enabled FROM priority_granularities ORDER BY position ASC')
        granularities = [dict(row) for row in cursor.fetchall()]

        self._return_connection(conn)

        return {
            'pairs': pairs,
            'periods': periods,
            'timeframes': timeframes,
            'granularities': granularities,
            'populated': populated
        }

    def get_priority_pairs(self) -> List[Dict]:
        """Get all trading pairs ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_pairs ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_priority_periods(self) -> List[Dict]:
        """Get all historical periods ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_periods ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_priority_timeframes(self) -> List[Dict]:
        """Get all timeframes ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_timeframes ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_enabled_priority_pairs(self) -> List[Dict]:
        """Get enabled trading pairs ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_pairs WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_enabled_priority_periods(self) -> List[Dict]:
        """Get enabled historical periods ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_periods WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_enabled_priority_timeframes(self) -> List[Dict]:
        """Get enabled timeframes ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_timeframes WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_priority_granularities(self) -> List[Dict]:
        """Get all granularities ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_granularities ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_enabled_priority_granularities(self) -> List[Dict]:
        """Get enabled granularities ordered by position."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM priority_granularities WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        self._return_connection(conn)
        return [dict(row) for row in rows]

    def get_priority_setting(self, key: str) -> Optional[str]:
        """Get a priority setting value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM priority_settings WHERE key = %s', (key,))
        row = cursor.fetchone()
        self._return_connection(conn)
        return row[0] if row else None

    def set_priority_setting(self, key: str, value: str):
        """Set a priority setting value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO priority_settings (key, value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
        ''', (key, value, datetime.now().isoformat()))
        conn.commit()
        self._return_connection(conn)

    def reorder_priority_list_new(self, list_type: str, id_order: List[int]) -> bool:
        """Reorder items in a specific priority list."""
        table_map = {
            'pairs': 'priority_pairs',
            'periods': 'priority_periods',
            'timeframes': 'priority_timeframes',
            'granularities': 'priority_granularities'
        }
        table = table_map.get(list_type)
        if not table:
            return False

        conn = self._get_connection()
        cursor = conn.cursor()

        for position, item_id in enumerate(id_order, start=1):
            cursor.execute(f'UPDATE {table} SET position = %s WHERE id = %s', (position, item_id))

        conn.commit()
        self._return_connection(conn)
        return True

    def toggle_priority_list_item(self, list_type: str, item_id: int) -> Optional[bool]:
        """Toggle enabled status in a specific list."""
        table_map = {
            'pairs': 'priority_pairs',
            'periods': 'priority_periods',
            'timeframes': 'priority_timeframes',
            'granularities': 'priority_granularities'
        }
        table = table_map.get(list_type)
        if not table:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f'SELECT enabled FROM {table} WHERE id = %s', (item_id,))
        row = cursor.fetchone()
        if not row:
            self._return_connection(conn)
            return None

        new_status = 0 if row[0] else 1
        cursor.execute(f'UPDATE {table} SET enabled = %s WHERE id = %s', (new_status, item_id))

        conn.commit()
        self._return_connection(conn)
        return bool(new_status)

    def reset_priority_pairs(self, pairs: List[str]):
        """Reset trading pairs to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_pairs')

        for pos, pair in enumerate(pairs, start=1):
            cursor.execute('''
                INSERT INTO priority_pairs (position, value, label, enabled)
                VALUES (%s, %s, %s, 1)
            ''', (pos, pair, pair))

        conn.commit()
        self._return_connection(conn)

    def reset_priority_periods(self, periods: List[Dict]):
        """Reset historical periods to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_periods')

        for pos, period in enumerate(periods, start=1):
            cursor.execute('''
                INSERT INTO priority_periods (position, value, label, months, enabled)
                VALUES (%s, %s, %s, %s, 1)
            ''', (pos, period['label'], period['label'], period['months']))

        conn.commit()
        self._return_connection(conn)

    def reset_priority_timeframes(self, timeframes: List[Dict]):
        """Reset timeframes to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_timeframes')

        for pos, tf in enumerate(timeframes, start=1):
            cursor.execute('''
                INSERT INTO priority_timeframes (position, value, label, minutes, enabled)
                VALUES (%s, %s, %s, %s, 1)
            ''', (pos, tf['label'], tf['label'], tf['minutes']))

        conn.commit()
        self._return_connection(conn)

    def reset_priority_granularities(self, granularities: List[Dict]):
        """Reset granularities to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_granularities')

        for pos, gran in enumerate(granularities, start=1):
            cursor.execute('''
                INSERT INTO priority_granularities (position, value, label, n_trials, enabled)
                VALUES (%s, %s, %s, %s, 1)
            ''', (pos, gran['label'], gran['label'], gran['n_trials']))

        conn.commit()
        self._return_connection(conn)

    def enable_all_priority_items(self):
        """Enable all items in all priority lists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE priority_pairs SET enabled = 1')
        cursor.execute('UPDATE priority_periods SET enabled = 1')
        cursor.execute('UPDATE priority_timeframes SET enabled = 1')
        cursor.execute('UPDATE priority_granularities SET enabled = 1')
        conn.commit()
        self._return_connection(conn)

    def disable_all_priority_items(self):
        """Disable all items in all priority lists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE priority_pairs SET enabled = 0')
        cursor.execute('UPDATE priority_periods SET enabled = 0')
        cursor.execute('UPDATE priority_timeframes SET enabled = 0')
        cursor.execute('UPDATE priority_granularities SET enabled = 0')
        conn.commit()
        self._return_connection(conn)

    def has_priority_lists_populated(self) -> bool:
        """Check if the new priority lists have any data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM priority_pairs')
        count = cursor.fetchone()[0]
        self._return_connection(conn)
        return count > 0

    # =========================================================================
    # OPTIMIZED QUERIES FOR PERFORMANCE
    # =========================================================================

    def get_elite_counts(self) -> Dict[str, int]:
        """Get elite status counts using SQL aggregation."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COALESCE(elite_status, 'pending') as status,
                COUNT(*) as count
            FROM strategies
            GROUP BY COALESCE(elite_status, 'pending')
        ''')

        counts = {
            'elite': 0,
            'partial': 0,
            'failed': 0,
            'pending': 0
        }

        for row in cursor.fetchall():
            status, count = row
            if status in counts:
                counts[status] = count
            elif status is None or status == 'pending':
                counts['pending'] += count

        self._return_connection(conn)
        return counts

    def get_db_stats_optimized(self) -> Dict[str, Any]:
        """Get database statistics using SQL aggregation."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total_strategies,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT timeframe) as unique_timeframes,
                SUM(CASE WHEN elite_status IN ('validated', 'elite', 'partial', 'failed') THEN 1 ELSE 0 END) as validated_count
            FROM strategies
        ''')

        row = cursor.fetchone()
        self._return_connection(conn)

        return {
            'total_strategies': row[0] or 0,
            'unique_symbols': row[1] or 0,
            'unique_timeframes': row[2] or 0,
            'elite_count': row[3] or 0
        }

    def get_strategies_paginated(self, limit: int = 500, offset: int = 0,
                                  symbol: str = None, timeframe: str = None,
                                  sort_by: str = 'id', sort_order: str = 'DESC') -> Dict:
        """Get strategies with pagination support."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        where_clauses = []
        params = []

        if symbol:
            where_clauses.append('symbol = %s')
            params.append(symbol)

        if timeframe:
            where_clauses.append('timeframe = %s')
            params.append(timeframe)

        where_sql = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''

        valid_sort_columns = {'id', 'composite_score', 'win_rate', 'profit_factor',
                              'total_pnl', 'created_at', 'elite_score'}
        if sort_by not in valid_sort_columns:
            sort_by = 'id'

        sort_direction = 'DESC' if sort_order.upper() == 'DESC' else 'ASC'

        # Get total count
        cursor.execute(f'SELECT COUNT(*) FROM strategies{where_sql}', params)
        total = cursor.fetchone()['count']

        # Get paginated results
        query = f'''
            SELECT * FROM strategies{where_sql}
            ORDER BY {sort_by} {sort_direction}
            LIMIT %s OFFSET %s
        '''
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()
        self._return_connection(conn)

        strategies = [self._row_to_dict(row) for row in rows]

        return {
            'strategies': strategies,
            'total': total,
            'has_more': offset + len(strategies) < total,
            'limit': limit,
            'offset': offset
        }

    def get_total_strategy_count(self) -> int:
        """Get total count of strategies."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM strategies')
        count = cursor.fetchone()[0]
        self._return_connection(conn)
        return count

    def get_elite_strategies_filtered(self, status_filter: str = None,
                                       limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get elite strategies with SQL-level filtering."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if status_filter:
            cursor.execute('''
                SELECT * FROM strategies
                WHERE elite_status = %s
                ORDER BY elite_score DESC
                LIMIT %s OFFSET %s
            ''', (status_filter, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM strategies
                WHERE elite_status IS NOT NULL
                ORDER BY elite_score DESC
                LIMIT %s OFFSET %s
            ''', (limit, offset))

        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row, parse_equity_curve=False) for row in rows]

    def get_elite_leaderboard(self, limit: int = 20) -> Dict:
        """Get top validated strategies by score."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Include both new 'validated' status and legacy 'elite'/'partial'/'failed' statuses
        # All validated strategies sorted by score - the score does the ranking
        cursor.execute('''
            SELECT * FROM strategies
            WHERE elite_status IN ('validated', 'elite', 'partial', 'failed')
            ORDER BY elite_score DESC
            LIMIT %s
        ''', (limit,))
        rows = cursor.fetchall()

        cursor.execute("""
            SELECT COUNT(*) FROM strategies
            WHERE elite_status IN ('validated', 'elite', 'partial', 'failed')
        """)
        total_validated = cursor.fetchone()['count']

        self._return_connection(conn)

        return {
            "leaderboard": [self._row_to_dict(row, parse_equity_curve=False) for row in rows],
            "total_validated": total_validated
        }

    def get_elite_stats_optimized(self) -> Dict:
        """Get elite validation statistics using SQL aggregation."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COALESCE(elite_status, 'pending') as status,
                COUNT(*) as count
            FROM strategies
            GROUP BY COALESCE(elite_status, 'pending')
        ''')

        # Simplified status model: validated, pending, untestable, skipped
        status_counts = {
            'validated': 0, 'pending': 0, 'untestable': 0, 'skipped': 0
        }

        for row in cursor.fetchall():
            status, count = row
            if status in status_counts:
                status_counts[status] = count
            elif status in ['elite', 'partial', 'failed']:
                # Legacy status - count as validated
                status_counts['validated'] += count
            else:
                # NULL or unknown - count as pending
                status_counts['pending'] += count

        cursor.execute('SELECT COUNT(*) FROM strategies')
        total = cursor.fetchone()[0]

        cursor.execute('''
            SELECT AVG(COALESCE(elite_score, 0))
            FROM strategies
            WHERE elite_status = 'validated'
        ''')
        avg_score = cursor.fetchone()[0] or 0

        self._return_connection(conn)

        return {
            "total_strategies": total,
            "status_counts": status_counts,
            "avg_elite_score": round(avg_score, 2)
        }

    def get_top_strategies_per_market(self, top_n: int = 10, symbol: str = None,
                                       timeframe: str = None, min_win_rate: float = 0.0,
                                       total_limit: int = 500) -> List[Dict]:
        """Get top N strategies per (symbol, timeframe) pair."""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        where_parts = ['win_rate >= %s']
        params = [min_win_rate]

        if symbol:
            where_parts.append('symbol = %s')
            params.append(symbol)

        if timeframe:
            where_parts.append('timeframe = %s')
            params.append(timeframe)

        where_clause = ' AND '.join(where_parts)

        query = f'''
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, timeframe
                        ORDER BY composite_score DESC
                    ) as rank
                FROM strategies
                WHERE {where_clause}
            )
            SELECT * FROM ranked
            WHERE rank <= %s
            ORDER BY composite_score DESC
            LIMIT %s
        '''

        params.extend([top_n, total_limit])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._return_connection(conn)

        return [self._row_to_dict(row, parse_equity_curve=False) for row in rows]

    def get_pending_validation_count(self) -> int:
        """Get count of strategies pending validation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM strategies
            WHERE elite_status IS NULL OR elite_status = 'pending'
        ''')
        count = cursor.fetchone()[0]
        self._return_connection(conn)
        return count


# Singleton instance for easy access
_db_instance: Optional[StrategyDatabase] = None


def get_strategy_db(database_url: str = None) -> StrategyDatabase:
    """Get or create the strategy database singleton."""
    global _db_instance

    if database_url is None:
        database_url = os.environ.get('DATABASE_URL')

    if _db_instance is None:
        _db_instance = StrategyDatabase(database_url)
    return _db_instance
