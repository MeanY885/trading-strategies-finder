"""
STRATEGY DATABASE
==================
SQLite persistence layer for storing winning strategies across sessions.
Allows loading historical results and comparing performance over time.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class StrategyDatabase:
    """
    SQLite database for persisting strategy results.

    Schema:
    - strategies: Core strategy results with metrics
    - strategy_trades: Individual trade records
    - optimization_runs: Track optimization sessions
    """

    def __init__(self, db_path: str = "data/strategies.db"):
        self.db_path = db_path

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._init_database()
        self._init_priority_table()

        # Auto-clean duplicates on startup
        self._auto_deduplicate()

    def _get_connection(self):
        """
        Get a database connection with proper settings for concurrent access.
        Uses WAL mode for better read/write concurrency.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=30000")  # Wait up to 30s for lock
        return conn

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Enable WAL mode for better concurrent read/write performance
        # WAL allows multiple readers while one writer is active
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set busy timeout to 30 seconds - wait for lock instead of failing immediately
        cursor.execute("PRAGMA busy_timeout=30000")
        # Synchronous=NORMAL is safe with WAL and faster than FULL
        cursor.execute("PRAGMA synchronous=NORMAL")

        # Main strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                optimization_run_id INTEGER,

                -- Equity curve (JSON)
                equity_curve TEXT
            )
        ''')

        # Optimization runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
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

        # Completed optimizations tracking table - for accurate resume functionality
        # Tracks exactly which (pair, period, timeframe, granularity) combinations have been completed
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                period_label TEXT NOT NULL,
                timeframe_label TEXT NOT NULL,
                granularity_label TEXT NOT NULL,
                strategies_found INTEGER DEFAULT 0,
                completed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'binance',
                UNIQUE(pair, period_label, timeframe_label, granularity_label)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_completed_combo ON completed_optimizations(pair, period_label, timeframe_label, granularity_label)')

        # Create indexes for fast querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_win_rate ON strategies(win_rate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_profit_factor ON strategies(profit_factor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_composite_score ON strategies(composite_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON strategies(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON strategies(timeframe)')
        # Elite validation indexes for fast filtering
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_status ON strategies(elite_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_score ON strategies(elite_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_combo ON strategies(symbol, timeframe, elite_status, elite_score)')
        # Performance indexes for common query patterns
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elite_status_score ON strategies(elite_status, elite_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON strategies(symbol, timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON strategies(created_at DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_validation ON strategies(elite_status, composite_score DESC)')

        # Migration: Add missing columns to existing databases
        cursor.execute("PRAGMA table_info(strategies)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        migration_columns = [
            ('indicator_params', 'TEXT'),
            ('tuning_improved', 'INTEGER DEFAULT 0'),
            ('tuning_score_before', 'REAL'),
            ('tuning_score_after', 'REAL'),
            ('tuning_improvement_pct', 'REAL'),
            # Bidirectional trading fields
            ('trade_mode', "TEXT DEFAULT 'single'"),  # 'long', 'short', 'bidirectional'
            ('long_trades', 'INTEGER DEFAULT 0'),
            ('long_wins', 'INTEGER DEFAULT 0'),
            ('long_pnl', 'REAL DEFAULT 0'),
            ('short_trades', 'INTEGER DEFAULT 0'),
            ('short_wins', 'INTEGER DEFAULT 0'),
            ('short_pnl', 'REAL DEFAULT 0'),
            ('flip_count', 'INTEGER DEFAULT 0'),
            # Elite strategy validation fields
            ('elite_status', "TEXT DEFAULT 'pending'"),  # 'pending', 'elite', 'failed', 'partial'
            ('elite_validated_at', 'TEXT'),              # Last validation timestamp
            ('elite_periods_passed', 'INTEGER DEFAULT 0'),  # Count of periods that passed
            ('elite_periods_total', 'INTEGER DEFAULT 0'),   # Total periods tested
            ('elite_validation_data', 'TEXT'),           # JSON with per-period results
            ('elite_score', 'REAL DEFAULT 0'),           # Consistency score 0-100

            # Dual Pool Architecture fields (TP/SL vs Indicator Exit)
            ('pool', "TEXT DEFAULT 'tp_sl'"),            # 'tp_sl' or 'indicator_exit'
            ('exit_type', "TEXT DEFAULT 'fixed_tp_sl'"), # 'fixed_tp_sl', 'trailing_stop', 'indicator_exit'
            ('exit_indicator', 'TEXT'),                  # 'supertrend', 'ema_cross', 'psar', 'mcginley', etc.
            ('trailing_atr_mult', 'REAL'),               # ATR multiplier for trailing stops
            ('strategy_style', "TEXT DEFAULT 'unknown'"), # 'trend_following', 'mean_reversion', 'breakout'

            # Trend-following metrics
            ('avg_trade_duration_hours', 'REAL DEFAULT 0'),  # Average holding period
            ('mfe_capture_ratio', 'REAL DEFAULT 0'),         # How much of max favorable excursion was captured
            ('avg_winner_pct', 'REAL DEFAULT 0'),            # Average winning trade %
            ('avg_loser_pct', 'REAL DEFAULT 0'),             # Average losing trade %
            ('risk_reward_ratio', 'REAL DEFAULT 0'),         # avg_winner / avg_loser
            ('trend_following_score', 'REAL DEFAULT 0'),     # Score using trend-following weights
        ]

        for col_name, col_type in migration_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f'ALTER TABLE strategies ADD COLUMN {col_name} {col_type}')
                    print(f"Added missing column: {col_name}")
                except Exception as e:
                    print(f"Column {col_name} migration skipped: {e}")

        conn.commit()
        conn.close()

        print(f"Strategy database initialized: {self.db_path}")

    def _auto_deduplicate(self):
        """Automatically remove duplicates on startup."""
        removed = self.remove_duplicates()
        if removed > 0:
            print(f"Auto-cleaned {removed} duplicate strategies")

    def start_optimization_run(self, symbol: str = None, timeframe: str = None,
                               data_source: str = None, data_rows: int = 0,
                               capital: float = 1000.0, risk_percent: float = 2.0) -> int:
        """Start a new optimization run and return its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO optimization_runs
            (symbol, timeframe, data_source, data_rows, capital, risk_percent, status)
            VALUES (?, ?, ?, ?, ?, ?, 'running')
        ''', (symbol, timeframe, data_source, data_rows, capital, risk_percent))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"Started optimization run #{run_id}")
        return run_id

    def complete_optimization_run(self, run_id: int, strategies_tested: int,
                                  profitable_found: int):
        """Mark an optimization run as complete."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE optimization_runs
            SET completed_at = ?, strategies_tested = ?, profitable_found = ?, status = 'completed'
            WHERE id = ?
        ''', (datetime.now().isoformat(), strategies_tested, profitable_found, run_id))

        conn.commit()
        conn.close()

        print(f"Completed optimization run #{run_id}: {profitable_found} profitable strategies")

    def save_strategy(self, result: Any, run_id: int = None,
                      symbol: str = None, timeframe: str = None,
                      data_source: str = None, data_start: str = None,
                      data_end: str = None,
                      indicator_params: Dict = None,
                      tuning_info: Dict = None) -> Optional[int]:
        """
        Save a strategy result to the database.

        Args:
            result: StrategyResult dataclass
            run_id: Optimization run ID
            symbol: Trading pair (e.g., BTCGBP)
            timeframe: Candle timeframe (e.g., 15m)
            data_source: Data source (e.g., Kraken)
            data_start: Start date of data
            data_end: End date of data
            indicator_params: Dict of tuned indicator parameters (Phase 2)
            tuning_info: Dict with tuning results (improved, before_score, after_score, improvement_pct)

        Returns:
            Strategy ID in database, or None if duplicate
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract params
        params = result.params if hasattr(result, 'params') else {}
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

        # Check for duplicate - same strategy with same key metrics
        strategy_name = getattr(result, 'strategy_name', 'unknown')
        total_trades = getattr(result, 'total_trades', 0)
        win_rate = getattr(result, 'win_rate', 0)
        total_pnl = getattr(result, 'total_pnl', 0)
        profit_factor = getattr(result, 'profit_factor', 0)

        cursor.execute('''
            SELECT id FROM strategies
            WHERE strategy_name = ?
              AND symbol = ?
              AND timeframe = ?
              AND ABS(tp_percent - ?) < 0.01
              AND ABS(sl_percent - ?) < 0.01
              AND total_trades = ?
              AND ABS(win_rate - ?) < 0.01
              AND ABS(total_pnl - ?) < 0.01
              AND ABS(profit_factor - ?) < 0.01
            LIMIT 1
        ''', (strategy_name, symbol, timeframe, tp_percent, sl_percent,
              total_trades, win_rate, total_pnl, profit_factor))

        existing = cursor.fetchone()
        if existing:
            conn.close()
            print(f"Skipping duplicate strategy: {strategy_name} (TP={tp_percent}%, SL={sl_percent}%)")
            return None  # Return None to indicate duplicate was skipped

        # Convert found_by list to JSON
        found_by = json.dumps(result.found_by) if hasattr(result, 'found_by') else '[]'

        # Convert equity curve to JSON
        equity_curve = json.dumps(result.equity_curve) if hasattr(result, 'equity_curve') else '[]'

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

        # Use getattr with defaults for compatibility with both old and new result formats
        cursor.execute('''
            INSERT INTO strategies
            (strategy_name, strategy_category, params, total_trades, win_rate,
             profit_factor, total_pnl, max_drawdown, equity_r_squared, recovery_factor,
             sharpe_ratio, composite_score, tp_percent, sl_percent,
             indicator_params, tuning_improved, tuning_score_before, tuning_score_after, tuning_improvement_pct,
             val_pnl, val_profit_factor, val_win_rate, found_by, data_source, symbol,
             timeframe, data_start, data_end, optimization_run_id, equity_curve,
             trade_mode, long_trades, long_wins, long_pnl, short_trades, short_wins, short_pnl, flip_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            # Bidirectional fields
            trade_mode,
            getattr(result, 'long_trades', 0),
            getattr(result, 'long_wins', 0),
            getattr(result, 'long_pnl', 0.0),
            getattr(result, 'short_trades', 0),
            getattr(result, 'short_wins', 0),
            getattr(result, 'short_pnl', 0.0),
            getattr(result, 'flip_count', 0)
        ))

        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return strategy_id

    def save_strategies_batch(self, results: List[Any], run_id: int = None,
                              symbol: str = None, timeframe: str = None,
                              data_source: str = None, data_start: str = None,
                              data_end: str = None) -> int:
        """Save multiple strategy results efficiently."""
        saved = 0
        for result in results:
            try:
                self.save_strategy(result, run_id, symbol, timeframe,
                                   data_source, data_start, data_end)
                saved += 1
            except Exception as e:
                print(f"Error saving strategy {result.strategy_name}: {e}")
        return saved

    def get_top_strategies(self, limit: int = 10, symbol: str = None,
                           timeframe: str = None, min_trades: int = 3,
                           min_win_rate: float = 0.0) -> List[Dict]:
        """
        Get top strategies by composite score.

        Args:
            limit: Maximum number of results
            symbol: Filter by trading pair
            timeframe: Filter by timeframe
            min_trades: Minimum number of trades required
            min_win_rate: Minimum win rate required

        Returns:
            List of strategy dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT * FROM strategies
            WHERE total_trades >= ? AND win_rate >= ?
        '''
        params = [min_trades, min_win_rate]

        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        query += ' ORDER BY composite_score DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_best_by_win_rate(self, limit: int = 10, min_trades: int = 5) -> List[Dict]:
        """Get strategies with highest win rate (min trades filter)."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM strategies
            WHERE total_trades >= ?
            ORDER BY win_rate DESC
            LIMIT ?
        ''', (min_trades, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_best_by_profit_factor(self, limit: int = 10, min_trades: int = 5) -> List[Dict]:
        """Get strategies with highest profit factor."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM strategies
            WHERE total_trades >= ? AND profit_factor > 0
            ORDER BY profit_factor DESC
            LIMIT ?
        ''', (min_trades, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def search_strategies(self, strategy_name: str = None, category: str = None,
                          min_win_rate: float = None, min_pnl: float = None,
                          symbol: str = None, timeframe: str = None) -> List[Dict]:
        """Search strategies with various filters."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM strategies WHERE 1=1'
        params = []

        if strategy_name:
            query += ' AND strategy_name LIKE ?'
            params.append(f'%{strategy_name}%')

        if category:
            query += ' AND strategy_category LIKE ?'
            params.append(f'%{category}%')

        if min_win_rate is not None:
            query += ' AND win_rate >= ?'
            params.append(min_win_rate)

        if min_pnl is not None:
            query += ' AND total_pnl >= ?'
            params.append(min_pnl)

        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        query += ' ORDER BY composite_score DESC LIMIT 100'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict]:
        """Get a single strategy by ID."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_dict(row) if row else None

    def get_optimization_run_by_id(self, run_id: int) -> Optional[Dict]:
        """Get optimization run by ID to access risk_percent and other settings."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM optimization_runs WHERE id = ?', (run_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_optimization_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent optimization runs."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM optimization_runs
            ORDER BY started_at DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

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

        conn.close()

        return {
            'total_strategies': total_strategies,
            'total_optimization_runs': total_runs,
            'profitable_strategies': profitable_strategies,
            'avg_win_rate': round(avg_win_rate, 2),
            'best_composite_score': round(best_score, 4),
            'symbols_tested': symbols,
            'timeframes_tested': timeframes
        }

    def get_filter_options(self) -> Dict:
        """Get distinct symbols, timeframes, and date range for filter dropdowns."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get distinct symbols
        cursor.execute("SELECT DISTINCT symbol FROM strategies WHERE symbol IS NOT NULL ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]

        # Get distinct timeframes
        cursor.execute("SELECT DISTINCT timeframe FROM strategies WHERE timeframe IS NOT NULL ORDER BY timeframe")
        timeframes = [row[0] for row in cursor.fetchall()]

        # Get date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM strategies")
        date_row = cursor.fetchone()
        date_range = {
            "min": date_row[0] if date_row[0] else None,
            "max": date_row[1] if date_row[1] else None
        }

        conn.close()
        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "date_range": date_range
        }

    def _row_to_dict(self, row: sqlite3.Row, parse_equity_curve: bool = True) -> Dict:
        """
        Convert a database row to a dictionary with parsed JSON fields.

        Args:
            row: SQLite Row object
            parse_equity_curve: If False, skip parsing the large equity_curve field for performance
        """
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
            # Don't include the raw JSON string - set to empty for performance
            d['equity_curve'] = []

        return d

    def update_elite_status(self, strategy_id: int, elite_status: str,
                             periods_passed: int, periods_total: int,
                             validation_data: str = None,
                             elite_score: float = 0) -> bool:
        """
        Update elite validation status for a strategy.

        Args:
            strategy_id: Strategy ID
            elite_status: 'pending', 'elite', 'partial', 'failed'
            periods_passed: Number of validation periods that passed
            periods_total: Total number of testable periods
            validation_data: JSON string with per-period results
            elite_score: Consistency score 0-100

        Returns:
            True if updated, False if strategy not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE strategies
            SET elite_status = ?,
                elite_validated_at = ?,
                elite_periods_passed = ?,
                elite_periods_total = ?,
                elite_validation_data = ?,
                elite_score = ?
            WHERE id = ?
        ''', (elite_status, datetime.now().isoformat(), periods_passed,
              periods_total, validation_data, elite_score, strategy_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def get_elite_strategies_optimized(self, top_n_per_market: int = 10) -> List[Dict]:
        """
        Get top N validated strategies per pair/timeframe, efficiently at the database level.
        Uses window functions for efficient per-group limiting.

        Args:
            top_n_per_market: Max strategies to return per (symbol, timeframe) pair

        Returns:
            List of elite strategies, sorted by elite_score descending
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use ROW_NUMBER() window function to get top N per market efficiently
        # This avoids loading all strategies into memory
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
            WHERE rank <= ?
            ORDER BY elite_score DESC
        ''', (top_n_per_market,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_strategies_pending_validation(self, limit: int = 100) -> List[Dict]:
        """
        Get strategies that need elite validation, ordered by composite score.
        Efficient query that only fetches pending strategies.
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM strategies
            WHERE elite_status IS NULL OR elite_status = 'pending'
            ORDER BY composite_score DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies from the database."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM strategies ORDER BY id DESC')
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def delete_strategy(self, strategy_id: int) -> bool:
        """Delete a strategy by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM strategies WHERE id = ?', (strategy_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def remove_duplicates(self) -> int:
        """Remove duplicate strategies, keeping the most recent (highest ID) of each group."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Find duplicates - same strategy with same key metrics
        # Keep the one with highest ID (most recent)
        cursor.execute('''
            DELETE FROM strategies
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM strategies
                GROUP BY strategy_name, symbol, timeframe,
                         ROUND(tp_percent, 1), ROUND(sl_percent, 1),
                         total_trades, ROUND(win_rate, 1),
                         ROUND(total_pnl, 1), ROUND(profit_factor, 2)
            )
        ''')

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        print(f"Removed {deleted} duplicate strategies from database")
        return deleted

    def clear_all(self) -> int:
        """Clear entire database - all strategies, runs, and tracking data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM strategies')
        count = cursor.fetchone()[0]

        # Clear all strategy and optimization data
        cursor.execute('DELETE FROM strategies')
        cursor.execute('DELETE FROM optimization_runs')
        cursor.execute('DELETE FROM completed_optimizations')

        # Clear all priority queue tables
        cursor.execute('DELETE FROM priority_items')
        cursor.execute('DELETE FROM priority_pairs')
        cursor.execute('DELETE FROM priority_periods')
        cursor.execute('DELETE FROM priority_timeframes')
        cursor.execute('DELETE FROM priority_granularities')
        cursor.execute('DELETE FROM priority_settings')

        conn.commit()
        conn.close()

        print(f"Cleared entire database: {count} strategies + all tracking data")
        return count

    # =========================================================================
    # COMPLETED OPTIMIZATIONS TRACKING - For accurate resume functionality
    # =========================================================================

    def record_completed_optimization(self, pair: str, period_label: str,
                                       timeframe_label: str, granularity_label: str,
                                       strategies_found: int = 0, source: str = 'binance'):
        """
        Record that an optimization combination has been completed.
        Uses INSERT OR REPLACE to update if already exists.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO completed_optimizations
            (pair, period_label, timeframe_label, granularity_label, strategies_found, completed_at, source)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ''', (pair, period_label, timeframe_label, granularity_label, strategies_found, source))

        conn.commit()
        conn.close()

    def is_optimization_completed(self, pair: str, period_label: str,
                                   timeframe_label: str, granularity_label: str) -> bool:
        """Check if a specific combination has been completed."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 1 FROM completed_optimizations
            WHERE pair = ? AND period_label = ? AND timeframe_label = ? AND granularity_label = ?
        ''', (pair, period_label, timeframe_label, granularity_label))

        result = cursor.fetchone() is not None
        conn.close()
        return result

    def get_completed_optimizations(self, with_timestamps: bool = False) -> dict:
        """
        Get completed optimization combinations.

        Args:
            with_timestamps: If True, returns dict mapping combo key to completed_at timestamp.
                           If False, returns set of combo keys (legacy behavior).

        Returns:
            If with_timestamps=True: dict of {(pair, period, timeframe, granularity): completed_at}
            If with_timestamps=False: set of (pair, period_label, timeframe_label, granularity_label)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT pair, period_label, timeframe_label, granularity_label, completed_at
            FROM completed_optimizations
        ''')

        if with_timestamps:
            # Return dict with timestamps for period boundary checking
            completed = {
                (row[0], row[1], row[2], row[3]): row[4]
                for row in cursor.fetchall()
            }
        else:
            # Legacy behavior: return set of keys only
            completed = {(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()}

        conn.close()
        return completed

    def get_completed_optimizations_count(self) -> int:
        """Get count of completed optimizations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM completed_optimizations')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear_completed_optimizations(self, pair: str = None, granularity_label: str = None):
        """
        Clear completed optimization records.
        - No args: clear all
        - pair only: clear all for that pair
        - granularity only: clear all for that granularity
        - both: clear specific pair+granularity combos
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if pair and granularity_label:
            cursor.execute('DELETE FROM completed_optimizations WHERE pair = ? AND granularity_label = ?',
                          (pair, granularity_label))
        elif pair:
            cursor.execute('DELETE FROM completed_optimizations WHERE pair = ?', (pair,))
        elif granularity_label:
            cursor.execute('DELETE FROM completed_optimizations WHERE granularity_label = ?', (granularity_label,))
        else:
            cursor.execute('DELETE FROM completed_optimizations')

        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # =========================================================================
    # PRIORITY QUEUE MANAGEMENT
    # =========================================================================

    def _init_priority_table(self):
        """Create priority_items table if it doesn't exist (legacy)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pair, period_label, timeframe_label, granularity_label)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority_position ON priority_items(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority_enabled ON priority_items(enabled)')

        # Initialize new 3-list priority tables
        self._init_priority_lists_tables(cursor)

        conn.commit()
        conn.close()

    def _init_priority_lists_tables(self, cursor):
        """Create separate priority tables for pairs, periods, and timeframes."""
        # Trading Pairs priority
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Historical Periods priority
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                months REAL NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Timeframes priority
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_timeframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Granularities priority
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_granularities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL,
                value TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                n_trials INTEGER NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Global priority settings (granularity, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS priority_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pairs_position ON priority_pairs(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_periods_position ON priority_periods(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframes_position ON priority_timeframes(position)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_granularities_position ON priority_granularities(position)')

    def get_priority_list(self) -> List[Dict]:
        """Get all priority items ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM priority_items
            ORDER BY position ASC
        ''')

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def add_priority_item(self, pair: str, period_label: str, period_months: float,
                          timeframe_label: str, timeframe_minutes: int,
                          granularity_label: str, granularity_trials: int,
                          source: str = 'binance') -> Optional[int]:
        """Add a new priority item at the end of the list."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get next position
        cursor.execute('SELECT COALESCE(MAX(position), 0) + 1 FROM priority_items')
        next_position = cursor.fetchone()[0]

        try:
            cursor.execute('''
                INSERT INTO priority_items
                (position, pair, period_label, period_months, timeframe_label,
                 timeframe_minutes, granularity_label, granularity_trials, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (next_position, pair, period_label, period_months,
                  timeframe_label, timeframe_minutes, granularity_label,
                  granularity_trials, source))

            item_id = cursor.lastrowid
            conn.commit()
            return item_id
        except sqlite3.IntegrityError:
            return None  # Duplicate
        finally:
            conn.close()

    def delete_priority_item(self, item_id: int) -> bool:
        """Delete a priority item and reorder remaining items."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM priority_items WHERE id = ?', (item_id,))
        deleted = cursor.rowcount > 0

        if deleted:
            # Reorder positions to eliminate gaps
            cursor.execute('''
                WITH numbered AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY position) as new_pos
                    FROM priority_items
                )
                UPDATE priority_items
                SET position = (SELECT new_pos FROM numbered WHERE numbered.id = priority_items.id)
            ''')

        conn.commit()
        conn.close()
        return deleted

    def reorder_priority_items(self, id_order: List[int]) -> bool:
        """Update positions based on new order of IDs."""
        conn = self._get_connection()
        cursor = conn.cursor()

        for position, item_id in enumerate(id_order, start=1):
            cursor.execute('''
                UPDATE priority_items
                SET position = ?, updated_at = ?
                WHERE id = ?
            ''', (position, datetime.now().isoformat(), item_id))

        conn.commit()
        conn.close()
        return True

    def toggle_priority_item(self, item_id: int) -> Optional[bool]:
        """Toggle enabled status. Returns new status or None if not found."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT enabled FROM priority_items WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        new_status = 0 if row[0] else 1
        cursor.execute('UPDATE priority_items SET enabled = ?, updated_at = ? WHERE id = ?',
                       (new_status, datetime.now().isoformat(), item_id))

        conn.commit()
        conn.close()
        return bool(new_status)

    def clear_priority_items(self) -> int:
        """Clear all priority items. Returns count deleted."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM priority_items')
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM priority_items')

        conn.commit()
        conn.close()
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
        """
        Get all priority lists in a single database connection.
        Much faster than calling each method separately.
        Returns dict with 'pairs', 'periods', 'timeframes', 'granularities', and 'populated'.
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if populated
        cursor.execute('SELECT COUNT(*) FROM priority_pairs')
        pairs_count = cursor.fetchone()[0]
        populated = pairs_count > 0

        # Fetch all lists in single connection
        cursor.execute('SELECT * FROM priority_pairs ORDER BY position ASC')
        pairs = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM priority_periods ORDER BY position ASC')
        periods = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM priority_timeframes ORDER BY position ASC')
        timeframes = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM priority_granularities ORDER BY position ASC')
        granularities = [dict(row) for row in cursor.fetchall()]

        conn.close()

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
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_pairs ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_priority_periods(self) -> List[Dict]:
        """Get all historical periods ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_periods ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_priority_timeframes(self) -> List[Dict]:
        """Get all timeframes ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_timeframes ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_enabled_priority_pairs(self) -> List[Dict]:
        """Get enabled trading pairs ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_pairs WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_enabled_priority_periods(self) -> List[Dict]:
        """Get enabled historical periods ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_periods WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_enabled_priority_timeframes(self) -> List[Dict]:
        """Get enabled timeframes ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_timeframes WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_priority_granularities(self) -> List[Dict]:
        """Get all granularities ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_granularities ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_enabled_priority_granularities(self) -> List[Dict]:
        """Get enabled granularities ordered by position."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM priority_granularities WHERE enabled = 1 ORDER BY position ASC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_priority_setting(self, key: str) -> Optional[str]:
        """Get a priority setting value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM priority_settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def set_priority_setting(self, key: str, value: str):
        """Set a priority setting value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO priority_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
        ''', (key, value, datetime.now().isoformat(), value, datetime.now().isoformat()))
        conn.commit()
        conn.close()

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
            cursor.execute(f'UPDATE {table} SET position = ? WHERE id = ?', (position, item_id))

        conn.commit()
        conn.close()
        return True

    def toggle_priority_list_item(self, list_type: str, item_id: int) -> Optional[bool]:
        """Toggle enabled status in a specific list. Returns new status."""
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

        cursor.execute(f'SELECT enabled FROM {table} WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        new_status = 0 if row[0] else 1
        cursor.execute(f'UPDATE {table} SET enabled = ? WHERE id = ?', (new_status, item_id))

        conn.commit()
        conn.close()
        return bool(new_status)

    def reset_priority_pairs(self, pairs: List[str]):
        """Reset trading pairs to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_pairs')

        for pos, pair in enumerate(pairs, start=1):
            cursor.execute('''
                INSERT INTO priority_pairs (position, value, label, enabled)
                VALUES (?, ?, ?, 1)
            ''', (pos, pair, pair))

        conn.commit()
        conn.close()

    def reset_priority_periods(self, periods: List[Dict]):
        """Reset historical periods to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_periods')

        for pos, period in enumerate(periods, start=1):
            cursor.execute('''
                INSERT INTO priority_periods (position, value, label, months, enabled)
                VALUES (?, ?, ?, ?, 1)
            ''', (pos, period['label'], period['label'], period['months']))

        conn.commit()
        conn.close()

    def reset_priority_timeframes(self, timeframes: List[Dict]):
        """Reset timeframes to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_timeframes')

        for pos, tf in enumerate(timeframes, start=1):
            cursor.execute('''
                INSERT INTO priority_timeframes (position, value, label, minutes, enabled)
                VALUES (?, ?, ?, ?, 1)
            ''', (pos, tf['label'], tf['label'], tf['minutes']))

        conn.commit()
        conn.close()

    def reset_priority_granularities(self, granularities: List[Dict]):
        """Reset granularities to defaults."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM priority_granularities')

        for pos, gran in enumerate(granularities, start=1):
            cursor.execute('''
                INSERT INTO priority_granularities (position, value, label, n_trials, enabled)
                VALUES (?, ?, ?, ?, 1)
            ''', (pos, gran['label'], gran['label'], gran['n_trials']))

        conn.commit()
        conn.close()

    def enable_all_priority_items(self):
        """Enable all items in all priority lists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE priority_pairs SET enabled = 1')
        cursor.execute('UPDATE priority_periods SET enabled = 1')
        cursor.execute('UPDATE priority_timeframes SET enabled = 1')
        cursor.execute('UPDATE priority_granularities SET enabled = 1')
        conn.commit()
        conn.close()

    def disable_all_priority_items(self):
        """Disable all items in all priority lists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE priority_pairs SET enabled = 0')
        cursor.execute('UPDATE priority_periods SET enabled = 0')
        cursor.execute('UPDATE priority_timeframes SET enabled = 0')
        cursor.execute('UPDATE priority_granularities SET enabled = 0')
        conn.commit()
        conn.close()

    def has_priority_lists_populated(self) -> bool:
        """Check if the new priority lists have any data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM priority_pairs')
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

    # =========================================================================
    # OPTIMIZED QUERIES FOR PERFORMANCE
    # =========================================================================

    def get_elite_counts(self) -> Dict[str, int]:
        """
        Get elite status counts using SQL aggregation instead of loading all rows.
        Much faster than: sum(1 for s in get_all_strategies() if s.get('elite_status') == 'elite')
        """
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

        conn.close()
        return counts

    def get_db_stats_optimized(self) -> Dict[str, Any]:
        """
        Get database statistics using SQL aggregation.
        Much faster than loading all strategies and counting in Python.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Single query for all counts
        cursor.execute('''
            SELECT
                COUNT(*) as total_strategies,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT timeframe) as unique_timeframes,
                SUM(CASE WHEN elite_status = 'elite' THEN 1 ELSE 0 END) as elite_count
            FROM strategies
        ''')

        row = cursor.fetchone()
        conn.close()

        return {
            'total_strategies': row[0] or 0,
            'unique_symbols': row[1] or 0,
            'unique_timeframes': row[2] or 0,
            'elite_count': row[3] or 0
        }

    def get_strategies_paginated(self, limit: int = 500, offset: int = 0,
                                  symbol: str = None, timeframe: str = None,
                                  sort_by: str = 'id', sort_order: str = 'DESC') -> Dict:
        """
        Get strategies with pagination support.
        Returns {strategies: [...], total: N, has_more: bool, limit: N, offset: N}
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build WHERE clause
        where_clauses = []
        params = []

        if symbol:
            where_clauses.append('symbol = ?')
            params.append(symbol)

        if timeframe:
            where_clauses.append('timeframe = ?')
            params.append(timeframe)

        where_sql = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''

        # Validate sort column to prevent SQL injection
        valid_sort_columns = {'id', 'composite_score', 'win_rate', 'profit_factor',
                              'total_pnl', 'created_at', 'elite_score'}
        if sort_by not in valid_sort_columns:
            sort_by = 'id'

        sort_direction = 'DESC' if sort_order.upper() == 'DESC' else 'ASC'

        # Get total count
        cursor.execute(f'SELECT COUNT(*) FROM strategies{where_sql}', params)
        total = cursor.fetchone()[0]

        # Get paginated results
        query = f'''
            SELECT * FROM strategies{where_sql}
            ORDER BY {sort_by} {sort_direction}
            LIMIT ? OFFSET ?
        '''
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()
        conn.close()

        strategies = [self._row_to_dict(row) for row in rows]

        return {
            'strategies': strategies,
            'total': total,
            'has_more': offset + len(strategies) < total,
            'limit': limit,
            'offset': offset
        }

    def get_total_strategy_count(self) -> int:
        """Get total count of strategies (very fast)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM strategies')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # =========================================================================
    # OPTIMIZED QUERY METHODS FOR API ROUTES
    # =========================================================================

    def get_elite_strategies_filtered(
        self,
        status_filter: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get elite strategies with SQL-level filtering.
        Much faster than loading all strategies and filtering in Python.

        Args:
            status_filter: Filter by elite_status ('elite', 'partial', 'failed', 'pending')
            limit: Maximum number of results
            offset: Offset for pagination
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status_filter:
            cursor.execute('''
                SELECT * FROM strategies
                WHERE elite_status = ?
                ORDER BY elite_score DESC
                LIMIT ? OFFSET ?
            ''', (status_filter, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM strategies
                WHERE elite_status IS NOT NULL
                ORDER BY elite_score DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))

        rows = cursor.fetchall()
        conn.close()

        # Skip equity_curve parsing for list views
        return [self._row_to_dict(row, parse_equity_curve=False) for row in rows]

    def get_elite_leaderboard(self, limit: int = 20) -> Dict:
        """
        Get top elite strategies efficiently using SQL.

        Returns:
            Dict with 'leaderboard' list and 'total_elite' count
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get leaderboard with LIMIT
        cursor.execute('''
            SELECT * FROM strategies
            WHERE elite_status = 'elite'
            ORDER BY elite_score DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()

        # Get total count efficiently
        cursor.execute("SELECT COUNT(*) FROM strategies WHERE elite_status = 'elite'")
        total_elite = cursor.fetchone()[0]

        conn.close()

        return {
            "leaderboard": [self._row_to_dict(row, parse_equity_curve=False) for row in rows],
            "total_elite": total_elite
        }

    def get_elite_stats_optimized(self) -> Dict:
        """
        Get elite validation statistics using SQL aggregation.
        Much faster than loading all strategies and counting in Python.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get counts by status in a single query
        cursor.execute('''
            SELECT
                COALESCE(elite_status, 'not_validated') as status,
                COUNT(*) as count
            FROM strategies
            GROUP BY COALESCE(elite_status, 'not_validated')
        ''')

        status_counts = {
            'elite': 0,
            'partial': 0,
            'failed': 0,
            'pending': 0,
            'untestable': 0,
            'skipped': 0,
            'not_validated': 0
        }

        for row in cursor.fetchall():
            status, count = row
            if status in status_counts:
                status_counts[status] = count
            else:
                status_counts['not_validated'] += count

        # Get total count
        cursor.execute('SELECT COUNT(*) FROM strategies')
        total = cursor.fetchone()[0]

        # Get average elite score for elite strategies
        cursor.execute('''
            SELECT AVG(COALESCE(elite_score, 0))
            FROM strategies
            WHERE elite_status = 'elite'
        ''')
        avg_score = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_strategies": total,
            "status_counts": status_counts,
            "avg_elite_score": round(avg_score, 2)
        }

    def get_top_strategies_per_market(
        self,
        top_n: int = 10,
        symbol: str = None,
        timeframe: str = None,
        min_win_rate: float = 0.0,
        total_limit: int = 500
    ) -> List[Dict]:
        """
        Get top N strategies per (symbol, timeframe) pair using SQL window functions.
        Much faster than loading all and grouping in Python.

        Args:
            top_n: Max strategies per market
            symbol: Filter by symbol (optional)
            timeframe: Filter by timeframe (optional)
            min_win_rate: Minimum win rate filter
            total_limit: Maximum total results
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build WHERE clause
        where_parts = ['win_rate >= ?']
        params = [min_win_rate]

        if symbol:
            where_parts.append('symbol = ?')
            params.append(symbol)

        if timeframe:
            where_parts.append('timeframe = ?')
            params.append(timeframe)

        where_clause = ' AND '.join(where_parts)

        # Use window function to get top N per market
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
            WHERE rank <= ?
            ORDER BY composite_score DESC
            LIMIT ?
        '''

        params.extend([top_n, total_limit])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row, parse_equity_curve=False) for row in rows]

    def get_pending_validation_count(self) -> int:
        """Get count of strategies pending validation (very fast)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM strategies
            WHERE elite_status IS NULL OR elite_status = 'pending'
        ''')
        count = cursor.fetchone()[0]
        conn.close()
        return count


# Singleton instance for easy access
_db_instance: Optional[StrategyDatabase] = None
_migration_done: bool = False


def _run_migration(db_path: str):
    """Run database migration to add any missing columns."""
    global _migration_done
    if _migration_done:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if strategies table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
    if not cursor.fetchone():
        conn.close()
        return

    # Get existing columns
    cursor.execute("PRAGMA table_info(strategies)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    migration_columns = [
        ('indicator_params', 'TEXT'),
        ('tuning_improved', 'INTEGER DEFAULT 0'),
        ('tuning_score_before', 'REAL'),
        ('tuning_score_after', 'REAL'),
        ('tuning_improvement_pct', 'REAL'),
    ]

    for col_name, col_type in migration_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f'ALTER TABLE strategies ADD COLUMN {col_name} {col_type}')
                print(f"Migration: Added column {col_name}")
            except Exception as e:
                print(f"Migration warning for {col_name}: {e}")

    conn.commit()
    conn.close()
    _migration_done = True
    print("Database migration check complete")


def get_strategy_db(db_path: str = None) -> StrategyDatabase:
    """Get or create the strategy database singleton."""
    global _db_instance

    # Determine proper database path
    if db_path is None:
        from pathlib import Path
        backend_dir = Path(__file__).parent
        project_dir = backend_dir.parent

        # Check if running in Docker
        if Path("/app").exists():
            db_path = "/app/data/strategies.db"
        else:
            data_dir = project_dir / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "strategies.db")

    # Always run migration check first
    _run_migration(db_path)

    if _db_instance is None:
        _db_instance = StrategyDatabase(db_path)
    return _db_instance
