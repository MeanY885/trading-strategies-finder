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
from dataclasses import asdict
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

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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

        # Create indexes for fast querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_win_rate ON strategies(win_rate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_profit_factor ON strategies(profit_factor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_composite_score ON strategies(composite_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON strategies(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON strategies(timeframe)')

        # Migration: Add missing columns to existing databases
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
                    print(f"Added missing column: {col_name}")
                except Exception as e:
                    print(f"Column {col_name} migration skipped: {e}")

        conn.commit()
        conn.close()

        print(f"Strategy database initialized: {self.db_path}")

    def start_optimization_run(self, symbol: str = None, timeframe: str = None,
                               data_source: str = None, data_rows: int = 0,
                               capital: float = 1000.0, risk_percent: float = 2.0) -> int:
        """Start a new optimization run and return its ID."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
                      tuning_info: Dict = None) -> int:
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
            Strategy ID in database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract params
        params = result.params if hasattr(result, 'params') else {}
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

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

        # Use getattr with defaults for compatibility with both old and new result formats
        cursor.execute('''
            INSERT INTO strategies
            (strategy_name, strategy_category, params, total_trades, win_rate,
             profit_factor, total_pnl, max_drawdown, equity_r_squared, recovery_factor,
             sharpe_ratio, composite_score, tp_percent, sl_percent,
             indicator_params, tuning_improved, tuning_score_before, tuning_score_after, tuning_improvement_pct,
             val_pnl, val_profit_factor, val_win_rate, found_by, data_source, symbol,
             timeframe, data_start, data_end, optimization_run_id, equity_curve)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            equity_curve
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_dict(row) if row else None

    def get_optimization_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent optimization runs."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to a dictionary with parsed JSON fields."""
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

        if d.get('equity_curve'):
            try:
                d['equity_curve'] = json.loads(d['equity_curve'])
            except:
                d['equity_curve'] = []

        return d

    def delete_strategy(self, strategy_id: int) -> bool:
        """Delete a strategy by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM strategies WHERE id = ?', (strategy_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def clear_all(self) -> int:
        """Clear all strategies (use with caution!)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM strategies')
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM strategies')
        cursor.execute('DELETE FROM optimization_runs')

        conn.commit()
        conn.close()

        print(f"Cleared {count} strategies from database")
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


def get_strategy_db(db_path: str = "data/strategies.db") -> StrategyDatabase:
    """Get or create the strategy database singleton."""
    global _db_instance

    # Always run migration check first
    _run_migration(db_path)

    if _db_instance is None:
        _db_instance = StrategyDatabase(db_path)
    return _db_instance
