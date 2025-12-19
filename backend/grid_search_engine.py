"""
EXHAUSTIVE GRID SEARCH ENGINE
=============================
Tests ALL combinations of strategies, directions, TP%, and SL%.
No strategy is off the table.

This is the engine that finds profitable strategies like:
- SHORT 0.6% TP / 5% SL = £30 profit
- SHORT 2.6% TP / 4-5% SL = £100 profit

Author: BTCGBP ML Optimizer
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import os
import time

from exact_match_backtester import (
    ExactMatchBacktester, StrategyResult, StrategyDatabase, load_csv_data
)


@dataclass
class GridSearchConfig:
    """Configuration for grid search"""

    # Directions to test
    directions: List[str] = field(default_factory=lambda: ["long", "short"])

    # TP percentages to test (wide range as requested)
    tp_percents: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
        3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0
    ])

    # SL percentages to test (wide range)
    sl_percents: List[float] = field(default_factory=lambda: [
        0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0
    ])

    # Entry conditions to test
    entry_conditions: List[str] = field(default_factory=lambda: [
        "always",
        "rsi_oversold",
        "bb_touch",
        "sma_cross",
        "consecutive_red",
        "consecutive_green",
        "price_drop",
        "price_rise",
        "adx_sideways",
        "adx_trending",
        "ema_trend",
        "every_n_bars",
    ])

    # Entry condition parameters to test
    entry_params_grid: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "always": [{}],
        "rsi_oversold": [
            {"rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80},
            {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
            {"rsi_period": 7, "rsi_oversold": 25, "rsi_overbought": 75},
            {"rsi_period": 21, "rsi_oversold": 35, "rsi_overbought": 65},
        ],
        "bb_touch": [
            {"bb_period": 20},
        ],
        "sma_cross": [
            {"sma_fast": 10, "sma_slow": 50},
            {"sma_fast": 20, "sma_slow": 100},
            {"sma_fast": 10, "sma_slow": 20},
        ],
        "consecutive_red": [
            {"consec_count": 2},
            {"consec_count": 3},
            {"consec_count": 4},
            {"consec_count": 5},
        ],
        "consecutive_green": [
            {"consec_count": 2},
            {"consec_count": 3},
            {"consec_count": 4},
            {"consec_count": 5},
        ],
        "price_drop": [
            {"drop_percent": 1.0, "lookback": 10},
            {"drop_percent": 2.0, "lookback": 10},
            {"drop_percent": 3.0, "lookback": 20},
            {"drop_percent": 5.0, "lookback": 20},
        ],
        "price_rise": [
            {"rise_percent": 1.0, "lookback": 10},
            {"rise_percent": 2.0, "lookback": 10},
            {"rise_percent": 3.0, "lookback": 20},
            {"rise_percent": 5.0, "lookback": 20},
        ],
        "adx_sideways": [
            {"adx_threshold": 20, "rsi_oversold": 30, "rsi_overbought": 70},
            {"adx_threshold": 25, "rsi_oversold": 30, "rsi_overbought": 70},
            {"adx_threshold": 30, "rsi_oversold": 35, "rsi_overbought": 65},
        ],
        "adx_trending": [
            {"adx_threshold": 20},
            {"adx_threshold": 25},
            {"adx_threshold": 30},
        ],
        "ema_trend": [
            {"ema_period": 9},
            {"ema_period": 21},
            {"ema_period": 55},
        ],
        "every_n_bars": [
            {"n_bars": 5},
            {"n_bars": 10},
            {"n_bars": 20},
        ],
    })

    # Minimum trades required for valid strategy
    min_trades: int = 5

    # Use parallel processing
    use_parallel: bool = True
    max_workers: int = 4


class GridSearchEngine:
    """
    Exhaustive grid search engine that tests ALL strategy combinations.

    Example usage:
        engine = GridSearchEngine(df)
        results = engine.run_full_search()
        top_strategies = engine.get_top_n(10)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: GridSearchConfig = None,
        initial_capital: float = 1000.0,
        db_path: str = None
    ):
        self.df = df
        self.config = config or GridSearchConfig()
        self.initial_capital = initial_capital
        self.backtester = ExactMatchBacktester(df, initial_capital)
        self.db = StrategyDatabase(db_path) if db_path else StrategyDatabase()

        self.results: List[StrategyResult] = []
        self.search_progress = {
            'total_combinations': 0,
            'tested': 0,
            'profitable': 0,
            'current': '',
            'start_time': None,
            'elapsed': 0
        }

    def count_combinations(self) -> int:
        """Count total number of strategy combinations to test"""
        total = 0
        for entry_cond in self.config.entry_conditions:
            params_list = self.config.entry_params_grid.get(entry_cond, [{}])
            total += (
                len(self.config.directions) *
                len(self.config.tp_percents) *
                len(self.config.sl_percents) *
                len(params_list)
            )
        return total

    def _test_single_combination(
        self,
        direction: str,
        tp_percent: float,
        sl_percent: float,
        entry_condition: str,
        entry_params: Dict
    ) -> Optional[StrategyResult]:
        """Test a single strategy combination"""
        try:
            result = self.backtester.run_backtest(
                direction=direction,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                entry_condition=entry_condition,
                entry_params=entry_params
            )
            return result
        except Exception as e:
            print(f"Error testing {direction} {entry_condition} TP{tp_percent} SL{sl_percent}: {e}")
            return None

    def run_full_search(
        self,
        progress_callback: Callable[[Dict], None] = None,
        save_to_db: bool = True
    ) -> List[StrategyResult]:
        """
        Run exhaustive grid search across ALL combinations.

        Args:
            progress_callback: Optional callback for progress updates
            save_to_db: Whether to save results to SQLite database

        Returns:
            List of all StrategyResults, sorted by total P&L
        """
        self.results = []
        self.search_progress['total_combinations'] = self.count_combinations()
        self.search_progress['tested'] = 0
        self.search_progress['profitable'] = 0
        self.search_progress['start_time'] = time.time()

        print(f"Starting exhaustive grid search...")
        print(f"Total combinations to test: {self.search_progress['total_combinations']:,}")
        print("=" * 60)

        # Generate all combinations
        combinations = []
        for entry_condition in self.config.entry_conditions:
            params_list = self.config.entry_params_grid.get(entry_condition, [{}])

            for direction in self.config.directions:
                for tp in self.config.tp_percents:
                    for sl in self.config.sl_percents:
                        for params in params_list:
                            combinations.append((
                                direction, tp, sl, entry_condition, params
                            ))

        # Test each combination
        for i, (direction, tp, sl, entry_cond, params) in enumerate(combinations):
            self.search_progress['tested'] = i + 1
            self.search_progress['current'] = f"{entry_cond} {direction} TP{tp}% SL{sl}%"
            self.search_progress['elapsed'] = time.time() - self.search_progress['start_time']

            result = self._test_single_combination(direction, tp, sl, entry_cond, params)

            if result and result.total_trades >= self.config.min_trades:
                self.results.append(result)

                if result.total_pnl_gbp > 0:
                    self.search_progress['profitable'] += 1

                # Save to database
                if save_to_db:
                    try:
                        self.db.save_strategy(result)
                    except Exception as e:
                        pass  # Ignore DB errors

            # Progress callback
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(self.search_progress)

            # Print progress every 500 combinations
            if (i + 1) % 500 == 0:
                elapsed = time.time() - self.search_progress['start_time']
                rate = (i + 1) / elapsed
                remaining = (self.search_progress['total_combinations'] - i - 1) / rate
                print(f"Progress: {i+1}/{self.search_progress['total_combinations']} "
                      f"({(i+1)/self.search_progress['total_combinations']*100:.1f}%) "
                      f"Profitable: {self.search_progress['profitable']} "
                      f"ETA: {remaining/60:.1f}min")

        # Sort by total P&L
        self.results.sort(key=lambda r: r.total_pnl_gbp, reverse=True)

        total_time = time.time() - self.search_progress['start_time']
        print("=" * 60)
        print(f"Grid search complete!")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Strategies tested: {len(combinations):,}")
        print(f"Valid strategies (>={self.config.min_trades} trades): {len(self.results):,}")
        print(f"Profitable strategies: {self.search_progress['profitable']:,}")

        return self.results

    def run_quick_search(
        self,
        directions: List[str] = None,
        tp_range: Tuple[float, float] = (0.3, 5.0),
        sl_range: Tuple[float, float] = (1.0, 10.0),
        step: float = 0.5,
        entry_conditions: List[str] = None
    ) -> List[StrategyResult]:
        """
        Run a quick search with limited parameter space.

        Good for finding winning strategies quickly before full search.
        """
        directions = directions or ["short"]
        entry_conditions = entry_conditions or ["always", "rsi_oversold", "consecutive_red"]

        # Generate TP and SL ranges
        tp_percents = np.arange(tp_range[0], tp_range[1] + step, step).tolist()
        sl_percents = np.arange(sl_range[0], sl_range[1] + step, step).tolist()

        print(f"Quick search: {len(directions)} directions x {len(tp_percents)} TPs x {len(sl_percents)} SLs x {len(entry_conditions)} conditions")

        self.results = []

        for entry_cond in entry_conditions:
            params_list = self.config.entry_params_grid.get(entry_cond, [{}])[:1]  # Only first params set

            for direction in directions:
                for tp in tp_percents:
                    for sl in sl_percents:
                        for params in params_list:
                            result = self._test_single_combination(
                                direction, tp, sl, entry_cond, params
                            )
                            if result and result.total_trades >= self.config.min_trades:
                                self.results.append(result)
                                self.db.save_strategy(result)

        self.results.sort(key=lambda r: r.total_pnl_gbp, reverse=True)
        return self.results

    def get_top_n(
        self,
        n: int = 10,
        min_win_rate: float = 0,
        min_profit_factor: float = 0,
        direction: str = None
    ) -> List[StrategyResult]:
        """
        Get top N strategies by P&L with optional filters.

        Args:
            n: Number of strategies to return
            min_win_rate: Minimum win rate percentage
            min_profit_factor: Minimum profit factor
            direction: Filter by direction ("long" or "short")

        Returns:
            List of top N StrategyResult objects
        """
        filtered = self.results

        if min_win_rate > 0:
            filtered = [r for r in filtered if r.win_rate >= min_win_rate]

        if min_profit_factor > 0:
            filtered = [r for r in filtered if r.profit_factor >= min_profit_factor]

        if direction:
            filtered = [r for r in filtered if r.direction == direction]

        return filtered[:n]

    def get_best_by_category(self) -> Dict[str, StrategyResult]:
        """
        Get best strategy for each entry condition.

        Returns:
            Dict mapping entry_condition to best StrategyResult
        """
        best = {}

        for result in self.results:
            cond = result.entry_condition
            if cond not in best or result.total_pnl_gbp > best[cond].total_pnl_gbp:
                best[cond] = result

        return best

    def print_summary(self, top_n: int = 10):
        """Print summary of search results"""
        print("\n" + "=" * 80)
        print("TOP PERFORMING STRATEGIES")
        print("=" * 80)

        top = self.get_top_n(top_n)

        for i, r in enumerate(top, 1):
            print(f"\n#{i} | {r.strategy_name}")
            print(f"    Direction: {r.direction.upper()}")
            print(f"    Entry: {r.entry_condition}")
            print(f"    TP: {r.tp_percent}% | SL: {r.sl_percent}%")
            print(f"    Trades: {r.total_trades} | Win Rate: {r.win_rate:.1f}%")
            print(f"    Total P&L: £{r.total_pnl_gbp:.2f}")
            print(f"    Profit Factor: {r.profit_factor:.2f}")
            print(f"    Max Drawdown: £{r.max_drawdown_gbp:.2f}")

        # Summary stats
        profitable = [r for r in self.results if r.total_pnl_gbp > 0]
        print("\n" + "-" * 80)
        print("SUMMARY")
        print(f"Total strategies tested: {len(self.results)}")
        print(f"Profitable strategies: {len(profitable)} ({len(profitable)/len(self.results)*100:.1f}%)")

        if profitable:
            avg_profit = np.mean([r.total_pnl_gbp for r in profitable])
            print(f"Average profit (profitable only): £{avg_profit:.2f}")

        # Best by direction
        longs = [r for r in self.results if r.direction == "long"]
        shorts = [r for r in self.results if r.direction == "short"]

        if longs:
            best_long = max(longs, key=lambda r: r.total_pnl_gbp)
            print(f"\nBest LONG: TP{best_long.tp_percent}%/SL{best_long.sl_percent}% "
                  f"= £{best_long.total_pnl_gbp:.2f} ({best_long.win_rate:.1f}% WR)")

        if shorts:
            best_short = max(shorts, key=lambda r: r.total_pnl_gbp)
            print(f"Best SHORT: TP{best_short.tp_percent}%/SL{best_short.sl_percent}% "
                  f"= £{best_short.total_pnl_gbp:.2f} ({best_short.win_rate:.1f}% WR)")

    def export_results_json(self, filepath: str):
        """Export all results to JSON file"""
        data = {
            'search_config': {
                'directions': self.config.directions,
                'tp_percents': self.config.tp_percents,
                'sl_percents': self.config.sl_percents,
                'entry_conditions': self.config.entry_conditions,
            },
            'data_info': {
                'start': str(self.df['time'].iloc[0]),
                'end': str(self.df['time'].iloc[-1]),
                'candles': len(self.df),
            },
            'results': [
                {
                    'strategy_id': r.strategy_id,
                    'strategy_name': r.strategy_name,
                    'direction': r.direction,
                    'tp_percent': r.tp_percent,
                    'sl_percent': r.sl_percent,
                    'entry_condition': r.entry_condition,
                    'params': r.params,
                    'total_trades': r.total_trades,
                    'win_rate': round(r.win_rate, 2),
                    'total_pnl_gbp': round(r.total_pnl_gbp, 2),
                    'profit_factor': round(r.profit_factor, 2),
                    'max_drawdown_gbp': round(r.max_drawdown_gbp, 2),
                }
                for r in self.results[:100]  # Top 100 only
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results exported to {filepath}")


def find_best_strategy(
    csv_path: str,
    quick: bool = True
) -> List[StrategyResult]:
    """
    Convenience function to find best strategy from CSV data.

    Args:
        csv_path: Path to CSV file with OHLC data
        quick: If True, run quick search. If False, run full search.

    Returns:
        List of top strategies
    """
    print(f"Loading data from {csv_path}...")
    df = load_csv_data(csv_path)
    print(f"Loaded {len(df)} candles: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    engine = GridSearchEngine(df)

    if quick:
        print("\nRunning QUICK search (SHORT-focused, limited parameters)...")
        results = engine.run_quick_search(
            directions=["short", "long"],
            tp_range=(0.3, 5.0),
            sl_range=(1.0, 10.0),
            step=0.3,
            entry_conditions=["always", "rsi_oversold", "consecutive_red", "price_drop"]
        )
    else:
        print("\nRunning FULL exhaustive search...")
        results = engine.run_full_search()

    engine.print_summary(top_n=10)

    return results


if __name__ == "__main__":
    # Test with sample data
    sample_path = "/Users/chriseddisford/Downloads/KRAKEN_BTCGBP, 1.csv"

    if os.path.exists(sample_path):
        results = find_best_strategy(sample_path, quick=True)
    else:
        print(f"Sample data not found at {sample_path}")
        print("Please provide a CSV file with columns: time, open, high, low, close")
