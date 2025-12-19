"""
STRATEGY FINDER - Main Orchestration Script
============================================
The ONE script to run when you want to find profitable strategies.

This is what you asked for:
1. Load data from CSV
2. Test ALL strategy combinations (direction, TP%, SL%, entry conditions)
3. Store results in SQLite database (persistent)
4. Generate exact-match Pine Scripts
5. Create equity curve visualizations

Usage:
    python strategy_finder.py /path/to/data.csv

Author: BTCGBP ML Optimizer
Date: 2025-12-19
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exact_match_backtester import (
    ExactMatchBacktester, StrategyResult, StrategyDatabase, load_csv_data
)
from grid_search_engine import GridSearchEngine, GridSearchConfig, find_best_strategy
from exact_match_pinescript import ExactMatchPineGenerator, generate_pine_script_from_result

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Equity curves will be saved as data only.")


class StrategyFinder:
    """
    Main orchestration class for finding profitable strategies.

    This replaces the complex ML optimizer with a simple, exact-match system
    that guarantees Python backtest = TradingView results.
    """

    def __init__(
        self,
        csv_path: str,
        output_dir: str = None,
        db_path: str = None
    ):
        """
        Initialize Strategy Finder.

        Args:
            csv_path: Path to CSV file with OHLC data
            output_dir: Directory for output files (Pine Scripts, charts)
            db_path: Path to SQLite database for persistence
        """
        self.csv_path = csv_path

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'output',
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up database
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data',
                'strategies.db'
            )
        self.db = StrategyDatabase(db_path)

        # Load data
        print(f"Loading data from {csv_path}...")
        self.df = load_csv_data(csv_path)
        print(f"Loaded {len(self.df)} candles")
        print(f"Date range: {self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}")

        # Detect timeframe
        if len(self.df) > 1:
            time_diff = (self.df['time'].iloc[1] - self.df['time'].iloc[0]).total_seconds()
            if time_diff <= 60:
                self.timeframe = "1m"
            elif time_diff <= 300:
                self.timeframe = "5m"
            elif time_diff <= 900:
                self.timeframe = "15m"
            elif time_diff <= 3600:
                self.timeframe = "1h"
            elif time_diff <= 14400:
                self.timeframe = "4h"
            else:
                self.timeframe = "1d"
            print(f"Detected timeframe: {self.timeframe}")

        # Initialize engine
        self.engine = GridSearchEngine(self.df, db_path=db_path)
        self.results: List[StrategyResult] = []

    def run_quick_search(
        self,
        focus_direction: str = None,
        tp_range: tuple = (0.3, 5.0),
        sl_range: tuple = (1.0, 10.0)
    ) -> List[StrategyResult]:
        """
        Run a quick search for profitable strategies.

        Good for initial exploration before full search.

        Args:
            focus_direction: "long", "short", or None for both
            tp_range: (min, max) TP percentage
            sl_range: (min, max) SL percentage

        Returns:
            List of StrategyResult objects
        """
        print("\n" + "=" * 60)
        print("QUICK SEARCH")
        print("=" * 60)

        directions = [focus_direction] if focus_direction else ["short", "long"]

        self.results = self.engine.run_quick_search(
            directions=directions,
            tp_range=tp_range,
            sl_range=sl_range,
            step=0.5,
            entry_conditions=["always", "rsi_oversold", "consecutive_red", "price_drop"]
        )

        return self.results

    def run_full_search(
        self,
        custom_config: GridSearchConfig = None
    ) -> List[StrategyResult]:
        """
        Run exhaustive search across ALL strategy combinations.

        This is the "no strategy off the table" search you asked for.

        Args:
            custom_config: Optional custom search configuration

        Returns:
            List of StrategyResult objects
        """
        print("\n" + "=" * 60)
        print("FULL EXHAUSTIVE SEARCH")
        print("=" * 60)

        if custom_config:
            self.engine.config = custom_config

        self.results = self.engine.run_full_search(save_to_db=True)
        return self.results

    def get_top_strategies(
        self,
        n: int = 10,
        min_win_rate: float = 0,
        direction: str = None
    ) -> List[StrategyResult]:
        """
        Get top N strategies by P&L.

        Args:
            n: Number of strategies to return
            min_win_rate: Minimum win rate filter
            direction: Filter by direction

        Returns:
            List of top StrategyResult objects
        """
        return self.engine.get_top_n(n, min_win_rate=min_win_rate, direction=direction)

    def generate_pine_script(
        self,
        result: StrategyResult,
        filename: str = None
    ) -> str:
        """
        Generate Pine Script for a strategy and save to file.

        Args:
            result: StrategyResult object
            filename: Output filename (auto-generated if None)

        Returns:
            Pine Script code string
        """
        script = ExactMatchPineGenerator.generate(
            strategy_name=result.strategy_name,
            direction=result.direction,
            tp_percent=result.tp_percent,
            sl_percent=result.sl_percent,
            entry_condition=result.entry_condition,
            entry_params=result.params,
            metrics={
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl_gbp': result.total_pnl_gbp,
                'profit_factor': result.profit_factor,
                'max_drawdown_gbp': result.max_drawdown_gbp,
            }
        )

        # Save to file
        if filename is None:
            filename = f"{result.entry_condition}_{result.direction}_tp{result.tp_percent}_sl{result.sl_percent}.pine"

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(script)

        print(f"Pine Script saved to: {filepath}")
        return script

    def generate_top_pine_scripts(self, n: int = 5):
        """Generate Pine Scripts for top N strategies"""
        top = self.get_top_strategies(n)

        print(f"\nGenerating Pine Scripts for top {n} strategies...")

        for i, result in enumerate(top, 1):
            filename = f"top_{i}_{result.direction}_{result.entry_condition}_tp{result.tp_percent}_sl{result.sl_percent}.pine"
            self.generate_pine_script(result, filename)

    def create_equity_chart(
        self,
        result: StrategyResult,
        filename: str = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Create equity curve chart for a strategy.

        Args:
            result: StrategyResult object
            filename: Output filename (auto-generated if None)
            show: Whether to display the chart

        Returns:
            Path to saved chart, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            # Save equity data as JSON instead
            data_file = os.path.join(self.output_dir, f"{result.strategy_id}_equity.json")
            with open(data_file, 'w') as f:
                json.dump({
                    'strategy': result.strategy_name,
                    'equity_curve': result.equity_curve,
                    'trades': result.trades
                }, f, indent=2)
            print(f"Equity data saved to: {data_file}")
            return data_file

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Plot equity curve
        ax1 = axes[0]
        ax1.plot(result.equity_curve, color='blue', linewidth=1.5)
        ax1.fill_between(range(len(result.equity_curve)), result.equity_curve,
                         result.equity_curve[0], alpha=0.3,
                         color='green' if result.total_pnl_gbp > 0 else 'red')
        ax1.axhline(y=result.equity_curve[0], color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f"{result.strategy_name}\n{result.direction.upper()} | TP {result.tp_percent}% | SL {result.sl_percent}%",
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel("Equity (£)")
        ax1.grid(True, alpha=0.3)

        # Add stats text
        stats_text = f"Trades: {result.total_trades} | Win Rate: {result.win_rate:.1f}%\n"
        stats_text += f"P&L: £{result.total_pnl_gbp:.2f} | PF: {result.profit_factor:.2f}\n"
        stats_text += f"Max DD: £{result.max_drawdown_gbp:.2f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot trade P&L bars
        ax2 = axes[1]
        if result.trades:
            pnls = [t.get('pnl_gbp', 0) for t in result.trades if t.get('pnl_gbp') is not None]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel("Trade P&L (£)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

        # Plot price with trade markers
        ax3 = axes[2]
        if len(self.df) > 0:
            ax3.plot(self.df['close'].values, color='black', linewidth=0.5, alpha=0.7)
            ax3.set_ylabel("Price (£)")
            ax3.set_xlabel("Bar #")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        if filename is None:
            filename = f"{result.strategy_id}_equity.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        print(f"Equity chart saved to: {filepath}")
        return filepath

    def create_summary_report(self, n_top: int = 10) -> str:
        """
        Create comprehensive summary report.

        Args:
            n_top: Number of top strategies to include

        Returns:
            Path to report file
        """
        report_path = os.path.join(self.output_dir, "strategy_report.md")

        with open(report_path, 'w') as f:
            f.write(f"# Strategy Search Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Data File:** {self.csv_path}\n")
            f.write(f"**Candles:** {len(self.df)}\n")
            f.write(f"**Date Range:** {self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}\n")
            f.write(f"**Timeframe:** {self.timeframe}\n\n")

            f.write("---\n\n")

            # Summary stats
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Strategies Tested:** {len(self.results)}\n")
            profitable = [r for r in self.results if r.total_pnl_gbp > 0]
            f.write(f"- **Profitable Strategies:** {len(profitable)} ({len(profitable)/len(self.results)*100:.1f}%)\n")

            if profitable:
                avg_profit = sum(r.total_pnl_gbp for r in profitable) / len(profitable)
                f.write(f"- **Average Profit (profitable only):** £{avg_profit:.2f}\n")

            # Best by direction
            longs = [r for r in self.results if r.direction == "long" and r.total_pnl_gbp > 0]
            shorts = [r for r in self.results if r.direction == "short" and r.total_pnl_gbp > 0]

            f.write(f"- **Profitable LONG strategies:** {len(longs)}\n")
            f.write(f"- **Profitable SHORT strategies:** {len(shorts)}\n\n")

            # Top strategies table
            f.write("## Top Performing Strategies\n\n")
            f.write("| Rank | Direction | Entry | TP% | SL% | Trades | Win Rate | P&L | PF |\n")
            f.write("|------|-----------|-------|-----|-----|--------|----------|------|----|\n")

            for i, r in enumerate(self.get_top_strategies(n_top), 1):
                f.write(f"| {i} | {r.direction.upper()} | {r.entry_condition} | "
                        f"{r.tp_percent}% | {r.sl_percent}% | {r.total_trades} | "
                        f"{r.win_rate:.1f}% | £{r.total_pnl_gbp:.2f} | {r.profit_factor:.2f} |\n")

            f.write("\n---\n\n")

            # Best by entry condition
            f.write("## Best Strategy per Entry Condition\n\n")
            best_by_entry = self.engine.get_best_by_category()

            for entry_cond, result in sorted(best_by_entry.items(), key=lambda x: -x[1].total_pnl_gbp):
                f.write(f"### {entry_cond}\n")
                f.write(f"- **Direction:** {result.direction.upper()}\n")
                f.write(f"- **TP/SL:** {result.tp_percent}% / {result.sl_percent}%\n")
                f.write(f"- **Trades:** {result.total_trades}\n")
                f.write(f"- **Win Rate:** {result.win_rate:.1f}%\n")
                f.write(f"- **P&L:** £{result.total_pnl_gbp:.2f}\n")
                f.write(f"- **Profit Factor:** {result.profit_factor:.2f}\n\n")

            # Pine script info
            f.write("---\n\n")
            f.write("## Generated Pine Scripts\n\n")
            f.write("Pine Script files are saved in the output directory. ")
            f.write("Import them into TradingView to verify the backtest results.\n\n")
            f.write("**Important:** Make sure to:\n")
            f.write("1. Use the EXACT same trading pair (BTCGBP, Kraken)\n")
            f.write("2. Use the EXACT same timeframe\n")
            f.write("3. Set the date range to match the CSV data\n\n")

        print(f"Report saved to: {report_path}")
        return report_path

    def run_complete_analysis(
        self,
        quick_first: bool = True,
        full_search: bool = False,
        generate_scripts: bool = True,
        generate_charts: bool = True,
        top_n: int = 10
    ):
        """
        Run complete analysis workflow.

        This is the main function to call for a full analysis.

        Args:
            quick_first: Start with quick search
            full_search: Run exhaustive full search
            generate_scripts: Generate Pine Scripts for top strategies
            generate_charts: Generate equity curve charts
            top_n: Number of top strategies to process
        """
        print("\n" + "=" * 80)
        print("STRATEGY FINDER - Complete Analysis")
        print("=" * 80)
        print(f"Data: {self.csv_path}")
        print(f"Output: {self.output_dir}")
        print("=" * 80 + "\n")

        # Run search
        if quick_first:
            print("Phase 1: Quick Search...")
            self.run_quick_search()
            self.engine.print_summary(top_n)

        if full_search:
            print("\nPhase 2: Full Exhaustive Search...")
            self.run_full_search()
            self.engine.print_summary(top_n)

        # Generate outputs
        if generate_scripts:
            print(f"\nGenerating Pine Scripts for top {top_n} strategies...")
            self.generate_top_pine_scripts(top_n)

        if generate_charts:
            print(f"\nGenerating equity charts for top {min(5, top_n)} strategies...")
            for result in self.get_top_strategies(min(5, top_n)):
                self.create_equity_chart(result)

        # Generate report
        print("\nGenerating summary report...")
        self.create_summary_report(top_n)

        # Export results
        results_path = os.path.join(self.output_dir, "all_results.json")
        self.engine.export_results_json(results_path)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Strategies tested: {len(self.results)}")
        print(f"Profitable strategies: {len([r for r in self.results if r.total_pnl_gbp > 0])}")

        if self.results:
            best = self.results[0]
            print(f"\nBEST STRATEGY:")
            print(f"  {best.strategy_name}")
            print(f"  Direction: {best.direction.upper()}")
            print(f"  TP: {best.tp_percent}% | SL: {best.sl_percent}%")
            print(f"  Trades: {best.total_trades} | Win Rate: {best.win_rate:.1f}%")
            print(f"  Total P&L: £{best.total_pnl_gbp:.2f}")
            print(f"  Pine Script: {self.output_dir}/top_1_*.pine")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Find profitable trading strategies from OHLC data"
    )
    parser.add_argument("csv_path", help="Path to CSV file with OHLC data")
    parser.add_argument("--output", "-o", help="Output directory", default=None)
    parser.add_argument("--quick", action="store_true", help="Run quick search only")
    parser.add_argument("--full", action="store_true", help="Run full exhaustive search")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of top strategies")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-scripts", action="store_true", help="Skip Pine Script generation")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    # Create finder and run analysis
    finder = StrategyFinder(args.csv_path, output_dir=args.output)

    finder.run_complete_analysis(
        quick_first=True,
        full_search=args.full,
        generate_scripts=not args.no_scripts,
        generate_charts=not args.no_charts,
        top_n=args.top
    )


if __name__ == "__main__":
    main()
