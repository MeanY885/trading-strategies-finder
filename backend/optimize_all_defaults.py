"""
Optimize All Strategy Defaults
==============================
Runs optimization for ALL strategies and outputs optimized parameters
that should be embedded as defaults in the Pine Script generator.

Usage:
    cd btcgbp-ml-optimizer/backend
    python optimize_all_defaults.py

Output:
    Creates optimized_defaults.json with best parameters for each strategy
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from unified_optimizer import (
    StrategyRegistry,
    UnifiedBacktester,
    PARAM_RANGES
)


def run_strategy_optimization(backtester, strategy_name: str, category: str,
                               param_names: list, n_trials: int = 50) -> dict:
    """
    Run Optuna optimization for a single strategy.
    Returns best parameters found.
    """

    def objective(trial):
        params = {}

        # Sample strategy-specific parameters
        for param_name in param_names:
            if param_name in PARAM_RANGES:
                min_val, max_val = PARAM_RANGES[param_name]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    step = 0.1 if max_val - min_val > 1 else 0.05
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)

        # Always optimize risk management
        params['sl_atr_mult'] = trial.suggest_float('sl_atr_mult', 0.5, 5.0, step=0.1)
        params['tp_ratio'] = trial.suggest_float('tp_ratio', 0.5, 5.0, step=0.1)

        try:
            result = backtester.run_backtest(strategy_name, category, params)
            return result.composite_score
        except Exception:
            return -1000

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_trial and study.best_trial.value > -1000:
        best_params = dict(study.best_params)
        # Round floats for cleaner output
        for k, v in best_params.items():
            if isinstance(v, float):
                best_params[k] = round(v, 1)
        return {
            'params': best_params,
            'score': round(study.best_trial.value, 2)
        }
    return None


def main():
    print("=" * 70)
    print("STRATEGY PARAMETER OPTIMIZER")
    print("Finding optimal defaults for ALL strategies")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"

    # Use the largest dataset available
    data_files = list(data_dir.glob("*.csv"))
    data_files = [f for f in data_files if f.stat().st_size > 50000]  # >50KB

    if not data_files:
        print("ERROR: No suitable data files found")
        return

    # Pick the one with most data
    data_file = max(data_files, key=lambda f: f.stat().st_size)
    print(f"\nUsing data file: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} candles")

    # Ensure proper column names
    df.columns = [c.lower() for c in df.columns]
    if 'time' not in df.columns and 'datetime' in df.columns:
        df['time'] = df['datetime']
    if 'time' not in df.columns and 'date' in df.columns:
        df['time'] = df['date']

    # Create backtester
    print("\nCalculating indicators (this may take a minute)...")
    backtester = UnifiedBacktester(df, capital=1000.0, risk_percent=2.0)

    # Get all strategies
    strategies = StrategyRegistry.get_all_strategies()
    print(f"\nOptimizing {len(strategies)} strategies...")
    print("-" * 70)

    optimized_defaults = {}
    n_trials = 30  # Trials per strategy (balance speed vs quality)

    for i, strategy in enumerate(strategies):
        name = strategy['name']
        display_name = strategy['display_name']
        category = StrategyRegistry.CATEGORIES.get(strategy['category'], strategy['category'])
        params = strategy.get('params', [])

        print(f"[{i+1}/{len(strategies)}] {display_name}...", end=" ", flush=True)

        try:
            result = run_strategy_optimization(
                backtester, name, category, params, n_trials
            )

            if result:
                optimized_defaults[name] = result
                print(f"Score: {result['score']:.2f}")
            else:
                print("No valid result")

        except Exception as e:
            print(f"Error: {str(e)[:50]}")

    # Save results
    output_file = Path(__file__).parent / "optimized_defaults.json"
    with open(output_file, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'data_file': data_file.name,
            'n_trials_per_strategy': n_trials,
            'strategies': optimized_defaults
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"COMPLETE! Optimized {len(optimized_defaults)} strategies")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # Print summary of top strategies
    print("\nTOP 10 STRATEGIES BY SCORE:")
    sorted_strats = sorted(
        optimized_defaults.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:10]

    for name, data in sorted_strats:
        print(f"  {name}: {data['score']:.2f}")
        print(f"    Params: {data['params']}")


if __name__ == "__main__":
    main()
