"""
ML Optimizer using Optuna for Bayesian Optimization
"""
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from strategy import SidewaysScalperStrategy


class StrategyOptimizer:
    """
    Uses Optuna (Bayesian optimization) to find optimal strategy parameters
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize optimizer with historical data
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df
        self.best_params = None
        self.best_score = None
        self.all_results = []
        
        # Split data into train (70%) and validation (30%) sets
        split_idx = int(len(df) * 0.7)
        self.train_df = df.iloc[:split_idx].copy()
        self.val_df = df.iloc[split_idx:].copy()
        
        print(f"Training data: {len(self.train_df)} candles")
        print(f"Validation data: {len(self.val_df)} candles")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - returns a score to maximize
        """
        # Sample parameters
        params = {
            # ADX parameters
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 40),
            "adx_emergency": trial.suggest_int("adx_emergency", 30, 50),
            
            # Bollinger Bands
            "bb_length": trial.suggest_int("bb_length", 10, 40),
            "bb_mult": trial.suggest_float("bb_mult", 1.2, 3.5, step=0.1),
            
            # RSI
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 85),
            
            # Risk Management
            "sl_fixed": trial.suggest_float("sl_fixed", 30, 400, step=10),
            "tp_ratio": trial.suggest_float("tp_ratio", 0.8, 4.0, step=0.1),
        }
        
        # Ensure adx_emergency > adx_threshold
        if params["adx_emergency"] <= params["adx_threshold"]:
            params["adx_emergency"] = params["adx_threshold"] + 5
        
        # Ensure rsi_overbought > rsi_oversold + 20
        if params["rsi_overbought"] <= params["rsi_oversold"] + 20:
            params["rsi_overbought"] = params["rsi_oversold"] + 25
        
        # Run strategy on training data
        strategy = SidewaysScalperStrategy(
            adx_threshold=params["adx_threshold"],
            adx_emergency=params["adx_emergency"],
            bb_length=params["bb_length"],
            bb_mult=params["bb_mult"],
            rsi_oversold=params["rsi_oversold"],
            rsi_overbought=params["rsi_overbought"],
            sl_fixed=params["sl_fixed"],
            tp_ratio=params["tp_ratio"]
        )
        
        train_results = strategy.backtest(self.train_df)
        
        # Calculate composite score
        # We want to maximize:
        # - Profit factor (higher is better)
        # - Win rate (higher is better)
        # - Total PnL (higher is better)
        # - Number of trades (need enough trades for statistical significance)
        # While minimizing:
        # - Max drawdown
        
        # Penalize if too few trades
        if train_results["total_trades"] < 10:
            return -100
        
        # Penalize if profit factor is too low
        profit_factor = train_results["profit_factor"]
        if profit_factor < 0.5:
            return -50
        
        # Calculate composite score
        win_rate_score = train_results["win_rate"] / 100  # 0 to 1
        pf_score = min(profit_factor, 3) / 3  # Cap at 3, normalize to 0-1
        
        # Normalize PnL (use sigmoid-like function)
        pnl = train_results["total_pnl"]
        pnl_score = np.tanh(pnl / 500)  # -1 to 1
        
        # Trade frequency score (want 20-100 trades per month ideally)
        months = len(self.train_df) / (4 * 24 * 30)  # Approximate months
        trades_per_month = train_results["total_trades"] / max(months, 1)
        trade_freq_score = min(trades_per_month / 50, 1)  # Cap at 1
        
        # Drawdown penalty
        dd_penalty = min(train_results["max_drawdown"] / 200, 1)  # Penalty increases with DD
        
        # Composite score
        score = (
            0.35 * pf_score +
            0.25 * win_rate_score +
            0.25 * pnl_score +
            0.10 * trade_freq_score -
            0.05 * dd_penalty
        )
        
        # Store result
        self.all_results.append({
            "trial": trial.number,
            "params": params.copy(),
            "train_results": {
                "total_trades": train_results["total_trades"],
                "win_rate": train_results["win_rate"],
                "profit_factor": train_results["profit_factor"],
                "total_pnl": train_results["total_pnl"],
                "max_drawdown": train_results["max_drawdown"]
            },
            "score": score
        })
        
        return score
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: int = 300,
        callback: Optional[Callable] = None
    ) -> Tuple[Dict, float, List]:
        """
        Run optimization
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            callback: Optional callback function(trial_num, total, params, score)
            
        Returns:
            Tuple of (best_params, best_score, all_results)
        """
        # Suppress Optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create Optuna study
        sampler = TPESampler(seed=42)  # Reproducible
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        # Track progress for callback
        self._callback = callback
        self._n_trials = n_trials
        
        # Custom callback to track progress - called after EVERY trial
        def optuna_callback(study, trial):
            if callback:
                try:
                    callback(
                        trial.number + 1,
                        n_trials,
                        dict(trial.params) if trial.params else {},
                        trial.value if trial.value is not None else 0
                    )
                except Exception as e:
                    print(f"Callback error: {e}")
        
        print(f"Starting optimization with {n_trials} trials...")
        
        # Run optimization - disable progress bar as we use our own callback
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[optuna_callback],
            show_progress_bar=False  # Disable built-in progress bar
        )
        
        # Get best parameters
        best_trial = study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        
        # Ensure constraints
        if self.best_params["adx_emergency"] <= self.best_params["adx_threshold"]:
            self.best_params["adx_emergency"] = self.best_params["adx_threshold"] + 5
        if self.best_params["rsi_overbought"] <= self.best_params["rsi_oversold"] + 20:
            self.best_params["rsi_overbought"] = self.best_params["rsi_oversold"] + 25
        
        # Validate on held-out data
        print("\n" + "="*50)
        print("VALIDATION RESULTS (Out-of-sample)")
        print("="*50)
        
        val_strategy = SidewaysScalperStrategy(
            adx_threshold=self.best_params["adx_threshold"],
            adx_emergency=self.best_params["adx_emergency"],
            bb_length=self.best_params["bb_length"],
            bb_mult=self.best_params["bb_mult"],
            rsi_oversold=self.best_params["rsi_oversold"],
            rsi_overbought=self.best_params["rsi_overbought"],
            sl_fixed=self.best_params["sl_fixed"],
            tp_ratio=self.best_params["tp_ratio"]
        )
        
        val_results = val_strategy.backtest(self.val_df)
        
        print(f"Validation Trades: {val_results['total_trades']}")
        print(f"Validation Win Rate: {val_results['win_rate']}%")
        print(f"Validation Profit Factor: {val_results['profit_factor']}")
        print(f"Validation Total PnL: £{val_results['total_pnl']}")
        print(f"Validation Max Drawdown: £{val_results['max_drawdown']}")
        
        # Add validation results to best params
        self.best_params["validation_results"] = val_results
        
        return self.best_params, self.best_score, self.all_results
    
    def get_parameter_importance(self, study: optuna.Study) -> Dict[str, float]:
        """Get importance of each parameter"""
        try:
            importances = optuna.importance.get_param_importances(study)
            return dict(importances)
        except:
            return {}


class WalkForwardOptimizer:
    """
    Walk-forward optimization for more robust parameter selection
    """
    
    def __init__(self, df: pd.DataFrame, n_splits: int = 5):
        self.df = df
        self.n_splits = n_splits
    
    def optimize(self, n_trials_per_split: int = 50) -> Dict:
        """
        Run walk-forward optimization
        
        Splits data into windows and optimizes on each,
        then validates on the next window
        """
        results = []
        window_size = len(self.df) // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Training window
            train_start = i * window_size
            train_end = (i + 2) * window_size
            
            # Validation window
            val_start = train_end
            val_end = min(val_start + window_size, len(self.df))
            
            train_df = self.df.iloc[train_start:train_end].copy()
            val_df = self.df.iloc[val_start:val_end].copy()
            
            print(f"\nWalk-Forward Split {i+1}/{self.n_splits}")
            print(f"Train: {train_df['time'].min()} to {train_df['time'].max()}")
            print(f"Val: {val_df['time'].min()} to {val_df['time'].max()}")
            
            # Optimize on training data
            optimizer = StrategyOptimizer(pd.concat([train_df, val_df]))
            optimizer.train_df = train_df
            optimizer.val_df = val_df
            
            best_params, best_score, _ = optimizer.optimize(n_trials=n_trials_per_split)
            
            results.append({
                "split": i + 1,
                "params": best_params,
                "score": best_score,
                "validation": best_params.get("validation_results", {})
            })
        
        # Average the parameters across all splits
        avg_params = self._average_params(results)
        
        return {
            "splits": results,
            "averaged_params": avg_params
        }
    
    def _average_params(self, results: List[Dict]) -> Dict:
        """Average parameters across all splits"""
        param_names = ["adx_threshold", "adx_emergency", "bb_length", "bb_mult",
                      "rsi_oversold", "rsi_overbought", "sl_fixed", "tp_ratio"]
        
        avg = {}
        for param in param_names:
            values = [r["params"][param] for r in results if param in r["params"]]
            if values:
                avg[param] = round(np.mean(values), 2) if isinstance(values[0], float) else int(np.mean(values))
        
        return avg

