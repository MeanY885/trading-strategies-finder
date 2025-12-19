"""
Ensemble Strategy Optimizer
============================
Combines multiple optimization methods for robust strategy discovery:
- Random Search (unbiased exploration)
- Bayesian TPE (smart directed search)  
- CMA-ES (best for continuous parameters)

Only strategies that multiple methods agree on make the final cut.
Focus: Maximum Profit + Smooth Equity Curve
Speed: Not a concern - exhaustive testing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
import concurrent.futures
from collections import defaultdict

from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, MACD, CCIIndicator


@dataclass
class TradeRecord:
    """Single trade for backtest"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str


@dataclass  
class StrategyCandidate:
    """A strategy + parameter combination found by an optimizer"""
    strategy_type: str
    params: Dict[str, Any]
    
    # Performance metrics
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    max_drawdown: float = 0.0
    equity_r_squared: float = 0.0  # Smoothness of equity curve
    recovery_factor: float = 0.0   # PnL / Max Drawdown
    
    # Composite score
    composite_score: float = 0.0
    
    # Which methods found this
    found_by: List[str] = field(default_factory=list)
    
    # Trade details for equity curve
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class UnifiedBacktester:
    """
    Unified backtester that can test any strategy type with any parameters.
    Calculates comprehensive metrics including equity curve smoothness.
    """
    
    # All strategy types we support
    STRATEGY_TYPES = [
        "bb_rsi",           # Bollinger Bands + RSI
        "bb_stoch",         # Bollinger Bands + Stochastic  
        "keltner_rsi",      # Keltner Channel + RSI
        "stoch_extreme",    # Stochastic Extremes
        "rsi_extreme",      # RSI Extremes
        "williams_r",       # Williams %R
        "cci_extreme",      # CCI Extremes
        "bb_cci",           # BB + CCI combo
        "stoch_rsi",        # Stochastic + RSI combo
        "triple_confirm",   # BB + RSI + Stoch triple
        "macd_bb",          # MACD + BB combo
        "adx_rsi",          # ADX trend + RSI
        "squeeze_breakout", # BB Squeeze breakout
        "mean_reversion",   # Pure mean reversion
        "momentum_burst",   # Momentum burst strategy
    ]
    
    # WIDE parameter ranges - test everything
    PARAM_RANGES = {
        # Bollinger Bands
        "bb_length": (5, 50),
        "bb_mult": (0.5, 4.0),
        
        # RSI
        "rsi_length": (3, 30),
        "rsi_oversold": (5, 45),
        "rsi_overbought": (55, 95),
        
        # Stochastic
        "stoch_k": (3, 21),
        "stoch_d": (2, 9),
        "stoch_oversold": (5, 35),
        "stoch_overbought": (65, 95),
        
        # ADX
        "adx_length": (5, 30),
        "adx_threshold": (10, 50),
        
        # CCI
        "cci_length": (5, 30),
        "cci_threshold": (50, 200),
        
        # Williams %R
        "willr_length": (5, 30),
        "willr_oversold": (-95, -70),
        "willr_overbought": (-30, -5),
        
        # Keltner
        "kc_length": (5, 30),
        "kc_mult": (0.5, 3.0),
        
        # MACD
        "macd_fast": (5, 20),
        "macd_slow": (15, 40),
        "macd_signal": (3, 15),
        
        # Risk Management
        "sl_atr_mult": (0.5, 5.0),
        "tp_ratio": (0.5, 5.0),
    }
    
    def __init__(self, df: pd.DataFrame, capital: float = 1000.0, 
                 risk_percent: float = 2.0):
        self.df = df.copy()
        self.capital = capital
        self.risk_percent = risk_percent
        
        # Pre-calculate ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        self.df['atr'] = atr.average_true_range()
        
    def calculate_indicators(self, params: Dict) -> pd.DataFrame:
        """Calculate all indicators based on parameters"""
        df = self.df.copy()
        
        # Bollinger Bands
        bb_len = params.get('bb_length', 20)
        bb_mult = params.get('bb_mult', 2.0)
        bb = BollingerBands(df['close'], window=bb_len, window_dev=bb_mult)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = bb.bollinger_wband()
        
        # RSI
        rsi_len = params.get('rsi_length', 14)
        df['rsi'] = RSIIndicator(df['close'], window=rsi_len).rsi()
        
        # Stochastic
        stoch_k = params.get('stoch_k', 14)
        stoch_d = params.get('stoch_d', 3)
        stoch = StochasticOscillator(df['high'], df['low'], df['close'],
                                      window=stoch_k, smooth_window=stoch_d)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx_len = params.get('adx_length', 14)
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=adx_len)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # CCI
        cci_len = params.get('cci_length', 20)
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], 
                                  window=cci_len).cci()
        
        # Williams %R
        willr_len = params.get('willr_length', 14)
        df['willr'] = WilliamsRIndicator(df['high'], df['low'], df['close'],
                                          lbp=willr_len).williams_r()
        
        # Keltner Channel
        kc_len = params.get('kc_length', 20)
        kc = KeltnerChannel(df['high'], df['low'], df['close'], window=kc_len)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        macd = MACD(df['close'], window_slow=macd_slow, window_fast=macd_fast,
                    window_sign=macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        return df
    
    def get_signals(self, strategy_type: str, df: pd.DataFrame, 
                    params: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate long/short signals based on strategy type.
        Returns: (long_signal, short_signal, sideways_filter)
        """
        rsi_os = params.get('rsi_oversold', 30)
        rsi_ob = params.get('rsi_overbought', 70)
        stoch_os = params.get('stoch_oversold', 20)
        stoch_ob = params.get('stoch_overbought', 80)
        adx_thresh = params.get('adx_threshold', 25)
        cci_thresh = params.get('cci_threshold', 100)
        willr_os = params.get('willr_oversold', -80)
        willr_ob = params.get('willr_overbought', -20)
        
        # Default sideways filter (ADX below threshold)
        sideways = df['adx'] < adx_thresh
        
        if strategy_type == "bb_rsi":
            long_sig = (df['close'] <= df['bb_lower']) & (df['rsi'] < rsi_os)
            short_sig = (df['close'] >= df['bb_upper']) & (df['rsi'] > rsi_ob)
            
        elif strategy_type == "bb_stoch":
            long_sig = (df['close'] <= df['bb_lower']) & (df['stoch_k'] < stoch_os)
            short_sig = (df['close'] >= df['bb_upper']) & (df['stoch_k'] > stoch_ob)
            
        elif strategy_type == "keltner_rsi":
            long_sig = (df['close'] <= df['kc_lower']) & (df['rsi'] < rsi_os)
            short_sig = (df['close'] >= df['kc_upper']) & (df['rsi'] > rsi_ob)
            
        elif strategy_type == "stoch_extreme":
            long_sig = (df['stoch_k'] < stoch_os) & (df['stoch_d'] < stoch_os)
            short_sig = (df['stoch_k'] > stoch_ob) & (df['stoch_d'] > stoch_ob)
            
        elif strategy_type == "rsi_extreme":
            long_sig = df['rsi'] < rsi_os
            short_sig = df['rsi'] > rsi_ob
            
        elif strategy_type == "williams_r":
            long_sig = df['willr'] < willr_os
            short_sig = df['willr'] > willr_ob
            
        elif strategy_type == "cci_extreme":
            long_sig = df['cci'] < -cci_thresh
            short_sig = df['cci'] > cci_thresh
            
        elif strategy_type == "bb_cci":
            long_sig = (df['close'] <= df['bb_lower']) & (df['cci'] < -cci_thresh)
            short_sig = (df['close'] >= df['bb_upper']) & (df['cci'] > cci_thresh)
            
        elif strategy_type == "stoch_rsi":
            long_sig = (df['stoch_k'] < stoch_os) & (df['rsi'] < rsi_os + 10)
            short_sig = (df['stoch_k'] > stoch_ob) & (df['rsi'] > rsi_ob - 10)
            
        elif strategy_type == "triple_confirm":
            long_sig = ((df['close'] <= df['bb_lower']) & 
                       (df['rsi'] < rsi_os) & 
                       (df['stoch_k'] < stoch_os))
            short_sig = ((df['close'] >= df['bb_upper']) & 
                        (df['rsi'] > rsi_ob) & 
                        (df['stoch_k'] > stoch_ob))
            
        elif strategy_type == "macd_bb":
            long_sig = ((df['close'] <= df['bb_lower']) & 
                       (df['macd_hist'] > df['macd_hist'].shift(1)))
            short_sig = ((df['close'] >= df['bb_upper']) & 
                        (df['macd_hist'] < df['macd_hist'].shift(1)))
            
        elif strategy_type == "adx_rsi":
            # Trend following with RSI confirmation
            sideways = pd.Series([True] * len(df), index=df.index)  # No sideways filter
            long_sig = ((df['adx'] > adx_thresh) & 
                       (df['di_plus'] > df['di_minus']) & 
                       (df['rsi'] > 50) & (df['rsi'] < 70))
            short_sig = ((df['adx'] > adx_thresh) & 
                        (df['di_minus'] > df['di_plus']) & 
                        (df['rsi'] < 50) & (df['rsi'] > 30))
            
        elif strategy_type == "squeeze_breakout":
            avg_width = df['bb_width'].rolling(50).mean()
            in_squeeze = df['bb_width'] < avg_width * 0.75
            squeeze_release = in_squeeze.shift(1) & ~in_squeeze
            sideways = pd.Series([True] * len(df), index=df.index)
            long_sig = squeeze_release & (df['close'] > df['bb_mid'])
            short_sig = squeeze_release & (df['close'] < df['bb_mid'])
            
        elif strategy_type == "mean_reversion":
            # Pure mean reversion - far from mean
            z_score = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            long_sig = z_score < -2
            short_sig = z_score > 2
            
        elif strategy_type == "momentum_burst":
            # Strong momentum continuation
            sideways = pd.Series([True] * len(df), index=df.index)
            rsi_momentum = df['rsi'] > df['rsi'].shift(1)
            long_sig = ((df['rsi'] > 60) & (df['rsi'] < 80) & 
                       rsi_momentum & (df['macd_hist'] > 0))
            short_sig = ((df['rsi'] < 40) & (df['rsi'] > 20) & 
                        ~rsi_momentum & (df['macd_hist'] < 0))
        else:
            # Default fallback
            long_sig = pd.Series([False] * len(df), index=df.index)
            short_sig = pd.Series([False] * len(df), index=df.index)
        
        return long_sig, short_sig, sideways
    
    def run_backtest(self, strategy_type: str, params: Dict) -> StrategyCandidate:
        """
        Run full backtest for a strategy + params combination.
        Returns comprehensive metrics including equity curve smoothness.
        """
        # Calculate indicators
        df = self.calculate_indicators(params)
        
        # Get signals
        long_sig, short_sig, sideways = self.get_signals(strategy_type, df, params)
        
        # Risk parameters
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        
        trades = []
        position = None
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # Skip if indicators not ready
            if pd.isna(row['rsi']) or pd.isna(row['atr']):
                continue
            
            # Manage existing position
            if position is not None:
                exit_price = None
                exit_reason = None
                
                atr = position['atr']
                sl_dist = atr * sl_mult
                tp_dist = sl_dist * tp_ratio
                
                if position['direction'] == 'long':
                    sl_price = position['entry_price'] - sl_dist
                    tp_price = position['entry_price'] + tp_dist
                    
                    if row['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                else:
                    sl_price = position['entry_price'] + sl_dist
                    tp_price = position['entry_price'] - tp_dist
                    
                    if row['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                
                if exit_price:
                    # Calculate PnL with position sizing
                    risk_amount = self.capital * (self.risk_percent / 100)
                    pos_size = risk_amount / sl_dist if sl_dist > 0 else 0
                    
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) * pos_size
                    else:
                        pnl = (position['entry_price'] - exit_price) * pos_size
                    
                    trades.append(TradeRecord(
                        entry_time=position['entry_time'],
                        exit_time=row['time'],
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        pnl=pnl,
                        exit_reason=exit_reason
                    ))
                    position = None
            
            # Check for new entry
            if position is None:
                # Apply sideways filter
                if not sideways.iloc[i]:
                    continue
                
                if long_sig.iloc[i]:
                    position = {
                        'direction': 'long',
                        'entry_time': row['time'],
                        'entry_price': row['close'],
                        'atr': row['atr']
                    }
                elif short_sig.iloc[i]:
                    position = {
                        'direction': 'short',
                        'entry_time': row['time'],
                        'entry_price': row['close'],
                        'atr': row['atr']
                    }
        
        # Calculate metrics
        return self._calculate_metrics(strategy_type, params, trades)
    
    def _calculate_metrics(self, strategy_type: str, params: Dict,
                          trades: List[TradeRecord]) -> StrategyCandidate:
        """Calculate comprehensive metrics including equity smoothness"""
        
        candidate = StrategyCandidate(
            strategy_type=strategy_type,
            params=params.copy(),
            trades=trades
        )
        
        if len(trades) < 5:
            candidate.composite_score = -100
            return candidate
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        candidate.total_trades = len(trades)
        candidate.total_pnl = sum(pnls)
        candidate.win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        candidate.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10 if gross_profit > 0 else 0)
        
        # Equity curve
        equity = np.cumsum(pnls)
        candidate.equity_curve = equity.tolist()
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        candidate.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Recovery factor
        candidate.recovery_factor = candidate.total_pnl / (candidate.max_drawdown + 1)
        
        # Equity curve smoothness (R² of linear fit)
        if len(equity) > 1:
            x = np.arange(len(equity))
            slope, intercept = np.polyfit(x, equity, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((equity - predicted) ** 2)
            ss_tot = np.sum((equity - np.mean(equity)) ** 2)
            candidate.equity_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Penalize negative slope
            if slope < 0:
                candidate.equity_r_squared *= -1
        
        # Composite score (what we optimize for)
        # Focus: Profit + Smooth equity curve
        pf_score = min(candidate.profit_factor / 3, 1)  # Cap at 3
        wr_score = candidate.win_rate / 100
        eq_score = max(0, candidate.equity_r_squared)  # 0-1
        rf_score = min(candidate.recovery_factor / 10, 1)
        pnl_score = np.tanh(candidate.total_pnl / 1000)  # Normalize
        
        # Trade frequency - want enough trades but not too few
        trades_per_month = candidate.total_trades / max(len(self.df) / (4 * 24 * 30), 1)
        freq_score = min(trades_per_month / 30, 1) if trades_per_month > 5 else trades_per_month / 10
        
        candidate.composite_score = (
            0.30 * pf_score +      # Profit factor
            0.20 * wr_score +      # Win rate
            0.25 * eq_score +      # Equity smoothness (important!)
            0.15 * rf_score +      # Recovery factor
            0.05 * pnl_score +     # Raw PnL
            0.05 * freq_score      # Trade frequency
        )
        
        return candidate


class EnsembleOptimizer:
    """
    Main ensemble optimizer that combines multiple optimization methods.
    
    Methods:
    - Random Search (unbiased exploration)
    - Bayesian TPE (directed search)
    - CMA-ES (continuous optimization)
    
    Only strategies found by multiple methods make the final cut.
    """
    
    def __init__(self, df: pd.DataFrame, capital: float = 1000.0,
                 risk_percent: float = 2.0, status_callback: Optional[Dict] = None):
        self.df = df
        self.capital = capital
        self.risk_percent = risk_percent
        self.status = status_callback or {}
        
        # Split data: 70% train, 30% validation
        split_idx = int(len(df) * 0.7)
        self.train_df = df.iloc[:split_idx].copy()
        self.val_df = df.iloc[split_idx:].copy()
        
        self.backtester = UnifiedBacktester(self.train_df, capital, risk_percent)
        self.val_backtester = UnifiedBacktester(self.val_df, capital, risk_percent)
        
        # Results from each method
        self.random_results: List[StrategyCandidate] = []
        self.bayesian_results: List[StrategyCandidate] = []
        self.cmaes_results: List[StrategyCandidate] = []
        
        # Final consensus results
        self.consensus_results: List[StrategyCandidate] = []
    
    def _update_status(self, message: str, progress: int):
        if self.status:
            self.status['message'] = message
            self.status['progress'] = progress
    
    def _create_trial_params(self, trial: optuna.Trial, strategy_type: str) -> Dict:
        """Create parameters from Optuna trial"""
        params = {'strategy_type': strategy_type}
        
        # Get relevant params based on strategy type
        if strategy_type in ['bb_rsi', 'bb_stoch', 'bb_cci', 'triple_confirm', 'macd_bb']:
            params['bb_length'] = trial.suggest_int('bb_length', 5, 50)
            params['bb_mult'] = trial.suggest_float('bb_mult', 0.5, 4.0, step=0.1)
        
        if strategy_type in ['bb_rsi', 'keltner_rsi', 'rsi_extreme', 'stoch_rsi', 
                             'triple_confirm', 'adx_rsi', 'mean_reversion', 'momentum_burst']:
            params['rsi_length'] = trial.suggest_int('rsi_length', 3, 30)
            params['rsi_oversold'] = trial.suggest_int('rsi_oversold', 5, 45)
            params['rsi_overbought'] = trial.suggest_int('rsi_overbought', 55, 95)
        
        if strategy_type in ['bb_stoch', 'stoch_extreme', 'stoch_rsi', 'triple_confirm']:
            params['stoch_k'] = trial.suggest_int('stoch_k', 3, 21)
            params['stoch_d'] = trial.suggest_int('stoch_d', 2, 9)
            params['stoch_oversold'] = trial.suggest_int('stoch_oversold', 5, 35)
            params['stoch_overbought'] = trial.suggest_int('stoch_overbought', 65, 95)
        
        if strategy_type in ['keltner_rsi']:
            params['kc_length'] = trial.suggest_int('kc_length', 5, 30)
            params['kc_mult'] = trial.suggest_float('kc_mult', 0.5, 3.0, step=0.1)
        
        if strategy_type in ['adx_rsi']:
            params['adx_length'] = trial.suggest_int('adx_length', 5, 30)
        
        params['adx_threshold'] = trial.suggest_int('adx_threshold', 10, 50)
        
        if strategy_type in ['cci_extreme', 'bb_cci']:
            params['cci_length'] = trial.suggest_int('cci_length', 5, 30)
            params['cci_threshold'] = trial.suggest_int('cci_threshold', 50, 200)
        
        if strategy_type == 'williams_r':
            params['willr_length'] = trial.suggest_int('willr_length', 5, 30)
            params['willr_oversold'] = trial.suggest_int('willr_oversold', -95, -70)
            params['willr_overbought'] = trial.suggest_int('willr_overbought', -30, -5)
        
        if strategy_type in ['macd_bb', 'momentum_burst']:
            params['macd_fast'] = trial.suggest_int('macd_fast', 5, 20)
            params['macd_slow'] = trial.suggest_int('macd_slow', 15, 40)
            params['macd_signal'] = trial.suggest_int('macd_signal', 3, 15)
        
        # Risk management - always included
        params['sl_atr_mult'] = trial.suggest_float('sl_atr_mult', 0.5, 5.0, step=0.1)
        params['tp_ratio'] = trial.suggest_float('tp_ratio', 0.5, 5.0, step=0.1)
        
        return params
    
    def run_random_search(self, n_trials: int = 500) -> List[StrategyCandidate]:
        """Phase 1a: Random search - unbiased exploration"""
        self._update_status("Running Random Search (unbiased exploration)...", 5)
        
        results = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        for strategy_type in UnifiedBacktester.STRATEGY_TYPES:
            self._update_status(f"Random: Testing {strategy_type}...", 
                               5 + int(25 * len(results) / (len(UnifiedBacktester.STRATEGY_TYPES) * n_trials // len(UnifiedBacktester.STRATEGY_TYPES))))
            
            trials_per_strategy = n_trials // len(UnifiedBacktester.STRATEGY_TYPES)
            
            study = optuna.create_study(
                direction="maximize",
                sampler=RandomSampler(seed=42)
            )
            
            def objective(trial):
                params = self._create_trial_params(trial, strategy_type)
                candidate = self.backtester.run_backtest(strategy_type, params)
                return candidate.composite_score
            
            study.optimize(objective, n_trials=trials_per_strategy, show_progress_bar=False)
            
            # Get top candidates from this strategy type
            for trial in study.trials:
                if trial.value and trial.value > 0:
                    params = dict(trial.params)
                    params['strategy_type'] = strategy_type
                    candidate = self.backtester.run_backtest(strategy_type, params)
                    candidate.found_by = ['random']
                    results.append(candidate)
        
        # Sort and keep top N
        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.random_results = results[:100]  # Top 100
        
        return self.random_results
    
    def run_bayesian_search(self, n_trials: int = 500) -> List[StrategyCandidate]:
        """Phase 1b: Bayesian TPE - smart directed search"""
        self._update_status("Running Bayesian Optimization (smart search)...", 30)
        
        results = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        for strategy_type in UnifiedBacktester.STRATEGY_TYPES:
            self._update_status(f"Bayesian: Testing {strategy_type}...",
                               30 + int(25 * len(results) / (len(UnifiedBacktester.STRATEGY_TYPES) * n_trials // len(UnifiedBacktester.STRATEGY_TYPES))))
            
            trials_per_strategy = n_trials // len(UnifiedBacktester.STRATEGY_TYPES)
            
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42)
            )
            
            def objective(trial):
                params = self._create_trial_params(trial, strategy_type)
                candidate = self.backtester.run_backtest(strategy_type, params)
                return candidate.composite_score
            
            study.optimize(objective, n_trials=trials_per_strategy, show_progress_bar=False)
            
            for trial in study.trials:
                if trial.value and trial.value > 0:
                    params = dict(trial.params)
                    params['strategy_type'] = strategy_type
                    candidate = self.backtester.run_backtest(strategy_type, params)
                    candidate.found_by = ['bayesian']
                    results.append(candidate)
        
        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.bayesian_results = results[:100]
        
        return self.bayesian_results
    
    def run_cmaes_search(self, n_trials: int = 300) -> List[StrategyCandidate]:
        """Phase 1c: CMA-ES - best for continuous parameters"""
        self._update_status("Running CMA-ES Optimization (continuous params)...", 55)
        
        results = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        for strategy_type in UnifiedBacktester.STRATEGY_TYPES:
            self._update_status(f"CMA-ES: Testing {strategy_type}...",
                               55 + int(20 * len(results) / (len(UnifiedBacktester.STRATEGY_TYPES) * n_trials // len(UnifiedBacktester.STRATEGY_TYPES))))
            
            trials_per_strategy = n_trials // len(UnifiedBacktester.STRATEGY_TYPES)
            
            try:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=CmaEsSampler(seed=42)
                )
                
                def objective(trial):
                    params = self._create_trial_params(trial, strategy_type)
                    candidate = self.backtester.run_backtest(strategy_type, params)
                    return candidate.composite_score
                
                study.optimize(objective, n_trials=trials_per_strategy, show_progress_bar=False)
                
                for trial in study.trials:
                    if trial.value and trial.value > 0:
                        params = dict(trial.params)
                        params['strategy_type'] = strategy_type
                        candidate = self.backtester.run_backtest(strategy_type, params)
                        candidate.found_by = ['cmaes']
                        results.append(candidate)
            except Exception as e:
                print(f"CMA-ES failed for {strategy_type}: {e}")
                continue
        
        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.cmaes_results = results[:100]
        
        return self.cmaes_results
    
    def find_consensus(self, min_methods: int = 2) -> List[StrategyCandidate]:
        """
        Phase 2: Find strategies that multiple methods agree on.
        
        A strategy makes the cut if it appears in top results from
        at least min_methods different optimization methods.
        """
        self._update_status("Finding consensus strategies...", 80)
        
        # Combine all results
        all_candidates = []
        all_candidates.extend(self.random_results)
        all_candidates.extend(self.bayesian_results)
        all_candidates.extend(self.cmaes_results)
        
        # Group by strategy_type + similar params
        strategy_groups = defaultdict(list)
        
        for candidate in all_candidates:
            # Create a signature for grouping similar strategies
            sig_parts = [candidate.strategy_type]
            
            # Round params for grouping
            for key in sorted(candidate.params.keys()):
                val = candidate.params[key]
                if isinstance(val, float):
                    val = round(val, 1)
                sig_parts.append(f"{key}:{val}")
            
            sig = "|".join(sig_parts)
            strategy_groups[sig].append(candidate)
        
        # Find consensus: strategies found by multiple methods
        consensus = []
        
        for sig, candidates in strategy_groups.items():
            # Get unique methods that found this strategy
            methods = set()
            for c in candidates:
                methods.update(c.found_by)
            
            if len(methods) >= min_methods:
                # Take the best version
                best = max(candidates, key=lambda x: x.composite_score)
                best.found_by = list(methods)
                consensus.append(best)
        
        # Sort by composite score
        consensus.sort(key=lambda x: x.composite_score, reverse=True)
        
        self._update_status(f"Found {len(consensus)} consensus strategies", 85)
        
        return consensus
    
    def validate_on_holdout(self, candidates: List[StrategyCandidate]) -> List[StrategyCandidate]:
        """
        Phase 3: Validate top candidates on held-out data.
        Only keeps strategies that also perform well out-of-sample.
        """
        self._update_status("Validating on held-out data...", 90)
        
        validated = []
        
        for candidate in candidates[:30]:  # Validate top 30
            # Re-run on validation data
            val_result = self.val_backtester.run_backtest(
                candidate.strategy_type, 
                candidate.params
            )
            
            # Only keep if still profitable on validation
            if val_result.total_pnl > 0 and val_result.profit_factor > 1:
                # Combine train + val metrics
                candidate.params['val_pnl'] = val_result.total_pnl
                candidate.params['val_pf'] = val_result.profit_factor
                candidate.params['val_wr'] = val_result.win_rate
                
                # Adjust score based on validation
                train_weight = 0.6
                val_weight = 0.4
                candidate.composite_score = (
                    train_weight * candidate.composite_score +
                    val_weight * val_result.composite_score
                )
                
                validated.append(candidate)
        
        # Sort by adjusted score
        validated.sort(key=lambda x: x.composite_score, reverse=True)
        
        return validated
    
    def optimize(self, n_trials_per_method: int = 500) -> List[StrategyCandidate]:
        """
        Run full ensemble optimization pipeline.
        
        Returns top 5 strategies that:
        1. Multiple optimization methods agree on
        2. Validated on held-out data
        3. Have smooth equity curves
        """
        self._update_status("Starting Ensemble Optimization...", 0)
        
        # Phase 1: Run all optimization methods
        self._update_status("Phase 1: Running multiple optimization methods...", 2)
        
        self.run_random_search(n_trials_per_method)
        self.run_bayesian_search(n_trials_per_method)
        self.run_cmaes_search(int(n_trials_per_method * 0.6))  # CMA-ES is slower
        
        # Phase 2: Find consensus
        self._update_status("Phase 2: Finding consensus strategies...", 78)
        consensus = self.find_consensus(min_methods=2)
        
        if not consensus:
            # Fallback: if no consensus, take best from each method
            self._update_status("No consensus found, using best from each method...", 82)
            consensus = []
            if self.random_results:
                consensus.extend(self.random_results[:5])
            if self.bayesian_results:
                consensus.extend(self.bayesian_results[:5])
            if self.cmaes_results:
                consensus.extend(self.cmaes_results[:5])
            consensus.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Phase 3: Validate on holdout
        self._update_status("Phase 3: Validating on held-out data...", 88)
        validated = self.validate_on_holdout(consensus)
        
        # Final top 5
        self.consensus_results = validated[:5]
        
        self._update_status(f"Complete! Top {len(self.consensus_results)} strategies found.", 100)
        
        return self.consensus_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of optimization results"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "capital": self.capital,
                "risk_percent": self.risk_percent,
                "train_candles": len(self.train_df),
                "validation_candles": len(self.val_df)
            },
            "methods_used": ["Random Search", "Bayesian TPE", "CMA-ES"],
            "strategies_tested": len(UnifiedBacktester.STRATEGY_TYPES),
            "top_5": []
        }
        
        for i, candidate in enumerate(self.consensus_results[:5], 1):
            strat_report = {
                "rank": i,
                "strategy_type": candidate.strategy_type,
                "found_by_methods": candidate.found_by,
                "params": {k: round(v, 2) if isinstance(v, float) else v 
                          for k, v in candidate.params.items() 
                          if not k.startswith('val_')},
                "metrics": {
                    "total_trades": candidate.total_trades,
                    "win_rate": round(candidate.win_rate, 1),
                    "profit_factor": round(candidate.profit_factor, 2),
                    "total_pnl": round(candidate.total_pnl, 2),
                    "max_drawdown": round(candidate.max_drawdown, 2),
                    "equity_smoothness": round(candidate.equity_r_squared, 3),
                    "recovery_factor": round(candidate.recovery_factor, 2),
                    "composite_score": round(candidate.composite_score, 3)
                },
                "validation": {
                    "val_pnl": round(candidate.params.get('val_pnl', 0), 2),
                    "val_profit_factor": round(candidate.params.get('val_pf', 0), 2),
                    "val_win_rate": round(candidate.params.get('val_wr', 0), 1)
                },
                "equity_curve": candidate.equity_curve[-100:] if candidate.equity_curve else []  # Last 100 points
            }
            report["top_5"].append(strat_report)
        
        return report


def run_ensemble_optimization(df: pd.DataFrame, capital: float = 1000.0,
                             risk_percent: float = 2.0, 
                             n_trials: int = 500,
                             status: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for ensemble optimization.
    
    Args:
        df: OHLCV DataFrame with 'time', 'open', 'high', 'low', 'close' columns
        capital: Starting capital
        risk_percent: Risk per trade as percentage
        n_trials: Trials per optimization method (more = slower but better)
        status: Optional dict for progress updates
    
    Returns:
        Report with top 5 strategies
    """
    optimizer = EnsembleOptimizer(df, capital, risk_percent, status)
    optimizer.optimize(n_trials_per_method=n_trials)
    return optimizer.generate_report()



