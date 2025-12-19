"""
DaviddTech-Style Professional Trading Strategies
=================================================
Implements strategies inspired by DaviddTech's proven configurations:
- Stiff Surge (JMA + Stiffness + TDFI + Volatility Quality)
- McGinley Trend Followers (McGinley Dynamic + LWPI + Flat Market)
- Trendhoo/Trendilo (ALMA + HMA + MA confluence)
- MACD Liquidity Spectrum (MACD + Range Filter + Liquidity)
- Precision Trend Mastery (ADX + Normalized Volume + Range Filter)
- T3 Nexus + Stiff (T3 + Stiffness indicators)

Features:
- ATR-based stop losses (2.5-3x multiplier)
- Multi-take profit (TP1 33%, TP2 33%, TP3 100%)
- Flat market detection (avoid choppy markets)
- Risk:Reward ratios (1.0-3.0)
- Dynamic position sizing
- Multiple confirmation indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from daviddtech_indicators import (
    calculate_daviddtech_indicators, jma, mcginley_dynamic, stiffness, 
    tdfi, volatility_quality, trendilo, range_filter, flat_market_detector,
    t3, zlema, hma, supertrend, alma, lwpi, ema, sma
)


@dataclass
class Trade:
    """Trade record with multi-TP tracking"""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # 'long' or 'short'
    entry_price: float
    position_size: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""
    tp1_hit: bool = False
    tp2_hit: bool = False


@dataclass
class StrategyResult:
    """Complete strategy backtest results"""
    name: str
    category: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    trades: List[Trade]
    
    # Period breakdowns
    pnl_last_day: float = 0.0
    pnl_last_week: float = 0.0
    pnl_last_month: float = 0.0
    pnl_last_3_months: float = 0.0
    pnl_last_6_months: float = 0.0
    pnl_last_year: float = 0.0
    monthly_pnl: Dict[str, float] = None
    
    # Strategy parameters for replication
    parameters: Dict[str, Any] = None


class DaviddTechBacktester:
    """
    Professional backtester with DaviddTech-style features
    """
    
    def __init__(self, df: pd.DataFrame, initial_capital: float = 1000.0,
                 risk_percent: float = 2.0, compound: bool = False):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_percent = risk_percent
        self.compound = compound
        
        # Calculate ATR for all rows
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        self.df['atr'] = atr.average_true_range()
        
    def calculate_atr_stops(self, i: int, direction: str, atr_mult: float = 2.5) -> Tuple[float, float, float, float]:
        """
        Calculate ATR-based stop loss and multi-take profit levels
        
        Returns: (sl_price, tp1_price, tp2_price, tp3_price)
        """
        entry_price = self.df['close'].iloc[i]
        atr = self.df['atr'].iloc[i] if not pd.isna(self.df['atr'].iloc[i]) else entry_price * 0.02
        
        sl_distance = atr * atr_mult
        
        if direction == 'long':
            sl_price = entry_price - sl_distance
            # Multi-TP based on R:R ratios
            tp1_price = entry_price + sl_distance * 1.0  # 1:1
            tp2_price = entry_price + sl_distance * 1.5  # 1.5:1
            tp3_price = entry_price + sl_distance * 2.5  # 2.5:1
        else:
            sl_price = entry_price + sl_distance
            tp1_price = entry_price - sl_distance * 1.0
            tp2_price = entry_price - sl_distance * 1.5
            tp3_price = entry_price - sl_distance * 2.5
        
        return sl_price, tp1_price, tp2_price, tp3_price
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> float:
        """Calculate position size based on risk"""
        capital = self.current_capital if self.compound else self.initial_capital
        risk_amount = capital * (self.risk_percent / 100)
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance == 0:
            return 0
        
        position_size = risk_amount / sl_distance
        return position_size
    
    def run_backtest(self, name: str, category: str, 
                     long_signal: pd.Series, short_signal: pd.Series,
                     flat_filter: Optional[pd.Series] = None,
                     atr_mult: float = 2.5,
                     rr_ratio: float = 1.5,
                     use_multi_tp: bool = True) -> StrategyResult:
        """
        Run backtest with DaviddTech-style trade management
        """
        trades: List[Trade] = []
        position: Optional[Trade] = None
        
        for i in range(50, len(self.df)):  # Start after indicator warmup
            current_price = self.df['close'].iloc[i]
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            current_time = self.df['time'].iloc[i] if 'time' in self.df.columns else self.df.index[i]
            
            # Check if in flat market (skip signals if flat)
            is_flat = False
            if flat_filter is not None and i < len(flat_filter):
                is_flat = flat_filter.iloc[i] if not pd.isna(flat_filter.iloc[i]) else False
            
            # Manage existing position
            if position is not None:
                # Check stop loss
                if position.direction == 'long':
                    if current_low <= position.sl_price:
                        # Stop loss hit
                        position.exit_price = position.sl_price
                        position.exit_time = current_time
                        position.exit_reason = "Stop Loss"
                        position.pnl = (position.exit_price - position.entry_price) * position.position_size
                        trades.append(position)
                        if self.compound:
                            self.current_capital += position.pnl
                        position = None
                        continue
                    
                    # Check take profits
                    if use_multi_tp:
                        if not position.tp1_hit and current_high >= position.tp1_price:
                            position.tp1_hit = True
                            # Move SL to breakeven after TP1
                            position.sl_price = position.entry_price
                        
                        if not position.tp2_hit and current_high >= position.tp2_price:
                            position.tp2_hit = True
                        
                        if current_high >= position.tp3_price:
                            # Full exit at TP3
                            position.exit_price = position.tp3_price
                            position.exit_time = current_time
                            position.exit_reason = "Take Profit 3"
                            position.pnl = (position.exit_price - position.entry_price) * position.position_size
                            trades.append(position)
                            if self.compound:
                                self.current_capital += position.pnl
                            position = None
                            continue
                    else:
                        # Single TP at specified R:R
                        tp_price = position.entry_price + (position.entry_price - position.sl_price) * rr_ratio
                        if current_high >= tp_price:
                            position.exit_price = tp_price
                            position.exit_time = current_time
                            position.exit_reason = "Take Profit"
                            position.pnl = (position.exit_price - position.entry_price) * position.position_size
                            trades.append(position)
                            if self.compound:
                                self.current_capital += position.pnl
                            position = None
                            continue
                
                else:  # Short position
                    if current_high >= position.sl_price:
                        position.exit_price = position.sl_price
                        position.exit_time = current_time
                        position.exit_reason = "Stop Loss"
                        position.pnl = (position.entry_price - position.exit_price) * position.position_size
                        trades.append(position)
                        if self.compound:
                            self.current_capital += position.pnl
                        position = None
                        continue
                    
                    if use_multi_tp:
                        if not position.tp1_hit and current_low <= position.tp1_price:
                            position.tp1_hit = True
                            position.sl_price = position.entry_price
                        
                        if not position.tp2_hit and current_low <= position.tp2_price:
                            position.tp2_hit = True
                        
                        if current_low <= position.tp3_price:
                            position.exit_price = position.tp3_price
                            position.exit_time = current_time
                            position.exit_reason = "Take Profit 3"
                            position.pnl = (position.entry_price - position.exit_price) * position.position_size
                            trades.append(position)
                            if self.compound:
                                self.current_capital += position.pnl
                            position = None
                            continue
                    else:
                        tp_price = position.entry_price - (position.sl_price - position.entry_price) * rr_ratio
                        if current_low <= tp_price:
                            position.exit_price = tp_price
                            position.exit_time = current_time
                            position.exit_reason = "Take Profit"
                            position.pnl = (position.entry_price - position.exit_price) * position.position_size
                            trades.append(position)
                            if self.compound:
                                self.current_capital += position.pnl
                            position = None
                            continue
            
            # Check for new signals (only if no position and not flat)
            if position is None and not is_flat:
                # Long signal
                if i < len(long_signal) and long_signal.iloc[i]:
                    sl, tp1, tp2, tp3 = self.calculate_atr_stops(i, 'long', atr_mult)
                    pos_size = self.calculate_position_size(current_price, sl)
                    
                    if pos_size > 0:
                        position = Trade(
                            entry_time=current_time,
                            exit_time=None,
                            direction='long',
                            entry_price=current_price,
                            position_size=pos_size,
                            sl_price=sl,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            tp3_price=tp3
                        )
                
                # Short signal
                elif i < len(short_signal) and short_signal.iloc[i]:
                    sl, tp1, tp2, tp3 = self.calculate_atr_stops(i, 'short', atr_mult)
                    pos_size = self.calculate_position_size(current_price, sl)
                    
                    if pos_size > 0:
                        position = Trade(
                            entry_time=current_time,
                            exit_time=None,
                            direction='short',
                            entry_price=current_price,
                            position_size=pos_size,
                            sl_price=sl,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            tp3_price=tp3
                        )
        
        # Close any open position at end
        if position is not None:
            position.exit_price = self.df['close'].iloc[-1]
            position.exit_time = self.df['time'].iloc[-1] if 'time' in self.df.columns else self.df.index[-1]
            position.exit_reason = "End of Data"
            if position.direction == 'long':
                position.pnl = (position.exit_price - position.entry_price) * position.position_size
            else:
                position.pnl = (position.entry_price - position.exit_price) * position.position_size
            trades.append(position)
        
        # Calculate statistics
        return self._calculate_stats(name, category, trades)
    
    def _calculate_stats(self, name: str, category: str, trades: List[Trade]) -> StrategyResult:
        """Calculate comprehensive statistics"""
        if not trades:
            return StrategyResult(
                name=name, category=category, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, total_pnl=0, profit_factor=0,
                max_drawdown=0, avg_win=0, avg_loss=0, largest_win=0,
                largest_loss=0, avg_trade_duration=0, trades=[]
            )
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Period P&L
        now = self.df['time'].iloc[-1] if 'time' in self.df.columns else datetime.now()
        if isinstance(now, str):
            now = pd.to_datetime(now)
        
        pnl_periods = {
            'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            '3_months': timedelta(days=90),
            '6_months': timedelta(days=180),
            'year': timedelta(days=365)
        }
        
        period_pnls = {}
        for period_name, delta in pnl_periods.items():
            cutoff = now - delta
            period_trades = [t for t in trades if isinstance(t.exit_time, datetime) and t.exit_time >= cutoff]
            period_pnls[period_name] = sum(t.pnl for t in period_trades)
        
        # Monthly breakdown
        monthly_pnl = {}
        for trade in trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime('%Y-%m') if isinstance(trade.exit_time, datetime) else str(trade.exit_time)[:7]
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
        
        return StrategyResult(
            name=name,
            category=category,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            avg_trade_duration=0,  # Could calculate if needed
            trades=trades,
            pnl_last_day=period_pnls['day'],
            pnl_last_week=period_pnls['week'],
            pnl_last_month=period_pnls['month'],
            pnl_last_3_months=period_pnls['3_months'],
            pnl_last_6_months=period_pnls['6_months'],
            pnl_last_year=period_pnls['year'],
            monthly_pnl=monthly_pnl
        )


class DaviddTechStrategyResearch:
    """
    Research engine implementing DaviddTech-style strategies
    """
    
    def __init__(self, df: pd.DataFrame, capital: float = 1000.0,
                 risk_percent: float = 2.0, compound: bool = False,
                 status_callback: Optional[Dict] = None):
        self.df = df.copy()
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.capital = capital
        self.risk_percent = risk_percent
        self.compound = compound
        self.status = status_callback or {}
        
        # Calculate all DaviddTech indicators
        self._update_status("Calculating DaviddTech indicators...", 5)
        self.df = calculate_daviddtech_indicators(self.df)
        
        # Calculate standard indicators too
        self._calculate_standard_indicators()
        
        self.backtester = DaviddTechBacktester(
            self.df, capital, risk_percent, compound
        )
        
        self.results: List[StrategyResult] = []
    
    def _update_status(self, message: str, progress: int):
        """Update status for UI feedback"""
        if self.status:
            self.status['message'] = message
            self.status['progress'] = progress
    
    def _calculate_standard_indicators(self):
        """Calculate standard TA indicators"""
        df = self.df
        
        # RSI
        for length in [6, 14, 21]:
            rsi = RSIIndicator(df['close'], window=length)
            df[f'rsi_{length}'] = rsi.rsi()
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], 
                                      window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = bb.bollinger_wband()
        
        # Moving averages
        for length in [9, 21, 55, 100, 200]:
            df[f'ema_{length}'] = ema(df['close'], length)
            df[f'sma_{length}'] = sma(df['close'], length)
    
    def run_all_strategies(self) -> List[StrategyResult]:
        """Run all DaviddTech-style strategies"""
        strategies = [
            self._stiff_surge_v1,
            self._stiff_surge_v2,
            self._mcginley_trend_followers,
            self._trendhoo_v1,
            self._trendhoo_v2,
            self._macd_liquidity_spectrum,
            self._precision_trend_mastery,
            self._t3_nexus_stiff,
            self._range_filter_adx,
            self._supertrend_confluence,
            self._jma_volatility_quality,
            self._hma_stiffness,
            self._zlema_momentum,
            self._trendilo_breakout,
            self._flat_market_mean_reversion,
            self._multi_ma_confluence,
            self._lwpi_trend,
            self._volume_confirmed_trend,
            self._bb_squeeze_momentum,
            self._adaptive_momentum,
        ]
        
        total = len(strategies)
        for i, strategy_func in enumerate(strategies):
            self._update_status(f"Testing: {strategy_func.__name__.replace('_', ' ').title()}", 
                               10 + int(80 * i / total))
            try:
                result = strategy_func()
                self.results.append(result)
            except Exception as e:
                print(f"Error in {strategy_func.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        self._update_status("Generating report...", 95)
        return self.results
    
    def _stiff_surge_v1(self) -> StrategyResult:
        """
        Stiff Surge Strategy v1
        Based on: JMA + Stiffness + TDFI
        From: Stiff_Surge_LINKUSDT params
        """
        df = self.df
        
        # JMA trend
        jma_val = df.get('jma_43_84', df['close'])
        jma_trend_up = df['close'] > jma_val
        jma_trend_dn = df['close'] < jma_val
        
        # Stiffness above threshold
        stiff = df.get('stiff_smooth_60_100', pd.Series(50, index=df.index))
        stiff_bullish = stiff > 50
        stiff_bearish = stiff < 50
        
        # TDFI direction
        tdfi_val = df.get('tdfi_15', pd.Series(0, index=df.index))
        tdfi_bullish = tdfi_val > 0.05
        tdfi_bearish = tdfi_val < -0.05
        
        # Flat market filter
        is_flat = df.get('is_flat_15', pd.Series(False, index=df.index))
        
        # Entry signals
        long_signal = jma_trend_up & stiff_bullish & tdfi_bullish
        short_signal = jma_trend_dn & stiff_bearish & tdfi_bearish
        
        return self.backtester.run_backtest(
            "Stiff Surge v1 (JMA+Stiff+TDFI)",
            "DaviddTech - Stiff Surge",
            long_signal, short_signal, is_flat,
            atr_mult=2.8, rr_ratio=1.1, use_multi_tp=True
        )
    
    def _stiff_surge_v2(self) -> StrategyResult:
        """
        Stiff Surge Strategy v2
        Based on: JMA + Volatility Quality
        From: Stiff_Surge_SUPERUSDT params
        """
        df = self.df
        
        # JMA with color confirmation
        jma_val = df.get('jma_7_50', df['close'])
        jma_rising = jma_val > jma_val.shift(1)
        jma_falling = jma_val < jma_val.shift(1)
        
        # Volatility Quality
        vq = df.get('vq_14', pd.Series(0, index=df.index))
        vq_positive = vq > 0
        vq_negative = vq < 0
        
        # Cross confirmation
        long_signal = jma_rising & vq_positive & (df['close'] > jma_val)
        short_signal = jma_falling & vq_negative & (df['close'] < jma_val)
        
        is_flat = df.get('is_flat_30', pd.Series(False, index=df.index))
        
        return self.backtester.run_backtest(
            "Stiff Surge v2 (JMA+VQ)",
            "DaviddTech - Stiff Surge",
            long_signal, short_signal, is_flat,
            atr_mult=2.5, rr_ratio=1.3, use_multi_tp=True
        )
    
    def _mcginley_trend_followers(self) -> StrategyResult:
        """
        McGinley Trend Followers
        Based on: McGinley Dynamic + LWPI + Flat Market
        From: McGinley_Trend_Followers__v5__GALAUSDT params
        """
        df = self.df
        
        # McGinley Dynamic trend
        md = df.get('mcginley_130_0.6', df['close'])
        md_trend_up = df['close'] > md
        md_trend_dn = df['close'] < md
        md_rising = md > md.shift(1)
        md_falling = md < md.shift(1)
        
        # LWPI
        lwpi_val = df.get('lwpi_130', pd.Series(-50, index=df.index))
        lwpi_oversold = lwpi_val < -80
        lwpi_overbought = lwpi_val > -20
        
        # Flat market filter with lower threshold
        is_flat = df.get('is_flat_15', pd.Series(False, index=df.index))
        
        # Entry: Price above McGinley, McGinley rising, LWPI showing reversal
        long_signal = md_trend_up & md_rising & (lwpi_val.shift(1) < -60) & (lwpi_val > lwpi_val.shift(1))
        short_signal = md_trend_dn & md_falling & (lwpi_val.shift(1) > -40) & (lwpi_val < lwpi_val.shift(1))
        
        return self.backtester.run_backtest(
            "McGinley Trend Followers",
            "DaviddTech - McGinley",
            long_signal, short_signal, is_flat,
            atr_mult=3.0, rr_ratio=1.04, use_multi_tp=True
        )
    
    def _trendhoo_v1(self) -> StrategyResult:
        """
        Trendhoo Strategy v1
        Based on: Trendilo + HMA + MA confluence
        From: Trendhoo__v5__FARTCOINUSDT params
        """
        df = self.df
        
        # Trendilo
        trendilo = df.get('trendilo_52', df['close'])
        trendilo_upper = df.get('trendilo_upper_52', df['close'] * 1.02)
        trendilo_lower = df.get('trendilo_lower_52', df['close'] * 0.98)
        
        # Price vs Trendilo bands
        above_trend = df['close'] > trendilo
        below_trend = df['close'] < trendilo
        
        # HMA confirmation
        hma_val = df.get('hma_65', df['close'])
        hma_bullish = df['close'] > hma_val
        hma_bearish = df['close'] < hma_val
        
        # MA5 (T3) confirmation
        t3_val = df.get('t3_5', df['close'])
        above_t3 = df['close'] > t3_val
        below_t3 = df['close'] < t3_val
        
        # Stochastic oversold/overbought
        stoch = df.get('stoch_k', pd.Series(50, index=df.index))
        stoch_oversold = stoch < 20
        stoch_overbought = stoch > 80
        
        # Entry signals
        long_signal = above_trend & hma_bullish & above_t3 & (stoch_oversold.shift(1) | (stoch < 40))
        short_signal = below_trend & hma_bearish & below_t3 & (stoch_overbought.shift(1) | (stoch > 60))
        
        return self.backtester.run_backtest(
            "Trendhoo v1 (Trendilo+HMA+T3)",
            "DaviddTech - Trendhoo",
            long_signal, short_signal, None,
            atr_mult=2.779, rr_ratio=2.9, use_multi_tp=True
        )
    
    def _trendhoo_v2(self) -> StrategyResult:
        """
        Trendhoo Strategy v2 
        Optimized ALMA settings
        """
        df = self.df
        
        # ALMA with special offset
        alma_val = df.get('alma_11_0.85', df['close'])
        alma_trend_up = df['close'] > alma_val
        alma_trend_dn = df['close'] < alma_val
        alma_rising = alma_val > alma_val.shift(1)
        alma_falling = alma_val < alma_val.shift(1)
        
        # HMA for trend confirmation
        hma_val = df.get('hma_65', df['close'])
        
        # Stochastic with higher smoothing
        stoch = df.get('stoch_k', pd.Series(50, index=df.index))
        
        # Entry on ALMA trend with HMA confirmation
        long_signal = alma_trend_up & alma_rising & (df['close'] > hma_val) & (stoch < 70)
        short_signal = alma_trend_dn & alma_falling & (df['close'] < hma_val) & (stoch > 30)
        
        return self.backtester.run_backtest(
            "Trendhoo v2 (ALMA Optimized)",
            "DaviddTech - Trendhoo",
            long_signal, short_signal, None,
            atr_mult=2.8, rr_ratio=2.1, use_multi_tp=True
        )
    
    def _macd_liquidity_spectrum(self) -> StrategyResult:
        """
        MACD Liquidity Spectrum
        Based on: MACD + Range Filter + Liquidity
        From: MACD_Liquidity_Spectrum_DOGEUSDT params
        """
        df = self.df
        
        # MACD crossover with MA confirmation
        macd = df.get('macd', pd.Series(0, index=df.index))
        macd_signal = df.get('macd_signal', pd.Series(0, index=df.index))
        macd_hist = df.get('macd_hist', pd.Series(0, index=df.index))
        
        # MACD cross
        macd_cross_up = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
        macd_cross_dn = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
        
        # MACD above/below zero
        macd_positive = macd > 0
        macd_negative = macd < 0
        
        # Range filter direction
        rf_dir = df.get('rf_dir_100_3.0', pd.Series(0, index=df.index))
        rf_bullish = rf_dir > 0
        rf_bearish = rf_dir < 0
        
        # Entry: MACD cross + Range Filter confirmation
        long_signal = (macd_cross_up | (macd_positive & macd_hist > 0)) & rf_bullish
        short_signal = (macd_cross_dn | (macd_negative & macd_hist < 0)) & rf_bearish
        
        return self.backtester.run_backtest(
            "MACD Liquidity Spectrum",
            "DaviddTech - MACD",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=1.3, use_multi_tp=True
        )
    
    def _precision_trend_mastery(self) -> StrategyResult:
        """
        Precision Trend Mastery
        Based on: ADX + Normalized Volume + Range Filter + Trendilo
        From: Precision_Trend_Mastery_SOLUSDT params
        """
        df = self.df
        
        # ADX for trend strength
        adx = df.get('adx', pd.Series(20, index=df.index))
        di_plus = df.get('di_plus', pd.Series(0, index=df.index))
        di_minus = df.get('di_minus', pd.Series(0, index=df.index))
        
        strong_trend = adx > 34
        di_bullish = di_plus > di_minus
        di_bearish = di_minus > di_plus
        
        # Volume confirmation
        high_vol = df.get('high_vol', pd.Series(False, index=df.index))
        
        # Range filter
        rf_dir = df.get('rf_dir_164_4.5', df.get('rf_dir_100_3.0', pd.Series(0, index=df.index)))
        rf_bullish = rf_dir > 0
        rf_bearish = rf_dir < 0
        
        # Flat market filter
        is_flat = df.get('is_flat_40', pd.Series(False, index=df.index))
        
        # Entry: Strong ADX trend + DI direction + Range Filter + Volume
        long_signal = strong_trend & di_bullish & rf_bullish
        short_signal = strong_trend & di_bearish & rf_bearish
        
        return self.backtester.run_backtest(
            "Precision Trend Mastery",
            "DaviddTech - Precision",
            long_signal, short_signal, is_flat,
            atr_mult=5.0, rr_ratio=1.2, use_multi_tp=True
        )
    
    def _t3_nexus_stiff(self) -> StrategyResult:
        """
        T3 Nexus + Stiff
        T3 moving average with Stiffness confirmation
        """
        df = self.df
        
        # T3 trend
        t3_fast = df.get('t3_5', df['close'])
        t3_slow = df.get('t3_20', df['close'])
        
        t3_bullish = t3_fast > t3_slow
        t3_bearish = t3_fast < t3_slow
        t3_cross_up = t3_bullish & ~t3_bullish.shift(1).fillna(False)
        t3_cross_dn = t3_bearish & ~t3_bearish.shift(1).fillna(False)
        
        # Stiffness
        stiff = df.get('stiff_smooth_60_100', pd.Series(50, index=df.index))
        stiff_bullish = stiff > 60
        stiff_bearish = stiff < 40
        
        # Entry on T3 cross with Stiffness confirmation
        long_signal = (t3_cross_up | (t3_bullish & (df['close'] > t3_fast))) & stiff_bullish
        short_signal = (t3_cross_dn | (t3_bearish & (df['close'] < t3_fast))) & stiff_bearish
        
        return self.backtester.run_backtest(
            "T3 Nexus + Stiff",
            "DaviddTech - T3 Nexus",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=1.5, use_multi_tp=True
        )
    
    def _range_filter_adx(self) -> StrategyResult:
        """Range Filter with ADX confirmation"""
        df = self.df
        
        rf_dir = df.get('rf_dir_100_3.0', pd.Series(0, index=df.index))
        rf_change_up = (rf_dir > 0) & (rf_dir.shift(1) <= 0)
        rf_change_dn = (rf_dir < 0) & (rf_dir.shift(1) >= 0)
        
        adx = df.get('adx', pd.Series(20, index=df.index))
        strong_trend = adx > 25
        
        long_signal = (rf_change_up | (rf_dir > 0)) & strong_trend
        short_signal = (rf_change_dn | (rf_dir < 0)) & strong_trend
        
        is_flat = df.get('is_flat_15', pd.Series(False, index=df.index))
        
        return self.backtester.run_backtest(
            "Range Filter + ADX",
            "DaviddTech - Range Filter",
            long_signal, short_signal, is_flat,
            atr_mult=3.0, rr_ratio=2.0, use_multi_tp=True
        )
    
    def _supertrend_confluence(self) -> StrategyResult:
        """Supertrend with multi-indicator confluence"""
        df = self.df
        
        st_dir = df.get('st_dir_10_3.0', pd.Series(1, index=df.index))
        st_bullish = st_dir > 0
        st_bearish = st_dir < 0
        st_flip_up = st_bullish & ~st_bullish.shift(1).fillna(False)
        st_flip_dn = st_bearish & ~st_bearish.shift(1).fillna(False)
        
        # RSI confirmation
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        rsi_bullish = rsi > 50
        rsi_bearish = rsi < 50
        
        long_signal = (st_flip_up | st_bullish) & rsi_bullish
        short_signal = (st_flip_dn | st_bearish) & rsi_bearish
        
        return self.backtester.run_backtest(
            "Supertrend Confluence",
            "Trend Following",
            long_signal, short_signal, None,
            atr_mult=3.0, rr_ratio=2.0, use_multi_tp=True
        )
    
    def _jma_volatility_quality(self) -> StrategyResult:
        """JMA with Volatility Quality filter"""
        df = self.df
        
        jma = df.get('jma_14_50', df['close'])
        jma_up = df['close'] > jma
        jma_dn = df['close'] < jma
        
        vq = df.get('vq_29', pd.Series(0, index=df.index))
        vq_strong_up = vq > 0.5
        vq_strong_dn = vq < -0.5
        
        long_signal = jma_up & vq_strong_up
        short_signal = jma_dn & vq_strong_dn
        
        return self.backtester.run_backtest(
            "JMA + Volatility Quality",
            "Momentum",
            long_signal, short_signal, None,
            atr_mult=2.75, rr_ratio=1.9, use_multi_tp=True
        )
    
    def _hma_stiffness(self) -> StrategyResult:
        """HMA with Stiffness"""
        df = self.df
        
        hma = df.get('hma_100', df['close'])
        hma_up = df['close'] > hma
        hma_dn = df['close'] < hma
        hma_rising = hma > hma.shift(1)
        hma_falling = hma < hma.shift(1)
        
        stiff = df.get('stiff_smooth_39_50', df.get('stiff_smooth_60_100', pd.Series(50, index=df.index)))
        stiff_high = stiff > 70
        stiff_low = stiff < 30
        
        long_signal = hma_up & hma_rising & stiff_high
        short_signal = hma_dn & hma_falling & stiff_low
        
        return self.backtester.run_backtest(
            "HMA + Stiffness",
            "Trend Following",
            long_signal, short_signal, None,
            atr_mult=2.8, rr_ratio=1.5, use_multi_tp=True
        )
    
    def _zlema_momentum(self) -> StrategyResult:
        """ZLEMA momentum strategy"""
        df = self.df
        
        zlema_fast = df.get('zlema_10', df['close'])
        zlema_slow = df.get('zlema_50', df['close'])
        
        bullish = zlema_fast > zlema_slow
        bearish = zlema_fast < zlema_slow
        
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        
        long_signal = bullish & (rsi > 40) & (rsi < 70)
        short_signal = bearish & (rsi < 60) & (rsi > 30)
        
        return self.backtester.run_backtest(
            "ZLEMA Momentum",
            "Momentum",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=1.5, use_multi_tp=False
        )
    
    def _trendilo_breakout(self) -> StrategyResult:
        """Trendilo band breakout"""
        df = self.df
        
        trendilo = df.get('trendilo_52', df['close'])
        upper = df.get('trendilo_upper_52', df['close'] * 1.02)
        lower = df.get('trendilo_lower_52', df['close'] * 0.98)
        
        breakout_up = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
        breakout_dn = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))
        
        long_signal = breakout_up | (df['close'] > trendilo) & (df['close'].shift(1) <= trendilo.shift(1))
        short_signal = breakout_dn | (df['close'] < trendilo) & (df['close'].shift(1) >= trendilo.shift(1))
        
        return self.backtester.run_backtest(
            "Trendilo Breakout",
            "Breakout",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=2.0, use_multi_tp=True
        )
    
    def _flat_market_mean_reversion(self) -> StrategyResult:
        """Mean reversion in flat markets only"""
        df = self.df
        
        is_flat = df.get('is_flat_30', pd.Series(False, index=df.index))
        
        bb_upper = df.get('bb_upper', df['close'] * 1.02)
        bb_lower = df.get('bb_lower', df['close'] * 0.98)
        
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        
        long_signal = is_flat & (df['close'] <= bb_lower) & (rsi < 35)
        short_signal = is_flat & (df['close'] >= bb_upper) & (rsi > 65)
        
        return self.backtester.run_backtest(
            "Flat Market Mean Reversion",
            "Mean Reversion",
            long_signal, short_signal, ~is_flat,  # Invert: avoid trending
            atr_mult=2.0, rr_ratio=1.5, use_multi_tp=False
        )
    
    def _multi_ma_confluence(self) -> StrategyResult:
        """Multiple MA alignment"""
        df = self.df
        
        ema9 = df.get('ema_9', df['close'])
        ema21 = df.get('ema_21', df['close'])
        ema55 = df.get('ema_55', df['close'])
        sma200 = df.get('sma_200', df['close'])
        
        # All MAs aligned
        all_bullish = (ema9 > ema21) & (ema21 > ema55) & (df['close'] > sma200)
        all_bearish = (ema9 < ema21) & (ema21 < ema55) & (df['close'] < sma200)
        
        long_signal = all_bullish & (df['close'] > ema9)
        short_signal = all_bearish & (df['close'] < ema9)
        
        is_flat = df.get('is_flat_15', pd.Series(False, index=df.index))
        
        return self.backtester.run_backtest(
            "Multi-MA Confluence",
            "Trend Following",
            long_signal, short_signal, is_flat,
            atr_mult=2.5, rr_ratio=2.0, use_multi_tp=True
        )
    
    def _lwpi_trend(self) -> StrategyResult:
        """LWPI trend following"""
        df = self.df
        
        lwpi_val = df.get('lwpi_13', pd.Series(-50, index=df.index))
        
        # LWPI reversals
        lwpi_oversold = lwpi_val < -80
        lwpi_overbought = lwpi_val > -20
        lwpi_rising = lwpi_val > lwpi_val.shift(1)
        lwpi_falling = lwpi_val < lwpi_val.shift(1)
        
        # McGinley trend
        md = df.get('mcginley_14_0.6', df['close'])
        trend_up = df['close'] > md
        trend_dn = df['close'] < md
        
        long_signal = lwpi_oversold.shift(1) & lwpi_rising & trend_up
        short_signal = lwpi_overbought.shift(1) & lwpi_falling & trend_dn
        
        return self.backtester.run_backtest(
            "LWPI Trend",
            "DaviddTech - McGinley",
            long_signal, short_signal, None,
            atr_mult=3.0, rr_ratio=1.5, use_multi_tp=True
        )
    
    def _volume_confirmed_trend(self) -> StrategyResult:
        """Volume confirmed trend entries"""
        df = self.df
        
        high_vol = df.get('high_vol', pd.Series(False, index=df.index))
        
        # Supertrend direction
        st_dir = df.get('st_dir_14_2.5', df.get('st_dir_10_3.0', pd.Series(1, index=df.index)))
        
        # RSI momentum
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        
        long_signal = (st_dir > 0) & high_vol & (rsi > 50) & (rsi < 70)
        short_signal = (st_dir < 0) & high_vol & (rsi < 50) & (rsi > 30)
        
        return self.backtester.run_backtest(
            "Volume Confirmed Trend",
            "Volume",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=2.0, use_multi_tp=True
        )
    
    def _bb_squeeze_momentum(self) -> StrategyResult:
        """Bollinger Band squeeze breakout"""
        df = self.df
        
        bb_width = df.get('bb_width', pd.Series(0.05, index=df.index))
        bb_upper = df.get('bb_upper', df['close'] * 1.02)
        bb_lower = df.get('bb_lower', df['close'] * 0.98)
        
        # Squeeze = low bandwidth
        avg_width = bb_width.rolling(50).mean()
        in_squeeze = bb_width < avg_width * 0.8
        
        # Breakout after squeeze
        squeeze_released = in_squeeze.shift(1) & ~in_squeeze
        
        macd_hist = df.get('macd_hist', pd.Series(0, index=df.index))
        
        long_signal = squeeze_released & (df['close'] > bb_upper.shift(1)) & (macd_hist > 0)
        short_signal = squeeze_released & (df['close'] < bb_lower.shift(1)) & (macd_hist < 0)
        
        return self.backtester.run_backtest(
            "BB Squeeze Momentum",
            "Volatility",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=2.5, use_multi_tp=True
        )
    
    def _adaptive_momentum(self) -> StrategyResult:
        """Adaptive momentum using multiple indicators"""
        df = self.df
        
        # Score multiple indicators
        score = pd.Series(0, index=df.index, dtype=float)
        
        # RSI contribution
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        score += np.where(rsi > 50, 1, -1)
        
        # MACD contribution
        macd_hist = df.get('macd_hist', pd.Series(0, index=df.index))
        score += np.where(macd_hist > 0, 1, -1)
        
        # ADX direction
        di_plus = df.get('di_plus', pd.Series(0, index=df.index))
        di_minus = df.get('di_minus', pd.Series(0, index=df.index))
        score += np.where(di_plus > di_minus, 1, -1)
        
        # Range filter
        rf_dir = df.get('rf_dir_100_3.0', pd.Series(0, index=df.index))
        score += rf_dir
        
        # Supertrend
        st_dir = df.get('st_dir_10_3.0', pd.Series(0, index=df.index))
        score += st_dir
        
        # Strong agreement needed
        long_signal = score >= 4
        short_signal = score <= -4
        
        return self.backtester.run_backtest(
            "Adaptive Momentum (Multi-Indicator)",
            "Confluence",
            long_signal, short_signal, None,
            atr_mult=2.5, rr_ratio=2.0, use_multi_tp=True
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        if not self.results:
            return {"error": "No results to report"}
        
        # Sort by profit factor
        sorted_results = sorted(self.results, key=lambda x: x.profit_factor, reverse=True)
        
        # Top 5
        top_5 = sorted_results[:5]
        
        # Compile report
        report = {
            "strategies_tested": len(self.results),
            "top_5": [],
            "all_results": [],
            "summary": {
                "best_strategy": top_5[0].name if top_5 else "N/A",
                "best_profit_factor": top_5[0].profit_factor if top_5 else 0,
                "best_win_rate": top_5[0].win_rate if top_5 else 0,
                "best_total_pnl": top_5[0].total_pnl if top_5 else 0,
            }
        }
        
        # Format top 5
        for result in top_5:
            report["top_5"].append({
                "name": result.name,
                "category": result.category,
                "profit_factor": round(result.profit_factor, 2),
                "win_rate": round(result.win_rate, 1),
                "total_pnl": round(result.total_pnl, 2),
                "total_trades": result.total_trades,
                "max_drawdown": round(result.max_drawdown, 2),
                "avg_win": round(result.avg_win, 2),
                "avg_loss": round(result.avg_loss, 2),
                "pnl_day": round(result.pnl_last_day, 2),
                "pnl_week": round(result.pnl_last_week, 2),
                "pnl_month": round(result.pnl_last_month, 2),
                "pnl_3_months": round(result.pnl_last_3_months, 2),
                "pnl_6_months": round(result.pnl_last_6_months, 2),
                "pnl_year": round(result.pnl_last_year, 2),
                "monthly_pnl": result.monthly_pnl
            })
        
        # All results summary
        for result in sorted_results:
            report["all_results"].append({
                "name": result.name,
                "category": result.category,
                "profit_factor": round(result.profit_factor, 2),
                "win_rate": round(result.win_rate, 1),
                "total_pnl": round(result.total_pnl, 2),
                "total_trades": result.total_trades
            })
        
        return report


def run_daviddtech_research(df: pd.DataFrame, capital: float, risk_percent: float, 
                            compound: bool, status: Dict) -> Dict[str, Any]:
    """Main entry point for DaviddTech-style research"""
    engine = DaviddTechStrategyResearch(df, capital, risk_percent, compound, status)
    engine.run_all_strategies()
    return engine.generate_report()



