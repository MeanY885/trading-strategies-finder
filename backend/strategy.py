"""
Sideways Scalper Strategy - Python Implementation for Backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import ta


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # "long" or "short"
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    position_size: float = 0.01


class SidewaysScalperStrategy:
    """
    Python implementation of the BTCGBP Sideways Scalper strategy
    for backtesting and optimization
    """
    
    def __init__(
        self,
        # Market Filter
        adx_length: int = 14,
        adx_threshold: int = 25,
        adx_emergency: int = 35,
        # Bollinger Bands
        bb_length: int = 20,
        bb_mult: float = 2.0,
        # RSI
        rsi_length: int = 14,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        # Risk Management
        sl_fixed: float = 100,
        tp_ratio: float = 1.5,
        # Position
        position_size: float = 0.01,
        commission: float = 0.001  # 0.1%
    ):
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold
        self.adx_emergency = adx_emergency
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.rsi_length = rsi_length
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_fixed = sl_fixed
        self.tp_ratio = tp_ratio
        self.position_size = position_size
        self.commission = commission
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_length,
            window_dev=self.bb_mult
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.rsi_length
        ).rsi()
        
        # ADX
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.adx_length
        )
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # ATR (for reference)
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range()
        
        # Market regime
        df['is_sideways'] = df['adx'] < self.adx_threshold
        df['is_emergency'] = df['adx'] >= self.adx_emergency
        
        # Entry signals
        df['long_signal'] = (
            df['is_sideways'] &
            (df['close'] <= df['bb_lower']) &
            (df['rsi'] < self.rsi_oversold)
        )
        
        df['short_signal'] = (
            df['is_sideways'] &
            (df['close'] >= df['bb_upper']) &
            (df['rsi'] > self.rsi_overbought)
        )
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Returns dict with performance metrics
        """
        df = self.calculate_indicators(df)
        
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        
        # Calculate SL and TP distances
        sl_distance = self.sl_fixed
        tp_distance = sl_distance * self.tp_ratio
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Skip if indicators not ready
            if pd.isna(row['adx']) or pd.isna(row['rsi']) or pd.isna(row['bb_upper']):
                continue
            
            # Check if we have an open position
            if current_trade is not None:
                # Check for exit conditions
                exit_reason = None
                exit_price = None
                
                if current_trade.direction == "long":
                    # Long position exits
                    stop_price = current_trade.entry_price - sl_distance
                    take_profit_price = current_trade.entry_price + tp_distance
                    
                    # Check stop loss (using low of candle)
                    if row['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                    # Check take profit (using high of candle)
                    elif row['high'] >= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = "take_profit"
                    # Emergency exit
                    elif row['is_emergency']:
                        exit_price = row['close']
                        exit_reason = "emergency"
                        
                else:  # short position
                    stop_price = current_trade.entry_price + sl_distance
                    take_profit_price = current_trade.entry_price - tp_distance
                    
                    # Check stop loss (using high of candle)
                    if row['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                    # Check take profit (using low of candle)
                    elif row['low'] <= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = "take_profit"
                    # Emergency exit
                    elif row['is_emergency']:
                        exit_price = row['close']
                        exit_reason = "emergency"
                
                # Process exit if triggered
                if exit_reason:
                    current_trade.exit_time = row['time']
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = exit_reason
                    
                    # Calculate PnL
                    if current_trade.direction == "long":
                        gross_pnl = (exit_price - current_trade.entry_price) * self.position_size
                    else:
                        gross_pnl = (current_trade.entry_price - exit_price) * self.position_size
                    
                    # Subtract commission
                    commission_cost = (current_trade.entry_price + exit_price) * self.position_size * self.commission
                    current_trade.pnl = gross_pnl - commission_cost
                    
                    trades.append(current_trade)
                    current_trade = None
            
            # Check for new entry signals (only if no position)
            if current_trade is None:
                if row['long_signal']:
                    current_trade = Trade(
                        entry_time=row['time'],
                        entry_price=row['close'],
                        direction="long",
                        position_size=self.position_size
                    )
                elif row['short_signal']:
                    current_trade = Trade(
                        entry_time=row['time'],
                        entry_price=row['close'],
                        direction="short",
                        position_size=self.position_size
                    )
        
        # Close any open position at the end
        if current_trade is not None:
            last_row = df.iloc[-1]
            current_trade.exit_time = last_row['time']
            current_trade.exit_price = last_row['close']
            current_trade.exit_reason = "end_of_data"
            
            if current_trade.direction == "long":
                gross_pnl = (last_row['close'] - current_trade.entry_price) * self.position_size
            else:
                gross_pnl = (current_trade.entry_price - last_row['close']) * self.position_size
            
            commission_cost = (current_trade.entry_price + last_row['close']) * self.position_size * self.commission
            current_trade.pnl = gross_pnl - commission_cost
            trades.append(current_trade)
        
        # Calculate metrics
        return self._calculate_metrics(trades, df)
    
    def _calculate_metrics(self, trades: List[Trade], df: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trades"""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "gross_profit": 0,
                "gross_loss": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "trades": []
            }
        
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p <= 0]
        
        total_pnl = sum(pnls)
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10 if gross_profit > 0 else 0)
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 4)  # Annualized for 15min
        else:
            sharpe_ratio = 0
        
        # Trade details for display
        trade_details = []
        for t in trades[-50:]:  # Last 50 trades
            trade_details.append({
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "direction": t.direction,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "exit_reason": t.exit_reason,
                "pnl": round(t.pnl, 2) if t.pnl else None
            })
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "trades": trade_details,
            "exit_reasons": {
                "stop_loss": sum(1 for t in trades if t.exit_reason == "stop_loss"),
                "take_profit": sum(1 for t in trades if t.exit_reason == "take_profit"),
                "emergency": sum(1 for t in trades if t.exit_reason == "emergency"),
                "end_of_data": sum(1 for t in trades if t.exit_reason == "end_of_data")
            }
        }
    
    def get_params(self) -> Dict:
        """Return current parameters as dict"""
        return {
            "adx_length": self.adx_length,
            "adx_threshold": self.adx_threshold,
            "adx_emergency": self.adx_emergency,
            "bb_length": self.bb_length,
            "bb_mult": self.bb_mult,
            "rsi_length": self.rsi_length,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "sl_fixed": self.sl_fixed,
            "tp_ratio": self.tp_ratio
        }



