"""
Smart Money Concepts (SMC) Indicators

This module implements institutional trading concepts:
1. Order Blocks - Last opposite candle before significant move
2. Fair Value Gaps (FVG) - Imbalance gaps in price
3. Break of Structure (BOS) - Trend continuation signals
4. Change of Character (CHoCH) - Trend reversal signals
5. Failed Breakouts - Breakout reversals

These concepts are used by institutional traders and can provide
high-probability trading signals when combined with other indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrderBlock:
    """Represents an Order Block"""
    index: int
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    open_price: float
    close_price: float
    tested: bool = False
    strength: str = 'normal'  # 'weak', 'normal', 'strong'


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    index: int
    type: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    filled: bool = False
    strength: str = 'normal'


@dataclass
class StructureBreak:
    """Represents a Break of Structure or Change of Character"""
    index: int
    type: str  # 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    level: float
    confirmed: bool = False


class SMCIndicators:
    """
    Smart Money Concepts indicator calculations.

    Usage:
        smc = SMCIndicators(df)
        order_blocks = smc.detect_order_blocks()
        fvgs = smc.detect_fvg()
        bos = smc.detect_bos()
        choch = smc.detect_choch()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume (optional)
        """
        self.df = df.copy()
        self._calculate_swings()

    def _calculate_swings(self, lookback: int = 5):
        """Calculate swing highs and lows for structure analysis."""
        highs = self.df['high'].values
        lows = self.df['low'].values

        self.swing_highs = []
        self.swing_lows = []

        for i in range(lookback, len(self.df) - lookback):
            # Swing high: higher than lookback bars on both sides
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                self.swing_highs.append({'index': i, 'price': highs[i]})

            # Swing low: lower than lookback bars on both sides
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                self.swing_lows.append({'index': i, 'price': lows[i]})

    def detect_order_blocks(self, lookback: int = 10,
                           min_move_pct: float = 0.02) -> List[OrderBlock]:
        """
        Detect Order Blocks.

        Order Block = Last opposite candle before significant move
        - Bullish OB: Last red candle before strong upward move
        - Bearish OB: Last green candle before strong downward move

        Args:
            lookback: Number of bars to look back for significant moves
            min_move_pct: Minimum percentage move to qualify (2% default)

        Returns:
            List of OrderBlock objects
        """
        order_blocks = []
        closes = self.df['close'].values
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values

        for i in range(lookback, len(self.df)):
            # Check for significant upward move (bullish OB)
            move_up = (closes[i] - closes[i - lookback]) / closes[i - lookback]
            if move_up > min_move_pct:
                # Find last red (bearish) candle before the move
                for j in range(i - 1, max(i - lookback, 0), -1):
                    if closes[j] < opens[j]:  # Red candle
                        # Determine strength based on volume and size
                        candle_size = abs(closes[j] - opens[j])
                        avg_size = np.mean(np.abs(closes[max(0, j-20):j] - opens[max(0, j-20):j]))
                        strength = 'strong' if candle_size > avg_size * 1.5 else ('weak' if candle_size < avg_size * 0.5 else 'normal')

                        order_blocks.append(OrderBlock(
                            index=j,
                            type='bullish',
                            high=highs[j],
                            low=lows[j],
                            open_price=opens[j],
                            close_price=closes[j],
                            strength=strength
                        ))
                        break

            # Check for significant downward move (bearish OB)
            move_down = (closes[i - lookback] - closes[i]) / closes[i - lookback]
            if move_down > min_move_pct:
                # Find last green (bullish) candle before the move
                for j in range(i - 1, max(i - lookback, 0), -1):
                    if closes[j] > opens[j]:  # Green candle
                        candle_size = abs(closes[j] - opens[j])
                        avg_size = np.mean(np.abs(closes[max(0, j-20):j] - opens[max(0, j-20):j]))
                        strength = 'strong' if candle_size > avg_size * 1.5 else ('weak' if candle_size < avg_size * 0.5 else 'normal')

                        order_blocks.append(OrderBlock(
                            index=j,
                            type='bearish',
                            high=highs[j],
                            low=lows[j],
                            open_price=opens[j],
                            close_price=closes[j],
                            strength=strength
                        ))
                        break

        # Mark tested order blocks
        for ob in order_blocks:
            for i in range(ob.index + 1, len(self.df)):
                if ob.type == 'bullish':
                    # Bullish OB tested when price returns to the OB zone
                    if lows[i] <= ob.high and lows[i] >= ob.low:
                        ob.tested = True
                        break
                else:
                    # Bearish OB tested when price returns to the OB zone
                    if highs[i] >= ob.low and highs[i] <= ob.high:
                        ob.tested = True
                        break

        return order_blocks

    def detect_fvg(self, min_gap_pct: float = 0.001) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (FVG).

        FVG = Gap between candle 1's low and candle 3's high (bullish)
        or between candle 1's high and candle 3's low (bearish)

        Args:
            min_gap_pct: Minimum gap size as percentage of price

        Returns:
            List of FairValueGap objects
        """
        fvgs = []
        highs = self.df['high'].values
        lows = self.df['low'].values
        closes = self.df['close'].values

        for i in range(2, len(self.df)):
            # Bullish FVG: candle[i-2] high < candle[i] low
            gap_up = lows[i] - highs[i - 2]
            if gap_up > 0 and gap_up / closes[i] > min_gap_pct:
                # Determine strength based on gap size
                strength = 'strong' if gap_up / closes[i] > min_gap_pct * 3 else 'normal'

                fvgs.append(FairValueGap(
                    index=i - 1,  # Middle candle
                    type='bullish',
                    top=lows[i],
                    bottom=highs[i - 2],
                    strength=strength
                ))

            # Bearish FVG: candle[i-2] low > candle[i] high
            gap_down = lows[i - 2] - highs[i]
            if gap_down > 0 and gap_down / closes[i] > min_gap_pct:
                strength = 'strong' if gap_down / closes[i] > min_gap_pct * 3 else 'normal'

                fvgs.append(FairValueGap(
                    index=i - 1,
                    type='bearish',
                    top=lows[i - 2],
                    bottom=highs[i],
                    strength=strength
                ))

        # Mark filled FVGs
        for fvg in fvgs:
            for i in range(fvg.index + 1, len(self.df)):
                if fvg.type == 'bullish':
                    # Filled when price comes back down into the gap
                    if lows[i] <= fvg.top:
                        fvg.filled = True
                        break
                else:
                    # Filled when price comes back up into the gap
                    if highs[i] >= fvg.bottom:
                        fvg.filled = True
                        break

        return fvgs

    def detect_bos(self) -> List[StructureBreak]:
        """
        Detect Break of Structure (BOS).

        BOS = Price breaks previous swing high (bullish) or swing low (bearish)
        in the direction of the trend (continuation signal).

        Returns:
            List of StructureBreak objects
        """
        bos_list = []
        highs = self.df['high'].values
        lows = self.df['low'].values

        # Determine overall trend
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return bos_list

        # Bullish BOS: break of previous swing high in uptrend
        for i, sh in enumerate(self.swing_highs[1:], 1):
            prev_sh = self.swing_highs[i - 1]
            # Check if there's a higher low between them (uptrend)
            relevant_lows = [sl for sl in self.swing_lows
                           if sl['index'] > prev_sh['index'] and sl['index'] < sh['index']]
            if relevant_lows:
                # BOS confirmed when price breaks above previous swing high
                for j in range(sh['index'], min(sh['index'] + 10, len(self.df))):
                    if highs[j] > prev_sh['price']:
                        bos_list.append(StructureBreak(
                            index=j,
                            type='bos_bullish',
                            level=prev_sh['price'],
                            confirmed=True
                        ))
                        break

        # Bearish BOS: break of previous swing low in downtrend
        for i, sl in enumerate(self.swing_lows[1:], 1):
            prev_sl = self.swing_lows[i - 1]
            # Check if there's a lower high between them (downtrend)
            relevant_highs = [sh for sh in self.swing_highs
                            if sh['index'] > prev_sl['index'] and sh['index'] < sl['index']]
            if relevant_highs:
                # BOS confirmed when price breaks below previous swing low
                for j in range(sl['index'], min(sl['index'] + 10, len(self.df))):
                    if lows[j] < prev_sl['price']:
                        bos_list.append(StructureBreak(
                            index=j,
                            type='bos_bearish',
                            level=prev_sl['price'],
                            confirmed=True
                        ))
                        break

        return bos_list

    def detect_choch(self) -> List[StructureBreak]:
        """
        Detect Change of Character (CHoCH).

        CHoCH = Price breaks structure against the current trend
        (reversal signal).

        Returns:
            List of StructureBreak objects
        """
        choch_list = []
        highs = self.df['high'].values
        lows = self.df['low'].values

        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return choch_list

        # Detect uptrend reversal (CHoCH bearish)
        # Uptrend: higher highs and higher lows
        # CHoCH: break of recent swing low
        for i in range(len(self.swing_highs) - 1):
            sh1 = self.swing_highs[i]
            sh2 = self.swing_highs[i + 1] if i + 1 < len(self.swing_highs) else None

            if sh2 and sh2['price'] > sh1['price']:
                # We're in uptrend - look for swing low between them
                relevant_lows = [sl for sl in self.swing_lows
                               if sl['index'] > sh1['index'] and sl['index'] < sh2['index']]
                if relevant_lows:
                    recent_sl = relevant_lows[-1]
                    # CHoCH if price breaks this swing low
                    for j in range(sh2['index'], min(sh2['index'] + 20, len(self.df))):
                        if lows[j] < recent_sl['price']:
                            choch_list.append(StructureBreak(
                                index=j,
                                type='choch_bearish',
                                level=recent_sl['price'],
                                confirmed=True
                            ))
                            break

        # Detect downtrend reversal (CHoCH bullish)
        for i in range(len(self.swing_lows) - 1):
            sl1 = self.swing_lows[i]
            sl2 = self.swing_lows[i + 1] if i + 1 < len(self.swing_lows) else None

            if sl2 and sl2['price'] < sl1['price']:
                # We're in downtrend - look for swing high between them
                relevant_highs = [sh for sh in self.swing_highs
                                if sh['index'] > sl1['index'] and sh['index'] < sl2['index']]
                if relevant_highs:
                    recent_sh = relevant_highs[-1]
                    # CHoCH if price breaks this swing high
                    for j in range(sl2['index'], min(sl2['index'] + 20, len(self.df))):
                        if highs[j] > recent_sh['price']:
                            choch_list.append(StructureBreak(
                                index=j,
                                type='choch_bullish',
                                level=recent_sh['price'],
                                confirmed=True
                            ))
                            break

        return choch_list

    def detect_failed_breakout(self, lookback: int = 20,
                               return_threshold: float = 0.5) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Failed Breakouts.

        Failed breakout = Price breaks level then reverses
        - Failed breakdown: Broke below support, now back above
        - Failed breakout: Broke above resistance, now back below

        These are +2 score boost for reversal trades per the user's prompt.

        Args:
            lookback: Bars to look back for support/resistance
            return_threshold: How much price must return to qualify (0.5 = 50% of breakout)

        Returns:
            Tuple of (failed_breakdown Series, failed_breakout Series)
        """
        highs = self.df['high'].values
        lows = self.df['low'].values
        closes = self.df['close'].values

        recent_high = pd.Series(highs).rolling(lookback).max().values
        recent_low = pd.Series(lows).rolling(lookback).min().values

        failed_breakdown = np.zeros(len(self.df), dtype=bool)
        failed_breakout = np.zeros(len(self.df), dtype=bool)

        for i in range(lookback + 1, len(self.df)):
            # Failed breakdown: broke below support, now back above
            # Price went below recent low, then came back above it
            if lows[i - 1] < recent_low[i - 2]:  # Broke support
                if closes[i] > recent_low[i - 2]:  # Now back above
                    failed_breakdown[i] = True

            # Failed breakout: broke above resistance, now back below
            if highs[i - 1] > recent_high[i - 2]:  # Broke resistance
                if closes[i] < recent_high[i - 2]:  # Now back below
                    failed_breakout[i] = True

        return pd.Series(failed_breakdown), pd.Series(failed_breakout)

    def get_smc_signals(self, current_index: int) -> Dict:
        """
        Get SMC signals for a specific bar index.

        Returns a dictionary with all SMC signals active at that index,
        useful for strategy decision making.

        Args:
            current_index: Bar index to analyze

        Returns:
            Dictionary with signal information
        """
        order_blocks = self.detect_order_blocks()
        fvgs = self.detect_fvg()
        bos_signals = self.detect_bos()
        choch_signals = self.detect_choch()
        failed_bd, failed_bo = self.detect_failed_breakout()

        close = self.df['close'].iloc[current_index]
        high = self.df['high'].iloc[current_index]
        low = self.df['low'].iloc[current_index]

        signals = {
            'bullish_ob_test': False,
            'bearish_ob_test': False,
            'bullish_fvg_test': False,
            'bearish_fvg_test': False,
            'bos_bullish': False,
            'bos_bearish': False,
            'choch_bullish': False,
            'choch_bearish': False,
            'failed_breakdown': failed_bd.iloc[current_index] if current_index < len(failed_bd) else False,
            'failed_breakout': failed_bo.iloc[current_index] if current_index < len(failed_bo) else False,
            'untested_bullish_ob': [],
            'untested_bearish_ob': [],
            'unfilled_bullish_fvg': [],
            'unfilled_bearish_fvg': [],
        }

        # Check for untested order blocks that price is at
        for ob in order_blocks:
            if ob.index < current_index and not ob.tested:
                if ob.type == 'bullish':
                    signals['untested_bullish_ob'].append(ob)
                    if low <= ob.high and low >= ob.low:
                        signals['bullish_ob_test'] = True
                else:
                    signals['untested_bearish_ob'].append(ob)
                    if high >= ob.low and high <= ob.high:
                        signals['bearish_ob_test'] = True

        # Check for unfilled FVGs that price is at
        for fvg in fvgs:
            if fvg.index < current_index and not fvg.filled:
                if fvg.type == 'bullish':
                    signals['unfilled_bullish_fvg'].append(fvg)
                    if low <= fvg.top and low >= fvg.bottom:
                        signals['bullish_fvg_test'] = True
                else:
                    signals['unfilled_bearish_fvg'].append(fvg)
                    if high >= fvg.bottom and high <= fvg.top:
                        signals['bearish_fvg_test'] = True

        # Check for recent BOS/CHoCH
        for bos in bos_signals:
            if bos.index == current_index or bos.index == current_index - 1:
                if bos.type == 'bos_bullish':
                    signals['bos_bullish'] = True
                elif bos.type == 'bos_bearish':
                    signals['bos_bearish'] = True

        for choch in choch_signals:
            if choch.index == current_index or choch.index == current_index - 1:
                if choch.type == 'choch_bullish':
                    signals['choch_bullish'] = True
                elif choch.type == 'choch_bearish':
                    signals['choch_bearish'] = True

        return signals


# Convenience functions for strategy use
def calculate_smc_score_adjustment(signals: Dict) -> int:
    """
    Calculate score adjustment based on SMC signals.

    Based on user's prompt scoring adjustments:
    - Order Block/FVG: +2 for untested test
    - Failed Breakout: +2 for detected

    Args:
        signals: Dictionary from get_smc_signals()

    Returns:
        Score adjustment value
    """
    adjustment = 0

    # +2 for untested Order Block test
    if signals['bullish_ob_test'] or signals['bearish_ob_test']:
        adjustment += 2

    # +2 for untested FVG test
    if signals['bullish_fvg_test'] or signals['bearish_fvg_test']:
        adjustment += 2

    # +2 for failed breakout
    if signals['failed_breakdown'] or signals['failed_breakout']:
        adjustment += 2

    return adjustment


def get_smc_trade_direction(signals: Dict) -> Optional[str]:
    """
    Determine trade direction based on SMC signals.

    Args:
        signals: Dictionary from get_smc_signals()

    Returns:
        'long', 'short', or None
    """
    bullish_score = 0
    bearish_score = 0

    if signals['bullish_ob_test']:
        bullish_score += 2
    if signals['bearish_ob_test']:
        bearish_score += 2
    if signals['bullish_fvg_test']:
        bullish_score += 2
    if signals['bearish_fvg_test']:
        bearish_score += 2
    if signals['bos_bullish']:
        bullish_score += 1
    if signals['bos_bearish']:
        bearish_score += 1
    if signals['choch_bullish']:
        bullish_score += 2
    if signals['choch_bearish']:
        bearish_score += 2
    if signals['failed_breakdown']:
        bullish_score += 2  # Failed breakdown = bullish reversal
    if signals['failed_breakout']:
        bearish_score += 2  # Failed breakout = bearish reversal

    if bullish_score > bearish_score and bullish_score >= 2:
        return 'long'
    elif bearish_score > bullish_score and bearish_score >= 2:
        return 'short'
    return None
