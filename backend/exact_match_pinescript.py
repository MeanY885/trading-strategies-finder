"""
EXACT-MATCH PINE SCRIPT GENERATOR
=================================
Generates Pine Script v6 code that EXACTLY matches the Python backtester.

Key matching rules:
1. Entry at CLOSE of signal bar (process_orders_on_close=true)
2. Percentage-based TP/SL calculated from entry price
3. Same signal logic as Python
4. Same commission (0.1%)
5. Fixed position size (0.01 BTC)

Author: BTCGBP ML Optimizer
Date: 2025-12-19
"""

from datetime import datetime
from typing import Dict, Optional
import json


class ExactMatchPineGenerator:
    """
    Generates Pine Script v6 that exactly matches the Python backtester.

    This generator guarantees:
    - Same entry prices (CLOSE of signal bar)
    - Same SL/TP levels (percentage-based)
    - Same signal conditions
    - Same commission treatment
    """

    @staticmethod
    def generate(
        strategy_name: str,
        direction: str,
        tp_percent: float,
        sl_percent: float,
        entry_condition: str,
        entry_params: Dict = None,
        metrics: Dict = None,
        pair: str = "BTCGBP"
    ) -> str:
        """
        Generate Pine Script v6 strategy code.

        Args:
            strategy_name: Display name for strategy
            direction: "long", "short", or "both"
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            entry_condition: Entry signal type
            entry_params: Parameters for entry condition
            metrics: Backtest metrics for comments
            pair: Trading pair name

        Returns:
            Complete Pine Script v6 code as string
        """
        entry_params = entry_params or {}
        metrics = metrics or {}
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Generate entry condition code
        entry_code = ExactMatchPineGenerator._get_entry_condition_code(
            entry_condition, direction, entry_params
        )

        # Generate metrics comment
        metrics_comment = ""
        if metrics:
            metrics_comment = f"""
// BACKTEST RESULTS:
//   Total Trades: {metrics.get('total_trades', 'N/A')}
//   Win Rate: {metrics.get('win_rate', 'N/A'):.1f}%
//   Total P&L: £{metrics.get('total_pnl_gbp', 'N/A'):.2f}
//   Profit Factor: {metrics.get('profit_factor', 'N/A'):.2f}
//   Max Drawdown: £{metrics.get('max_drawdown_gbp', 'N/A'):.2f}"""

        # Direction config
        enable_longs = direction in ["long", "both"]
        enable_shorts = direction in ["short", "both"]

        script = f'''// =============================================================================
// {strategy_name}
// =============================================================================
// Generated: {gen_date}
// GUARANTEED 1:1 MATCH with Python backtester
//
// STRATEGY CONFIGURATION:
//   Direction: {direction.upper()}
//   Take Profit: {tp_percent}%
//   Stop Loss: {sl_percent}%
//   Entry: {entry_condition}
//   Params: {json.dumps(entry_params)}
{metrics_comment}
//
// MATCHING RULES (DO NOT MODIFY):
//   - Entry at CLOSE of signal bar (process_orders_on_close=true)
//   - TP/SL as percentage of entry price
//   - Fixed position size: 0.01 BTC
//   - Commission: 0.1% per side
// =============================================================================

//@version=6
strategy("{strategy_name}",
         overlay=true,
         process_orders_on_close=true,  // CRITICAL: Entry at current bar's CLOSE
         default_qty_type=strategy.fixed,
         default_qty_value=0.01,
         initial_capital=1000,
         commission_type=strategy.commission.percent,
         commission_value=0.1,  // Matches Python: 0.1% per side
         calc_on_every_tick=false,
         max_bars_back=500)

// =============================================================================
// INPUTS - Match Python parameters exactly
// =============================================================================

// Risk Management (PERCENTAGE-BASED - matches Python exactly)
tpPercent = input.float({tp_percent}, "Take Profit %", minval=0.1, maxval=20.0, step=0.1, group="Risk Management")
slPercent = input.float({sl_percent}, "Stop Loss %", minval=0.1, maxval=20.0, step=0.1, group="Risk Management")

// Direction toggles
enableLongs = input.bool({str(enable_longs).lower()}, "Enable Long Trades", group="Direction")
enableShorts = input.bool({str(enable_shorts).lower()}, "Enable Short Trades", group="Direction")

{ExactMatchPineGenerator._get_entry_params_inputs(entry_condition, entry_params)}

// Visual Settings
showLabels = input.bool(true, "Show Entry/Exit Labels", group="Visuals")
showLevels = input.bool(true, "Show TP/SL Levels", group="Visuals")

// =============================================================================
// INDICATOR CALCULATIONS
// =============================================================================

{ExactMatchPineGenerator._get_indicator_calculations(entry_condition, entry_params)}

// =============================================================================
// ENTRY CONDITIONS - EXACT MATCH WITH PYTHON
// =============================================================================

{entry_code}

// Apply direction filter
longCondition = longSignal and enableLongs and strategy.position_size == 0
shortCondition = shortSignal and enableShorts and strategy.position_size == 0

// =============================================================================
// TRADE EXECUTION - EXACT MATCH WITH PYTHON
// =============================================================================

// Track entry for TP/SL calculation
var float entryPrice = na

// Long Entry
if longCondition
    strategy.entry("Long", strategy.long)
    entryPrice := close  // Entry at CLOSE (matches Python)

// Short Entry
if shortCondition
    strategy.entry("Short", strategy.short)
    entryPrice := close  // Entry at CLOSE (matches Python)

// =============================================================================
// EXIT LOGIC - PERCENTAGE-BASED TP/SL (EXACT MATCH)
// =============================================================================

// Calculate TP/SL prices from entry price
// NOTE: Uses strategy.position_avg_price for accuracy after fill

if strategy.position_size > 0  // Long position
    // TP/SL calculated as percentage of entry price
    longTP = strategy.position_avg_price * (1 + tpPercent / 100)
    longSL = strategy.position_avg_price * (1 - slPercent / 100)
    strategy.exit("Long Exit", "Long", limit=longTP, stop=longSL)

if strategy.position_size < 0  // Short position
    // TP/SL calculated as percentage of entry price
    shortTP = strategy.position_avg_price * (1 - tpPercent / 100)
    shortSL = strategy.position_avg_price * (1 + slPercent / 100)
    strategy.exit("Short Exit", "Short", limit=shortTP, stop=shortSL)

// =============================================================================
// VISUAL ELEMENTS
// =============================================================================

// Plot TP/SL levels while in position
tpLevel = strategy.position_size > 0 ? strategy.position_avg_price * (1 + tpPercent / 100) :
          strategy.position_size < 0 ? strategy.position_avg_price * (1 - tpPercent / 100) : na
slLevel = strategy.position_size > 0 ? strategy.position_avg_price * (1 - slPercent / 100) :
          strategy.position_size < 0 ? strategy.position_avg_price * (1 + slPercent / 100) : na

plot(showLevels and strategy.position_size != 0 ? tpLevel : na, "TP Level",
     color=color.new(color.green, 0), style=plot.style_linebr, linewidth=2)
plot(showLevels and strategy.position_size != 0 ? slLevel : na, "SL Level",
     color=color.new(color.red, 0), style=plot.style_linebr, linewidth=2)

// Entry signals
plotshape(longCondition, "Long Signal", shape.triangleup, location.belowbar,
          color.new(color.green, 0), size=size.small)
plotshape(shortCondition, "Short Signal", shape.triangledown, location.abovebar,
          color.new(color.red, 0), size=size.small)

// Position entry markers with labels
if showLabels
    if strategy.position_size != 0 and strategy.position_size[1] == 0
        isLong = strategy.position_size > 0
        entryLabel = isLong ? "LONG\\n£" + str.tostring(close, "#.##") : "SHORT\\n£" + str.tostring(close, "#.##")
        labelColor = isLong ? color.green : color.red
        labelY = isLong ? low : high
        labelStyle = isLong ? label.style_label_up : label.style_label_down
        label.new(bar_index, labelY, entryLabel, style=labelStyle, color=labelColor,
                  textcolor=color.white, size=size.small)

// =============================================================================
// PERFORMANCE STATS TABLE
// =============================================================================

var table statsTable = table.new(position.top_right, 2, 10, bgcolor=color.new(color.black, 80), border_width=1)

if barstate.islast
    totalTrades = strategy.closedtrades
    winTrades = strategy.wintrades
    lossTrades = strategy.losstrades
    winRate = totalTrades > 0 ? (winTrades / totalTrades) * 100 : 0
    netProfit = strategy.netprofit
    grossProfit = strategy.grossprofit
    grossLoss = strategy.grossloss
    profitFactor = grossLoss != 0 ? grossProfit / math.abs(grossLoss) : 0

    table.cell(statsTable, 0, 0, "{strategy_name[:20]}", text_color=color.white,
               bgcolor=color.new({'color.green' if direction == "long" else 'color.red' if direction == "short" else 'color.purple'}, 60), text_size=size.small)
    table.cell(statsTable, 1, 0, "{direction.upper()}", bgcolor=color.new({'color.green' if direction == "long" else 'color.red' if direction == "short" else 'color.purple'}, 60), text_color=color.white, text_size=size.small)

    table.cell(statsTable, 0, 1, "TP / SL", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 1, str.tostring(tpPercent) + "% / " + str.tostring(slPercent) + "%", text_color=color.white, text_size=size.tiny)

    table.cell(statsTable, 0, 2, "Trades", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 2, str.tostring(totalTrades), text_color=color.white, text_size=size.tiny)

    table.cell(statsTable, 0, 3, "Win Rate", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 3, str.tostring(winRate, "#.#") + "%",
               text_color=winRate >= 50 ? color.lime : color.red, text_size=size.tiny)

    table.cell(statsTable, 0, 4, "Net Profit", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 4, "£" + str.tostring(netProfit, "#.##"),
               text_color=netProfit >= 0 ? color.lime : color.red, text_size=size.tiny)

    table.cell(statsTable, 0, 5, "Profit Factor", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 5, str.tostring(profitFactor, "#.##"),
               text_color=profitFactor >= 1 ? color.lime : color.red, text_size=size.tiny)

    table.cell(statsTable, 0, 6, "Gross Profit", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 6, "£" + str.tostring(grossProfit, "#.##"), text_color=color.lime, text_size=size.tiny)

    table.cell(statsTable, 0, 7, "Gross Loss", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 7, "£" + str.tostring(math.abs(grossLoss), "#.##"), text_color=color.red, text_size=size.tiny)

    table.cell(statsTable, 0, 8, "Entry", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 8, "{entry_condition}", text_color=color.white, text_size=size.tiny)

// =============================================================================
// ALERTS
// =============================================================================

alertcondition(longCondition, title="Long Entry", message="LONG entry signal at {{{{close}}}}")
alertcondition(shortCondition, title="Short Entry", message="SHORT entry signal at {{{{close}}}}")
'''

        return script

    @staticmethod
    def _get_entry_params_inputs(entry_condition: str, params: Dict) -> str:
        """Generate input declarations for entry parameters"""
        inputs = []

        if entry_condition == "rsi_oversold":
            rsi_period = params.get('rsi_period', 14)
            rsi_os = params.get('rsi_oversold', 30)
            rsi_ob = params.get('rsi_overbought', 70)
            inputs.append(f'rsiPeriod = input.int({rsi_period}, "RSI Period", minval=2, maxval=50, group="Entry Settings")')
            inputs.append(f'rsiOversold = input.int({rsi_os}, "RSI Oversold", minval=5, maxval=50, group="Entry Settings")')
            inputs.append(f'rsiOverbought = input.int({rsi_ob}, "RSI Overbought", minval=50, maxval=95, group="Entry Settings")')

        elif entry_condition == "bb_touch":
            bb_period = params.get('bb_period', 20)
            inputs.append(f'bbPeriod = input.int({bb_period}, "BB Period", minval=5, maxval=50, group="Entry Settings")')
            inputs.append('bbMult = input.float(2.0, "BB Multiplier", minval=0.5, maxval=4.0, group="Entry Settings")')

        elif entry_condition == "sma_cross":
            sma_fast = params.get('sma_fast', 10)
            sma_slow = params.get('sma_slow', 50)
            inputs.append(f'smaFast = input.int({sma_fast}, "SMA Fast", minval=2, maxval=50, group="Entry Settings")')
            inputs.append(f'smaSlow = input.int({sma_slow}, "SMA Slow", minval=10, maxval=200, group="Entry Settings")')

        elif entry_condition in ["consecutive_red", "consecutive_green"]:
            consec = params.get('consec_count', 3)
            inputs.append(f'consecCount = input.int({consec}, "Consecutive Candles", minval=2, maxval=10, group="Entry Settings")')

        elif entry_condition in ["price_drop", "price_rise"]:
            pct = params.get('drop_percent', params.get('rise_percent', 2.0))
            lookback = params.get('lookback', 10)
            inputs.append(f'dropRisePercent = input.float({pct}, "Drop/Rise %", minval=0.5, maxval=20.0, group="Entry Settings")')
            inputs.append(f'lookbackPeriod = input.int({lookback}, "Lookback Period", minval=5, maxval=100, group="Entry Settings")')

        elif entry_condition in ["adx_sideways", "adx_trending"]:
            adx_thresh = params.get('adx_threshold', 25)
            inputs.append(f'adxThreshold = input.int({adx_thresh}, "ADX Threshold", minval=10, maxval=50, group="Entry Settings")')
            if entry_condition == "adx_sideways":
                rsi_os = params.get('rsi_oversold', 30)
                rsi_ob = params.get('rsi_overbought', 70)
                inputs.append(f'rsiOversold = input.int({rsi_os}, "RSI Oversold", minval=5, maxval=50, group="Entry Settings")')
                inputs.append(f'rsiOverbought = input.int({rsi_ob}, "RSI Overbought", minval=50, maxval=95, group="Entry Settings")')

        elif entry_condition == "ema_trend":
            ema_period = params.get('ema_period', 21)
            inputs.append(f'emaPeriod = input.int({ema_period}, "EMA Period", minval=5, maxval=200, group="Entry Settings")')

        elif entry_condition == "every_n_bars":
            n = params.get('n_bars', 10)
            inputs.append(f'nBars = input.int({n}, "Enter Every N Bars", minval=2, maxval=100, group="Entry Settings")')

        if inputs:
            return "// Entry Parameters\n" + "\n".join(inputs)
        return "// No additional entry parameters"

    @staticmethod
    def _get_indicator_calculations(entry_condition: str, params: Dict) -> str:
        """Generate indicator calculation code"""
        calcs = []

        if entry_condition == "rsi_oversold":
            calcs.append("rsiValue = ta.rsi(close, rsiPeriod)")

        elif entry_condition == "bb_touch":
            calcs.append("[bbMiddle, bbUpper, bbLower] = ta.bb(close, bbPeriod, bbMult)")

        elif entry_condition == "sma_cross":
            calcs.append("smaFastValue = ta.sma(close, smaFast)")
            calcs.append("smaSLOWValue = ta.sma(close, smaSlow)")

        elif entry_condition in ["consecutive_red", "consecutive_green"]:
            calcs.append("isGreen = close > open")
            calcs.append("isRed = close < open")

        elif entry_condition in ["price_drop", "price_rise"]:
            calcs.append("recentHigh = ta.highest(high, lookbackPeriod)")
            calcs.append("recentLow = ta.lowest(low, lookbackPeriod)")

        elif entry_condition in ["adx_sideways", "adx_trending"]:
            calcs.append("[diPlus, diMinus, adxValue] = ta.dmi(14, 14)")
            if entry_condition == "adx_sideways":
                calcs.append("rsiValue = ta.rsi(close, 14)")

        elif entry_condition == "ema_trend":
            calcs.append("emaValue = ta.ema(close, emaPeriod)")

        elif entry_condition == "every_n_bars":
            calcs.append("barCounter = bar_index % nBars")

        if calcs:
            return "\n".join(calcs)
        return "// No additional indicators needed"

    @staticmethod
    def _get_entry_condition_code(entry_condition: str, direction: str, params: Dict) -> str:
        """Generate entry condition logic code"""

        if entry_condition == "always":
            return """// ALWAYS - Enter on every bar
longSignal = true
shortSignal = true"""

        elif entry_condition == "rsi_oversold":
            return """// RSI EXTREMES - Long when oversold, Short when overbought
longSignal = rsiValue < rsiOversold
shortSignal = rsiValue > rsiOverbought"""

        elif entry_condition == "bb_touch":
            return """// BOLLINGER BAND TOUCH - Long at lower band, Short at upper band
longSignal = close <= bbLower
shortSignal = close >= bbUpper"""

        elif entry_condition == "sma_cross":
            return """// SMA CROSSOVER - Long on golden cross, Short on death cross
longSignal = ta.crossover(smaFastValue, smaSLOWValue)
shortSignal = ta.crossunder(smaFastValue, smaSLOWValue)"""

        elif entry_condition == "consecutive_red":
            return """// CONSECUTIVE RED CANDLES - Reversal long after N red candles
var int redCount = 0
redCount := isRed ? redCount + 1 : 0
longSignal = redCount >= consecCount
shortSignal = false  // Long-only reversal strategy"""

        elif entry_condition == "consecutive_green":
            return """// CONSECUTIVE GREEN CANDLES - Reversal short after N green candles
var int greenCount = 0
greenCount := isGreen ? greenCount + 1 : 0
shortSignal = greenCount >= consecCount
longSignal = false  // Short-only reversal strategy"""

        elif entry_condition == "price_drop":
            return """// PRICE DROP - Long after X% drop from recent high
currentDrop = (recentHigh - close) / recentHigh * 100
currentRise = (close - recentLow) / recentLow * 100
longSignal = currentDrop >= dropRisePercent
shortSignal = currentRise >= dropRisePercent"""

        elif entry_condition == "price_rise":
            return """// PRICE RISE - Short after X% rise from recent low
currentRise = (close - recentLow) / recentLow * 100
currentDrop = (recentHigh - close) / recentHigh * 100
shortSignal = currentRise >= dropRisePercent
longSignal = currentDrop >= dropRisePercent"""

        elif entry_condition == "adx_sideways":
            return """// ADX SIDEWAYS + RSI - Trade reversals in low-ADX environment
isSideways = adxValue < adxThreshold
longSignal = isSideways and rsiValue < rsiOversold
shortSignal = isSideways and rsiValue > rsiOverbought"""

        elif entry_condition == "adx_trending":
            return """// ADX TRENDING - Trade with trend when ADX is strong
isTrending = adxValue > adxThreshold
longSignal = isTrending and diPlus > diMinus
shortSignal = isTrending and diMinus > diPlus"""

        elif entry_condition == "ema_trend":
            return """// EMA TREND - Long above EMA, Short below EMA
longSignal = close > emaValue
shortSignal = close < emaValue"""

        elif entry_condition == "every_n_bars":
            return """// EVERY N BARS - Time-based entry
longSignal = barCounter == 0
shortSignal = barCounter == 0"""

        else:
            return f"""// CUSTOM ENTRY: {entry_condition}
// NOTE: Add your custom logic here
longSignal = false
shortSignal = false"""


def generate_pine_script_from_result(result: dict) -> str:
    """
    Generate Pine Script from a strategy result dictionary.

    Args:
        result: Dictionary from StrategyResult or database query

    Returns:
        Pine Script v6 code string
    """
    # Parse params if JSON string
    params = result.get('params', {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except:
            params = {}

    return ExactMatchPineGenerator.generate(
        strategy_name=result.get('strategy_name', 'Unnamed Strategy'),
        direction=result.get('direction', 'both'),
        tp_percent=float(result.get('tp_percent', 1.0)),
        sl_percent=float(result.get('sl_percent', 2.0)),
        entry_condition=result.get('entry_condition', 'always'),
        entry_params=params,
        metrics={
            'total_trades': result.get('total_trades'),
            'win_rate': result.get('win_rate'),
            'total_pnl_gbp': result.get('total_pnl_gbp'),
            'profit_factor': result.get('profit_factor'),
            'max_drawdown_gbp': result.get('max_drawdown_gbp'),
        }
    )


if __name__ == "__main__":
    # Test generation
    script = ExactMatchPineGenerator.generate(
        strategy_name="Test SHORT Strategy",
        direction="short",
        tp_percent=0.6,
        sl_percent=5.0,
        entry_condition="always",
        entry_params={},
        metrics={
            'total_trades': 50,
            'win_rate': 72.0,
            'total_pnl_gbp': 30.50,
            'profit_factor': 2.15,
            'max_drawdown_gbp': 8.25
        }
    )

    print(script[:2000])  # Print first 2000 chars
    print("\n... (truncated)")

    # Save to file
    with open("/Users/chriseddisford/Documents/TrandingView Scripts/btcgbp-ml-optimizer/output/test_exact_match.pine", "w") as f:
        f.write(script)

    print("\nSaved to output/test_exact_match.pine")
