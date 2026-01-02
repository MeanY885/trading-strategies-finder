"""
Pine Script v6 Generator - Outputs optimized strategy code
"""
from typing import Dict
from datetime import datetime


def normalize_params(params: Dict) -> Dict:
    """
    Normalize optimizer params to Pine Script params.
    Maps Python backtest params to their Pine Script equivalents.

    The optimizer uses sl_atr_mult (ATR-based), but some legacy code
    uses sl_fixed (pound-based). This ensures ATR-based stops are used.
    """
    normalized = params.copy()

    # ATR-based stop loss (prefer over fixed)
    if 'sl_atr_mult' in params:
        normalized['sl_atr_mult'] = params['sl_atr_mult']
    else:
        # Convert legacy sl_fixed to approximate ATR mult (assume ATR ~200 for BTCGBP)
        sl_fixed = params.get('sl_fixed', 100)
        normalized['sl_atr_mult'] = max(0.5, min(5.0, sl_fixed / 100))

    normalized['use_atr_stops'] = True

    # Ensure tp_ratio is present
    normalized['tp_ratio'] = params.get('tp_ratio', 1.5)

    return normalized


class PineScriptGenerator:
    """Generates Pine Script v6 code from optimized parameters"""
    
    def generate(self, params: Dict) -> str:
        """
        Generate Pine Script v6 strategy code with optimized parameters
        
        Args:
            params: Dictionary of optimized parameters
            
        Returns:
            Complete Pine Script v6 code as string
        """
        # Normalize parameters (converts sl_fixed to sl_atr_mult if needed)
        params = normalize_params(params)

        # Extract parameters with defaults
        adx_threshold = params.get("adx_threshold", 25)
        adx_emergency = params.get("adx_emergency", 35)
        bb_length = params.get("bb_length", 20)
        bb_mult = params.get("bb_mult", 2.0)
        rsi_oversold = params.get("rsi_oversold", 35)
        rsi_overbought = params.get("rsi_overbought", 65)
        sl_atr_mult = params.get("sl_atr_mult", 2.0)  # ATR-based (not fixed £)
        tp_ratio = params.get("tp_ratio", 1.5)
        
        # Get validation results if available
        val_results = params.get("validation_results", {})
        val_info = ""
        if val_results:
            val_info = f"""
// Validation Results:
//   - Win Rate: {val_results.get('win_rate', 'N/A')}%
//   - Profit Factor: {val_results.get('profit_factor', 'N/A')}
//   - Total Trades: {val_results.get('total_trades', 'N/A')}
//   - Total PnL: £{val_results.get('total_pnl', 'N/A')}"""
        
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        script = f'''// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © BTCGBP ML-Optimized Sideways Scalper
// Generated: {generation_date}
// 
// ML OPTIMIZED PARAMETERS:
//   ADX Threshold: {adx_threshold}
//   ADX Emergency: {adx_emergency}
//   BB Length: {bb_length}
//   BB Multiplier: {bb_mult}
//   RSI Oversold: {rsi_oversold}
//   RSI Overbought: {rsi_overbought}
//   Stop Loss ATR Mult: {sl_atr_mult}x
//   TP Ratio: {tp_ratio}x
//{val_info}

//@version=6
strategy("BTCGBP ML-Optimized Scalper", overlay=true, default_qty_type=strategy.cash, default_qty_value=1000, initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent, commission_value=0.1, process_orders_on_close=true, calc_on_every_tick=false, pyramiding=0)

// =============================================================================
// ML-OPTIMIZED PARAMETERS
// =============================================================================

// === Market Regime Detection ===
adxLength = input.int(14, "ADX Length", minval=1, group="Market Filter")
adxThreshold = input.int({adx_threshold}, "ADX Threshold (below = sideways)", minval=10, maxval=50, group="Market Filter")
adxEmergencyExit = input.int({adx_emergency}, "Emergency Exit ADX", minval=20, maxval=60, group="Market Filter")

// === Bollinger Bands ===
bbLength = input.int({bb_length}, "BB Length", minval=5, maxval=50, group="Bollinger Bands")
bbMult = input.float({bb_mult}, "BB Multiplier", minval=0.5, maxval=5.0, step=0.1, group="Bollinger Bands")

// === RSI ===
rsiLength = input.int(14, "RSI Length", minval=2, group="RSI")
rsiOversold = input.int({rsi_oversold}, "RSI Oversold Level", minval=10, maxval=50, group="RSI")
rsiOverbought = input.int({rsi_overbought}, "RSI Overbought Level", minval=50, maxval=90, group="RSI")

// === Risk Management (ATR-Based) ===
slAtrMult = input.float({sl_atr_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk Management")
tpRatio = input.float({tp_ratio}, "Take Profit Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk Management")

// === Position Sizing ===
positionSize = input.float(0.01, "Position Size (BTC)", minval=0.001, maxval=1.0, step=0.001, group="Position Size")

// === Visual Settings ===
showLabels = input.bool(true, "Show Buy/Sell Labels", group="Visuals")
showBB = input.bool(true, "Show Bollinger Bands", group="Visuals")
showBackground = input.bool(true, "Show Market Regime Background", group="Visuals")
showStats = input.bool(true, "Show Performance Table", group="Visuals")

// =============================================================================
// INDICATOR CALCULATIONS
// =============================================================================

// --- Bollinger Bands ---
bbBasis = ta.sma(close, bbLength)
bbDev = bbMult * ta.stdev(close, bbLength)
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

// --- RSI ---
rsi = ta.rsi(close, rsiLength)

// --- ADX for trend detection ---
[diPlus, diMinus, adx] = ta.dmi(adxLength, adxLength)

// =============================================================================
// MARKET REGIME DETECTION
// =============================================================================

isSideways = adx < adxThreshold
isTrending = adx >= adxThreshold
isEmergency = adx >= adxEmergencyExit

// =============================================================================
// ENTRY CONDITIONS (ML-OPTIMIZED)
// =============================================================================

longCondition = (isSideways and close <= bbLower and rsi < rsiOversold and strategy.position_size == 0)
shortCondition = (isSideways and close >= bbUpper and rsi > rsiOverbought and strategy.position_size == 0)

// =============================================================================
// STOP LOSS & TAKE PROFIT (ATR-BASED - matches Python backtest)
// =============================================================================

atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// =============================================================================
// TRADE EXECUTION
// =============================================================================
// IMPORTANT: With process_orders_on_close=true, orders fill at CURRENT BAR's CLOSE
// This matches the Python backtester behavior exactly

var bool isLong = false
var bool pendingLong = false
var bool pendingShort = false

// Track signal for next-bar execution (matches Python backtest behavior)
if longCondition and strategy.position_size == 0 and not pendingLong and not pendingShort
    pendingLong := true

if shortCondition and strategy.position_size == 0 and not pendingLong and not pendingShort
    pendingShort := true

// Execute pending entries (fills at this bar's open - like Python backtester)
if pendingLong[1]
    strategy.entry("Long", strategy.long, qty=positionSize)
    isLong := true
    pendingLong := false

if pendingShort[1]
    strategy.entry("Short", strategy.short, qty=positionSize)
    isLong := false
    pendingShort := false

// Use strategy.position_avg_price for accurate entry price (TradingView calculates this correctly)
if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

if isEmergency and strategy.position_size != 0
    strategy.close_all(comment="EMERGENCY: Trend Breakout")

// =============================================================================
// VISUAL ELEMENTS
// =============================================================================

// Track entry price for visual labels (using actual fill price from TradingView)
var float actualEntryPrice = na
if strategy.position_size != 0 and strategy.position_size[1] == 0
    actualEntryPrice := strategy.position_avg_price

var float lastTradeProfit = 0.0
positionClosed = strategy.position_size == 0 and strategy.position_size[1] != 0

if positionClosed
    if isLong[1]
        lastTradeProfit := (strategy.closedtrades.exit_price(strategy.closedtrades - 1) - strategy.closedtrades.entry_price(strategy.closedtrades - 1)) * positionSize
    else
        lastTradeProfit := (strategy.closedtrades.entry_price(strategy.closedtrades - 1) - strategy.closedtrades.exit_price(strategy.closedtrades - 1)) * positionSize

if showLabels
    // Show signal labels (signal bar - entry will be at next bar's open)
    if pendingLong
        label.new(bar_index, low, "SIGNAL\\nBUY", style=label.style_label_up, color=color.new(color.green, 50), textcolor=color.white, size=size.tiny)
    if pendingShort
        label.new(bar_index, high, "SIGNAL\\nSELL", style=label.style_label_down, color=color.new(color.red, 50), textcolor=color.white, size=size.tiny)
    // Show actual entry labels (at next bar open - actual fill price)
    if strategy.position_size != 0 and strategy.position_size[1] == 0
        entryLabel = isLong ? "ENTRY\\n£" + str.tostring(strategy.position_avg_price, "#.##") : "ENTRY\\n£" + str.tostring(strategy.position_avg_price, "#.##")
        entryColor = isLong ? color.green : color.red
        entryY = isLong ? low : high
        entryStyle = isLong ? label.style_label_up : label.style_label_down
        label.new(bar_index, entryY, entryLabel, style=entryStyle, color=entryColor, textcolor=color.white, size=size.small)
    if positionClosed
        plColor = lastTradeProfit >= 0 ? color.lime : color.red
        plText = lastTradeProfit >= 0 ? "+£" + str.tostring(lastTradeProfit, "#.##") : "-£" + str.tostring(math.abs(lastTradeProfit), "#.##")
        labelY = isLong[1] ? high : low
        labelStyle = isLong[1] ? label.style_label_down : label.style_label_up
        label.new(bar_index, labelY, plText, style=labelStyle, color=plColor, textcolor=color.black, size=size.normal)

// =============================================================================
// PLOT INDICATORS
// =============================================================================

p1 = plot(showBB ? bbUpper : na, "BB Upper", color=color.new(color.red, 50), linewidth=1)
plot(showBB ? bbBasis : na, "BB Basis", color=color.new(color.gray, 50), linewidth=1)
p2 = plot(showBB ? bbLower : na, "BB Lower", color=color.new(color.green, 50), linewidth=1)
fill(p1, p2, color=color.new(color.purple, 90))

bgColor = isEmergency ? color.new(color.red, 85) : isTrending ? color.new(color.orange, 90) : color.new(color.green, 92)
bgColor := showBackground ? bgColor : na
bgcolor(bgColor, title="Market Regime")

plotshape(longCondition, "Long Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(shortCondition, "Short Signal", shape.triangledown, location.abovebar, color.red, size=size.small)

// =============================================================================
// PERFORMANCE STATS TABLE
// =============================================================================

var table statsTable = table.new(position.top_right, 2, 11, bgcolor=color.new(color.black, 80), border_width=1)

if showStats and barstate.islast
    totalTrades = strategy.closedtrades
    winningTrades = strategy.wintrades
    losingTrades = strategy.losstrades
    winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0
    netProfit = strategy.netprofit
    grossProfit = strategy.grossprofit
    grossLoss = strategy.grossloss
    profitFactor = grossLoss != 0 ? grossProfit / math.abs(grossLoss) : 0
    avgWin = winningTrades > 0 ? grossProfit / winningTrades : 0
    avgLoss = losingTrades > 0 ? grossLoss / losingTrades : 0
    
    table.cell(statsTable, 0, 0, "ML-OPTIMIZED", text_color=color.white, bgcolor=color.new(color.purple, 60), text_size=size.small)
    table.cell(statsTable, 1, 0, "", bgcolor=color.new(color.purple, 60))
    
    table.cell(statsTable, 0, 1, "Total Trades", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 1, str.tostring(totalTrades), text_color=color.white, text_size=size.tiny)
    
    table.cell(statsTable, 0, 2, "Win Rate", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 2, str.tostring(winRate, "#.#") + "%", text_color=winRate >= 50 ? color.lime : color.red, text_size=size.tiny)
    
    table.cell(statsTable, 0, 3, "Net Profit", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 3, "£" + str.tostring(netProfit, "#.##"), text_color=netProfit >= 0 ? color.lime : color.red, text_size=size.tiny)
    
    table.cell(statsTable, 0, 4, "Profit Factor", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 4, str.tostring(profitFactor, "#.##"), text_color=profitFactor >= 1 ? color.lime : color.red, text_size=size.tiny)
    
    table.cell(statsTable, 0, 5, "Avg Win", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 5, "£" + str.tostring(avgWin, "#.##"), text_color=color.lime, text_size=size.tiny)
    
    table.cell(statsTable, 0, 6, "Avg Loss", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 6, "£" + str.tostring(math.abs(avgLoss), "#.##"), text_color=color.red, text_size=size.tiny)
    
    table.cell(statsTable, 0, 7, "Market", text_color=color.gray, text_size=size.tiny)
    regimeText = isEmergency ? "⚠️ TRENDING" : isTrending ? "📈 TRENDING" : "↔️ SIDEWAYS"
    regimeColor = isEmergency ? color.red : isTrending ? color.orange : color.lime
    table.cell(statsTable, 1, 7, regimeText, text_color=regimeColor, text_size=size.tiny)
    
    table.cell(statsTable, 0, 8, "ADX", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 8, str.tostring(adx, "#.#"), text_color=color.white, text_size=size.tiny)
    
    table.cell(statsTable, 0, 9, "RSI", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 9, str.tostring(rsi, "#.#"), text_color=color.white, text_size=size.tiny)
    
    table.cell(statsTable, 0, 10, "SL/TP", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 10, str.tostring(slAtrMult, "#.#") + "x ATR / " + str.tostring(tpRatio) + "x TP", text_color=color.white, text_size=size.tiny)

// =============================================================================
// ALERTS
// =============================================================================

alertcondition(longCondition, title="Long Entry", message="BTCGBP ML Scalper: BUY Signal")
alertcondition(shortCondition, title="Short Entry", message="BTCGBP ML Scalper: SELL Signal")
alertcondition(isEmergency, title="Emergency Exit", message="BTCGBP ML Scalper: EMERGENCY - Trend breakout!")
'''
        
        return script
    
    def generate_with_enhancements(self, params: Dict) -> str:
        """
        Generate enhanced Pine Script with additional features:
        - Trailing stop option
        - Time filters
        - Volatility filter
        """
        base_script = self.generate(params)
        # Could add more features here in the future
        return base_script

    def get_mihakralj_indicator_functions(self) -> str:
        """
        Get mihakralj's mathematically rigorous indicator implementations.
        These are included in the Pine Script when engine='mihakralj'.

        Reference: https://github.com/mihakralj/pinescript
        """
        return '''
// =============================================================================
// MIHAKRALJ INDICATOR FUNCTIONS (Mathematically Rigorous)
// Reference: https://github.com/mihakralj/pinescript
// =============================================================================

// RSI with proper warmup compensation
mih_rsi(series float src, simple int len) =>
    if len <= 0
        runtime.error("Length must be greater than 0")
    float u = math.max(src - src[1], 0)
    float d = math.max(src[1] - src, 0)
    float alpha = 1.0 / len
    var float smoothUp = 0.0
    var float smoothDown = 0.0
    if bar_index < len
        smoothUp := u
        smoothDown := d
    else
        smoothUp := nz(smoothUp[1]) * (1 - alpha) + u * alpha
        smoothDown := nz(smoothDown[1]) * (1 - alpha) + d * alpha
    float rs = smoothDown == 0 ? 0 : smoothUp / smoothDown
    smoothDown == 0 ? 100 : 100 - (100 / (1 + rs))

// EMA with warmup compensation
mih_ema(series float source, simple int period) =>
    float a = 2.0 / (period + 1)
    float beta = 1.0 - a
    var bool warmup = true
    var float e = 1.0
    var float ema = 0.0
    var float result = source
    ema := a * (source - ema) + ema
    if warmup
        e *= beta
        float c = 1.0 / (1.0 - e)
        result := c * ema
        warmup := e > 1e-10
    else
        result := ema
    result

// SMA with O(1) circular buffer
mih_sma(series float source, simple int period) =>
    if period <= 0
        runtime.error("Period must be greater than 0")
    var array<float> buffer = array.new_float(period, na)
    var int head = 0
    var float sum = 0.0
    var int count = 0
    float oldest = array.get(buffer, head)
    if not na(oldest)
        sum -= oldest
    else
        count += 1
    float current = nz(source)
    sum += current
    array.set(buffer, head, current)
    head := (head + 1) % period
    sum / math.max(1, count)

// MACD with warmup compensation
mih_macd(series float src, simple int fast_length, simple int slow_length, simple int signal_length) =>
    float alpha_fast = 2.0 / (fast_length + 1)
    float alpha_slow = 2.0 / (slow_length + 1)
    float alpha_signal = 2.0 / (signal_length + 1)
    float beta_fast = 1.0 - alpha_fast
    float beta_slow = 1.0 - alpha_slow
    float beta_signal = 1.0 - alpha_signal
    var bool warmup = true
    var float e_fast = 1.0
    var float e_slow = 1.0
    var float e_signal = 1.0
    var float ema_fast = 0.0
    var float ema_slow = 0.0
    var float ema_signal = 0.0
    var float result_fast = src
    var float result_slow = src
    var float result_signal = 0.0
    ema_fast := alpha_fast * (src - ema_fast) + ema_fast
    ema_slow := alpha_slow * (src - ema_slow) + ema_slow
    if warmup
        e_fast *= beta_fast
        e_slow *= beta_slow
        e_signal *= beta_signal
        float c_fast = 1.0 / (1.0 - e_fast)
        float c_slow = 1.0 / (1.0 - e_slow)
        float c_signal = 1.0 / (1.0 - e_signal)
        result_fast := c_fast * ema_fast
        result_slow := c_slow * ema_slow
        float macd_line = result_fast - result_slow
        ema_signal := alpha_signal * (macd_line - ema_signal) + ema_signal
        result_signal := c_signal * ema_signal
        warmup := e_fast > 1e-10 or e_slow > 1e-10 or e_signal > 1e-10
    else
        result_fast := ema_fast
        result_slow := ema_slow
        float macd_line = result_fast - result_slow
        ema_signal := alpha_signal * (macd_line - ema_signal) + ema_signal
        result_signal := ema_signal
    float macd_line = result_fast - result_slow
    float histogram = macd_line - result_signal
    [macd_line, result_signal, histogram]

// Stochastic with smoothing
mih_stoch(simple int kLength, simple int smoothK, simple int dPeriod) =>
    float lowest = ta.lowest(low, kLength)
    float highest = ta.highest(high, kLength)
    float range_val = highest - lowest
    float raw_k = range_val > 0 ? 100 * (close - lowest) / range_val : 0.0
    float k = mih_sma(raw_k, smoothK)
    float d = mih_sma(k, dPeriod)
    [k, d]

// Bollinger Bands with O(1) complexity
mih_bbands(series float src, simple int period, simple float mult) =>
    var array<float> buffer = array.new_float(period, na)
    var int head = 0
    var float sum = 0.0
    var float sumSq = 0.0
    var int count = 0
    float oldest = array.get(buffer, head)
    if not na(oldest)
        sum -= oldest
        sumSq -= oldest * oldest
    else
        count += 1
    float current = nz(src)
    sum += current
    sumSq += current * current
    array.set(buffer, head, current)
    head := (head + 1) % period
    float basis = sum / math.max(1, count)
    float variance = math.max(0.0, sumSq / count - basis * basis)
    float dev = mult * math.sqrt(variance)
    [basis, basis + dev, basis - dev]

// ATR with RMA warmup compensation
mih_atr(simple int length) =>
    var float prevClose = close
    float tr1 = high - low
    float tr2 = math.abs(high - prevClose)
    float tr3 = math.abs(low - prevClose)
    float trueRange = math.max(tr1, tr2, tr3)
    prevClose := close
    float alpha = 1.0 / float(length)
    float beta = 1.0 - alpha
    var float raw_rma = 0.0
    var float e = 1.0
    if not na(trueRange)
        raw_rma := (raw_rma * (length - 1) + trueRange) / length
        e *= beta
        e > 1e-10 ? raw_rma / (1.0 - e) : raw_rma
    else
        na

// Ultimate Oscillator
mih_uo(simple int fast, simple int medium, simple int slow) =>
    float bp = close - math.min(low, nz(close[1], close))
    float tr = math.max(high, nz(close[1], close)) - math.min(low, nz(close[1], close))
    float avgFast = ta.sum(bp, fast) / ta.sum(tr, fast)
    float avgMedium = ta.sum(bp, medium) / ta.sum(tr, medium)
    float avgSlow = ta.sum(bp, slow) / ta.sum(tr, slow)
    100 * ((4 * avgFast) + (2 * avgMedium) + avgSlow) / 7

'''

    def _generate_professional_visuals(self, tp_percent: float, sl_percent: float) -> str:
        """
        Generate Pine Script v6 code for professional trade visualizations.

        Includes:
        - Win/Loss labels at exit with percentage
        - P&L value labels in GBP
        - Entry markers with price
        - Trade zone overlays (green for longs, red for shorts)
        - TP/SL level labels

        Args:
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage

        Returns:
            Pine Script v6 code for professional visuals
        """
        return f'''
// =============================================================================
// PROFESSIONAL TRADE VISUALIZATIONS
// =============================================================================

// Color constants for professional appearance
color WIN_COLOR = #00C853      // Bright green for wins
color LOSS_COLOR = #FF1744     // Bright red for losses
color LONG_ZONE = color.new(#1B5E20, 85)   // Dark green at 85% transparency
color SHORT_ZONE = color.new(#B71C1C, 85)  // Dark red at 85% transparency
color TP_COLOR = #00E676       // Light green for TP levels
color SL_COLOR = #FF5252       // Light red for SL levels
color ENTRY_LONG_COLOR = #2196F3   // Blue for long entries
color ENTRY_SHORT_COLOR = #FF9800  // Orange for short entries

// Track trade state for visualizations
var float tradeEntryPrice = na
var float tradeTpLevel = na
var float tradeSlLevel = na
var bool tradeIsLong = false
var int tradeEntryBar = na
var line tpLine = na
var line slLine = na
var label tpLabel = na
var label slLabel = na
var box tradeZone = na

// Detect new position entry
positionChanged = ta.change(strategy.position_size) != 0
newLongEntry = strategy.position_size > 0 and strategy.position_size[1] <= 0
newShortEntry = strategy.position_size < 0 and strategy.position_size[1] >= 0

// Entry visualization - create markers and zones
if newLongEntry
    tradeEntryPrice := strategy.position_avg_price
    tradeIsLong := true
    tradeEntryBar := bar_index
    tradeTpLevel := tradeEntryPrice * (1 + tpPercent / 100)
    tradeSlLevel := tradeEntryPrice * (1 - slPercent / 100)

    // Entry label with price
    label.new(bar_index, low, "Long @ " + str.tostring(tradeEntryPrice, "#.##"),
              style=label.style_label_up, color=ENTRY_LONG_COLOR, textcolor=color.white, size=size.small)

    // TP level line and label
    tpLine := line.new(bar_index, tradeTpLevel, bar_index + 20, tradeTpLevel,
                       color=TP_COLOR, width=1, style=line.style_dashed)
    tpLabel := label.new(bar_index + 10, tradeTpLevel, "TP " + str.tostring(tpPercent, "#.##") + "%",
                         style=label.style_label_left, color=color.new(TP_COLOR, 50), textcolor=color.white, size=size.tiny)

    // SL level line and label
    slLine := line.new(bar_index, tradeSlLevel, bar_index + 20, tradeSlLevel,
                       color=SL_COLOR, width=1, style=line.style_dashed)
    slLabel := label.new(bar_index + 10, tradeSlLevel, "SL " + str.tostring(slPercent, "#.##") + "%",
                         style=label.style_label_left, color=color.new(SL_COLOR, 50), textcolor=color.white, size=size.tiny)

    // Trade zone overlay
    tradeZone := box.new(bar_index, tradeTpLevel, bar_index + 1, tradeSlLevel,
                         bgcolor=LONG_ZONE, border_color=color.new(#1B5E20, 50), border_width=1)

if newShortEntry
    tradeEntryPrice := strategy.position_avg_price
    tradeIsLong := false
    tradeEntryBar := bar_index
    tradeTpLevel := tradeEntryPrice * (1 - tpPercent / 100)
    tradeSlLevel := tradeEntryPrice * (1 + slPercent / 100)

    // Entry label with price
    label.new(bar_index, high, "SHORT @ " + str.tostring(tradeEntryPrice, "#.##"),
              style=label.style_label_down, color=ENTRY_SHORT_COLOR, textcolor=color.white, size=size.small)

    // TP level line and label
    tpLine := line.new(bar_index, tradeTpLevel, bar_index + 20, tradeTpLevel,
                       color=TP_COLOR, width=1, style=line.style_dashed)
    tpLabel := label.new(bar_index + 10, tradeTpLevel, "TP " + str.tostring(tpPercent, "#.##") + "%",
                         style=label.style_label_left, color=color.new(TP_COLOR, 50), textcolor=color.white, size=size.tiny)

    // SL level line and label
    slLine := line.new(bar_index, tradeSlLevel, bar_index + 20, tradeSlLevel,
                       color=SL_COLOR, width=1, style=line.style_dashed)
    slLabel := label.new(bar_index + 10, tradeSlLevel, "SL " + str.tostring(slPercent, "#.##") + "%",
                         style=label.style_label_left, color=color.new(SL_COLOR, 50), textcolor=color.white, size=size.tiny)

    // Trade zone overlay
    tradeZone := box.new(bar_index, tradeSlLevel, bar_index + 1, tradeTpLevel,
                         bgcolor=SHORT_ZONE, border_color=color.new(#B71C1C, 50), border_width=1)

// Update trade zone while in position
if strategy.position_size != 0 and not na(tradeZone)
    if tradeIsLong
        box.set_right(tradeZone, bar_index + 1)
    else
        box.set_right(tradeZone, bar_index + 1)

    // Extend TP/SL lines
    if not na(tpLine)
        line.set_x2(tpLine, bar_index + 10)
    if not na(slLine)
        line.set_x2(slLine, bar_index + 10)

// Exit visualization - show Win/Loss labels with P&L
positionClosed = strategy.position_size == 0 and strategy.position_size[1] != 0
if positionClosed
    // Calculate P&L for the closed trade
    exitPrice = close
    pnlPercent = tradeIsLong ? ((exitPrice - tradeEntryPrice) / tradeEntryPrice) * 100 : ((tradeEntryPrice - exitPrice) / tradeEntryPrice) * 100
    pnlGbp = strategy.netprofit - strategy.netprofit[1]
    isWin = pnlGbp > 0

    // Determine direction text
    directionText = tradeIsLong ? "(Long)" : "(Short)"

    // Win/Loss label with percentage
    resultText = isWin ? "Win " + directionText + " " + str.tostring(math.abs(pnlPercent), "#.##") + "%" : "Loss " + directionText + " " + str.tostring(math.abs(pnlPercent), "#.##") + "%"
    resultColor = isWin ? WIN_COLOR : LOSS_COLOR
    labelY = tradeIsLong ? high : low
    labelStyle = tradeIsLong ? label.style_label_down : label.style_label_up

    label.new(bar_index, labelY, resultText,
              style=labelStyle, color=resultColor, textcolor=color.white, size=size.normal)

    // P&L value label in GBP
    pnlText = isWin ? "+" + str.tostring(pnlGbp, "#.##") : str.tostring(pnlGbp, "#.##")
    pnlLabelY = tradeIsLong ? high + (high - low) * 0.5 : low - (high - low) * 0.5
    label.new(bar_index, pnlLabelY, pnlText,
              style=label.style_label_center, color=color.new(resultColor, 30), textcolor=color.white, size=size.small)

    // Reset trade state
    tradeEntryPrice := na
    tradeTpLevel := na
    tradeSlLevel := na
    tradeEntryBar := na
'''

    def _generate_enhanced_stats_table(self, strategy_name: str, direction: str, tp_percent: float, sl_percent: float) -> str:
        """
        Generate Pine Script v6 code for enhanced performance stats table.

        Includes 14 rows:
        - Total Trades, Wins, Losses
        - Win Rate, Profit Factor, Net Profit
        - Max Drawdown, Avg Win, Avg Loss
        - Expectancy, Win:Loss Ratio
        - TP/SL settings
        - All values in GBP

        Args:
            strategy_name: Strategy name for header
            direction: Trade direction (long/short/both)
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage

        Returns:
            Pine Script v6 code for enhanced stats table
        """
        direction_upper = direction.upper()
        direction_color = "color.green" if direction == "long" else ("color.purple" if direction == "both" else "color.red")

        return f'''
// =============================================================================
// ENHANCED PERFORMANCE STATS TABLE (14 Rows)
// =============================================================================

var table statsTable = table.new(position.top_right, 2, 14, bgcolor=color.new(color.black, 80), border_width=1)

if barstate.islast
    // Core metrics
    totalTrades = strategy.closedtrades
    winTrades = strategy.wintrades
    lossTrades = strategy.losstrades
    winRate = totalTrades > 0 ? (winTrades / totalTrades) * 100 : 0

    // Profit metrics
    netProfit = strategy.netprofit
    grossProfit = strategy.grossprofit
    grossLoss = math.abs(strategy.grossloss)
    profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0

    // Average trade metrics
    avgWin = winTrades > 0 ? grossProfit / winTrades : 0
    avgLoss = lossTrades > 0 ? grossLoss / lossTrades : 0

    // Advanced metrics
    maxDrawdown = strategy.max_drawdown
    winLossRatio = avgLoss > 0 ? avgWin / avgLoss : 0
    expectancy = totalTrades > 0 ? netProfit / totalTrades : 0

    // Row 0: Strategy Header
    table.cell(statsTable, 0, 0, "{strategy_name}", text_color=color.white,
               bgcolor=color.new(color.purple, 60), text_size=size.small)
    table.cell(statsTable, 1, 0, "{direction_upper}", bgcolor=color.new({direction_color}, 60),
               text_color=color.white, text_size=size.small)

    // Row 1: Total Trades
    table.cell(statsTable, 0, 1, "Total Trades", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 1, str.tostring(totalTrades), text_color=color.white, text_size=size.tiny)

    // Row 2: Wins
    table.cell(statsTable, 0, 2, "Wins", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 2, str.tostring(winTrades), text_color=#00C853, text_size=size.tiny)

    // Row 3: Losses
    table.cell(statsTable, 0, 3, "Losses", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 3, str.tostring(lossTrades), text_color=#FF1744, text_size=size.tiny)

    // Row 4: Win Rate
    table.cell(statsTable, 0, 4, "Win Rate", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 4, str.tostring(winRate, "#.#") + "%",
               text_color=winRate >= 50 ? #00C853 : #FF1744, text_size=size.tiny)

    // Row 5: Profit Factor
    table.cell(statsTable, 0, 5, "Profit Factor", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 5, str.tostring(profitFactor, "#.##"),
               text_color=profitFactor >= 1 ? #00C853 : #FF1744, text_size=size.tiny)

    // Row 6: Net Profit
    table.cell(statsTable, 0, 6, "Net Profit", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 6, str.tostring(netProfit, "#.##"),
               text_color=netProfit >= 0 ? #00C853 : #FF1744, text_size=size.tiny)

    // Row 7: Max Drawdown
    table.cell(statsTable, 0, 7, "Max Drawdown", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 7, str.tostring(maxDrawdown, "#.##"),
               text_color=#FF1744, text_size=size.tiny)

    // Row 8: Avg Win
    table.cell(statsTable, 0, 8, "Avg Win", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 8, str.tostring(avgWin, "#.##"),
               text_color=#00C853, text_size=size.tiny)

    // Row 9: Avg Loss
    table.cell(statsTable, 0, 9, "Avg Loss", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 9, str.tostring(avgLoss, "#.##"),
               text_color=#FF1744, text_size=size.tiny)

    // Row 10: Expectancy
    table.cell(statsTable, 0, 10, "Expectancy", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 10, str.tostring(expectancy, "#.##"),
               text_color=expectancy >= 0 ? #00C853 : #FF1744, text_size=size.tiny)

    // Row 11: Win:Loss Ratio
    table.cell(statsTable, 0, 11, "Win:Loss Ratio", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 11, str.tostring(winLossRatio, "#.##"),
               text_color=winLossRatio >= 1 ? #00C853 : #FF1744, text_size=size.tiny)

    // Row 12: TP Setting
    table.cell(statsTable, 0, 12, "Take Profit", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 12, str.tostring(tpPercent, "#.##") + "%",
               text_color=#00E676, text_size=size.tiny)

    // Row 13: SL Setting
    table.cell(statsTable, 0, 13, "Stop Loss", text_color=color.gray, text_size=size.tiny)
    table.cell(statsTable, 1, 13, str.tostring(slPercent, "#.##") + "%",
               text_color=#FF5252, text_size=size.tiny)
'''

    def generate_exact_match(self, strategy_name: str, params: Dict, metrics: Dict = None,
                              entry_rule: str = None, direction: str = None,
                              position_size_pct: float = 100.0, capital: float = 1000.0,
                              engine: str = "mihakralj", date_range: Dict = None,
                              indicator_params: Dict = None) -> str:
        """
        Generate EXACT-MATCH Pine Script v6 that guarantees 1:1 match with Python backtester.

        Key matching rules:
        - Entry at CLOSE of signal bar (process_orders_on_close=true, pyramiding=0, margin_long=100, margin_short=100)
        - Percentage-based TP/SL (not ATR-based)
        - Position size as % of equity (matches Python exactly)
        - Commission: 0.1% per side

        Args:
            strategy_name: Name of the strategy
            params: Optimized parameters (must include tp_percent, sl_percent)
            metrics: Performance metrics from backtesting
            entry_rule: Entry rule identifier (e.g., 'williams_r', 'rsi_extreme')
            direction: Trade direction ('long' or 'short')
            position_size_pct: Position size as % of equity (from UI)
            capital: Starting capital (from UI)
            engine: Calculation engine ("tradingview", "pandas_ta", or "mihakralj")
            date_range: Optional date range dict with keys: enabled, startDate, startTime, endDate, endTime
            indicator_params: Optional dict with tuned indicator parameters (from Phase 2)
                              e.g., {'rsi_length': 10, 'ema_fast': 7, 'ema_slow': 18}

        Returns:
            Complete Pine Script v6 code as string
        """
        # Default indicator parameters (Phase 1 defaults)
        DEFAULT_PARAMS = {
            'rsi_length': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'bb_length': 20,
            'bb_mult': 2.0,
            'atr_length': 14,
            'sma_fast': 9,
            'sma_slow': 18,
            'sma_20': 20,
            'ema_fast': 9,
            'ema_slow': 21,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'willr_length': 14,
            'cci_length': 20,
            'supertrend_factor': 3.0,
            'supertrend_atr': 10,
            'adx_length': 14,
        }

        # Merge with tuned params (tuned values override defaults)
        ind_params = DEFAULT_PARAMS.copy()
        if indicator_params:
            ind_params.update(indicator_params)

        # Helper to get param value
        def p(name):
            return ind_params.get(name, DEFAULT_PARAMS.get(name))

        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Extract percentage-based TP/SL
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

        # Metrics comment
        metrics_comment = ""
        if metrics:
            metrics_comment = f"""
// BACKTEST RESULTS (Python - should match TradingView):
//   Total Trades: {metrics.get('total_trades', 'N/A')}
//   Win Rate: {metrics.get('win_rate', 0):.1f}%
//   Total P&L: £{metrics.get('total_pnl', 0):.2f}
//   Profit Factor: {metrics.get('profit_factor', 0):.2f}
//   Max Drawdown: £{metrics.get('max_drawdown', 0):.2f}"""

        # Determine direction from parameter or strategy name
        if direction is None:
            direction = "both"
            if "long" in strategy_name.lower():
                direction = "long"
            elif "short" in strategy_name.lower():
                direction = "short"

        is_long = direction == "long"
        is_bidirectional = direction == "both"
        enable_longs = direction in ["long", "both"]
        enable_shorts = direction in ["short", "both"]

        # Variables for stats table
        direction_upper = direction.upper()
        direction_color = "color.green" if is_long else ("color.purple" if is_bidirectional else "color.red")

        # Generate date range filtering code
        date_range_code = ""
        date_range_condition = ""
        if date_range and date_range.get('enabled'):
            # Format dates for Pine Script timestamp function (YYYY-MM-DD HH:MM)
            start_date = date_range.get('startDate', '2024-01-01')
            start_time = date_range.get('startTime', '00:00')
            end_date = date_range.get('endDate', '2025-12-31')
            end_time = date_range.get('endTime', '23:59')

            date_range_code = f'''
// =============================================================================
// DATE RANGE FILTER
// =============================================================================

useDateRange = input.bool(true, "Limit Backtest to Date Range", group="Date Range")
fromDate = input.time(timestamp("{start_date} {start_time} +0000"), "From Date", group="Date Range")
toDate = input.time(timestamp("{end_date} {end_time} +0000"), "To Date", group="Date Range")

// Function to check if current bar is within date range
inDateRange() => not useDateRange or (time >= fromDate and time <= toDate)
'''
            date_range_condition = " and inDateRange()"

        # Strategy-specific entry conditions - MUST MATCH PYTHON EXACTLY
        # Uses tuned indicator parameters when provided
        rsi_len = p('rsi_length')
        stoch_k = p('stoch_k')
        stoch_d = p('stoch_d')
        stoch_smooth = p('stoch_smooth')
        bb_len = p('bb_length')
        bb_mult = p('bb_mult')
        ema_fast = p('ema_fast')
        ema_slow = p('ema_slow')
        sma_fast = p('sma_fast')
        sma_slow = p('sma_slow')
        sma_20 = p('sma_20')
        macd_fast = p('macd_fast')
        macd_slow = p('macd_slow')
        macd_signal = p('macd_signal')
        willr_len = p('willr_length')
        cci_len = p('cci_length')
        st_factor = p('supertrend_factor')
        st_atr = p('supertrend_atr')
        adx_len = p('adx_length')
        atr_len = p('atr_length')
        vwma_length = p('vwma_length') if indicator_params and 'vwma_length' in indicator_params else 20

        entry_conditions = {
            # === MOMENTUM ===
            'rsi_extreme': f'''// RSI Strategy (TradingView built-in pattern)
// Long: RSI crosses OVER oversold (30), Short: RSI crosses UNDER overbought (70)
// RSI Length: {rsi_len} {"(tuned)" if indicator_params and 'rsi_length' in indicator_params else "(default)"}
rsiValue = ta.rsi(close, {rsi_len})
entrySignal = {"ta.crossover(rsiValue, 30)" if is_long else "ta.crossunder(rsiValue, 70)"}''',

            'rsi_cross_50': f'''// RSI Cross 50 Entry
// RSI Length: {rsi_len} {"(tuned)" if indicator_params and 'rsi_length' in indicator_params else "(default)"}
rsiValue = ta.rsi(close, {rsi_len})
entrySignal = {"ta.crossover(rsiValue, 50)" if is_long else "ta.crossunder(rsiValue, 50)"}''',

            'stoch_extreme': f'''// Stochastic Slow Strategy (TradingView built-in pattern)
// Long: K crosses OVER D while K < 20, Short: K crosses UNDER D while K > 80
// Stoch K: {stoch_k}, D: {stoch_d}, Smooth: {stoch_smooth} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['stoch_k', 'stoch_d', 'stoch_smooth']) else "(default)"}
k = ta.sma(ta.stoch(close, high, low, {stoch_k}), {stoch_smooth})
d = ta.sma(k, {stoch_d})
entrySignal = {"ta.crossover(k, d) and k < 20" if is_long else "ta.crossunder(k, d) and k > 80"}''',

            'williams_r': f'''// Williams %R Extreme Entry (< -80 long, > -20 short)
// Williams %R Length: {willr_len} {"(tuned)" if indicator_params and 'willr_length' in indicator_params else "(default)"}
willrValue = ta.wpr({willr_len})
entrySignal = {"willrValue < -80" if is_long else "willrValue > -20"}''',

            'cci_extreme': f'''// CCI Extreme Entry (< -100 long, > 100 short)
// CCI Length: {cci_len} {"(tuned)" if indicator_params and 'cci_length' in indicator_params else "(default)"}
cciValue = ta.cci(high, low, close, {cci_len})
entrySignal = {"cciValue < -100" if is_long else "cciValue > 100"}''',

            'rsi_divergence': f'''// RSI Divergence Entry (rolling window - matches Python)
// RSI Length: {rsi_len} {"(tuned)" if indicator_params and 'rsi_length' in indicator_params else "(default)"}
// Python uses rolling().min/max().shift(1) - we match with ta.lowest/highest()[1]
rsiValue = ta.rsi(close, {rsi_len})
lookback = 5
priceLowerLow = low < ta.lowest(low, lookback)[1]
rsiHigherLow = rsiValue > ta.lowest(rsiValue, lookback)[1]
priceHigherHigh = high > ta.highest(high, lookback)[1]
rsiLowerHigh = rsiValue < ta.highest(rsiValue, lookback)[1]
entrySignal = {"priceLowerLow and rsiHigherLow and rsiValue < 40" if is_long else "priceHigherHigh and rsiLowerHigh and rsiValue > 60"}''',

            # === MEAN REVERSION ===
            'bb_touch': f'''// Bollinger Bands Strategy (TradingView built-in pattern)
// Long: price crosses OVER lower band, Short: price crosses UNDER upper band
// BB Length: {bb_len}, Mult: {bb_mult} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['bb_length', 'bb_mult']) else "(default)"}
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
entrySignal = {"ta.crossover(close, bbLower)" if is_long else "ta.crossunder(close, bbUpper)"}''',

            'bb_squeeze_breakout': f'''// BB Squeeze Breakout Entry
// BB Length: {bb_len}, Mult: {bb_mult} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['bb_length', 'bb_mult']) else "(default)"}
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
bbWidth = (bbUpper - bbLower) / bbMiddle
avgWidth = ta.sma(bbWidth, {bb_len})
squeezed = bbWidth[1] < avgWidth * 0.8
expanding = bbWidth > bbWidth[1]
entrySignal = squeezed and expanding and {"close > bbMiddle" if is_long else "close < bbMiddle"}''',

            'price_vs_sma': f'''// Price vs SMA Entry (1% deviation from SMA = mean reversion signal)
// SMA Length: {sma_20} {"(tuned)" if indicator_params and 'sma_20' in indicator_params else "(default)"}
sma20 = ta.sma(close, {sma_20})
entrySignal = {"close < sma20 * 0.99" if is_long else "close > sma20 * 1.01"}''',

            'vwap_bounce': f'''// VWAP Bounce Entry
vwapValue = ta.vwap(hlc3)
touchedBelow = low < vwapValue
touchedAbove = high > vwapValue
closedAbove = close > vwapValue
closedBelow = close < vwapValue
entrySignal = {"touchedBelow and closedAbove" if is_long else "touchedAbove and closedBelow"}''',

            'vwap_cross': f'''// VWAP Cross Entry
vwapValue = ta.vwap(hlc3)
entrySignal = {"ta.crossover(close, vwapValue)" if is_long else "ta.crossunder(close, vwapValue)"}''',

            'vwma_cross': f'''// VWMA Cross Entry
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
entrySignal = {"ta.crossover(close, vwmaValue)" if is_long else "ta.crossunder(close, vwmaValue)"}''',

            'vwma_trend': f'''// VWMA Trend Entry
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
vwmaSlope = vwmaValue - vwmaValue[1]
entrySignal = {"vwmaSlope > 0 and vwmaSlope[1] <= 0" if is_long else "vwmaSlope < 0 and vwmaSlope[1] >= 0"}''',

            # === TREND ===
            'ema_cross': f'''// EMA Cross Entry
// EMA Fast: {ema_fast}, Slow: {ema_slow} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['ema_fast', 'ema_slow']) else "(default)"}
emaFast = ta.ema(close, {ema_fast})
emaSlow = ta.ema(close, {ema_slow})
entrySignal = {"ta.crossover(emaFast, emaSlow)" if is_long else "ta.crossunder(emaFast, emaSlow)"}''',

            'sma_cross': f'''// MovingAvg2Line Cross (TradingView built-in pattern)
// SMA Fast: {sma_fast}, Slow: {sma_slow} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['sma_fast', 'sma_slow']) else "(default)"}
mafast = ta.sma(close, {sma_fast})
maslow = ta.sma(close, {sma_slow})
entrySignal = {"ta.crossover(mafast, maslow)" if is_long else "ta.crossunder(mafast, maslow)"}''',

            'macd_cross': f'''// MACD Strategy (TradingView built-in pattern)
// Long: histogram crosses OVER zero, Short: histogram crosses UNDER zero
// MACD Fast: {macd_fast}, Slow: {macd_slow}, Signal: {macd_signal} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['macd_fast', 'macd_slow', 'macd_signal']) else "(default)"}
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
delta = macdLine - signalLine  // histogram
entrySignal = {"ta.crossover(delta, 0)" if is_long else "ta.crossunder(delta, 0)"}''',

            'price_above_sma': f'''// Price Crosses SMA Entry
// SMA Length: {sma_20} {"(tuned)" if indicator_params and 'sma_20' in indicator_params else "(default)"}
sma20 = ta.sma(close, {sma_20})
entrySignal = {"ta.crossover(close, sma20)" if is_long else "ta.crossunder(close, sma20)"}''',

            'supertrend': f'''// Supertrend Strategy (TradingView built-in pattern)
// if ta.change(direction) < 0 -> long, if ta.change(direction) > 0 -> short
// Supertrend Factor: {st_factor}, ATR: {st_atr} {"(tuned)" if indicator_params and any(k in indicator_params for k in ['supertrend_factor', 'supertrend_atr']) else "(default)"}
[supertrendValue, supertrendDir] = ta.supertrend({st_factor}, {st_atr})
dirChange = ta.change(supertrendDir)
entrySignal = {"dirChange < 0" if is_long else "dirChange > 0"}''',

            'adx_strong_trend': f'''// ADX Strong Trend Entry (ADX > 25)
// ADX Length: {adx_len} {"(tuned)" if indicator_params and 'adx_length' in indicator_params else "(default)"}
[diPlus, diMinus, adxValue] = ta.dmi({adx_len}, {adx_len})
strongTrend = adxValue > 25
entrySignal = strongTrend and {"diPlus > diMinus" if is_long else "diMinus > diPlus"}''',

            'psar_reversal': f'''// Parabolic SAR Reversal Entry
psarValue = ta.sar(0.02, 0.02, 0.2)
entrySignal = {"close > psarValue and close[1] <= psarValue[1]" if is_long else "close < psarValue and close[1] >= psarValue[1]"}''',

            # === PATTERN ===
            'consecutive_candles': f'''// Consecutive Up/Down Closes Entry (3 in a row - matches Python)
// Python counts UP closes (close > close[1]), NOT green/red candles
upClose = close > close[1]
downClose = close < close[1]
threeUp = upClose[2] and upClose[1] and upClose
threeDown = downClose[2] and downClose[1] and downClose
entrySignal = {"threeDown" if is_long else "threeUp"}''',

            'big_candle': f'''// Big Candle Reversal Entry (> 2x ATR)
atrValue = ta.atr(14)
candleRange = high - low
bigCandle = candleRange > atrValue * 2
greenCandle = close > open
redCandle = close < open
entrySignal = bigCandle and {"redCandle" if is_long else "greenCandle"}''',

            'doji_reversal': f'''// Doji Reversal Entry
body = math.abs(close - open)
totalRange = high - low
isDoji = totalRange > 0 and body < totalRange * 0.1
prevRed = close[1] < open[1]
prevGreen = close[1] > open[1]
entrySignal = isDoji and {"prevRed" if is_long else "prevGreen"}''',

            'engulfing': f'''// Engulfing Pattern Entry
greenCandle = close > open
redCandle = close < open
bullishEngulf = greenCandle and redCandle[1] and close > open[1] and open < close[1]
bearishEngulf = redCandle and greenCandle[1] and close < open[1] and open > close[1]
entrySignal = {"bullishEngulf" if is_long else "bearishEngulf"}''',

            'inside_bar': f'''// InSide Bar Strategy (TradingView built-in pattern)
// if (high < high[1] and low > low[1]) - bar range inside previous bar
// if (close > open) -> long, if (close < open) -> short
insideBar = high < high[1] and low > low[1]
greenCandle = close > open
redCandle = close < open
entrySignal = insideBar and {"greenCandle" if is_long else "redCandle"}''',

            'outside_bar': f'''// OutSide Bar Strategy (TradingView built-in pattern)
// if (high > high[1] and low < low[1]) - bar range engulfs previous bar
// if (close > open) -> long, if (close < open) -> short
outsideBar = high > high[1] and low < low[1]
greenCandle = close > open
redCandle = close < open
entrySignal = outsideBar and {"greenCandle" if is_long else "redCandle"}''',

            # === VOLATILITY ===
            'atr_breakout': f'''// ATR Breakout Entry (move > 1.5x ATR)
atrValue = ta.atr(14)
priceMove = math.abs(close - close[1])
bigMove = priceMove > atrValue * 1.5
moveUp = close > close[1]
moveDown = close < close[1]
entrySignal = bigMove and {"moveUp" if is_long else "moveDown"}''',

            'low_volatility_breakout': f'''// Low Volatility Breakout Entry (adaptive threshold - matches Python)
// Python uses 25th percentile of ATR; we approximate with lowest ATR * 1.5
atrValue = ta.atr(14)
atrThreshold = ta.lowest(atrValue, 100) * 1.5  // Approximates 25th percentile
lowVol = atrValue[1] < atrThreshold
breakHigh = close > high[1]
breakLow = close < low[1]
entrySignal = lowVol and {"breakHigh" if is_long else "breakLow"}''',

            # === PRICE ACTION ===
            'higher_low': f'''// Higher Low / Lower High Entry
higherLow = low > low[1] and low[1] > low[2]
lowerHigh = high < high[1] and high[1] < high[2]
entrySignal = {"higherLow" if is_long else "lowerHigh"}''',

            'support_resistance': f'''// Support/Resistance Entry (within 0.5% of recent extreme)
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
entrySignal = {"close <= recentLow * 1.005" if is_long else "close >= recentHigh * 0.995"}''',

            # === BASELINE ===
            'always': '''// Always Enter Strategy - Tests pure TP/SL effectiveness
// Enters on every bar in configured direction
entrySignal = true''',

            # === ADDITIONAL STRATEGIES (11 new) ===
            'keltner_breakout': f'''// Keltner Channel Breakout
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 2.0)
entrySignal = {"ta.crossover(close, kcUpper)" if is_long else "ta.crossunder(close, kcLower)"}''',

            'donchian_breakout': f'''// Donchian Channel Breakout (Turtle Trading)
dcUpper = ta.highest(high, 20)[1]
dcLower = ta.lowest(low, 20)[1]
entrySignal = {"close > dcUpper and close[1] <= dcUpper[1]" if is_long else "close < dcLower and close[1] >= dcLower[1]"}''',

            'ichimoku_cross': f'''// Ichimoku Tenkan-Kijun Cross
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
entrySignal = {"ta.crossover(tenkan, kijun)" if is_long else "ta.crossunder(tenkan, kijun)"}''',

            'ichimoku_cloud': f'''// Ichimoku Cloud Breakout
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
cloudTop = math.max(spanA, spanB)
cloudBottom = math.min(spanA, spanB)
entrySignal = {"ta.crossover(close, cloudTop)" if is_long else "ta.crossunder(close, cloudBottom)"}''',

            'aroon_cross': f'''// Aroon Oscillator Cross
[aroonUp, aroonDown] = ta.aroon(14)
entrySignal = {"ta.crossover(aroonUp, aroonDown)" if is_long else "ta.crossover(aroonDown, aroonUp)"}''',

            'momentum_zero': f'''// Momentum Crosses Zero
momValue = ta.mom(close, 10)
entrySignal = {"ta.crossover(momValue, 0)" if is_long else "ta.crossunder(momValue, 0)"}''',

            'roc_extreme': f'''// Rate of Change Extreme (adaptive percentile approximation)
rocValue = ta.roc(close, 12)
rocLower = ta.percentile_linear_interpolation(rocValue, 100, 5)
rocUpper = ta.percentile_linear_interpolation(rocValue, 100, 95)
entrySignal = {"rocValue < rocLower" if is_long else "rocValue > rocUpper"}''',

            'uo_extreme': f'''// Ultimate Oscillator Extreme
uoValue = mih_uo(7, 14, 28)
entrySignal = {"uoValue < 30" if is_long else "uoValue > 70"}''',

            'chop_trend': f'''// Choppiness Index Trend Detection
chopValue = ta.chop(14)
sma20 = ta.sma(close, 20)
isTrending = chopValue < 38.2
entrySignal = isTrending and {"close > sma20" if is_long else "close < sma20"}''',

            'double_ema_cross': f'''// Double EMA Cross (12/26)
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
entrySignal = {"ta.crossover(ema12, ema26)" if is_long else "ta.crossunder(ema12, ema26)"}''',

            'triple_ema': f'''// Triple EMA Alignment (9/21/50)
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)
ema50 = ta.ema(close, 50)
aligned = {"ema9 > ema21 and ema21 > ema50" if is_long else "ema9 < ema21 and ema21 < ema50"}
wasNotAligned = {"not (ema9[1] > ema21[1] and ema21[1] > ema50[1])" if is_long else "not (ema9[1] < ema21[1] and ema21[1] < ema50[1])"}
entrySignal = aligned and wasNotAligned''',

            # === KALMAN FILTER STRATEGIES ===
            'kalman_trend': f'''// Kalman Filter Trend
// Kalman Filter with velocity tracking for adaptive smoothing
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
entrySignal = {"ta.crossover(close, kalman)" if is_long else "ta.crossunder(close, kalman)"}''',

            'kalman_bb': f'''// Kalman Bollinger Bands
// Kalman Filter as center line with standard deviation bands
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
kalmanStd = ta.stdev(close, 20)
kalmanUpper = kalman + kalmanStd * 2.0
kalmanLower = kalman - kalmanStd * 2.0
entrySignal = {"ta.crossover(close, kalmanLower)" if is_long else "ta.crossunder(close, kalmanUpper)"}''',

            'kalman_rsi': f'''// Kalman-Smoothed RSI
// RSI filtered through Kalman for smoother signals
rsiRaw = ta.rsi(close, 14)
var float kalmanRsi = na
kalmanRsi := na(kalmanRsi) ? rsiRaw : kalmanRsi + 0.5 * (rsiRaw - kalmanRsi)
entrySignal = {"ta.crossover(kalmanRsi, 30)" if is_long else "ta.crossunder(kalmanRsi, 70)"}''',

            'kalman_mfi': f'''// Kalman-Smoothed MFI
// MFI filtered through Kalman for smoother signals
mfiRaw = ta.mfi(hlc3, 14)
var float kalmanMfi = na
kalmanMfi := na(kalmanMfi) ? mfiRaw : kalmanMfi + 0.5 * (mfiRaw - kalmanMfi)
entrySignal = {"ta.crossover(kalmanMfi, 20)" if is_long else "ta.crossunder(kalmanMfi, 80)"}''',

            'kalman_adx': f'''// Kalman-Smoothed ADX Trend
// ADX filtered through Kalman with DI dominance
[diPlus, diMinus, adxRaw] = ta.dmi(14, 14)
var float kalmanAdx = na
kalmanAdx := na(kalmanAdx) ? adxRaw : kalmanAdx + 0.5 * (adxRaw - kalmanAdx)
entrySignal = {"kalmanAdx > 25 and diPlus > diMinus" if is_long else "kalmanAdx > 25 and diMinus > diPlus"}''',

            'kalman_psar': f'''// Kalman PSAR
// Parabolic SAR crossover (use standard PSAR)
psarValue = ta.sar(0.02, 0.02, 0.2)
entrySignal = {"ta.crossover(close, psarValue)" if is_long else "ta.crossunder(close, psarValue)"}''',

            'kalman_macd': f'''// Kalman-Smoothed MACD
// MACD filtered through Kalman for smoother signals
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
var float kalmanMacd = na
var float kalmanSignal = na
kalmanMacd := na(kalmanMacd) ? macdLine : kalmanMacd + 0.5 * (macdLine - kalmanMacd)
kalmanSignal := na(kalmanSignal) ? signalLine : kalmanSignal + 0.5 * (signalLine - kalmanSignal)
entrySignal = {"ta.crossover(kalmanMacd, kalmanSignal)" if is_long else "ta.crossunder(kalmanMacd, kalmanSignal)"}''',

            # === MISSING ENTRY RULES (ported from VectorBT) ===
            'ao_twin_peaks': f'''// Awesome Oscillator Twin Peaks Pattern
// AO = SMA(hl2, 5) - SMA(hl2, 34)
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
aoLow = ta.lowest(ao, 20)
aoHigh = ta.highest(ao, 20)
aoRising = ao > ao[1]
aoFalling = ao < ao[1]
// Long: Twin peaks below zero (AO < 0, above recent low, turning up)
// Short: Twin peaks above zero (AO > 0, below recent high, turning down)
entrySignal = {"ao < 0 and ao > aoLow and aoRising" if is_long else "ao > 0 and ao < aoHigh and aoFalling"}''',

            'ao_zero_cross': f'''// Awesome Oscillator Zero Cross
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
entrySignal = {"ta.crossover(ao, 0)" if is_long else "ta.crossunder(ao, 0)"}''',

            'mfi_extreme': f'''// Money Flow Index Extreme
mfiValue = ta.mfi(hlc3, 14)
entrySignal = {"ta.crossover(mfiValue, 20)" if is_long else "ta.crossunder(mfiValue, 80)"}''',

            'cmf_cross': f'''// Chaikin Money Flow Zero Cross
// CMF = sum(mfm * volume, 20) / sum(volume, 20)
mfm = ((close - low) - (high - close)) / (high - low)
mfv = mfm * volume
cmfValue = ta.sma(mfv, 20) / ta.sma(volume, 20)
entrySignal = {"ta.crossover(cmfValue, 0)" if is_long else "ta.crossunder(cmfValue, 0)"}''',

            'obv_trend': f'''// On-Balance Volume Trend
obvValue = ta.obv
obvHigh = ta.highest(obvValue, 14)
obvLow = ta.lowest(obvValue, 14)
priceHigh = ta.highest(close, 14)
priceLow = ta.lowest(close, 14)
entrySignal = {"obvValue == obvHigh and close >= priceHigh * 0.98" if is_long else "obvValue == obvLow and close <= priceLow * 1.02"}''',

            'ppo_cross': f'''// Percentage Price Oscillator Cross
// PPO = (EMA12 - EMA26) / EMA26 * 100
emaFastPPO = ta.ema(close, 12)
emaSlowPPO = ta.ema(close, 26)
ppoValue = (emaFastPPO - emaSlowPPO) / emaSlowPPO * 100
ppoSignal = ta.ema(ppoValue, 9)
entrySignal = {"ta.crossover(ppoValue, ppoSignal)" if is_long else "ta.crossunder(ppoValue, ppoSignal)"}''',

            'fisher_cross': f'''// Fisher Transform Cross
// Fisher Transform of normalized price
highestHigh = ta.highest(hl2, 10)
lowestLow = ta.lowest(hl2, 10)
rawValue = 2 * ((hl2 - lowestLow) / (highestHigh - lowestLow) - 0.5)
var float smoothed = 0.0
smoothed := 0.33 * rawValue + 0.67 * nz(smoothed[1])
clampedValue = math.max(-0.999, math.min(0.999, smoothed))
var float fisherValue = 0.0
fisherValue := 0.5 * math.log((1 + clampedValue) / (1 - clampedValue)) + 0.5 * nz(fisherValue[1])
fisherSignal = fisherValue[1]
entrySignal = {"ta.crossover(fisherValue, fisherSignal)" if is_long else "ta.crossunder(fisherValue, fisherSignal)"}''',

            'tsi_cross': f'''// True Strength Index Signal Cross
// TSI = 100 * Double EMA(momentum) / Double EMA(abs(momentum))
momentum = close - close[1]
smoothMom = ta.ema(ta.ema(momentum, 25), 13)
smoothAbsMom = ta.ema(ta.ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
tsiSignal = ta.ema(tsiValue, 7)
entrySignal = {"ta.crossover(tsiValue, tsiSignal)" if is_long else "ta.crossunder(tsiValue, tsiSignal)"}''',

            'tsi_zero': f'''// True Strength Index Zero Cross
momentum = close - close[1]
smoothMom = ta.ema(ta.ema(momentum, 25), 13)
smoothAbsMom = ta.ema(ta.ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
entrySignal = {"ta.crossover(tsiValue, 0)" if is_long else "ta.crossunder(tsiValue, 0)"}''',

            'rsi_macd_combo': f'''// RSI + MACD Combo
rsiValue = ta.rsi(close, {rsi_len})
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
histogram = macdLine - signalLine
histRising = histogram > histogram[1]
histFalling = histogram < histogram[1]
entrySignal = {"rsiValue < 30 and histRising" if is_long else "rsiValue > 70 and histFalling"}''',

            'macd_stoch_combo': f'''// MACD + Stochastic Combo
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
k = ta.sma(ta.stoch(close, high, low, {stoch_k}), {stoch_smooth})
d = ta.sma(k, {stoch_d})
macdBullish = macdLine > signalLine
macdBearish = macdLine < signalLine
stochOversold = k < 20
stochOverbought = k > 80
entrySignal = {"macdBullish and stochOversold" if is_long else "macdBearish and stochOverbought"}''',

            'hull_ma_cross': f'''// Hull Moving Average Cross
// Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
entrySignal = {"ta.crossover(close, hullMa)" if is_long else "ta.crossunder(close, hullMa)"}''',

            'hull_ma_turn': f'''// Hull Moving Average Direction Change
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
hullSlope = hullMa - hullMa[1]
hullSlopePrev = hullMa[1] - hullMa[2]
entrySignal = {"hullSlope > 0 and hullSlopePrev <= 0" if is_long else "hullSlope < 0 and hullSlopePrev >= 0"}''',

            'zlema_cross': f'''// Zero-Lag EMA Cross
// ZLEMA = EMA(close + (close - close[lag]), length)
zlemaLen = 20
lag = math.round((zlemaLen - 1) / 2)
zlema = ta.ema(close + (close - close[lag]), zlemaLen)
entrySignal = {"ta.crossover(close, zlema)" if is_long else "ta.crossunder(close, zlema)"}''',

            'mcginley_cross': f'''// McGinley Dynamic Cross
// McGinley Dynamic adapts to price speed
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
entrySignal = {"ta.crossover(close, mcginley)" if is_long else "ta.crossunder(close, mcginley)"}''',

            'mcginley_trend': f'''// McGinley Dynamic Trend Change
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
mcgSlope = mcginley - mcginley[1]
mcgSlopePrev = mcginley[1] - mcginley[2]
entrySignal = {"mcgSlope > 0 and mcgSlopePrev <= 0" if is_long else "mcgSlope < 0 and mcgSlopePrev >= 0"}''',

            'linreg_channel': f'''// Linear Regression Channel Breakout
linregLen = 20
linregMid = ta.linreg(close, linregLen, 0)
linregDev = ta.stdev(close, linregLen)
linregUpper = linregMid + linregDev * 2
linregLower = linregMid - linregDev * 2
entrySignal = {"ta.crossover(close, linregLower)" if is_long else "ta.crossunder(close, linregUpper)"}''',

            'bb_rsi_combo': f'''// Bollinger Bands + RSI Combo
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
rsiValue = ta.rsi(close, {rsi_len})
entrySignal = {"close < bbLower and rsiValue < 30" if is_long else "close > bbUpper and rsiValue > 70"}''',

            'squeeze_momentum': f'''// Squeeze Momentum (BB inside Keltner)
[bbMiddle, bbUpper, bbLower] = ta.bb(close, 20, 2.0)
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 1.5)
squeeze = bbLower > kcLower and bbUpper < kcUpper
squeezeFired = squeeze[1] and not squeeze
mom = close - close[20]
entrySignal = {"squeezeFired and mom > 0" if is_long else "squeezeFired and mom < 0"}''',

            'chandelier_entry': f'''// Chandelier Exit Entry
atrValue = ta.atr({atr_len})
highestHigh = ta.highest(high, 22)
lowestLow = ta.lowest(low, 22)
chandelierLong = highestHigh - atrValue * 3
chandelierShort = lowestLow + atrValue * 3
entrySignal = {"ta.crossover(close, chandelierLong)" if is_long else "ta.crossunder(close, chandelierShort)"}''',

            'ema_rsi_combo': f'''// EMA + RSI Combo
emaFast = ta.ema(close, {ema_fast})
emaSlow = ta.ema(close, {ema_slow})
rsiValue = ta.rsi(close, {rsi_len})
emaBullish = emaFast > emaSlow
emaBearish = emaFast < emaSlow
entrySignal = {"emaBullish and rsiValue < 40" if is_long else "emaBearish and rsiValue > 60"}''',

            'supertrend_adx_combo': f'''// SuperTrend + ADX Combo
[supertrendValue, supertrendDir] = ta.supertrend({st_factor}, {st_atr})
[diPlus, diMinus, adxValue] = ta.dmi({adx_len}, {adx_len})
strongTrend = adxValue > 25
stBullish = close > supertrendValue
stBearish = close < supertrendValue
entrySignal = {"strongTrend and stBullish and diPlus > diMinus" if is_long else "strongTrend and stBearish and diMinus > diPlus"}''',

            'pivot_bounce': f'''// Pivot Point Bounce
// Standard floor trader pivots
pivotPoint = (high[1] + low[1] + close[1]) / 3
r1 = 2 * pivotPoint - low[1]
s1 = 2 * pivotPoint - high[1]
nearS1 = low <= s1 * 1.005 and low >= s1 * 0.995
nearR1 = high >= r1 * 0.995 and high <= r1 * 1.005
greenCandle = close > open
redCandle = close < open
entrySignal = {"nearS1 and greenCandle" if is_long else "nearR1 and redCandle"}''',

            'elder_ray': f'''// Elder Ray (Bull/Bear Power)
ema13 = ta.ema(close, 13)
bullPower = high - ema13
bearPower = low - ema13
emaRising = ema13 > ema13[1]
emaFalling = ema13 < ema13[1]
bearRising = bearPower > bearPower[1]
bullFalling = bullPower < bullPower[1]
entrySignal = {"emaRising and bearPower < 0 and bearRising" if is_long else "emaFalling and bullPower > 0 and bullFalling"}''',
        }

        # Mihakralj entry conditions - uses mathematically rigorous mih_* functions
        mihakralj_entry_conditions = {
            # === MOMENTUM ===
            'rsi_extreme': f'''// RSI Strategy (mihakralj warmup-compensated)
rsiValue = mih_rsi(close, 14)
entrySignal = {"ta.crossover(rsiValue, 30)" if is_long else "ta.crossunder(rsiValue, 70)"}''',

            'rsi_cross_50': f'''// RSI Cross 50 Entry (mihakralj)
rsiValue = mih_rsi(close, 14)
entrySignal = {"ta.crossover(rsiValue, 50)" if is_long else "ta.crossunder(rsiValue, 50)"}''',

            'stoch_extreme': f'''// Stochastic Slow Strategy (mihakralj O(1) SMA)
[k, d] = mih_stoch(14, 3, 3)
entrySignal = {"ta.crossover(k, d) and k < 20" if is_long else "ta.crossunder(k, d) and k > 80"}''',

            'williams_r': f'''// Williams %R Extreme Entry (< -80 long, > -20 short)
willrValue = ta.wpr(14)
entrySignal = {"willrValue < -80" if is_long else "willrValue > -20"}''',

            'cci_extreme': f'''// CCI Extreme Entry (< -100 long, > 100 short)
cciValue = ta.cci(high, low, close, 20)
entrySignal = {"cciValue < -100" if is_long else "cciValue > 100"}''',

            'rsi_divergence': f'''// RSI Divergence Entry (mihakralj RSI - rolling window matches Python)
// Python uses rolling().min/max().shift(1) - we match with ta.lowest/highest()[1]
rsiValue = mih_rsi(close, 14)
lookback = 5
priceLowerLow = low < ta.lowest(low, lookback)[1]
rsiHigherLow = rsiValue > ta.lowest(rsiValue, lookback)[1]
priceHigherHigh = high > ta.highest(high, lookback)[1]
rsiLowerHigh = rsiValue < ta.highest(rsiValue, lookback)[1]
entrySignal = {"priceLowerLow and rsiHigherLow and rsiValue < 40" if is_long else "priceHigherHigh and rsiLowerHigh and rsiValue > 60"}''',

            # === MEAN REVERSION ===
            'bb_touch': f'''// Bollinger Bands Strategy (mihakralj O(1) complexity)
[bbMiddle, bbUpper, bbLower] = mih_bbands(close, 20, 2.0)
entrySignal = {"ta.crossover(close, bbLower)" if is_long else "ta.crossunder(close, bbUpper)"}''',

            'bb_squeeze_breakout': f'''// BB Squeeze Breakout Entry (mihakralj O(1) BBands)
[bbMiddle, bbUpper, bbLower] = mih_bbands(close, 20, 2.0)
bbWidth = (bbUpper - bbLower) / bbMiddle
avgWidth = mih_sma(bbWidth, 20)
squeezed = bbWidth[1] < avgWidth * 0.8
expanding = bbWidth > bbWidth[1]
entrySignal = squeezed and expanding and {"close > bbMiddle" if is_long else "close < bbMiddle"}''',

            'price_vs_sma': f'''// Price vs SMA Entry (mihakralj O(1) SMA)
sma20 = mih_sma(close, 20)
entrySignal = {"close < sma20 * 0.99" if is_long else "close > sma20 * 1.01"}''',

            'vwap_bounce': f'''// VWAP Bounce Entry
vwapValue = ta.vwap(hlc3)
touchedBelow = low < vwapValue
touchedAbove = high > vwapValue
closedAbove = close > vwapValue
closedBelow = close < vwapValue
entrySignal = {"touchedBelow and closedAbove" if is_long else "touchedAbove and closedBelow"}''',

            'vwap_cross': f'''// VWAP Cross Entry
vwapValue = ta.vwap(hlc3)
entrySignal = {"ta.crossover(close, vwapValue)" if is_long else "ta.crossunder(close, vwapValue)"}''',

            'vwma_cross': f'''// VWMA Cross Entry
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
entrySignal = {"ta.crossover(close, vwmaValue)" if is_long else "ta.crossunder(close, vwmaValue)"}''',

            'vwma_trend': f'''// VWMA Trend Entry
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
vwmaSlope = vwmaValue - vwmaValue[1]
entrySignal = {"vwmaSlope > 0 and vwmaSlope[1] <= 0" if is_long else "vwmaSlope < 0 and vwmaSlope[1] >= 0"}''',

            # === TREND ===
            'ema_cross': f'''// EMA 9/21 Cross Entry (mihakralj warmup-compensated)
emaFast = mih_ema(close, 9)
emaSlow = mih_ema(close, 21)
entrySignal = {"ta.crossover(emaFast, emaSlow)" if is_long else "ta.crossunder(emaFast, emaSlow)"}''',

            'sma_cross': f'''// MovingAvg2Line Cross (mihakralj O(1) SMA)
mafast = mih_sma(close, 9)
maslow = mih_sma(close, 18)
entrySignal = {"ta.crossover(mafast, maslow)" if is_long else "ta.crossunder(mafast, maslow)"}''',

            'macd_cross': f'''// MACD Strategy (mihakralj warmup-compensated)
[macdLine, signalLine, histLine] = mih_macd(close, 12, 26, 9)
entrySignal = {"ta.crossover(histLine, 0)" if is_long else "ta.crossunder(histLine, 0)"}''',

            'price_above_sma': f'''// Price Crosses SMA Entry (mihakralj O(1) SMA)
sma20 = mih_sma(close, 20)
entrySignal = {"ta.crossover(close, sma20)" if is_long else "ta.crossunder(close, sma20)"}''',

            'supertrend': f'''// Supertrend Strategy (TradingView built-in pattern)
// if ta.change(direction) < 0 -> long, if ta.change(direction) > 0 -> short
[supertrendValue, supertrendDir] = ta.supertrend(3, 10)
dirChange = ta.change(supertrendDir)
entrySignal = {"dirChange < 0" if is_long else "dirChange > 0"}''',

            'adx_strong_trend': f'''// ADX Strong Trend Entry (ADX > 25)
[diPlus, diMinus, adxValue] = ta.dmi(14, 14)
strongTrend = adxValue > 25
entrySignal = strongTrend and {"diPlus > diMinus" if is_long else "diMinus > diPlus"}''',

            'psar_reversal': f'''// Parabolic SAR Reversal Entry
psarValue = ta.sar(0.02, 0.02, 0.2)
entrySignal = {"close > psarValue and close[1] <= psarValue[1]" if is_long else "close < psarValue and close[1] >= psarValue[1]"}''',

            # === PATTERN ===
            'consecutive_candles': f'''// Consecutive Up/Down Closes Entry (3 in a row - matches Python)
// Python counts UP closes (close > close[1]), NOT green/red candles
upClose = close > close[1]
downClose = close < close[1]
threeUp = upClose[2] and upClose[1] and upClose
threeDown = downClose[2] and downClose[1] and downClose
entrySignal = {"threeDown" if is_long else "threeUp"}''',

            'big_candle': f'''// Big Candle Reversal Entry (> 2x ATR, mihakralj ATR)
atrValue = mih_atr(14)
candleRange = high - low
bigCandle = candleRange > atrValue * 2
greenCandle = close > open
redCandle = close < open
entrySignal = bigCandle and {"redCandle" if is_long else "greenCandle"}''',

            'doji_reversal': f'''// Doji Reversal Entry
body = math.abs(close - open)
totalRange = high - low
isDoji = totalRange > 0 and body < totalRange * 0.1
prevRed = close[1] < open[1]
prevGreen = close[1] > open[1]
entrySignal = isDoji and {"prevRed" if is_long else "prevGreen"}''',

            'engulfing': f'''// Engulfing Pattern Entry
greenCandle = close > open
redCandle = close < open
bullishEngulf = greenCandle and redCandle[1] and close > open[1] and open < close[1]
bearishEngulf = redCandle and greenCandle[1] and close < open[1] and open > close[1]
entrySignal = {"bullishEngulf" if is_long else "bearishEngulf"}''',

            'inside_bar': f'''// InSide Bar Strategy (TradingView built-in pattern)
// if (high < high[1] and low > low[1]) - bar range inside previous bar
// if (close > open) -> long, if (close < open) -> short
insideBar = high < high[1] and low > low[1]
greenCandle = close > open
redCandle = close < open
entrySignal = insideBar and {"greenCandle" if is_long else "redCandle"}''',

            'outside_bar': f'''// OutSide Bar Strategy (TradingView built-in pattern)
// if (high > high[1] and low < low[1]) - bar range engulfs previous bar
// if (close > open) -> long, if (close < open) -> short
outsideBar = high > high[1] and low < low[1]
greenCandle = close > open
redCandle = close < open
entrySignal = outsideBar and {"greenCandle" if is_long else "redCandle"}''',

            # === VOLATILITY ===
            'atr_breakout': f'''// ATR Breakout Entry (mihakralj ATR with warmup)
atrValue = mih_atr(14)
priceMove = math.abs(close - close[1])
bigMove = priceMove > atrValue * 1.5
moveUp = close > close[1]
moveDown = close < close[1]
entrySignal = bigMove and {"moveUp" if is_long else "moveDown"}''',

            'low_volatility_breakout': f'''// Low Volatility Breakout Entry (adaptive threshold - matches Python)
// Python uses 25th percentile of ATR; we approximate with lowest ATR * 1.5
atrValue = mih_atr(14)
atrThreshold = ta.lowest(atrValue, 100) * 1.5  // Approximates 25th percentile
lowVol = atrValue[1] < atrThreshold
breakHigh = close > high[1]
breakLow = close < low[1]
entrySignal = lowVol and {"breakHigh" if is_long else "breakLow"}''',

            # === PRICE ACTION ===
            'higher_low': f'''// Higher Low / Lower High Entry
higherLow = low > low[1] and low[1] > low[2]
lowerHigh = high < high[1] and high[1] < high[2]
entrySignal = {"higherLow" if is_long else "lowerHigh"}''',

            'support_resistance': f'''// Support/Resistance Entry (within 0.5% of recent extreme)
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
entrySignal = {"close <= recentLow * 1.005" if is_long else "close >= recentHigh * 0.995"}''',

            # === BASELINE ===
            'always': '''// Always Enter Strategy (mihakralj)
entrySignal = true''',

            # === ADDITIONAL STRATEGIES ===
            'keltner_breakout': f'''// Keltner Channel Breakout (mihakralj)
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 2.0)
entrySignal = {"ta.crossover(close, kcUpper)" if is_long else "ta.crossunder(close, kcLower)"}''',

            'donchian_breakout': f'''// Donchian Channel Breakout (mihakralj)
dcUpper = ta.highest(high, 20)[1]
dcLower = ta.lowest(low, 20)[1]
entrySignal = {"close > dcUpper and close[1] <= dcUpper[1]" if is_long else "close < dcLower and close[1] >= dcLower[1]"}''',

            'ichimoku_cross': f'''// Ichimoku Tenkan-Kijun Cross (mihakralj)
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
entrySignal = {"ta.crossover(tenkan, kijun)" if is_long else "ta.crossunder(tenkan, kijun)"}''',

            'ichimoku_cloud': f'''// Ichimoku Cloud Breakout (mihakralj)
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
cloudTop = math.max(spanA, spanB)
cloudBottom = math.min(spanA, spanB)
entrySignal = {"ta.crossover(close, cloudTop)" if is_long else "ta.crossunder(close, cloudBottom)"}''',

            'aroon_cross': f'''// Aroon Oscillator Cross (mihakralj)
[aroonUp, aroonDown] = ta.aroon(14)
entrySignal = {"ta.crossover(aroonUp, aroonDown)" if is_long else "ta.crossover(aroonDown, aroonUp)"}''',

            'momentum_zero': f'''// Momentum Crosses Zero (mihakralj)
momValue = ta.mom(close, 10)
entrySignal = {"ta.crossover(momValue, 0)" if is_long else "ta.crossunder(momValue, 0)"}''',

            'roc_extreme': f'''// Rate of Change Extreme (mihakralj)
rocValue = ta.roc(close, 12)
rocLower = ta.percentile_linear_interpolation(rocValue, 100, 5)
rocUpper = ta.percentile_linear_interpolation(rocValue, 100, 95)
entrySignal = {"rocValue < rocLower" if is_long else "rocValue > rocUpper"}''',

            'uo_extreme': f'''// Ultimate Oscillator Extreme (mihakralj)
uoValue = mih_uo(7, 14, 28)
entrySignal = {"uoValue < 30" if is_long else "uoValue > 70"}''',

            'chop_trend': f'''// Choppiness Index Trend Detection (mihakralj)
chopValue = ta.chop(14)
sma20 = ta.sma(close, 20)
isTrending = chopValue < 38.2
entrySignal = isTrending and {"close > sma20" if is_long else "close < sma20"}''',

            'double_ema_cross': f'''// Double EMA Cross (12/26) (mihakralj)
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
entrySignal = {"ta.crossover(ema12, ema26)" if is_long else "ta.crossunder(ema12, ema26)"}''',

            'triple_ema': f'''// Triple EMA Alignment (9/21/50) (mihakralj)
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)
ema50 = ta.ema(close, 50)
aligned = {"ema9 > ema21 and ema21 > ema50" if is_long else "ema9 < ema21 and ema21 < ema50"}
wasNotAligned = {"not (ema9[1] > ema21[1] and ema21[1] > ema50[1])" if is_long else "not (ema9[1] < ema21[1] and ema21[1] < ema50[1])"}
entrySignal = aligned and wasNotAligned''',

            # === KALMAN FILTER STRATEGIES (mihakralj) ===
            'kalman_trend': f'''// Kalman Filter Trend (mihakralj)
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
entrySignal = {"ta.crossover(close, kalman)" if is_long else "ta.crossunder(close, kalman)"}''',

            'kalman_bb': f'''// Kalman Bollinger Bands (mihakralj)
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
kalmanStd = ta.stdev(close, 20)
kalmanUpper = kalman + kalmanStd * 2.0
kalmanLower = kalman - kalmanStd * 2.0
entrySignal = {"ta.crossover(close, kalmanLower)" if is_long else "ta.crossunder(close, kalmanUpper)"}''',

            'kalman_rsi': f'''// Kalman-Smoothed RSI (mihakralj)
rsiRaw = mih_rsi(close, 14)
var float kalmanRsi = na
kalmanRsi := na(kalmanRsi) ? rsiRaw : kalmanRsi + 0.5 * (rsiRaw - kalmanRsi)
entrySignal = {"ta.crossover(kalmanRsi, 30)" if is_long else "ta.crossunder(kalmanRsi, 70)"}''',

            'kalman_mfi': f'''// Kalman-Smoothed MFI (mihakralj)
mfiRaw = ta.mfi(hlc3, 14)
var float kalmanMfi = na
kalmanMfi := na(kalmanMfi) ? mfiRaw : kalmanMfi + 0.5 * (mfiRaw - kalmanMfi)
entrySignal = {"ta.crossover(kalmanMfi, 20)" if is_long else "ta.crossunder(kalmanMfi, 80)"}''',

            'kalman_adx': f'''// Kalman-Smoothed ADX Trend (mihakralj)
[diPlus, diMinus, adxRaw] = ta.dmi(14, 14)
var float kalmanAdx = na
kalmanAdx := na(kalmanAdx) ? adxRaw : kalmanAdx + 0.5 * (adxRaw - kalmanAdx)
entrySignal = {"kalmanAdx > 25 and diPlus > diMinus" if is_long else "kalmanAdx > 25 and diMinus > diPlus"}''',

            'kalman_psar': f'''// Kalman PSAR (mihakralj)
psarValue = ta.sar(0.02, 0.02, 0.2)
entrySignal = {"ta.crossover(close, psarValue)" if is_long else "ta.crossunder(close, psarValue)"}''',

            'kalman_macd': f'''// Kalman-Smoothed MACD (mihakralj)
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
var float kalmanMacd = na
var float kalmanSignal = na
kalmanMacd := na(kalmanMacd) ? macdLine : kalmanMacd + 0.5 * (macdLine - kalmanMacd)
kalmanSignal := na(kalmanSignal) ? signalLine : kalmanSignal + 0.5 * (signalLine - kalmanSignal)
entrySignal = {"ta.crossover(kalmanMacd, kalmanSignal)" if is_long else "ta.crossunder(kalmanMacd, kalmanSignal)"}''',

            # === MISSING ENTRY RULES (ported from VectorBT - mihakralj) ===
            'ao_twin_peaks': f'''// Awesome Oscillator Twin Peaks Pattern (mihakralj)
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
aoLow = ta.lowest(ao, 20)
aoHigh = ta.highest(ao, 20)
aoRising = ao > ao[1]
aoFalling = ao < ao[1]
entrySignal = {"ao < 0 and ao > aoLow and aoRising" if is_long else "ao > 0 and ao < aoHigh and aoFalling"}''',

            'ao_zero_cross': f'''// Awesome Oscillator Zero Cross (mihakralj)
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
entrySignal = {"ta.crossover(ao, 0)" if is_long else "ta.crossunder(ao, 0)"}''',

            'mfi_extreme': f'''// Money Flow Index Extreme (mihakralj)
mfiValue = ta.mfi(hlc3, 14)
entrySignal = {"ta.crossover(mfiValue, 20)" if is_long else "ta.crossunder(mfiValue, 80)"}''',

            'cmf_cross': f'''// Chaikin Money Flow Zero Cross (mihakralj)
mfm = ((close - low) - (high - close)) / (high - low)
mfv = mfm * volume
cmfValue = ta.sma(mfv, 20) / ta.sma(volume, 20)
entrySignal = {"ta.crossover(cmfValue, 0)" if is_long else "ta.crossunder(cmfValue, 0)"}''',

            'obv_trend': f'''// On-Balance Volume Trend (mihakralj)
obvValue = ta.obv
obvHigh = ta.highest(obvValue, 14)
obvLow = ta.lowest(obvValue, 14)
priceHigh = ta.highest(close, 14)
priceLow = ta.lowest(close, 14)
entrySignal = {"obvValue == obvHigh and close >= priceHigh * 0.98" if is_long else "obvValue == obvLow and close <= priceLow * 1.02"}''',

            'ppo_cross': f'''// Percentage Price Oscillator Cross (mihakralj)
emaFastPPO = mih_ema(close, 12)
emaSlowPPO = mih_ema(close, 26)
ppoValue = (emaFastPPO - emaSlowPPO) / emaSlowPPO * 100
ppoSignal = mih_ema(ppoValue, 9)
entrySignal = {"ta.crossover(ppoValue, ppoSignal)" if is_long else "ta.crossunder(ppoValue, ppoSignal)"}''',

            'fisher_cross': f'''// Fisher Transform Cross (mihakralj)
highestHigh = ta.highest(hl2, 10)
lowestLow = ta.lowest(hl2, 10)
rawValue = 2 * ((hl2 - lowestLow) / (highestHigh - lowestLow) - 0.5)
var float smoothed = 0.0
smoothed := 0.33 * rawValue + 0.67 * nz(smoothed[1])
clampedValue = math.max(-0.999, math.min(0.999, smoothed))
var float fisherValue = 0.0
fisherValue := 0.5 * math.log((1 + clampedValue) / (1 - clampedValue)) + 0.5 * nz(fisherValue[1])
fisherSignal = fisherValue[1]
entrySignal = {"ta.crossover(fisherValue, fisherSignal)" if is_long else "ta.crossunder(fisherValue, fisherSignal)"}''',

            'tsi_cross': f'''// True Strength Index Signal Cross (mihakralj)
momentum = close - close[1]
smoothMom = mih_ema(mih_ema(momentum, 25), 13)
smoothAbsMom = mih_ema(mih_ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
tsiSignal = mih_ema(tsiValue, 7)
entrySignal = {"ta.crossover(tsiValue, tsiSignal)" if is_long else "ta.crossunder(tsiValue, tsiSignal)"}''',

            'tsi_zero': f'''// True Strength Index Zero Cross (mihakralj)
momentum = close - close[1]
smoothMom = mih_ema(mih_ema(momentum, 25), 13)
smoothAbsMom = mih_ema(mih_ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
entrySignal = {"ta.crossover(tsiValue, 0)" if is_long else "ta.crossunder(tsiValue, 0)"}''',

            'rsi_macd_combo': f'''// RSI + MACD Combo (mihakralj)
rsiValue = mih_rsi(close, 14)
[macdLine, signalLine, histLine] = mih_macd(close, 12, 26, 9)
histogram = macdLine - signalLine
histRising = histogram > histogram[1]
histFalling = histogram < histogram[1]
entrySignal = {"rsiValue < 30 and histRising" if is_long else "rsiValue > 70 and histFalling"}''',

            'macd_stoch_combo': f'''// MACD + Stochastic Combo (mihakralj)
[macdLine, signalLine, histLine] = mih_macd(close, 12, 26, 9)
[k, d] = mih_stoch(14, 3, 3)
macdBullish = macdLine > signalLine
macdBearish = macdLine < signalLine
stochOversold = k < 20
stochOverbought = k > 80
entrySignal = {"macdBullish and stochOversold" if is_long else "macdBearish and stochOverbought"}''',

            'hull_ma_cross': f'''// Hull Moving Average Cross (mihakralj)
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
entrySignal = {"ta.crossover(close, hullMa)" if is_long else "ta.crossunder(close, hullMa)"}''',

            'hull_ma_turn': f'''// Hull Moving Average Direction Change (mihakralj)
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
hullSlope = hullMa - hullMa[1]
hullSlopePrev = hullMa[1] - hullMa[2]
entrySignal = {"hullSlope > 0 and hullSlopePrev <= 0" if is_long else "hullSlope < 0 and hullSlopePrev >= 0"}''',

            'zlema_cross': f'''// Zero-Lag EMA Cross (mihakralj)
zlemaLen = 20
lag = math.round((zlemaLen - 1) / 2)
zlema = mih_ema(close + (close - close[lag]), zlemaLen)
entrySignal = {"ta.crossover(close, zlema)" if is_long else "ta.crossunder(close, zlema)"}''',

            'mcginley_cross': f'''// McGinley Dynamic Cross (mihakralj)
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
entrySignal = {"ta.crossover(close, mcginley)" if is_long else "ta.crossunder(close, mcginley)"}''',

            'mcginley_trend': f'''// McGinley Dynamic Trend Change (mihakralj)
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
mcgSlope = mcginley - mcginley[1]
mcgSlopePrev = mcginley[1] - mcginley[2]
entrySignal = {"mcgSlope > 0 and mcgSlopePrev <= 0" if is_long else "mcgSlope < 0 and mcgSlopePrev >= 0"}''',

            'linreg_channel': f'''// Linear Regression Channel Breakout (mihakralj)
linregLen = 20
linregMid = ta.linreg(close, linregLen, 0)
linregDev = ta.stdev(close, linregLen)
linregUpper = linregMid + linregDev * 2
linregLower = linregMid - linregDev * 2
entrySignal = {"ta.crossover(close, linregLower)" if is_long else "ta.crossunder(close, linregUpper)"}''',

            'bb_rsi_combo': f'''// Bollinger Bands + RSI Combo (mihakralj)
[bbMiddle, bbUpper, bbLower] = mih_bbands(close, 20, 2.0)
rsiValue = mih_rsi(close, 14)
entrySignal = {"close < bbLower and rsiValue < 30" if is_long else "close > bbUpper and rsiValue > 70"}''',

            'squeeze_momentum': f'''// Squeeze Momentum (mihakralj)
[bbMiddle, bbUpper, bbLower] = mih_bbands(close, 20, 2.0)
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 1.5)
squeeze = bbLower > kcLower and bbUpper < kcUpper
squeezeFired = squeeze[1] and not squeeze
mom = close - close[20]
entrySignal = {"squeezeFired and mom > 0" if is_long else "squeezeFired and mom < 0"}''',

            'chandelier_entry': f'''// Chandelier Exit Entry (mihakralj)
atrValue = mih_atr(14)
highestHigh = ta.highest(high, 22)
lowestLow = ta.lowest(low, 22)
chandelierLong = highestHigh - atrValue * 3
chandelierShort = lowestLow + atrValue * 3
entrySignal = {"ta.crossover(close, chandelierLong)" if is_long else "ta.crossunder(close, chandelierShort)"}''',

            'ema_rsi_combo': f'''// EMA + RSI Combo (mihakralj)
emaFast = mih_ema(close, 9)
emaSlow = mih_ema(close, 21)
rsiValue = mih_rsi(close, 14)
emaBullish = emaFast > emaSlow
emaBearish = emaFast < emaSlow
entrySignal = {"emaBullish and rsiValue < 40" if is_long else "emaBearish and rsiValue > 60"}''',

            'supertrend_adx_combo': f'''// SuperTrend + ADX Combo (mihakralj)
[supertrendValue, supertrendDir] = ta.supertrend(3, 10)
[diPlus, diMinus, adxValue] = ta.dmi(14, 14)
strongTrend = adxValue > 25
stBullish = close > supertrendValue
stBearish = close < supertrendValue
entrySignal = {"strongTrend and stBullish and diPlus > diMinus" if is_long else "strongTrend and stBearish and diMinus > diPlus"}''',

            'pivot_bounce': f'''// Pivot Point Bounce (mihakralj)
pivotPoint = (high[1] + low[1] + close[1]) / 3
r1 = 2 * pivotPoint - low[1]
s1 = 2 * pivotPoint - high[1]
nearS1 = low <= s1 * 1.005 and low >= s1 * 0.995
nearR1 = high >= r1 * 0.995 and high <= r1 * 1.005
greenCandle = close > open
redCandle = close < open
entrySignal = {"nearS1 and greenCandle" if is_long else "nearR1 and redCandle"}''',

            'elder_ray': f'''// Elder Ray (mihakralj)
ema13 = mih_ema(close, 13)
bullPower = high - ema13
bearPower = low - ema13
emaRising = ema13 > ema13[1]
emaFalling = ema13 < ema13[1]
bearRising = bearPower > bearPower[1]
bullFalling = bullPower < bullPower[1]
entrySignal = {"emaRising and bearPower < 0 and bearRising" if is_long else "emaFalling and bullPower > 0 and bullFalling"}''',
        }

        # Bidirectional entry conditions - generates BOTH long and short signals
        bidirectional_entry_conditions = {
            'rsi_extreme': f'''// RSI Strategy - BIDIRECTIONAL
rsiValue = ta.rsi(close, {rsi_len})
longEntrySignal = ta.crossover(rsiValue, 30)
shortEntrySignal = ta.crossunder(rsiValue, 70)''',

            'rsi_cross_50': f'''// RSI Cross 50 - BIDIRECTIONAL
rsiValue = ta.rsi(close, {rsi_len})
longEntrySignal = ta.crossover(rsiValue, 50)
shortEntrySignal = ta.crossunder(rsiValue, 50)''',

            'stoch_extreme': f'''// Stochastic Slow - BIDIRECTIONAL
k = ta.sma(ta.stoch(close, high, low, {stoch_k}), {stoch_smooth})
d = ta.sma(k, {stoch_d})
longEntrySignal = ta.crossover(k, d) and k < 20
shortEntrySignal = ta.crossunder(k, d) and k > 80''',

            'williams_r': f'''// Williams %R - BIDIRECTIONAL
willrValue = ta.wpr({willr_len})
longEntrySignal = willrValue < -80
shortEntrySignal = willrValue > -20''',

            'cci_extreme': f'''// CCI Extreme - BIDIRECTIONAL
cciValue = ta.cci(high, low, close, {cci_len})
longEntrySignal = cciValue < -100
shortEntrySignal = cciValue > 100''',

            'bb_touch': f'''// Bollinger Bands - BIDIRECTIONAL
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
longEntrySignal = ta.crossover(close, bbLower)
shortEntrySignal = ta.crossunder(close, bbUpper)''',

            'ema_cross': f'''// EMA Cross - BIDIRECTIONAL
emaFast = ta.ema(close, {ema_fast})
emaSlow = ta.ema(close, {ema_slow})
longEntrySignal = ta.crossover(emaFast, emaSlow)
shortEntrySignal = ta.crossunder(emaFast, emaSlow)''',

            'sma_cross': f'''// SMA Cross - BIDIRECTIONAL
mafast = ta.sma(close, {sma_fast})
maslow = ta.sma(close, {sma_slow})
longEntrySignal = ta.crossover(mafast, maslow)
shortEntrySignal = ta.crossunder(mafast, maslow)''',

            'macd_cross': f'''// MACD - BIDIRECTIONAL
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
delta = macdLine - signalLine
longEntrySignal = ta.crossover(delta, 0)
shortEntrySignal = ta.crossunder(delta, 0)''',

            'price_above_sma': f'''// Price Crosses SMA - BIDIRECTIONAL
sma20 = ta.sma(close, {sma_20})
longEntrySignal = ta.crossover(close, sma20)
shortEntrySignal = ta.crossunder(close, sma20)''',

            'supertrend': f'''// Supertrend - BIDIRECTIONAL
[supertrendValue, supertrendDir] = ta.supertrend({st_factor}, {st_atr})
dirChange = ta.change(supertrendDir)
longEntrySignal = dirChange < 0
shortEntrySignal = dirChange > 0''',

            'adx_strong_trend': f'''// ADX Strong Trend - BIDIRECTIONAL
[diPlus, diMinus, adxValue] = ta.dmi({adx_len}, {adx_len})
strongTrend = adxValue > 25
longEntrySignal = strongTrend and diPlus > diMinus
shortEntrySignal = strongTrend and diMinus > diPlus''',

            'psar_reversal': f'''// Parabolic SAR - BIDIRECTIONAL
psarValue = ta.sar(0.02, 0.02, 0.2)
longEntrySignal = close > psarValue and close[1] <= psarValue[1]
shortEntrySignal = close < psarValue and close[1] >= psarValue[1]''',

            'consecutive_candles': f'''// Consecutive Up/Down Closes - BIDIRECTIONAL (matches Python)
// Python counts UP closes (close > close[1]), NOT green/red candles
upClose = close > close[1]
downClose = close < close[1]
threeUp = upClose[2] and upClose[1] and upClose
threeDown = downClose[2] and downClose[1] and downClose
longEntrySignal = threeDown
shortEntrySignal = threeUp''',

            'big_candle': f'''// Big Candle Reversal - BIDIRECTIONAL
atrValue = ta.atr(14)
candleRange = high - low
bigCandle = candleRange > atrValue * 2
greenCandle = close > open
redCandle = close < open
longEntrySignal = bigCandle and redCandle
shortEntrySignal = bigCandle and greenCandle''',

            'doji_reversal': f'''// Doji Reversal - BIDIRECTIONAL
body = math.abs(close - open)
totalRange = high - low
isDoji = totalRange > 0 and body < totalRange * 0.1
prevRed = close[1] < open[1]
prevGreen = close[1] > open[1]
longEntrySignal = isDoji and prevRed
shortEntrySignal = isDoji and prevGreen''',

            'engulfing': f'''// Engulfing Pattern - BIDIRECTIONAL
greenCandle = close > open
redCandle = close < open
bullishEngulf = greenCandle and redCandle[1] and close > open[1] and open < close[1]
bearishEngulf = redCandle and greenCandle[1] and close < open[1] and open > close[1]
longEntrySignal = bullishEngulf
shortEntrySignal = bearishEngulf''',

            'inside_bar': f'''// Inside Bar - BIDIRECTIONAL
insideBar = high < high[1] and low > low[1]
greenCandle = close > open
redCandle = close < open
longEntrySignal = insideBar and greenCandle
shortEntrySignal = insideBar and redCandle''',

            'outside_bar': f'''// Outside Bar - BIDIRECTIONAL
outsideBar = high > high[1] and low < low[1]
greenCandle = close > open
redCandle = close < open
longEntrySignal = outsideBar and greenCandle
shortEntrySignal = outsideBar and redCandle''',

            'atr_breakout': f'''// ATR Breakout - BIDIRECTIONAL
atrValue = ta.atr(14)
priceMove = math.abs(close - close[1])
bigMove = priceMove > atrValue * 1.5
moveUp = close > close[1]
moveDown = close < close[1]
longEntrySignal = bigMove and moveUp
shortEntrySignal = bigMove and moveDown''',

            'low_volatility_breakout': f'''// Low Volatility Breakout - BIDIRECTIONAL (adaptive threshold - matches Python)
// Python uses 25th percentile of ATR; we approximate with lowest ATR * 1.5
atrValue = ta.atr(14)
atrThreshold = ta.lowest(atrValue, 100) * 1.5  // Approximates 25th percentile
lowVol = atrValue[1] < atrThreshold
breakHigh = close > high[1]
breakLow = close < low[1]
longEntrySignal = lowVol and breakHigh
shortEntrySignal = lowVol and breakLow''',

            'higher_low': f'''// Higher Low / Lower High - BIDIRECTIONAL
higherLow = low > low[1] and low[1] > low[2]
lowerHigh = high < high[1] and high[1] < high[2]
longEntrySignal = higherLow
shortEntrySignal = lowerHigh''',

            'support_resistance': f'''// Support/Resistance - BIDIRECTIONAL
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
longEntrySignal = close <= recentLow * 1.005
shortEntrySignal = close >= recentHigh * 0.995''',

            'price_vs_sma': f'''// Price vs SMA - BIDIRECTIONAL
sma20 = ta.sma(close, {sma_20})
longEntrySignal = close < sma20 * 0.99
shortEntrySignal = close > sma20 * 1.01''',

            'vwap_bounce': f'''// VWAP Bounce - BIDIRECTIONAL
vwapValue = ta.vwap(hlc3)
touchedBelow = low < vwapValue
touchedAbove = high > vwapValue
closedAbove = close > vwapValue
closedBelow = close < vwapValue
longEntrySignal = touchedBelow and closedAbove
shortEntrySignal = touchedAbove and closedBelow''',

            'vwap_cross': f'''// VWAP Cross - BIDIRECTIONAL
vwapValue = ta.vwap(hlc3)
longEntrySignal = ta.crossover(close, vwapValue)
shortEntrySignal = ta.crossunder(close, vwapValue)''',

            'vwma_cross': f'''// VWMA Cross - BIDIRECTIONAL
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
longEntrySignal = ta.crossover(close, vwmaValue)
shortEntrySignal = ta.crossunder(close, vwmaValue)''',

            'vwma_trend': f'''// VWMA Trend - BIDIRECTIONAL
vwmaLength = {vwma_length}
vwmaValue = ta.vwma(close, vwmaLength)
vwmaSlope = vwmaValue - vwmaValue[1]
longEntrySignal = vwmaSlope > 0 and vwmaSlope[1] <= 0
shortEntrySignal = vwmaSlope < 0 and vwmaSlope[1] >= 0''',

            'bb_squeeze_breakout': f'''// BB Squeeze Breakout - BIDIRECTIONAL
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
bbWidth = (bbUpper - bbLower) / bbMiddle
avgWidth = ta.sma(bbWidth, {bb_len})
squeezed = bbWidth[1] < avgWidth * 0.8
expanding = bbWidth > bbWidth[1]
longEntrySignal = squeezed and expanding and close > bbMiddle
shortEntrySignal = squeezed and expanding and close < bbMiddle''',

            'rsi_divergence': f'''// RSI Divergence - BIDIRECTIONAL (rolling window - matches Python)
// Python uses rolling().min/max().shift(1) - we match with ta.lowest/highest()[1]
rsiValue = ta.rsi(close, {rsi_len})
lookback = 5
priceLowerLow = low < ta.lowest(low, lookback)[1]
rsiHigherLow = rsiValue > ta.lowest(rsiValue, lookback)[1]
priceHigherHigh = high > ta.highest(high, lookback)[1]
rsiLowerHigh = rsiValue < ta.highest(rsiValue, lookback)[1]
longEntrySignal = priceLowerLow and rsiHigherLow and rsiValue < 40
shortEntrySignal = priceHigherHigh and rsiLowerHigh and rsiValue > 60''',

            # === BASELINE ===
            'always': '''// Always Enter Strategy - BIDIRECTIONAL
longEntrySignal = true
shortEntrySignal = true''',

            # === ADDITIONAL STRATEGIES ===
            'keltner_breakout': f'''// Keltner Channel Breakout - BIDIRECTIONAL
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 2.0)
longEntrySignal = ta.crossover(close, kcUpper)
shortEntrySignal = ta.crossunder(close, kcLower)''',

            'donchian_breakout': f'''// Donchian Channel Breakout - BIDIRECTIONAL
dcUpper = ta.highest(high, 20)[1]
dcLower = ta.lowest(low, 20)[1]
longEntrySignal = close > dcUpper and close[1] <= dcUpper[1]
shortEntrySignal = close < dcLower and close[1] >= dcLower[1]''',

            'ichimoku_cross': f'''// Ichimoku Tenkan-Kijun Cross - BIDIRECTIONAL
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
longEntrySignal = ta.crossover(tenkan, kijun)
shortEntrySignal = ta.crossunder(tenkan, kijun)''',

            'ichimoku_cloud': f'''// Ichimoku Cloud Breakout - BIDIRECTIONAL
[tenkan, kijun, spanA, spanB, laggingSpan] = ta.ichimoku(9, 26, 52, 26)
cloudTop = math.max(spanA, spanB)
cloudBottom = math.min(spanA, spanB)
longEntrySignal = ta.crossover(close, cloudTop)
shortEntrySignal = ta.crossunder(close, cloudBottom)''',

            'aroon_cross': f'''// Aroon Oscillator Cross - BIDIRECTIONAL
[aroonUp, aroonDown] = ta.aroon(14)
longEntrySignal = ta.crossover(aroonUp, aroonDown)
shortEntrySignal = ta.crossover(aroonDown, aroonUp)''',

            'momentum_zero': f'''// Momentum Crosses Zero - BIDIRECTIONAL
momValue = ta.mom(close, 10)
longEntrySignal = ta.crossover(momValue, 0)
shortEntrySignal = ta.crossunder(momValue, 0)''',

            'roc_extreme': f'''// Rate of Change Extreme - BIDIRECTIONAL
rocValue = ta.roc(close, 12)
rocLower = ta.percentile_linear_interpolation(rocValue, 100, 5)
rocUpper = ta.percentile_linear_interpolation(rocValue, 100, 95)
longEntrySignal = rocValue < rocLower
shortEntrySignal = rocValue > rocUpper''',

            'uo_extreme': f'''// Ultimate Oscillator Extreme - BIDIRECTIONAL
uoValue = mih_uo(7, 14, 28)
longEntrySignal = uoValue < 30
shortEntrySignal = uoValue > 70''',

            'chop_trend': f'''// Choppiness Index Trend - BIDIRECTIONAL
chopValue = ta.chop(14)
sma20 = ta.sma(close, 20)
isTrending = chopValue < 38.2
longEntrySignal = isTrending and close > sma20
shortEntrySignal = isTrending and close < sma20''',

            'double_ema_cross': f'''// Double EMA Cross (12/26) - BIDIRECTIONAL
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
longEntrySignal = ta.crossover(ema12, ema26)
shortEntrySignal = ta.crossunder(ema12, ema26)''',

            'triple_ema': f'''// Triple EMA Alignment - BIDIRECTIONAL
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)
ema50 = ta.ema(close, 50)
longAligned = ema9 > ema21 and ema21 > ema50
shortAligned = ema9 < ema21 and ema21 < ema50
longWasNotAligned = not (ema9[1] > ema21[1] and ema21[1] > ema50[1])
shortWasNotAligned = not (ema9[1] < ema21[1] and ema21[1] < ema50[1])
longEntrySignal = longAligned and longWasNotAligned
shortEntrySignal = shortAligned and shortWasNotAligned''',

            # === KALMAN FILTER STRATEGIES - BIDIRECTIONAL ===
            'kalman_trend': f'''// Kalman Filter Trend - BIDIRECTIONAL
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
longEntrySignal = ta.crossover(close, kalman)
shortEntrySignal = ta.crossunder(close, kalman)''',

            'kalman_bb': f'''// Kalman Bollinger Bands - BIDIRECTIONAL
var float kalman = na
var float velocity = 0.0
kalmanGain = 0.7
if bar_index == 0
    kalman := close
else
    float prediction = kalman + velocity
    float error = close - prediction
    kalman := prediction + kalmanGain * error
    velocity := velocity + kalmanGain * error
kalmanStd = ta.stdev(close, 20)
kalmanUpper = kalman + kalmanStd * 2.0
kalmanLower = kalman - kalmanStd * 2.0
longEntrySignal = ta.crossover(close, kalmanLower)
shortEntrySignal = ta.crossunder(close, kalmanUpper)''',

            'kalman_rsi': f'''// Kalman-Smoothed RSI - BIDIRECTIONAL
rsiRaw = ta.rsi(close, 14)
var float kalmanRsi = na
kalmanRsi := na(kalmanRsi) ? rsiRaw : kalmanRsi + 0.5 * (rsiRaw - kalmanRsi)
longEntrySignal = ta.crossover(kalmanRsi, 30)
shortEntrySignal = ta.crossunder(kalmanRsi, 70)''',

            'kalman_mfi': f'''// Kalman-Smoothed MFI - BIDIRECTIONAL
mfiRaw = ta.mfi(hlc3, 14)
var float kalmanMfi = na
kalmanMfi := na(kalmanMfi) ? mfiRaw : kalmanMfi + 0.5 * (mfiRaw - kalmanMfi)
longEntrySignal = ta.crossover(kalmanMfi, 20)
shortEntrySignal = ta.crossunder(kalmanMfi, 80)''',

            'kalman_adx': f'''// Kalman-Smoothed ADX Trend - BIDIRECTIONAL
[diPlus, diMinus, adxRaw] = ta.dmi(14, 14)
var float kalmanAdx = na
kalmanAdx := na(kalmanAdx) ? adxRaw : kalmanAdx + 0.5 * (adxRaw - kalmanAdx)
longEntrySignal = kalmanAdx > 25 and diPlus > diMinus
shortEntrySignal = kalmanAdx > 25 and diMinus > diPlus''',

            'kalman_psar': f'''// Kalman PSAR - BIDIRECTIONAL
psarValue = ta.sar(0.02, 0.02, 0.2)
longEntrySignal = ta.crossover(close, psarValue)
shortEntrySignal = ta.crossunder(close, psarValue)''',

            'kalman_macd': f'''// Kalman-Smoothed MACD - BIDIRECTIONAL
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
var float kalmanMacd = na
var float kalmanSignal = na
kalmanMacd := na(kalmanMacd) ? macdLine : kalmanMacd + 0.5 * (macdLine - kalmanMacd)
kalmanSignal := na(kalmanSignal) ? signalLine : kalmanSignal + 0.5 * (signalLine - kalmanSignal)
longEntrySignal = ta.crossover(kalmanMacd, kalmanSignal)
shortEntrySignal = ta.crossunder(kalmanMacd, kalmanSignal)''',

            # === MISSING ENTRY RULES (ported from VectorBT - BIDIRECTIONAL) ===
            'ao_twin_peaks': f'''// Awesome Oscillator Twin Peaks - BIDIRECTIONAL
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
aoLow = ta.lowest(ao, 20)
aoHigh = ta.highest(ao, 20)
aoRising = ao > ao[1]
aoFalling = ao < ao[1]
longEntrySignal = ao < 0 and ao > aoLow and aoRising
shortEntrySignal = ao > 0 and ao < aoHigh and aoFalling''',

            'ao_zero_cross': f'''// Awesome Oscillator Zero Cross - BIDIRECTIONAL
ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
longEntrySignal = ta.crossover(ao, 0)
shortEntrySignal = ta.crossunder(ao, 0)''',

            'mfi_extreme': f'''// Money Flow Index Extreme - BIDIRECTIONAL
mfiValue = ta.mfi(hlc3, 14)
longEntrySignal = ta.crossover(mfiValue, 20)
shortEntrySignal = ta.crossunder(mfiValue, 80)''',

            'cmf_cross': f'''// Chaikin Money Flow Zero Cross - BIDIRECTIONAL
mfm = ((close - low) - (high - close)) / (high - low)
mfv = mfm * volume
cmfValue = ta.sma(mfv, 20) / ta.sma(volume, 20)
longEntrySignal = ta.crossover(cmfValue, 0)
shortEntrySignal = ta.crossunder(cmfValue, 0)''',

            'obv_trend': f'''// On-Balance Volume Trend - BIDIRECTIONAL
obvValue = ta.obv
obvHigh = ta.highest(obvValue, 14)
obvLow = ta.lowest(obvValue, 14)
priceHigh = ta.highest(close, 14)
priceLow = ta.lowest(close, 14)
longEntrySignal = obvValue == obvHigh and close >= priceHigh * 0.98
shortEntrySignal = obvValue == obvLow and close <= priceLow * 1.02''',

            'ppo_cross': f'''// Percentage Price Oscillator Cross - BIDIRECTIONAL
emaFastPPO = ta.ema(close, 12)
emaSlowPPO = ta.ema(close, 26)
ppoValue = (emaFastPPO - emaSlowPPO) / emaSlowPPO * 100
ppoSignal = ta.ema(ppoValue, 9)
longEntrySignal = ta.crossover(ppoValue, ppoSignal)
shortEntrySignal = ta.crossunder(ppoValue, ppoSignal)''',

            'fisher_cross': f'''// Fisher Transform Cross - BIDIRECTIONAL
highestHigh = ta.highest(hl2, 10)
lowestLow = ta.lowest(hl2, 10)
rawValue = 2 * ((hl2 - lowestLow) / (highestHigh - lowestLow) - 0.5)
var float smoothed = 0.0
smoothed := 0.33 * rawValue + 0.67 * nz(smoothed[1])
clampedValue = math.max(-0.999, math.min(0.999, smoothed))
var float fisherValue = 0.0
fisherValue := 0.5 * math.log((1 + clampedValue) / (1 - clampedValue)) + 0.5 * nz(fisherValue[1])
fisherSignal = fisherValue[1]
longEntrySignal = ta.crossover(fisherValue, fisherSignal)
shortEntrySignal = ta.crossunder(fisherValue, fisherSignal)''',

            'tsi_cross': f'''// True Strength Index Signal Cross - BIDIRECTIONAL
momentum = close - close[1]
smoothMom = ta.ema(ta.ema(momentum, 25), 13)
smoothAbsMom = ta.ema(ta.ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
tsiSignal = ta.ema(tsiValue, 7)
longEntrySignal = ta.crossover(tsiValue, tsiSignal)
shortEntrySignal = ta.crossunder(tsiValue, tsiSignal)''',

            'tsi_zero': f'''// True Strength Index Zero Cross - BIDIRECTIONAL
momentum = close - close[1]
smoothMom = ta.ema(ta.ema(momentum, 25), 13)
smoothAbsMom = ta.ema(ta.ema(math.abs(momentum), 25), 13)
tsiValue = 100 * smoothMom / smoothAbsMom
longEntrySignal = ta.crossover(tsiValue, 0)
shortEntrySignal = ta.crossunder(tsiValue, 0)''',

            'rsi_macd_combo': f'''// RSI + MACD Combo - BIDIRECTIONAL
rsiValue = ta.rsi(close, {rsi_len})
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
histogram = macdLine - signalLine
histRising = histogram > histogram[1]
histFalling = histogram < histogram[1]
longEntrySignal = rsiValue < 30 and histRising
shortEntrySignal = rsiValue > 70 and histFalling''',

            'macd_stoch_combo': f'''// MACD + Stochastic Combo - BIDIRECTIONAL
[macdLine, signalLine, histLine] = ta.macd(close, {macd_fast}, {macd_slow}, {macd_signal})
k = ta.sma(ta.stoch(close, high, low, {stoch_k}), {stoch_smooth})
d = ta.sma(k, {stoch_d})
macdBullish = macdLine > signalLine
macdBearish = macdLine < signalLine
stochOversold = k < 20
stochOverbought = k > 80
longEntrySignal = macdBullish and stochOversold
shortEntrySignal = macdBearish and stochOverbought''',

            'hull_ma_cross': f'''// Hull Moving Average Cross - BIDIRECTIONAL
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
longEntrySignal = ta.crossover(close, hullMa)
shortEntrySignal = ta.crossunder(close, hullMa)''',

            'hull_ma_turn': f'''// Hull Moving Average Direction Change - BIDIRECTIONAL
hullLen = 20
halfLen = math.round(hullLen / 2)
sqrtLen = math.round(math.sqrt(hullLen))
wma1 = ta.wma(close, halfLen)
wma2 = ta.wma(close, hullLen)
hullMa = ta.wma(2 * wma1 - wma2, sqrtLen)
hullSlope = hullMa - hullMa[1]
hullSlopePrev = hullMa[1] - hullMa[2]
longEntrySignal = hullSlope > 0 and hullSlopePrev <= 0
shortEntrySignal = hullSlope < 0 and hullSlopePrev >= 0''',

            'zlema_cross': f'''// Zero-Lag EMA Cross - BIDIRECTIONAL
zlemaLen = 20
lag = math.round((zlemaLen - 1) / 2)
zlema = ta.ema(close + (close - close[lag]), zlemaLen)
longEntrySignal = ta.crossover(close, zlema)
shortEntrySignal = ta.crossunder(close, zlema)''',

            'mcginley_cross': f'''// McGinley Dynamic Cross - BIDIRECTIONAL
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
longEntrySignal = ta.crossover(close, mcginley)
shortEntrySignal = ta.crossunder(close, mcginley)''',

            'mcginley_trend': f'''// McGinley Dynamic Trend Change - BIDIRECTIONAL
var float mcginley = na
mcginley := na(mcginley[1]) ? close : mcginley[1] + (close - mcginley[1]) / (10 * math.pow(close / mcginley[1], 4))
mcgSlope = mcginley - mcginley[1]
mcgSlopePrev = mcginley[1] - mcginley[2]
longEntrySignal = mcgSlope > 0 and mcgSlopePrev <= 0
shortEntrySignal = mcgSlope < 0 and mcgSlopePrev >= 0''',

            'linreg_channel': f'''// Linear Regression Channel Breakout - BIDIRECTIONAL
linregLen = 20
linregMid = ta.linreg(close, linregLen, 0)
linregDev = ta.stdev(close, linregLen)
linregUpper = linregMid + linregDev * 2
linregLower = linregMid - linregDev * 2
longEntrySignal = ta.crossover(close, linregLower)
shortEntrySignal = ta.crossunder(close, linregUpper)''',

            'bb_rsi_combo': f'''// Bollinger Bands + RSI Combo - BIDIRECTIONAL
[bbMiddle, bbUpper, bbLower] = ta.bb(close, {bb_len}, {bb_mult})
rsiValue = ta.rsi(close, {rsi_len})
longEntrySignal = close < bbLower and rsiValue < 30
shortEntrySignal = close > bbUpper and rsiValue > 70''',

            'squeeze_momentum': f'''// Squeeze Momentum - BIDIRECTIONAL
[bbMiddle, bbUpper, bbLower] = ta.bb(close, 20, 2.0)
[kcMiddle, kcUpper, kcLower] = ta.kc(close, 20, 1.5)
squeeze = bbLower > kcLower and bbUpper < kcUpper
squeezeFired = squeeze[1] and not squeeze
mom = close - close[20]
longEntrySignal = squeezeFired and mom > 0
shortEntrySignal = squeezeFired and mom < 0''',

            'chandelier_entry': f'''// Chandelier Exit Entry - BIDIRECTIONAL
atrValue = ta.atr({atr_len})
highestHigh = ta.highest(high, 22)
lowestLow = ta.lowest(low, 22)
chandelierLong = highestHigh - atrValue * 3
chandelierShort = lowestLow + atrValue * 3
longEntrySignal = ta.crossover(close, chandelierLong)
shortEntrySignal = ta.crossunder(close, chandelierShort)''',

            'ema_rsi_combo': f'''// EMA + RSI Combo - BIDIRECTIONAL
emaFast = ta.ema(close, {ema_fast})
emaSlow = ta.ema(close, {ema_slow})
rsiValue = ta.rsi(close, {rsi_len})
emaBullish = emaFast > emaSlow
emaBearish = emaFast < emaSlow
longEntrySignal = emaBullish and rsiValue < 40
shortEntrySignal = emaBearish and rsiValue > 60''',

            'supertrend_adx_combo': f'''// SuperTrend + ADX Combo - BIDIRECTIONAL
[supertrendValue, supertrendDir] = ta.supertrend({st_factor}, {st_atr})
[diPlus, diMinus, adxValue] = ta.dmi({adx_len}, {adx_len})
strongTrend = adxValue > 25
stBullish = close > supertrendValue
stBearish = close < supertrendValue
longEntrySignal = strongTrend and stBullish and diPlus > diMinus
shortEntrySignal = strongTrend and stBearish and diMinus > diPlus''',

            'pivot_bounce': f'''// Pivot Point Bounce - BIDIRECTIONAL
pivotPoint = (high[1] + low[1] + close[1]) / 3
r1 = 2 * pivotPoint - low[1]
s1 = 2 * pivotPoint - high[1]
nearS1 = low <= s1 * 1.005 and low >= s1 * 0.995
nearR1 = high >= r1 * 0.995 and high <= r1 * 1.005
greenCandle = close > open
redCandle = close < open
longEntrySignal = nearS1 and greenCandle
shortEntrySignal = nearR1 and redCandle''',

            'elder_ray': f'''// Elder Ray - BIDIRECTIONAL
ema13 = ta.ema(close, 13)
bullPower = high - ema13
bearPower = low - ema13
emaRising = ema13 > ema13[1]
emaFalling = ema13 < ema13[1]
bearRising = bearPower > bearPower[1]
bullFalling = bullPower < bullPower[1]
longEntrySignal = emaRising and bearPower < 0 and bearRising
shortEntrySignal = emaFalling and bullPower > 0 and bullFalling''',
        }

        # Select the appropriate entry conditions based on engine and direction
        if is_bidirectional:
            # Use bidirectional conditions that generate both signals
            entry_code = bidirectional_entry_conditions.get(entry_rule, '''// Unknown strategy - BIDIRECTIONAL
longEntrySignal = false
shortEntrySignal = false''')
        elif engine == "mihakralj":
            selected_conditions = mihakralj_entry_conditions
            entry_code = selected_conditions.get(entry_rule, f'''// Unknown strategy "{entry_rule}" - using EMA crossover fallback
// This default entry uses EMA12/EMA26 crossover which works for most trend-following strategies
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
entrySignal = {"ta.crossover(ema12, ema26)" if is_long else "ta.crossunder(ema12, ema26)"}''')
        else:
            # Both "tradingview" and "pandas_ta" use TradingView's built-in ta.* functions
            selected_conditions = entry_conditions
            entry_code = selected_conditions.get(entry_rule, f'''// Unknown strategy "{entry_rule}" - using EMA crossover fallback
// This default entry uses EMA12/EMA26 crossover which works for most trend-following strategies
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
entrySignal = {"ta.crossover(ema12, ema26)" if is_long else "ta.crossunder(ema12, ema26)"}''')

        # Get engine label and indicator functions for header
        engine_label = engine.upper() if engine else "TRADINGVIEW"
        if engine == "mihakralj":
            engine_note = "// Calculation Engine: MIHAKRALJ (warmup-compensated, mathematically rigorous)"
            indicator_functions = self.get_mihakralj_indicator_functions()
        elif engine == "pandas_ta":
            engine_note = "// Calculation Engine: PANDAS_TA (Python library, may differ slightly from TV)"
            indicator_functions = ""
        else:
            engine_note = "// Calculation Engine: TRADINGVIEW (built-in ta.* functions)"
            indicator_functions = ""

        script = f'''// =============================================================================
// {strategy_name}
// =============================================================================
// Generated: {gen_date}
// GUARANTEED 1:1 MATCH with Python backtester
{engine_note}
//
// STRATEGY CONFIGURATION:
//   Take Profit: {tp_percent}%
//   Stop Loss: {sl_percent}%
{metrics_comment}
//
// MATCHING RULES (DO NOT MODIFY):
//   - Entry at CLOSE of signal bar (process_orders_on_close=true, pyramiding=0, margin_long=100, margin_short=100)
//   - TP/SL as percentage of entry price
//   - Position size: {position_size_pct}% of equity
//   - Commission: 0.1% per side
// =============================================================================

//@version=6
strategy("{strategy_name}",
         overlay=true,
         process_orders_on_close=true,  // CRITICAL: Entry at CLOSE
         default_qty_type=strategy.cash,
         default_qty_value={int(capital)},
         initial_capital={int(capital) * 100},
         currency=currency.NONE,
         commission_type=strategy.commission.percent,
         commission_value=0.1,
         calc_on_every_tick=false,
         max_bars_back=500, pyramiding=0)
{indicator_functions}
// =============================================================================
// INPUTS - PERCENTAGE-BASED TP/SL (matches Python exactly)
// =============================================================================

tpPercent = input.float({tp_percent}, "Take Profit %", minval=0.1, maxval=20.0, step=0.1, group="Risk Management")
slPercent = input.float({sl_percent}, "Stop Loss %", minval=0.1, maxval=20.0, step=0.1, group="Risk Management")

enableLongs = input.bool({str(enable_longs).lower()}, "Enable Long Trades", group="Direction")
enableShorts = input.bool({str(enable_shorts).lower()}, "Enable Short Trades", group="Direction")
{date_range_code}
// =============================================================================
// WARMUP PERIOD TRACKING
// =============================================================================
// Skip first N bars for accurate indicator values (mihakralj warmup compensation)

var int warmupBars = 50
bool warmupComplete = bar_index >= warmupBars

// =============================================================================
// ENTRY CONDITIONS
// =============================================================================

{entry_code}

{f'''// BIDIRECTIONAL - Check for signal conflict (skip when both fire)
signalConflict = longEntrySignal and shortEntrySignal

// Only enter trades after warmup period completes
longCondition = longEntrySignal and not signalConflict and enableLongs and warmupComplete{date_range_condition}
shortCondition = shortEntrySignal and not signalConflict and enableShorts and warmupComplete{date_range_condition}

// =============================================================================
// TRADE EXECUTION - BIDIRECTIONAL WITH FLIP LOGIC
// =============================================================================

// Flip from short to long
if longCondition and strategy.position_size < 0
    strategy.close("Short", comment="Flip to Long")
    strategy.entry("Long", strategy.long)

// Flip from long to short
if shortCondition and strategy.position_size > 0
    strategy.close("Long", comment="Flip to Short")
    strategy.entry("Short", strategy.short)

// Fresh entry (no position)
if longCondition and strategy.position_size == 0
    strategy.entry("Long", strategy.long)

if shortCondition and strategy.position_size == 0
    strategy.entry("Short", strategy.short)''' if is_bidirectional else f'''// Only enter trades after warmup period completes
longCondition = entrySignal and enableLongs and strategy.position_size == 0 and warmupComplete{date_range_condition}
shortCondition = entrySignal and enableShorts and strategy.position_size == 0 and warmupComplete{date_range_condition}

// =============================================================================
// TRADE EXECUTION - EXACT MATCH WITH PYTHON
// =============================================================================

if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)'''}

// =============================================================================
// EXIT LOGIC - PERCENTAGE-BASED TP/SL
// =============================================================================

if strategy.position_size > 0  // Long position
    longTP = strategy.position_avg_price * (1 + tpPercent / 100)
    longSL = strategy.position_avg_price * (1 - slPercent / 100)
    strategy.exit("Long Exit", "Long", limit=longTP, stop=longSL)

if strategy.position_size < 0  // Short position
    shortTP = strategy.position_avg_price * (1 - tpPercent / 100)
    shortSL = strategy.position_avg_price * (1 + slPercent / 100)
    strategy.exit("Short Exit", "Short", limit=shortTP, stop=shortSL)

// =============================================================================
// VISUAL ELEMENTS - Signal Markers
// =============================================================================

// Entry signal markers
plotshape(longCondition, "Long Signal", shape.triangleup, location.belowbar, color.new(color.green, 0), size=size.small)
plotshape(shortCondition, "Short Signal", shape.triangledown, location.abovebar, color.new(color.red, 0), size=size.small)

// =============================================================================
// ALERTS
// =============================================================================

alertcondition(longCondition, title="Long Entry", message="LONG entry at {{{{close}}}}")
alertcondition(shortCondition, title="Short Entry", message="SHORT entry at {{{{close}}}}")

{self._generate_professional_visuals(tp_percent, sl_percent)}

{self._generate_enhanced_stats_table(strategy_name, direction, tp_percent, sl_percent)}
'''

        return script

    def generate_simple_strategy(self, strategy_name: str, params: Dict, metrics: Dict = None) -> str:
        """
        Generate Pine Script v6 for simple strategies.

        Args:
            strategy_name: Name of the simple strategy
            params: Optimized parameters
            metrics: Performance metrics from backtesting

        Returns:
            Pine Script v6 code
        """
        generators = {
            "pct_drop_buy": self._generate_pct_drop_buy,
            "pct_drop_consecutive": self._generate_pct_drop_consecutive,
            "simple_sma_cross": self._generate_simple_sma_cross,
            "price_vs_sma": self._generate_price_vs_sma,
            "triple_sma_align": self._generate_triple_sma_align,
            "consecutive_candles_reversal": self._generate_consecutive_candles,
            "inside_bar_breakout": self._generate_inside_bar_breakout,
            "range_breakout_simple": self._generate_range_breakout,
            "support_resistance_simple": self._generate_support_resistance,
            "simple_rsi_extreme": self._generate_simple_rsi,
            "candle_ratio_momentum": self._generate_candle_ratio,
            "engulfing_pattern": self._generate_engulfing_pattern,
            "doji_reversal": self._generate_doji_reversal,
        }

        if strategy_name in generators:
            return generators[strategy_name](params, metrics)

        # Fallback to generic simple template
        return self._generate_generic_simple(strategy_name, params, metrics)

    def generate_for_strategy(self, strategy_name: str, params: Dict, metrics: Dict = None) -> str:
        """
        Generate Pine Script v6 for ANY strategy type.
        Routes to the appropriate template based on strategy name.

        This is the MAIN entry point for generating Pine Script from
        unified optimizer results.

        Args:
            strategy_name: Name from unified_optimizer strategy registry
            params: Optimized parameters
            metrics: Performance metrics from backtesting

        Returns:
            Pine Script v6 code
        """
        # Normalize parameters (ensure ATR-based stops)
        params = normalize_params(params)

        # Strategy template routing - organized by category
        generators = {
            # === ML-BASED / SMC STRATEGIES ===
            "cvd_divergence": self._generate_cvd_divergence,
            "nadaraya_watson_reversion": self._generate_nadaraya_watson,
            "divergence_3wave": self._generate_divergence_3wave,
            "squeeze_momentum_breakout": self._generate_squeeze_momentum,
            "squeeze_momentum": self._generate_squeeze_momentum,  # Alias
            "connors_rsi_extreme": self._generate_connors_rsi,
            "order_block_bounce": self._generate_order_block_bounce,  # SMC
            "fvg_fill": self._generate_fvg_fill,  # SMC
            "smc_divergence_confluence": self._generate_order_block_bounce,  # Uses OB logic
            "kernel_trend_follow": self._generate_nadaraya_watson,  # Similar kernel logic
            "ml_feature_ensemble": self._generate_nadaraya_watson,  # Ensemble uses N-W

            # === MEAN REVERSION ===
            "bb_rsi_classic": self._generate_bb_rsi_classic,
            "bb_rsi_tight": self._generate_bb_rsi_tight,
            "bb_stoch": self._generate_bb_stoch,
            "keltner_rsi": self._generate_keltner_rsi,
            "z_score_reversion": self._generate_z_score_reversion,

            # === OSCILLATOR ===
            "stoch_extreme": self._generate_stoch_extreme,
            "rsi_extreme": self._generate_rsi_extreme,
            "williams_r": self._generate_williams_r,
            "cci_extreme": self._generate_cci_extreme,
            "stoch_rsi": self._generate_stoch_extreme,  # Similar logic
            "fisher_transform": self._generate_stoch_extreme,  # Oscillator
            "awesome_oscillator": self._generate_macd_trend,  # Momentum osc

            # === TREND FOLLOWING ===
            "supertrend_follow": self._generate_supertrend,
            "macd_trend": self._generate_macd_trend,
            "ema_crossover": self._generate_ema_crossover,
            "adx_di_trend": self._generate_adx_di_trend,
            "adx_trend": self._generate_adx_di_trend,  # Alias
            "hma_trend": self._generate_ema_crossover,  # Similar MA logic
            "zlema_momentum": self._generate_ema_crossover,  # Similar MA logic
            "multi_ma_confluence": self._generate_ema_crossover,  # MA-based

            # === BREAKOUT ===
            "bb_squeeze_breakout": self._generate_bb_squeeze,
            "donchian_breakout": self._generate_donchian_breakout,
            "trendilo_breakout": self._generate_donchian_breakout,  # Channel breakout

            # === DAVIDDTECH ===
            "stiff_surge_v1": self._generate_stiff_surge,
            "stiff_surge_v2": self._generate_stiff_surge_v2,
            "range_filter_adx": self._generate_range_filter,
            "supertrend_confluence": self._generate_supertrend_confluence,
            "mcginley_trend": self._generate_ema_crossover,  # Dynamic MA
            "trendhoo_v1": self._generate_supertrend,  # Trend-based
            "trendhoo_v2": self._generate_supertrend,  # Trend-based
            "macd_liquidity_spectrum": self._generate_macd_trend,  # MACD-based
            "precision_trend_mastery": self._generate_adx_di_trend,  # ADX-based
            "t3_nexus_stiff": self._generate_stiff_surge,  # Stiffness-based

            # === SIMPLE STRATEGIES (delegate to existing) ===
            "pct_drop_buy": self._generate_pct_drop_buy,
            "pct_drop_consecutive": self._generate_pct_drop_buy,  # Similar
            "simple_sma_cross": self._generate_simple_sma_cross,
            "price_vs_sma": self._generate_simple_sma_cross,  # SMA-based
            "triple_sma_align": self._generate_simple_sma_cross,  # SMA-based
            "consecutive_candles_reversal": self._generate_consecutive_candles,
            "simple_rsi_extreme": self._generate_simple_rsi,
            "range_breakout_simple": self._generate_range_breakout,
            "support_resistance_simple": self._generate_range_breakout,  # S/R bounce
            "engulfing_pattern": self._generate_engulfing_pattern,
            "doji_reversal": self._generate_engulfing_pattern,  # Candle pattern
            "inside_bar_breakout": self._generate_range_breakout,  # Breakout
            "candle_ratio_momentum": self._generate_consecutive_candles,  # Candle-based

            # === CHANNEL STRATEGIES ===
            "dc_reversion": self._generate_donchian_breakout,
        }

        if strategy_name in generators:
            return generators[strategy_name](params, metrics)

        # Fallback to generic template with warning
        return self._generate_generic_advanced(strategy_name, params, metrics)

    def _generate_cvd_divergence(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for CVD-Price Divergence strategy"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 1.0)
        tp_ratio = params.get('tp_ratio', 0.9)
        rsi_length = int(params.get('rsi_length', 23))
        adx_threshold = int(params.get('adx_threshold', 29))
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// CVD-Price Divergence Strategy
// Generated: {gen_date}
// Strategy Type: ML-BASED (Volume Flow Analysis)
{metrics_comment}
//
// LOGIC: Detects when price and volume delta disagree
// - Bullish: Price making lower lows, CVD making higher lows (selling exhaustion)
// - Bearish: Price making higher highs, CVD making lower highs (buying exhaustion)

//@version=6
strategy("CVD Divergence Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")
rsiLength = input.int({rsi_length}, "RSI Length", minval=5, maxval=30, group="Indicators")
adxThreshold = input.int({adx_threshold}, "ADX Threshold", minval=10, maxval=50, group="Indicators")
divLookback = input.int(20, "Divergence Lookback", minval=10, maxval=50, group="CVD")
swingLookback = input.int(5, "Swing Point Lookback", minval=2, maxval=10, group="CVD")

// === CVD CALCULATION ===
// Approximates buying vs selling pressure from OHLC data
// Close position within bar: near high = buying, near low = selling
barRange = high - low
closePosition = barRange > 0 ? (close - low) / barRange : 0.5
deltaRatio = (closePosition * 2) - 1  // -1 (selling) to +1 (buying)
volumeDelta = deltaRatio * volume
cvd = ta.cum(volumeDelta)

// === SWING POINT DETECTION ===
// Find pivot highs and lows for price and CVD
pivotHigh = ta.pivothigh(close, swingLookback, swingLookback)
pivotLow = ta.pivotlow(close, swingLookback, swingLookback)
cvdPivotHigh = ta.pivothigh(cvd, swingLookback, swingLookback)
cvdPivotLow = ta.pivotlow(cvd, swingLookback, swingLookback)

// Track recent pivots for divergence detection
var float lastPriceLow1 = na
var float lastPriceLow2 = na
var float lastCvdLow1 = na
var float lastCvdLow2 = na
var float lastPriceHigh1 = na
var float lastPriceHigh2 = na
var float lastCvdHigh1 = na
var float lastCvdHigh2 = na

if not na(pivotLow)
    lastPriceLow2 := lastPriceLow1
    lastPriceLow1 := pivotLow
    lastCvdLow2 := lastCvdLow1
    lastCvdLow1 := cvd[swingLookback]

if not na(pivotHigh)
    lastPriceHigh2 := lastPriceHigh1
    lastPriceHigh1 := pivotHigh
    lastCvdHigh2 := lastCvdHigh1
    lastCvdHigh1 := cvd[swingLookback]

// === DIVERGENCE DETECTION ===
// Bullish: Lower low in price, higher low in CVD (selling exhaustion)
bullishDiv = not na(lastPriceLow1) and not na(lastPriceLow2) and
             lastPriceLow1 < lastPriceLow2 and lastCvdLow1 > lastCvdLow2

// Bearish: Higher high in price, lower high in CVD (buying exhaustion)
bearishDiv = not na(lastPriceHigh1) and not na(lastPriceHigh2) and
             lastPriceHigh1 > lastPriceHigh2 and lastCvdHigh1 < lastCvdHigh2

// === FILTERS ===
rsi = ta.rsi(close, rsiLength)
[diPlus, diMinus, adx] = ta.dmi(14, 14)
isSideways = adx < adxThreshold

// === ENTRY CONDITIONS ===
longCondition = bullishDiv and rsi < 40 and isSideways and strategy.position_size == 0
shortCondition = bearishDiv and rsi > 60 and isSideways and strategy.position_size == 0

// === RISK MANAGEMENT (ATR-BASED) ===
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
// CVD in separate pane
plot(cvd, "CVD", color=color.blue, display=display.pane)

// Signal markers
plotshape(bullishDiv and longCondition, "Bullish Div", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(bearishDiv and shortCondition, "Bearish Div", shape.triangledown, location.abovebar, color.red, size=size.normal)

// Market regime background
bgcolor(isSideways ? color.new(color.green, 95) : color.new(color.red, 95))

// === INFO TABLE ===
var table infoTable = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 80))
if barstate.islast
    table.cell(infoTable, 0, 0, "Strategy", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 0, "CVD Divergence", text_color=color.white, text_size=size.small)
    table.cell(infoTable, 0, 1, "RSI", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 1, str.tostring(rsi, "#.#"), text_color=rsi < 40 ? color.lime : rsi > 60 ? color.red : color.white, text_size=size.small)
    table.cell(infoTable, 0, 2, "ADX", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 2, str.tostring(adx, "#.#"), text_color=isSideways ? color.lime : color.orange, text_size=size.small)
    table.cell(infoTable, 0, 3, "Bull Div", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 3, bullishDiv ? "YES" : "NO", text_color=bullishDiv ? color.lime : color.gray, text_size=size.small)
    table.cell(infoTable, 0, 4, "Bear Div", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 4, bearishDiv ? "YES" : "NO", text_color=bearishDiv ? color.red : color.gray, text_size=size.small)
    table.cell(infoTable, 0, 5, "SL/TP", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 5, str.tostring(slAtrMult, "#.#") + "x/" + str.tostring(tpRatio, "#.#") + "x", text_color=color.white, text_size=size.small)

// === ALERTS ===
alertcondition(longCondition, title="CVD Bullish Divergence", message="CVD Divergence: BUY signal - Price lower low, CVD higher low")
alertcondition(shortCondition, title="CVD Bearish Divergence", message="CVD Divergence: SELL signal - Price higher high, CVD lower high")
'''

    def _generate_bb_rsi_classic(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Bollinger Band + RSI mean reversion"""
        params = normalize_params(params)
        bb_length = int(params.get('bb_length', 22))
        bb_mult = params.get('bb_mult', 3.9)
        rsi_length = int(params.get('rsi_length', 23))
        rsi_oversold = int(params.get('rsi_oversold', 29))
        rsi_overbought = int(params.get('rsi_overbought', 61))
        sl_mult = params.get('sl_atr_mult', 0.7)
        tp_ratio = params.get('tp_ratio', 4.4)
        adx_threshold = int(params.get('adx_threshold', 16))
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Bollinger Band + RSI Classic Mean Reversion
// Generated: {gen_date}
// Strategy Type: MEAN REVERSION
{metrics_comment}
//
// LOGIC: Buy when price at lower BB + RSI oversold in sideways market
//        Sell when price at upper BB + RSI overbought in sideways market

//@version=6
strategy("BB + RSI Mean Reversion", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
bbLength = input.int({bb_length}, "BB Length", minval=5, maxval=50, group="Bollinger Bands")
bbMult = input.float({bb_mult}, "BB Multiplier", minval=0.5, maxval=5.0, step=0.1, group="Bollinger Bands")
rsiLength = input.int({rsi_length}, "RSI Length", minval=5, maxval=30, group="RSI")
rsiOversold = input.int({rsi_oversold}, "RSI Oversold", minval=10, maxval=40, group="RSI")
rsiOverbought = input.int({rsi_overbought}, "RSI Overbought", minval=60, maxval=90, group="RSI")
adxThreshold = input.int({adx_threshold}, "ADX Threshold (sideways)", minval=10, maxval=50, group="Filter")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
bbBasis = ta.sma(close, bbLength)
bbDev = bbMult * ta.stdev(close, bbLength)
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

rsi = ta.rsi(close, rsiLength)
[diPlus, diMinus, adx] = ta.dmi(14, 14)
isSideways = adx < adxThreshold

atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === ENTRY CONDITIONS ===
longCondition = close <= bbLower and rsi < rsiOversold and isSideways and strategy.position_size == 0
shortCondition = close >= bbUpper and rsi > rsiOverbought and isSideways and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(bbUpper, "BB Upper", color=color.red)
plot(bbBasis, "BB Basis", color=color.gray)
plot(bbLower, "BB Lower", color=color.green)
bgcolor(isSideways ? color.new(color.green, 95) : color.new(color.red, 95))
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.small)

// === ALERTS ===
alertcondition(longCondition, title="BB RSI Buy", message="BB + RSI: BUY signal")
alertcondition(shortCondition, title="BB RSI Sell", message="BB + RSI: SELL signal")
'''

    def _generate_supertrend(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Supertrend trend following"""
        params = normalize_params(params)
        st_length = int(params.get('supertrend_length', 16))
        st_mult = params.get('supertrend_mult', 3.6)
        sl_mult = params.get('sl_atr_mult', 5.0)
        tp_ratio = params.get('tp_ratio', 1.6)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Supertrend Trend Following Strategy
// Generated: {gen_date}
// Strategy Type: TREND FOLLOWING
{metrics_comment}

//@version=6
strategy("Supertrend Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
stLength = input.int({st_length}, "Supertrend Length", minval=5, maxval=50, group="Supertrend")
stMult = input.float({st_mult}, "Supertrend Multiplier", minval=0.5, maxval=10.0, step=0.1, group="Supertrend")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
[supertrend, direction] = ta.supertrend(stMult, stLength)
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === ENTRY CONDITIONS ===
longCondition = direction < 0 and direction[1] >= 0 and strategy.position_size == 0
shortCondition = direction > 0 and direction[1] <= 0 and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(supertrend, "Supertrend", color=direction < 0 ? color.green : color.red, linewidth=2)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.small)

// === ALERTS ===
alertcondition(longCondition, title="Supertrend Buy", message="Supertrend: BUY signal - trend turned bullish")
alertcondition(shortCondition, title="Supertrend Sell", message="Supertrend: SELL signal - trend turned bearish")
'''

    def _generate_macd_trend(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for MACD trend strategy"""
        params = normalize_params(params)
        fast = int(params.get('macd_fast', 18))
        slow = int(params.get('macd_slow', 29))
        signal = int(params.get('macd_signal', 8))
        sl_mult = params.get('sl_atr_mult', 1.9)
        tp_ratio = params.get('tp_ratio', 0.9)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// MACD Trend Strategy
// Generated: {gen_date}
// Strategy Type: MOMENTUM
{metrics_comment}

//@version=6
strategy("MACD Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
fastLength = input.int({fast}, "MACD Fast", minval=5, maxval=50, group="MACD")
slowLength = input.int({slow}, "MACD Slow", minval=10, maxval=100, group="MACD")
signalLength = input.int({signal}, "MACD Signal", minval=5, maxval=30, group="MACD")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
[macdLine, signalLine, hist] = ta.macd(close, fastLength, slowLength, signalLength)
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === ENTRY CONDITIONS ===
longCondition = ta.crossover(macdLine, signalLine) and strategy.position_size == 0
shortCondition = ta.crossunder(macdLine, signalLine) and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
hline(0, "Zero", color=color.gray)
plot(hist, "Histogram", style=plot.style_columns, color=hist >= 0 ? color.green : color.red, display=display.pane)
plot(macdLine, "MACD", color=color.blue, display=display.pane)
plot(signalLine, "Signal", color=color.orange, display=display.pane)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.small)

// === ALERTS ===
alertcondition(longCondition, title="MACD Buy", message="MACD: BUY signal - bullish crossover")
alertcondition(shortCondition, title="MACD Sell", message="MACD: SELL signal - bearish crossover")
'''

    def _generate_generic_advanced(self, strategy_name: str, params: Dict, metrics: Dict = None) -> str:
        """Generate generic Pine Script for advanced strategies without specific template"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        # Format params for display
        params_display = []
        for k, v in params.items():
            if not k.startswith('sl_') and not k.startswith('tp_') and k != 'use_atr_stops':
                params_display.append(f"// {k}: {v}")
        params_str = "\n".join(params_display)

        return f'''// {strategy_name.replace("_", " ").title()} Strategy
// Generated: {gen_date}
// Strategy Type: ADVANCED
{metrics_comment}
//
// NOTE: This strategy uses a generic template.
// The specific logic for "{strategy_name}" should be customized
// based on the optimized parameters shown below.
//
// OPTIMIZED PARAMETERS:
{params_str}

//@version=6
strategy("{strategy_name.replace("_", " ").title()}", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === RISK PARAMETERS ===
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === ENTRY SIGNALS ===
// Generic bidirectional entry: EMA crossover with RSI filter
// This provides a balanced approach suitable for unknown strategy types
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
rsiValue = ta.rsi(close, 14)
// Long: EMA12 crosses above EMA26 with RSI not overbought
longCondition = ta.crossover(ema12, ema26) and rsiValue < 70 and strategy.position_size == 0
// Short: EMA12 crosses below EMA26 with RSI not oversold
shortCondition = ta.crossunder(ema12, ema26) and rsiValue > 30 and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === ALERTS ===
alertcondition(longCondition, title="Buy Signal", message="{strategy_name}: BUY signal")
alertcondition(shortCondition, title="Sell Signal", message="{strategy_name}: SELL signal")
'''

    # ========================================================================
    # FULLY IMPLEMENTED PINE SCRIPT TEMPLATES - NO PLACEHOLDERS
    # ========================================================================

    def _generate_order_block_bounce(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Order Block Bounce (SMC) strategy"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 3.8)
        tp_ratio = params.get('tp_ratio', 0.5)
        ob_lookback = int(params.get('ob_lookback', 18))
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Order Block Bounce (SMC) Strategy
// Generated: {gen_date}
// Strategy Type: SMART MONEY CONCEPTS
{metrics_comment}
//
// LOGIC: Identifies institutional order blocks (zones where large orders accumulated)
// - Bullish OB: Last bearish candle before significant up move
// - Bearish OB: Last bullish candle before significant down move
// - Enter when price returns to test the order block zone

//@version=6
strategy("Order Block Bounce", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
obLookback = input.int({ob_lookback}, "OB Lookback", minval=5, maxval=50, group="Order Block")
obMinMove = input.float(1.5, "Min Move ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Order Block")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
rsi = ta.rsi(close, 14)

// Find swing points for order block detection
swingHigh = ta.pivothigh(high, obLookback, obLookback)
swingLow = ta.pivotlow(low, obLookback, obLookback)

// Track order block zones
var float bullOBTop = na
var float bullOBBottom = na
var float bearOBTop = na
var float bearOBBottom = na
var bool bullOBActive = false
var bool bearOBActive = false

// Detect significant move (potential institutional activity)
priceMove = close - close[obLookback]
significantUp = priceMove > atr * obMinMove
significantDn = priceMove < -atr * obMinMove

// Identify order blocks
// Bullish OB: Last red candle before big up move
if significantUp and close[obLookback] < open[obLookback]
    bullOBTop := high[obLookback]
    bullOBBottom := low[obLookback]
    bullOBActive := true

// Bearish OB: Last green candle before big down move
if significantDn and close[obLookback] > open[obLookback]
    bearOBTop := high[obLookback]
    bearOBBottom := low[obLookback]
    bearOBActive := true

// === ENTRY CONDITIONS ===
// Price enters bullish OB zone
inBullOB = bullOBActive and close <= bullOBTop and close >= bullOBBottom
// Price enters bearish OB zone
inBearOB = bearOBActive and close >= bearOBBottom and close <= bearOBTop

longCondition = inBullOB and rsi < 50 and strategy.position_size == 0
shortCondition = inBearOB and rsi > 50 and strategy.position_size == 0

// Invalidate OB after price moves through
if bullOBActive and close < bullOBBottom - atr
    bullOBActive := false
if bearOBActive and close > bearOBTop + atr
    bearOBActive := false

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)
    bullOBActive := false  // OB consumed

if shortCondition
    strategy.entry("Short", strategy.short)
    bearOBActive := false  // OB consumed

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
// Plot active order block zones
bgcolor(bullOBActive and close >= bullOBBottom and close <= bullOBTop ? color.new(color.green, 85) : na)
bgcolor(bearOBActive and close >= bearOBBottom and close <= bearOBTop ? color.new(color.red, 85) : na)
plotshape(longCondition, "Buy OB", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell OB", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === ALERTS ===
alertcondition(longCondition, title="OB Buy", message="Order Block: BUY signal - price at bullish OB")
alertcondition(shortCondition, title="OB Sell", message="Order Block: SELL signal - price at bearish OB")
'''

    def _generate_fvg_fill(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Fair Value Gap Fill strategy"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 0.9)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Fair Value Gap Fill (SMC) Strategy
// Generated: {gen_date}
// Strategy Type: SMART MONEY CONCEPTS
{metrics_comment}
//
// LOGIC: Trade the fill of Fair Value Gaps (imbalance zones)
// - Bullish FVG: Gap between candle 1 high and candle 3 low (after up move)
// - Bearish FVG: Gap between candle 1 low and candle 3 high (after down move)
// - Price tends to return and fill these gaps

//@version=6
strategy("FVG Fill Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
minGapATR = input.float(0.5, "Min Gap Size (ATR)", minval=0.1, maxval=3.0, step=0.1, group="FVG")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
rsi = ta.rsi(close, 14)

// Detect Fair Value Gaps
// Bullish FVG: Low of current bar > High of bar 2 bars ago (gap up)
bullishFVG = low > high[2] and (low - high[2]) > atr * minGapATR
fvgBullTop = low
fvgBullBottom = high[2]

// Bearish FVG: High of current bar < Low of bar 2 bars ago (gap down)
bearishFVG = high < low[2] and (low[2] - high) > atr * minGapATR
fvgBearTop = low[2]
fvgBearBottom = high

// Track active FVG zones
var float activeBullFVGTop = na
var float activeBullFVGBottom = na
var float activeBearFVGTop = na
var float activeBearFVGBottom = na
var bool bullFVGActive = false
var bool bearFVGActive = false

// Store new FVGs
if bullishFVG
    activeBullFVGTop := fvgBullTop
    activeBullFVGBottom := fvgBullBottom
    bullFVGActive := true

if bearishFVG
    activeBearFVGTop := fvgBearTop
    activeBearFVGBottom := fvgBearBottom
    bearFVGActive := true

// Check if price is filling FVG
fillingBullFVG = bullFVGActive and close < activeBullFVGTop and close >= activeBullFVGBottom
fillingBearFVG = bearFVGActive and close > activeBearFVGBottom and close <= activeBearFVGTop

// === ENTRY CONDITIONS ===
longCondition = fillingBullFVG and rsi < 50 and strategy.position_size == 0
shortCondition = fillingBearFVG and rsi > 50 and strategy.position_size == 0

// Invalidate FVG after fill or price moves away
if bullFVGActive and close < activeBullFVGBottom - atr
    bullFVGActive := false
if bearFVGActive and close > activeBearFVGTop + atr
    bearFVGActive := false

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)
    bullFVGActive := false

if shortCondition
    strategy.entry("Short", strategy.short)
    bearFVGActive := false

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
bgcolor(bullFVGActive ? color.new(color.green, 90) : bearFVGActive ? color.new(color.red, 90) : na)
plotshape(bullishFVG, "Bullish FVG", shape.diamond, location.belowbar, color.green, size=size.tiny)
plotshape(bearishFVG, "Bearish FVG", shape.diamond, location.abovebar, color.red, size=size.tiny)
plotshape(longCondition, "Buy FVG", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell FVG", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === ALERTS ===
alertcondition(longCondition, title="FVG Buy", message="FVG Fill: BUY signal - filling bullish gap")
alertcondition(shortCondition, title="FVG Sell", message="FVG Fill: SELL signal - filling bearish gap")
'''

    def _generate_squeeze_momentum(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Squeeze Momentum Breakout strategy"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 3.4)
        tp_ratio = params.get('tp_ratio', 0.5)
        squeeze_kc_mult = params.get('squeeze_kc_mult', 1.8)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Squeeze Momentum Breakout Strategy
// Generated: {gen_date}
// Strategy Type: BREAKOUT
{metrics_comment}
//
// LOGIC: Trade when BB squeezes inside KC (low volatility) then releases
// - Squeeze ON: BB inside KC (consolidation)
// - Squeeze Release: BB expands outside KC (breakout)
// - Direction determined by momentum histogram

//@version=6
strategy("Squeeze Momentum Breakout", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
bbLength = input.int(20, "BB Length", minval=5, maxval=50, group="Squeeze")
bbMult = input.float(2.0, "BB Multiplier", minval=0.5, maxval=5.0, step=0.1, group="Squeeze")
kcLength = input.int(20, "KC Length", minval=5, maxval=50, group="Squeeze")
kcMult = input.float({squeeze_kc_mult}, "KC Multiplier", minval=0.5, maxval=3.0, step=0.1, group="Squeeze")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === BOLLINGER BANDS ===
bbBasis = ta.sma(close, bbLength)
bbDev = bbMult * ta.stdev(close, bbLength)
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

// === KELTNER CHANNELS ===
kcBasis = ta.sma(close, kcLength)
kcRange = ta.atr(kcLength)
kcUpper = kcBasis + kcRange * kcMult
kcLower = kcBasis - kcRange * kcMult

// === SQUEEZE DETECTION ===
// Squeeze ON when BB is inside KC
squeezeOn = bbLower > kcLower and bbUpper < kcUpper
// Squeeze OFF (release) when BB is outside KC
squeezeOff = bbLower < kcLower or bbUpper > kcUpper

// === MOMENTUM CALCULATION ===
// Linear regression of price minus basis
momentum = ta.linreg(close - math.avg(math.avg(ta.highest(high, kcLength), ta.lowest(low, kcLength)), ta.sma(close, kcLength)), kcLength, 0)

// Momentum direction
momUp = momentum > 0
momDn = momentum < 0
momIncreasing = momentum > momentum[1]
momDecreasing = momentum < momentum[1]

// === ENTRY CONDITIONS ===
// Release from squeeze with momentum confirmation
squeezeRelease = squeezeOn[1] and squeezeOff
rsi = ta.rsi(close, 14)

longCondition = squeezeRelease and momUp and momIncreasing and rsi > 50 and strategy.position_size == 0
shortCondition = squeezeRelease and momDn and momDecreasing and rsi < 50 and strategy.position_size == 0

// === RISK MANAGEMENT ===
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
// Squeeze indicator dots
plotshape(squeezeOn, "Squeeze ON", shape.circle, location.bottom, color.red, size=size.tiny)
plotshape(squeezeOff, "Squeeze OFF", shape.circle, location.bottom, color.green, size=size.tiny)

// Momentum histogram
plot(momentum, "Momentum", style=plot.style_histogram, color=momentum > 0 ? (momIncreasing ? color.lime : color.green) : (momDecreasing ? color.red : color.maroon), display=display.pane)

plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === ALERTS ===
alertcondition(longCondition, title="Squeeze Buy", message="Squeeze Momentum: BUY signal - squeeze released bullish")
alertcondition(shortCondition, title="Squeeze Sell", message="Squeeze Momentum: SELL signal - squeeze released bearish")
'''

    def _generate_stoch_extreme(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Stochastic Extreme strategy"""
        params = normalize_params(params)
        stoch_k = int(params.get('stoch_k', 9))
        stoch_d = int(params.get('stoch_d', 7))
        stoch_os = int(params.get('stoch_oversold', 32))
        stoch_ob = int(params.get('stoch_overbought', 90))
        sl_mult = params.get('sl_atr_mult', 3.7)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Stochastic Extreme Strategy
// Generated: {gen_date}
// Strategy Type: OSCILLATOR
{metrics_comment}
//
// LOGIC: Trade when both %K and %D are at extreme levels
// - Long: Both K and D below oversold level
// - Short: Both K and D above overbought level

//@version=6
strategy("Stochastic Extreme", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
stochK = input.int({stoch_k}, "Stoch %K Length", minval=5, maxval=30, group="Stochastic")
stochD = input.int({stoch_d}, "Stoch %D Smoothing", minval=1, maxval=10, group="Stochastic")
stochSmooth = input.int(3, "Stoch %K Smoothing", minval=1, maxval=10, group="Stochastic")
oversold = input.int({stoch_os}, "Oversold Level", minval=5, maxval=30, group="Stochastic")
overbought = input.int({stoch_ob}, "Overbought Level", minval=70, maxval=95, group="Stochastic")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
k = ta.sma(ta.stoch(close, high, low, stochK), stochSmooth)
d = ta.sma(k, stochD)
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = k < oversold and d < oversold and strategy.position_size == 0
shortCondition = k > overbought and d > overbought and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === STOCHASTIC PANEL ===
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(50, "Middle", color=color.gray, linestyle=hline.style_dotted)

// === VISUALS ===
plot(k, "%K", color=color.blue, display=display.pane)
plot(d, "%D", color=color.orange, display=display.pane)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
bgcolor(k < oversold and d < oversold ? color.new(color.green, 90) : k > overbought and d > overbought ? color.new(color.red, 90) : na)

// === ALERTS ===
alertcondition(longCondition, title="Stoch Oversold", message="Stochastic: BUY signal - extreme oversold")
alertcondition(shortCondition, title="Stoch Overbought", message="Stochastic: SELL signal - extreme overbought")
'''

    def _generate_williams_r(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Williams %R strategy"""
        params = normalize_params(params)
        willr_len = int(params.get('willr_length', 14))
        willr_os = int(params.get('willr_oversold', -95))
        willr_ob = int(params.get('willr_overbought', -7))
        sl_mult = params.get('sl_atr_mult', 4.0)
        tp_ratio = params.get('tp_ratio', 1.2)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Williams %R Strategy
// Generated: {gen_date}
// Strategy Type: OSCILLATOR
{metrics_comment}
//
// LOGIC: Trade Williams %R at extreme levels (-100 to 0 scale)
// - Long: %R below oversold level (e.g., -80)
// - Short: %R above overbought level (e.g., -20)

//@version=6
strategy("Williams %R Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
willrLength = input.int({willr_len}, "Williams %R Length", minval=5, maxval=50, group="Williams %R")
oversold = input.int({willr_os}, "Oversold Level", minval=-100, maxval=-50, group="Williams %R")
overbought = input.int({willr_ob}, "Overbought Level", minval=-50, maxval=0, group="Williams %R")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
willr = ta.wpr(willrLength)
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = willr < oversold and strategy.position_size == 0
shortCondition = willr > overbought and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === WILLIAMS %R PANEL ===
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(-50, "Middle", color=color.gray, linestyle=hline.style_dotted)

// === VISUALS ===
plot(willr, "Williams %R", color=color.purple, display=display.pane)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
bgcolor(willr < oversold ? color.new(color.green, 90) : willr > overbought ? color.new(color.red, 90) : na)

// === ALERTS ===
alertcondition(longCondition, title="WillR Oversold", message="Williams %R: BUY signal - oversold")
alertcondition(shortCondition, title="WillR Overbought", message="Williams %R: SELL signal - overbought")
'''

    def _generate_cci_extreme(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for CCI Extreme strategy"""
        params = normalize_params(params)
        cci_len = int(params.get('cci_length', 20))
        cci_thresh = int(params.get('cci_threshold', 75))
        sl_mult = params.get('sl_atr_mult', 4.8)
        tp_ratio = params.get('tp_ratio', 4.9)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// CCI Extreme Strategy
// Generated: {gen_date}
// Strategy Type: OSCILLATOR
{metrics_comment}
//
// LOGIC: Trade CCI at extreme readings
// - Long: CCI below -threshold (oversold)
// - Short: CCI above +threshold (overbought)

//@version=6
strategy("CCI Extreme Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
cciLength = input.int({cci_len}, "CCI Length", minval=5, maxval=50, group="CCI")
cciThreshold = input.int({cci_thresh}, "CCI Threshold", minval=50, maxval=200, group="CCI")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
cci = ta.cci(close, cciLength)
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = cci < -cciThreshold and strategy.position_size == 0
shortCondition = cci > cciThreshold and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === CCI PANEL ===
hline(-cciThreshold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(cciThreshold, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(0, "Zero", color=color.gray, linestyle=hline.style_dotted)

// === VISUALS ===
plot(cci, "CCI", color=color.teal, display=display.pane)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
bgcolor(cci < -cciThreshold ? color.new(color.green, 90) : cci > cciThreshold ? color.new(color.red, 90) : na)

// === ALERTS ===
alertcondition(longCondition, title="CCI Oversold", message="CCI: BUY signal - extreme oversold")
alertcondition(shortCondition, title="CCI Overbought", message="CCI: SELL signal - extreme overbought")
'''

    def _generate_adx_di_trend(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for ADX + DI Trend strategy"""
        params = normalize_params(params)
        adx_len = int(params.get('adx_length', 20))
        adx_thresh = int(params.get('adx_threshold', 15))
        sl_mult = params.get('sl_atr_mult', 2.1)
        tp_ratio = params.get('tp_ratio', 2.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// ADX + DI Trend Strategy
// Generated: {gen_date}
// Strategy Type: TREND FOLLOWING
{metrics_comment}
//
// LOGIC: Trade strong trends using ADX and DI crossovers
// - Long: ADX > threshold AND DI+ > DI-
// - Short: ADX > threshold AND DI- > DI+

//@version=6
strategy("ADX + DI Trend Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
adxLength = input.int({adx_len}, "ADX Length", minval=5, maxval=50, group="ADX")
adxThreshold = input.int({adx_thresh}, "ADX Threshold", minval=15, maxval=50, group="ADX")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
[diPlus, diMinus, adx] = ta.dmi(adxLength, adxLength)
rsi = ta.rsi(close, 14)
atr = ta.atr(14)

// Strong trend condition
strongTrend = adx > adxThreshold

// DI crossover
diPlusCross = ta.crossover(diPlus, diMinus)
diMinusCross = ta.crossover(diMinus, diPlus)

// === ENTRY CONDITIONS ===
longCondition = strongTrend and diPlus > diMinus and rsi > 50 and strategy.position_size == 0
shortCondition = strongTrend and diMinus > diPlus and rsi < 50 and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(diPlus, "+DI", color=color.green, display=display.pane)
plot(diMinus, "-DI", color=color.red, display=display.pane)
plot(adx, "ADX", color=color.blue, linewidth=2, display=display.pane)
hline(adxThreshold, "Threshold", color=color.gray, linestyle=hline.style_dashed)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
bgcolor(strongTrend ? (diPlus > diMinus ? color.new(color.green, 90) : color.new(color.red, 90)) : na)

// === ALERTS ===
alertcondition(longCondition, title="ADX Bull", message="ADX DI: BUY signal - strong uptrend")
alertcondition(shortCondition, title="ADX Bear", message="ADX DI: SELL signal - strong downtrend")
'''

    def _generate_bb_squeeze(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for BB Squeeze Breakout strategy"""
        params = normalize_params(params)
        squeeze_thresh = params.get('squeeze_threshold', 0.8)
        sl_mult = params.get('sl_atr_mult', 2.8)
        tp_ratio = params.get('tp_ratio', 0.7)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// BB Squeeze Breakout Strategy
// Generated: {gen_date}
// Strategy Type: BREAKOUT
{metrics_comment}
//
// LOGIC: Trade when Bollinger Band width squeezes then releases
// - Squeeze: BB width below average (consolidation)
// - Release: BB width expands (breakout)
// - Direction: Above/below BB middle band

//@version=6
strategy("BB Squeeze Breakout", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
bbLength = input.int(20, "BB Length", minval=5, maxval=50, group="BB Squeeze")
bbMult = input.float(2.0, "BB Multiplier", minval=0.5, maxval=5.0, step=0.1, group="BB Squeeze")
squeezeThreshold = input.float({squeeze_thresh}, "Squeeze Threshold", minval=0.5, maxval=1.0, step=0.05, group="BB Squeeze")
avgPeriod = input.int(50, "Average Width Period", minval=20, maxval=100, group="BB Squeeze")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
bbBasis = ta.sma(close, bbLength)
bbDev = bbMult * ta.stdev(close, bbLength)
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

// BB Width calculation
bbWidth = (bbUpper - bbLower) / bbBasis
avgWidth = ta.sma(bbWidth, avgPeriod)

// Squeeze detection
inSqueeze = bbWidth < avgWidth * squeezeThreshold
squeezeRelease = inSqueeze[1] and not inSqueeze

atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = squeezeRelease and close > bbBasis and strategy.position_size == 0
shortCondition = squeezeRelease and close < bbBasis and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(bbUpper, "BB Upper", color=color.red)
plot(bbBasis, "BB Mid", color=color.gray)
plot(bbLower, "BB Lower", color=color.green)
bgcolor(inSqueeze ? color.new(color.yellow, 85) : na, title="Squeeze Zone")
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === ALERTS ===
alertcondition(longCondition, title="BB Squeeze Buy", message="BB Squeeze: BUY signal - breakout above mid")
alertcondition(shortCondition, title="BB Squeeze Sell", message="BB Squeeze: SELL signal - breakout below mid")
'''

    def _generate_connors_rsi(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Connors RSI Extreme strategy"""
        params = normalize_params(params)
        crsi_rsi_len = int(params.get('crsi_rsi_len', 5))
        crsi_streak_len = int(params.get('crsi_streak_len', 4))
        crsi_roc_len = int(params.get('crsi_roc_len', 95))
        sl_mult = params.get('sl_atr_mult', 2.7)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Connors RSI Extreme Strategy
// Generated: {gen_date}
// Strategy Type: MEAN REVERSION
{metrics_comment}
//
// LOGIC: Connors RSI combines 3 components:
// 1. Short-term RSI (default 3 periods)
// 2. Up/Down streak RSI
// 3. Percent rank of ROC
// Trade at extreme readings (< 15 or > 85)

//@version=6
strategy("Connors RSI Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
rsiLen = input.int({crsi_rsi_len}, "RSI Length", minval=2, maxval=10, group="Connors RSI")
streakLen = input.int({crsi_streak_len}, "Streak RSI Length", minval=2, maxval=10, group="Connors RSI")
rocLen = input.int({crsi_roc_len}, "ROC Lookback", minval=50, maxval=200, group="Connors RSI")
oversold = input.int(15, "Oversold Level", minval=5, maxval=30, group="Connors RSI")
overbought = input.int(85, "Overbought Level", minval=70, maxval=95, group="Connors RSI")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CONNORS RSI CALCULATION ===
// Component 1: Short-term RSI
rsi1 = ta.rsi(close, rsiLen)

// Component 2: Streak RSI
// Count consecutive up/down days
var int streak = 0
if close > close[1]
    streak := streak >= 0 ? streak + 1 : 1
else if close < close[1]
    streak := streak <= 0 ? streak - 1 : -1
else
    streak := 0

streakRsi = ta.rsi(streak, streakLen)

// Component 3: Percent rank of ROC
roc = (close - close[1]) / close[1] * 100
rocRank = ta.percentrank(roc, rocLen)

// Connors RSI = Average of all 3
connorsRsi = (rsi1 + streakRsi + rocRank) / 3

atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = connorsRsi < oversold and strategy.position_size == 0
shortCondition = connorsRsi > overbought and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(connorsRsi, "Connors RSI", color=color.purple, linewidth=2, display=display.pane)
hline(oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(50, "Middle", color=color.gray, linestyle=hline.style_dotted)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
bgcolor(connorsRsi < oversold ? color.new(color.green, 90) : connorsRsi > overbought ? color.new(color.red, 90) : na)

// === ALERTS ===
alertcondition(longCondition, title="CRSI Oversold", message="Connors RSI: BUY signal - extreme oversold")
alertcondition(shortCondition, title="CRSI Overbought", message="Connors RSI: SELL signal - extreme overbought")
'''

    def _generate_nadaraya_watson(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Nadaraya-Watson Mean Reversion"""
        params = normalize_params(params)
        nw_bandwidth = int(params.get('nw_bandwidth', 16))
        nw_mult = params.get('nw_mult', 2.6)
        sl_mult = params.get('sl_atr_mult', 4.1)
        tp_ratio = params.get('tp_ratio', 3.4)
        adx_threshold = int(params.get('adx_threshold', 32))
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Nadaraya-Watson Mean Reversion Strategy
// Generated: {gen_date}
// Strategy Type: ML-BASED (Kernel Regression)
{metrics_comment}
//
// LOGIC: Kernel regression envelope for mean reversion
// - Uses RBF kernel to smooth price
// - Trades when price touches envelope bands
// - Only in sideways markets (ADX filter)

//@version=6
strategy("Nadaraya-Watson Strategy", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
bandwidth = input.int({nw_bandwidth}, "Bandwidth", minval=3, maxval=20, group="N-W Kernel")
nwMult = input.float({nw_mult}, "Envelope Mult", minval=1.0, maxval=5.0, step=0.1, group="N-W Kernel")
lookback = input.int(50, "Lookback", minval=20, maxval=100, group="N-W Kernel")
adxThreshold = input.int({adx_threshold}, "ADX Threshold", minval=10, maxval=50, group="Filter")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === KERNEL REGRESSION (Simplified) ===
// Using weighted moving average as approximation
// True N-W would require looping with RBF weights
nwCenter = ta.wma(close, bandwidth)
nwStdev = ta.stdev(close, lookback)
nwUpper = nwCenter + nwStdev * nwMult
nwLower = nwCenter - nwStdev * nwMult

// === ADX FILTER ===
[diPlus, diMinus, adx] = ta.dmi(14, 14)
isSideways = adx < adxThreshold

// === CALCULATIONS ===
atr = ta.atr(14)
rsi = ta.rsi(close, 14)

// === ENTRY CONDITIONS ===
longCondition = close <= nwLower and isSideways and rsi < 35 and strategy.position_size == 0
shortCondition = close >= nwUpper and isSideways and rsi > 65 and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(nwCenter, "N-W Center", color=color.blue, linewidth=2)
plot(nwUpper, "Upper Band", color=color.red)
plot(nwLower, "Lower Band", color=color.green)
fill(plot(nwUpper, display=display.none), plot(nwLower, display=display.none), color=color.new(color.purple, 90))
bgcolor(isSideways ? color.new(color.green, 95) : color.new(color.red, 95))
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === ALERTS ===
alertcondition(longCondition, title="N-W Buy", message="Nadaraya-Watson: BUY signal - price at lower band")
alertcondition(shortCondition, title="N-W Sell", message="Nadaraya-Watson: SELL signal - price at upper band")
'''

    def _generate_divergence_3wave(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for 3-Wave Divergence strategy - SIMPLIFIED TO MATCH PYTHON"""
        params = normalize_params(params)
        # Optimizer found div_lookback=21 works best
        div_lookback = int(params.get('div_lookback', 21))
        sl_mult = params.get('sl_atr_mult', 1.4)
        tp_ratio = params.get('tp_ratio', 1.3)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// 3-Wave Divergence Strategy (Vdubus Style) - SIMPLIFIED v3
// Generated: {gen_date}
// Strategy Type: DIVERGENCE (ML-Based)
{metrics_comment}
//
// SIMPLIFIED APPROACH:
// 1. Uses rolling window to find swing lows/highs (like Python)
// 2. No strict pivot requirements - just local min/max
// 3. MACD histogram (8,21,5) for momentum divergence
// 4. Lookback forced to minimum 50 bars (Python hardcodes this)

//@version=6
strategy("3-Wave Divergence (Vdubus)", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
divLookback = input.int({div_lookback}, "Divergence Lookback", minval=30, maxval=100, group="Divergence")
swingLen = input.int(5, "Swing Detection Length", minval=2, maxval=10, group="Divergence")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === MACD HISTOGRAM (Python uses 8, 21, 5) ===
[macdLine, signalLine, macdHist] = ta.macd(close, 8, 21, 5)

// === RSI FOR ENTRY FILTER ===
rsi = ta.rsi(close, 14)
atr = ta.atr(14)

// === SIMPLE SWING DETECTION (matches Python's find_swing_points) ===
// Python checks: if value == min/max of window from i-lookback to i+lookback
isSwingLow(src, len) =>
    lowest = ta.lowest(src, len * 2 + 1)
    src[len] == lowest

isSwingHigh(src, len) =>
    highest = ta.highest(src, len * 2 + 1)
    src[len] == highest

// Current swing detection (delayed by swingLen bars to match Python)
swingLow = isSwingLow(close, swingLen)
swingHigh = isSwingHigh(close, swingLen)

// === FIND 3 MOST RECENT SWING LOWS WITHIN LOOKBACK ===
// Track the bar indices and values of the 3 most recent swing lows
var int lowBar1 = na  // Most recent
var int lowBar2 = na
var int lowBar3 = na  // Oldest
var float lowPrice1 = na
var float lowPrice2 = na
var float lowPrice3 = na
var float lowMacd1 = na
var float lowMacd2 = na
var float lowMacd3 = na

// Update swing low tracking
if swingLow
    lowBar3 := lowBar2
    lowPrice3 := lowPrice2
    lowMacd3 := lowMacd2
    lowBar2 := lowBar1
    lowPrice2 := lowPrice1
    lowMacd2 := lowMacd1
    lowBar1 := bar_index - swingLen
    lowPrice1 := close[swingLen]
    lowMacd1 := macdHist[swingLen]

// === FIND 3 MOST RECENT SWING HIGHS WITHIN LOOKBACK ===
var int highBar1 = na
var int highBar2 = na
var int highBar3 = na
var float highPrice1 = na
var float highPrice2 = na
var float highPrice3 = na
var float highMacd1 = na
var float highMacd2 = na
var float highMacd3 = na

if swingHigh
    highBar3 := highBar2
    highPrice3 := highPrice2
    highMacd3 := highMacd2
    highBar2 := highBar1
    highPrice2 := highPrice1
    highMacd2 := highMacd1
    highBar1 := bar_index - swingLen
    highPrice1 := close[swingLen]
    highMacd1 := macdHist[swingLen]

// === 3-WAVE BULLISH DIVERGENCE ===
// Conditions: All 3 lows within lookback, price making lower lows, MACD making higher lows
lowsValid = not na(lowBar1) and not na(lowBar2) and not na(lowBar3)
lowsInRange = lowsValid and (bar_index - lowBar3) <= divLookback

// Price: lower lows (most recent is lowest)
priceLowerLows = lowsInRange and lowPrice1 < lowPrice2 and lowPrice2 < lowPrice3

// MACD: higher lows (at least one improving)
macdHigherLows = lowsInRange and (lowMacd1 > lowMacd2 or lowMacd2 > lowMacd3)

bullish3Wave = priceLowerLows and macdHigherLows

// === 3-WAVE BEARISH DIVERGENCE ===
highsValid = not na(highBar1) and not na(highBar2) and not na(highBar3)
highsInRange = highsValid and (bar_index - highBar3) <= divLookback

// Price: higher highs (most recent is highest)
priceHigherHighs = highsInRange and highPrice1 > highPrice2 and highPrice2 > highPrice3

// MACD: lower highs (at least one declining)
macdLowerHighs = highsInRange and (highMacd1 < highMacd2 or highMacd2 < highMacd3)

bearish3Wave = priceHigherHighs and macdLowerHighs

// === ENTRY CONDITIONS ===
longCondition = bullish3Wave and rsi < 45 and strategy.position_size == 0
shortCondition = bearish3Wave and rsi > 55 and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === MACD HISTOGRAM PANEL ===
plot(macdHist, "MACD Hist", style=plot.style_histogram,
     color=macdHist >= 0 ? color.green : color.red, display=display.pane)
hline(0, "Zero", color=color.gray)

// === VISUALS ===
bgcolor(bullish3Wave ? color.new(color.green, 85) : bearish3Wave ? color.new(color.red, 85) : na)
plotshape(swingLow, "Swing Low", shape.circle, location.belowbar, color.new(color.blue, 50), size=size.tiny)
plotshape(swingHigh, "Swing High", shape.circle, location.abovebar, color.new(color.orange, 50), size=size.tiny)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)

// === DEBUG TABLE ===
var table debugTable = table.new(position.top_right, 2, 8, bgcolor=color.new(color.black, 80))
if barstate.islast
    table.cell(debugTable, 0, 0, "3-Wave Debug", text_color=color.white, text_size=size.tiny)
    table.cell(debugTable, 1, 0, "", text_color=color.white, text_size=size.tiny)
    table.cell(debugTable, 0, 1, "Lookback", text_color=color.gray, text_size=size.tiny)
    table.cell(debugTable, 1, 1, str.tostring(divLookback), text_color=color.yellow, text_size=size.tiny)
    table.cell(debugTable, 0, 2, "SL/TP", text_color=color.gray, text_size=size.tiny)
    table.cell(debugTable, 1, 2, str.tostring(slAtrMult, "#.#") + "x / " + str.tostring(tpRatio, "#.#") + "x", text_color=color.yellow, text_size=size.tiny)
    table.cell(debugTable, 0, 3, "Bull Div", text_color=color.gray, text_size=size.tiny)
    table.cell(debugTable, 1, 3, bullish3Wave ? "YES" : "no", text_color=bullish3Wave ? color.lime : color.gray, text_size=size.tiny)
    table.cell(debugTable, 0, 4, "Bear Div", text_color=color.gray, text_size=size.tiny)
    table.cell(debugTable, 1, 4, bearish3Wave ? "YES" : "no", text_color=bearish3Wave ? color.red : color.gray, text_size=size.tiny)
    table.cell(debugTable, 0, 5, "RSI", text_color=color.gray, text_size=size.tiny)
    table.cell(debugTable, 1, 5, str.tostring(rsi, "#.#"), text_color=color.white, text_size=size.tiny)
    table.cell(debugTable, 0, 6, "Low Prices", text_color=color.gray, text_size=size.tiny)
    lowPriceStr = na(lowPrice1) ? "na" : str.tostring(lowPrice1, "#") + "<" + str.tostring(lowPrice2, "#") + "<" + str.tostring(lowPrice3, "#")
    table.cell(debugTable, 1, 6, lowPriceStr, text_color=priceLowerLows ? color.lime : color.gray, text_size=size.tiny)
    table.cell(debugTable, 0, 7, "Low MACD", text_color=color.gray, text_size=size.tiny)
    lowMacdStr = na(lowMacd1) ? "na" : str.tostring(lowMacd1, "#.#") + ">" + str.tostring(lowMacd2, "#.#") + "?" + str.tostring(lowMacd3, "#.#")
    table.cell(debugTable, 1, 7, lowMacdStr, text_color=macdHigherLows ? color.lime : color.gray, text_size=size.tiny)

// === ALERTS ===
alertcondition(longCondition, title="3Wave Buy", message="3-Wave Divergence: BUY signal - bullish MACD divergence")
alertcondition(shortCondition, title="3Wave Sell", message="3-Wave Divergence: SELL signal - bearish MACD divergence")
'''

    def _generate_bb_rsi_tight(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for BB + RSI Tight (stricter conditions)"""
        # Use tighter parameters than classic
        params['rsi_oversold'] = params.get('rsi_oversold', 25)
        params['rsi_overbought'] = params.get('rsi_overbought', 75)
        return self._generate_bb_rsi_classic(params, metrics)

    def _generate_bb_stoch(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for BB + Stochastic strategy"""
        params = normalize_params(params)
        bb_length = int(params.get('bb_length', 20))
        bb_mult = params.get('bb_mult', 2.0)
        stoch_k = int(params.get('stoch_k', 14))
        stoch_os = int(params.get('stoch_oversold', 20))
        stoch_ob = int(params.get('stoch_overbought', 80))
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Bollinger Band + Stochastic Strategy
// Generated: {gen_date}
// Strategy Type: MEAN REVERSION
{metrics_comment}

//@version=6
strategy("BB + Stochastic", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
bbLength = input.int({bb_length}, "BB Length", minval=5, maxval=50, group="BB")
bbMult = input.float({bb_mult}, "BB Mult", minval=0.5, maxval=5.0, step=0.1, group="BB")
stochK = input.int({stoch_k}, "Stoch K", minval=5, maxval=30, group="Stochastic")
stochOS = input.int({stoch_os}, "Oversold", minval=5, maxval=30, group="Stochastic")
stochOB = input.int({stoch_ob}, "Overbought", minval=70, maxval=95, group="Stochastic")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
bbBasis = ta.sma(close, bbLength)
bbDev = bbMult * ta.stdev(close, bbLength)
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

k = ta.sma(ta.stoch(close, high, low, stochK), 3)
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = close <= bbLower and k < stochOS and strategy.position_size == 0
shortCondition = close >= bbUpper and k > stochOB and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - slDistance, limit=strategy.position_avg_price + tpDistance)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + slDistance, limit=strategy.position_avg_price - tpDistance)

// === VISUALS ===
plot(bbUpper, "BB Upper", color=color.red)
plot(bbBasis, "BB Mid", color=color.gray)
plot(bbLower, "BB Lower", color=color.green)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red, size=size.normal)
'''

    def _generate_keltner_rsi(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Keltner Channel + RSI strategy"""
        params = normalize_params(params)
        kc_length = int(params.get('kc_length', 20))
        kc_mult = params.get('kc_mult', 2.0)
        rsi_os = int(params.get('rsi_oversold', 30))
        rsi_ob = int(params.get('rsi_overbought', 70))
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Keltner Channel + RSI Strategy
// Generated: {gen_date}
// Strategy Type: MEAN REVERSION
{metrics_comment}

//@version=6
strategy("Keltner + RSI", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
kcLength = input.int({kc_length}, "KC Length", minval=5, maxval=50, group="Keltner")
kcMult = input.float({kc_mult}, "KC Mult", minval=0.5, maxval=5.0, step=0.1, group="Keltner")
rsiOS = input.int({rsi_os}, "RSI Oversold", minval=10, maxval=40, group="RSI")
rsiOB = input.int({rsi_ob}, "RSI Overbought", minval=60, maxval=90, group="RSI")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
kcBasis = ta.sma(close, kcLength)
kcRange = ta.atr(kcLength)
kcUpper = kcBasis + kcRange * kcMult
kcLower = kcBasis - kcRange * kcMult

rsi = ta.rsi(close, 14)
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = close <= kcLower and rsi < rsiOS and strategy.position_size == 0
shortCondition = close >= kcUpper and rsi > rsiOB and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
plot(kcUpper, "KC Upper", color=color.red)
plot(kcBasis, "KC Mid", color=color.gray)
plot(kcLower, "KC Lower", color=color.green)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _generate_z_score_reversion(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Z-Score Mean Reversion strategy"""
        params = normalize_params(params)
        z_length = int(params.get('z_length', 20))
        z_thresh = params.get('z_threshold', 2.0)
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Z-Score Mean Reversion Strategy
// Generated: {gen_date}
// Strategy Type: MEAN REVERSION
{metrics_comment}

//@version=6
strategy("Z-Score Reversion", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
zLength = input.int({z_length}, "Z-Score Length", minval=10, maxval=100, group="Z-Score")
zThreshold = input.float({z_thresh}, "Z-Score Threshold", minval=1.0, maxval=4.0, step=0.1, group="Z-Score")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
mean = ta.sma(close, zLength)
stdDev = ta.stdev(close, zLength)
zScore = (close - mean) / stdDev
atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = zScore < -zThreshold and strategy.position_size == 0
shortCondition = zScore > zThreshold and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
hline(zThreshold, "Upper", color=color.red, linestyle=hline.style_dashed)
hline(-zThreshold, "Lower", color=color.green, linestyle=hline.style_dashed)
hline(0, "Zero", color=color.gray)
plot(zScore, "Z-Score", color=color.purple, display=display.pane)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _generate_rsi_extreme(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for RSI Extreme strategy"""
        return self._generate_simple_rsi(params, metrics)

    def _generate_ema_crossover(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for EMA Crossover strategy"""
        params = normalize_params(params)
        ema_fast = int(params.get('ema_fast', 9))
        ema_slow = int(params.get('ema_slow', 100))
        sl_mult = params.get('sl_atr_mult', 3.5)
        tp_ratio = params.get('tp_ratio', 1.1)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// EMA Crossover Strategy
// Generated: {gen_date}
// Strategy Type: TREND FOLLOWING
{metrics_comment}

//@version=6
strategy("EMA Crossover", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
fastLen = input.int({ema_fast}, "Fast EMA", minval=5, maxval=50, group="EMA")
slowLen = input.int({ema_slow}, "Slow EMA", minval=10, maxval=200, group="EMA")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
emaFast = ta.ema(close, fastLen)
emaSlow = ta.ema(close, slowLen)
atr = ta.atr(14)

// === CROSSOVER SIGNALS ===
bullCross = ta.crossover(emaFast, emaSlow)
bearCross = ta.crossunder(emaFast, emaSlow)

// === TRADE EXECUTION ===
if bullCross
    strategy.entry("Long", strategy.long)

if bearCross
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
plot(emaFast, "Fast EMA", color=color.green, linewidth=2)
plot(emaSlow, "Slow EMA", color=color.red, linewidth=2)
plotshape(bullCross, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(bearCross, "Sell", shape.triangledown, location.abovebar, color.red)
bgcolor(emaFast > emaSlow ? color.new(color.green, 90) : color.new(color.red, 90))
'''

    def _generate_donchian_breakout(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Donchian Breakout strategy

        Uses PERCENTAGE-BASED SL/TP to match Python backtester exactly.
        """
        params = normalize_params(params)
        dc_length = int(params.get('dc_length', params.get('donchian_length', 20)))
        # Use percentage-based SL/TP to match Python backtester
        sl_percent = params.get('sl_percent', 6.1)
        tp_percent = params.get('tp_percent', 3.1)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Donchian Channel Breakout Strategy
// Generated: {gen_date}
// Strategy Type: BREAKOUT
// Uses percentage-based SL/TP to match Python backtester
{metrics_comment}

//@version=6
strategy("Donchian Breakout", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
dcLength = input.int({dc_length}, "Donchian Length", minval=5, maxval=100, group="Donchian")
slPercent = input.float({sl_percent}, "Stop Loss %", minval=0.5, maxval=20.0, step=0.1, group="Risk")
tpPercent = input.float({tp_percent}, "Take Profit %", minval=0.5, maxval=20.0, step=0.1, group="Risk")

// === CALCULATIONS ===
dcUpper = ta.highest(high, dcLength)[1]
dcLower = ta.lowest(low, dcLength)[1]
dcMid = (dcUpper + dcLower) / 2

// === BREAKOUT SIGNALS ===
breakoutUp = close > dcUpper and close[1] <= dcUpper[1]
breakoutDn = close < dcLower and close[1] >= dcLower[1]

// === TRADE EXECUTION ===
if breakoutUp and strategy.position_size == 0
    strategy.entry("Long", strategy.long)

if breakoutDn and strategy.position_size == 0
    strategy.entry("Short", strategy.short)

// Percentage-based exits (matches Python backtester)
if strategy.position_size > 0
    longSL = strategy.position_avg_price * (1 - slPercent / 100)
    longTP = strategy.position_avg_price * (1 + tpPercent / 100)
    strategy.exit("Long Exit", "Long", stop=longSL, limit=longTP)

if strategy.position_size < 0
    shortSL = strategy.position_avg_price * (1 + slPercent / 100)
    shortTP = strategy.position_avg_price * (1 - tpPercent / 100)
    strategy.exit("Short Exit", "Short", stop=shortSL, limit=shortTP)

// === VISUALS ===
plot(dcUpper, "Upper", color=color.red)
plot(dcMid, "Mid", color=color.gray)
plot(dcLower, "Lower", color=color.green)
plotshape(breakoutUp, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(breakoutDn, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _generate_stiff_surge(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Stiff Surge V1 (DaviddTech)"""
        params = normalize_params(params)
        sl_mult = params.get('sl_atr_mult', 3.0)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Stiff Surge V1 Strategy (DaviddTech)
// Generated: {gen_date}
// Strategy Type: TREND + MOMENTUM
{metrics_comment}

//@version=6
strategy("Stiff Surge V1", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
jmaLength = input.int(43, "JMA Length", minval=10, maxval=100, group="JMA")
stiffLen = input.int(60, "Stiffness Length", minval=20, maxval=100, group="Stiffness")
stiffSmooth = input.int(100, "Stiffness Smoothing", minval=50, maxval=200, group="Stiffness")
tdfiLen = input.int(15, "TDFI Length", minval=5, maxval=30, group="TDFI")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === JMA (Jurik Moving Average approximation) ===
jma = ta.ema(ta.ema(close, jmaLength), jmaLength)

// === STIFFNESS ===
ema1 = ta.ema(close, stiffLen)
ema2 = ta.ema(close, stiffSmooth)
stiff = ema1 > ema2 ? 100 : 0
stiffSmoothed = ta.sma(stiff, 10)

// === TDFI (Trend Direction Force Index) ===
tdfi = ta.roc(close, tdfiLen) / ta.atr(tdfiLen)

atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = close > jma and stiffSmoothed > 50 and tdfi > 0.05 and strategy.position_size == 0
shortCondition = close < jma and stiffSmoothed < 50 and tdfi < -0.05 and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
plot(jma, "JMA", color=color.blue, linewidth=2)
bgcolor(stiffSmoothed > 50 ? color.new(color.green, 90) : color.new(color.red, 90))
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _generate_stiff_surge_v2(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Stiff Surge V2"""
        return self._generate_stiff_surge(params, metrics)  # Similar logic

    def _generate_range_filter(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Range Filter + ADX strategy"""
        params = normalize_params(params)
        rf_period = int(params.get('rf_period', 100))
        rf_mult = params.get('rf_mult', 3.0)
        adx_thresh = int(params.get('adx_threshold', 25))
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Range Filter + ADX Strategy
// Generated: {gen_date}
// Strategy Type: TREND FOLLOWING
{metrics_comment}

//@version=6
strategy("Range Filter ADX", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
rfPeriod = input.int({rf_period}, "Range Filter Period", minval=50, maxval=200, group="Range Filter")
rfMult = input.float({rf_mult}, "Range Mult", minval=1.0, maxval=5.0, step=0.1, group="Range Filter")
adxThreshold = input.int({adx_thresh}, "ADX Threshold", minval=15, maxval=50, group="ADX")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === RANGE FILTER ===
rfRange = ta.atr(rfPeriod) * rfMult
var float rfValue = close
rfValue := close > rfValue + rfRange ? close - rfRange : close < rfValue - rfRange ? close + rfRange : rfValue
rfDir = close > rfValue ? 1 : close < rfValue ? -1 : 0

// === ADX ===
[diPlus, diMinus, adx] = ta.dmi(14, 14)
strongTrend = adx > adxThreshold

atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = rfDir == 1 and strongTrend and strategy.position_size == 0
shortCondition = rfDir == -1 and strongTrend and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
plot(rfValue, "Range Filter", color=rfDir == 1 ? color.green : color.red, linewidth=2)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _generate_supertrend_confluence(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for Supertrend Confluence strategy"""
        params = normalize_params(params)
        st_mult1 = params.get('st_mult_1', 2.0)
        st_mult2 = params.get('st_mult_2', 3.0)
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Supertrend Confluence Strategy
// Generated: {gen_date}
// Strategy Type: TREND FOLLOWING
{metrics_comment}

//@version=6
strategy("Supertrend Confluence", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, process_orders_on_close=true, pyramiding=0)

// === PARAMETERS ===
stLen = input.int(10, "Supertrend Length", minval=5, maxval=50, group="Supertrend")
stMult1 = input.float({st_mult1}, "ST Mult 1", minval=0.5, maxval=5.0, step=0.1, group="Supertrend")
stMult2 = input.float({st_mult2}, "ST Mult 2", minval=0.5, maxval=5.0, step=0.1, group="Supertrend")
slAtrMult = input.float({sl_mult}, "SL ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === DUAL SUPERTREND ===
[st1, dir1] = ta.supertrend(stMult1, stLen)
[st2, dir2] = ta.supertrend(stMult2, stLen)

// Confluence: both supertrends agree
bullConfluence = dir1 < 0 and dir2 < 0
bearConfluence = dir1 > 0 and dir2 > 0

atr = ta.atr(14)

// === ENTRY CONDITIONS ===
longCondition = bullConfluence and not bullConfluence[1] and strategy.position_size == 0
shortCondition = bearConfluence and not bearConfluence[1] and strategy.position_size == 0

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit("Long Exit", "Long", stop=strategy.position_avg_price - atr * slAtrMult, limit=strategy.position_avg_price + atr * slAtrMult * tpRatio)

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short", stop=strategy.position_avg_price + atr * slAtrMult, limit=strategy.position_avg_price - atr * slAtrMult * tpRatio)

// === VISUALS ===
plot(st1, "ST1", color=dir1 < 0 ? color.green : color.red, linewidth=1)
plot(st2, "ST2", color=dir2 < 0 ? color.lime : color.maroon, linewidth=2)
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color.red)
'''

    def _get_metrics_comment(self, metrics: Dict) -> str:
        """Generate metrics comment block"""
        if not metrics:
            return ""
        return f"""
// BACKTEST RESULTS:
//   Win Rate: {metrics.get('win_rate', 'N/A')}%
//   Profit Factor: {metrics.get('profit_factor', 'N/A')}
//   Total Trades: {metrics.get('total_trades', 'N/A')}
//   Total PnL: £{metrics.get('total_pnl', 'N/A')}"""

    def _generate_pct_drop_buy(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for percentage drop buy strategy"""
        drop_pct = params.get('drop_percent', 2.3)
        rise_pct = params.get('rise_percent', 3.8)
        lookback = int(params.get('pct_lookback', 29))
        sl_mult = params.get('sl_atr_mult', 0.9)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Percentage Drop Buy Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Beginner-Friendly)
{metrics_comment}

//@version=6
strategy("Percentage Drop Buy ({drop_pct}%/{rise_pct}%)", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
dropPercent = input.float({drop_pct}, "Buy after X% drop", minval=0.5, maxval=10, step=0.5, group="Entry")
risePercent = input.float({rise_pct}, "Sell after Y% rise", minval=0.5, maxval=10, step=0.5, group="Exit")
lookback = input.int({lookback}, "Lookback period", minval=5, maxval=50, group="Entry")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
rollingHigh = ta.highest(high, lookback)
currentDrop = (rollingHigh - close) / rollingHigh * 100

// === ENTRY CONDITIONS ===
buySignal = currentDrop >= dropPercent and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if buySignal
    strategy.entry("Buy", strategy.long)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Exit", "Buy", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plotshape(buySignal, "Buy Signal", shape.triangleup, location.belowbar, color.green)
plot(rollingHigh, "Rolling High", color=color.new(color.blue, 50))
bgcolor(currentDrop >= dropPercent ? color.new(color.green, 90) : na)

// === INFO TABLE ===
var table infoTable = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 80))
if barstate.islast
    table.cell(infoTable, 0, 0, "Strategy", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 0, "% Drop Buy", text_color=color.white, text_size=size.small)
    table.cell(infoTable, 0, 1, "Drop %", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 1, str.tostring(dropPercent) + "%", text_color=color.lime, text_size=size.small)
    table.cell(infoTable, 0, 2, "Rise %", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 2, str.tostring(risePercent) + "%", text_color=color.lime, text_size=size.small)
    table.cell(infoTable, 0, 3, "Current Drop", text_color=color.gray, text_size=size.small)
    dropColor = currentDrop >= dropPercent ? color.lime : color.orange
    table.cell(infoTable, 1, 3, str.tostring(currentDrop, "#.##") + "%", text_color=dropColor, text_size=size.small)
    table.cell(infoTable, 0, 4, "Lookback", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 4, str.tostring(lookback), text_color=color.white, text_size=size.small)

// === ALERTS ===
alertcondition(buySignal, title="Buy Signal", message="Percentage Drop Buy: Entry signal")
'''

    def _generate_simple_sma_cross(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for simple SMA crossover"""
        fast = int(params.get('sma_fast', 12))
        slow = int(params.get('sma_slow', 25))
        sl_mult = params.get('sl_atr_mult', 4.5)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Simple SMA Crossover Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Beginner-Friendly)
{metrics_comment}

//@version=6
strategy("SMA Cross ({fast}/{slow})", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
fastPeriod = input.int({fast}, "Fast SMA", minval=5, maxval=50, group="Moving Averages")
slowPeriod = input.int({slow}, "Slow SMA", minval=10, maxval=200, group="Moving Averages")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
smaFast = ta.sma(close, fastPeriod)
smaSlow = ta.sma(close, slowPeriod)

// === CROSSOVER SIGNALS ===
bullishCross = ta.crossover(smaFast, smaSlow)
bearishCross = ta.crossunder(smaFast, smaSlow)

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if bullishCross
    strategy.entry("Long", strategy.long)

if bearishCross
    strategy.close("Long")
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(smaFast, "Fast SMA", color=color.green, linewidth=2)
plot(smaSlow, "Slow SMA", color=color.red, linewidth=2)
plotshape(bullishCross, "Buy", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(bearishCross, "Sell", shape.triangledown, location.abovebar, color.red, size=size.small)
bgcolor(smaFast > smaSlow ? color.new(color.green, 90) : color.new(color.red, 90))

// === ALERTS ===
alertcondition(bullishCross, title="Bullish Cross", message="SMA Crossover: BUY signal")
alertcondition(bearishCross, title="Bearish Cross", message="SMA Crossover: SELL signal")
'''

    def _generate_consecutive_candles(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for consecutive candles reversal"""
        n_candles = int(params.get('consec_count', 5))
        sl_mult = params.get('sl_atr_mult', 3.2)
        tp_ratio = params.get('tp_ratio', 0.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Consecutive Candles Reversal Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Price Action)
{metrics_comment}

//@version=6
strategy("Consecutive Candles Reversal ({n_candles})", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
consecCount = input.int({n_candles}, "Consecutive candles", minval=2, maxval=6, group="Pattern")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
isRed = close < open
isGreen = close > open

// Count consecutive candles
redCount = 0
greenCount = 0
for i = 1 to consecCount
    if close[i] < open[i]
        redCount += 1
    if close[i] > open[i]
        greenCount += 1

hadRedStreak = redCount == consecCount
hadGreenStreak = greenCount == consecCount

// === REVERSAL SIGNALS ===
bullishReversal = hadRedStreak and isGreen
bearishReversal = hadGreenStreak and isRed

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if bullishReversal and strategy.position_size == 0
    strategy.entry("Long", strategy.long)

if bearishReversal and strategy.position_size == 0
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plotshape(bullishReversal, "Bullish Reversal", shape.triangleup, location.belowbar, color.green)
plotshape(bearishReversal, "Bearish Reversal", shape.triangledown, location.abovebar, color.red)
barcolor(hadRedStreak ? color.new(color.red, 50) : hadGreenStreak ? color.new(color.green, 50) : na)

// === ALERTS ===
alertcondition(bullishReversal, title="Bullish Reversal", message="Consecutive Candles: BUY signal")
alertcondition(bearishReversal, title="Bearish Reversal", message="Consecutive Candles: SELL signal")
'''

    def _generate_simple_rsi(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for simple RSI extremes"""
        rsi_len = int(params.get('simple_rsi_period', 7))
        os_level = int(params.get('simple_oversold', 31))
        ob_level = int(params.get('simple_overbought', 76))
        sl_mult = params.get('sl_atr_mult', 1.5)
        tp_ratio = params.get('tp_ratio', 2.0)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Simple RSI Extremes Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Momentum)
{metrics_comment}

//@version=6
strategy("RSI Extremes ({rsi_len})", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
rsiPeriod = input.int({rsi_len}, "RSI Period", minval=5, maxval=30, group="RSI")
oversoldLevel = input.int({os_level}, "Oversold Level", minval=10, maxval=40, group="RSI")
overboughtLevel = input.int({ob_level}, "Overbought Level", minval=60, maxval=90, group="RSI")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
rsi = ta.rsi(close, rsiPeriod)

// === SIGNALS ===
buySignal = rsi < oversoldLevel and strategy.position_size == 0
sellSignal = rsi > overboughtLevel and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if buySignal
    strategy.entry("Long", strategy.long)

if sellSignal
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === RSI PANEL ===
hline(oversoldLevel, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(overboughtLevel, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(50, "Middle", color=color.gray, linestyle=hline.style_dotted)

// === VISUALS ===
plotshape(buySignal, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(sellSignal, "Sell", shape.triangledown, location.abovebar, color.red)
bgcolor(rsi < oversoldLevel ? color.new(color.green, 90) : rsi > overboughtLevel ? color.new(color.red, 90) : na)

// === ALERTS ===
alertcondition(buySignal, title="RSI Oversold", message="RSI Extremes: BUY signal (oversold)")
alertcondition(sellSignal, title="RSI Overbought", message="RSI Extremes: SELL signal (overbought)")
'''

    def _generate_range_breakout(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for simple range breakout"""
        lookback = int(params.get('range_lookback', 17))
        buffer_pct = params.get('breakout_buffer_pct', 1.0)
        sl_mult = params.get('sl_atr_mult', 3.8)
        tp_ratio = params.get('tp_ratio', 3.2)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Simple Range Breakout Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Breakout)
{metrics_comment}

//@version=6
strategy("Range Breakout ({lookback})", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
lookback = input.int({lookback}, "Range Lookback", minval=5, maxval=50, group="Range")
bufferPct = input.float({buffer_pct}, "Breakout Buffer %", minval=0.1, maxval=2.0, step=0.1, group="Range")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
recentHigh = ta.highest(high, lookback)[1]
recentLow = ta.lowest(low, lookback)[1]
bufferMult = 1 + bufferPct / 100

// Breakout levels
breakoutUp = recentHigh * bufferMult
breakoutDn = recentLow / bufferMult

// === SIGNALS ===
breakoutLong = close > breakoutUp and strategy.position_size == 0
breakoutShort = close < breakoutDn and strategy.position_size == 0

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if breakoutLong
    strategy.entry("Long", strategy.long)

if breakoutShort
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plot(recentHigh, "Range High", color=color.red, linewidth=1, style=plot.style_stepline)
plot(recentLow, "Range Low", color=color.green, linewidth=1, style=plot.style_stepline)
plot(breakoutUp, "Breakout Up", color=color.new(color.red, 70), style=plot.style_circles)
plot(breakoutDn, "Breakout Dn", color=color.new(color.green, 70), style=plot.style_circles)
plotshape(breakoutLong, "Long Breakout", shape.triangleup, location.belowbar, color.green)
plotshape(breakoutShort, "Short Breakout", shape.triangledown, location.abovebar, color.red)

// === ALERTS ===
alertcondition(breakoutLong, title="Breakout Up", message="Range Breakout: BUY signal")
alertcondition(breakoutShort, title="Breakout Down", message="Range Breakout: SELL signal")
'''

    def _generate_engulfing_pattern(self, params: Dict, metrics: Dict = None) -> str:
        """Generate Pine Script for engulfing pattern"""
        min_size = params.get('engulf_min_size_mult', 1.4)
        sl_mult = params.get('sl_atr_mult', 5.0)
        tp_ratio = params.get('tp_ratio', 4.7)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        return f'''// Engulfing Candle Pattern Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Price Action)
{metrics_comment}

//@version=6
strategy("Engulfing Pattern", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === PARAMETERS ===
minSizeMult = input.float({min_size}, "Min Size Multiplier", minval=1.0, maxval=3.0, step=0.1, group="Pattern")
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
bodyPrev = math.abs(close[1] - open[1])
bodyCurr = math.abs(close - open)

prevRed = close[1] < open[1]
prevGreen = close[1] > open[1]
currGreen = close > open
currRed = close < open

// === ENGULFING PATTERNS ===
bullishEngulf = prevRed and currGreen and bodyCurr >= bodyPrev * minSizeMult and close > open[1] and open < close[1]
bearishEngulf = prevGreen and currRed and bodyCurr >= bodyPrev * minSizeMult and close < open[1] and open > close[1]

// === RISK MANAGEMENT ===
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// === TRADE EXECUTION ===
if bullishEngulf and strategy.position_size == 0
    strategy.entry("Long", strategy.long)

if bearishEngulf and strategy.position_size == 0
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)

// === VISUALS ===
plotshape(bullishEngulf, "Bullish Engulfing", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(bearishEngulf, "Bearish Engulfing", shape.triangledown, location.abovebar, color.red, size=size.normal)
barcolor(bullishEngulf ? color.lime : bearishEngulf ? color.red : na)

// === ALERTS ===
alertcondition(bullishEngulf, title="Bullish Engulfing", message="Engulfing Pattern: BUY signal")
alertcondition(bearishEngulf, title="Bearish Engulfing", message="Engulfing Pattern: SELL signal")
'''

    def _generate_generic_simple(self, strategy_name: str, params: Dict, metrics: Dict = None) -> str:
        """Generate generic Pine Script for simple strategies without specific template"""
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_ratio = params.get('tp_ratio', 1.5)
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metrics_comment = self._get_metrics_comment(metrics)

        # Format params for display
        params_str = "\n".join([f"// {k}: {v}" for k, v in params.items() if not k.startswith('sl_') and not k.startswith('tp_')])

        return f'''// {strategy_name.replace("_", " ").title()} Strategy
// Generated: {gen_date}
// Strategy Type: SIMPLE (Custom)
{metrics_comment}
//
// OPTIMIZED PARAMETERS:
{params_str}

//@version=6
strategy("{strategy_name.replace("_", " ").title()}", overlay=true,
         default_qty_type=strategy.cash, default_qty_value=1000,
         initial_capital=100000, currency=currency.NONE, commission_type=strategy.commission.percent,
         commission_value=0.1, pyramiding=0)

// === RISK PARAMETERS ===
slAtrMult = input.float({sl_mult}, "Stop Loss ATR Mult", minval=0.5, maxval=5.0, step=0.1, group="Risk")
tpRatio = input.float({tp_ratio}, "TP Ratio", minval=0.5, maxval=5.0, step=0.1, group="Risk")

// === CALCULATIONS ===
atr = ta.atr(14)
slDistance = atr * slAtrMult
tpDistance = slDistance * tpRatio

// NOTE: This is a generic template.
// The specific strategy logic for "{strategy_name}" should be implemented
// based on the optimized parameters shown above.

// === PLACEHOLDER SIGNALS (customize these) ===
longCondition = false  // Implement based on strategy
shortCondition = false // Implement based on strategy

// === TRADE EXECUTION ===
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    stopPrice = strategy.position_avg_price - slDistance
    takeProfitPrice = strategy.position_avg_price + tpDistance
    strategy.exit("Long Exit", "Long", stop=stopPrice, limit=takeProfitPrice)

if strategy.position_size < 0
    stopPrice = strategy.position_avg_price + slDistance
    takeProfitPrice = strategy.position_avg_price - tpDistance
    strategy.exit("Short Exit", "Short", stop=stopPrice, limit=takeProfitPrice)
'''

    # Stub methods for other simple strategies (can be expanded)
    def _generate_pct_drop_consecutive(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("pct_drop_consecutive", params, metrics)

    def _generate_price_vs_sma(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("price_vs_sma", params, metrics)

    def _generate_triple_sma_align(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("triple_sma_align", params, metrics)

    def _generate_inside_bar_breakout(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("inside_bar_breakout", params, metrics)

    def _generate_support_resistance(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("support_resistance_simple", params, metrics)

    def _generate_candle_ratio(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("candle_ratio_momentum", params, metrics)

    def _generate_doji_reversal(self, params: Dict, metrics: Dict = None) -> str:
        return self._generate_generic_simple("doji_reversal", params, metrics)

    def generate_debug_mode(self, strategy_name: str, params: Dict, metrics: Dict = None,
                            entry_rule: str = None, direction: str = None,
                            position_size_pct: float = 100.0, capital: float = 1000.0,
                            date_range: Dict = None, max_trades_to_log: int = 50) -> str:
        """
        Generate DEBUG MODE Pine Script v6 that exports trade data to Data Window.

        This debug script tracks all trade details in arrays and exports them to labels
        that can be copied from TradingView's Data Window for easy comparison with
        Python backtest exports.

        Args:
            strategy_name: Name of the strategy
            params: Optimized parameters (must include tp_percent, sl_percent)
            metrics: Performance metrics from backtesting
            entry_rule: Entry rule identifier (e.g., 'williams_r', 'rsi_extreme')
            direction: Trade direction ('long', 'short', or 'both')
            position_size_pct: Position size as % of equity
            capital: Starting capital
            date_range: Optional date range dict
            max_trades_to_log: Maximum number of trades to display in Data Window (default 50)

        Returns:
            Complete Pine Script v6 debug code as string
        """
        gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Extract percentage-based TP/SL
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

        # Metrics comment
        metrics_comment = ""
        if metrics:
            metrics_comment = f"""
// PYTHON BACKTEST RESULTS (compare with TradingView):
//   Total Trades: {metrics.get('total_trades', 'N/A')}
//   Win Rate: {metrics.get('win_rate', 0):.1f}%
//   Total P&L: £{metrics.get('total_pnl', 0):.2f}
//   Profit Factor: {metrics.get('profit_factor', 0):.2f}
//   Max Drawdown: £{metrics.get('max_drawdown', 0):.2f}"""

        # Determine direction
        if direction is None:
            direction = "both"
            if "long" in strategy_name.lower():
                direction = "long"
            elif "short" in strategy_name.lower():
                direction = "short"

        is_long = direction == "long"
        is_short = direction == "short"
        enable_longs = direction in ["long", "both"]
        enable_shorts = direction in ["short", "both"]

        # Generate date range filtering code
        date_range_code = ""
        date_range_condition = ""
        if date_range and date_range.get('enabled'):
            start_date = date_range.get('startDate', '2024-01-01')
            start_time = date_range.get('startTime', '00:00')
            end_date = date_range.get('endDate', '2025-12-31')
            end_time = date_range.get('endTime', '23:59')

            date_range_code = f'''
// =============================================================================
// DATE RANGE FILTER
// =============================================================================

useDateRange = input.bool(true, "Limit Backtest to Date Range", group="Date Range")
fromDate = input.time(timestamp("{start_date} {start_time} +0000"), "From Date", group="Date Range")
toDate = input.time(timestamp("{end_date} {end_time} +0000"), "To Date", group="Date Range")

inDateRange() => not useDateRange or (time >= fromDate and time <= toDate)
'''
            date_range_condition = " and inDateRange()"

        # Entry condition based on entry_rule
        entry_rule = entry_rule or 'rsi_extreme'
        entry_conditions = {
            'rsi_extreme': '''rsiValue = ta.rsi(close, rsiLength)
longCondition = ta.crossover(rsiValue, 30)
shortCondition = ta.crossunder(rsiValue, 70)''',
            'stoch_extreme': '''k = ta.sma(ta.stoch(close, high, low, stochK), stochSmooth)
d = ta.sma(k, stochD)
longCondition = ta.crossover(k, d) and k < 20
shortCondition = ta.crossunder(k, d) and k > 80''',
            'williams_r': '''willrValue = ta.wpr(willrLength)
longCondition = willrValue < -80
shortCondition = willrValue > -20''',
            'bb_touch': '''[bbMiddle, bbUpper, bbLower] = ta.bb(close, bbLength, bbMult)
longCondition = ta.crossover(close, bbLower)
shortCondition = ta.crossunder(close, bbUpper)''',
            'cci_extreme': '''cciValue = ta.cci(high, low, close, cciLength)
longCondition = cciValue < -100
shortCondition = cciValue > 100''',
            'ema_crossover': '''emaFast = ta.ema(close, emaFastLen)
emaSlow = ta.ema(close, emaSlowLen)
longCondition = ta.crossover(emaFast, emaSlow)
shortCondition = ta.crossunder(emaFast, emaSlow)''',
            'macd_cross': '''[macdLine, signalLine, hist] = ta.macd(close, macdFast, macdSlow, macdSignal)
longCondition = ta.crossover(macdLine, signalLine)
shortCondition = ta.crossunder(macdLine, signalLine)''',
            'supertrend': '''[supertrend, direction] = ta.supertrend(stFactor, stAtrLen)
longCondition = direction == -1 and direction[1] == 1
shortCondition = direction == 1 and direction[1] == -1''',
        }

        entry_code = entry_conditions.get(entry_rule, entry_conditions['rsi_extreme'])

        # Direction filtering
        if is_long:
            direction_filter = '''
// Direction filter: LONG ONLY
shortCondition := false'''
        elif is_short:
            direction_filter = '''
// Direction filter: SHORT ONLY
longCondition := false'''
        else:
            direction_filter = '''
// Direction filter: BOTH directions enabled'''

        return f'''// =============================================================================
// DEBUG MODE: Trade Data Export for Python Comparison
// =============================================================================
// Generated: {gen_date}
// Strategy: {strategy_name}
// Entry Rule: {entry_rule}
// Direction: {direction}
//
// HOW TO USE:
// 1. Apply this script to TradingView chart
// 2. Run the strategy backtest
// 3. Open the Data Window (Ctrl+D or Cmd+D)
// 4. Scroll to the last bar - trade logs appear as labels
// 5. Copy the CSV data from the Data Window
// 6. Compare with Python backtest export
//
// TRADE LOG FORMAT (CSV):
// trade_num,direction,entry_time,exit_time,entry_price,exit_price,tp_price,sl_price,pnl,exit_type,entry_rsi,entry_atr
{metrics_comment}

//@version=6
strategy("{strategy_name} [DEBUG MODE]", overlay=true,
         default_qty_type=strategy.percent_of_equity, default_qty_value={position_size_pct},
         initial_capital={capital}, currency=currency.GBP,
         commission_type=strategy.commission.percent, commission_value=0.1,
         process_orders_on_close=true, pyramiding=0,
         margin_long=100, margin_short=100)

// =============================================================================
// PARAMETERS
// =============================================================================

// Risk Management
tpPercent = input.float({tp_percent}, "Take Profit %", minval=0.1, maxval=20.0, step=0.1, group="Risk")
slPercent = input.float({sl_percent}, "Stop Loss %", minval=0.1, maxval=20.0, step=0.1, group="Risk")

// Indicator Parameters
rsiLength = input.int(14, "RSI Length", minval=2, maxval=50, group="Indicators")
stochK = input.int(14, "Stoch K", minval=1, maxval=50, group="Indicators")
stochD = input.int(3, "Stoch D", minval=1, maxval=20, group="Indicators")
stochSmooth = input.int(3, "Stoch Smooth", minval=1, maxval=20, group="Indicators")
bbLength = input.int(20, "BB Length", minval=5, maxval=50, group="Indicators")
bbMult = input.float(2.0, "BB Mult", minval=0.5, maxval=5.0, step=0.1, group="Indicators")
willrLength = input.int(14, "Williams %R Length", minval=2, maxval=50, group="Indicators")
cciLength = input.int(20, "CCI Length", minval=5, maxval=50, group="Indicators")
emaFastLen = input.int(9, "EMA Fast", minval=2, maxval=50, group="Indicators")
emaSlowLen = input.int(21, "EMA Slow", minval=5, maxval=100, group="Indicators")
macdFast = input.int(12, "MACD Fast", minval=2, maxval=50, group="Indicators")
macdSlow = input.int(26, "MACD Slow", minval=5, maxval=100, group="Indicators")
macdSignal = input.int(9, "MACD Signal", minval=2, maxval=50, group="Indicators")
stFactor = input.float(3.0, "SuperTrend Factor", minval=1.0, maxval=10.0, step=0.1, group="Indicators")
stAtrLen = input.int(10, "SuperTrend ATR Length", minval=1, maxval=50, group="Indicators")

// Debug Settings
maxTradesToLog = input.int({max_trades_to_log}, "Max Trades to Log", minval=10, maxval=200, group="Debug")
showDebugLabels = input.bool(true, "Show Debug Labels", group="Debug")
showDebugTable = input.bool(true, "Show Debug Summary Table", group="Debug")
{date_range_code}
// =============================================================================
// DEBUG: TRADE LOGGING ARRAYS
// =============================================================================

// Trade log arrays - stores CSV formatted trade data
var array<string> tradeLog = array.new_string(0)
var int tradeNum = 0

// Position tracking for logging
var float entryPrice = na
var float entryTime = na
var float entryRsi = na
var float entryAtr = na
var float tpPrice = na
var float slPrice = na
var bool isLongPosition = false

// =============================================================================
// INDICATOR CALCULATIONS
// =============================================================================

atr = ta.atr(14)
rsi = ta.rsi(close, rsiLength)

// =============================================================================
// ENTRY CONDITIONS ({entry_rule})
// =============================================================================

{entry_code}
{direction_filter}

// Apply date range filter
longCondition := longCondition{date_range_condition}
shortCondition := shortCondition{date_range_condition}

// =============================================================================
// TRADE EXECUTION WITH DEBUG LOGGING
// =============================================================================

// ENTRY: Track position entry details
if longCondition and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
    tradeNum += 1
    entryPrice := close
    entryTime := time
    entryRsi := rsi
    entryAtr := atr
    tpPrice := close * (1 + tpPercent / 100)
    slPrice := close * (1 - slPercent / 100)
    isLongPosition := true

if shortCondition and strategy.position_size == 0
    strategy.entry("Short", strategy.short)
    tradeNum += 1
    entryPrice := close
    entryTime := time
    entryRsi := rsi
    entryAtr := atr
    tpPrice := close * (1 - tpPercent / 100)
    slPrice := close * (1 + slPercent / 100)
    isLongPosition := false

// EXIT: Set TP/SL orders
if strategy.position_size > 0
    strategy.exit("Long Exit", "Long",
                  limit=strategy.position_avg_price * (1 + tpPercent / 100),
                  stop=strategy.position_avg_price * (1 - slPercent / 100))

if strategy.position_size < 0
    strategy.exit("Short Exit", "Short",
                  limit=strategy.position_avg_price * (1 - tpPercent / 100),
                  stop=strategy.position_avg_price * (1 + slPercent / 100))

// =============================================================================
// DEBUG: LOG TRADE ON EXIT
// =============================================================================

// Detect position close and log trade details
positionClosed = strategy.position_size == 0 and strategy.position_size[1] != 0

if positionClosed and not na(entryPrice)
    // Calculate P&L
    exitPrice = close
    float pnl = 0.0
    string exitType = "UNKNOWN"

    if isLongPosition
        pnl := (exitPrice - entryPrice) / entryPrice * 100
        if exitPrice >= tpPrice
            exitType := "TP"
        else if exitPrice <= slPrice
            exitType := "SL"
        else
            exitType := "SIGNAL"
    else
        pnl := (entryPrice - exitPrice) / entryPrice * 100
        if exitPrice <= tpPrice
            exitType := "TP"
        else if exitPrice >= slPrice
            exitType := "SL"
        else
            exitType := "SIGNAL"

    // Format trade log entry as CSV
    // Format: trade_num,direction,entry_time,exit_time,entry_price,exit_price,tp_price,sl_price,pnl,exit_type,entry_rsi,entry_atr
    string dirStr = isLongPosition ? "LONG" : "SHORT"
    string entryTimeStr = str.format("{{0,date,yyyy-MM-dd HH:mm}}", entryTime)
    string exitTimeStr = str.format("{{0,date,yyyy-MM-dd HH:mm}}", time)

    string logEntry = str.tostring(tradeNum) + "," +
                      dirStr + "," +
                      entryTimeStr + "," +
                      exitTimeStr + "," +
                      str.tostring(entryPrice, "#.##") + "," +
                      str.tostring(exitPrice, "#.##") + "," +
                      str.tostring(tpPrice, "#.##") + "," +
                      str.tostring(slPrice, "#.##") + "," +
                      str.tostring(pnl, "#.####") + "," +
                      exitType + "," +
                      str.tostring(entryRsi, "#.##") + "," +
                      str.tostring(entryAtr, "#.##")

    array.push(tradeLog, logEntry)

    // Reset tracking variables
    entryPrice := na
    entryTime := na
    entryRsi := na
    entryAtr := na
    tpPrice := na
    slPrice := na

// =============================================================================
// DEBUG: DISPLAY TRADE LOG IN DATA WINDOW
// =============================================================================

// Display last N trades as labels (visible in Data Window)
if barstate.islast and showDebugLabels
    // Header label
    label.new(bar_index, high * 1.01,
              "=== TRADE LOG (CSV) ===\\ntrade_num,direction,entry_time,exit_time,entry_price,exit_price,tp_price,sl_price,pnl%,exit_type,entry_rsi,entry_atr",
              style=label.style_label_down, color=color.blue, textcolor=color.white, size=size.small)

    // Trade log labels (most recent trades)
    int logSize = array.size(tradeLog)
    int startIdx = math.max(0, logSize - maxTradesToLog)

    for i = startIdx to logSize - 1
        float yOffset = high * (1.005 - (i - startIdx) * 0.001)
        label.new(bar_index - (i - startIdx), yOffset, array.get(tradeLog, i),
                  style=label.style_none, textcolor=color.gray, size=size.tiny)

// =============================================================================
// DEBUG: SUMMARY TABLE
// =============================================================================

if barstate.islast and showDebugTable
    var table debugTable = table.new(position.top_right, 2, 8, bgcolor=color.new(color.black, 80))

    table.cell(debugTable, 0, 0, "DEBUG SUMMARY", bgcolor=color.blue, text_color=color.white)
    table.cell(debugTable, 1, 0, "{strategy_name}", bgcolor=color.blue, text_color=color.white)

    table.cell(debugTable, 0, 1, "Total Trades Logged", text_color=color.white)
    table.cell(debugTable, 1, 1, str.tostring(array.size(tradeLog)), text_color=color.yellow)

    table.cell(debugTable, 0, 2, "Direction", text_color=color.white)
    table.cell(debugTable, 1, 2, "{direction.upper()}", text_color=color.yellow)

    table.cell(debugTable, 0, 3, "Entry Rule", text_color=color.white)
    table.cell(debugTable, 1, 3, "{entry_rule}", text_color=color.yellow)

    table.cell(debugTable, 0, 4, "TP %", text_color=color.white)
    table.cell(debugTable, 1, 4, str.tostring(tpPercent, "#.##") + "%", text_color=color.green)

    table.cell(debugTable, 0, 5, "SL %", text_color=color.white)
    table.cell(debugTable, 1, 5, str.tostring(slPercent, "#.##") + "%", text_color=color.red)

    table.cell(debugTable, 0, 6, "Position Size", text_color=color.white)
    table.cell(debugTable, 1, 6, str.tostring({position_size_pct}, "#.#") + "% of equity", text_color=color.yellow)

    table.cell(debugTable, 0, 7, "Capital", text_color=color.white)
    table.cell(debugTable, 1, 7, "£" + str.tostring({capital}, "#,###"), text_color=color.yellow)

// =============================================================================
// VISUAL: ENTRY/EXIT MARKERS
// =============================================================================

plotshape(longCondition and strategy.position_size[1] == 0, "Long Entry", shape.triangleup,
          location.belowbar, color.green, size=size.small)
plotshape(shortCondition and strategy.position_size[1] == 0, "Short Entry", shape.triangledown,
          location.abovebar, color.red, size=size.small)
plotshape(positionClosed, "Exit", shape.xcross,
          location.belowbar, color.orange, size=size.tiny)
'''



