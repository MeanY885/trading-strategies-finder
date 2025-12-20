"""
TradingView Built-in Strategies Reference
==========================================
This file contains the EXACT logic from TradingView's built-in strategies.
Use this as the authoritative reference when implementing Python backtests.

Source: TradingView's official strategy templates
Version: Pine Script v6

CRITICAL NOTES FOR MATCHING:
1. Most strategies use STOP orders, not market orders
2. OCA (One-Cancels-All) logic cancels pending orders when conditions change
3. Entry at close vs entry at stop price - different execution models
4. Commission and slippage handling differs

For EXACT matching, strategies should be rewritten to use market orders
with process_orders_on_close=true, which fills at close price.
"""

# =============================================================================
# STRATEGY REFERENCE DICTIONARY
# Each entry contains:
# - pine_script: Exact TradingView code
# - order_type: "market" or "stop"
# - entry_logic: Description of when entry occurs
# - exit_logic: How exits are handled
# - notes: Important matching considerations
# =============================================================================

TRADINGVIEW_STRATEGIES = {

    # =========================================================================
    # RSI STRATEGY
    # =========================================================================
    "rsi": {
        "name": "RSI Strategy",
        "pine_script": '''
//@version=6
strategy("RSI Strategy", overlay=true)
length = input(14, "Length")
overSold = input(30, "Oversold")
overBought = input(70, "Overbought")
price = close
vrsi = ta.rsi(price, length)
co = ta.crossover(vrsi, overSold)
cu = ta.crossunder(vrsi, overBought)
if (not na(vrsi))
    if (co)
        strategy.entry("RsiLE", strategy.long, comment="RsiLE")
    if (cu)
        strategy.entry("RsiSE", strategy.short, comment="RsiSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "RSI crosses OVER oversold level (30)",
            "short": "RSI crosses UNDER overbought level (70)",
        },
        "exit_logic": "No explicit exit - position reversal only",
        "notes": [
            "Uses ta.crossover/crossunder (edge detection, not level)",
            "Default RSI length = 14",
            "No TP/SL - pure reversal system",
            "Entry at bar close (default strategy behavior)",
        ],
        "python_signal": {
            "long": "(rsi > 30) & (rsi.shift(1) <= 30)",
            "short": "(rsi < 70) & (rsi.shift(1) >= 70)",
        }
    },

    # =========================================================================
    # STOCHASTIC SLOW STRATEGY
    # =========================================================================
    "stochastic_slow": {
        "name": "Stochastic Slow Strategy",
        "pine_script": '''
//@version=6
strategy("Stochastic Slow Strategy", overlay=true)
length = input.int(14, "Length", minval=1)
OverBought = input(80, "Overbought")
OverSold = input(20, "Oversold")
smoothK = 3
smoothD = 3
k = ta.sma(ta.stoch(close, high, low, length), smoothK)
d = ta.sma(k, smoothD)
co = ta.crossover(k, d)
cu = ta.crossunder(k, d)
if (not na(k) and not na(d))
    if (co and k < OverSold)
        strategy.entry("StochLE", strategy.long, comment="StochLE")
    if (cu and k > OverBought)
        strategy.entry("StochSE", strategy.short, comment="StochSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "K crosses OVER D while K < 20 (oversold zone)",
            "short": "K crosses UNDER D while K > 80 (overbought zone)",
        },
        "exit_logic": "No explicit exit - position reversal only",
        "notes": [
            "smoothK = 3, smoothD = 3 (hardcoded)",
            "K is double-smoothed: SMA of raw stoch, then D is SMA of K",
            "Check BOTH: 1) K crosses D AND 2) K is in extreme zone",
            "The zone check is AFTER the crossover, using current K value",
        ],
        "python_signal": {
            "long": "k_cross_d_over & (k < 20)",
            "short": "k_cross_d_under & (k > 80)",
        }
    },

    # =========================================================================
    # BOLLINGER BANDS STRATEGY
    # =========================================================================
    "bollinger_bands": {
        "name": "Bollinger Bands Strategy",
        "pine_script": '''
//@version=6
strategy("Bollinger Bands Strategy", overlay=true)
source = close
length = input.int(20, minval=1)
mult = input.float(2.0, minval=0.001, maxval=50)
basis = ta.sma(source, length)
dev = mult * ta.stdev(source, length)
upper = basis + dev
lower = basis - dev
buyEntry = ta.crossover(source, lower)
sellEntry = ta.crossunder(source, upper)
if (ta.crossover(source, lower))
    strategy.entry("BBandLE", strategy.long, stop=lower, oca_name="BollingerBands", oca_type=strategy.oca.cancel, comment="BBandLE")
else
    strategy.cancel(id="BBandLE")
if (ta.crossunder(source, upper))
    strategy.entry("BBandSE", strategy.short, stop=upper, oca_name="BollingerBands", oca_type=strategy.oca.cancel, comment="BBandSE")
else
    strategy.cancel(id="BBandSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "When price crosses OVER lower band, place STOP order at lower band",
            "short": "When price crosses UNDER upper band, place STOP order at upper band",
        },
        "exit_logic": "OCA cancel - pending orders cancelled if crossover no longer valid",
        "notes": [
            "CRITICAL: Uses STOP orders at band levels, NOT market orders",
            "Entry only fills if price retraces to the band level",
            "OCA cancel removes pending orders when condition changes",
            "For market order equivalent: entry on crossover at close price",
            "ta.stdev uses population std (ddof=0)",
        ],
        "python_signal_market_equivalent": {
            "long": "(close > bb_lower) & (close.shift(1) <= bb_lower.shift(1))",
            "short": "(close < bb_upper) & (close.shift(1) >= bb_upper.shift(1))",
        }
    },

    # =========================================================================
    # MACD STRATEGY
    # =========================================================================
    "macd": {
        "name": "MACD Strategy",
        "pine_script": '''
//@version=6
strategy("MACD Strategy", overlay=true)
fastLength = input(12, "Fast length")
slowlength = input(26, "Slow length")
MACDLength = input(9, "MACD length")
MACD = ta.ema(close, fastLength) - ta.ema(close, slowlength)
aMACD = ta.ema(MACD, MACDLength)
delta = MACD - aMACD
if (ta.crossover(delta, 0))
    strategy.entry("MacdLE", strategy.long, comment="MacdLE")
if (ta.crossunder(delta, 0))
    strategy.entry("MacdSE", strategy.short, comment="MacdSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Histogram (MACD - Signal) crosses OVER zero",
            "short": "Histogram (MACD - Signal) crosses UNDER zero",
        },
        "exit_logic": "Position reversal only",
        "notes": [
            "Uses histogram (delta), not MACD line crossing signal",
            "delta = MACD - aMACD (histogram)",
            "Crossover of histogram through zero line",
        ],
        "python_signal": {
            "long": "(histogram > 0) & (histogram.shift(1) <= 0)",
            "short": "(histogram < 0) & (histogram.shift(1) >= 0)",
        }
    },

    # =========================================================================
    # MOVING AVERAGE 2-LINE CROSS
    # =========================================================================
    "ma2line_cross": {
        "name": "MovingAvg2Line Cross",
        "pine_script": '''
//@version=6
strategy("MovingAvg2Line Cross", overlay=true)
fastLength = input(9)
slowLength = input(18)
price = close
mafast = ta.sma(price, fastLength)
maslow = ta.sma(price, slowLength)
if (ta.crossover(mafast, maslow))
    strategy.entry("MA2CrossLE", strategy.long, comment="MA2CrossLE")
if (ta.crossunder(mafast, maslow))
    strategy.entry("MA2CrossSE", strategy.short, comment="MA2CrossSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Fast SMA(9) crosses OVER slow SMA(18)",
            "short": "Fast SMA(9) crosses UNDER slow SMA(18)",
        },
        "python_signal": {
            "long": "(sma_9 > sma_18) & (sma_9.shift(1) <= sma_18.shift(1))",
            "short": "(sma_9 < sma_18) & (sma_9.shift(1) >= sma_18.shift(1))",
        }
    },

    # =========================================================================
    # MOMENTUM STRATEGY
    # =========================================================================
    "momentum": {
        "name": "Momentum Strategy",
        "pine_script": '''
//@version=6
strategy("Momentum Strategy", overlay=true)
length = input(12)
price = close
momentum(seria, length) =>
    mom = seria - seria[length]
    mom
mom0 = momentum(price, length)
mom1 = momentum(mom0, 1)
if (mom0 > 0 and mom1 > 0)
    strategy.entry("MomLE", strategy.long, stop=high+syminfo.mintick, comment="MomLE")
else
    strategy.cancel("MomLE")
if (mom0 < 0 and mom1 < 0)
    strategy.entry("MomSE", strategy.short, stop=low-syminfo.mintick, comment="MomSE")
else
    strategy.cancel("MomSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "mom0 > 0 AND mom1 > 0, with STOP at high + tick",
            "short": "mom0 < 0 AND mom1 < 0, with STOP at low - tick",
        },
        "notes": [
            "STOP orders at high/low + mintick",
            "Cancelled if momentum condition no longer true",
            "mom0 = close - close[12], mom1 = mom0 - mom0[1]",
        ],
        "python_signal_market_equivalent": {
            "long": "(mom0 > 0) & (mom1 > 0)",
            "short": "(mom0 < 0) & (mom1 < 0)",
        }
    },

    # =========================================================================
    # CHANNEL BREAKOUT STRATEGY
    # =========================================================================
    "channel_breakout": {
        "name": "ChannelBreakOutStrategy",
        "pine_script": '''
//@version=6
strategy("ChannelBreakOutStrategy", overlay=true)
length = input.int(title="Length", minval=1, maxval=1000, defval=5)
upBound = ta.highest(high, length)
downBound = ta.lowest(low, length)
if (not na(close[length]))
    strategy.entry("ChBrkLE", strategy.long, stop=upBound + syminfo.mintick, comment="ChBrkLE")
strategy.entry("ChBrkSE", strategy.short, stop=downBound - syminfo.mintick, comment="ChBrkSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "STOP order at highest high of last 5 bars + tick",
            "short": "STOP order at lowest low of last 5 bars - tick",
        },
        "notes": [
            "Always has pending orders at channel boundaries",
            "Fills when price breaks out of channel",
            "No cancel logic - orders always pending",
        ]
    },

    # =========================================================================
    # SUPERTREND STRATEGY
    # =========================================================================
    "supertrend": {
        "name": "Supertrend Strategy",
        "pine_script": '''
//@version=6
strategy("Supertrend Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=15)

atrPeriod = input(10, "ATR Length")
factor = input.float(3.0, "Factor", step = 0.01)

[_, direction] = ta.supertrend(factor, atrPeriod)

if ta.change(direction) < 0
    strategy.entry("My Long Entry Id", strategy.long)

if ta.change(direction) > 0
    strategy.entry("My Short Entry Id", strategy.short)
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Supertrend direction changes from positive to negative (ta.change < 0)",
            "short": "Supertrend direction changes from negative to positive (ta.change > 0)",
        },
        "notes": [
            "ta.supertrend returns [supertrend_value, direction]",
            "direction: 1 = bearish (price below), -1 = bullish (price above)",
            "Change from 1 to -1 = long signal (ta.change(direction) = -2, which is < 0)",
            "Change from -1 to 1 = short signal (ta.change(direction) = 2, which is > 0)",
            "Uses 15% of equity per trade",
        ],
        "python_signal": {
            "long": "(direction == -1) & (direction.shift(1) == 1)",  # direction changed from 1 to -1
            "short": "(direction == 1) & (direction.shift(1) == -1)",  # direction changed from -1 to 1
        }
    },

    # =========================================================================
    # PARABOLIC SAR STRATEGY
    # =========================================================================
    "parabolic_sar": {
        "name": "Parabolic SAR Strategy",
        "pine_script": '''
//@version=6
strategy("Parabolic SAR Strategy", overlay=true)
start = input(0.02)
increment = input(0.02)
maximum = input(0.2)
var bool uptrend = false
var float EP = na
var float SAR = na
var float AF = start
var float nextBarSAR = na
if bar_index > 0
    firstTrendBar = false
    SAR := nextBarSAR
    if bar_index == 1
        float prevSAR = na
        float prevEP = na
        lowPrev = low[1]
        highPrev = high[1]
        closeCur = close
        closePrev = close[1]
        if closeCur > closePrev
            uptrend := true
            EP := high
            prevSAR := lowPrev
            prevEP := high
        else
            uptrend := false
            EP := low
            prevSAR := highPrev
            prevEP := low
        firstTrendBar := true
        SAR := prevSAR + start * (prevEP - prevSAR)
    if uptrend
        if SAR > low
            firstTrendBar := true
            uptrend := false
            SAR := math.max(EP, high)
            EP := low
            AF := start
    else
        if SAR < high
            firstTrendBar := true
            uptrend := true
            SAR := math.min(EP, low)
            EP := high
            AF := start
    if not firstTrendBar
        if uptrend
            if high > EP
                EP := high
                AF := math.min(AF + increment, maximum)
        else
            if low < EP
                EP := low
                AF := math.min(AF + increment, maximum)
    if uptrend
        SAR := math.min(SAR, low[1])
        if bar_index > 1
            SAR := math.min(SAR, low[2])
    else
        SAR := math.max(SAR, high[1])
        if bar_index > 1
            SAR := math.max(SAR, high[2])
    nextBarSAR := SAR + AF * (EP - SAR)
    if barstate.isconfirmed
        if uptrend
            strategy.entry("ParSE", strategy.short, stop=nextBarSAR, comment="ParSE")
            strategy.cancel("ParLE")
        else
            strategy.entry("ParLE", strategy.long, stop=nextBarSAR, comment="ParLE")
            strategy.cancel("ParSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "In downtrend, STOP order at nextBarSAR level",
            "short": "In uptrend, STOP order at nextBarSAR level",
        },
        "notes": [
            "COMPLEX: Manual SAR calculation, not ta.sar()",
            "Uses STOP orders at calculated SAR level",
            "Cancels opposite direction order each bar",
            "For matching: use ta.sar() but results may differ slightly",
        ]
    },

    # =========================================================================
    # CONSECUTIVE UP/DOWN STRATEGY
    # =========================================================================
    "consecutive_updn": {
        "name": "Consecutive Up/Down Strategy",
        "pine_script": '''
//@version=6
strategy("Consecutive Up/Down Strategy", overlay=true)
consecutiveBarsUp = input(3, "Consecutive bars up")
consecutiveBarsDown = input(3, "Consecutive bars down")
price = close
ups = 0.0
ups := price > price[1] ? nz(ups[1]) + 1 : 0
dns = 0.0
dns := price < price[1] ? nz(dns[1]) + 1 : 0
if (ups >= consecutiveBarsUp)
    strategy.entry("ConsUpLE", strategy.long, comment="ConsUpLE")
if (dns >= consecutiveBarsDown)
    strategy.entry("ConsDnSE", strategy.short, comment="ConsDnSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "3 or more consecutive up closes",
            "short": "3 or more consecutive down closes",
        },
        "notes": [
            "Cumulative counter resets to 0 on opposite direction",
            "Entry when counter >= 3",
        ],
        "python_signal": {
            "long": "consecutive_up >= 3",
            "short": "consecutive_down >= 3",
        }
    },

    # =========================================================================
    # INSIDE BAR STRATEGY
    # =========================================================================
    "inside_bar": {
        "name": "InSide Bar Strategy",
        "pine_script": '''
//@version=6
strategy("InSide Bar Strategy", overlay=true)
if (high < high[1] and low > low[1])
    if (close > open)
        strategy.entry("InsBarLE", strategy.long, comment="InsBarLE")
    if (close < open)
        strategy.entry("InsBarSE", strategy.short, comment="InsBarSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Inside bar (H < H[1] and L > L[1]) AND bar is green (C > O)",
            "short": "Inside bar AND bar is red (C < O)",
        },
        "python_signal": {
            "long": "(high < high.shift(1)) & (low > low.shift(1)) & (close > open)",
            "short": "(high < high.shift(1)) & (low > low.shift(1)) & (close < open)",
        }
    },

    # =========================================================================
    # OUTSIDE BAR STRATEGY
    # =========================================================================
    "outside_bar": {
        "name": "OutSide Bar Strategy",
        "pine_script": '''
//@version=6
strategy("OutSide Bar Strategy", overlay=true)
if (high > high[1] and low < low[1])
    if (close > open)
        strategy.entry("OutBarLE", strategy.long, comment="OutBarLE")
    if (close < open)
        strategy.entry("OutBarSE", strategy.short, comment="OutBarSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Outside bar (H > H[1] and L < L[1]) AND bar is green",
            "short": "Outside bar AND bar is red",
        },
        "python_signal": {
            "long": "(high > high.shift(1)) & (low < low.shift(1)) & (close > open)",
            "short": "(high > high.shift(1)) & (low < low.shift(1)) & (close < open)",
        }
    },

    # =========================================================================
    # PIVOT EXTENSION STRATEGY
    # =========================================================================
    "pivot_extension": {
        "name": "Pivot Extension Strategy",
        "pine_script": '''
//@version=6
strategy("Pivot Extension Strategy", overlay=true)
leftBars = input(4, "Pivot Lookback Left")
rightBars = input(2, "Pivot Lookback Right")
ph = ta.pivothigh(leftBars, rightBars)
pl = ta.pivotlow(leftBars, rightBars)
if (not na(pl))
    strategy.entry("PivExtLE", strategy.long, comment="PivExtLE")
if (not na(ph))
    strategy.entry("PivExtSE", strategy.short, comment="PivExtSE")
''',
        "order_type": "market",
        "entry_logic": {
            "long": "Pivot low detected (not na(pl))",
            "short": "Pivot high detected (not na(ph))",
        },
        "notes": [
            "Pivot detection has rightBars lag (default 2)",
            "Signal appears 2 bars after the actual pivot",
        ]
    },

    # =========================================================================
    # PIVOT REVERSAL STRATEGY
    # =========================================================================
    "pivot_reversal": {
        "name": "Pivot Reversal Strategy",
        "pine_script": '''
//@version=6
strategy("Pivot Reversal Strategy", overlay=true)
leftBars = input(4, "Pivot Lookback Left")
rightBars = input(2, "Pivot Lookback Right")
swh = ta.pivothigh(leftBars, rightBars)
swl = ta.pivotlow(leftBars, rightBars)
swh_cond = not na(swh)
hprice = 0.0
hprice := swh_cond ? swh : hprice[1]
le = false
le := swh_cond ? true : (le[1] and high > hprice ? false : le[1])
if (le)
    strategy.entry("PivRevLE", strategy.long, comment="PivRevLE", stop=hprice + syminfo.mintick)
swl_cond = not na(swl)
lprice = 0.0
lprice := swl_cond ? swl : lprice[1]
se = false
se := swl_cond ? true : (se[1] and low < lprice ? false : se[1])
if (se)
    strategy.entry("PivRevSE", strategy.short, comment="PivRevSE", stop=lprice - syminfo.mintick)
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "STOP order at pivot high + tick, cancelled when breached",
            "short": "STOP order at pivot low - tick, cancelled when breached",
        },
        "notes": [
            "Tracks last pivot high/low price",
            "Pending order until price exceeds pivot level",
            "Complex state tracking with le/se flags",
        ]
    },

    # =========================================================================
    # PRICE CHANNEL STRATEGY
    # =========================================================================
    "price_channel": {
        "name": "Price Channel Strategy",
        "pine_script": '''
//@version=6
strategy("Price Channel Strategy", overlay=true)
length = input(20)
hh = ta.highest(high, length)
ll = ta.lowest(low, length)
if (not na(close[length]))
    strategy.entry("PChLE", strategy.long, comment="PChLE", stop=hh)
    strategy.entry("PChSE", strategy.short, comment="PChSE", stop=ll)
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "STOP order at 20-bar highest high",
            "short": "STOP order at 20-bar lowest low",
        },
        "notes": [
            "Always has pending orders at channel extremes",
            "Similar to Donchian Channel breakout",
        ]
    },

    # =========================================================================
    # KELTNER CHANNELS STRATEGY
    # =========================================================================
    "keltner_channels": {
        "name": "Keltner Channels Strategy",
        "pine_script": '''
//@version=6
strategy(title="Keltner Channels Strategy", overlay=true)
length = input.int(20, minval=1)
mult = input.float(2.0, "Multiplier")
src = input(close, title="Source")
exp = input(true, "Use Exponential MA", display = display.data_window)
BandsStyle = input.string("Average True Range", options = ["Average True Range", "True Range", "Range"], title="Bands Style", display = display.data_window)
atrlength = input(10, "ATR Length", display = display.data_window)
esma(source, length)=>
    s = ta.sma(source, length)
    e = ta.ema(source, length)
    exp ? e : s
ma = esma(src, length)
rangema = BandsStyle == "True Range" ? ta.tr(true) : BandsStyle == "Average True Range" ? ta.atr(atrlength) : ta.rma(high - low, length)
upper = ma + rangema * mult
lower = ma - rangema * mult
crossUpper = ta.crossover(src, upper)
crossLower = ta.crossunder(src, lower)
bprice = 0.0
bprice := crossUpper ? high+syminfo.mintick : nz(bprice[1])
sprice = 0.0
sprice := crossLower ? low -syminfo.mintick : nz(sprice[1])
crossBcond = false
crossBcond := crossUpper ? true : crossBcond[1]
crossScond = false
crossScond := crossLower ? true : crossScond[1]
cancelBcond = crossBcond and (src < ma or high >= bprice )
cancelScond = crossScond and (src > ma or low <= sprice )
if (cancelBcond)
    strategy.cancel("KltChLE")
if (crossUpper)
    strategy.entry("KltChLE", strategy.long, stop=bprice, comment="KltChLE")
if (cancelScond)
    strategy.cancel("KltChSE")
if (crossLower)
    strategy.entry("KltChSE", strategy.short, stop=sprice, comment="KltChSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "Price crosses over upper Keltner, STOP at high + tick",
            "short": "Price crosses under lower Keltner, STOP at low - tick",
        },
        "notes": [
            "Complex cancel conditions based on price reverting to MA",
            "Default uses EMA for middle band, ATR for width",
        ]
    },

    # =========================================================================
    # VOLTY EXPAN CLOSE STRATEGY
    # =========================================================================
    "volty_expan_close": {
        "name": "Volty Expan Close Strategy",
        "pine_script": '''
//@version=6
strategy("Volty Expan Close Strategy", overlay=true)
length = input(5, "Length")
numATRs = input(0.75, "ATR Mult")
atrs = ta.sma(ta.tr, length)*numATRs
if (not na(close[length]))
    strategy.entry("VltClsLE", strategy.long, stop=close+atrs, comment = "VltClsLE")
    strategy.entry("VltClsSE", strategy.short, stop=close-atrs, comment = "VltClsSE")
''',
        "order_type": "stop",
        "entry_logic": {
            "long": "STOP at close + 0.75 * SMA(TR, 5)",
            "short": "STOP at close - 0.75 * SMA(TR, 5)",
        },
        "notes": [
            "Uses SMA of True Range, not ATR (which uses RMA)",
            "Both orders always pending",
        ]
    },
}


# =============================================================================
# HELPER FUNCTIONS FOR MATCHING
# =============================================================================

def get_strategy_order_type(strategy_name: str) -> str:
    """Get the order type for a strategy."""
    strategy = TRADINGVIEW_STRATEGIES.get(strategy_name, {})
    return strategy.get("order_type", "market")


def get_python_signal(strategy_name: str, direction: str) -> str:
    """Get the Python signal logic for a strategy."""
    strategy = TRADINGVIEW_STRATEGIES.get(strategy_name, {})
    signals = strategy.get("python_signal", strategy.get("python_signal_market_equivalent", {}))
    return signals.get(direction, "False")


def list_market_order_strategies() -> list:
    """List strategies that use market orders (easier to match)."""
    return [name for name, data in TRADINGVIEW_STRATEGIES.items()
            if data.get("order_type") == "market"]


def list_stop_order_strategies() -> list:
    """List strategies that use stop orders (harder to match exactly)."""
    return [name for name, data in TRADINGVIEW_STRATEGIES.items()
            if data.get("order_type") == "stop"]


# =============================================================================
# NOTES ON MATCHING
# =============================================================================
"""
STRATEGIES THAT USE MARKET ORDERS (Easier to Match):
- RSI Strategy
- Stochastic Slow Strategy
- MACD Strategy
- MovingAvg2Line Cross
- Consecutive Up/Down Strategy
- Inside Bar Strategy
- Outside Bar Strategy
- Pivot Extension Strategy
- Supertrend Strategy

STRATEGIES THAT USE STOP ORDERS (Harder to Match Exactly):
- Bollinger Bands Strategy
- Momentum Strategy
- Channel Breakout Strategy
- Parabolic SAR Strategy
- Pivot Reversal Strategy
- Price Channel Strategy
- Keltner Channels Strategy
- Volty Expan Close Strategy

For stop order strategies, you have two options:
1. Implement stop order logic in Python (complex, requires intrabar simulation)
2. Convert to market order equivalent (entry on crossover at close price)

Option 2 is simpler but will produce different trade entries and results.
"""
