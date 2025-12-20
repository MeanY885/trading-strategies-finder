# TradingView Indicator Reference

This document contains TradingView's official indicator formulas to ensure our Python backtester
calculates indicators identically. This is CRITICAL for matching results.

## Sources
- TradingView Built-in Indicators: Pine Editor → "Indicators" → "Built-ins"
- GitHub Reference: https://github.com/TWODS-CAPITAL/Trading-View-Indicators
- **QuanTAlib (276 indicators)**: https://github.com/mihakralj/pinescript - Mathematically rigorous Pine Script implementations

---

# MOMENTUM INDICATORS

## RSI (Relative Strength Index)

### TradingView's RSI Strategy Source:
```pinescript
//@version=5
strategy("RSI Strategy", overlay=true)

rsiLength = input.int(14, minval=1)
rsiOverbought = input.int(70, minval=50, maxval=100)
rsiOversold = input.int(30, minval=0, maxval=50)

rsiValue = ta.rsi(close, rsiLength)

// Entry signals - CROSSOVER/CROSSUNDER, not state-based
longCondition = ta.crossover(rsiValue, rsiOversold)   // RSI crosses UP through 30
shortCondition = ta.crossunder(rsiValue, rsiOverbought) // RSI crosses DOWN through 70

if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
```

### Formula:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

Where:
- Average Gain = RMA of gains over period (Wilder's smoothing)
- Average Loss = RMA of losses over period (Wilder's smoothing)
- RMA (Relative Moving Average) = alpha * src + (1 - alpha) * rma[1], where alpha = 1/length
```

### Key Note:
TradingView uses **RMA (Wilder's smoothing)** NOT SMA for averaging gains/losses.

---

## Stochastic Oscillator

### TradingView's Stochastic Strategy Source:
```pinescript
//@version=5
strategy("Stochastic Strategy", overlay=true)

periodK = input.int(14, minval=1)
smoothK = input.int(3, minval=1)  // %K smoothing
periodD = input.int(3, minval=1)  // %D period

// Stochastic calculation - SMOOTHED K
k = ta.sma(ta.stoch(close, high, low, periodK), smoothK)
d = ta.sma(k, periodD)

// Entry signals - %K crosses %D in oversold/overbought zones
longCondition = ta.crossover(k, d) and k < 20
shortCondition = ta.crossunder(k, d) and k > 80

if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
```

### Formula:
```
%K (raw) = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
%K (smoothed) = SMA(%K raw, smoothK)  // smoothK = 3 by default
%D = SMA(%K smoothed, periodD)        // periodD = 3 by default

Where:
- Lowest Low = ta.lowest(low, periodK)
- Highest High = ta.highest(high, periodK)
```

### Key Note:
Our Python code uses `smooth_k=3` to match TradingView's default smoothing.

---

## Williams %R

### TradingView Source:
```pinescript
//@version=5
indicator("Williams %R")

length = input.int(14, minval=1)
upper = ta.highest(high, length)
lower = ta.lowest(low, length)
willr = 100 * (close - upper) / (upper - lower)

// Range: 0 to -100
// Overbought: > -20
// Oversold: < -80
```

### Formula:
```
Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

Entry signals:
- Long: willr < -80 (oversold)
- Short: willr > -20 (overbought)
```

---

## CCI (Commodity Channel Index)

### TradingView Source:
```pinescript
//@version=5
indicator("CCI")

length = input.int(20, minval=1)
tp = (high + low + close) / 3
ma = ta.sma(tp, length)
// Mean Absolute Deviation
mad = ta.sma(math.abs(tp - ma), length)
cci = (tp - ma) / (0.015 * mad)
```

### Formula:
```
Typical Price (TP) = (High + Low + Close) / 3
CCI = (TP - SMA(TP, length)) / (0.015 * MAD(TP, length))

Where MAD = Mean Absolute Deviation

Entry signals:
- Long: CCI < -100
- Short: CCI > 100
```

---

## RSI Divergence

### TradingView Pattern:
```pinescript
//@version=5
rsiValue = ta.rsi(close, 14)
lookback = 5

// Bullish divergence: Price makes lower low, RSI makes higher low
priceLowerLow = low < ta.lowest(low, lookback)[1]
rsiHigherLow = rsiValue > ta.valuewhen(priceLowerLow[1], rsiValue, 0)
bullishDiv = priceLowerLow and rsiHigherLow and rsiValue < 40

// Bearish divergence: Price makes higher high, RSI makes lower high
priceHigherHigh = high > ta.highest(high, lookback)[1]
rsiLowerHigh = rsiValue < ta.valuewhen(priceHigherHigh[1], rsiValue, 0)
bearishDiv = priceHigherHigh and rsiLowerHigh and rsiValue > 60
```

---

# MEAN REVERSION INDICATORS

## Bollinger Bands

### TradingView's Bollinger Bands Strategy Source:
```pinescript
//@version=5
strategy("Bollinger Bands Strategy", overlay=true)

length = input.int(20, minval=1)
mult = input.float(2.0, minval=0.001, maxval=50)
src = close

basis = ta.sma(src, length)
dev = mult * ta.stdev(src, length)
upper = basis + dev
lower = basis - dev

// Entry signals - CROSSOVER through bands
longCondition = ta.crossover(close, lower)   // Price crosses UP through lower band
shortCondition = ta.crossunder(close, upper) // Price crosses DOWN through upper band

if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
```

### Formula:
```
Middle Band (Basis) = SMA(close, length)
Standard Deviation = ta.stdev(close, length)  // Population std dev (divides by N)
Upper Band = Basis + (multiplier * StdDev)
Lower Band = Basis - (multiplier * StdDev)
BB Width = (Upper - Lower) / Middle
```

### Key Note:
TradingView uses **population standard deviation** (divides by N, not N-1).

---

## BB Squeeze Breakout

### TradingView Pattern:
```pinescript
//@version=5
[bbMid, bbUpper, bbLower] = ta.bb(close, 20, 2)
bbWidth = (bbUpper - bbLower) / bbMid
avgWidth = ta.sma(bbWidth, 20)

// Squeeze: width below 80% of average
squeezed = bbWidth[1] < avgWidth * 0.8
// Expanding: width increasing
expanding = bbWidth > bbWidth[1]

// Breakout from squeeze
longCondition = squeezed and expanding and close > bbMid
shortCondition = squeezed and expanding and close < bbMid
```

---

## VWAP (Volume Weighted Average Price)

### TradingView Source:
```pinescript
//@version=5
indicator("VWAP")

// Calculates from session start
vwapValue = ta.vwap(hlc3)

// Or with anchor:
// vwapValue = ta.vwap(hlc3, anchor)
```

### Formula:
```
VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
Typical Price = (High + Low + Close) / 3

Entry signals (VWAP Bounce):
- Long: low < vwap AND close > vwap (touched below, closed above)
- Short: high > vwap AND close < vwap (touched above, closed below)
```

---

# TREND INDICATORS

## MACD (Moving Average Convergence Divergence)

### TradingView's MACD Strategy Source:
```pinescript
//@version=5
strategy("MACD Strategy", overlay=true)

fastLength = input.int(12)
slowLength = input.int(26)
signalLength = input.int(9)

[macdLine, signalLine, histLine] = ta.macd(close, fastLength, slowLength, signalLength)

// Entry signals - Histogram crosses zero (NOT macd crosses signal)
longCondition = ta.crossover(histLine, 0)    // Histogram crosses UP through zero
shortCondition = ta.crossunder(histLine, 0)  // Histogram crosses DOWN through zero

if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
```

### Formula:
```
MACD Line = EMA(close, fastLength) - EMA(close, slowLength)
Signal Line = EMA(MACD Line, signalLength)
Histogram = MACD Line - Signal Line
```

---

## EMA (Exponential Moving Average)

### TradingView Formula:
```pinescript
// EMA formula
alpha = 2 / (length + 1)
EMA = alpha * close + (1 - alpha) * EMA[1]

// First value is SMA
EMA[0] = SMA(close, length)
```

---

## SMA (Simple Moving Average)

### TradingView Formula:
```
SMA = SUM(close, length) / length
```

---

## Moving Average Cross

### TradingView's MovingAvg2Line Cross Strategy Source:
```pinescript
//@version=5
strategy("Moving Average 2-Line Cross", overlay=true)

fastLength = input.int(9, "Fast MA")
slowLength = input.int(18, "Slow MA")

maFast = ta.sma(close, fastLength)
maSlow = ta.sma(close, slowLength)

longCondition = ta.crossover(maFast, maSlow)
shortCondition = ta.crossunder(maFast, maSlow)

if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
```

---

## SuperTrend

### TradingView Source:
```pinescript
//@version=5
indicator("SuperTrend", overlay=true)

atrPeriod = input.int(10)
factor = input.float(3.0)

[supertrend, direction] = ta.supertrend(factor, atrPeriod)
// direction: 1 = bullish (below price), -1 = bearish (above price)

// Entry on direction change
longCondition = direction == 1 and direction[1] == -1
shortCondition = direction == -1 and direction[1] == 1
```

### Formula:
```
ATR = ta.atr(atrPeriod)
HL2 = (high + low) / 2

Upper Band = HL2 + (factor * ATR)
Lower Band = HL2 - (factor * ATR)

Direction changes when price crosses bands
```

---

## ADX (Average Directional Index)

### TradingView Source:
```pinescript
//@version=5
indicator("ADX")

adxlen = input.int(14)
dilen = input.int(14)

[diplus, diminus, adx] = ta.dmi(dilen, adxlen)

// Strong trend: ADX > 25
// DI+ > DI- = bullish
// DI- > DI+ = bearish

strongTrend = adx > 25
longCondition = strongTrend and diplus > diminus
shortCondition = strongTrend and diminus > diplus
```

### Formula:
```
+DM = high - high[1] (if positive and > -DM, else 0)
-DM = low[1] - low (if positive and > +DM, else 0)
TR = True Range

+DI = 100 * RMA(+DM, len) / RMA(TR, len)
-DI = 100 * RMA(-DM, len) / RMA(TR, len)
DX = 100 * |+DI - -DI| / (+DI + -DI)
ADX = RMA(DX, len)
```

---

## Parabolic SAR

### TradingView Source:
```pinescript
//@version=5
indicator("Parabolic SAR", overlay=true)

start = input.float(0.02, step=0.01)
inc = input.float(0.02, step=0.01)
max = input.float(0.2, step=0.01)

psar = ta.sar(start, inc, max)

// Entry on crossover
longCondition = close > psar and close[1] <= psar[1]
shortCondition = close < psar and close[1] >= psar[1]
```

### Formula:
```
SAR(n) = SAR(n-1) + AF * (EP - SAR(n-1))

Where:
- AF = Acceleration Factor (starts at 0.02, increases by 0.02 each new EP, max 0.2)
- EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
```

---

# VOLATILITY INDICATORS

## ATR (Average True Range)

### TradingView Source:
```pinescript
//@version=5
indicator("ATR")

length = input.int(14)
atr = ta.atr(length)

// True Range
tr = math.max(high - low, math.max(math.abs(high - close[1]), math.abs(low - close[1])))

// ATR uses RMA (Wilder's smoothing), NOT SMA
atrValue = ta.rma(tr, length)
```

### Key Note:
TradingView uses **RMA (Wilder's smoothing)** for ATR, not SMA.

---

## ATR Breakout

### Pattern:
```pinescript
atrValue = ta.atr(14)
candleRange = high - low

// Big move: range > 1.5x ATR
bigMove = candleRange > atrValue * 1.5

longCondition = bigMove and close > open
shortCondition = bigMove and close < open
```

---

## Low Volatility Breakout

### Pattern:
```pinescript
atrValue = ta.atr(14)
avgAtr = ta.sma(atrValue, 20)

// Low volatility: ATR below 80% of average
lowVol = atrValue < avgAtr * 0.8
expanding = atrValue > atrValue[1]

longCondition = lowVol[1] and expanding and close > open
shortCondition = lowVol[1] and expanding and close < open
```

---

# CANDLESTICK PATTERNS

## Consecutive Candles

### Pattern:
```pinescript
isGreen = close > open
isRed = close < open

// 3 consecutive candles
consecutiveRed = isRed and isRed[1] and isRed[2]
consecutiveGreen = isGreen and isGreen[1] and isGreen[2]

// Reversal entry after consecutive candles
longCondition = consecutiveRed
shortCondition = consecutiveGreen
```

---

## Big Candle Reversal

### Pattern:
```pinescript
atrValue = ta.atr(14)
candleRange = high - low
isGreen = close > open
isRed = close < open

// Large candle: range > 2x ATR
largeCandle = candleRange > atrValue * 2

longCondition = largeCandle and isRed   // Large red candle = potential reversal up
shortCondition = largeCandle and isGreen // Large green candle = potential reversal down
```

---

## Doji Reversal

### Pattern:
```pinescript
body = math.abs(close - open)
range = high - low
isDoji = body < range * 0.1  // Body less than 10% of range

// Doji after trend
upTrend = close > ta.sma(close, 5)
downTrend = close < ta.sma(close, 5)

longCondition = isDoji and downTrend
shortCondition = isDoji and upTrend
```

---

## Engulfing Pattern

### Pattern:
```pinescript
prevBody = math.abs(close[1] - open[1])
currBody = math.abs(close - open)

// Bullish engulfing: red candle followed by larger green candle
bullishEngulfing = close[1] < open[1] and  // Previous red
                   close > open and         // Current green
                   currBody > prevBody and  // Current larger
                   close > open[1] and      // Current close above prev open
                   open < close[1]          // Current open below prev close

// Bearish engulfing: green candle followed by larger red candle
bearishEngulfing = close[1] > open[1] and  // Previous green
                   close < open and         // Current red
                   currBody > prevBody and  // Current larger
                   close < open[1] and      // Current close below prev open
                   open > close[1]          // Current open above prev close
```

---

# PRICE ACTION

## Higher Low / Lower High

### Pattern:
```pinescript
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
prevLow = ta.lowest(low, 20)[5]
prevHigh = ta.highest(high, 20)[5]

// Higher low pattern
higherLow = recentLow > prevLow and close > open

// Lower high pattern
lowerHigh = recentHigh < prevHigh and close < open
```

---

## Support/Resistance Bounce

### Pattern:
```pinescript
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
tolerance = ta.atr(14) * 0.5

// At support
atSupport = low <= recentLow + tolerance and close > open

// At resistance
atResistance = high >= recentHigh - tolerance and close < open
```

---

# REFERENCE LIBRARIES

## QuanTAlib (276 Indicators)
**URL:** https://github.com/mihakralj/pinescript

Categories:
- Finite Impulse Response Trends
- Infinite Impulse Response Trends
- Oscillators
- Momentum Indicators
- Volatility Indicators
- Volume Indicators
- Trend Dynamics
- Cycles Indicators
- Price Channels and Bands
- Stop and Reverse Indicators
- Signal Filters
- Numerics
- Statistics
- Error Metrics

**Key Advantage:** Mathematically rigorous implementations with proper initialization.

## TWODS-CAPITAL Trading-View-Indicators
**URL:** https://github.com/TWODS-CAPITAL/Trading-View-Indicators

Contains additional indicator implementations and entry/exit logic patterns.

---

# pandas_ta vs TradingView Comparison

| Indicator | pandas_ta | TradingView | Match? |
|-----------|-----------|-------------|--------|
| RSI | Uses RMA | Uses RMA | ✓ |
| Stochastic | SMA smoothing, smooth_k param | SMA smoothing | ✓ (with smooth_k=3) |
| Bollinger | Population StdDev | Population StdDev | ✓ |
| MACD | Standard EMA | Standard EMA | ✓ |
| ATR | Uses RMA | Uses RMA | ✓ |
| SMA | Standard | Standard | ✓ |
| EMA | Standard | Standard | ✓ |
| Williams %R | Standard | Standard | ✓ |
| CCI | Standard | Standard | ✓ |
| ADX | Uses RMA | Uses RMA | ✓ |
| SuperTrend | Standard | Standard | ✓ |
| PSAR | Standard | Standard | ✓ |

### Main Causes of Differences:
1. **Data source differences** - Yahoo Finance vs TradingView's data feed (MAIN ISSUE)
2. **Entry signal logic** - State-based vs crossover events (FIXED)
3. **Stochastic smoothing** - Need smooth_k=3 (FIXED)
4. **Floating point precision** - Minor rounding differences
5. **Bar timing** - When exactly the bar closes

---

# Strategy Entry Logic Summary

All strategies use **crossover/crossunder** events, not state-based conditions:

| Strategy | Entry Logic |
|----------|-------------|
| RSI Extreme | `ta.crossover(rsi, 30)` for long |
| RSI Cross 50 | `ta.crossover(rsi, 50)` for long |
| Stochastic | `ta.crossover(k, d) and k < 20` for long |
| Bollinger Touch | `ta.crossover(close, bbLower)` for long |
| MACD Cross | `ta.crossover(histogram, 0)` for long |
| EMA Cross | `ta.crossover(ema9, ema21)` for long |
| SMA Cross | `ta.crossover(smaFast, smaSlow)` for long |
| Price vs SMA | `close < sma20 * 0.99` for long |
| SuperTrend | `direction == 1 and direction[1] == -1` for long |
| PSAR | `close > psar and close[1] <= psar[1]` for long |
| ADX | `adx > 25 and diPlus > diMinus` for long |

---

# Changes Log

## December 2024
- Added `smooth_k=3` to Stochastic calculation to match TradingView
- Updated all Pine Script templates to use TradingView crossover patterns
- Documented all 25 strategy entry conditions
- Added QuanTAlib reference (276 indicators)
- Documented ADX, PSAR, VWAP, Williams %R, CCI formulas
- Documented candlestick patterns (engulfing, doji, consecutive)
- Documented price action patterns (higher low, support/resistance)
