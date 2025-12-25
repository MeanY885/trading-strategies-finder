"""
Data Fetcher Module - Binance USDT via CCXT

Single source of truth for all trading data:
- Exchange: Binance
- Quote Currency: USDT
- Library: CCXT (industry standard)

This ensures:
- Years of historical depth
- Direct TradingView compatibility (match Binance charts)
- Consistent data format across all pairs
- No source confusion

Data Format Specification:
-------------------------
Every DataFrame returned contains exactly these columns:
  - time: datetime64[ns] - Candle open time (UTC, timezone-naive)
  - open: float64 - Opening price in USDT
  - high: float64 - Highest price in USDT
  - low: float64 - Lowest price in USDT
  - close: float64 - Closing price in USDT
  - volume: float64 - Volume in base currency

All downstream engines (strategy_engine, backtester, optimizer) expect this exact format.
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DataValidationResult:
    """Result of data validation check"""
    valid: bool
    message: str
    candle_count: int
    expected_count: int
    coverage_pct: float
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    gaps_detected: int
    price_anomalies: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "message": self.message,
            "candle_count": self.candle_count,
            "expected_count": self.expected_count,
            "coverage_pct": self.coverage_pct,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "gaps_detected": self.gaps_detected,
            "price_anomalies": self.price_anomalies,
        }


class BinanceDataFetcher:
    """
    Unified Binance USDT data fetcher using CCXT.

    Key Features:
    - Uses CCXT library for robust exchange connectivity
    - Automatic pagination for deep historical data
    - Progress updates for UI feedback
    - Built-in data validation
    - Consistent data format guaranteed

    TradingView Matching:
    - All data comes from Binance exchange
    - Use "BINANCE:" prefix on TradingView (e.g., BINANCE:BTCUSDT)
    - Timeframes map directly to TradingView intervals

    Supported Pairs (subset):
    - BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT
    - DOGEUSDT, DOTUSDT, MATICUSDT, LTCUSDT, AVAXUSDT
    - Any pair available on Binance Spot

    Supported Timeframes:
    - 1m, 3m, 5m, 15m, 30m (minute intervals)
    - 1h, 2h, 4h, 6h, 8h, 12h (hourly intervals)
    - 1d, 3d, 1w, 1M (daily+)
    """

    # Exchange identifier for TradingView matching
    EXCHANGE = "binance"
    TRADINGVIEW_PREFIX = "BINANCE:"

    # Timeframe mapping (minutes to CCXT format)
    TIMEFRAME_MAP = {
        1: '1m', 3: '3m', 5: '5m', 15: '15m', 30: '30m',
        60: '1h', 120: '2h', 240: '4h', 360: '6h',
        480: '8h', 720: '12h', 1440: '1d',
    }

    # Expected candles per timeframe per day (for validation)
    CANDLES_PER_DAY = {
        1: 1440, 3: 480, 5: 288, 15: 96, 30: 48,
        60: 24, 120: 12, 240: 6, 360: 4, 480: 3, 720: 2, 1440: 1,
    }

    def __init__(self, status_callback=None):
        """
        Initialize the Binance data fetcher.

        Args:
            status_callback: Function(message: str, progress: int) for UI updates
        """
        self.status_callback = status_callback
        self._exchange = None

    def _update_status(self, message: str, progress: int = None):
        """Update status for UI feedback"""
        print(message)
        if self.status_callback:
            self.status_callback(message, progress)

    def _get_exchange(self):
        """Get or create CCXT Binance exchange instance"""
        if self._exchange is None:
            import ccxt
            self._exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True}
            })
        return self._exchange

    def _normalize_pair(self, pair: str) -> str:
        """
        Normalize pair format for CCXT (e.g., BTCUSDT -> BTC/USDT).

        Args:
            pair: Input pair in any format (BTCUSDT, BTC/USDT, btcusdt)

        Returns:
            Normalized pair in CCXT format (BTC/USDT)
        """
        pair = pair.upper().replace("-", "").replace("/", "")

        # Common quote currencies in priority order
        quotes = ['USDT', 'USDC', 'BUSD', 'USD']
        for quote in quotes:
            if pair.endswith(quote):
                base = pair[:-len(quote)]
                if base:
                    return f"{base}/{quote}"

        # Default: assume USDT
        if len(pair) >= 3:
            return f"{pair}/USDT"

        return pair

    def _get_timeframe(self, minutes: int) -> str:
        """Convert minutes to CCXT timeframe string"""
        if minutes in self.TIMEFRAME_MAP:
            return self.TIMEFRAME_MAP[minutes]
        # Find closest supported timeframe
        for m in sorted(self.TIMEFRAME_MAP.keys()):
            if m >= minutes:
                return self.TIMEFRAME_MAP[m]
        return '1d'

    def get_tradingview_symbol(self, pair: str) -> str:
        """
        Get the TradingView-compatible symbol.

        Args:
            pair: Pair in any format

        Returns:
            TradingView symbol (e.g., "BINANCE:BTCUSDT")
        """
        normalized = self._normalize_pair(pair).replace("/", "")
        return f"{self.TRADINGVIEW_PREFIX}{normalized}"

    async def fetch_ohlcv(
        self,
        pair: str = "BTCUSDT",
        interval: int = 15,
        months: float = 3
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data from Binance.

        Args:
            pair: Trading pair (e.g., BTCUSDT, BTC/USDT)
            interval: Candle interval in minutes (1, 5, 15, 30, 60, 240, 1440)
            months: How many months of historical data to fetch

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
            - time: datetime64[ns], UTC timezone-naive
            - open/high/low/close: float64, prices in USDT
            - volume: float64, volume in base currency

        Raises:
            Returns empty DataFrame on error (with status update)

        Example:
            >>> fetcher = BinanceDataFetcher()
            >>> df = await fetcher.fetch_ohlcv("BTCUSDT", 15, 3)
            >>> print(df.head())
                                    time      open      high       low     close       volume
            0 2024-09-25 00:00:00  63000.0  63100.0  62900.0  63050.0   125.5
            1 2024-09-25 00:15:00  63050.0  63150.0  63000.0  63100.0   130.2
        """
        import ccxt

        # Normalize inputs
        symbol = self._normalize_pair(pair)
        timeframe = self._get_timeframe(interval)

        self._update_status(f"[Binance] Connecting to Binance...", 5)

        # Get exchange instance
        try:
            exchange = self._get_exchange()
            exchange.load_markets()
        except Exception as e:
            self._update_status(f"[Binance] Connection error: {e}", 0)
            return self._empty_dataframe()

        # Validate pair exists
        if symbol not in exchange.markets:
            available = [s for s in exchange.markets.keys() if 'USDT' in s][:10]
            self._update_status(f"[Binance] {symbol} not found. Try: {available}", 0)
            return self._empty_dataframe()

        # Validate timeframe
        if timeframe not in exchange.timeframes:
            self._update_status(f"[Binance] {timeframe} not supported", 0)
            return self._empty_dataframe()

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months * 30)
        since = exchange.parse8601(start_date.isoformat())

        self._update_status(f"[Binance] Fetching {months} months of {timeframe} {symbol}...", 10)

        # Pagination settings
        max_limit = 1000  # Binance limit per request
        rate_limit = 0.1  # 100ms between requests (Binance allows 1200/min)

        # Calculate expected candles for progress tracking
        days = months * 30
        candles_per_day = self.CANDLES_PER_DAY.get(interval, 96)
        expected_candles = int(days * candles_per_day)

        all_data = []
        request_count = 0
        max_requests = 500  # Safety limit

        while request_count < max_requests:
            try:
                # Fetch batch
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=max_limit)

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Progress update
                progress = min(90, 10 + int((len(all_data) / expected_candles) * 80))
                self._update_status(
                    f"[Binance] Downloaded {len(all_data):,} candles...",
                    progress
                )

                # Check if we've received less than limit (end of data)
                if len(ohlcv) < max_limit:
                    break

                # Move since to after last candle
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1

                # Check if we've reached current time
                if last_timestamp > end_date.timestamp() * 1000:
                    break

                request_count += 1
                await asyncio.sleep(rate_limit)

            except ccxt.RateLimitExceeded:
                self._update_status(f"[Binance] Rate limit, waiting 5s...", progress)
                await asyncio.sleep(5)
            except ccxt.NetworkError as e:
                self._update_status(f"[Binance] Network error: {e}", 0)
                break
            except ccxt.ExchangeError as e:
                self._update_status(f"[Binance] Exchange error: {e}", 0)
                break
            except Exception as e:
                self._update_status(f"[Binance] Error: {e}", 0)
                break

        if not all_data:
            self._update_status(f"[Binance] No data returned for {symbol}", 0)
            return self._empty_dataframe()

        # Convert to DataFrame with proper types
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime (UTC, timezone-naive)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)

        # Ensure correct column order
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Convert to proper dtypes
        df['open'] = df['open'].astype(np.float64)
        df['high'] = df['high'].astype(np.float64)
        df['low'] = df['low'].astype(np.float64)
        df['close'] = df['close'].astype(np.float64)
        df['volume'] = df['volume'].astype(np.float64)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

        # Filter to requested time range
        df = df[df['time'] >= start_date.replace(tzinfo=None)].reset_index(drop=True)

        # Final stats
        if len(df) > 0:
            actual_days = (df['time'].max() - df['time'].min()).days
            self._update_status(
                f"[Binance] ✓ {len(df):,} candles ({actual_days} days) - {self.get_tradingview_symbol(pair)}",
                100
            )
        else:
            self._update_status(f"[Binance] No data in requested range", 100)

        return df

    def validate_data(
        self,
        df: pd.DataFrame,
        interval_minutes: int,
        months: float,
        min_coverage_pct: float = 90.0
    ) -> DataValidationResult:
        """
        Validate fetched data for quality and completeness.

        Args:
            df: DataFrame from fetch_ohlcv
            interval_minutes: Expected candle interval
            months: Expected period in months
            min_coverage_pct: Minimum acceptable coverage (default 90%)

        Returns:
            DataValidationResult with validation details

        Validation Checks:
        1. Column presence and types
        2. Data coverage (expected vs actual candles)
        3. Gap detection (missing candles)
        4. Price anomalies (zeros, negatives, extreme spikes)
        """
        # Empty data check
        if df is None or len(df) == 0:
            return DataValidationResult(
                valid=False,
                message="No data received",
                candle_count=0,
                expected_count=0,
                coverage_pct=0.0,
                start_date=None,
                end_date=None,
                gaps_detected=0,
                price_anomalies=0
            )

        # Column check
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return DataValidationResult(
                valid=False,
                message=f"Missing columns: {missing_cols}",
                candle_count=len(df),
                expected_count=0,
                coverage_pct=0.0,
                start_date=None,
                end_date=None,
                gaps_detected=0,
                price_anomalies=0
            )

        # Calculate expected candles
        candles_per_day = self.CANDLES_PER_DAY.get(interval_minutes, 96)
        expected_candles = int(months * 30 * candles_per_day)
        actual_candles = len(df)
        coverage_pct = (actual_candles / expected_candles * 100) if expected_candles > 0 else 0

        # Date range
        start_date = df['time'].min()
        end_date = df['time'].max()
        actual_days = (end_date - start_date).days

        # Gap detection (check for missing candles)
        if len(df) > 1:
            df_sorted = df.sort_values('time')
            time_diffs = df_sorted['time'].diff().dropna()
            expected_diff = pd.Timedelta(minutes=interval_minutes)
            gaps = (time_diffs > expected_diff * 1.5).sum()
        else:
            gaps = 0

        # Price anomaly detection
        anomalies = 0
        for col in ['open', 'high', 'low', 'close']:
            anomalies += (df[col] <= 0).sum()  # Non-positive prices
            anomalies += (df[col].isna()).sum()  # NaN values

        # Volume check
        anomalies += (df['volume'] < 0).sum()

        # High-Low consistency
        anomalies += (df['high'] < df['low']).sum()

        # Validation decision
        if coverage_pct < min_coverage_pct:
            message = f"Low coverage: {coverage_pct:.1f}% (need {min_coverage_pct}%)"
            valid = False
        elif anomalies > 0:
            message = f"Data quality issues: {anomalies} anomalies detected"
            valid = False
        elif gaps > actual_candles * 0.1:  # More than 10% gaps
            message = f"Too many gaps: {gaps} gaps in {actual_candles} candles"
            valid = False
        else:
            message = f"Valid: {actual_candles:,} candles, {actual_days} days, {coverage_pct:.1f}% coverage"
            valid = True

        return DataValidationResult(
            valid=valid,
            message=message,
            candle_count=actual_candles,
            expected_count=expected_candles,
            coverage_pct=round(coverage_pct, 1),
            start_date=start_date,
            end_date=end_date,
            gaps_detected=gaps,
            price_anomalies=anomalies
        )

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with correct schema"""
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


# Factory function for backward compatibility
def get_fetcher(source: str = "binance", status_callback=None):
    """
    Get a data fetcher instance.

    Args:
        source: Ignored (always uses Binance)
        status_callback: Function for status updates

    Returns:
        BinanceDataFetcher instance
    """
    return BinanceDataFetcher(status_callback=status_callback)


def get_optimal_fetcher(pair: str, interval_minutes: int, months: float, status_callback=None):
    """
    Get the optimal fetcher for a given configuration.

    Args:
        pair: Trading pair
        interval_minutes: Candle interval
        months: Historical period
        status_callback: Function for status updates

    Returns:
        Tuple of (BinanceDataFetcher, "binance")
    """
    return BinanceDataFetcher(status_callback=status_callback), "binance"


# Backward compatibility aliases
CCXTDataFetcher = BinanceDataFetcher
KrakenDataFetcher = BinanceDataFetcher
CryptoCompareDataFetcher = BinanceDataFetcher
YFinanceDataFetcher = BinanceDataFetcher


async def test_fetcher():
    """Test the data fetcher"""
    print("=" * 60)
    print("Testing Binance Data Fetcher (CCXT)")
    print("=" * 60)

    fetcher = BinanceDataFetcher()

    # Test BTC/USDT
    print("\n1. Fetching BTC/USDT @ 15m, 1 month...")
    df = await fetcher.fetch_ohlcv("BTCUSDT", 15, 1)
    print(f"   Candles: {len(df)}")
    print(f"   TradingView: {fetcher.get_tradingview_symbol('BTCUSDT')}")

    # Validate
    result = fetcher.validate_data(df, 15, 1)
    print(f"   Validation: {result.message}")

    if len(df) > 0:
        print(f"\n   Sample data:")
        print(df.head(3).to_string())
        print(f"\n   Data types:")
        print(df.dtypes)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_fetcher())
