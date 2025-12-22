"""
Data Fetcher Module - Binance & Yahoo Finance APIs
Provides historical OHLCV data for backtesting and optimization
"""
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from abc import ABC, abstractmethod


class DataFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    @abstractmethod
    async def fetch_ohlcv(self, pair: str, interval: int, months: float) -> pd.DataFrame:
        """Fetch OHLCV data - returns DataFrame with: time, open, high, low, close, volume"""
        pass


class BinanceDataFetcher(DataFetcher):
    """
    Fetches historical OHLCV data from Binance public API.
    No API key required. Supports years of historical data.

    Best for: USDT pairs (BTCUSDT, ETHUSDT, etc.)
    """

    BASE_URL = "https://api.binance.com/api/v3"

    INTERVAL_MAP = {
        1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m",
        60: "1h", 120: "2h", 240: "4h", 360: "6h",
        480: "8h", 720: "12h", 1440: "1d",
    }

    def __init__(self, status_callback=None):
        self.session = None
        self.status_callback = status_callback

    def _update_status(self, message: str, progress: int = None):
        print(message)
        if self.status_callback:
            self.status_callback(message, progress)
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_ohlcv(
        self,
        pair: str = "BTCUSDT",
        interval: int = 15,
        months: float = 12
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        session = await self._get_session()

        binance_interval = self._get_binance_interval(interval)
        symbol = pair.upper().replace("/", "").replace("-", "")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=months * 30)

        end_ts = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)

        self._update_status(f"[Binance] Connecting to Binance API...", 10)
        self._update_status(f"[Binance] Fetching {months} months of {interval}m {symbol} data...", 20)

        all_data = []
        current_start = start_ts
        request_count = 0

        # Estimate total requests needed for progress calculation
        total_ms = end_ts - start_ts
        candles_per_request = 1000
        ms_per_candle = interval * 60 * 1000
        estimated_requests = max(1, total_ms // (candles_per_request * ms_per_candle))

        while current_start < end_ts and request_count < 500:
            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }

            try:
                async with session.get(f"{self.BASE_URL}/klines", params=params) as response:
                    if response.status != 200:
                        error = await response.text()
                        self._update_status(f"[Binance] Error: {response.status} - {error}", 0)
                        break

                    klines = await response.json()
                    if not klines:
                        break

                    for k in klines:
                        all_data.append({
                            "time": datetime.utcfromtimestamp(k[0] / 1000),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })

                    current_start = klines[-1][0] + 1

                    # Update progress every 5 requests
                    if request_count % 5 == 0:
                        progress = min(90, 20 + int((request_count / max(1, estimated_requests)) * 70))
                        self._update_status(f"[Binance] Downloading... {len(all_data):,} candles", progress)

                    if len(klines) < 1000:
                        break

                    request_count += 1
                    await asyncio.sleep(0.1)

            except Exception as e:
                self._update_status(f"[Binance] Error: {e}", 0)
                break

        await self.close()

        if not all_data:
            self._update_status("[Binance] Warning: No data fetched", 0)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data)
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        days = (df['time'].max() - df['time'].min()).days
        self._update_status(f"[Binance] ✓ Fetched {len(df):,} candles ({days} days)", 100)

        return df
    
    def _get_binance_interval(self, minutes: int) -> str:
        if minutes in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[minutes]
        for m in sorted(self.INTERVAL_MAP.keys()):
            if m >= minutes:
                return self.INTERVAL_MAP[m]
        return "15m"


class YFinanceDataFetcher(DataFetcher):
    """
    Fetches historical OHLCV data from Yahoo Finance.
    No API key required. Supports GBP pairs.
    """
    
    INTERVAL_MAP = {
        1: "1m", 5: "5m", 15: "15m", 30: "30m",
        60: "1h", 1440: "1d",
    }

    # Yahoo Finance maximum historical data by interval (in days)
    # Source: Yahoo Finance API limitations for intraday data
    MAX_HISTORY_DAYS = {
        1: 7,      # 1m: max 7 days
        5: 60,     # 5m: max 60 days
        15: 60,    # 15m: max 60 days
        30: 60,    # 30m: max 60 days
        60: 730,   # 1h: max 730 days (2 years)
        1440: 9999 # 1d: effectively unlimited
    }
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
    
    def _update_status(self, message: str, progress: int = None):
        print(message)
        if self.status_callback:
            self.status_callback(message, progress)
    
    async def fetch_ohlcv(
        self,
        pair: str = "BTC-GBP",
        interval: int = 15,
        months: float = 12
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_sync, pair, interval, months)
    
    def _fetch_sync(self, pair: str, interval: int, months: float) -> pd.DataFrame:
        """Synchronous fetch using yfinance"""
        try:
            import yfinance as yf
        except ImportError:
            self._update_status("[YFinance] Error: yfinance not installed", 0)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        
        # Format symbol for Yahoo
        symbol = pair.upper().replace("/", "-")
        if "-" not in symbol:
            if symbol.endswith("GBP"):
                symbol = symbol[:-3] + "-GBP"
            elif symbol.endswith("USD"):
                symbol = symbol[:-3] + "-USD"
            elif symbol.endswith("EUR"):
                symbol = symbol[:-3] + "-EUR"
        
        yf_interval = self._get_yf_interval(interval)
        
        # Calculate period - yfinance only accepts: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
        # For intraday data (< 1h), minimum period is 5d for reliable data
        days = int(months * 30)

        # Check Yahoo Finance limits for this interval
        max_days = self.MAX_HISTORY_DAYS.get(interval, 9999)
        if days > max_days:
            self._update_status(
                f"Error: Yahoo Finance only provides {max_days} days of {interval}m data. "
                f"Please select a shorter period (max {max_days} days) or use Binance for longer history.",
                0
            )
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        # For intraday intervals, enforce minimum 5d period
        if interval < 60:
            if days < 5:
                days = 5
        
        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "max"
        
        self._update_status(f"Connecting to Yahoo Finance...", 10)

        # Retry logic for flaky yfinance connections
        max_retries = 3
        df = None

        for attempt in range(max_retries):
            try:
                self._update_status(f"Fetching {symbol} @ {yf_interval} (attempt {attempt + 1})...", 30)

                # Use yf.download() which is more reliable than ticker.history()
                self._update_status(f"Downloading {period} of data...", 50)

                df = yf.download(
                    symbol,
                    period=period,
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=False,  # Use raw prices for consistency with Binance
                    prepost=False
                )

                if df is not None and not df.empty:
                    break  # Success

                if attempt < max_retries - 1:
                    self._update_status(f"Empty response, retrying...", 40)
                    import time
                    time.sleep(1)  # Brief delay before retry

            except Exception as e:
                if attempt < max_retries - 1:
                    self._update_status(f"Retry {attempt + 1} failed: {e}", 40)
                    import time
                    time.sleep(1)
                else:
                    raise  # Re-raise on final attempt

        if df is None or df.empty:
            self._update_status(f"No data for {symbol} - try Binance or a different pair", 0)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        try:
            self._update_status(f"Processing {len(df)} candles...", 80)

            # Normalize columns
            df = df.reset_index()

            # Handle MultiIndex columns from yfinance (common in v0.2.40+)
            # yfinance returns columns like ('Open', 'BTC-GBP') for single tickers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)  # Drop ticker level, keep OHLCV names

            # Lowercase all column names
            df.columns = [str(c).lower() for c in df.columns]
            
            # Handle different column name formats
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'time'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'time'})
            
            # Select only OHLCV columns
            cols_needed = ['time', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in cols_needed if c in df.columns]
            df = df[available_cols].copy()
            
            # Ensure time is timezone-naive datetime
            df['time'] = pd.to_datetime(df['time'])
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            
            days = (df['time'].max() - df['time'].min()).days
            self._update_status(f"✓ Fetched {len(df)} candles ({days} days)", 100)
            
            return df
            
        except Exception as e:
            self._update_status(f"Error: {e} - try Binance instead", 0)
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    def _get_yf_interval(self, minutes: int) -> str:
        if minutes in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[minutes]
        for m in sorted(self.INTERVAL_MAP.keys()):
            if m >= minutes:
                return self.INTERVAL_MAP[m]
        return "1h"


class KrakenDataFetcher(DataFetcher):
    """
    Fetches historical OHLCV data from Kraken public API.
    No API key required. Best for GBP pairs.

    Kraken intervals: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 (minutes)
    """

    BASE_URL = "https://api.kraken.com/0/public"

    # Kraken supported intervals (in minutes)
    INTERVAL_MAP = {
        1: 1, 5: 5, 15: 15, 30: 30,
        60: 60, 240: 240, 1440: 1440,
    }

    # Map common pair names to Kraken format
    # Kraken uses XBT instead of BTC
    PAIR_MAP = {
        'BTCGBP': 'XBTGBP', 'BTC-GBP': 'XBTGBP', 'BTC/GBP': 'XBTGBP',
        'ETHGBP': 'ETHGBP', 'ETH-GBP': 'ETHGBP', 'ETH/GBP': 'ETHGBP',
        'XRPGBP': 'XRPGBP', 'XRP-GBP': 'XRPGBP', 'XRP/GBP': 'XRPGBP',
        'ADAGBP': 'ADAGBP', 'ADA-GBP': 'ADAGBP', 'ADA/GBP': 'ADAGBP',
        'SOLGBP': 'SOLGBP', 'SOL-GBP': 'SOLGBP', 'SOL/GBP': 'SOLGBP',
        'DOGEGBP': 'XDGGBP', 'DOGE-GBP': 'XDGGBP', 'DOGE/GBP': 'XDGGBP',  # Kraken uses XDG for DOGE
        'TRXGBP': 'TRXGBP', 'TRX-GBP': 'TRXGBP', 'TRX/GBP': 'TRXGBP',
        'BNBGBP': 'BNBGBP', 'BNB-GBP': 'BNBGBP', 'BNB/GBP': 'BNBGBP',
        # EUR pairs as backup
        'BTCEUR': 'XBTEUR', 'BTC-EUR': 'XBTEUR',
        'ETHEUR': 'ETHEUR', 'ETH-EUR': 'ETHEUR',
    }

    def __init__(self, status_callback=None):
        self.session = None
        self.status_callback = status_callback

    def _update_status(self, message: str, progress: int = None):
        print(message)
        if self.status_callback:
            self.status_callback(message, progress)

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    def _get_kraken_pair(self, pair: str) -> str:
        """Convert user pair format to Kraken format"""
        pair_upper = pair.upper().replace("/", "").replace("-", "")
        # Check direct mapping
        for key, value in self.PAIR_MAP.items():
            if key.replace("-", "").replace("/", "") == pair_upper:
                return value
        # If no mapping, try as-is with XBT substitution
        if pair_upper.startswith('BTC'):
            return 'XBT' + pair_upper[3:]
        return pair_upper

    def _get_kraken_interval(self, minutes: int) -> int:
        """Get closest supported Kraken interval"""
        if minutes in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[minutes]
        # Find closest supported interval
        for m in sorted(self.INTERVAL_MAP.keys()):
            if m >= minutes:
                return self.INTERVAL_MAP[m]
        return 1440  # Default to 1d

    async def fetch_ohlcv(
        self,
        pair: str = "BTCGBP",
        interval: int = 15,
        months: float = 3
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Kraken"""
        session = await self._get_session()

        kraken_pair = self._get_kraken_pair(pair)
        kraken_interval = self._get_kraken_interval(interval)

        # Calculate start time
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=months * 30)
        since = int(start_time.timestamp())

        self._update_status(f"[Kraken] Connecting to Kraken API...", 10)
        self._update_status(f"[Kraken] Fetching {months} months of {interval}m {pair} data...", 20)

        all_data = []
        request_count = 0
        max_requests = 100  # Safety limit

        while request_count < max_requests:
            params = {
                "pair": kraken_pair,
                "interval": kraken_interval,
                "since": since
            }

            try:
                async with session.get(f"{self.BASE_URL}/OHLC", params=params) as response:
                    if response.status != 200:
                        error = await response.text()
                        self._update_status(f"[Kraken] Error: {response.status} - {error}", 0)
                        break

                    data = await response.json()

                    if data.get('error') and len(data['error']) > 0:
                        self._update_status(f"[Kraken] API Error: {data['error']}", 0)
                        break

                    result = data.get('result', {})

                    # Find the OHLC data (key varies by pair)
                    ohlc_data = None
                    for key, value in result.items():
                        if key != 'last' and isinstance(value, list):
                            ohlc_data = value
                            break

                    if not ohlc_data:
                        break

                    # Parse OHLC data: [time, open, high, low, close, vwap, volume, count]
                    for candle in ohlc_data:
                        candle_time = datetime.utcfromtimestamp(candle[0])
                        if candle_time > end_time:
                            continue
                        all_data.append({
                            "time": candle_time,
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[6])
                        })

                    # Update progress
                    progress = min(90, 20 + request_count * 10)
                    self._update_status(f"[Kraken] Downloading... {len(all_data):,} candles", progress)

                    # Get 'last' timestamp for pagination
                    last_ts = result.get('last')
                    if not last_ts or last_ts <= since:
                        break

                    # Check if we've reached current time
                    if len(ohlc_data) < 720:  # Kraken returns max 720 candles
                        break

                    since = last_ts
                    request_count += 1
                    await asyncio.sleep(0.5)  # Rate limiting

            except Exception as e:
                self._update_status(f"[Kraken] Error: {e}", 0)
                break

        await self.close()

        if not all_data:
            self._update_status(f"[Kraken] No data found for {pair}. Try a different pair or use Binance.", 0)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data)
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        # Filter to requested time range
        df = df[df['time'] >= start_time].reset_index(drop=True)

        days = (df['time'].max() - df['time'].min()).days if len(df) > 0 else 0
        self._update_status(f"[Kraken] ✓ Fetched {len(df):,} candles ({days} days)", 100)

        return df


def get_fetcher(source: str = "binance", status_callback=None) -> DataFetcher:
    """Factory function to get appropriate data fetcher"""
    if source.lower() == "yfinance":
        return YFinanceDataFetcher(status_callback=status_callback)
    elif source.lower() == "kraken":
        return KrakenDataFetcher(status_callback=status_callback)
    return BinanceDataFetcher(status_callback=status_callback)


async def main():
    """Test the data fetchers"""
    print("=" * 50)
    print("Testing Binance (BTCUSDT)")
    print("=" * 50)
    binance = BinanceDataFetcher()
    df = await binance.fetch_ohlcv(pair="BTCUSDT", interval=15, months=1)
    print(f"Shape: {df.shape}")
    print(df.head(3))

    print("\n" + "=" * 50)
    print("Testing Kraken (BTCGBP)")
    print("=" * 50)
    kraken = KrakenDataFetcher()
    df = await kraken.fetch_ohlcv(pair="BTCGBP", interval=15, months=1)
    print(f"Shape: {df.shape}")
    print(df.head(3))

    print("\n" + "=" * 50)
    print("Testing YFinance (BTC-GBP)")
    print("=" * 50)
    yf = YFinanceDataFetcher()
    df = await yf.fetch_ohlcv(pair="BTC-GBP", interval=60, months=3)
    print(f"Shape: {df.shape}")
    print(df.head(3))


if __name__ == "__main__":
    asyncio.run(main())
