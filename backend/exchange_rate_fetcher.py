"""
Exchange Rate Fetcher Module - Historical USD/GBP rates from Frankfurter API
Free API, no key required, daily rates back to 1999
"""
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from logging_config import log


class ExchangeRateFetcher:
    """
    Fetches historical USD/GBP exchange rates from Frankfurter API.

    Features:
    - No API key required
    - Daily rates updated at 16:00 CET
    - Historical data back to 1999
    - Efficient time-series endpoint for bulk fetching
    - In-memory caching to avoid repeat API calls
    """

    BASE_URL = "https://api.frankfurter.dev/v1"
    DEFAULT_RATE = 0.79  # Fallback rate if API fails
    REQUEST_TIMEOUT = 30  # Timeout in seconds to prevent indefinite hangs

    def __init__(self, status_callback=None):
        self.session = None
        self.status_callback = status_callback
        self._rate_cache: Dict[str, float] = {}  # date_str -> rate
        self._rates_loaded = False

    def _update_status(self, message: str, progress: int = None):
        log(message)
        if self.status_callback:
            self.status_callback(message, progress)

    async def _get_session(self):
        """Reuse aiohttp session with timeout configuration"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_rates_for_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch USD/GBP rates for a date range using time-series endpoint.

        Returns DataFrame with columns: date, usd_gbp_rate
        """
        session = await self._get_session()

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        self._update_status(f"[ExchangeRate] Fetching USD/GBP rates from {start_str} to {end_str}...")

        url = f"{self.BASE_URL}/{start_str}..{end_str}"
        params = {"base": "USD", "symbols": "GBP"}

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._update_status(f"[ExchangeRate] API error: {response.status} - {error_text}")
                    return pd.DataFrame(columns=['date', 'usd_gbp_rate'])

                data = await response.json()

                # Parse response: {"rates": {"2024-01-01": {"GBP": 0.786}, ...}}
                rates_dict = data.get('rates', {})

                rows = []
                for date_str, currencies in rates_dict.items():
                    gbp_rate = currencies.get('GBP', self.DEFAULT_RATE)
                    rows.append({
                        'date': pd.to_datetime(date_str).date(),
                        'usd_gbp_rate': gbp_rate
                    })
                    self._rate_cache[date_str] = gbp_rate

                self._rates_loaded = True
                self._update_status(f"[ExchangeRate] Loaded {len(rows)} daily exchange rates")

                df = pd.DataFrame(rows)
                if not df.empty:
                    df = df.sort_values('date').reset_index(drop=True)
                return df

        except asyncio.TimeoutError:
            self._update_status(f"[ExchangeRate] Request timeout after {self.REQUEST_TIMEOUT}s")
            return pd.DataFrame(columns=['date', 'usd_gbp_rate'])
        except aiohttp.ClientError as e:
            self._update_status(f"[ExchangeRate] Connection error: {e}")
            return pd.DataFrame(columns=['date', 'usd_gbp_rate'])
        except Exception as e:
            self._update_status(f"[ExchangeRate] Error fetching rates: {e}")
            return pd.DataFrame(columns=['date', 'usd_gbp_rate'])
        finally:
            await self.close()

    def get_rate_for_date(self, dt: datetime) -> float:
        """
        Get cached rate for a specific date.
        Falls back to nearest available date or default if not found.
        """
        if dt is None:
            return self.DEFAULT_RATE

        date_str = dt.strftime('%Y-%m-%d')

        # Direct match
        if date_str in self._rate_cache:
            return self._rate_cache[date_str]

        # Find nearest date in cache (for weekends/holidays)
        if self._rate_cache:
            cache_dates = sorted(self._rate_cache.keys())

            # Find the most recent rate before or on this date
            nearest_rate = self.DEFAULT_RATE
            for d in cache_dates:
                if d <= date_str:
                    nearest_rate = self._rate_cache[d]
                else:
                    break

            # If no earlier date found, use first available
            if nearest_rate == self.DEFAULT_RATE and cache_dates:
                nearest_rate = self._rate_cache[cache_dates[0]]

            return nearest_rate

        return self.DEFAULT_RATE

    def convert_usd_to_gbp(self, usd_amount: float, dt: datetime) -> float:
        """Convert USD amount to GBP using historical rate"""
        rate = self.get_rate_for_date(dt)
        return usd_amount * rate

    def is_loaded(self) -> bool:
        """Check if rates have been loaded"""
        return self._rates_loaded and len(self._rate_cache) > 0

    def get_cache_stats(self) -> Dict:
        """Get stats about cached rates"""
        if not self._rate_cache:
            return {"loaded": False, "count": 0}

        dates = sorted(self._rate_cache.keys())
        rates = list(self._rate_cache.values())

        return {
            "loaded": True,
            "count": len(self._rate_cache),
            "start_date": dates[0],
            "end_date": dates[-1],
            "min_rate": min(rates),
            "max_rate": max(rates),
            "avg_rate": sum(rates) / len(rates)
        }


# Singleton instance for caching across requests
_exchange_fetcher_instance: Optional[ExchangeRateFetcher] = None


def get_exchange_fetcher(status_callback=None) -> ExchangeRateFetcher:
    """Get or create singleton exchange rate fetcher"""
    global _exchange_fetcher_instance
    if _exchange_fetcher_instance is None:
        _exchange_fetcher_instance = ExchangeRateFetcher(status_callback)
    return _exchange_fetcher_instance


def reset_exchange_fetcher():
    """Reset the singleton (useful for testing or clearing cache)"""
    global _exchange_fetcher_instance
    _exchange_fetcher_instance = None


async def preload_exchange_rates(start_date: datetime, end_date: datetime, status_callback=None):
    """
    Preload exchange rates for a period (called before backtest).
    Uses singleton pattern so rates are cached for subsequent calls.
    """
    fetcher = get_exchange_fetcher(status_callback)

    # Only fetch if not already loaded or if we need a different range
    if not fetcher.is_loaded():
        await fetcher.fetch_rates_for_period(start_date, end_date)

    return fetcher
