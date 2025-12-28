"""
COMPARISON ROUTES
=================
API endpoints for TradingView comparison functionality.
Extracted from main.py for better modularity.
"""
import io
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd

from state import app_state

router = APIRouter(prefix="/api", tags=["comparison"])

# Store comparison results
comparison_data = {
    "tv_trades": None,
    "our_trades": None,
    "comparison": None,
    "strategy_rank": None,
    "strategy_name": None
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compare_trades(tv_trades: list, our_trades: list) -> dict:
    """
    Compare TradingView trades vs our backtester trades.

    Returns detailed comparison with:
    - Summary stats
    - Trade-by-trade matching
    - Discrepancies highlighted
    """
    # Summary comparison
    tv_total_pnl = sum(t['pnl'] for t in tv_trades) if tv_trades else 0
    our_total_pnl = sum(t['pnl'] for t in our_trades) if our_trades else 0

    tv_wins = len([t for t in tv_trades if t['pnl'] > 0])
    our_wins = len([t for t in our_trades if t.get('pnl', 0) > 0])

    tv_win_rate = (tv_wins / len(tv_trades) * 100) if tv_trades else 0
    our_win_rate = (our_wins / len(our_trades) * 100) if our_trades else 0

    # Try to match trades by time (within 1 hour tolerance)
    matched_trades = []
    unmatched_tv = []
    unmatched_ours = []

    our_trades_copy = list(our_trades)

    for tv_trade in tv_trades:
        tv_entry = pd.to_datetime(tv_trade['entry_time'])
        matched = False

        for i, our_trade in enumerate(our_trades_copy):
            our_entry = pd.to_datetime(our_trade['entry_time'])
            time_diff = abs((tv_entry - our_entry).total_seconds() / 3600)  # hours

            if time_diff <= 1:  # Within 1 hour
                pnl_diff = tv_trade['pnl'] - our_trade['pnl']
                matched_trades.append({
                    'tv_trade': tv_trade,
                    'our_trade': our_trade,
                    'time_diff_hours': round(time_diff, 2),
                    'pnl_diff': round(pnl_diff, 2),
                    'pnl_diff_pct': round((pnl_diff / abs(our_trade['pnl']) * 100) if our_trade['pnl'] != 0 else 0, 1),
                    'match_quality': 'exact' if time_diff < 0.1 else 'close'
                })
                our_trades_copy.pop(i)
                matched = True
                break

        if not matched:
            unmatched_tv.append(tv_trade)

    unmatched_ours = our_trades_copy

    return {
        "summary": {
            "tradingview": {
                "trade_count": len(tv_trades),
                "total_pnl": round(tv_total_pnl, 2),
                "win_rate": round(tv_win_rate, 1),
                "wins": tv_wins,
                "losses": len(tv_trades) - tv_wins
            },
            "our_system": {
                "trade_count": len(our_trades),
                "total_pnl": round(our_total_pnl, 2),
                "win_rate": round(our_win_rate, 1),
                "wins": our_wins,
                "losses": len(our_trades) - our_wins
            },
            "difference": {
                "trade_count_diff": len(our_trades) - len(tv_trades),
                "pnl_diff": round(our_total_pnl - tv_total_pnl, 2),
                "win_rate_diff": round(our_win_rate - tv_win_rate, 1)
            }
        },
        "matched_trades": matched_trades,
        "unmatched_tv": unmatched_tv,
        "unmatched_ours": unmatched_ours,
        "match_rate": round(len(matched_trades) / len(tv_trades) * 100, 1) if tv_trades else 0
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/upload-tv-comparison/{rank}")
async def upload_tv_comparison(rank: int, file: UploadFile = File(...)):
    """
    Upload TradingView trade list export CSV for comparison.

    TradingView format (Entry long + Exit long rows per trade):
    Trade #,Type,Date/Time,Signal,Price GBP,Position size (qty),...

    Args:
        rank: Strategy rank to compare against (1-10)
        file: TradingView exported trade list CSV
    """
    global comparison_data

    unified_status = app_state.get_unified_status()
    if unified_status.get("report") is None:
        raise HTTPException(status_code=400, detail="Run optimization first before comparing")

    top_10 = unified_status["report"].get("top_10", [])
    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    try:
        contents = await file.read()

        # Parse TradingView CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = df.columns.str.strip()

        # TradingView exports Entry + Exit rows per trade
        # Group by Trade # to combine entry/exit into single trade records
        tv_trades = []

        for trade_num in df['Trade #'].unique():
            trade_rows = df[df['Trade #'] == trade_num]

            entry_row = trade_rows[trade_rows['Type'].str.contains('Entry', case=False, na=False)]
            exit_row = trade_rows[trade_rows['Type'].str.contains('Exit', case=False, na=False)]

            if len(entry_row) == 0 or len(exit_row) == 0:
                continue

            entry_row = entry_row.iloc[0]
            exit_row = exit_row.iloc[0]

            # Extract direction from Signal or Type
            direction = 'long' if 'long' in str(entry_row.get('Signal', '')).lower() or 'long' in str(entry_row.get('Type', '')).lower() else 'short'

            tv_trades.append({
                'trade_num': int(trade_num),
                'direction': direction.upper(),
                'entry_time': str(entry_row.get('Date/Time', '')),
                'exit_time': str(exit_row.get('Date/Time', '')),
                'entry_price': float(entry_row.get('Price GBP', 0)),
                'exit_price': float(exit_row.get('Price GBP', 0)),
                'pnl': float(exit_row.get('Net P&L GBP', 0)),
                'pnl_pct': float(str(exit_row.get('Net P&L %', '0')).replace('%', '')),
                'cumulative_pnl': float(exit_row.get('Cumulative P&L GBP', 0))
            })

        # Get our trades for this strategy
        our_strategy = top_10[rank - 1]
        our_trades_list = our_strategy.get("trades_list", [])

        # Perform comparison
        comparison = compare_trades(tv_trades, our_trades_list)

        # Store for UI access
        comparison_data["tv_trades"] = tv_trades
        comparison_data["our_trades"] = our_trades_list
        comparison_data["comparison"] = comparison
        comparison_data["strategy_rank"] = rank
        comparison_data["strategy_name"] = our_strategy["strategy_name"]

        return {
            "status": "success",
            "tv_trade_count": len(tv_trades),
            "our_trade_count": len(our_trades_list),
            "comparison": comparison
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.get("/comparison")
async def get_comparison():
    """Get the current comparison data between TradingView and our system"""
    if comparison_data["comparison"] is None:
        raise HTTPException(status_code=404, detail="No comparison data. Upload TradingView trade export first.")

    return {
        "strategy_rank": comparison_data["strategy_rank"],
        "strategy_name": comparison_data["strategy_name"],
        "comparison": comparison_data["comparison"],
        "tv_trades": comparison_data["tv_trades"],
        "our_trades": comparison_data["our_trades"]
    }


@router.delete("/comparison")
async def clear_comparison():
    """Clear comparison data"""
    global comparison_data
    comparison_data = {
        "tv_trades": None,
        "our_trades": None,
        "comparison": None,
        "strategy_rank": None,
        "strategy_name": None
    }
    return {"status": "cleared"}


# =============================================================================
# INDICATOR COMPARISON ENDPOINTS
# =============================================================================

@router.get("/indicator-engines")
async def list_indicator_engines():
    """List available indicator calculation engines"""
    return {
        "engines": [
            {
                "id": "tradingview",
                "name": "TradingView Default",
                "description": "Matches TradingView's built-in ta.* functions exactly"
            },
            {
                "id": "pandas_ta",
                "name": "pandas_ta",
                "description": "pandas_ta library implementation (current default)"
            },
            {
                "id": "mihakralj",
                "name": "mihakralj/QuanTAlib",
                "description": "Mathematically rigorous implementations with proper warmup"
            }
        ],
        "current": "pandas_ta",
        "recommendation": "mihakralj - best for matching Pine Script output"
    }


@router.get("/indicator-comparison")
async def get_indicator_comparison():
    """
    Compare indicator calculations across all three engines:
    - TradingView Default (matches ta.* functions)
    - pandas_ta (current library)
    - mihakralj (mathematically rigorous)

    Fetches fresh BTCUSDT data for comparison.
    """
    try:
        from data_fetcher import BinanceDataFetcher
        from indicator_engines import MultiEngineCalculator

        # Fetch fresh data for comparison
        fetcher = BinanceDataFetcher()
        df = await fetcher.fetch_ohlcv('BTCUSDT', 60, 1)  # 1 month of 1h data

        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="Failed to fetch data for comparison")

        # Use last 100 bars for comparison
        df_sample = df.tail(100).copy()
        calc = MultiEngineCalculator(df_sample)

        # Get comparison summary
        summary = calc.get_comparison_summary()

        # Get detailed comparison for each indicator (last 20 values)
        detailed = {}

        # RSI comparison
        rsi_df = calc.compare_indicator('rsi', length=14).tail(20)
        detailed['rsi'] = {
            'values': {
                'tradingview': rsi_df['tradingview'].round(4).tolist(),
                'pandas_ta': rsi_df['pandas_ta'].round(4).tolist(),
                'mihakralj': rsi_df['mihakralj'].round(4).tolist(),
            },
            'timestamps': [str(t) for t in rsi_df.index.tolist()],
        }

        # EMA comparison
        ema_df = calc.compare_indicator('ema', length=20).tail(20)
        detailed['ema'] = {
            'values': {
                'tradingview': ema_df['tradingview'].round(2).tolist(),
                'pandas_ta': ema_df['pandas_ta'].round(2).tolist(),
                'mihakralj': ema_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in ema_df.index.tolist()],
        }

        # SMA comparison
        sma_df = calc.compare_indicator('sma', length=20).tail(20)
        detailed['sma'] = {
            'values': {
                'tradingview': sma_df['tradingview'].round(2).tolist(),
                'pandas_ta': sma_df['pandas_ta'].round(2).tolist(),
                'mihakralj': sma_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in sma_df.index.tolist()],
        }

        # ATR comparison
        atr_df = calc.compare_indicator('atr', length=14).tail(20)
        detailed['atr'] = {
            'values': {
                'tradingview': atr_df['tradingview'].round(2).tolist(),
                'pandas_ta': atr_df['pandas_ta'].round(2).tolist(),
                'mihakralj': atr_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in atr_df.index.tolist()],
        }

        # MACD comparison (special handling for tuple return)
        macd_tv = calc.macd_tradingview()
        macd_pta = calc.macd_pandas_ta()
        macd_mih = calc.macd_mihakralj()

        detailed['macd'] = {
            'values': {
                'tradingview': macd_tv[0].tail(20).round(2).tolist(),
                'pandas_ta': macd_pta[0].tail(20).round(2).tolist(),
                'mihakralj': macd_mih[0].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in macd_tv[0].tail(20).index.tolist()],
        }

        # Stochastic comparison
        stoch_tv = calc.stoch_tradingview()
        stoch_pta = calc.stoch_pandas_ta()
        stoch_mih = calc.stoch_mihakralj()

        detailed['stochastic'] = {
            'values': {
                'tradingview': stoch_tv[0].tail(20).round(2).tolist(),
                'pandas_ta': stoch_pta[0].tail(20).round(2).tolist(),
                'mihakralj': stoch_mih[0].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in stoch_tv[0].tail(20).index.tolist()],
        }

        # Bollinger Bands comparison
        bb_tv = calc.bbands_tradingview()
        bb_pta = calc.bbands_pandas_ta()
        bb_mih = calc.bbands_mihakralj()

        detailed['bollinger_upper'] = {
            'values': {
                'tradingview': bb_tv[1].tail(20).round(2).tolist(),
                'pandas_ta': bb_pta[1].tail(20).round(2).tolist(),
                'mihakralj': bb_mih[1].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in bb_tv[1].tail(20).index.tolist()],
        }

        return {
            "summary": summary,
            "detailed": detailed,
            "sample_size": len(df_sample),
            "engines": ["tradingview", "pandas_ta", "mihakralj"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing indicators: {str(e)}")
