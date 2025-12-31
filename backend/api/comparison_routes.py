"""
COMPARISON ROUTES
=================
API endpoints for TradingView comparison functionality.
Extracted from main.py for better modularity.
"""
import io
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd

from state import app_state
from strategy_database import get_strategy_db

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


def categorize_matches_by_tolerance(tv_trades: list, our_trades: list) -> dict:
    """
    Categorize trade matches by time tolerance windows.

    Returns:
        - exact_matches: < 1 minute difference
        - close_matches: 1 minute to 1 hour difference
        - timing_mismatches: same trade (direction/price match) but timing differs > 1 hour
        - missing_in_tv: trades in our system not found in TradingView
        - missing_in_ours: trades in TradingView not found in our system
    """
    exact_matches = []
    close_matches = []
    timing_mismatches = []

    our_trades_copy = list(our_trades)
    our_used_indices = set()

    for tv_idx, tv_trade in enumerate(tv_trades):
        tv_entry = pd.to_datetime(tv_trade['entry_time'])
        best_match = None
        best_match_idx = None
        best_time_diff = float('inf')

        for i, our_trade in enumerate(our_trades_copy):
            if i in our_used_indices:
                continue

            our_entry = pd.to_datetime(our_trade['entry_time'])
            time_diff_seconds = abs((tv_entry - our_entry).total_seconds())

            # Check if this is a better match
            if time_diff_seconds < best_time_diff:
                best_time_diff = time_diff_seconds
                best_match = our_trade
                best_match_idx = i

        if best_match is not None:
            pnl_diff = tv_trade['pnl'] - best_match.get('pnl', 0)
            pnl_diff_pct = (pnl_diff / abs(best_match['pnl']) * 100) if best_match.get('pnl', 0) != 0 else 0

            match_entry = {
                'tv_trade': tv_trade,
                'our_trade': best_match,
                'time_diff_seconds': round(best_time_diff, 1),
                'time_diff_hours': round(best_time_diff / 3600, 2),
                'pnl_diff': round(pnl_diff, 2),
                'pnl_diff_pct': round(pnl_diff_pct, 2),
                'tv_index': tv_idx,
                'our_index': best_match_idx
            }

            if best_time_diff < 60:  # Less than 1 minute
                exact_matches.append(match_entry)
                our_used_indices.add(best_match_idx)
            elif best_time_diff < 3600:  # Less than 1 hour
                close_matches.append(match_entry)
                our_used_indices.add(best_match_idx)
            elif best_time_diff < 86400:  # Less than 24 hours - potential timing mismatch
                # Check if direction matches (suggests same trade, different timing)
                tv_direction = tv_trade.get('direction', '').upper()
                our_direction = best_match.get('direction', '').upper()
                if tv_direction == our_direction:
                    timing_mismatches.append(match_entry)
                    our_used_indices.add(best_match_idx)

    # Find unmatched trades
    missing_in_ours = [tv_trades[i] for i in range(len(tv_trades))
                       if not any(m['tv_index'] == i for m in exact_matches + close_matches + timing_mismatches)]
    missing_in_tv = [our_trades_copy[i] for i in range(len(our_trades_copy))
                     if i not in our_used_indices]

    return {
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'timing_mismatches': timing_mismatches,
        'missing_in_tv': missing_in_tv,
        'missing_in_ours': missing_in_ours
    }


def detect_root_causes(analysis: dict, tv_trades: list, our_trades: list) -> list:
    """
    Analyze patterns to detect root causes of discrepancies.

    Detects:
        - Heikin Ashi/Renko chart usage (PnL inflated 30%+)
        - 1-bar timing offset (execution timing issue)
        - Commission mismatch (small consistent PnL diff)
        - Warmup period issues (first 50 trades diverge)
        - Data source differences (random mismatches)
    """
    import numpy as np

    root_causes = []
    all_matches = analysis['exact_matches'] + analysis['close_matches']

    if not all_matches:
        root_causes.append({
            'cause': 'No matching trades found',
            'confidence': 1.0,
            'evidence': 'Unable to match any trades between systems',
            'severity': 'high'
        })
        return root_causes

    # === 1. Heikin Ashi / Renko Detection ===
    # Check if TradingView PnL is consistently inflated vs our PnL
    pnl_ratios = []
    for match in all_matches:
        our_pnl = match['our_trade'].get('pnl', 0)
        tv_pnl = match['tv_trade'].get('pnl', 0)
        if our_pnl != 0 and tv_pnl != 0:
            # TV PnL / Our PnL ratio
            pnl_ratios.append(tv_pnl / our_pnl)

    if pnl_ratios:
        mean_ratio = np.mean(pnl_ratios)
        std_ratio = np.std(pnl_ratios)

        # Heikin Ashi typically inflates PnL by 30%+ with low variance
        if mean_ratio > 1.30 and std_ratio < 0.15:
            confidence = min(0.95, 0.7 + (mean_ratio - 1.30) * 2)
            root_causes.append({
                'cause': 'Heikin Ashi or Renko chart detected',
                'confidence': round(confidence, 2),
                'evidence': f'TradingView PnL is {round((mean_ratio - 1) * 100, 1)}% higher on average (std: {round(std_ratio * 100, 1)}%)',
                'severity': 'high'
            })
        elif mean_ratio < 0.70 and std_ratio < 0.15:
            confidence = min(0.95, 0.7 + (0.70 - mean_ratio) * 2)
            root_causes.append({
                'cause': 'Our system may be using smoothed data',
                'confidence': round(confidence, 2),
                'evidence': f'Our PnL is {round((1 - mean_ratio) * 100, 1)}% higher on average',
                'severity': 'high'
            })

    # === 2. One-Bar Offset Detection ===
    # Check if time differences cluster around common bar intervals
    time_diffs = [m['time_diff_seconds'] for m in all_matches]

    if time_diffs:
        # Common timeframe intervals in seconds
        timeframe_intervals = {
            60: '1m', 300: '5m', 900: '15m', 1800: '30m',
            3600: '1h', 14400: '4h', 86400: '1D'
        }

        for interval, tf_name in timeframe_intervals.items():
            # Check if most diffs are close to this interval
            close_to_interval = sum(1 for d in time_diffs if abs(d - interval) < interval * 0.1)
            if close_to_interval / len(time_diffs) > 0.5:
                confidence = round(close_to_interval / len(time_diffs), 2)
                root_causes.append({
                    'cause': f'1-bar timing offset ({tf_name} timeframe)',
                    'confidence': confidence,
                    'evidence': f'{close_to_interval}/{len(time_diffs)} trades offset by ~{tf_name}',
                    'severity': 'medium'
                })
                break

    # === 3. Commission Mismatch Detection ===
    # Small consistent PnL differences (0.1% - 5%)
    pnl_diff_pcts = [abs(m['pnl_diff_pct']) for m in all_matches if m['pnl_diff_pct'] != 0]

    if pnl_diff_pcts:
        mean_diff_pct = np.mean(pnl_diff_pcts)
        std_diff_pct = np.std(pnl_diff_pcts)

        # Commission mismatch: small consistent differences
        if 0.1 < mean_diff_pct < 5.0 and std_diff_pct < mean_diff_pct * 0.5:
            confidence = round(min(0.9, 0.6 + (1 - std_diff_pct / mean_diff_pct) * 0.3), 2)
            root_causes.append({
                'cause': 'Commission/fee mismatch',
                'confidence': confidence,
                'evidence': f'Consistent {round(mean_diff_pct, 2)}% PnL difference (std: {round(std_diff_pct, 2)}%)',
                'severity': 'low'
            })

    # === 4. Warmup Period Issue Detection ===
    # First 50 trades have significantly lower match rate
    if len(tv_trades) > 60:
        first_50_tv = tv_trades[:50]
        rest_tv = tv_trades[50:]

        # Count matches in first 50
        first_50_matches = sum(1 for m in all_matches if m['tv_index'] < 50)
        rest_matches = sum(1 for m in all_matches if m['tv_index'] >= 50)

        first_50_rate = first_50_matches / 50 if first_50_matches > 0 else 0
        rest_rate = rest_matches / len(rest_tv) if rest_tv else 0

        # Warmup issue if first 50 match rate is less than half of rest
        if first_50_rate < rest_rate * 0.5 and rest_rate > 0.5:
            confidence = round(min(0.9, (rest_rate - first_50_rate) / rest_rate), 2)
            root_causes.append({
                'cause': 'Indicator warmup period mismatch',
                'confidence': confidence,
                'evidence': f'First 50 trades: {round(first_50_rate * 100, 1)}% match rate vs {round(rest_rate * 100, 1)}% for rest',
                'severity': 'medium'
            })

    # === 5. Data Source Difference Detection ===
    # Random mismatches with no clear pattern (catch-all)
    total_trades = max(len(tv_trades), len(our_trades))
    match_rate = len(all_matches) / total_trades if total_trades > 0 else 0
    missing_count = len(analysis['missing_in_tv']) + len(analysis['missing_in_ours'])

    # If significant mismatches and no other high-confidence cause found
    high_confidence_causes = [c for c in root_causes if c['confidence'] > 0.7]

    if match_rate < 0.9 and missing_count > 5 and not high_confidence_causes:
        # Check for randomness in missing trades
        if analysis['missing_in_tv'] and analysis['missing_in_ours']:
            confidence = round(min(0.8, 0.5 + (1 - match_rate) * 0.5), 2)
            root_causes.append({
                'cause': 'Data source difference',
                'confidence': confidence,
                'evidence': f'{missing_count} unmatched trades, {round(match_rate * 100, 1)}% overall match rate',
                'severity': 'medium'
            })

    # === 6. Trade Count Mismatch ===
    count_diff = abs(len(tv_trades) - len(our_trades))
    if count_diff > 0:
        pct_diff = count_diff / max(len(tv_trades), len(our_trades)) * 100
        if pct_diff > 10:
            root_causes.append({
                'cause': 'Significant trade count difference',
                'confidence': round(min(0.9, pct_diff / 100), 2),
                'evidence': f'TV: {len(tv_trades)} trades, Ours: {len(our_trades)} trades ({round(pct_diff, 1)}% diff)',
                'severity': 'high' if pct_diff > 25 else 'medium'
            })

    # Sort by confidence (highest first)
    root_causes.sort(key=lambda x: x['confidence'], reverse=True)

    return root_causes


def generate_recommendations(root_causes: list) -> list:
    """
    Generate actionable recommendations based on detected root causes.
    """
    recommendations = []

    cause_to_recommendations = {
        'Heikin Ashi or Renko chart detected': [
            'Switch TradingView chart to standard candlesticks (not Heikin Ashi or Renko)',
            'Verify Pine Script strategy uses regular OHLC values, not smoothed candles',
            'Check if strategy.entry() is using Heikin Ashi close prices'
        ],
        'Our system may be using smoothed data': [
            'Review our indicator calculations for unintended smoothing',
            'Compare raw OHLC data between systems',
            'Check for double-smoothing in indicator chain'
        ],
        '1-bar timing offset': [
            'Check order execution timing (market vs limit orders)',
            'Verify both systems use the same bar close/open for signals',
            'Review calc_on_every_tick setting in TradingView'
        ],
        'Commission/fee mismatch': [
            'Align commission settings between TradingView and our backtester',
            'Check TradingView Strategy Properties for commission percentage',
            'Verify slippage settings match'
        ],
        'Indicator warmup period mismatch': [
            'Increase lookback period in our system to match TradingView',
            'Add explicit warmup handling for indicators with long periods',
            'Compare indicator values during first 50 bars between systems'
        ],
        'Data source difference': [
            'Verify both systems use the same data source (exchange, pair)',
            'Check for timezone differences in data timestamps',
            'Compare raw OHLCV data for specific dates where trades diverge'
        ],
        'Significant trade count difference': [
            'Check if TradingView strategy has max trades or position limits',
            'Verify date range is identical in both systems',
            'Review entry/exit conditions for edge cases'
        ],
        'No matching trades found': [
            'Verify the correct strategy rank was selected',
            'Check if date ranges overlap between CSV and backtest',
            'Ensure TradingView export contains the expected trades'
        ]
    }

    for cause in root_causes:
        cause_name = cause['cause']
        # Match partial cause names
        for key, recs in cause_to_recommendations.items():
            if key.lower() in cause_name.lower():
                for rec in recs:
                    if rec not in recommendations:
                        recommendations.append(rec)
                break

    # Add general recommendations if few specific ones found
    if len(recommendations) < 2:
        recommendations.extend([
            'Export indicator values from both systems and compare directly',
            'Run comparison on a shorter time period for detailed analysis'
        ])

    return recommendations


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


@router.get("/debug-comparison/{rank}")
async def debug_comparison(rank: int):
    """
    Enhanced root cause analysis for trade comparison discrepancies.

    Provides detailed analysis including:
    - Multiple tolerance windows for matching (exact, close, timing mismatches)
    - Root cause detection (Heikin Ashi, timing offset, commission, warmup, data source)
    - Actionable recommendations

    Args:
        rank: Strategy rank (1-10) - must have comparison data uploaded first

    Returns:
        Structured analysis with summary, categorized matches, root causes, and recommendations
    """
    # Check if comparison data exists
    if comparison_data["comparison"] is None:
        raise HTTPException(
            status_code=404,
            detail="No comparison data. Upload TradingView trade export first using /api/upload-tv-comparison/{rank}"
        )

    # Validate rank matches stored comparison
    if comparison_data["strategy_rank"] != rank:
        raise HTTPException(
            status_code=400,
            detail=f"Comparison data is for rank {comparison_data['strategy_rank']}, not rank {rank}. Upload new comparison or use correct rank."
        )

    tv_trades = comparison_data["tv_trades"]
    our_trades = comparison_data["our_trades"]

    if not tv_trades or not our_trades:
        raise HTTPException(
            status_code=400,
            detail="Missing trade data. Re-upload comparison CSV."
        )

    try:
        # Perform enhanced analysis with multiple tolerance windows
        analysis = categorize_matches_by_tolerance(tv_trades, our_trades)

        # Detect root causes of discrepancies
        root_causes = detect_root_causes(analysis, tv_trades, our_trades)

        # Generate actionable recommendations
        recommendations = generate_recommendations(root_causes)

        # Build response
        return {
            "strategy_rank": comparison_data["strategy_rank"],
            "strategy_name": comparison_data["strategy_name"],
            "summary": comparison_data["comparison"]["summary"],
            "analysis": {
                "exact_matches": analysis["exact_matches"],
                "close_matches": analysis["close_matches"],
                "timing_mismatches": analysis["timing_mismatches"],
                "missing_in_tv": analysis["missing_in_tv"],
                "missing_in_ours": analysis["missing_in_ours"],
                "match_counts": {
                    "exact": len(analysis["exact_matches"]),
                    "close": len(analysis["close_matches"]),
                    "timing_mismatch": len(analysis["timing_mismatches"]),
                    "missing_in_tv": len(analysis["missing_in_tv"]),
                    "missing_in_ours": len(analysis["missing_in_ours"])
                }
            },
            "root_causes": root_causes,
            "recommendations": recommendations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")


# =============================================================================
# ADVANCED STRATEGY DEBUGGER
# =============================================================================

@router.post("/debug-strategy/{strategy_id}")
async def debug_strategy(strategy_id: int, file: UploadFile = File(...)):
    """
    Advanced Strategy Debugger - Compare TradingView trades against a specific strategy.

    This endpoint:
    1. Fetches strategy details and trades from database
    2. Parses uploaded TradingView CSV
    3. Performs trade-by-trade comparison
    4. Detects root causes of discrepancies
    5. Generates actionable recommendations

    Args:
        strategy_id: Database ID of the strategy to debug
        file: TradingView "List of Trades" CSV export

    Returns:
        Comprehensive comparison with root cause analysis
    """
    try:
        # 1. Fetch strategy from database
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(strategy_id)

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Extract strategy details
        strategy_info = {
            "id": strategy_id,
            "name": strategy.get("strategy_name", "Unknown"),
            "symbol": strategy.get("symbol", ""),
            "timeframe": strategy.get("timeframe", ""),
            "direction": strategy.get("direction", ""),
            "entry_rule": strategy.get("entry_rule", ""),
            "tp_percent": strategy.get("tp_percent", 0),
            "sl_percent": strategy.get("sl_percent", 0),
            "total_trades": strategy.get("total_trades", 0),
            "win_rate": strategy.get("win_rate", 0),
            "profit_factor": strategy.get("profit_factor", 0),
            "total_pnl": strategy.get("total_pnl", 0),
            "max_drawdown": strategy.get("max_drawdown", 0),
        }

        # Get our trades from database
        our_trades_raw = strategy.get("trades_list", [])

        # Parse trades_list if it's a JSON string
        if isinstance(our_trades_raw, str):
            try:
                our_trades_raw = json.loads(our_trades_raw)
            except json.JSONDecodeError:
                our_trades_raw = []

        # Normalize our trades format
        our_trades = []
        for t in our_trades_raw:
            our_trades.append({
                'trade_num': t.get('trade_num', len(our_trades) + 1),
                'direction': str(t.get('direction', '')).upper(),
                'entry_time': str(t.get('entry_time', '')),
                'exit_time': str(t.get('exit_time', '')),
                'entry_price': float(t.get('entry_price', 0)),
                'exit_price': float(t.get('exit_price', 0)),
                'pnl': float(t.get('pnl', 0)),
                'pnl_pct': float(t.get('pnl_pct', t.get('pnl_percent', 0))),
                'exit_reason': t.get('exit_reason', t.get('result', '')),
            })

        # 2. Parse TradingView CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = df.columns.str.strip()

        # TradingView exports Entry + Exit rows per trade
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
            direction = 'LONG' if 'long' in str(entry_row.get('Signal', '')).lower() or 'long' in str(entry_row.get('Type', '')).lower() else 'SHORT'

            # Try to get price from various possible column names
            entry_price = 0
            exit_price = 0
            pnl = 0
            pnl_pct = 0

            # Price columns (TradingView uses different names)
            for col in ['Price GBP', 'Price', 'Price USD', 'Price USDT']:
                if col in entry_row.index:
                    try:
                        entry_price = float(entry_row.get(col, 0))
                        exit_price = float(exit_row.get(col, 0))
                        break
                    except (ValueError, TypeError):
                        continue

            # PnL columns
            for col in ['Net P&L GBP', 'Net P&L', 'Profit GBP', 'Profit', 'Net P&L USD', 'Net P&L USDT']:
                if col in exit_row.index:
                    try:
                        pnl = float(exit_row.get(col, 0))
                        break
                    except (ValueError, TypeError):
                        continue

            # PnL % columns
            for col in ['Net P&L %', 'Profit %', 'Return %']:
                if col in exit_row.index:
                    try:
                        pnl_pct = float(str(exit_row.get(col, '0')).replace('%', ''))
                        break
                    except (ValueError, TypeError):
                        continue

            tv_trades.append({
                'trade_num': int(trade_num),
                'direction': direction,
                'entry_time': str(entry_row.get('Date/Time', '')),
                'exit_time': str(exit_row.get('Date/Time', '')),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })

        # 3. Perform comparison
        comparison = compare_trades(tv_trades, our_trades)

        # 4. Run enhanced analysis with multiple tolerance windows
        analysis = categorize_matches_by_tolerance(tv_trades, our_trades)

        # 5. Detect root causes of discrepancies
        root_causes = detect_root_causes(analysis, tv_trades, our_trades)

        # 6. Generate actionable recommendations
        recommendations = generate_recommendations(root_causes)

        return {
            "strategy": strategy_info,
            "our_trades": our_trades,
            "tv_trades": tv_trades,
            "comparison": comparison,
            "analysis": {
                "exact_matches": analysis["exact_matches"],
                "close_matches": analysis["close_matches"],
                "timing_mismatches": analysis["timing_mismatches"],
                "missing_in_tv": analysis["missing_in_tv"],
                "missing_in_ours": analysis["missing_in_ours"],
                "match_counts": {
                    "exact": len(analysis["exact_matches"]),
                    "close": len(analysis["close_matches"]),
                    "timing_mismatch": len(analysis["timing_mismatches"]),
                    "missing_in_tv": len(analysis["missing_in_tv"]),
                    "missing_in_ours": len(analysis["missing_in_ours"])
                }
            },
            "root_causes": root_causes,
            "recommendations": recommendations
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during debug analysis: {str(e)}")


@router.get("/debug-strategy/{strategy_id}/info")
async def get_debug_strategy_info(strategy_id: int):
    """
    Get strategy info for debug modal (without CSV upload).
    Returns strategy details and our trades.
    """
    try:
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(strategy_id)

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Extract strategy details
        strategy_info = {
            "id": strategy_id,
            "name": strategy.get("strategy_name", "Unknown"),
            "symbol": strategy.get("symbol", ""),
            "timeframe": strategy.get("timeframe", ""),
            "direction": strategy.get("direction", ""),
            "entry_rule": strategy.get("entry_rule", ""),
            "tp_percent": strategy.get("tp_percent", 0),
            "sl_percent": strategy.get("sl_percent", 0),
            "total_trades": strategy.get("total_trades", 0),
            "win_rate": strategy.get("win_rate", 0),
            "profit_factor": strategy.get("profit_factor", 0),
            "total_pnl": strategy.get("total_pnl", 0),
            "max_drawdown": strategy.get("max_drawdown", 0),
        }

        # Get our trades from database
        our_trades_raw = strategy.get("trades_list", [])

        # Parse trades_list if it's a JSON string
        if isinstance(our_trades_raw, str):
            try:
                our_trades_raw = json.loads(our_trades_raw)
            except json.JSONDecodeError:
                our_trades_raw = []

        # Normalize our trades format
        our_trades = []
        for t in our_trades_raw:
            our_trades.append({
                'trade_num': t.get('trade_num', len(our_trades) + 1),
                'direction': str(t.get('direction', '')).upper(),
                'entry_time': str(t.get('entry_time', '')),
                'exit_time': str(t.get('exit_time', '')),
                'entry_price': float(t.get('entry_price', 0)),
                'exit_price': float(t.get('exit_price', 0)),
                'pnl': float(t.get('pnl', 0)),
                'pnl_pct': float(t.get('pnl_pct', t.get('pnl_percent', 0))),
                'exit_reason': t.get('exit_reason', t.get('result', '')),
            })

        return {
            "strategy": strategy_info,
            "our_trades": our_trades
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching strategy info: {str(e)}")


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
