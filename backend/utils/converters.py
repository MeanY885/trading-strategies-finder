"""
CONVERTERS
==========
Shared conversion utilities to avoid code duplication across API routes.
"""
from typing import Dict, List, Set, Any


def dict_to_strategy_result(d: dict) -> 'StrategyResult':
    """
    Convert a dictionary to StrategyResult dataclass.

    Used when strategy data is stored as dict (e.g., in reports) but needs
    to be converted back to StrategyResult for Pine Script generation.

    Args:
        d: Dictionary containing strategy data with 'metrics' nested dict

    Returns:
        StrategyResult instance
    """
    from strategy_engine import StrategyResult

    metrics = d.get("metrics", {})

    return StrategyResult(
        strategy_name=d.get("strategy_name", "Unknown"),
        strategy_category=d.get("strategy_category", ""),
        direction=d.get("direction", "long"),
        entry_rule=d.get("entry_rule", ""),
        tp_percent=d.get("tp_percent", 2.0),
        sl_percent=d.get("sl_percent", 1.0),
        total_trades=metrics.get("total_trades", 0),
        wins=metrics.get("wins", 0),
        losses=metrics.get("losses", 0),
        win_rate=metrics.get("win_rate", 0),
        total_pnl=metrics.get("total_pnl", 0),
        total_pnl_percent=metrics.get("total_pnl_percent", 0),
        profit_factor=metrics.get("profit_factor", 0),
        max_drawdown=metrics.get("max_drawdown", 0),
        max_drawdown_percent=metrics.get("max_drawdown_pct", 0),
        avg_trade=metrics.get("avg_trade", 0),
        avg_trade_percent=metrics.get("avg_trade_percent", 0),
        composite_score=metrics.get("composite_score", 0),
        trades_list=[],  # Empty list, detailed trades not needed for Pine Script
    )


def get_pending_queue_items(
    combinations: List[Dict],
    cycle_index: int,
    running_items: List[Dict],
    max_items: int = 5,
    lookahead: int = 10
) -> List[Dict]:
    """
    Get pending queue items for UI display, excluding currently running items.

    Filters out items that are already running (by index or pair) to avoid
    showing duplicate entries in the task queue.

    Args:
        combinations: Full list of combination dicts from autonomous optimizer
        cycle_index: Current position in the cycle (starting point for pending)
        running_items: List of currently running items with 'index' and 'pair' keys
        max_items: Maximum number of pending items to return (default 5)
        lookahead: How far ahead to scan for pending items (default 10)

    Returns:
        List of pending item dicts with index, pair, period, timeframe, granularity, status
    """
    total = len(combinations)
    if total == 0:
        return []

    # Build exclusion sets
    running_indices: Set[int] = {r.get("index") for r in running_items if r.get("index") is not None}
    running_pairs: Set[str] = {r.get("pair") for r in running_items if r.get("pair")}

    pending_items: List[Dict] = []

    for i in range(cycle_index, min(cycle_index + lookahead, total)):
        # Skip if this index is already running
        if i in running_indices:
            continue

        combo = combinations[i]
        pair = combo.get("pair", "")

        # Skip if this pair is already running (even with different settings)
        if pair in running_pairs:
            continue

        pending_items.append({
            "index": i,
            "pair": pair,
            "period": combo.get("period", ""),
            "timeframe": combo.get("timeframe", ""),
            "granularity": combo.get("granularity", ""),
            "status": "pending"
        })

        if len(pending_items) >= max_items:
            break

    return pending_items
