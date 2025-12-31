"""
PRIORITY ROUTES
===============
API endpoints for priority queue management.
Extracted from main.py for better modularity.
"""
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import AUTONOMOUS_CONFIG
from strategy_database import get_strategy_db
from services.cache import invalidate_priority_cache
from logging_config import log

router = APIRouter(prefix="/api/priority", tags=["priority"])

# Check database availability
try:
    get_strategy_db()
    HAS_DATABASE = True
except Exception:
    HAS_DATABASE = False


# =============================================================================
# REQUEST MODELS
# =============================================================================

class PriorityAddRequest(BaseModel):
    pair: str
    period_label: str
    timeframe_label: str
    granularity_label: str


class PriorityReorderRequest(BaseModel):
    order: List[int]


class PriorityListReorderRequest(BaseModel):
    order: List[int]


class GranularityRequest(BaseModel):
    granularity: str


# =============================================================================
# LEGACY PRIORITY ENDPOINTS (Single combined list)
# =============================================================================

@router.get("/list")
def get_priority_list():
    """Get all priority items ordered by position."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    items = db.get_priority_list()

    # Add display field and convert enabled to bool
    for item in items:
        item['display'] = f"{item['pair']}, {item['period_label']}, {item['timeframe_label']}, {item['granularity_label']}"
        item['enabled'] = bool(item['enabled'])

    return {"items": items, "total": len(items)}


@router.get("/available")
def get_priority_available():
    """Get available options for creating priority items."""
    config = AUTONOMOUS_CONFIG

    return {
        "pairs": config["pairs"].get("binance", []),
        "periods": config["periods"],
        "timeframes": config["timeframes"],
        "granularities": config["granularities"]
    }


@router.post("/add")
def add_priority_item(request: PriorityAddRequest):
    """Add a new priority item."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    config = AUTONOMOUS_CONFIG

    # Validate pair
    valid_pairs = config["pairs"].get("binance", [])
    if request.pair not in valid_pairs:
        raise HTTPException(status_code=400, detail=f"Invalid pair: {request.pair}")

    # Find period
    period = next((p for p in config["periods"] if p["label"] == request.period_label), None)
    if not period:
        raise HTTPException(status_code=400, detail=f"Invalid period: {request.period_label}")

    # Find timeframe
    timeframe = next((t for t in config["timeframes"] if t["label"] == request.timeframe_label), None)
    if not timeframe:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {request.timeframe_label}")

    # Find granularity
    granularity = next((g for g in config["granularities"] if g["label"] == request.granularity_label), None)
    if not granularity:
        raise HTTPException(status_code=400, detail=f"Invalid granularity: {request.granularity_label}")

    db = get_strategy_db()
    item_id = db.add_priority_item(
        pair=request.pair,
        period_label=period["label"],
        period_months=period["months"],
        timeframe_label=timeframe["label"],
        timeframe_minutes=timeframe["minutes"],
        granularity_label=granularity["label"],
        granularity_trials=granularity["n_trials"]
    )

    if item_id is None:
        return {"success": False, "message": "Combination already exists in priority list"}

    invalidate_priority_cache()
    log(f"[Priority] Added legacy item - cache invalidated")

    return {
        "success": True,
        "id": item_id,
        "message": f"Added {request.pair}, {request.period_label}, {request.timeframe_label}, {request.granularity_label}"
    }


@router.delete("/{item_id}")
def delete_priority_item(item_id: int):
    """Remove a priority item."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    success = db.delete_priority_item(item_id)

    if success:
        invalidate_priority_cache()
        log(f"[Priority] Deleted legacy item {item_id} - cache invalidated")

    return {"success": success, "message": f"Removed item {item_id}" if success else "Item not found"}


@router.post("/reorder")
def reorder_priority(request: PriorityReorderRequest):
    """Reorder priority items."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    success = db.reorder_priority_items(request.order)

    if success:
        invalidate_priority_cache()
        log(f"[Priority] Reordered legacy items - cache invalidated")

    return {"success": success, "message": f"Reordered {len(request.order)} items"}


@router.patch("/{item_id}/toggle")
def toggle_priority_item(item_id: int):
    """Toggle enabled status of a priority item."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    new_status = db.toggle_priority_item(item_id)

    if new_status is None:
        raise HTTPException(status_code=404, detail="Item not found")

    invalidate_priority_cache()
    log(f"[Priority] Toggled legacy item {item_id} - cache invalidated")

    return {"success": True, "enabled": new_status}


@router.post("/populate-defaults")
def populate_default_priority():
    """Populate priority list with a sensible default ordering."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    config = AUTONOMOUS_CONFIG
    db = get_strategy_db()

    added = 0
    # Priority: Start with 0.5% granularity (fastest), 15m timeframe, 1 month period
    priority_combinations = [
        ("15m", "1 month", "0.5%"),
        ("15m", "1 week", "0.5%"),
        ("5m", "1 month", "0.5%"),
        ("30m", "1 month", "0.5%"),
        ("1h", "1 month", "0.5%"),
        ("4h", "1 month", "0.5%"),
    ]

    for tf_label, period_label, gran_label in priority_combinations:
        period = next((p for p in config["periods"] if p["label"] == period_label), None)
        timeframe = next((t for t in config["timeframes"] if t["label"] == tf_label), None)
        granularity = next((g for g in config["granularities"] if g["label"] == gran_label), None)

        if period and timeframe and granularity:
            for pair in config["pairs"].get("binance", []):
                item_id = db.add_priority_item(
                    pair=pair,
                    period_label=period["label"],
                    period_months=period["months"],
                    timeframe_label=timeframe["label"],
                    timeframe_minutes=timeframe["minutes"],
                    granularity_label=granularity["label"],
                    granularity_trials=granularity["n_trials"]
                )
                if item_id:
                    added += 1

    return {"success": True, "added": added}


@router.post("/populate-all")
def populate_all_priority():
    """Populate priority list with ALL possible combinations."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    config = AUTONOMOUS_CONFIG
    db = get_strategy_db()

    added = 0
    for granularity in config["granularities"]:
        for timeframe in config["timeframes"]:
            for period in config["periods"]:
                for pair in config["pairs"].get("binance", []):
                    item_id = db.add_priority_item(
                        pair=pair,
                        period_label=period["label"],
                        period_months=period["months"],
                        timeframe_label=timeframe["label"],
                        timeframe_minutes=timeframe["minutes"],
                        granularity_label=granularity["label"],
                        granularity_trials=granularity["n_trials"]
                    )
                    if item_id:
                        added += 1

    return {"success": True, "added": added}


@router.post("/clear")
def clear_priority():
    """Clear all priority items."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    count = db.clear_priority_items()

    invalidate_priority_cache()
    log(f"[Priority] Cleared {count} legacy items - cache invalidated")

    return {"success": True, "deleted": count}


# =============================================================================
# NEW 4-LIST PRIORITY SYSTEM ENDPOINTS
# =============================================================================

@router.get("/lists")
def get_priority_lists():
    """Get all four priority lists and settings - optimized single DB call."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    config = AUTONOMOUS_CONFIG

    # Get all lists in one optimized call
    data = db.get_all_priority_lists()

    # Check if lists are populated, if not populate with defaults
    if not data['populated']:
        db.reset_priority_pairs(config["pairs"].get("binance", []))
        db.reset_priority_periods(config["periods"])
        db.reset_priority_timeframes(config["timeframes"])
        db.reset_priority_granularities(config["granularities"])
        # Re-fetch after populating
        data = db.get_all_priority_lists()

    # Convert enabled to bool
    for p in data['pairs']:
        p['enabled'] = bool(p['enabled'])
    for p in data['periods']:
        p['enabled'] = bool(p['enabled'])
    for t in data['timeframes']:
        t['enabled'] = bool(t['enabled'])
    for g in data['granularities']:
        g['enabled'] = bool(g['enabled'])

    return {
        "pairs": data['pairs'],
        "periods": data['periods'],
        "timeframes": data['timeframes'],
        "granularities": data['granularities']
    }


@router.post("/{list_type}/reorder")
def reorder_priority_list(list_type: str, request: PriorityListReorderRequest):
    """Reorder items in a specific list."""
    if list_type not in ['pairs', 'periods', 'timeframes', 'granularities']:
        raise HTTPException(status_code=400, detail="Invalid list type")

    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    success = db.reorder_priority_list_new(list_type, request.order)

    if success:
        invalidate_priority_cache()
        log(f"[Priority] Reordered {list_type} - cache invalidated")

    return {"success": success}


@router.patch("/{list_type}/{item_id}/toggle")
def toggle_priority_list_item(list_type: str, item_id: int):
    """Toggle enabled status in a specific list."""
    if list_type not in ['pairs', 'periods', 'timeframes', 'granularities']:
        raise HTTPException(status_code=400, detail="Invalid list type")

    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    new_status = db.toggle_priority_list_item(list_type, item_id)

    if new_status is None:
        raise HTTPException(status_code=404, detail="Item not found")

    invalidate_priority_cache()
    log(f"[Priority] Toggled {list_type} item {item_id} to {new_status} - cache invalidated")

    return {"success": True, "enabled": new_status}


@router.post("/granularity")
def set_priority_granularity(request: GranularityRequest):
    """Set the global granularity setting."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    db.set_priority_setting("granularity", request.granularity)

    return {"success": True}


@router.post("/reset-defaults")
def reset_priority_defaults():
    """Reset all priority settings to defaults."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    config = AUTONOMOUS_CONFIG

    db.reset_priority_pairs(config["pairs"].get("binance", []))
    db.reset_priority_periods(config["periods"])
    db.reset_priority_timeframes(config["timeframes"])
    db.reset_priority_granularities(config["granularities"])

    invalidate_priority_cache()
    log("[Priority] Reset to defaults - cache invalidated")

    return {"success": True}


@router.post("/enable-all")
def enable_all_priority():
    """Enable all items in all lists."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    db.enable_all_priority_items()

    invalidate_priority_cache()
    log("[Priority] Enabled all items - cache invalidated")

    return {"success": True}


@router.post("/disable-all")
def disable_all_priority():
    """Disable all items in all lists."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    db = get_strategy_db()
    db.disable_all_priority_items()

    invalidate_priority_cache()
    log("[Priority] Disabled all items - cache invalidated")

    return {"success": True}
