"""
VALIDATION ROUTES
=================
API endpoints for strategy validation functionality.
Extracted from main.py for better modularity.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strategy_database import get_strategy_db

router = APIRouter(prefix="/api", tags=["validation"])

# Check database availability
try:
    get_strategy_db()
    HAS_DATABASE = True
except Exception:
    HAS_DATABASE = False


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ValidateStrategyRequest(BaseModel):
    strategy_id: int
    capital: Optional[float] = None
    position_size_pct: Optional[float] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/validate-strategy")
async def validate_strategy(request: ValidateStrategyRequest):
    """
    Validate a strategy by testing it against different time periods.
    Uses EXACT SAME configuration from original:
    - Binance data source
    - Same symbol & timeframe
    - Same TP% and SL%
    - Same entry_rule and direction
    - Same indicator parameters
    """
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        from data_fetcher import BinanceDataFetcher
        from strategy_engine import StrategyEngine

        db = get_strategy_db()
        strategy = db.get_strategy_by_id(request.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Extract ALL parameters from the strategy
        params = strategy.get('params', {})
        symbol = strategy.get('symbol', 'BTCUSDT')  # Default to USDT

        # === CHECK FOR SUPPORTED PAIRS (Binance USDT only) ===
        supported_quotes = ['USDT', 'USDC', 'BUSD']
        symbol_supported = any(symbol.endswith(q) for q in supported_quotes)
        if not symbol_supported:
            raise HTTPException(
                status_code=400,
                detail=f"Symbol {symbol} not supported. Only Binance USDT pairs are available."
            )

        # Always use Binance (only USDT/USDC/BUSD pairs are supported)
        data_source = 'binance'

        timeframe = strategy.get('timeframe', '15m')
        tp_percent = strategy.get('tp_percent', 2.0)
        sl_percent = strategy.get('sl_percent', 5.0)
        entry_rule = params.get('entry_rule', 'rsi_oversold')
        direction = params.get('direction', strategy.get('trade_mode', 'long'))

        # Get position size and original period from optimization run
        original_position_size_pct = 100.0
        original_capital = 1000.0
        original_months = 1.0
        run_id = strategy.get('optimization_run_id')
        if run_id:
            run = db.get_optimization_run_by_id(run_id)
            if run:
                original_position_size_pct = run.get('risk_percent', 100.0)
                original_capital = run.get('capital', 1000.0)
                original_months = run.get('months', 1.0)
        original_position_size_pct = params.get('position_size_pct', original_position_size_pct)
        original_capital = params.get('capital', original_capital)

        # Use request overrides if provided
        capital = request.capital if request.capital is not None else original_capital
        position_size_pct = request.position_size_pct if request.position_size_pct is not None else original_position_size_pct

        # Calculate original period display text
        original_days = int(original_months * 30)
        if original_days < 7:
            original_period_text = f"{original_days} days"
        elif original_days < 30:
            original_period_text = f"{original_days // 7} week{'s' if original_days >= 14 else ''}"
        elif original_days < 60:
            original_period_text = f"{original_days} days"
        else:
            original_period_text = f"{original_months:.1f} months".replace('.0 ', ' ')

        # Convert timeframe to minutes for data limit check
        tf_minutes = int(timeframe.replace('m', '').replace('h', '')) if 'm' in timeframe else int(timeframe.replace('h', '')) * 60

        # Binance data limits by timeframe (in days)
        data_limits = {
            1: 365,
            5: 1825,
            15: 2555,
            30: 2555,
            60: 2555,
            240: 2555,
            1440: 3650,
        }
        max_days = data_limits.get(tf_minutes, 1825)

        # Validation periods (3-year and 5-year removed - cause timeouts)
        validation_periods = [
            {"period": "1 week", "months": 0.25, "days": 7},
            {"period": "2 weeks", "months": 0.5, "days": 14},
            {"period": "1 month", "months": 1.0, "days": 30},
            {"period": "3 months", "months": 3.0, "days": 90},
            {"period": "6 months", "months": 6.0, "days": 180},
            {"period": "9 months", "months": 9.0, "days": 270},
            {"period": "1 year", "months": 12.0, "days": 365},
            {"period": "2 years", "months": 24.0, "days": 730},
        ]

        # Original metrics (baseline)
        original_metrics = {
            "total_trades": strategy.get('total_trades', 0),
            "win_rate": strategy.get('win_rate', 0),
            "total_pnl": strategy.get('total_pnl', 0),
            "profit_factor": strategy.get('profit_factor', 0),
            "max_drawdown": strategy.get('max_drawdown', 0),
        }

        # Run validation for each period
        validations = []
        for vp in validation_periods:
            if vp["days"] > max_days:
                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": None,
                    "status": "limit_exceeded",
                    "message": f"Exceeds {data_source} {tf_minutes}m limit of {max_days} days"
                })
                continue

            try:
                # Fetch fresh data for this period - always use Binance
                fetcher = BinanceDataFetcher()

                df = await fetcher.fetch_ohlcv(pair=symbol, interval=tf_minutes, months=vp["months"])

                if len(df) < 50:
                    validations.append({
                        "period": vp["period"],
                        "months": vp["months"],
                        "metrics": None,
                        "status": "insufficient_data",
                        "message": f"Only {len(df)} candles available"
                    })
                    continue

                # Create engine and run backtest
                engine = StrategyEngine(df)
                result = engine.backtest(
                    strategy=entry_rule,
                    direction=direction,
                    tp_percent=tp_percent,
                    sl_percent=sl_percent,
                    initial_capital=capital,
                    position_size_pct=position_size_pct,
                    commission_pct=0.1
                )

                # Determine status based on comparison with original
                status = "consistent"
                if result.total_trades == 0:
                    status = "no_trades"
                elif result.win_rate < original_metrics["win_rate"] * 0.8:
                    status = "degraded"
                elif result.profit_factor < original_metrics["profit_factor"] * 0.8:
                    status = "degraded"
                elif result.win_rate < original_metrics["win_rate"] * 0.95:
                    status = "minor_drop"

                # Calculate % return
                return_pct = round((result.total_pnl / capital) * 100, 1)

                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": {
                        "total_trades": result.total_trades,
                        "win_rate": round(result.win_rate, 2),
                        "total_pnl": round(result.total_pnl, 2),
                        "profit_factor": round(result.profit_factor, 2),
                        "max_drawdown": round(result.max_drawdown, 2),
                        "return_pct": return_pct,
                    },
                    "status": status,
                    "message": None
                })

            except Exception as e:
                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": None,
                    "status": "error",
                    "message": str(e)
                })

        return {
            "strategy": {
                "id": strategy.get('id'),
                "name": strategy.get('strategy_name'),
                "symbol": symbol,
                "timeframe": timeframe,
                "data_source": data_source,
                "tp_percent": tp_percent,
                "sl_percent": sl_percent,
                "entry_rule": entry_rule,
                "direction": direction,
                "capital": capital,
                "position_size_pct": position_size_pct,
            },
            "original": {
                **original_metrics,
                "period": original_period_text,
            },
            "validations": validations
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
