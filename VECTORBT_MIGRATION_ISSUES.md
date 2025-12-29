# VectorBT Migration Issues Report (Commit af3a502)

**Generated:** 2025-12-29
**Reviewed by:** 20 parallel code analysis agents

---

## Executive Summary

The VectorBT integration (commit af3a502) adds significant performance improvements but introduces several issues that need addressing before going live. This report categorizes all findings by severity.

---

## CRITICAL ISSUES (Must Fix Before Go-Live)

### 1. Python/NumPy Version Incompatibility
**File:** `backend/requirements.txt`
- **numba>=0.59.0** does NOT support Python 3.13 (your current version)
- **FIX:** Change to `numba>=0.61.0`
- **vectorbt** uses deprecated `np.float_` which was removed in NumPy 2.0
- **TA-Lib** needs version `>=0.6.0` for Python 3.13/NumPy 2.x compatibility

### 2. Missing `log` Import in VectorBT Path
**File:** `backend/strategy_engine.py:4532, 4838-4839, 4966-4967, 4979-4980`
```python
# Line 4532 - log() used without import
if use_vectorbt:
    log(f"[StrategyEngine] VectorBT path selected...")  # NameError!
```
- **Impact:** `NameError: name 'log' is not defined` when VectorBT is enabled
- **FIX:** Add `from logging_config import log` at module level or inside the method

### 3. Database Connection Pool Too Small
**Files:** `backend/strategy_database.py:80-85`, `backend/async_database.py:31-37`
```python
maxconn=10  # TOO SMALL for 16-24 concurrent sessions
```
- **Impact:** Connection exhaustion causing hangs
- **FIX:** Increase to `maxconn=30-40` for 16-24 concurrent sessions

### 4. Duplicate State Management (Race Conditions)
**Files:** `backend/services/autonomous_optimizer.py:47-55` vs `backend/state.py:133-142`
- Two separate `running_optimizations` dictionaries exist
- Abort signaling in `state.py` modifies `app_state.current_optimization_status`
- But `autonomous_optimizer.py` uses its own `current_optimization_status`
- **Impact:** Race conditions, abort signals not reaching running tasks

---

## HIGH PRIORITY ISSUES

### 5. Inconsistent MAX_CONCURRENT Calculation
**File:** `backend/services/autonomous_optimizer.py:75-87` vs `:1104-1114`
- `init_async_primitives()`: Uses `cpu_count - 4` for 32 cores = **28 concurrent**
- `_run_parallel_optimizer()`: Uses `cpu_count // 4` for 32 cores = **8 concurrent**
- **Impact:** Unpredictable parallelization behavior

### 6. DataFrame Shared Without Thread-Safety
**File:** `backend/state.py:158-166`
```python
def get_dataframe(self) -> Any:
    return self.current_df  # Returns by reference, not copy!
```
- VectorBT modifies DataFrames in-place (`df.set_index('time', inplace=True)`)
- **Impact:** Race conditions during concurrent access

### 7. Mixed Sync/Async Lock Usage
**File:** `backend/services/autonomous_optimizer.py:1180-1181, 1218-1221`
```python
with running_optimizations_lock:  # sync lock in async function!
```
- **Impact:** Event loop blocking, potential deadlocks

### 8. Frontend/Backend API Mismatch
**File:** `backend/api/optimization_routes.py:32-40`
- Frontend sends `risk_percent` and `date_range`
- Backend model doesn't accept these fields
- Missing `use_vectorbt` parameter in API

### 9. VectorBT Strategy Mismatch
**File:** `backend/services/vectorbt_engine.py:108-124`
- VectorBT implements only 14 strategies
- StrategyEngine has 30+ strategies
- **Impact:** Strategies silently skipped when using VectorBT

### 10. Path Traversal Vulnerability
**File:** `backend/api/data_routes.py:296`
```python
@router.post("/load-file/{filename}")
async def load_data_file(filename: str):
    file_path = DATA_DIR / filename  # Not sanitized!
```
- **Impact:** Security vulnerability allowing file access outside DATA_DIR

---

## MEDIUM PRIORITY ISSUES

### 11. No VectorBT Status in Frontend
- Frontend has no indicator showing VectorBT status/availability
- No display of speedup metrics

### 12. WebSocket Throttle Race Condition
**File:** `backend/services/websocket_manager.py:95-100`
- Time check and update not atomic
- Can cause duplicate broadcasts within throttle interval

### 13. Cache Unbounded Growth
**File:** `backend/services/cache.py:31`
- `TTLCache` has no maximum entry limit
- **Impact:** Memory exhaustion under high load

### 14. OHLCV Cache 7-Day TTL
**File:** `backend/services/ohlcv_cache.py:185`
- Week-old data could be used for strategy validation
- **Impact:** Strategies may miss recent market conditions

### 15. Elite Validator Not Using VectorBT
**File:** `backend/services/elite_validator.py:403-413`
- Uses standard `StrategyEngine.backtest()`, not VectorBT
- Loses 100x speedup for validation

### 16. Async Pool Initialization Race
**File:** `backend/async_database.py:22-37`
- No asyncio.Lock protecting pool creation
- Multiple tasks could create duplicate pools

### 17. Missing VectorBT Fields in Database
**File:** `backend/strategy_database.py:110-191`
- VectorBTResult has fields not in database schema:
  - `total_pnl_percent`, `avg_trade`, `buy_hold_return`, `vs_buy_hold`, `consistency_score`, etc.

### 18. Hardcoded VectorBT Frequency
**File:** `backend/services/vectorbt_engine.py:399`
```python
freq='1D',  # Comment says "will be adjusted" but code doesn't
```
- **Impact:** Sharpe ratio and annualized metrics may be incorrect for non-daily data

### 19. NaN Values Not Cleaned in Data Fetcher
**File:** `backend/data_fetcher.py:354-358`
- No `dropna()` or validation before passing to VectorBT
- **Impact:** NaN values propagate through indicators, breaking signal generation

### 20. Bare Except Clauses
**Files:** Multiple locations
- `strategy_database.py:651-666`
- `elite_validator.py:318, 340, 684, 809`
- `autonomous_optimizer.py:1004-1005`
- Catches all exceptions including `SystemExit`, `KeyboardInterrupt`

---

## LOW PRIORITY ISSUES

### 21. Deprecated asyncio.get_event_loop()
**Files:** `autonomous_optimizer.py:656`, `elite_validator.py:415`, `optimization_routes.py:145`
- Deprecated in Python 3.10+, should use `asyncio.get_running_loop()`

### 22. Unused Variables
- `running_validations_lock` in elite_validator.py (never used)
- `VECTORBT_AVAILABLE` in config.py (never imported by other files)
- `ELITE_MEM_THRESHOLD` in elite_validator.py (defined but not used)

### 23. Missing CORS Middleware
**File:** `backend/main.py`
- No CORSMiddleware added to FastAPI app
- Only works if frontend/backend share same origin

### 24. Duplicate Code Patterns
- Dict-to-StrategyResult conversion duplicated in 3 places in optimization_routes.py
- Pending items logic duplicated between autonomous_routes.py and websocket_manager.py

### 25. History List O(n) Insertions
**File:** `backend/state.py:304-306`
```python
self.autonomous_history.insert(0, entry)  # O(n) operation
```
- Should use `collections.deque(maxlen=500)` instead

---

## Recommended Fix Priority

### Before Go-Live (CRITICAL):
1. Fix `numba>=0.61.0` in requirements.txt
2. Add `log` import to strategy_engine.py VectorBT paths
3. Increase database pool size to 30-40
4. Unify state management to use only `app_state`

### First Week:
5. Fix MAX_CONCURRENT calculation inconsistency
6. Add DataFrame copy in state.py `get_dataframe()`
7. Fix sync/async lock usage in autonomous_optimizer.py
8. Add missing `use_vectorbt` parameter to optimization API
9. Sanitize filename in data_routes.py
10. Handle NaN values in data_fetcher.py

### First Month:
11-25. Address remaining medium and low priority issues

---

## Quick Reference: Files Changed in af3a502

| File | Lines Added | Key Changes |
|------|-------------|-------------|
| `backend/config.py` | +34 | USE_VECTORBT config, dynamic concurrency |
| `backend/requirements.txt` | +4 | vectorbt, numba dependencies |
| `backend/services/autonomous_optimizer.py` | +21 | Dynamic MAX_CONCURRENT |
| `backend/services/elite_validator.py` | +16 | 8 max validations, CPU scaling |
| `backend/services/vectorbt_engine.py` | +662 | **New file** - VectorBT backtesting |
| `backend/services/websocket_manager.py` | +2 | Throttle 0.5s -> 1.0s |
| `backend/strategy_engine.py` | +225 | VectorBT integration, dynamic workers |

---

## Testing Recommendations

1. Run full test suite with Python 3.13 after updating numba
2. Test with 16+ concurrent sessions to verify pool sizing
3. Verify VectorBT fallback works when vectorbt not installed
4. Compare results between VectorBT and standard engine for same strategies
5. Load test WebSocket with 20+ clients
6. Verify no data loss in priority queue under high concurrency
