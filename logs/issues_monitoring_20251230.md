# Docker Container Monitoring - Issues Found
**Started:** 2025-12-30 17:37:40 GMT
**Duration:** 4 hours (until ~21:37:40 GMT)
**Log File:** `monitoring_4hour_20251230_173806.log`

---

## Critical Issues

### 1. Task Stalled and Aborted - BTCUSDT 15m
**Time:** 17:39:13
**Severity:** HIGH
**Component:** ProgressWatchdog / Autonomous Optimizer

**Details:**
- Task `auto_0_BTCUSDT_15m` stalled at 31.0% progress
- Watchdog detected 50 consecutive measurements without progress change
- Velocity trend reported as "stalled"
- Task was aborted by the watchdog

**Log excerpt:**
```
[17:39:13] [ProgressWatchdog] WARNING: auto_0_BTCUSDT_15m appears stalled
[17:39:13] [ProgressWatchdog]   Progress: 31.0%
[17:39:13] [ProgressWatchdog]   Unchanged for: 50 measurements
[17:39:13] [ProgressWatchdog]   Velocity trend: stalled
[17:39:13] [ProgressWatchdog] ABORT: auto_0_BTCUSDT_15m - Task stalled at 31.0% for 50 consecutive measurements
```

**Potential causes to investigate:**
- Thread pool deadlock or resource contention
- Memory pressure causing slowdown
- VectorBT computation hitting edge case
- Progress reporting issue (progress may be advancing but not reported)

---

## Minor Issues

### 2. Connection Refused on Startup
**Time:** 17:37:33
**Severity:** LOW (Expected during startup)
**Component:** Frontend -> Backend

**Details:**
- Frontend attempted to connect to backend health endpoint before backend was ready
- This is expected during container startup due to `depends_on` only waiting for container start, not readiness

**Log excerpt:**
```
frontend-1  | [error] connect() failed (111: Connection refused) while connecting to upstream
```

**Note:** This resolved itself once backend was ready. No action needed unless it persists.

---

## Observations

### Healthy Operations
- Elite validation service started successfully at 17:39:42
- ETHUSDT 15m task started successfully at 17:39:25
- VectorBT engine initialized correctly with 720 bars
- Throughput system is actively adjusting concurrency (3 -> 4 workers)
- WebSocket broadcasts working correctly
- Binance API connections successful

### Performance Metrics Observed
- Throughput: ~32-35 combinations/sec
- CPU usage: ~9.3% at spawn time
- Available memory: 7.1GB
- Max concurrent set to 4 for autonomous, 1 for elite validation

---

## Issues Added During 4-Hour Monitoring

*This section will be updated as new issues are discovered*

| Time | Issue | Severity | Notes |
|------|-------|----------|-------|
| 17:39:13 | BTCUSDT task aborted (stalled at 31%) | HIGH | See Issue #1 |
| | | | |

---

## Recommendations for Review

1. **Investigate BTCUSDT stall** - Check why task stalled at exactly 31%. Could be:
   - Specific strategy combination causing infinite loop
   - Progress callback not firing
   - Resource exhaustion at that point

2. **Consider health check for backend** - Add proper healthcheck to docker-compose for backend so frontend waits for actual readiness

3. **Monitor for pattern** - Watch if 31% stall recurs with other tasks
