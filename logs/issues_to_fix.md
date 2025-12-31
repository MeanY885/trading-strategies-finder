# Docker Monitoring Issues Log
Started: 2025-12-30 18:37 UTC
Monitoring period: 4 hours

---

## Issue #1: Startup Race Condition (502 Errors)
**Time:** 2025-12-30 18:37:21 UTC
**Severity:** Low
**Component:** Frontend/Backend startup timing

**Description:**
During container startup, the frontend becomes ready before the backend, causing temporary 502 errors:
```
frontend-1: connect() failed (111: Connection refused) while connecting to upstream
```

**Impact:**
- Users accessing the site immediately after startup see 502 errors for ~4 seconds
- System self-recovers once backend is ready

**Potential Fixes:**
1. Add health check dependency in docker-compose so frontend waits for backend
2. Add retry logic with backoff in nginx upstream configuration
3. Add a startup probe/readiness check to backend

---

## Observations (no issues)

### Startup Sequence
- PostgreSQL: Started and healthy at 18:29:57
- Backend: Started at 18:37:21, fully ready at 18:37:25 (~4 seconds)
- Frontend: Ready at 18:37:21
- Auto-scaling: Detected 11 cores, 10.8GB RAM -> 4 concurrent tasks
- VectorBT: Enabled and available
- WebSocket: Event loop registered successfully
- Cache warm-up: Completed successfully

### System Configuration
- Cores per task: 2
- Max concurrent: 4
- Total threads: 8
- Memory per task: 2.0GB

---

*This file will be updated as monitoring continues...*
