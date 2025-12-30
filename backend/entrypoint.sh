#!/bin/bash
# =============================================================================
# DOCKER ENTRYPOINT
# =============================================================================
# Auto-tunes optimizer settings on first boot, then starts the application.
#
# Features:
# - Runs quick benchmark (~15s) to find optimal CORES_PER_TASK
# - Caches result in .auto_tuned (skips on subsequent boots)
# - Force re-tune with: docker run -e FORCE_AUTOTUNE=1 ...
# =============================================================================

set -e

TUNE_FILE="/app/backend/.auto_tuned"

echo "================================================"
echo "BTCGBP ML Optimizer - Starting up"
echo "================================================"

# Check if we need to auto-tune
SHOULD_TUNE=false

if [ "$FORCE_AUTOTUNE" = "1" ]; then
    echo "FORCE_AUTOTUNE=1, running benchmark..."
    SHOULD_TUNE=true
elif [ ! -f "$TUNE_FILE" ]; then
    echo "First boot detected, running auto-tune..."
    SHOULD_TUNE=true
elif [ -n "$CORES_PER_TASK" ]; then
    echo "CORES_PER_TASK=${CORES_PER_TASK} set via environment, skipping auto-tune"
else
    echo "Using cached auto-tune settings from previous boot"
    cat "$TUNE_FILE" | grep -v "^#"
fi

# Run auto-tune if needed
if [ "$SHOULD_TUNE" = true ]; then
    echo ""
    echo "Running quick benchmark to optimize for this hardware..."
    echo "------------------------------------------------"

    # Run quick auto-tune (should take ~15 seconds)
    python /app/backend/tools/auto_tune.py --quick

    echo "------------------------------------------------"
    echo "Auto-tune complete!"
    echo ""
fi

# Show final config
echo ""
echo "Starting with configuration:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/{print $2}')"
if [ -f "$TUNE_FILE" ]; then
    echo "  CORES_PER_TASK: $(grep CORES_PER_TASK $TUNE_FILE | cut -d= -f2)"
fi
echo ""

# Start the main application
echo "Starting optimizer backend..."
exec python /app/backend/main.py
