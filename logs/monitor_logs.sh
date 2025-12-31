#!/bin/bash
# Docker log monitoring script - runs every 5 minutes for 4 hours
# Started: $(date)

LOG_DIR="/Users/chriseddisford/Documents/TrandingView Scripts/btcgbp-ml-optimizer/logs"
LOG_FILE="$LOG_DIR/docker_monitor_$(date +%Y%m%d_%H%M%S).log"
ISSUES_FILE="$LOG_DIR/issues_found.log"

cd "/Users/chriseddisford/Documents/TrandingView Scripts/btcgbp-ml-optimizer"

echo "=== Docker Log Monitoring Started ===" | tee "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Will run for 4 hours (48 intervals at 5 minutes each)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Initialize issues file
echo "=== Issues Found During Monitoring ===" > "$ISSUES_FILE"
echo "Started: $(date)" >> "$ISSUES_FILE"
echo "" >> "$ISSUES_FILE"

for i in {1..48}; do
    echo "" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
    echo "=== Log Check #$i - $(date) ===" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"

    # Capture logs from last 5 minutes
    docker compose logs --since 5m 2>&1 >> "$LOG_FILE"

    # Check for errors and warnings
    ERRORS=$(docker compose logs --since 5m 2>&1 | grep -iE "(error|exception|fatal|failed|warning|critical)" | head -20)

    if [ -n "$ERRORS" ]; then
        echo "" >> "$ISSUES_FILE"
        echo "--- Check #$i - $(date) ---" >> "$ISSUES_FILE"
        echo "$ERRORS" >> "$ISSUES_FILE"
    fi

    # Check container health
    echo "" >> "$LOG_FILE"
    echo "--- Container Status ---" >> "$LOG_FILE"
    docker compose ps >> "$LOG_FILE" 2>&1

    # Only sleep if not the last iteration
    if [ $i -lt 48 ]; then
        sleep 300  # 5 minutes
    fi
done

echo "" >> "$LOG_FILE"
echo "=== Monitoring Complete ===" >> "$LOG_FILE"
echo "End time: $(date)" >> "$LOG_FILE"

echo "" >> "$ISSUES_FILE"
echo "=== Monitoring Complete ===" >> "$ISSUES_FILE"
echo "End time: $(date)" >> "$ISSUES_FILE"
