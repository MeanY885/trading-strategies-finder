        // Strategy History State
        let historyStrategies = [];
        let historySortBy = 'composite_score';
        let historySortOrder = 'desc';
        let historyFilters = {
            symbol: '',
            timeframe: '',
            period: ''
        };
        let historyInitialized = false;
        let validationInitialized = false;
        let eliteInitialized = false;

        // Elite Strategies State
        let eliteStrategiesData = [];
        let eliteFilteredData = [];
        let eliteSortBy = 'score';
        let eliteSortOrder = 'desc';
        let eliteFilters = {
            search: '',
            symbol: '',
            status: '',
            direction: ''
        };
        let autonomousInitialized = false;
        let autonomousPollingInterval = null;

        // Priority Queue State (3-list system - actual state in priority section)
        let priorityInitialized = false;

        // Expanded row tracking - prevents O(n²) querySelectorAll loops
        let currentExpandedHistoryRow = null;
        let currentExpandedHistoryId = null;
        let currentExpandedEliteRow = null;
        let currentExpandedEliteId = null;

        // Queue render batching - prevents layout thrashing
        let pendingQueueRender = null;
        let queueRenderScheduled = false;

        // Strategy result batching - accumulates results for batch DOM insert
        let pendingStrategyResults = [];
        let strategyResultsScheduled = false;

        // =============================================================================
        // HELPER FUNCTIONS
        // =============================================================================

        /**
         * Format seconds into a human-readable ETA string
         * @param {number} seconds - Total seconds remaining
         * @returns {string} Formatted string like "15m 22s" or "1h 5m"
         */
        function formatETA(seconds) {
            if (seconds === null || seconds === undefined || seconds < 0) {
                return 'calculating...';
            }

            seconds = Math.round(seconds);

            if (seconds < 60) {
                return `${seconds}s`;
            }

            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;

            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            }

            return `${minutes}m ${secs}s`;
        }

        // =============================================================================
        // WEBSOCKET CLIENT - Real-time updates (replaces polling)
        // =============================================================================

        let wsConnection = null;
        let wsReconnectAttempts = 0;
        let wsMaxReconnectAttempts = 10;
        let wsReconnectDelay = 1000; // Start with 1 second
        let wsConnected = false;
        let wsPingInterval = null;  // Track ping interval to prevent memory leak

        // UI update throttling - prevent excessive DOM updates from rapid WebSocket messages
        const uiThrottleState = {
            pending: {},       // Pending messages per type
            timeouts: {},      // Timeout IDs per type
            minInterval: 100   // Minimum ms between UI updates per type (10 updates/sec max)
        };

        // Types that should be throttled (frequent status updates)
        const throttledMessageTypes = new Set([
            'autonomous_status',
            'elite_status',
            'optimization_status',
            'data_status'
            // Note: strategy_result uses its own batching (see handleStrategyResult)
        ]);

        function initWebSocket() {
            // Clear existing ping interval before creating new one
            if (wsPingInterval) {
                clearInterval(wsPingInterval);
                wsPingInterval = null;
            }

            // Determine WebSocket URL based on current location
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/status`;

            console.log('[WebSocket] Connecting to:', wsUrl);
            updateConnectionStatus('connecting');

            try {
                wsConnection = new WebSocket(wsUrl);

                wsConnection.onopen = function(event) {
                    console.log('[WebSocket] Connected');
                    wsConnected = true;
                    wsReconnectAttempts = 0;
                    wsReconnectDelay = 1000;
                    updateConnectionStatus('connected');

                    // Stop any legacy polling
                    stopPolling();
                    stopAutonomousPolling();
                };

                wsConnection.onmessage = function(event) {
                    // Handle plain text responses (like "pong")
                    if (event.data === 'pong') {
                        return; // Keepalive response, ignore
                    }

                    try {
                        const message = JSON.parse(event.data);
                        handleWebSocketMessage(message);
                    } catch (e) {
                        console.error('[WebSocket] Error parsing message:', e, event.data);
                    }
                };

                wsConnection.onerror = function(error) {
                    console.error('[WebSocket] Error:', error);
                    updateConnectionStatus('error');
                };

                wsConnection.onclose = function(event) {
                    console.log('[WebSocket] Disconnected, code:', event.code);
                    wsConnected = false;
                    updateConnectionStatus('disconnected');

                    // Always reconnect with exponential backoff (no polling fallback)
                    wsReconnectAttempts++;
                    const delay = Math.min(wsReconnectDelay * Math.pow(1.5, wsReconnectAttempts - 1), 30000);
                    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${wsReconnectAttempts})`);
                    setTimeout(initWebSocket, delay);
                };

                // Send periodic pings to keep connection alive
                wsPingInterval = setInterval(() => {
                    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
                        wsConnection.send('ping');
                    }
                }, 25000);

            } catch (e) {
                console.error('[WebSocket] Failed to create connection:', e);
                updateConnectionStatus('error');
                // Retry connection after delay (no polling fallback)
                setTimeout(initWebSocket, 5000);
            }
        }

        function handleWebSocketMessage(message) {
            const type = message.type;
            console.log('[WebSocket] Message received, type:', type);

            // First check if this is a response to a pending request
            if (message.id && handleWsResponse(message)) {
                return; // Handled as request response
            }

            // Apply throttling for frequent message types to prevent UI thrashing
            if (throttledMessageTypes.has(type)) {
                // Store latest message (overwrites any pending)
                uiThrottleState.pending[type] = message;

                // If we already have a timeout scheduled, let it handle this
                if (uiThrottleState.timeouts[type]) {
                    return;
                }

                // Schedule the actual UI update
                uiThrottleState.timeouts[type] = setTimeout(() => {
                    const pendingMessage = uiThrottleState.pending[type];
                    uiThrottleState.timeouts[type] = null;
                    uiThrottleState.pending[type] = null;

                    if (pendingMessage) {
                        processWebSocketMessage(pendingMessage);
                    }
                }, uiThrottleState.minInterval);
                return;
            }

            // Non-throttled messages get immediate processing
            processWebSocketMessage(message);
        }

        // Actual message processing (called after throttling)
        function processWebSocketMessage(message) {
            const type = message.type;

            switch(type) {
                case 'full_state':
                    // Initial state on connect - update all UI components
                    if (message.data) updateDataStatus(message.data);
                    if (message.optimization) updateOptimizationStatus(message.optimization);
                    if (message.autonomous) updateAutonomousUI(message.autonomous);
                    if (message.elite) updateEliteUI(message.elite);
                    // Queue data for autonomous tab
                    if (message.queue) {
                        cachedQueueData = message.queue;
                        renderTaskQueueFromCache(message.queue);
                    }
                    break;

                case 'data_status':
                    if (message.data) updateDataStatus(message.data);
                    break;

                case 'optimization_status':
                    if (message.optimization) updateOptimizationStatus(message.optimization);
                    break;

                case 'autonomous_status':
                    console.log('[WebSocket] Received autonomous_status:', message);
                    if (message.autonomous) updateAutonomousUI(message.autonomous);
                    // Queue data is now included in autonomous_status broadcasts
                    if (message.queue) {
                        cachedQueueData = message.queue;
                        renderTaskQueueFromCache(message.queue);
                    }
                    break;

                case 'elite_status':
                    if (message.elite) updateEliteUI(message.elite);
                    break;

                case 'strategy_result':
                    // Individual strategy result from optimization
                    if (message.result) handleStrategyResult(message.result);
                    break;

                case 'heartbeat':
                    // Server heartbeat - connection is alive
                    break;

                // Data responses (when not matched by request ID)
                case 'strategies_data':
                case 'elite_data':
                case 'queue_data':
                case 'priority_data':
                case 'db_stats':
                case 'error':
                    // These are handled by handleWsResponse above
                    break;

                default:
                    console.log('[WebSocket] Unknown message type:', type, message);
            }
        }

        function updateConnectionStatus(status) {
            const indicator = document.getElementById('wsConnectionStatus');
            if (!indicator) return;

            const statusMap = {
                'connecting': { text: '🔄 Connecting...', class: 'ws-connecting' },
                'connected': { text: '🟢 Live', class: 'ws-connected' },
                'disconnected': { text: '🔴 Disconnected', class: 'ws-disconnected' },
                'error': { text: '⚠️ Error', class: 'ws-error' }
            };

            const info = statusMap[status] || { text: status, class: '' };
            indicator.textContent = info.text;
            indicator.className = 'ws-status ' + info.class;
        }

        // Helper to update optimization status from WebSocket
        function updateOptimizationStatus(status) {
            if (!status) return;

            // Update progress bar if running
            const progressBar = document.getElementById('unifiedProgress');
            const progressFill = document.getElementById('unifiedProgressFill');
            const statusText = document.getElementById('unifiedStatusText');

            if (progressFill && status.progress !== undefined) {
                progressFill.style.width = `${status.progress}%`;
            }
            if (statusText && status.message) {
                statusText.textContent = status.message;
            }

            // Update button state
            const runBtn = document.getElementById('runUnifiedBtn');
            if (runBtn) {
                runBtn.disabled = status.running;
                runBtn.innerHTML = status.running ?
                    '<span class="spinner"></span> Running...' :
                    '🔬 Find Strategies';
            }
        }

        // Helper to update autonomous UI from WebSocket
        function updateAutonomousUI(status) {
            if (!status) return;

            // Update badge
            const badge = document.getElementById('autonomousStatusBadge');
            const message = document.getElementById('autonomousStatusMessage');
            const parallelCount = status.parallel_count || 0;
            const maxParallel = status.max_parallel || 1;

            if (badge) {
                if (parallelCount > 0) {
                    badge.className = 'status-badge success';
                    badge.textContent = `Running ${parallelCount}`;
                } else if (status.running) {
                    badge.className = 'status-badge success';
                    badge.textContent = 'Running';
                } else if (status.paused) {
                    badge.className = 'status-badge warning';
                    badge.textContent = 'Paused';
                } else if (status.enabled && status.auto_running) {
                    badge.className = 'status-badge success';
                    badge.textContent = 'Active';
                } else if (status.enabled) {
                    badge.className = 'status-badge neutral';
                    badge.textContent = 'Waiting';
                } else {
                    badge.className = 'status-badge neutral';
                    badge.textContent = 'Off';
                }
            }

            if (message) {
                message.textContent = status.message || 'Not running';
            }

            // Update toggle switch state
            const toggle = document.getElementById('autonomousToggle');
            if (toggle && toggle.checked !== status.enabled) {
                toggle.checked = status.enabled;
                const label = document.getElementById('autonomousToggleLabel');
                if (label) {
                    label.textContent = status.enabled ? 'ON' : 'OFF';
                    label.style.color = status.enabled ? 'var(--success)' : 'var(--text-secondary)';
                }
            }

            // Update cycle progress (shown inline in status bar)
            const cycleProgress = document.getElementById('autonomousCycleProgress');
            if (cycleProgress && status.running) {
                const completed = status.completed_count || 0;
                const total = status.total_combinations || 0;
                if (total > 0) {
                    const pct = Math.round((completed / total) * 100);
                    cycleProgress.textContent = `${completed} / ${total} (${pct}%)`;
                }
            } else if (cycleProgress) {
                cycleProgress.textContent = '';
            }

            // Update summary stats
            const autoCompletedCount = document.getElementById('autoCompletedCount');
            const autoSkippedCount = document.getElementById('autoSkippedCount');
            const autoErrorCount = document.getElementById('autoErrorCount');
            const autoTotalCombinations = document.getElementById('autoTotalCombinations');
            const autoLastCompleted = document.getElementById('autoLastCompleted');

            if (autoCompletedCount) autoCompletedCount.textContent = status.completed_count || 0;
            if (autoSkippedCount) autoSkippedCount.textContent = status.skipped_count || 0;
            if (autoErrorCount) autoErrorCount.textContent = status.error_count || 0;
            if (autoTotalCombinations) autoTotalCombinations.textContent = status.total_combinations || 0;
            if (autoLastCompleted) {
                autoLastCompleted.textContent = status.last_completed_at
                    ? new Date(status.last_completed_at).toLocaleTimeString()
                    : '-';
            }

            // Note: Queue rendering is handled separately via renderTaskQueueFromCache
            // when queue data is received in the autonomous_status message

            // Update history table when status changes
            updateAutonomousHistory();
        }

        // Helper to update elite UI from WebSocket
        function updateEliteUI(status) {
            if (!status) return;

            // Only update if elite tab is visible
            const eliteTab = document.getElementById('elite-tab');
            if (!eliteTab || eliteTab.style.display === 'none') return;

            // Update progress indicator (text badge)
            const progressEl = document.getElementById('eliteProgress');
            const progressCount = document.getElementById('eliteProgressCount');
            // Update queue elements
            const queueContainer = document.getElementById('eliteValidationQueue');
            const queueBody = document.getElementById('eliteQueueBody');
            const queueParallelCount = document.getElementById('eliteQueueParallelCount');

            if (status.running && status.total > 0) {
                // Show queue when validating
                if (queueContainer) queueContainer.style.display = 'block';

                // Update progress count in status bar
                if (progressCount) {
                    progressCount.textContent = `${status.processed} / ${status.total}`;
                }

                // Update parallel count display with resource info
                if (queueParallelCount) {
                    const parallelCount = status.parallel_count || 0;
                    const maxParallel = status.max_parallel || 1;
                    const cpuPct = status.cpu_percent ? `CPU: ${Math.round(status.cpu_percent)}%` : '';
                    // Show memory available for validation (after reserved for DB/Frontend)
                    const memForVal = status.memory_for_validation_gb;
                    const memGb = memForVal !== undefined
                        ? `Avail: ${memForVal.toFixed(1)}GB`
                        : (status.memory_available_gb ? `Mem: ${status.memory_available_gb.toFixed(1)}GB` : '');
                    const resourceInfo = [cpuPct, memGb].filter(x => x).join(' | ');
                    queueParallelCount.textContent = `${parallelCount}/${maxParallel} parallel${resourceInfo ? ' | ' + resourceInfo : ''}`;
                }

                // Render validation queue
                if (queueBody) {
                    renderEliteValidationQueue(queueBody, status);
                }

                if (status.paused) {
                    // Paused state
                    if (progressEl) {
                        progressEl.textContent = `⏸ Paused`;
                        progressEl.className = 'elite-progress paused';
                    }
                } else {
                    // Active validation
                    if (progressEl) {
                        progressEl.textContent = `⚡ Validating`;
                        progressEl.className = 'elite-progress running';
                    }
                }
            } else {
                // Hide queue when not running
                if (queueContainer) queueContainer.style.display = 'none';
                if (progressCount) progressCount.textContent = '';

                if (progressEl) {
                    if (status.message) {
                        progressEl.textContent = `✓ ${status.message}`;
                    } else {
                        progressEl.textContent = '✓ Auto-validation active';
                    }
                    progressEl.className = 'elite-progress';
                }
            }

            // Refresh elite data periodically (not on every message to avoid overload)
            if (status.running === false) {
                loadEliteData();
            }
        }

        // Render the elite validation queue with progress bars (like Auto Optimizer)
        function renderEliteValidationQueue(container, status) {
            const running = status.running_validations || [];
            const pending = status.pending_queue || [];

            if (running.length === 0 && pending.length === 0) {
                container.innerHTML = `
                    <div style="padding: 1rem; text-align: center; color: var(--text-muted);">
                        No pending validations
                    </div>
                `;
                return;
            }

            let html = '';

            // Format running item WITH progress bar (full-width grid layout like optimizer queue)
            const formatRunningItem = (item) => {
                const progress = item.progress || 0;
                const currentPeriod = item.current_period || '...';
                const periodIndex = item.period_index !== undefined ? item.period_index + 1 : 0;
                const totalPeriods = item.total_periods || 10;
                const periodProgress = `${currentPeriod} - ${periodIndex}/${totalPeriods} (${progress}%)`;

                return `
                    <div class="task-queue-item in-progress">
                        <span class="task-status-icon"><span class="spinner"></span></span>
                        <span>${item.symbol || '?'}</span>
                        <span>${item.timeframe || '?'}</span>
                        <span title="${item.name || ''}">${(item.name || '?').substring(0, 12)}</span>
                        <span>${item.tp_sl || '-'}</span>
                        <div class="task-progress">
                            <div class="task-progress-bar">
                                <div class="task-progress-fill" style="width: ${progress}%"></div>
                            </div>
                            <span class="task-progress-text">${periodProgress}</span>
                        </div>
                    </div>
                `;
            };

            // Format pending item (no progress bar, full-width grid layout)
            const formatPendingItem = (item) => {
                return `
                    <div class="task-queue-item pending">
                        <span class="task-status-icon pending">○</span>
                        <span>${item.symbol || '?'}</span>
                        <span>${item.timeframe || '?'}</span>
                        <span title="${item.name || ''}">${(item.name || '?').substring(0, 12)}</span>
                        <span>${item.tp_sl || '-'}</span>
                        <span class="task-result">Pending</span>
                    </div>
                `;
            };

            // Running validations with progress bars
            for (const item of running) {
                html += formatRunningItem(item);
            }

            // Pending validations
            for (const item of pending) {
                html += formatPendingItem(item);
            }

            // Show remaining count if more pending (use same style as optimizer queue)
            const totalPending = (status.total || 0) - (status.processed || 0) - running.length;
            if (totalPending > pending.length) {
                html += `
                    <div class="task-queue-more">
                        +${totalPending - pending.length} more pending...
                    </div>
                `;
            }

            container.innerHTML = html;
        }

        // Handle individual strategy results (replaces SSE)
        // Uses batching to prevent DOM thrashing - accumulates results then batch inserts
        function handleStrategyResult(result) {
            if (!result.strategy_name) return;

            // Accumulate result
            pendingStrategyResults.push(result);

            // Schedule batch render if not already scheduled
            if (strategyResultsScheduled) return;
            strategyResultsScheduled = true;

            requestAnimationFrame(() => {
                strategyResultsScheduled = false;
                const results = pendingStrategyResults;
                pendingStrategyResults = [];

                if (results.length === 0) return;

                const resultsBody = document.getElementById('streamingResultsBody');
                if (!resultsBody) return;

                // Build all rows as DocumentFragment (single DOM insert)
                const fragment = document.createDocumentFragment();
                results.forEach(result => {
                    const row = document.createElement('tr');
                    const metrics = result.metrics || {};
                    const pnl = metrics.total_pnl || 0;
                    const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';

                    row.innerHTML = `
                        <td>${result.strategy_name || '-'}</td>
                        <td>${result.direction || '-'}</td>
                        <td>${metrics.win_rate?.toFixed(1) || '-'}%</td>
                        <td>${metrics.profit_factor?.toFixed(2) || '-'}</td>
                        <td class="${pnlClass}">£${pnl.toFixed(2)}</td>
                    `;
                    fragment.appendChild(row);
                });

                // Single DOM operation: insert all new rows at top
                resultsBody.insertBefore(fragment, resultsBody.firstChild);

                // Batch trim excess rows
                const maxRows = 50;
                while (resultsBody.children.length > maxRows) {
                    resultsBody.removeChild(resultsBody.lastChild);
                }
            });
        }

        // =============================================================================
        // WEBSOCKET DATA REQUESTS - Request data via WebSocket instead of HTTP
        // =============================================================================

        let wsRequestId = 0;
        const wsPendingRequests = new Map();

        function wsRequest(type, timeout = 30000) {
            return new Promise((resolve, reject) => {
                if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket not connected'));
                    return;
                }

                const id = ++wsRequestId;
                const timer = setTimeout(() => {
                    wsPendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }, timeout);

                wsPendingRequests.set(id, { resolve, reject, timer });

                wsConnection.send(JSON.stringify({ type, id }));
            });
        }

        function handleWsResponse(message) {
            const id = message.id;
            if (id && wsPendingRequests.has(id)) {
                const { resolve, reject, timer } = wsPendingRequests.get(id);
                clearTimeout(timer);
                wsPendingRequests.delete(id);
                // Check if this is an error response
                if (message.type === 'error') {
                    reject(new Error(message.error || 'Unknown server error'));
                } else {
                    resolve(message);
                }
                return true;
            }
            return false;
        }

        // Cache for data to avoid re-requests
        let cachedStrategies = null;
        let cachedEliteData = null;
        let cachedPriorityData = null;
        let cachedQueueData = null;

        // WebSocket data loading functions
        async function loadStrategiesViaWs() {
            try {
                const response = await wsRequest('get_strategies', 30000);
                cachedStrategies = response.data;
                return response.data;
            } catch (e) {
                console.error('[WS] Failed to load strategies:', e);
                throw e;
            }
        }

        async function loadEliteViaWs() {
            try {
                const response = await wsRequest('get_elite', 30000);
                cachedEliteData = response;
                return response;
            } catch (e) {
                console.error('[WS] Failed to load elite data:', e);
                throw e;
            }
        }

        async function loadQueueViaWs() {
            try {
                const response = await wsRequest('get_queue', 15000);
                cachedQueueData = response.data;
                return response.data;
            } catch (e) {
                console.error('[WS] Failed to load queue:', e);
                throw e;
            }
        }

        async function loadPriorityViaWs() {
            try {
                // Use longer timeout - database may be busy during optimization
                const response = await wsRequest('get_priority', 15000);
                cachedPriorityData = response.data;
                return response.data;
            } catch (e) {
                console.error('[WS] Failed to load priority:', e);
                // WebSocket-only - retry rather than HTTP fallback
                throw e;
            }
        }

        async function loadDbStatsViaWs() {
            try {
                const response = await wsRequest('get_db_stats');
                return response.data;
            } catch (e) {
                console.error('[WS] Failed to load db stats:', e);
                return { total_strategies: 0, unique_symbols: 0, unique_timeframes: 0, elite_count: 0 };
            }
        }

        // Tab Navigation
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.style.display = 'none';
            });

            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName + '-tab').style.display = 'block';
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

            // Initialize history tab on first view
            if (tabName === 'history' && !historyInitialized) {
                initHistoryTab();
            }

            // Initialize validation tab on first view
            if (tabName === 'validation' && !validationInitialized) {
                initValidationTab();
            }

            // Initialize elite tab on first view
            if (tabName === 'elite' && !eliteInitialized) {
                initEliteTab();
            }

            // Initialize autonomous tab on first view
            if (tabName === 'autonomous' && !autonomousInitialized) {
                initAutonomousTab();
            }

            // Initialize tools tab on first view
            if (tabName === 'tools') {
                refreshDbStats();
                if (!priorityInitialized) {
                    initPriorityManager();
                }
            }
        }

        // === TOOLS TAB FUNCTIONS ===

        async function clearDatabase() {
            if (!confirm('Are you sure you want to delete ALL strategies from the database?\n\nThis will remove:\n- All Strategy History\n- All Elite Strategies\n- All validation data\n\nThis action cannot be undone!')) {
                return;
            }

            // Double confirmation
            if (!confirm('FINAL WARNING: This will permanently delete all strategy data. Continue?')) {
                return;
            }

            try {
                const response = await fetch('/api/db/clear', { method: 'POST' });
                const result = await response.json();

                if (response.ok) {
                    alert(`Database cleared successfully!\n\nDeleted ${result.deleted || 0} strategies.`);
                    refreshDbStats();
                    // Refresh other tabs if they were loaded
                    if (historyInitialized) loadHistoryStrategies();
                    if (eliteInitialized) loadEliteStrategies();
                } else {
                    alert('Error clearing database: ' + (result.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error clearing database: ' + e.message);
            }
        }

        async function resetEliteValidation() {
            if (!confirm('Reset all elite validation data?\n\nThis will:\n- Set all strategies to "pending"\n- Clear all elite scores\n- Clear all validation data\n\nStrategies will be re-validated with the current scoring system.')) {
                return;
            }

            try {
                const response = await fetch('/api/elite/reset-all', { method: 'POST' });
                const result = await response.json();

                if (response.ok) {
                    alert(`Elite validation reset!\n\n${result.reset_count || 0} strategies will be re-validated.`);
                    refreshDbStats();
                    if (eliteInitialized) loadEliteStrategies();
                } else {
                    alert('Error resetting elite validation: ' + (result.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error resetting elite validation: ' + e.message);
            }
        }

        async function refreshDbStats() {
            try {
                const response = await fetch('/api/db/strategies');
                const strategies = await response.json();

                const total = strategies.length;
                const validated = strategies.filter(s => s.elite_status && s.elite_status !== 'pending' && s.elite_status !== 'untestable').length;
                const pending = strategies.filter(s => !s.elite_status || s.elite_status === 'pending').length;

                document.getElementById('dbTotalStrategies').textContent = total;
                document.getElementById('dbEliteStrategies').textContent = validated;
                document.getElementById('dbPendingValidation').textContent = pending;
            } catch (e) {
                console.error('Error fetching db stats:', e);
            }
        }

        async function resetAutonomousCycle() {
            if (!confirm('Reset the autonomous optimizer cycle to the beginning?')) {
                return;
            }

            try {
                const response = await fetch('/api/autonomous/reset-cycle', { method: 'POST' });
                const result = await response.json();

                if (response.ok) {
                    alert('Cycle reset successfully! Optimizer will restart from the beginning.');
                } else {
                    alert('Error resetting cycle: ' + (result.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error resetting cycle: ' + e.message);
            }
        }

        // === PRIORITY QUEUE MANAGEMENT (4-List System) ===

        // Priority state - 4 separate lists
        let priorityPairs = [];
        let priorityPeriods = [];
        let priorityTimeframes = [];
        let priorityGranularities = [];
        let draggedItem = null;
        let draggedListType = null;

        async function initPriorityManager() {
            priorityInitialized = true;
            await loadPriorityLists();
        }

        async function loadPriorityLists() {
            try {
                let data;
                // Try WebSocket first (faster), fall back to HTTP if not connected
                try {
                    data = await loadPriorityViaWs();
                } catch (wsError) {
                    console.log('[Priority] WebSocket failed, falling back to HTTP:', wsError.message);
                    const response = await fetch('/api/priority/lists');
                    data = await response.json();
                }

                priorityPairs = data.pairs || [];
                priorityPeriods = data.periods || [];
                priorityTimeframes = data.timeframes || [];
                priorityGranularities = data.granularities || [];

                renderPriorityList('pairs', priorityPairs);
                renderPriorityList('periods', priorityPeriods);
                renderPriorityList('timeframes', priorityTimeframes);
                renderPriorityList('granularities', priorityGranularities);

                updateCombinationCount();
            } catch (e) {
                console.error('Error loading priority lists:', e);
            }
        }

        function renderPriorityList(listType, items) {
            const containerId = `${listType}PriorityList`;
            const countId = `${listType}Count`;
            const container = document.getElementById(containerId);
            const countEl = document.getElementById(countId);

            if (!container) return;

            const enabledCount = items.filter(i => i.enabled).length;
            countEl.textContent = `${enabledCount}/${items.length}`;

            container.innerHTML = items.map((item, index) => `
                <div class="priority-item-simple ${item.enabled ? '' : 'disabled'}"
                     data-id="${item.id}"
                     data-list-type="${listType}"
                     data-position="${index + 1}"
                     draggable="true"
                     ondragstart="handleDragStart(event)"
                     ondragend="handleDragEnd(event)"
                     ondragover="handleDragOver(event)"
                     ondrop="handleDrop(event)">
                    <span class="drag-handle">&#x2630;</span>
                    <span class="position-badge">${index + 1}</span>
                    <span class="item-label">${item.label}</span>
                    <button class="toggle-btn ${item.enabled ? 'enabled' : 'disabled-state'}"
                            onclick="togglePriorityListItem('${listType}', ${item.id})"
                            title="${item.enabled ? 'Disable' : 'Enable'}">
                        ${item.enabled ? '&#x2714;' : '&#x2716;'}
                    </button>
                </div>
            `).join('');
        }

        // Unified drag handlers for all 3 lists
        function handleDragStart(e) {
            draggedItem = e.target.closest('.priority-item-simple');
            draggedListType = draggedItem.dataset.listType;
            draggedItem.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', draggedItem.dataset.id);
        }

        function handleDragEnd(e) {
            if (draggedItem) {
                draggedItem.classList.remove('dragging');
                draggedItem = null;
                draggedListType = null;
            }
            document.querySelectorAll('.priority-item-simple').forEach(item => {
                item.classList.remove('drag-over');
            });
        }

        function handleDragOver(e) {
            e.preventDefault();
            const target = e.target.closest('.priority-item-simple');

            // Only allow dropping within the same list
            if (target && target !== draggedItem &&
                target.dataset.listType === draggedListType) {
                document.querySelectorAll('.priority-item-simple').forEach(item => {
                    item.classList.remove('drag-over');
                });
                target.classList.add('drag-over');
            }
        }

        async function handleDrop(e) {
            e.preventDefault();
            const target = e.target.closest('.priority-item-simple');

            if (!target || target === draggedItem ||
                target.dataset.listType !== draggedListType) return;

            const listType = draggedListType;
            const draggedId = parseInt(draggedItem.dataset.id);
            const targetId = parseInt(target.dataset.id);

            // Get the appropriate list
            let list;
            switch (listType) {
                case 'pairs': list = priorityPairs; break;
                case 'periods': list = priorityPeriods; break;
                case 'timeframes': list = priorityTimeframes; break;
                case 'granularities': list = priorityGranularities; break;
            }

            // Reorder in local array
            const draggedIndex = list.findIndex(i => i.id === draggedId);
            const targetIndex = list.findIndex(i => i.id === targetId);

            const [removed] = list.splice(draggedIndex, 1);
            list.splice(targetIndex, 0, removed);

            // Re-render immediately
            renderPriorityList(listType, list);

            // Persist to backend
            const newOrder = list.map(item => item.id);
            try {
                await fetch(`/api/priority/${listType}/reorder`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ order: newOrder })
                });
            } catch (e) {
                console.error('Error saving order:', e);
                loadPriorityLists(); // Reload on error
            }
        }

        async function togglePriorityListItem(listType, itemId) {
            try {
                await fetch(`/api/priority/${listType}/${itemId}/toggle`, { method: 'PATCH' });
                await loadPriorityLists();
            } catch (e) {
                console.error('Error toggling item:', e);
            }
        }

        function updateCombinationCount() {
            const enabledPairs = priorityPairs.filter(p => p.enabled).length;
            const enabledPeriods = priorityPeriods.filter(p => p.enabled).length;
            const enabledTimeframes = priorityTimeframes.filter(t => t.enabled).length;
            const enabledGranularities = priorityGranularities.filter(g => g.enabled).length;

            const total = enabledPairs * enabledPeriods * enabledTimeframes * enabledGranularities;
            const el = document.getElementById('totalCombinations');
            if (el) {
                el.textContent = total.toLocaleString();
            }
        }

        async function resetPriorityToDefaults() {
            if (!confirm('Reset all priority settings to defaults?')) return;

            try {
                await fetch('/api/priority/reset-defaults', { method: 'POST' });
                await loadPriorityLists();
            } catch (e) {
                console.error('Error resetting:', e);
            }
        }

        async function enableAllPriorityItems() {
            try {
                await fetch('/api/priority/enable-all', { method: 'POST' });
                await loadPriorityLists();
            } catch (e) {
                console.error('Error:', e);
            }
        }

        async function disableAllPriorityItems() {
            try {
                await fetch('/api/priority/disable-all', { method: 'POST' });
                await loadPriorityLists();
            } catch (e) {
                console.error('Error:', e);
            }
        }

        // Set up event delegation for history table (only once)
        let historyTableDelegationSetup = false;
        function setupHistoryTableDelegation() {
            if (historyTableDelegationSetup) return;
            historyTableDelegationSetup = true;

            const tbody = document.getElementById('history-tbody');
            if (!tbody) return;

            tbody.addEventListener('click', (e) => {
                // Don't handle clicks on buttons or action buttons
                if (e.target.closest('.action-btn-group') || e.target.closest('button')) {
                    return;
                }

                const row = e.target.closest('.strategy-row');
                if (!row) return;

                const strategyId = parseInt(row.dataset.strategyId, 10);
                const hasVariants = row.dataset.hasVariants === 'true';
                const groupKey = row.dataset.groupKey;

                if (hasVariants && groupKey) {
                    // Find the group from historyStrategies
                    const groups = {};
                    historyStrategies.forEach(strategy => {
                        const key = `${strategy.strategy_name}|${strategy.symbol}|${strategy.timeframe}`;
                        if (!groups[key]) groups[key] = [];
                        groups[key].push(strategy);
                    });
                    const group = groups[groupKey];
                    if (group) toggleVariants(row, group);
                } else if (strategyId) {
                    toggleStrategyDetails(strategyId, row);
                }
            });
        }

        // Initialize History Tab
        async function initHistoryTab() {
            historyInitialized = true;
            try {
                // Try WebSocket first (much faster than HTTP polling)
                // Fall back to HTTP if WebSocket fails
                try {
                    historyStrategies = await loadStrategiesViaWs();
                } catch (wsError) {
                    console.log('[History] WebSocket failed, falling back to HTTP:', wsError.message);
                    const response = await fetch('/api/db/strategies?limit=500');
                    historyStrategies = await response.json();
                }

                // Extract filter options from strategies
                const symbols = [...new Set(historyStrategies.map(s => s.symbol).filter(Boolean))].sort();
                const timeframes = [...new Set(historyStrategies.map(s => s.timeframe).filter(Boolean))].sort();

                // Populate symbol dropdown
                const symbolSelect = document.getElementById('filter-symbol');
                symbolSelect.innerHTML = '<option value="">All Pairs</option>';
                symbols.forEach(symbol => {
                    symbolSelect.innerHTML += `<option value="${symbol}">${symbol}</option>`;
                });

                // Populate timeframe dropdown
                const timeframeSelect = document.getElementById('filter-timeframe');
                timeframeSelect.innerHTML = '<option value="">All Timeframes</option>';
                timeframes.forEach(tf => {
                    timeframeSelect.innerHTML += `<option value="${tf}">${tf}</option>`;
                });

                // Sort and render (strategies already loaded)
                sortStrategiesArray();
                document.getElementById('history-count').textContent = `${historyStrategies.length} strategies`;
                document.getElementById('history-count').className = 'status-badge ' + (historyStrategies.length > 0 ? 'success' : 'neutral');

                const loading = document.getElementById('history-loading');
                const empty = document.getElementById('history-empty');
                const table = document.getElementById('history-table');

                if (historyStrategies.length > 0) {
                    loading.style.display = 'none';
                    table.style.display = 'table';
                    // Set up event delegation before rendering
                    setupHistoryTableDelegation();
                    renderHistoryTable();
                } else {
                    loading.style.display = 'none';
                    empty.style.display = 'block';
                }
            } catch (error) {
                console.error('Failed to initialize history tab:', error);
                // Reset flag so user can retry by switching tabs
                historyInitialized = false;
                document.getElementById('history-loading').textContent = 'Failed to load. Please try again.';
            }
        }

        // Load History Strategies
        async function loadHistoryStrategies() {
            const tbody = document.getElementById('history-tbody');
            const loading = document.getElementById('history-loading');
            const empty = document.getElementById('history-empty');
            const table = document.getElementById('history-table');

            // Show loading
            loading.style.display = 'block';
            empty.style.display = 'none';
            table.style.display = 'none';

            try {
                // Build query parameters - fetch more to allow client-side filtering
                const params = new URLSearchParams({
                    limit: '500'
                });

                if (historyFilters.symbol) params.append('symbol', historyFilters.symbol);
                if (historyFilters.timeframe) params.append('timeframe', historyFilters.timeframe);

                const response = await fetch(`/api/db/strategies?${params.toString()}`);
                historyStrategies = await response.json();

                // Filter by backtest data DURATION if specified (client-side)
                // Shows only strategies tested on at least X amount of data
                if (historyFilters.period) {
                    // Filter ranges must match calculatePeriodLabel() display thresholds exactly
                    let minDays = 0;
                    let maxDays = 0;
                    switch (historyFilters.period) {
                        case '1w': minDays = 0; maxDays = 10; break;      // Display: <= 10
                        case '2w': minDays = 11; maxDays = 20; break;     // Display: <= 20
                        case '1m': minDays = 21; maxDays = 45; break;     // Display: <= 45
                        case '3m': minDays = 46; maxDays = 100; break;    // Display: <= 100
                        case '6m': minDays = 101; maxDays = 200; break;   // Display: <= 200
                        case '9m': minDays = 201; maxDays = 300; break;   // Display: <= 300
                        case '1y': minDays = 301; maxDays = 400; break;   // Display: <= 400
                        case '2y': minDays = 401; maxDays = 800; break;   // Display: <= 800
                        case '3y': minDays = 801; maxDays = 1200; break;  // Display: <= 1200
                        case '5y': minDays = 1201; maxDays = 3650; break; // Display: > 1200
                    }

                    historyStrategies = historyStrategies.filter(s => {
                        // If no date info, HIDE the strategy when filter is active
                        if (!s.data_start || !s.data_end) return false;
                        // Check for invalid data (numeric indices instead of dates)
                        if (s.data_start === '0' || s.data_start === 0) return false;

                        try {
                            const startDate = new Date(s.data_start);
                            const endDate = new Date(s.data_end);

                            // Check for invalid dates
                            if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) return false;
                            // Check for epoch dates (indicates invalid data)
                            if (startDate.getFullYear() < 2000 || endDate.getFullYear() < 2000) return false;

                            const durationDays = (endDate - startDate) / (1000 * 60 * 60 * 24);
                            // Negative or unreasonable days indicates bad data
                            if (durationDays < 0 || durationDays > 3650) return false;

                            // Match strategies within the period range
                            return durationDays >= minDays && durationDays <= maxDays;
                        } catch (e) {
                            return false;
                        }
                    });
                }

                // Sort strategies
                sortStrategiesArray();

                // Update count
                document.getElementById('history-count').textContent = `${historyStrategies.length} strategies`;
                document.getElementById('history-count').className = 'status-badge ' + (historyStrategies.length > 0 ? 'success' : 'neutral');

                // Render table or show empty state
                if (historyStrategies.length > 0) {
                    loading.style.display = 'none';
                    table.style.display = 'table';
                    renderHistoryTable();
                } else {
                    loading.style.display = 'none';
                    empty.style.display = 'block';
                }
            } catch (error) {
                console.error('Failed to load strategies:', error);
                loading.textContent = 'Failed to load strategies.';
            }
        }

        // Sort strategies array
        function sortStrategiesArray() {
            historyStrategies.sort((a, b) => {
                let aVal = a[historySortBy];
                let bVal = b[historySortBy];

                // Special handling for period - calculate days from date range
                if (historySortBy === 'period') {
                    const getDays = (s) => {
                        if (!s.data_start || !s.data_end) return 0;
                        try {
                            return (new Date(s.data_end) - new Date(s.data_start)) / (1000 * 60 * 60 * 24);
                        } catch { return 0; }
                    };
                    aVal = getDays(a);
                    bVal = getDays(b);
                }

                // Handle null/undefined
                if (aVal === null || aVal === undefined) aVal = historySortOrder === 'desc' ? -Infinity : Infinity;
                if (bVal === null || bVal === undefined) bVal = historySortOrder === 'desc' ? -Infinity : Infinity;

                // String comparison for text fields
                if (typeof aVal === 'string') {
                    return historySortOrder === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }

                // Numeric comparison
                return historySortOrder === 'asc' ? aVal - bVal : bVal - aVal;
            });
        }

        // Render History Table - Groups strategies by name, shows best (lowest SL) first
        function renderHistoryTable() {
            const tbody = document.getElementById('history-tbody');
            tbody.innerHTML = '';
            // Reset expanded state since DOM is being rebuilt
            currentExpandedHistoryRow = null;
            currentExpandedHistoryId = null;

            // Group strategies by name+symbol+timeframe
            const groups = {};
            historyStrategies.forEach(strategy => {
                const key = `${strategy.strategy_name}|${strategy.symbol}|${strategy.timeframe}`;
                if (!groups[key]) {
                    groups[key] = [];
                }
                groups[key].push(strategy);
            });

            // Sort each group by SL (ascending - lowest first)
            Object.values(groups).forEach(group => {
                group.sort((a, b) => {
                    const slA = a.sl_percent || a.params?.sl_percent || 999;
                    const slB = b.sl_percent || b.params?.sl_percent || 999;
                    return slA - slB;
                });
            });

            // Convert to array and sort groups by the selected column
            const sortedGroups = Object.values(groups).sort((a, b) => {
                let aVal, bVal;

                // Special handling for period - calculate days from date range
                if (historySortBy === 'period') {
                    const getDays = (s) => {
                        if (!s.data_start || !s.data_end) return 0;
                        try {
                            return (new Date(s.data_end) - new Date(s.data_start)) / (1000 * 60 * 60 * 24);
                        } catch { return 0; }
                    };
                    aVal = getDays(a[0]);
                    bVal = getDays(b[0]);
                } else {
                    aVal = a[0][historySortBy];
                    bVal = b[0][historySortBy];
                }

                // Handle null/undefined
                if (aVal === null || aVal === undefined) aVal = historySortOrder === 'desc' ? -Infinity : Infinity;
                if (bVal === null || bVal === undefined) bVal = historySortOrder === 'desc' ? -Infinity : Infinity;

                // String comparison for text fields
                if (typeof aVal === 'string') {
                    return historySortOrder === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }

                // Numeric comparison
                return historySortOrder === 'asc' ? aVal - bVal : bVal - aVal;
            });

            // Use DocumentFragment for batch DOM insert (single reflow instead of N)
            const fragment = document.createDocumentFragment();
            let rowIndex = 0;
            sortedGroups.forEach(group => {
                const bestStrategy = group[0]; // Lowest SL
                const hasVariants = group.length > 1;
                rowIndex++;

                const row = document.createElement('tr');
                row.className = 'strategy-row';
                row.dataset.strategyId = bestStrategy.id;
                row.dataset.groupKey = `${bestStrategy.strategy_name}|${bestStrategy.symbol}|${bestStrategy.timeframe}`;
                row.dataset.hasVariants = hasVariants ? 'true' : 'false';
                row.style.cursor = 'pointer';

                // Win rate color class
                const winRateClass = bestStrategy.win_rate >= 50 ? 'win-rate-high' :
                                     bestStrategy.win_rate >= 40 ? 'win-rate-medium' : 'win-rate-low';

                // PnL color class
                const pnlClass = bestStrategy.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                const pnlPrefix = bestStrategy.total_pnl >= 0 ? '+' : '';

                // Drawdown color class
                const dd = bestStrategy.max_drawdown || 0;
                const ddClass = dd <= 5 ? 'dd-low' : dd <= 15 ? 'dd-medium' : 'dd-high';

                // Format date with time
                const createdDate = bestStrategy.created_at ? new Date(bestStrategy.created_at) : null;
                const dateStr = createdDate ? createdDate.toLocaleDateString('en-GB', {
                    day: '2-digit', month: 'short'
                }) + ' ' + createdDate.toLocaleTimeString('en-GB', {
                    hour: 'numeric', minute: '2-digit', hour12: true
                }) : '-';

                // TP/SL values
                const tpPercent = bestStrategy.tp_percent || bestStrategy.params?.tp_percent || '-';
                const slPercent = bestStrategy.sl_percent || bestStrategy.params?.sl_percent || '-';

                // Show variant count badge if there are variants
                const variantBadge = hasVariants ? `<span class="variant-badge" style="background: var(--primary); color: white; padding: 0.1rem 0.4rem; border-radius: 10px; font-size: 0.7rem; margin-left: 0.3rem;">${group.length}</span>` : '';

                const periodLabel = calculatePeriodLabel(bestStrategy.data_start, bestStrategy.data_end);
                row.innerHTML = `
                    <td><span class="expand-icon">${hasVariants ? '▶' : '•'}</span> ${rowIndex}</td>
                    <td title="${bestStrategy.strategy_name}">${truncateText(bestStrategy.strategy_name, 18)}${variantBadge}</td>
                    <td><span class="symbol-badge">${bestStrategy.symbol || '-'}</span></td>
                    <td><span class="period-badge">${periodLabel}</span></td>
                    <td><span class="tf-badge">${bestStrategy.timeframe || '-'}</span></td>
                    <td style="color: var(--success);">${typeof tpPercent === 'number' ? formatNumber(tpPercent, 1) : tpPercent}</td>
                    <td style="color: var(--danger);">${typeof slPercent === 'number' ? formatNumber(slPercent, 1) : slPercent}</td>
                    <td>${formatNumber(bestStrategy.composite_score, 1)}</td>
                    <td class="${winRateClass}">${formatNumber(bestStrategy.win_rate, 1)}%</td>
                    <td>${formatNumber(bestStrategy.profit_factor, 2)}</td>
                    <td class="${pnlClass}">${pnlPrefix}${formatNumber(bestStrategy.total_pnl, 2)}</td>
                    <td class="${ddClass}">${formatNumber(dd, 1)}%</td>
                    <td>${bestStrategy.total_trades || 0}</td>
                    <td>${dateStr}</td>
                    <td>
                        <div class="action-btn-group">
                            <button class="action-btn" onclick="event.stopPropagation(); copyPineScript(${bestStrategy.id})" title="Copy Pine Script">📋</button>
                            <button class="action-btn" onclick="event.stopPropagation(); downloadPineScript(${bestStrategy.id}, '${bestStrategy.strategy_name}')" title="Download Pine Script">📜</button>
                            <button class="action-btn" onclick="event.stopPropagation(); exportTradesCSV(${bestStrategy.id}, '${bestStrategy.strategy_name}')" title="Export Trades CSV">📊</button>
                            <button class="action-btn validate" onclick="event.stopPropagation(); validateFromHistory(${bestStrategy.id})" title="Validate Strategy">🔬</button>
                            <button class="action-btn danger" onclick="event.stopPropagation(); deleteStrategy(${bestStrategy.id}, '${bestStrategy.strategy_name}')" title="Delete">🗑️</button>
                        </div>
                    </td>
                `;

                // Event handling is done via delegation in setupHistoryTableDelegation()
                fragment.appendChild(row);

                // Create variants container row (hidden by default)
                if (hasVariants) {
                    const variantsRow = document.createElement('tr');
                    variantsRow.className = 'variants-row';
                    variantsRow.id = `variants-row-${bestStrategy.id}`;
                    variantsRow.style.display = 'none';
                    variantsRow.innerHTML = `<td colspan="15" class="variants-cell" style="padding: 0; background: var(--bg-secondary);"></td>`;
                    fragment.appendChild(variantsRow);
                }
            });

            // Single DOM operation: append all rows at once
            tbody.appendChild(fragment);

            // Update sort indicators
            updateSortIndicators();
        }

        // Toggle variants dropdown
        function toggleVariants(rowElement, group) {
            const bestStrategy = group[0];
            const variantsRow = document.getElementById(`variants-row-${bestStrategy.id}`);
            const expandIcon = rowElement.querySelector('.expand-icon');

            if (!variantsRow) return;

            // Close previously expanded row (O(1) instead of O(n²))
            if (currentExpandedHistoryRow && currentExpandedHistoryRow !== rowElement) {
                const prevVariantsRow = document.getElementById(`variants-row-${currentExpandedHistoryId}`);
                if (prevVariantsRow) prevVariantsRow.style.display = 'none';
                currentExpandedHistoryRow.classList.remove('expanded');
                const prevIcon = currentExpandedHistoryRow.querySelector('.expand-icon');
                if (prevIcon) prevIcon.textContent = '▶';
            }

            // Toggle current row
            if (variantsRow.style.display === 'none') {
                variantsRow.style.display = 'table-row';
                rowElement.classList.add('expanded');
                expandIcon.textContent = '▼';
                // Track expanded state
                currentExpandedHistoryRow = rowElement;
                currentExpandedHistoryId = bestStrategy.id;

                // Build variants table
                const variantsCell = variantsRow.querySelector('.variants-cell');
                let variantsHtml = `<div style="padding: 0.5rem 1rem 0.5rem 2rem;">
                    <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                        <thead>
                            <tr style="background: var(--bg-card);">
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">TP%</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">SL%</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">Score</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">Win Rate</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">PF</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">PnL</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">Actions</th>
                            </tr>
                        </thead>
                        <tbody>`;

                group.forEach((variant, idx) => {
                    const tp = variant.tp_percent || variant.params?.tp_percent || '-';
                    const sl = variant.sl_percent || variant.params?.sl_percent || '-';
                    const pnlClass = variant.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                    const pnlPrefix = variant.total_pnl >= 0 ? '+' : '';
                    const isBest = idx === 0;

                    variantsHtml += `
                        <tr style="border-bottom: 1px solid var(--border-subtle); ${isBest ? 'background: rgba(34, 197, 94, 0.1);' : ''}">
                            <td style="padding: 0.4rem; color: var(--success);">${typeof tp === 'number' ? formatNumber(tp, 1) : tp}${isBest ? ' <span style="color: var(--success); font-size: 0.7rem;">★ Best</span>' : ''}</td>
                            <td style="padding: 0.4rem; color: var(--danger);">${typeof sl === 'number' ? formatNumber(sl, 1) : sl}</td>
                            <td style="padding: 0.4rem;">${formatNumber(variant.composite_score, 1)}</td>
                            <td style="padding: 0.4rem; color: ${variant.win_rate >= 50 ? 'var(--success)' : 'var(--text-primary)'};">${formatNumber(variant.win_rate, 1)}%</td>
                            <td style="padding: 0.4rem;">${formatNumber(variant.profit_factor, 2)}</td>
                            <td style="padding: 0.4rem;" class="${pnlClass}">${pnlPrefix}${formatNumber(variant.total_pnl, 2)}</td>
                            <td style="padding: 0.4rem; text-align: center;">
                                <button class="action-btn" onclick="event.stopPropagation(); copyPineScript(${variant.id})" title="Copy Pine Script">📋</button>
                                <button class="action-btn" onclick="event.stopPropagation(); downloadPineScript(${variant.id}, '${variant.strategy_name}')" title="Download">📜</button>
                                <button class="action-btn" onclick="event.stopPropagation(); exportTradesCSV(${variant.id}, '${variant.strategy_name}')" title="Export Trades CSV">📊</button>
                                <button class="action-btn validate" onclick="event.stopPropagation(); validateFromHistory(${variant.id})" title="Validate">🔬</button>
                                <button class="action-btn danger" onclick="event.stopPropagation(); deleteStrategy(${variant.id}, '${variant.strategy_name}')" title="Delete">🗑️</button>
                            </td>
                        </tr>`;
                });

                variantsHtml += `</tbody></table></div>`;
                variantsCell.innerHTML = variantsHtml;
            } else {
                variantsRow.style.display = 'none';
                rowElement.classList.remove('expanded');
                expandIcon.textContent = '▶';
                // Clear expanded state
                currentExpandedHistoryRow = null;
                currentExpandedHistoryId = null;
            }
        }

        // Toggle inline strategy details
        async function toggleStrategyDetails(strategyId, rowElement) {
            const detailRow = document.getElementById(`detail-row-${strategyId}`);
            const detailContent = document.getElementById(`detail-content-${strategyId}`);
            const expandIcon = rowElement.querySelector('.expand-icon');

            // Close all other expanded rows
            document.querySelectorAll('.strategy-detail-row').forEach(r => {
                if (r.id !== `detail-row-${strategyId}`) {
                    r.style.display = 'none';
                }
            });
            document.querySelectorAll('.strategy-row').forEach(r => {
                r.classList.remove('expanded');
                const icon = r.querySelector('.expand-icon');
                if (icon) icon.textContent = '▶';
            });

            // Toggle current row
            if (detailRow.style.display === 'none') {
                detailRow.style.display = 'table-row';
                rowElement.classList.add('expanded');
                expandIcon.textContent = '▼';

                // Load details if not already loaded
                if (!detailContent.innerHTML || detailContent.innerHTML.includes('Loading')) {
                    detailContent.innerHTML = '<p style="text-align: center; padding: 1rem; color: var(--text-secondary);">Loading...</p>';
                    await loadInlineStrategyDetails(strategyId, detailContent);
                }
            } else {
                detailRow.style.display = 'none';
                rowElement.classList.remove('expanded');
                expandIcon.textContent = '▶';
            }
        }

        // Load strategy details inline
        async function loadInlineStrategyDetails(strategyId, container) {
            try {
                const response = await fetch(`/api/db/strategies/${strategyId}`);
                const strategy = await response.json();

                const winRateClass = strategy.win_rate >= 50 ? 'pnl-positive' : strategy.win_rate >= 40 ? '' : 'pnl-negative';
                const pnlClass = strategy.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                const ddClass = (strategy.max_drawdown || 0) <= 5 ? 'dd-low' : (strategy.max_drawdown || 0) <= 15 ? 'dd-medium' : 'dd-high';

                container.innerHTML = `
                    <div class="inline-detail-grid">
                        <div class="inline-detail-section">
                            <h4>Performance Metrics</h4>
                            <div class="mini-stats-grid">
                                <div class="mini-stat"><span class="label">Score</span><span class="value">${formatNumber(strategy.composite_score, 1)}</span></div>
                                <div class="mini-stat"><span class="label">Win Rate</span><span class="value ${winRateClass}">${formatNumber(strategy.win_rate, 1)}%</span></div>
                                <div class="mini-stat"><span class="label">Profit Factor</span><span class="value">${formatNumber(strategy.profit_factor, 2)}</span></div>
                                <div class="mini-stat"><span class="label">Total PnL</span><span class="value ${pnlClass}">${strategy.total_pnl >= 0 ? '+' : ''}${formatNumber(strategy.total_pnl, 2)}</span></div>
                                <div class="mini-stat"><span class="label">Trades</span><span class="value">${strategy.total_trades || 0}</span></div>
                                <div class="mini-stat"><span class="label">Max DD</span><span class="value ${ddClass}">${formatNumber(strategy.max_drawdown, 1)}%</span></div>
                                <div class="mini-stat"><span class="label">Sharpe</span><span class="value">${formatNumber(strategy.sharpe_ratio, 2)}</span></div>
                                <div class="mini-stat"><span class="label">Recovery</span><span class="value">${formatNumber(strategy.recovery_factor, 2)}</span></div>
                            </div>
                        </div>
                        <div class="inline-detail-section">
                            <h4>Parameters</h4>
                            <div class="params-grid">
                                <div><span class="label">Symbol:</span> <span class="symbol-badge">${strategy.symbol || '-'}</span></div>
                                <div><span class="label">Timeframe:</span> <span class="tf-badge">${strategy.timeframe || '-'}</span></div>
                                <div><span class="label">Take Profit:</span> ${formatNumber(strategy.tp_percent, 2)}%</div>
                                <div><span class="label">Stop Loss:</span> ${formatNumber(strategy.sl_percent, 2)}%</div>
                                <div><span class="label">Data Source:</span> ${strategy.data_source || '-'}</div>
                                <div><span class="label">Created:</span> ${strategy.created_at || '-'}</div>
                            </div>
                        </div>
                        <div class="inline-detail-section actions-section">
                            <h4>Actions</h4>
                            <div class="inline-actions">
                                <button class="btn btn-primary btn-sm" onclick="copyPineScript(${strategy.id})">📋 Copy Pine Script</button>
                                <button class="btn btn-secondary btn-sm" onclick="downloadPineScript(${strategy.id}, '${strategy.strategy_name}')">📜 Download</button>
                                <button class="btn btn-danger btn-sm" onclick="deleteStrategy(${strategy.id}, '${strategy.strategy_name}')">🗑️ Delete</button>
                            </div>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load strategy details:', error);
                container.innerHTML = '<p style="text-align: center; padding: 1rem; color: var(--danger);">Failed to load details.</p>';
            }
        }

        // Update sort indicators on headers
        function updateSortIndicators() {
            document.querySelectorAll('.history-table .sortable').forEach(th => {
                th.classList.remove('active', 'asc', 'desc');
                if (th.dataset.sort === historySortBy) {
                    th.classList.add('active', historySortOrder);
                }
            });
        }

        // Sort table by column
        function sortHistoryTable(column) {
            if (historySortBy === column) {
                // Toggle order
                historySortOrder = historySortOrder === 'desc' ? 'asc' : 'desc';
            } else {
                historySortBy = column;
                historySortOrder = 'desc'; // Default to descending for new column
            }

            updateSortIndicators();
            sortStrategiesArray();
            renderHistoryTable();
        }

        // Apply filters
        function applyHistoryFilters() {
            historyFilters.symbol = document.getElementById('filter-symbol').value;
            historyFilters.timeframe = document.getElementById('filter-timeframe').value;
            historyFilters.period = document.getElementById('filter-period').value;

            loadHistoryStrategies();
        }

        // Reset filters
        function resetHistoryFilters() {
            document.getElementById('filter-symbol').value = '';
            document.getElementById('filter-timeframe').value = '';
            document.getElementById('filter-period').value = '';

            historyFilters = { symbol: '', timeframe: '', period: '' };
            loadHistoryStrategies();
        }

        // ========== ELITE STRATEGIES SORTING & FILTERING ==========

        // Sort elite table
        function sortEliteTable(column) {
            if (eliteSortBy === column) {
                eliteSortOrder = eliteSortOrder === 'desc' ? 'asc' : 'desc';
            } else {
                eliteSortBy = column;
                eliteSortOrder = 'desc';
            }

            updateEliteSortIndicators();
            applyEliteSortAndFilter();
        }

        // Update sort indicators on elite headers
        function updateEliteSortIndicators() {
            document.querySelectorAll('.elite-table .sortable').forEach(th => {
                th.classList.remove('active', 'asc', 'desc');
                if (th.dataset.sort === eliteSortBy) {
                    th.classList.add('active', eliteSortOrder);
                }
            });
        }

        // Apply filters
        function applyEliteFilters() {
            eliteFilters.search = document.getElementById('elite-filter-search').value.toLowerCase();
            eliteFilters.symbol = document.getElementById('elite-filter-symbol').value;
            const statusEl = document.getElementById('elite-filter-status');
            eliteFilters.status = statusEl ? statusEl.value : '';
            eliteFilters.direction = document.getElementById('elite-filter-direction').value;

            applyEliteSortAndFilter();
        }

        // Reset elite filters
        function resetEliteFilters() {
            document.getElementById('elite-filter-search').value = '';
            document.getElementById('elite-filter-symbol').value = '';
            const statusEl = document.getElementById('elite-filter-status');
            if (statusEl) statusEl.value = '';
            document.getElementById('elite-filter-direction').value = '';

            eliteFilters = { search: '', symbol: '', status: '', direction: '' };
            applyEliteSortAndFilter();
        }

        // Apply both sorting and filtering to elite data
        function applyEliteSortAndFilter() {
            if (!eliteStrategiesData || eliteStrategiesData.length === 0) return;

            // First, filter the data
            eliteFilteredData = eliteStrategiesData.filter(s => {
                const direction = s.params?.direction || s.trade_mode || 'long';

                // Search filter
                if (eliteFilters.search && !s.strategy_name.toLowerCase().includes(eliteFilters.search)) {
                    return false;
                }

                // Symbol filter
                if (eliteFilters.symbol && s.symbol !== eliteFilters.symbol) {
                    return false;
                }

                // Status filter
                if (eliteFilters.status && s.elite_status !== eliteFilters.status) {
                    return false;
                }

                // Direction filter
                if (eliteFilters.direction && direction !== eliteFilters.direction) {
                    return false;
                }

                return true;
            });

            // Re-render the table with filtered and sorted data
            renderEliteTableFiltered(eliteFilteredData);
        }

        // Populate elite symbol filter dropdown
        function populateEliteSymbolFilter(strategies) {
            const select = document.getElementById('elite-filter-symbol');
            const currentValue = select.value;

            // Get unique symbols
            const symbols = [...new Set(strategies.map(s => s.symbol))].sort();

            // Clear existing options except first
            select.innerHTML = '<option value="">All Pairs</option>';

            symbols.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                select.appendChild(option);
            });

            // Restore selected value
            select.value = currentValue;
        }

        // ========== END ELITE SORTING & FILTERING ==========

        // View strategy details
        async function viewStrategyDetails(strategyId) {
            const panel = document.getElementById('strategy-detail-panel');
            const content = document.getElementById('strategy-detail-content');
            const nameEl = document.getElementById('detail-strategy-name');

            panel.style.display = 'block';
            content.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Loading...</p>';

            try {
                const response = await fetch(`/api/db/strategies/${strategyId}`);
                const strategy = await response.json();

                nameEl.textContent = strategy.strategy_name;

                // Build detail content
                const winRateClass = strategy.win_rate >= 50 ? 'pnl-positive' : strategy.win_rate >= 40 ? '' : 'pnl-negative';
                const pnlClass = strategy.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

                content.innerHTML = `
                    <div class="detail-stats-grid">
                        <div class="detail-stat">
                            <div class="detail-stat-label">Composite Score</div>
                            <div class="detail-stat-value">${formatNumber(strategy.composite_score, 1)}</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Win Rate</div>
                            <div class="detail-stat-value ${winRateClass}">${formatNumber(strategy.win_rate, 1)}%</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Profit Factor</div>
                            <div class="detail-stat-value">${formatNumber(strategy.profit_factor, 2)}</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Total PnL</div>
                            <div class="detail-stat-value ${pnlClass}">${strategy.total_pnl >= 0 ? '+' : ''}${formatNumber(strategy.total_pnl, 2)}</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Total Trades</div>
                            <div class="detail-stat-value">${strategy.total_trades || 0}</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Max Drawdown</div>
                            <div class="detail-stat-value pnl-negative">${formatNumber(strategy.max_drawdown, 2)}%</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Sharpe Ratio</div>
                            <div class="detail-stat-value">${formatNumber(strategy.sharpe_ratio, 2)}</div>
                        </div>
                        <div class="detail-stat">
                            <div class="detail-stat-label">Recovery Factor</div>
                            <div class="detail-stat-value">${formatNumber(strategy.recovery_factor, 2)}</div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <h3>Strategy Parameters</h3>
                        <table class="params-table" style="width: 100%;">
                            <tr><td>Symbol</td><td><span class="symbol-badge">${strategy.symbol || '-'}</span></td></tr>
                            <tr><td>Timeframe</td><td><span class="tf-badge">${strategy.timeframe || '-'}</span></td></tr>
                            <tr><td>Take Profit</td><td>${formatNumber(strategy.tp_percent, 2)}%</td></tr>
                            <tr><td>Stop Loss</td><td>${formatNumber(strategy.sl_percent, 2)}%</td></tr>
                            <tr><td>Data Range</td><td>${strategy.data_start || '-'} → ${strategy.data_end || '-'}</td></tr>
                            <tr><td>Data Source</td><td>${strategy.data_source || '-'}</td></tr>
                            <tr><td>Created</td><td>${strategy.created_at || '-'}</td></tr>
                        </table>
                    </div>

                    ${strategy.indicator_params ? `
                    <div class="detail-section">
                        <h3>Indicator Parameters</h3>
                        <pre style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem;">${JSON.stringify(strategy.indicator_params, null, 2)}</pre>
                    </div>
                    ` : ''}

                    <div style="margin-top: 1.5rem; display: flex; gap: 0.5rem;">
                        <button class="btn btn-primary" onclick="downloadPineScript(${strategy.id}, '${strategy.strategy_name}')">
                            📜 Download Pine Script
                        </button>
                        <button class="btn btn-secondary danger" style="background: var(--danger-bg); border-color: var(--danger); color: var(--danger);" onclick="deleteStrategy(${strategy.id}, '${strategy.strategy_name}')">
                            🗑️ Delete Strategy
                        </button>
                    </div>
                `;

                // Scroll to panel
                panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } catch (error) {
                console.error('Failed to load strategy details:', error);
                content.innerHTML = '<p style="text-align: center; color: var(--danger);">Failed to load strategy details.</p>';
            }
        }

        // Close strategy detail panel
        function closeStrategyDetail() {
            document.getElementById('strategy-detail-panel').style.display = 'none';
        }

        // Download Pine Script
        async function downloadPineScript(strategyId, strategyName) {
            try {
                const response = await fetch(`/api/db/strategies/${strategyId}/pinescript`);
                const data = await response.json();

                if (data.pinescript) {
                    // Create download
                    const blob = new Blob([data.pinescript], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${strategyName.replace(/[^a-z0-9]/gi, '_')}.pine`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                } else {
                    alert('Failed to generate Pine Script');
                }
            } catch (error) {
                console.error('Failed to download Pine Script:', error);
                alert('Failed to download Pine Script: ' + error.message);
            }
        }

        // Export Trades CSV
        async function exportTradesCSV(strategyId, strategyName) {
            try {
                showToast('Generating trades CSV...');
                const response = await fetch(`/api/export-trades/${strategyId}`);
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Export failed');
                }

                // Get filename from Content-Disposition header or generate one
                const disposition = response.headers.get('Content-Disposition');
                let filename = `${strategyName.replace(/[^a-z0-9]/gi, '_')}_trades.csv`;
                if (disposition) {
                    const match = disposition.match(/filename="?([^"]+)"?/);
                    if (match) filename = match[1];
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('Trades CSV downloaded!');
            } catch (error) {
                console.error('Failed to export trades:', error);
                showToast('Export failed: ' + error.message, 'error');
            }
        }

        // Copy Pine Script to clipboard
        async function copyPineScript(strategyId) {
            try {
                const response = await fetch(`/api/db/strategies/${strategyId}/pinescript`);
                const data = await response.json();

                if (data.pinescript) {
                    await navigator.clipboard.writeText(data.pinescript);
                    // Show brief success feedback
                    showToast('Pine Script copied to clipboard!');
                } else {
                    alert('Failed to generate Pine Script');
                }
            } catch (error) {
                console.error('Failed to copy Pine Script:', error);
                alert('Failed to copy Pine Script: ' + error.message);
            }
        }

        // Show Pine Script in modal (for Elite Strategies)
        async function showPineScript(strategyId) {
            try {
                const response = await fetch(`/api/db/strategies/${strategyId}/pinescript`);
                const data = await response.json();

                if (data.pinescript) {
                    const title = data.strategy_name || `Strategy #${strategyId}`;
                    showPineModal(title, data.pinescript);
                } else {
                    showToast('Failed to generate Pine Script');
                }
            } catch (error) {
                console.error('Failed to load Pine Script:', error);
                showToast('Failed to load Pine Script: ' + error.message);
            }
        }

        // Show toast notification
        function showToast(message) {
            const toast = document.createElement('div');
            toast.className = 'toast-notification';
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.classList.add('show'), 10);
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => document.body.removeChild(toast), 300);
            }, 2000);
        }

        // Delete strategy
        async function deleteStrategy(strategyId, strategyName) {
            if (!confirm(`Delete strategy "${strategyName}"?\n\nThis action cannot be undone.`)) {
                return;
            }

            try {
                const response = await fetch(`/api/db/strategies/${strategyId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    // Remove from array and re-render
                    historyStrategies = historyStrategies.filter(s => s.id !== strategyId);
                    renderHistoryTable();

                    // Close detail panel if showing this strategy
                    closeStrategyDetail();

                    // Update count
                    document.getElementById('history-count').textContent = `${historyStrategies.length} strategies`;
                } else {
                    alert('Failed to delete strategy');
                }
            } catch (error) {
                console.error('Failed to delete strategy:', error);
                alert('Failed to delete strategy: ' + error.message);
            }
        }

        // Helper: Format number with decimals
        function formatNumber(num, decimals = 2) {
            if (num === null || num === undefined || isNaN(num)) return '-';
            return Number(num).toFixed(decimals);
        }

        // Helper: Truncate text
        function truncateText(text, maxLength) {
            if (!text) return '-';
            return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        }

        function calculatePeriodLabel(dataStart, dataEnd) {
            if (!dataStart || !dataEnd) return '-';
            // Check for invalid data (numeric indices instead of dates)
            if (dataStart === '0' || dataStart === 0 || dataEnd === '0' || dataEnd === 0) return '-';
            try {
                const start = new Date(dataStart);
                const end = new Date(dataEnd);
                if (isNaN(start.getTime()) || isNaN(end.getTime())) return '-';
                // Check for epoch dates (indicates invalid data)
                if (start.getFullYear() < 2000 || end.getFullYear() < 2000) return '-';

                const days = (end - start) / (1000 * 60 * 60 * 24);
                // Negative or unreasonable days indicates bad data
                if (days < 0 || days > 3650) return '-';

                if (days <= 10) return '1 Week';
                if (days <= 20) return '2 Weeks';
                if (days <= 45) return '1 Month';
                if (days <= 100) return '3 Months';
                if (days <= 200) return '6 Months';
                if (days <= 300) return '9 Months';
                if (days <= 400) return '1 Year';
                if (days <= 800) return '2 Years';
                if (days <= 1200) return '3 Years';
                return '5 Years';
            } catch (e) {
                return '-';
            }
        }

        // Utility functions
        function formatTime() {
            const now = new Date();
            return now.toTimeString().split(' ')[0];
        }
        
        function addLog(message, type = 'normal') {
            const container = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';

            // Style based on type
            if (type === 'detail') {
                entry.style.color = '#888';
                entry.style.fontSize = '0.85em';
                entry.style.marginLeft = '20px';
                entry.innerHTML = message;
            } else {
                entry.innerHTML = `<span class="log-time">[${formatTime()}]</span> ${message}`;
            }

            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateDataStatus(status) {
            const container = document.getElementById('dataStatus');
            const loadingProgress = document.getElementById('dataLoadingProgress');
            const unifiedOverlay = document.getElementById('unifiedOverlay');

            if (status.loaded) {
                // Data loaded - update status
                container.innerHTML = `
                    <span class="status-badge success">✓ ${status.rows.toLocaleString()} candles loaded</span>
                    <p style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                        ${formatDateUK(status.start_date)} → ${formatDateUK(status.end_date)}
                    </p>
                `;

                // Hide loading progress
                loadingProgress.style.display = 'none';

                // Enable unified optimizer section
                if (unifiedOverlay) unifiedOverlay.style.display = 'none';

                // Show comparison card
                showComparisonCard();

                // Auto-populate backtest date range from loaded data
                updateBacktestDateRange(status.start_date, status.end_date);

                // Update data statistics
                if (status.stats) {
                    updateDataStats(status.stats);
                }

            } else if (status.fetching || (status.message && (status.message.includes('Fetching') || status.message.includes('Connecting') || status.message.includes('Downloading')))) {
                // Data is being fetched - show loading progress with REAL backend progress
                container.innerHTML = `<span class="status-badge warning">📥 ${status.message}</span>`;
                loadingProgress.style.display = 'block';

                // Use real backend progress instead of simulated
                const fill = document.getElementById('dataProgressFill');
                const msgEl = document.getElementById('dataProgressMessage');
                const pctEl = document.getElementById('dataProgressPercent');

                const progress = status.progress || 0;
                fill.style.width = `${progress}%`;
                msgEl.textContent = status.message;
                pctEl.textContent = `${progress}%`;

                // Stop any fake animation that might be running
                stopDataLoadingAnimation();

                // Keep overlay visible
                if (unifiedOverlay) unifiedOverlay.style.display = 'flex';

            } else if (status.message && status.message.startsWith('Error:')) {
                // Error occurred during fetch
                container.innerHTML = `<span class="status-badge danger">❌ ${status.message}</span>`;
                loadingProgress.style.display = 'none';
                stopDataLoadingAnimation();

                // Show overlay
                if (unifiedOverlay) unifiedOverlay.style.display = 'flex';

            } else {
                // No data loaded
                container.innerHTML = `<span class="status-badge danger">No data loaded</span>`;
                loadingProgress.style.display = 'none';

                // Show overlay
                if (unifiedOverlay) unifiedOverlay.style.display = 'flex';
            }
        }
        
        let dataLoadingAnimation = null;
        function animateDataLoading(message) {
            const fill = document.getElementById('dataProgressFill');
            const msgEl = document.getElementById('dataProgressMessage');
            const pctEl = document.getElementById('dataProgressPercent');
            
            msgEl.textContent = message;
            
            // Animate progress bar
            if (!dataLoadingAnimation) {
                let progress = 0;
                dataLoadingAnimation = setInterval(() => {
                    progress += Math.random() * 5;
                    if (progress > 90) progress = 90; // Cap at 90% until complete
                    fill.style.width = `${progress}%`;
                    pctEl.textContent = `${Math.round(progress)}%`;
                }, 500);
            }
        }
        
        function stopDataLoadingAnimation() {
            if (dataLoadingAnimation) {
                clearInterval(dataLoadingAnimation);
                dataLoadingAnimation = null;
            }
            const fill = document.getElementById('dataProgressFill');
            const pctEl = document.getElementById('dataProgressPercent');
            fill.style.width = '100%';
            pctEl.textContent = '100%';
        }

        // Format date to UK format DD/MM/YY
        function formatDateUK(dateStr) {
            if (!dateStr) return '-';
            const date = new Date(dateStr);
            const day = String(date.getDate()).padStart(2, '0');
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const year = String(date.getFullYear()).slice(-2);
            return `${day}/${month}/${year}`;
        }

        // Format currency with symbol based on currency type
        function formatCurrency(amount, currency = 'USD', decimals = 0) {
            const symbols = { 'USD': '$', 'GBP': '£', 'EUR': '€', 'USDT': '$' };
            const symbol = symbols[currency] || currency;
            const num = Number(amount) || 0;

            if (Math.abs(num) >= 1000) {
                return symbol + num.toLocaleString('en-GB', {
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                });
            }
            return symbol + num.toLocaleString('en-GB', {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            });
        }

        // Format dual currency display (USD | GBP)
        function formatDualCurrency(usdAmount, gbpAmount, showBoth = true) {
            if (!showBoth || gbpAmount === undefined || gbpAmount === null) {
                return formatCurrency(usdAmount, 'USD');
            }
            return `${formatCurrency(usdAmount, 'USD')} | ${formatCurrency(gbpAmount, 'GBP')}`;
        }

        // Get currency display based on report settings
        function getCurrencyDisplay(metrics, report) {
            const hasConversion = report?.currency_conversion_enabled ||
                                  (metrics?.display_currencies?.length > 1);
            const sourceCurrency = report?.source_currency || metrics?.source_currency || 'USD';

            return {
                hasConversion,
                sourceCurrency,
                formatPnl: (usd, gbp) => hasConversion
                    ? formatDualCurrency(usd, gbp)
                    : formatCurrency(usd, sourceCurrency)
            };
        }

        // Update backtest date range inputs from loaded data
        function updateBacktestDateRange(startDate, endDate) {
            console.log('updateBacktestDateRange called with:', startDate, endDate);

            const startInput = document.getElementById('backtestStartDate');
            const endInput = document.getElementById('backtestEndDate');
            const strategyDateRange = document.getElementById('strategyDateRange');

            if (startDate && startInput) {
                const start = startDate.split('T')[0];
                console.log('Setting start date to:', start);
                startInput.value = start;
            }
            if (endDate && endInput) {
                const end = endDate.split('T')[0];
                console.log('Setting end date to:', end);
                endInput.value = end;
            }

            // Update the strategy heading with date range (UK format)
            if (strategyDateRange && startDate && endDate) {
                strategyDateRange.textContent = `for ${formatDateUK(startDate)} → ${formatDateUK(endDate)}`;
            }
        }

        // Update data statistics display
        function updateDataStats(stats) {
            const statsSection = document.getElementById('dataStatsSection');
            if (!stats || !statsSection) return;

            statsSection.style.display = 'block';

            // Format currency based on pair (assume GBP for now)
            const formatPrice = (price) => {
                if (price >= 1000) {
                    return '£' + price.toLocaleString('en-GB', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
                }
                return '£' + price.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            };

            document.getElementById('statMinPrice').textContent = formatPrice(stats.min_price);
            document.getElementById('statAvgPrice').textContent = formatPrice(stats.avg_price);
            document.getElementById('statMaxPrice').textContent = formatPrice(stats.max_price);
            document.getElementById('statMaxUp').textContent = '+' + stats.max_up_swing.toFixed(2) + '%';
            document.getElementById('statMaxDown').textContent = stats.max_down_swing.toFixed(2) + '%';
            document.getElementById('statVolatility').textContent = stats.volatility_pct.toFixed(2) + '%';

            // Render daily trend timeline
            if (stats.daily_trends && stats.daily_trends.length > 0) {
                renderDailyTrendTimeline(stats.daily_trends);
            }
        }

        // Render the daily bull/bear timeline
        function renderDailyTrendTimeline(dailyTrends) {
            const container = document.getElementById('dailyTrendTimeline');
            const startLabel = document.getElementById('timelineStartDate');
            const endLabel = document.getElementById('timelineEndDate');

            if (!container || !dailyTrends || dailyTrends.length === 0) return;

            // Set date labels (UK format)
            startLabel.textContent = formatDateUK(dailyTrends[0].date);
            endLabel.textContent = formatDateUK(dailyTrends[dailyTrends.length - 1].date);

            const numDays = dailyTrends.length;

            // If more than 90 days, aggregate to weeks; if more than 365, aggregate to months
            let displayData = dailyTrends;
            let label = 'Daily';

            if (numDays > 365) {
                // Aggregate to months
                label = 'Monthly';
                const monthlyMap = {};
                dailyTrends.forEach(d => {
                    const month = d.date.substring(0, 7); // YYYY-MM
                    if (!monthlyMap[month]) monthlyMap[month] = { bull: 0, bear: 0, flat: 0 };
                    monthlyMap[month][d.trend]++;
                });
                displayData = Object.entries(monthlyMap).map(([month, counts]) => {
                    const dominant = counts.bull >= counts.bear ? (counts.bull > counts.flat ? 'bull' : 'flat') : (counts.bear > counts.flat ? 'bear' : 'flat');
                    return { date: month, trend: dominant, counts };
                });
            } else if (numDays > 90) {
                // Aggregate to weeks
                label = 'Weekly';
                const weeklyMap = {};
                dailyTrends.forEach(d => {
                    const date = new Date(d.date);
                    const weekStart = new Date(date);
                    weekStart.setDate(date.getDate() - date.getDay());
                    const weekKey = weekStart.toISOString().split('T')[0];
                    if (!weeklyMap[weekKey]) weeklyMap[weekKey] = { bull: 0, bear: 0, flat: 0 };
                    weeklyMap[weekKey][d.trend]++;
                });
                displayData = Object.entries(weeklyMap).map(([week, counts]) => {
                    const dominant = counts.bull >= counts.bear ? (counts.bull > counts.flat ? 'bull' : 'flat') : (counts.bear > counts.flat ? 'bear' : 'flat');
                    return { date: week, trend: dominant, counts };
                });
            }

            // Calculate bar width - ensure bars fill container
            const containerWidth = container.offsetWidth || 400;
            const barWidth = Math.max(4, Math.floor((containerWidth - (displayData.length * 2)) / displayData.length));

            // Build timeline HTML
            let html = '';
            displayData.forEach((day, index) => {
                let color, title;
                const dateFormatted = formatDateUK(day.date);
                if (day.trend === 'bull') {
                    color = '#10b981'; // green
                    title = `${dateFormatted}: Bull 📈`;
                } else if (day.trend === 'bear') {
                    color = '#ef4444'; // red
                    title = `${dateFormatted}: Bear 📉`;
                } else {
                    color = 'rgba(255,255,255,0.4)';
                    title = `${dateFormatted}: Flat ➡️`;
                }

                if (day.counts) {
                    title += ` (${day.counts.bull}↑ ${day.counts.bear}↓ ${day.counts.flat}→)`;
                }

                html += `<div style="width:${barWidth}px; height:24px; background:${color}; border-radius:2px; cursor:pointer;" title="${title}"></div>`;
            });

            container.innerHTML = html;

            // Update label
            const labelEl = container.parentElement.querySelector('div:first-child');
            if (labelEl) {
                labelEl.textContent = `${label} Trend Timeline (${numDays} days)`;
            }

            // Add summary below
            const bullDays = dailyTrends.filter(d => d.trend === 'bull').length;
            const bearDays = dailyTrends.filter(d => d.trend === 'bear').length;
            const flatDays = dailyTrends.filter(d => d.trend === 'flat').length;

            // Update the legend with counts
            const legendContainer = container.parentElement.querySelector('div:last-child span:nth-child(2)');
            if (legendContainer) {
                legendContainer.innerHTML = `
                    <span style="color:#10b981">● Bull (${bullDays})</span>
                    <span style="color:#ef4444">● Bear (${bearDays})</span>
                    <span style="color:rgba(255,255,255,0.6)">● Flat (${flatDays})</span>
                `;
            }
        }

        // Get current backtest date range settings for Pine Script generation
        function getBacktestDateRange() {
            const enabled = document.getElementById('enableDateRange').checked;
            if (!enabled) return null;

            const startDate = document.getElementById('backtestStartDate').value;
            const startTime = document.getElementById('backtestStartTime').value || '00:00';
            const endDate = document.getElementById('backtestEndDate').value;
            const endTime = document.getElementById('backtestEndTime').value || '23:59';

            if (!startDate || !endDate) return null;

            return {
                enabled: true,
                startDate: startDate,
                startTime: startTime,
                endDate: endDate,
                endTime: endTime,
                // Format for Pine Script timestamp function
                startTimestamp: `${startDate} ${startTime}`,
                endTimestamp: `${endDate} ${endTime}`
            };
        }

        // Toggle date range inputs visibility
        document.addEventListener('DOMContentLoaded', function() {
            const checkbox = document.getElementById('enableDateRange');
            const inputs = document.getElementById('dateRangeInputs');

            checkbox.addEventListener('change', function() {
                inputs.style.opacity = this.checked ? '1' : '0.5';
                inputs.style.pointerEvents = this.checked ? 'auto' : 'none';
            });
        });

        // Trading pair options - Binance USDT pairs only
        // Use BINANCE:SYMBOL on TradingView for matching charts
        const PAIR_OPTIONS = {
            binance: [
                { value: 'BTCUSDT', label: 'BTC/USDT (Bitcoin)' },
                { value: 'ETHUSDT', label: 'ETH/USDT (Ethereum)' },
                { value: 'BNBUSDT', label: 'BNB/USDT (Binance Coin)' },
                { value: 'XRPUSDT', label: 'XRP/USDT (Ripple)' },
                { value: 'SOLUSDT', label: 'SOL/USDT (Solana)' },
                { value: 'ADAUSDT', label: 'ADA/USDT (Cardano)' },
                { value: 'DOGEUSDT', label: 'DOGE/USDT (Dogecoin)' },
                { value: 'DOTUSDT', label: 'DOT/USDT (Polkadot)' },
                { value: 'MATICUSDT', label: 'MATIC/USDT (Polygon)' },
                { value: 'LTCUSDT', label: 'LTC/USDT (Litecoin)' },
                { value: 'AVAXUSDT', label: 'AVAX/USDT (Avalanche)' },
                { value: 'LINKUSDT', label: 'LINK/USDT (Chainlink)' },
            ],
        };

        // Timeframe options - All Binance supported intervals
        const INTERVAL_OPTIONS = {
            binance: [
                { value: '1', label: '1 minute' },
                { value: '3', label: '3 minutes' },
                { value: '5', label: '5 minutes' },
                { value: '15', label: '15 minutes' },
                { value: '30', label: '30 minutes' },
                { value: '60', label: '1 hour' },
                { value: '120', label: '2 hours' },
                { value: '240', label: '4 hours' },
                { value: '1440', label: '1 day' },
            ],
        };

        // Historical period options - Binance has years of data
        const PERIOD_OPTIONS = {
            binance: {
                default: [
                    { value: '0.25', label: '1 week' },
                    { value: '1', label: '1 month' },
                    { value: '3', label: '3 months' },
                    { value: '6', label: '6 months' },
                    { value: '12', label: '12 months' },
                    { value: '24', label: '24 months' },
                ]
            },
        };

        function updatePairOptions() {
            const source = document.getElementById('sourceSelect').value;
            const pairSelect = document.getElementById('pairSelect');

            // Update pairs based on source
            pairSelect.innerHTML = PAIR_OPTIONS[source].map(p =>
                `<option value="${p.value}">${p.label}</option>`
            ).join('');

            // Also update intervals and periods for this source
            updateIntervalOptions();
        }

        function updateIntervalOptions() {
            const source = document.getElementById('sourceSelect').value;
            const intervalSelect = document.getElementById('intervalSelect');
            const currentInterval = intervalSelect.value;

            // Update intervals based on source
            const intervals = INTERVAL_OPTIONS[source] || INTERVAL_OPTIONS.binance;
            intervalSelect.innerHTML = intervals.map(i =>
                `<option value="${i.value}">${i.label}</option>`
            ).join('');

            // Try to keep current selection if still valid
            if (intervals.some(i => i.value === currentInterval)) {
                intervalSelect.value = currentInterval;
            }

            // Update period options for the selected interval
            updatePeriodOptions();
        }

        function updatePeriodOptions() {
            const source = document.getElementById('sourceSelect').value;
            const interval = document.getElementById('intervalSelect').value;
            const monthsSelect = document.getElementById('monthsSelect');
            const currentMonths = monthsSelect.value;

            // Get period options for this source (use default for all intervals)
            const sourceOptions = PERIOD_OPTIONS[source] || PERIOD_OPTIONS.binance;
            const periods = sourceOptions.default || sourceOptions[interval] || sourceOptions['60'];

            monthsSelect.innerHTML = periods.map(p =>
                `<option value="${p.value}">${p.label}</option>`
            ).join('');

            // Try to keep current selection if still valid
            if (periods.some(p => p.value === currentMonths)) {
                monthsSelect.value = currentMonths;
            }
        }
        
        // API Functions
        async function fetchData() {
            const source = document.getElementById('sourceSelect').value;
            const pair = document.getElementById('pairSelect').value;
            const months = document.getElementById('monthsSelect').value;
            const interval = document.getElementById('intervalSelect').value;
            const btn = document.getElementById('fetchDataBtn');
            const loadingProgress = document.getElementById('dataLoadingProgress');
            
            const sourceNames = { binance: 'Binance', kraken: 'Kraken', yfinance: 'Yahoo Finance', cryptocompare: 'CryptoCompare' };
            const sourceName = sourceNames[source] || source;
            
            // Show loading UI
            btn.disabled = true;
            btn.innerHTML = `<div class="spinner"></div> Fetching...`;
            document.getElementById('uploadCsvBtn').disabled = true;
            
            // Show and reset progress (backend will provide real progress)
            loadingProgress.style.display = 'block';
            document.getElementById('dataProgressFill').style.width = '0%';
            document.getElementById('dataProgressMessage').textContent = `Connecting to ${sourceName}...`;
            document.getElementById('dataProgressPercent').textContent = '0%';

            // Stop any existing fake animation - backend will provide real progress
            stopDataLoadingAnimation();

            addLog(`Fetching ${months} months of ${interval}m ${pair} from ${sourceName}...`);
            
            try {
                const response = await fetch('/api/fetch-data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source, pair, interval: parseInt(interval), months: parseFloat(months) })
                });
                
                const data = await response.json();
                document.getElementById('dataProgressMessage').textContent = data.message;
                addLog(data.message);
                
                // Start fast polling for data fetch
                startFastPolling();
                
            } catch (error) {
                addLog(`Error: ${error.message}`);
                btn.disabled = false;
                btn.innerHTML = 'Fetch Data';
                document.getElementById('uploadCsvBtn').disabled = false;
                loadingProgress.style.display = 'none';
                stopDataLoadingAnimation();
            }
        }
        
        async function loadExistingData() {
            const btn = document.getElementById('fetchDataBtn');
            const loadingProgress = document.getElementById('dataLoadingProgress');
            
            btn.disabled = true;
            loadingProgress.style.display = 'block';
            document.getElementById('dataProgressMessage').textContent = 'Loading existing data...';
            document.getElementById('dataProgressFill').style.width = '50%';
            
            addLog('Loading existing data...');
            
            try {
                const response = await fetch('/api/load-existing-data', { method: 'POST' });
                const data = await response.json();
                
                document.getElementById('dataProgressFill').style.width = '100%';
                document.getElementById('dataProgressPercent').textContent = '100%';
                
                setTimeout(() => {
                    updateDataStatus(data);
                    addLog(`Loaded ${data.rows} candles`);
                    btn.disabled = false;
                    btn.innerHTML = 'Fetch Data';
                }, 500);
                
            } catch (error) {
                addLog(`Error: ${error.message}`);
                btn.disabled = false;
                loadingProgress.style.display = 'none';
            }
        }

        // ============ UNIFIED OPTIMIZATION ============

        let unifiedPolling = null;
        let sseEventSource = null;
        let streamedResults = [];  // Track streamed results

        async function runUnifiedOptimization() {
            const btn = document.getElementById('unifiedBtn');
            const capital = parseFloat(document.getElementById('unifiedCapital').value) || 1000;
            const riskPercent = parseFloat(document.getElementById('unifiedRisk').value) || 75;
            const nTrials = parseInt(document.getElementById('unifiedTrials').value) || 300;

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Finding profitable strategies...';

            document.getElementById('unifiedProgress').style.display = 'block';
            document.getElementById('unifiedResults').innerHTML = createStreamingResultsContainer();
            document.getElementById('unifiedProgressFill').style.width = '0%';
            document.getElementById('unifiedMessage').textContent = 'Starting multi-engine optimization...';

            // Reset streamed results
            streamedResults = [];

            addLog(`🔬 Running dual-engine optimization: TradingView + Native (TA-Lib)...`);

            try {
                // Get backtest date range settings
                const dateRange = getBacktestDateRange();

                const response = await fetch('/api/run-unified', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        capital: capital,
                        risk_percent: riskPercent,
                        n_trials: nTrials,
                        engine: 'all',  // Always run all engines for comparison
                        date_range: dateRange
                    })
                });

                const data = await response.json();
                addLog(data.message);

                // Start SSE streaming for real-time results
                startSSEStream();

                // Start polling for unified status (for progress updates)
                startUnifiedPolling();

            } catch (error) {
                addLog(`Error: ${error.message}`);
                btn.disabled = false;
                btn.innerHTML = 'Find Best Strategies';
            }
        }

        function createStreamingResultsContainer() {
            return `
                <div id="streamingResults" style="margin-bottom: 1.5rem;">
                    <h4 style="color: var(--accent-secondary); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span class="pulse-dot"></span> Live Results (streaming)
                    </h4>
                    <div id="streamingTable" style="max-height: 400px; overflow-y: auto; background: var(--bg-secondary); border-radius: 8px; padding: 0.5rem;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                            <thead style="position: sticky; top: 0; background: var(--bg-card);">
                                <tr style="color: var(--text-muted); text-align: left;">
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">#</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Eng</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Strategy</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">TP/SL</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Trades</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Win%</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">PF</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Return</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);" title="Max Drawdown">DD</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">B&H</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);" title="Consistency Score">Cons</th>
                                    <th style="padding: 0.5rem; border-bottom: 1px solid var(--border);">Score</th>
                                </tr>
                            </thead>
                            <tbody id="streamingTableBody"></tbody>
                        </table>
                    </div>
                    <p id="streamingCount" style="color: var(--text-muted); margin-top: 0.5rem; font-size: 0.85rem;">
                        Waiting for results...
                    </p>
                </div>
            `;
        }

        function startSSEStream() {
            // Close any existing connection
            if (sseEventSource) {
                sseEventSource.close();
            }

            sseEventSource = new EventSource('/api/unified-stream');

            sseEventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'connected') {
                        addLog('📡 Real-time stream connected');
                    } else if (data.type === 'strategy_result') {
                        handleStreamedResult(data);
                    } else if (data.type === 'tuning_progress') {
                        // Phase 2: Tuning progress update with detailed info
                        const params = data.params_being_tuned || [];
                        const numCombos = data.num_combinations || 0;
                        const paramStr = params.length > 0 ? params.join(', ') : 'none';

                        // Main progress message
                        const msg = `🔧 Tuning ${data.current}/${data.total}: ${data.strategy} (${data.direction})`;
                        document.getElementById('unifiedProgressMessage').textContent = msg;

                        // Detailed log with parameter info
                        if (params.length > 0) {
                            const detailMsg = `   └─ Testing ${numCombos} combinations: ${paramStr}`;
                            addLog(msg);
                            addLog(detailMsg, 'detail');
                        } else {
                            addLog(`${msg} - no tunable params`);
                        }
                    } else if (data.type === 'tuning_result') {
                        // Phase 2: Individual tuning result
                        handleTuningResult(data);
                    } else if (data.type === 'complete') {
                        addLog('✅ Streaming complete');
                        sseEventSource.close();
                        sseEventSource = null;
                    }
                } catch (e) {
                    console.error('SSE parse error:', e);
                }
            };

            sseEventSource.onerror = (error) => {
                console.error('SSE error:', error);
                // Don't close on error, let it reconnect
            };
        }

        function handleStreamedResult(data) {
            streamedResults.push(data);

            // Sort by COMPOSITE SCORE (balanced ranking)
            streamedResults.sort((a, b) => (b.composite_score || 0) - (a.composite_score || 0));

            // Update the streaming table
            const tbody = document.getElementById('streamingTableBody');
            if (tbody) {
                tbody.innerHTML = streamedResults.slice(0, 20).map((r, idx) => {
                    const wins = r.wins || 0;
                    const losses = r.losses || 0;
                    const total = r.total_trades || (wins + losses);
                    const returnPct = r.total_pnl_percent || 0;
                    const returnGBP = r.total_pnl || 0;
                    const score = r.composite_score || 0;
                    const vsBH = r.vs_buy_hold || 0;
                    const beatsBH = r.beats_buy_hold || false;
                    const maxDD = r.max_drawdown_percent || 0;
                    const consistency = r.consistency_score || 50;
                    const periodMetrics = r.period_metrics || {};

                    // Engine tag colors (TV = TradingView, NT = Native/TA-Lib)
                    const engineColors = { 'TV': '#2962FF', 'NT': '#22c55e' };
                    const engineTag = r.engine_tag || 'TV';
                    const engineColor = engineColors[engineTag] || '#666';

                    // DD color: green < 5%, yellow 5-15%, red > 15%
                    const ddColor = maxDD < 5 ? 'var(--success)' : maxDD < 15 ? 'var(--warning)' : 'var(--danger)';

                    // Consistency color: green >= 70, yellow >= 50, red < 50
                    const consColor = consistency >= 70 ? 'var(--success)' : consistency >= 50 ? 'var(--warning)' : 'var(--danger)';

                    // Build period metrics row
                    const availablePeriods = Object.keys(periodMetrics);
                    let periodRow = '';
                    if (availablePeriods.length > 0) {
                        const validPeriods = availablePeriods.filter(p => periodMetrics[p] && periodMetrics[p].trades > 0);
                        if (validPeriods.length > 0) {
                            const periodBoxes = validPeriods.map(p => {
                                const pm = periodMetrics[p];
                                const wrColor = pm.win_rate >= 60 ? 'var(--success)' : pm.win_rate >= 50 ? 'var(--warning)' : 'var(--danger)';
                                const pnlColor = pm.pnl >= 0 ? 'var(--success)' : 'var(--danger)';
                                const pnlPct = pm.pnl_pct || 0;
                                const pnlPctColor = pnlPct >= 0 ? 'var(--success)' : 'var(--danger)';
                                return `<div style="flex: 1; text-align: center; padding: 0.4rem 0.5rem; min-width: 100px;">
                                    <div style="color: var(--text-muted); font-weight: 600; font-size: 0.8rem; margin-bottom: 0.2rem;">${p.toUpperCase()}</div>
                                    <div style="font-size: 0.9rem; font-weight: 500;"><span style="color: ${wrColor};">${pm.win_rate?.toFixed(0) || '-'}%</span> <span style="color: ${pnlColor};">${pm.pnl >= 0 ? '+' : ''}£${pm.pnl?.toFixed(0) || '0'}</span></div>
                                    <div style="font-size: 0.75rem; margin-top: 0.1rem;"><span style="color: ${pnlPctColor};">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%</span> <span style="color: var(--text-muted);">${pm.trades} trades</span></div>
                                </div>`;
                            }).join('');

                            periodRow = `
                            <tr class="period-row" style="background: var(--bg-secondary);">
                                <td colspan="12" style="padding: 0.4rem 1rem;">
                                    <div style="display: flex; justify-content: flex-start; gap: 0.5rem; flex-wrap: wrap;">
                                        ${periodBoxes}
                                    </div>
                                </td>
                            </tr>`;
                        }
                    }

                    return `
                    <tr style="border-bottom: 1px solid var(--border-subtle); ${idx === 0 ? 'background: rgba(34, 197, 94, 0.1);' : ''}">
                        <td style="padding: 0.4rem 0.5rem; color: ${idx < 3 ? 'var(--success)' : 'var(--text-muted)'};">${idx + 1}</td>
                        <td style="padding: 0.4rem 0.5rem;">
                            <span style="background: ${engineColor}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">${engineTag}</span>
                        </td>
                        <td style="padding: 0.4rem 0.5rem;">
                            <span style="color: var(--text-primary);">${r.strategy_name}</span>
                            <span style="color: var(--text-muted); font-size: 0.75rem; display: block;">${r.strategy_category}</span>
                        </td>
                        <td style="padding: 0.4rem 0.5rem; color: #3b82f6; font-size: 0.8rem;">${(r.params?.tp_percent || 1.0).toFixed(1)}/${(r.params?.sl_percent || 3.0).toFixed(1)}%</td>
                        <td style="padding: 0.4rem 0.5rem;">
                            <span style="color: var(--success);">${wins}</span>/<span style="color: var(--danger);">${losses}</span>
                            <span style="color: var(--text-muted); font-size: 0.7rem;"> (${total})</span>
                        </td>
                        <td style="padding: 0.4rem 0.5rem; color: ${r.win_rate >= 60 ? 'var(--success)' : r.win_rate >= 50 ? 'var(--warning)' : 'var(--danger)'};">${r.win_rate.toFixed(1)}%</td>
                        <td style="padding: 0.4rem 0.5rem; color: var(--accent-secondary);">${(r.profit_factor || 0).toFixed(2)}</td>
                        <td style="padding: 0.4rem 0.5rem;">
                            <span style="color: ${returnPct >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight: 500;">${returnPct.toFixed(1)}%</span>
                            <span style="color: var(--text-muted); font-size: 0.65rem; display: block;">£${returnGBP.toFixed(0)}</span>
                        </td>
                        <td style="padding: 0.4rem 0.5rem; color: ${ddColor};">-${maxDD.toFixed(1)}%</td>
                        <td style="padding: 0.4rem 0.5rem; color: ${(returnPct - vsBH) >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight: 500;">
                            ${(returnPct - vsBH) >= 0 ? '+' : ''}${(returnPct - vsBH).toFixed(1)}%
                        </td>
                        <td style="padding: 0.4rem 0.5rem; color: ${consColor};">${consistency.toFixed(0)}</td>
                        <td style="padding: 0.4rem 0.5rem; color: var(--accent-primary); font-weight: 600;">${score.toFixed(0)}</td>
                    </tr>
                    ${periodRow}
                `}).join('');
            }

            // Update count
            const countEl = document.getElementById('streamingCount');
            if (countEl) {
                const profitable = streamedResults.filter(r => r.total_pnl > 0).length;
                const beatsBH = streamedResults.filter(r => r.beats_buy_hold).length;
                countEl.textContent = `${profitable} profitable | ${beatsBH} beat Buy & Hold (ranked by composite score)`;
            }
        }

        // Phase 2: Store tuning results - keyed by "strategyName_direction" for quick lookup
        let tuningResults = [];
        let tuningResultsMap = {};

        function handleTuningResult(data) {
            // Store the tuning result
            if (data.tuning) {
                tuningResults.push(data.tuning);

                // Build a key for quick lookup (e.g., "rsi_extreme_long_long" or "rsi_extreme_long")
                const key = `${data.tuning.strategy_name}_${data.tuning.direction}`.toLowerCase();
                tuningResultsMap[key] = data.tuning;

                // Also store by entry_rule if available
                if (data.tuning.entry_rule) {
                    const altKey = `${data.tuning.entry_rule}_${data.tuning.direction}`.toLowerCase();
                    tuningResultsMap[altKey] = data.tuning;
                }

                // Log the tuning result
                const t = data.tuning;
                const improved = t.is_improved ? '✅' : '➖';
                const scoreChange = t.improvements?.score || 0;
                addLog(`${improved} ${t.strategy_name} (${t.direction}): Score ${t.before?.score?.toFixed(1)} → ${t.after?.score?.toFixed(1)} (${scoreChange >= 0 ? '+' : ''}${scoreChange.toFixed(1)}%)`);

                // Update streaming table to show tuning column
                updateStreamingTableWithTuning();
            }
        }

        function updateStreamingTableWithTuning() {
            // This can be called to refresh the streaming table with tuning info
            // For now we just log - the full display happens in displayUnifiedReport
        }

        // Helper to find tuning result for a strategy
        function findTuningForStrategy(strat) {
            // Try multiple key patterns to find the matching tuning result
            const stratName = (strat.strategy_name || '').toLowerCase();
            const direction = (strat.direction || 'long').toLowerCase();
            const entryRule = (strat.entry_rule || strat.params?.entry_rule || '').toLowerCase();

            // Try direct match first
            let tuning = tuningResultsMap[`${stratName}_${direction}`];
            if (tuning) return tuning;

            // Try entry_rule match
            if (entryRule) {
                tuning = tuningResultsMap[`${entryRule}_${direction}`];
                if (tuning) return tuning;
            }

            // Try finding in array with partial matching
            return tuningResults.find(t => {
                const tName = (t.strategy_name || '').toLowerCase();
                const tDir = (t.direction || '').toLowerCase();
                return (tName === stratName || tName.includes(entryRule)) && tDir === direction;
            });
        }

        // Load tuning results from report payload
        function loadTuningResultsFromReport(report) {
            const tuningData = report.tuning_results || [];
            tuningResults = tuningData;
            tuningResultsMap = {};

            tuningData.forEach(t => {
                const key = `${t.strategy_name}_${t.direction}`.toLowerCase();
                tuningResultsMap[key] = t;

                if (t.entry_rule) {
                    const altKey = `${t.entry_rule}_${t.direction}`.toLowerCase();
                    tuningResultsMap[altKey] = t;
                }
            });

            if (tuningData.length > 0) {
                addLog(`📊 Loaded ${tuningData.length} tuning results from report`);
            }
        }

        // Helper function to render tuning comparison UI for a strategy
        function renderTuningSection(tuning) {
            if (!tuning || !tuning.tuned_params || Object.keys(tuning.tuned_params).length === 0) {
                return '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.5rem;">No tunable parameters for this strategy</div>';
            }

            const improved = tuning.is_improved;
            const paramsChanged = tuning.params_changed;

            // Determine status message and styling
            let statusMessage = '';
            let borderColor = 'var(--text-muted)';
            let bgColor = 'rgba(100, 100, 100, 0.05)';

            if (improved && paramsChanged) {
                statusMessage = '<span style="color: var(--success);">✓ Improved!</span>';
                borderColor = 'var(--success)';
                bgColor = 'rgba(34, 197, 94, 0.05)';
            } else if (paramsChanged && !improved) {
                statusMessage = '<span style="color: var(--warning);">Tested - No improvement</span>';
                borderColor = 'var(--warning)';
                bgColor = 'rgba(234, 179, 8, 0.05)';
            } else {
                // No params changed - defaults are optimal
                statusMessage = '<span style="color: var(--text-muted);">Defaults optimal</span>';
            }

            let paramRows = '';
            for (const [param, value] of Object.entries(tuning.tuned_params)) {
                const defaultVal = tuning.default_params?.[param];
                const changed = defaultVal !== undefined && defaultVal !== value;
                const changeIndicator = changed ?
                    `<span style="color: var(--success);">✓ Changed</span>` :
                    `<span style="color: var(--text-muted);">-</span>`;

                paramRows += `
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 0.3rem 0.5rem; color: var(--text-secondary);">${param}</td>
                        <td style="padding: 0.3rem 0.5rem; text-align: center; color: var(--text-muted);">${defaultVal ?? '-'}</td>
                        <td style="padding: 0.3rem 0.5rem; text-align: center; font-weight: 600; color: ${changed ? 'var(--success)' : 'var(--text-primary)'};">${value}</td>
                        <td style="padding: 0.3rem 0.5rem; text-align: center; font-size: 0.75rem;">${changeIndicator}</td>
                    </tr>
                `;
            }

            return `
                <div style="background: ${bgColor}; border-radius: 8px; padding: 0.75rem; border-left: 3px solid ${borderColor}; margin-top: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-primary);">
                            🔧 Indicator Tuning ${statusMessage}
                        </span>
                    </div>

                    <table style="width: 100%; font-size: 0.75rem; border-collapse: collapse;">
                        <thead>
                            <tr style="background: var(--bg-card); border-bottom: 1px solid var(--border);">
                                <th style="padding: 0.3rem 0.5rem; text-align: left; color: var(--text-muted);">Parameter</th>
                                <th style="padding: 0.3rem 0.5rem; text-align: center; color: var(--text-muted);">Before</th>
                                <th style="padding: 0.3rem 0.5rem; text-align: center; color: var(--text-muted);">After</th>
                                <th style="padding: 0.3rem 0.5rem; text-align: center; color: var(--text-muted);">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${paramRows}
                        </tbody>
                    </table>

                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.75rem; font-size: 0.75rem;">
                        <div style="text-align: center; padding: 0.4rem; background: var(--bg-card); border-radius: 4px;">
                            <div style="color: var(--text-muted);">Score</div>
                            <div style="font-weight: 600;">
                                ${tuning.before?.score?.toFixed(1) || 'N/A'} →
                                <span style="color: ${tuning.improvements?.score > 0 ? 'var(--success)' : tuning.improvements?.score < 0 ? 'var(--danger)' : 'var(--text-primary)'};">
                                    ${tuning.after?.score?.toFixed(1) || 'N/A'}
                                </span>
                            </div>
                            <div style="color: ${tuning.improvements?.score > 0 ? 'var(--success)' : tuning.improvements?.score < 0 ? 'var(--danger)' : 'var(--text-muted)'}; font-size: 0.7rem;">
                                ${tuning.improvements?.score > 0 ? '+' : ''}${tuning.improvements?.score?.toFixed(1) || 0}%
                            </div>
                        </div>
                        <div style="text-align: center; padding: 0.4rem; background: var(--bg-card); border-radius: 4px;">
                            <div style="color: var(--text-muted);">Win Rate</div>
                            <div style="font-weight: 600;">
                                ${tuning.before?.win_rate?.toFixed(1) || 'N/A'}% →
                                <span style="color: ${tuning.improvements?.win_rate > 0 ? 'var(--success)' : tuning.improvements?.win_rate < 0 ? 'var(--danger)' : 'var(--text-primary)'};">
                                    ${tuning.after?.win_rate?.toFixed(1) || 'N/A'}%
                                </span>
                            </div>
                        </div>
                        <div style="text-align: center; padding: 0.4rem; background: var(--bg-card); border-radius: 4px;">
                            <div style="color: var(--text-muted);">Profit Factor</div>
                            <div style="font-weight: 600;">
                                ${tuning.before?.profit_factor?.toFixed(2) || 'N/A'} →
                                <span style="color: ${tuning.improvements?.profit_factor > 0 ? 'var(--success)' : tuning.improvements?.profit_factor < 0 ? 'var(--danger)' : 'var(--text-primary)'};">
                                    ${tuning.after?.profit_factor?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                        </div>
                        <div style="text-align: center; padding: 0.4rem; background: var(--bg-card); border-radius: 4px;">
                            <div style="color: var(--text-muted);">P&L</div>
                            <div style="font-weight: 600;">
                                ${tuning.before?.pnl_percent?.toFixed(1) || 'N/A'}% →
                                <span style="color: ${tuning.improvements?.pnl > 0 ? 'var(--success)' : tuning.improvements?.pnl < 0 ? 'var(--danger)' : 'var(--text-primary)'};">
                                    ${tuning.after?.pnl_percent?.toFixed(1) || 'N/A'}%
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function startUnifiedPolling() {
            // DISABLED - WebSocket provides all updates
            // HTTP polling is wasteful and drains resources
            console.log('[Unified] Polling disabled - using WebSocket');
        }
        
        // Store the current report globally for engine switching
        let currentReport = null;
        let selectedEngine = 'tradingview';

        function displayUnifiedReport(report) {
            const container = document.getElementById('unifiedResults');
            currentReport = report;

            // Load tuning results from report
            loadTuningResultsFromReport(report);

            // Show the comparison card once we have results
            showComparisonCard();

            // Check if this is a comparison mode report
            const isComparisonMode = report.mode === 'comparison' && report.engine_reports;

            if (isComparisonMode) {
                displayComparisonReport(report, container);
            } else {
                displaySingleEngineReport(report, container);
            }
        }

        function displayComparisonReport(report, container) {
            const engineReports = report.engine_reports || {};
            const engines = report.engines || ['tradingview', 'native'];
            const bestEngine = report.best_engine || 'tradingview';

            // Combine all strategies from all engines with engine tag
            let allStrategies = [];
            engines.forEach(eng => {
                const engReport = engineReports[eng] || {};
                const top10 = engReport.top_10 || [];
                const engineTag = eng === 'tradingview' ? 'TV' : 'NT';
                top10.forEach((strat, originalRank) => {
                    allStrategies.push({
                        ...strat,
                        engine: eng,
                        engine_tag: engineTag,
                        original_rank: originalRank + 1  // Store original rank (1-indexed)
                    });
                });
            });

            // Sort by composite score (check both top-level and metrics)
            allStrategies.sort((a, b) => {
                const scoreA = a.metrics?.composite_score || a.composite_score || 0;
                const scoreB = b.metrics?.composite_score || b.composite_score || 0;
                return scoreB - scoreA;
            });

            // Take top 10 overall
            const topStrategies = allStrategies.slice(0, 10);

            // Engine tag colors
            const engineColors = { 'TV': '#2962FF', 'NT': '#22c55e' };

            let html = `
                <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-hover); border-radius: 8px;">
                    <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                        <strong>Data:</strong> ${report.data_rows?.toLocaleString() || 'N/A'} candles |
                        <strong>Symbol:</strong> ${report.symbol || 'N/A'} |
                        <strong>Timeframe:</strong> ${report.timeframe || 'N/A'} |
                        <span style="color: var(--text-muted);">Engine tags: </span>
                        <span style="background: #2962FF; color: white; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.7rem;">TV</span>=TradingView
                        <span style="background: #22c55e; color: white; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.7rem; margin-left: 0.5rem;">NT</span>=Native (TA-Lib)
                    </p>
                </div>

                <h3 style="color: var(--accent-primary); margin-bottom: 1rem;">🎯 Top 10 Strategies</h3>
            `;

            topStrategies.forEach((strat, idx) => {
                const m = strat.metrics || {};
                const engineTag = strat.engine_tag;
                const engineColor = engineColors[engineTag] || '#666';
                const engine = strat.engine;

                html += `
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem; ${idx === 0 ? 'border-color: var(--success); background: linear-gradient(135deg, rgba(34, 197, 94, 0.05), transparent);' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-weight: bold; color: ${idx < 3 ? 'var(--success)' : 'var(--text-muted)'};">#${idx + 1}</span>
                                <span style="background: ${engineColor}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">${engineTag}</span>
                                <span style="color: var(--text-primary); font-weight: 600;">${strat.strategy_name}</span>
                                <span style="color: var(--text-muted); font-size: 0.8rem;">${strat.strategy_category || ''}</span>
                            </div>
                            <span style="text-align: right;">
                                <span style="color: ${(m.total_pnl || 0) >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight: bold; font-size: 1.1rem;">${(strat.total_pnl_percent || m.total_pnl_percent || 0).toFixed(1)}%</span>
                                <span style="color: var(--text-muted); font-size: 0.75rem; display: block;">${m.total_pnl_gbp !== undefined && m.total_pnl_gbp !== m.total_pnl ? formatDualCurrency(m.total_pnl, m.total_pnl_gbp) : formatCurrency(m.total_pnl || 0, m.source_currency || 'GBP')}</span>
                            </span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; font-size: 0.85rem; margin-bottom: 0.75rem;">
                            <div>
                                <span style="color: var(--text-muted);">TP/SL:</span><br>
                                <span style="color: #3b82f6;">${(strat.params?.tp_percent || 1.0).toFixed(1)}/${(strat.params?.sl_percent || 3.0).toFixed(1)}%</span>
                            </div>
                            <div>
                                <span style="color: var(--text-muted);">Trades:</span><br>
                                <span style="color: var(--success);">${m.wins || 0}</span>/<span style="color: var(--danger);">${m.losses || 0}</span>
                            </div>
                            <div>
                                <span style="color: var(--text-muted);">Win Rate:</span><br>
                                <span style="color: ${(m.win_rate || 0) >= 60 ? 'var(--success)' : (m.win_rate || 0) >= 50 ? 'var(--warning)' : 'var(--danger)'};">${(m.win_rate || 0).toFixed(1)}%</span>
                            </div>
                            <div>
                                <span style="color: var(--text-muted);">Profit Factor:</span><br>
                                <span style="color: var(--accent-secondary);">${(m.profit_factor || 0).toFixed(2)}</span>
                            </div>
                            <div>
                                <span style="color: var(--text-muted);">Score:</span><br>
                                <span style="color: var(--accent-primary); font-weight: 600;">${(m.composite_score || strat.composite_score || 0).toFixed(0)}</span>
                            </div>
                        </div>
                        ${renderTuningSectionForCard(strat)}
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem;">
                            <button onclick="exportStrategyPineScriptWithEngine(${strat.original_rank}, '${engine}')" class="btn btn-success" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                                📋 Copy Pine
                            </button>
                            <button onclick="downloadStrategyPineScriptWithEngine(${strat.original_rank}, '${engine}')" class="btn btn-secondary" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                                ⬇️ Download
                            </button>
                            <button onclick="openStrategyInTradingViewWithEngine(${strat.original_rank}, '${engine}')" class="btn" style="font-size: 0.8rem; padding: 0.4rem 0.8rem; background: linear-gradient(135deg, #2962FF, #00BCD4);">
                                📈 View in TradingView
                            </button>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }

        function renderStrategyCard(strat, idx, engine) {
            const metrics = strat.metrics || {};
            const params = strat.params || {};
            const isTopPick = idx === 0;
            const isTop3 = idx < 3;
            const borderColor = isTopPick ? 'var(--success)' : (isTop3 ? 'var(--accent-primary)' : 'var(--border)');
            const hasOpenPosition = strat.has_open_position || false;
            const openPosition = strat.open_position || null;

            // Build open position warning HTML
            let openPositionWarning = '';
            if (hasOpenPosition && openPosition) {
                const isLosing = openPosition.unrealized_pnl < 0;
                openPositionWarning = `
                    <div style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: ${isLosing ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)'}; border: 1px solid ${isLosing ? 'var(--danger)' : 'var(--success)'}; border-radius: 6px; font-size: 0.75rem;">
                        <span style="font-weight: 600; color: ${isLosing ? 'var(--danger)' : 'var(--success)'};">⚠️ Open ${openPosition.direction.toUpperCase()} Position</span>
                        <span style="color: var(--text-muted); margin-left: 0.5rem;">
                            Entry: $${openPosition.entry_price?.toLocaleString() || '?'} → Now: $${openPosition.current_price?.toLocaleString() || '?'}
                        </span>
                        <span style="font-weight: 600; color: ${isLosing ? 'var(--danger)' : 'var(--success)'}; margin-left: 0.5rem;">
                            ${openPosition.unrealized_pnl >= 0 ? '+' : ''}£${openPosition.unrealized_pnl?.toFixed(2) || '0'} (${openPosition.unrealized_pnl_pct >= 0 ? '+' : ''}${openPosition.unrealized_pnl_pct?.toFixed(2) || '0'}%)
                        </span>
                    </div>
                `;
            }

            return `
                <div style="background: var(--bg-hover); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; border-left: 4px solid ${borderColor}; ${isTopPick ? 'box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);' : ''}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem;">
                        <div>
                            <h4 style="color: var(--text-primary); margin: 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem;">
                                ${isTopPick ? '<span style="font-size: 1.25rem;">👑</span>' : `<span style="color: var(--text-muted);">#${idx + 1}</span>`}
                                ${strat.strategy_name || 'Unknown Strategy'}
                                ${hasOpenPosition ? '<span style="background: var(--warning); color: #000; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.65rem; font-weight: 600;">OPEN TRADE</span>' : ''}
                            </h4>
                            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                                ${strat.strategy_category || 'Unknown'} | ${strat.direction || 'both'} | TP: ${params.tp_percent?.toFixed(1) || '?'}% / SL: ${params.sl_percent?.toFixed(1) || '?'}%
                            </div>
                            ${openPositionWarning}
                        </div>
                        <div style="text-align: right;">
                            <div style="color: ${metrics.total_pnl >= 0 ? 'var(--success)' : 'var(--danger)'}; font-size: 1.25rem; font-weight: 700;">
                                ${metrics.total_pnl >= 0 ? '+' : ''}${(metrics.total_pnl_percent || 0).toFixed(1)}%
                            </div>
                            <div style="font-size: 0.8rem; color: var(--text-muted);">
                                ${metrics.total_pnl_gbp !== undefined && metrics.total_pnl_gbp !== metrics.total_pnl ? formatDualCurrency(metrics.total_pnl, metrics.total_pnl_gbp) : formatCurrency(metrics.total_pnl || 0, metrics.source_currency || 'GBP')} | ${metrics.total_trades || 0} trades
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; font-size: 0.85rem; padding: 0.75rem; background: var(--bg-primary); border-radius: 8px;">
                        <div><span style="color: var(--text-muted);">P.Factor:</span> <strong>${metrics.profit_factor?.toFixed(2) || 'N/A'}</strong></div>
                        <div><span style="color: var(--text-muted);">Avg Trade:</span> <strong>${metrics.avg_trade_gbp !== undefined && metrics.avg_trade_gbp !== metrics.avg_trade ? formatDualCurrency(metrics.avg_trade, metrics.avg_trade_gbp) : formatCurrency(metrics.avg_trade || 0, metrics.source_currency || 'GBP')}</strong></div>
                        <div><span style="color: var(--text-muted);">Max DD:</span> <strong style="color: var(--danger);">${metrics.max_drawdown_gbp !== undefined && metrics.max_drawdown_gbp !== metrics.max_drawdown ? formatDualCurrency(metrics.max_drawdown, metrics.max_drawdown_gbp) : formatCurrency(metrics.max_drawdown || 0, metrics.source_currency || 'GBP')}</strong></div>
                        <div><span style="color: var(--text-muted);">Score:</span> <strong style="color: var(--accent-primary);">${metrics.composite_score?.toFixed(1) || 'N/A'}</strong></div>
                    </div>
                    ${renderTuningSectionForCard(strat)}
                    <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <button onclick="exportStrategyPineScriptWithEngine(${idx + 1}, '${engine}')" class="btn btn-success" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                            📋 Copy Pine
                        </button>
                        <button onclick="downloadStrategyPineScriptWithEngine(${idx + 1}, '${engine}')" class="btn btn-secondary" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                            ⬇️ Download .pine
                        </button>
                        <button onclick="openStrategyInTradingViewWithEngine(${idx + 1}, '${engine}')" class="btn" style="font-size: 0.8rem; padding: 0.4rem 0.8rem; background: linear-gradient(135deg, #2962FF, #00BCD4);">
                            📈 View in TradingView
                        </button>
                    </div>
                </div>
            `;
        }

        // Helper to render tuning section for a strategy card
        function renderTuningSectionForCard(strat) {
            const tuning = findTuningForStrategy(strat);
            if (!tuning) return '';
            return renderTuningSection(tuning);
        }

        function displaySingleEngineReport(report, container) {
            const top10 = report.top_10 || [];

            if (top10.length === 0) {
                container.innerHTML = `
                    <div style="padding: 2rem; text-align: center; background: var(--danger-bg); border-radius: 8px; border: 1px solid var(--danger);">
                        <h3 style="color: var(--danger); margin-bottom: 0.5rem;">No Profitable Strategies Found</h3>
                        <p style="color: var(--text-muted);">Try using more historical data or different trading pairs.</p>
                    </div>
                `;
                return;
            }

            const buyHoldReturn = report.buy_hold_return || 0;
            const beatsBHCount = report.beats_buy_hold_count || 0;
            const engine = report.engine || 'tradingview';

            let html = `
                <div style="margin-bottom: 1.5rem; padding: 1.25rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(139, 92, 246, 0.05)); border-radius: 12px; border-left: 4px solid var(--success);">
                    <h3 style="color: var(--success); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">🏆</span> Strategy Search Complete
                    </h3>
                    <p style="color: var(--text-secondary); margin: 0; font-size: 0.95rem;">
                        <strong>Data:</strong> ${report.data_rows?.toLocaleString() || 'N/A'} candles |
                        <strong>Symbol:</strong> ${report.symbol || 'N/A'} |
                        <strong>Timeframe:</strong> ${report.timeframe || 'N/A'} |
                        <strong>Engine:</strong> ${engine.toUpperCase()}
                    </p>
                </div>

                <h3 style="color: var(--accent-primary); margin-bottom: 1rem;">🎯 Top 10 Strategies</h3>
            `;

            top10.forEach((strat, idx) => {
                html += renderStrategyCard(strat, idx, engine);
            });

            container.innerHTML = html;

            top10.forEach((strat, idx) => {
                const metrics = strat.metrics || {};
                const params = strat.params || {};
                const equityCurve = strat.equity_curve || [];

                const isTopPick = idx === 0;
                const isTop3 = idx < 3;
                const borderColor = isTopPick ? 'var(--success)' : (isTop3 ? 'var(--accent-primary)' : 'var(--border)');

                html += `
                    <div style="background: var(--bg-hover); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; border-left: 4px solid ${borderColor}; ${isTopPick ? 'box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem;">
                            <div>
                                <h4 style="color: var(--text-primary); margin: 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem;">
                                    ${isTopPick ? '<span style="font-size: 1.25rem;">👑</span>' : `<span style="color: var(--text-muted);">#${idx + 1}</span>`}
                                    ${strat.strategy_name || 'Unknown Strategy'}
                                </h4>
                                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; display: flex; align-items: center; gap: 0.5rem;">
                                    ${strat.strategy_category || ''}
                                    ${(metrics.win_rate || 0) >= 60 ?
                                        `<span style="padding: 0.15rem 0.5rem; background: rgba(34, 197, 94, 0.2); color: #22c55e; border-radius: 4px; font-size: 0.65rem; font-weight: 600;">✓ High Win Rate</span>` :
                                        ''
                                    }
                                </div>
                            </div>
                            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                <span class="status-badge" style="font-size: 0.85rem; background: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                                    TP: ${(params.tp_percent || 1.0).toFixed(1)}% / SL: ${(params.sl_percent || 3.0).toFixed(1)}%
                                </span>
                                <span class="status-badge success" style="font-size: 0.9rem;">
                                    PF: ${metrics.profit_factor || 0}
                                </span>
                                <span class="status-badge ${(metrics.total_pnl || 0) >= 0 ? 'success' : 'danger'}" style="font-size: 0.9rem;">
                                    ${(metrics.total_pnl_percent || 0).toFixed(1)}% (£${(metrics.total_pnl || 0).toFixed(0)})
                                </span>
                            </div>
                        </div>

                        <!-- Metrics Grid -->
                        <div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Trades (W/L)</div>
                                <div style="font-weight: 600; font-size: 1.1rem;">
                                    ${metrics.total_trades || 0} <span style="font-size: 0.8rem; color: var(--text-muted);">(<span style="color: var(--success);">${metrics.wins || 0}</span>/<span style="color: var(--danger);">${metrics.losses || 0}</span>)</span>
                                </div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Win Rate</div>
                                <div style="color: ${(metrics.win_rate || 0) >= 60 ? 'var(--success)' : (metrics.win_rate || 0) >= 50 ? 'var(--warning)' : 'var(--danger)'}; font-weight: 600; font-size: 1.1rem;">${(metrics.win_rate || 0).toFixed(1)}%</div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Profit Factor</div>
                                <div style="color: ${(metrics.profit_factor || 0) >= 1.5 ? 'var(--success)' : 'var(--warning)'}; font-weight: 600; font-size: 1.1rem;">${(metrics.profit_factor || 0).toFixed(2)}</div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Return</div>
                                <div style="color: ${(metrics.total_pnl_percent || 0) >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight: 600; font-size: 1.1rem;">${(metrics.total_pnl_percent || 0).toFixed(1)}%</div>
                                <div style="color: var(--text-muted); font-size: 0.65rem;">${metrics.total_pnl_gbp !== undefined && metrics.total_pnl_gbp !== metrics.total_pnl ? formatDualCurrency(metrics.total_pnl, metrics.total_pnl_gbp) : formatCurrency(metrics.total_pnl || 0, metrics.source_currency || 'GBP')}</div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">B&H</div>
                                <div style="color: ${((metrics.total_pnl_percent || 0) - (metrics.vs_buy_hold || 0)) >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight: 600; font-size: 1.1rem;">${((metrics.total_pnl_percent || 0) - (metrics.vs_buy_hold || 0)) >= 0 ? '+' : ''}${((metrics.total_pnl_percent || 0) - (metrics.vs_buy_hold || 0)).toFixed(1)}%</div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Max DD</div>
                                <div style="color: var(--danger); font-weight: 600; font-size: 1.1rem;">${(metrics.max_drawdown_percent || 0).toFixed(1)}%</div>
                            </div>
                            <div style="text-align: center; padding: 0.6rem; background: var(--bg-card); border-radius: 6px;">
                                <div style="color: var(--text-muted); font-size: 0.7rem;">Score</div>
                                <div style="color: var(--accent-primary); font-weight: 600; font-size: 1.1rem;">${(metrics.composite_score || 0).toFixed(0)}</div>
                            </div>
                        </div>

                        <!-- Equity Curve -->
                        ${equityCurve.length > 0 ? `
                            <div style="background: var(--bg-primary); border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem; overflow: hidden;">
                                <div style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 0.5rem;">📈 Equity Curve</div>
                                <div style="height: 60px; display: flex; align-items: flex-end; gap: 1px;">
                                    ${equityCurve.slice(-50).map((val, i, arr) => {
                                        const min = Math.min(...arr);
                                        const max = Math.max(...arr);
                                        const range = max - min || 1;
                                        const height = ((val - min) / range) * 100;
                                        const color = val >= 0 ? 'var(--success)' : 'var(--danger)';
                                        return `<div style="flex: 1; height: ${Math.max(height, 2)}%; background: ${color}; border-radius: 1px 1px 0 0;"></div>`;
                                    }).join('')}
                                </div>
                            </div>
                        ` : ''}

                        <!-- Parameters -->
                        <details style="margin-top: 0.5rem;">
                            <summary style="cursor: pointer; color: var(--accent-secondary); font-size: 0.85rem;">⚙️ Strategy Parameters</summary>
                            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 0.5rem; margin-top: 0.75rem; font-size: 0.8rem;">
                                ${Object.entries(params).map(([key, val]) => `
                                    <div style="padding: 0.4rem 0.6rem; background: var(--bg-primary); border-radius: 4px;">
                                        <span style="color: var(--text-muted);">${key}:</span>
                                        <span style="color: var(--text-primary); font-family: 'JetBrains Mono', monospace; margin-left: 0.25rem;">${typeof val === 'number' ? val.toFixed(2) : val}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </details>

                        <!-- Action Buttons -->
                        <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <button onclick="exportStrategyPineScript(${idx + 1})" class="btn btn-success" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                                📋 Copy Pine
                            </button>
                            <button onclick="downloadStrategyPineScript(${idx + 1})" class="btn btn-secondary" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">
                                ⬇️ Download .pine
                            </button>
                            <button onclick="downloadStrategyTradesCSV(${idx + 1})" class="btn" style="font-size: 0.8rem; padding: 0.4rem 0.8rem; background: #0ea5e9;">
                                📊 Trades CSV
                            </button>
                            <button onclick="openStrategyInTradingView(${idx + 1})" class="btn" style="font-size: 0.8rem; padding: 0.4rem 0.8rem; background: linear-gradient(135deg, #2962FF, #00BCD4);">
                                📈 View in TradingView
                            </button>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }
        
        async function uploadCSV(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const btn = document.getElementById('uploadCsvBtn');
            const fetchBtn = document.getElementById('fetchDataBtn');
            const loadingProgress = document.getElementById('dataLoadingProgress');
            
            // Show loading UI
            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Uploading...';
            fetchBtn.disabled = true;
            
            loadingProgress.style.display = 'block';
            document.getElementById('dataProgressFill').style.width = '30%';
            document.getElementById('dataProgressMessage').textContent = `Uploading ${file.name}...`;
            document.getElementById('dataProgressPercent').textContent = '30%';
            
            addLog(`Uploading TradingView CSV: ${file.name}...`);
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                document.getElementById('dataProgressFill').style.width = '60%';
                document.getElementById('dataProgressPercent').textContent = '60%';
                document.getElementById('dataProgressMessage').textContent = 'Processing CSV...';
                
                const response = await fetch('/api/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('dataProgressFill').style.width = '100%';
                    document.getElementById('dataProgressPercent').textContent = '100%';
                    document.getElementById('dataProgressMessage').textContent = data.message;

                    addLog(data.message);
                    // WebSocket will receive status updates automatically
                } else {
                    addLog(`Error: ${data.detail || 'Upload failed'}`);
                    loadingProgress.style.display = 'none';
                }
            } catch (error) {
                addLog(`Error: ${error.message}`);
                loadingProgress.style.display = 'none';
            }
            
            // Reset buttons and file input
            btn.disabled = false;
            btn.innerHTML = '📁 Upload TradingView CSV';
            fetchBtn.disabled = false;
            event.target.value = '';
        }
        
        // Export Pine Script for a specific unified optimizer strategy (copies to clipboard)
        // Get selected Pine Script engine (legacy - for non-comparison mode)
        function getSelectedPineEngine() {
            const select = document.getElementById('pineEngineSelect');
            return select ? select.value : 'tradingview';
        }

        // Engine-specific versions for comparison mode
        async function exportStrategyPineScriptWithEngine(rank, engine) {
            try {
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`Generating Pine Script for strategy #${rank} [${engineLabel}]...`);
                const response = await fetch(`/api/unified-pinescript/${rank}?engine=${engine}`);

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to generate Pine Script');
                }

                const data = await response.json();

                // Sanitize and copy to clipboard
                const cleanScript = data.pinescript
                    .replace(/\r\n/g, '\n')  // Windows -> Unix
                    .replace(/\r/g, '\n')     // Old Mac -> Unix
                    .replace(/\uFEFF/g, '')   // Remove BOM
                    .replace(/[\u200B-\u200D\uFEFF]/g, '')  // Remove zero-width chars
                    .replace(/\u00A0/g, ' '); // Replace non-breaking space

                await navigator.clipboard.writeText(cleanScript);
                addLog(`✅ Pine Script copied to clipboard [${engineLabel}]`);

            } catch (error) {
                addLog(`❌ Error generating Pine Script: ${error.message}`);
            }
        }

        async function downloadStrategyPineScriptWithEngine(rank, engine) {
            try {
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`Downloading Pine Script for strategy #${rank} [${engineLabel}]...`);
                window.location.href = `/api/download-unified-pinescript/${rank}?engine=${engine}`;
            } catch (error) {
                addLog(`❌ Error downloading: ${error.message}`);
            }
        }

        async function openStrategyInTradingViewWithEngine(rank, engine) {
            try {
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`📈 Fetching TradingView link for strategy #${rank} [${engineLabel}]...`);

                const response = await fetch(`/api/tradingview-link/${rank}?engine=${engine}`);
                if (!response.ok) {
                    throw new Error('Failed to get TradingView link');
                }

                const data = await response.json();

                // Normalize line endings and copy to clipboard
                const cleanScript = data.pinescript.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
                await navigator.clipboard.writeText(cleanScript);
                addLog(`📋 Pine Script copied to clipboard [${engineLabel}]`);

                // Show warning if exchange was mapped
                if (data.exchange_warning) {
                    addLog(`⚠️ ${data.exchange_warning}`);
                }

                // Open TradingView in new tab
                window.open(data.tradingview_url, '_blank');
                addLog(`📈 TradingView opened: ${data.exchange}:${data.symbol} (${data.timeframe})`);
                addLog(`💡 Paste Pine Script into Pine Editor (Ctrl+V / Cmd+V)`);

            } catch (error) {
                addLog(`❌ Error: ${error.message}`);
            }
        }

        async function exportStrategyPineScript(rank) {
            try {
                const engine = getSelectedPineEngine();
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`Generating Pine Script for strategy #${rank} [${engineLabel}]...`);
                const response = await fetch(`/api/unified-pinescript/${rank}?engine=${engine}`);

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to generate Pine Script');
                }

                const data = await response.json();

                // Normalize line endings and copy to clipboard
                const cleanScript = data.pinescript.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
                await navigator.clipboard.writeText(cleanScript);

                addLog(`✅ Pine Script copied for "${data.strategy_name}" (Rank #${rank}) - TP: ${data.tp_percent?.toFixed(1) || '1.0'}% / SL: ${data.sl_percent?.toFixed(1) || '3.0'}% [${engineLabel}]`);

            } catch (error) {
                addLog(`❌ Error generating Pine Script: ${error.message}`);
            }
        }

        // Download Pine Script file for a specific strategy
        async function downloadStrategyPineScript(rank) {
            try {
                const engine = getSelectedPineEngine();
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`Downloading Pine Script for strategy #${rank} [${engineLabel}]...`);
                window.location.href = `/api/download-unified-pinescript/${rank}?engine=${engine}`;
            } catch (error) {
                addLog(`❌ Error downloading: ${error.message}`);
            }
        }

        // Download Trades CSV for the currently selected strategy
        async function downloadTradesCSV() {
            const select = document.getElementById('strategyRankSelect');
            const rank = select ? parseInt(select.value) : 1;

            try {
                addLog(`📊 Downloading trades CSV for strategy #${rank}...`);
                window.location.href = `/api/unified-trades-csv/${rank}`;
                addLog(`✅ Trades CSV download started`);
            } catch (error) {
                addLog(`❌ Error downloading CSV: ${error.message}`);
            }
        }

        // Download Trades CSV for a specific strategy rank
        async function downloadStrategyTradesCSV(rank) {
            try {
                addLog(`📊 Downloading trades CSV for strategy #${rank}...`);
                window.location.href = `/api/unified-trades-csv/${rank}`;
            } catch (error) {
                addLog(`❌ Error downloading CSV: ${error.message}`);
            }
        }

        // Open TradingView with correct symbol/timeframe and copy Pine Script to clipboard
        async function openInTradingView() {
            const select = document.getElementById('strategyRankSelect');
            const rank = select ? parseInt(select.value) : 1;
            await openStrategyInTradingView(rank);
        }

        // Open TradingView for a specific strategy rank
        async function openStrategyInTradingView(rank) {
            try {
                const engine = getSelectedPineEngine();
                const engineLabel = engine === 'native' ? 'NATIVE' : 'TRADINGVIEW';
                addLog(`📈 Fetching TradingView link for strategy #${rank} [${engineLabel}]...`);

                const response = await fetch(`/api/tradingview-link/${rank}?engine=${engine}`);
                if (!response.ok) {
                    throw new Error('Failed to get TradingView link');
                }

                const data = await response.json();

                // Normalize line endings and copy to clipboard
                const cleanScript = data.pinescript.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
                await navigator.clipboard.writeText(cleanScript);
                addLog(`📋 Pine Script copied to clipboard [${engineLabel}]`);

                // Show warning if exchange was mapped
                if (data.exchange_warning) {
                    addLog(`⚠️ ${data.exchange_warning}`);
                }

                // Open TradingView in new tab
                window.open(data.tradingview_url, '_blank');
                addLog(`📈 TradingView opened: ${data.exchange}:${data.symbol} (${data.timeframe})`);

                // Show instructions
                let message = `Pine Script copied to clipboard!\n\nIn TradingView:\n1. Open Pine Editor (bottom panel)\n2. Delete existing code\n3. Paste (Ctrl+V / Cmd+V)\n4. Click "Add to chart"\n\nChart: ${data.exchange}:${data.symbol}`;

                if (data.exchange_warning) {
                    message += `\n\n⚠️ ${data.exchange_warning}`;
                }

                alert(message);

            } catch (error) {
                addLog(`❌ Error: ${error.message}`);
                alert('Error: ' + error.message);
            }
        }
        
        // ============ TRADINGVIEW COMPARISON ============

        async function uploadTVComparison(event) {
            const file = event.target.files[0];
            if (!file) return;

            const rank = parseInt(document.getElementById('comparisonRankSelect').value) || 1;

            addLog(`📊 Uploading TradingView trade export for comparison with strategy #${rank}...`);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`/api/upload-tv-comparison/${rank}`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    addLog(`✅ Comparison complete: ${data.tv_trade_count} TV trades vs ${data.our_trade_count} our trades`);
                    displayComparisonResults(data.comparison);
                } else {
                    addLog(`❌ Error: ${data.detail || 'Upload failed'}`);
                }
            } catch (error) {
                addLog(`❌ Error: ${error.message}`);
            }

            event.target.value = '';
        }

        function displayComparisonResults(comparison) {
            const container = document.getElementById('comparisonResults');
            const summary = comparison.summary;
            const tv = summary.tradingview;
            const ours = summary.our_system;
            const diff = summary.difference;

            let html = `
                <!-- Summary Cards -->
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                    <!-- TradingView Results -->
                    <div style="background: linear-gradient(135deg, rgba(41, 98, 255, 0.1), rgba(41, 98, 255, 0.05)); border: 1px solid rgba(41, 98, 255, 0.3); border-radius: 12px; padding: 1rem;">
                        <h4 style="color: #2962FF; margin: 0 0 0.75rem 0; font-size: 1rem;">📺 TradingView</h4>
                        <div style="display: grid; gap: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Trades:</span>
                                <span style="font-weight: 600;">${tv.trade_count}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Total P&L:</span>
                                <span style="font-weight: 600; color: ${tv.total_pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${tv.total_pnl.toFixed(2)}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Win Rate:</span>
                                <span style="font-weight: 600; color: ${tv.win_rate >= 50 ? 'var(--success)' : 'var(--danger)'};">${tv.win_rate.toFixed(1)}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">W/L:</span>
                                <span><span style="color: var(--success);">${tv.wins}</span>/<span style="color: var(--danger);">${tv.losses}</span></span>
                            </div>
                        </div>
                    </div>

                    <!-- Our System Results -->
                    <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05)); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 1rem;">
                        <h4 style="color: var(--accent-primary); margin: 0 0 0.75rem 0; font-size: 1rem;">🎯 Our System</h4>
                        <div style="display: grid; gap: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Trades:</span>
                                <span style="font-weight: 600;">${ours.trade_count}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Total P&L:</span>
                                <span style="font-weight: 600; color: ${ours.total_pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${ours.total_pnl.toFixed(2)}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Win Rate:</span>
                                <span style="font-weight: 600; color: ${ours.win_rate >= 50 ? 'var(--success)' : 'var(--danger)'};">${ours.win_rate.toFixed(1)}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">W/L:</span>
                                <span><span style="color: var(--success);">${ours.wins}</span>/<span style="color: var(--danger);">${ours.losses}</span></span>
                            </div>
                        </div>
                    </div>

                    <!-- Difference -->
                    <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 1rem;">
                        <h4 style="color: var(--warning); margin: 0 0 0.75rem 0; font-size: 1rem;">📈 Difference</h4>
                        <div style="display: grid; gap: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Trade Count:</span>
                                <span style="font-weight: 600; color: ${diff.trade_count_diff === 0 ? 'var(--success)' : 'var(--warning)'};">${diff.trade_count_diff > 0 ? '+' : ''}${diff.trade_count_diff}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">P&L Diff:</span>
                                <span style="font-weight: 600; color: ${diff.pnl_diff >= 0 ? 'var(--success)' : 'var(--danger)'};">${diff.pnl_diff >= 0 ? '+' : ''}£${diff.pnl_diff.toFixed(2)}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Win Rate Diff:</span>
                                <span style="font-weight: 600; color: ${diff.win_rate_diff >= 0 ? 'var(--success)' : 'var(--danger)'};">${diff.win_rate_diff >= 0 ? '+' : ''}${diff.win_rate_diff.toFixed(1)}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted);">Match Rate:</span>
                                <span style="font-weight: 600; color: ${comparison.match_rate >= 80 ? 'var(--success)' : comparison.match_rate >= 50 ? 'var(--warning)' : 'var(--danger)'};">${comparison.match_rate.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Match Status -->
                <div style="padding: 1rem; margin-bottom: 1rem; border-radius: 8px; ${comparison.match_rate >= 80 ? 'background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3);' : comparison.match_rate >= 50 ? 'background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3);' : 'background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);'}">
                    ${comparison.match_rate >= 80 ?
                        `<span style="color: var(--success); font-weight: 600;">✅ Good Match!</span> ${comparison.match_rate.toFixed(1)}% of TradingView trades matched within 1 hour tolerance.` :
                        comparison.match_rate >= 50 ?
                        `<span style="color: var(--warning); font-weight: 600;">⚠️ Partial Match</span> Only ${comparison.match_rate.toFixed(1)}% matched. Check entry logic or data source differences.` :
                        `<span style="color: var(--danger); font-weight: 600;">❌ Low Match Rate</span> Only ${comparison.match_rate.toFixed(1)}% matched. Likely different entry conditions or data source.`
                    }
                </div>
            `;

            // Matched Trades Table
            if (comparison.matched_trades && comparison.matched_trades.length > 0) {
                html += `
                    <details open style="margin-bottom: 1rem;">
                        <summary style="cursor: pointer; color: var(--success); font-weight: 600; margin-bottom: 0.5rem;">
                            ✅ Matched Trades (${comparison.matched_trades.length})
                        </summary>
                        <div style="max-height: 300px; overflow-y: auto; background: var(--bg-secondary); border-radius: 8px;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
                                <thead style="position: sticky; top: 0; background: var(--bg-card);">
                                    <tr style="color: var(--text-muted);">
                                        <th style="padding: 0.5rem; text-align: left;">Time</th>
                                        <th style="padding: 0.5rem; text-align: right;">TV P&L</th>
                                        <th style="padding: 0.5rem; text-align: right;">Our P&L</th>
                                        <th style="padding: 0.5rem; text-align: right;">Diff</th>
                                        <th style="padding: 0.5rem; text-align: center;">Quality</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${comparison.matched_trades.map(m => `
                                        <tr style="border-bottom: 1px solid var(--border-subtle);">
                                            <td style="padding: 0.4rem 0.5rem; font-size: 0.75rem;">${m.tv_trade.entry_time}</td>
                                            <td style="padding: 0.4rem 0.5rem; text-align: right; color: ${m.tv_trade.pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${m.tv_trade.pnl.toFixed(2)}</td>
                                            <td style="padding: 0.4rem 0.5rem; text-align: right; color: ${m.our_trade.pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${m.our_trade.pnl.toFixed(2)}</td>
                                            <td style="padding: 0.4rem 0.5rem; text-align: right; color: ${Math.abs(m.pnl_diff) < 1 ? 'var(--success)' : 'var(--warning)'};">£${m.pnl_diff.toFixed(2)}</td>
                                            <td style="padding: 0.4rem 0.5rem; text-align: center;">
                                                ${m.match_quality === 'exact' ?
                                                    '<span style="color: var(--success);">●</span>' :
                                                    '<span style="color: var(--warning);">◐</span>'
                                                }
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </details>
                `;
            }

            // Unmatched TV Trades
            if (comparison.unmatched_tv && comparison.unmatched_tv.length > 0) {
                html += `
                    <details style="margin-bottom: 1rem;">
                        <summary style="cursor: pointer; color: #2962FF; font-weight: 600; margin-bottom: 0.5rem;">
                            📺 TradingView Only (${comparison.unmatched_tv.length} trades we missed)
                        </summary>
                        <div style="max-height: 200px; overflow-y: auto; background: var(--bg-secondary); border-radius: 8px; padding: 0.5rem;">
                            ${comparison.unmatched_tv.map(t => `
                                <div style="padding: 0.4rem; border-bottom: 1px solid var(--border-subtle); font-size: 0.8rem;">
                                    <span style="color: var(--text-muted);">${t.entry_time}</span>
                                    <span style="margin-left: 0.5rem; color: ${t.pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${t.pnl.toFixed(2)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </details>
                `;
            }

            // Unmatched Our Trades
            if (comparison.unmatched_ours && comparison.unmatched_ours.length > 0) {
                html += `
                    <details style="margin-bottom: 1rem;">
                        <summary style="cursor: pointer; color: var(--accent-primary); font-weight: 600; margin-bottom: 0.5rem;">
                            🎯 Our System Only (${comparison.unmatched_ours.length} extra trades)
                        </summary>
                        <div style="max-height: 200px; overflow-y: auto; background: var(--bg-secondary); border-radius: 8px; padding: 0.5rem;">
                            ${comparison.unmatched_ours.map(t => `
                                <div style="padding: 0.4rem; border-bottom: 1px solid var(--border-subtle); font-size: 0.8rem;">
                                    <span style="color: var(--text-muted);">${t.entry_time}</span>
                                    <span style="margin-left: 0.5rem; color: ${t.pnl >= 0 ? 'var(--success)' : 'var(--danger)'};">£${t.pnl.toFixed(2)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </details>
                `;
            }

            container.innerHTML = html;
        }

        async function clearComparison() {
            try {
                await fetch('/api/comparison', { method: 'DELETE' });
                document.getElementById('comparisonResults').innerHTML = '';
                addLog('Comparison data cleared');
            } catch (error) {
                addLog(`Error clearing comparison: ${error.message}`);
            }
        }

        // Show comparison card when optimization completes
        function showComparisonCard() {
            const card = document.getElementById('comparisonCard');
            if (card) card.style.display = 'block';
        }

        // Polling - uses sequential pattern to prevent request pileup
        let pollingActive = false;
        let fastPollingActive = false;
        let pollingDelay = 1500;

        function startPolling() {
            // DISABLED - WebSocket provides all updates
            console.log('[Status] Polling disabled - using WebSocket');
        }

        function startFastPolling() {
            // DISABLED - WebSocket provides all updates
            console.log('[Status] Fast polling disabled - using WebSocket');
        }

        function stopFastPolling() {
            fastPollingActive = false;
            pollingDelay = 1500;
            // Continue normal polling
        }

        function stopPolling() {
            pollingActive = false;
            fastPollingActive = false;
        }

        function schedulePoll() {
            if (!pollingActive) return;
            setTimeout(pollStatus, pollingDelay);
        }

        async function pollStatus() {
            if (!pollingActive) return;

            // Skip HTTP polling if WebSocket is connected - we get push updates
            if (wsConnected) {
                stopPolling();
                return;
            }

            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                updateDataStatus(data.data);

                // Handle data loading completion
                if (data.data.loaded) {
                    // Switch to normal polling if was fast polling
                    if (fastPollingActive) {
                        stopFastPolling();
                    }

                    // Stop data loading animation
                    stopDataLoadingAnimation();

                    // Complete progress bar
                    const fill = document.getElementById('dataProgressFill');
                    const pct = document.getElementById('dataProgressPercent');
                    const msg = document.getElementById('dataProgressMessage');
                    if (fill) fill.style.width = '100%';
                    if (pct) pct.textContent = '100%';
                    if (msg) msg.textContent = `✓ ${data.data.rows?.toLocaleString() || ''} candles loaded`;

                    // Re-enable buttons
                    const fetchBtn = document.getElementById('fetchDataBtn');
                    const uploadBtn = document.getElementById('uploadCsvBtn');
                    if (fetchBtn) {
                        fetchBtn.disabled = false;
                        fetchBtn.innerHTML = 'Fetch Data';
                    }
                    if (uploadBtn) uploadBtn.disabled = false;

                    // Hide progress after a delay
                    setTimeout(() => {
                        const loadingProgress = document.getElementById('dataLoadingProgress');
                        if (loadingProgress && data.data.loaded) {
                            loadingProgress.style.display = 'none';
                        }
                    }, 2000);
                }

            } catch (error) {
                console.error('Polling error:', error);
            }

            // Schedule next poll AFTER this one completes (prevents pileup)
            schedulePoll();
        }
        
        // Initial load
        document.addEventListener('DOMContentLoaded', async () => {
            addLog('System initialized');

            // Initialize WebSocket for real-time updates (replaces polling)
            initWebSocket();

            // Initialize source-specific options (pairs, intervals, periods)
            // Yahoo Finance is the default source
            updatePairOptions();

            // Initial state will come from WebSocket full_state message
            // NO HTTP FALLBACK - WebSocket is the only source of truth
            console.log('[Init] Waiting for WebSocket full_state message...');
        });

        // =====================================================
        // VALIDATION TAB FUNCTIONS
        // =====================================================

        // Initialize Validation Tab
        async function initValidationTab() {
            validationInitialized = true;
            try {
                const response = await fetch('/api/db/strategies?limit=500');
                const strategies = await response.json();

                const select = document.getElementById('validationStrategySelect');
                select.innerHTML = '<option value="">-- Select a strategy --</option>';

                // Store strategies globally for lookup
                window.validationStrategies = strategies;

                // Group by strategy name for easier selection
                strategies.forEach(strategy => {
                    const option = document.createElement('option');
                    option.value = strategy.id;
                    const pnl = strategy.total_pnl >= 0 ? `+${strategy.total_pnl.toFixed(2)}` : strategy.total_pnl.toFixed(2);
                    option.textContent = `${strategy.strategy_name} (${strategy.symbol} ${strategy.timeframe}) - ${strategy.win_rate.toFixed(1)}% WR, ${pnl}`;
                    select.appendChild(option);
                });
            } catch (e) {
                console.error('Failed to load strategies for validation:', e);
            }
        }

        // Handle strategy selection change - show original values
        async function onValidationStrategyChange() {
            const strategyId = document.getElementById('validationStrategySelect').value;
            const origCapitalEl = document.getElementById('validationOriginalCapital');
            const origPositionEl = document.getElementById('validationOriginalPosition');
            const capitalInput = document.getElementById('validationCapital');
            const positionInput = document.getElementById('validationPositionSize');

            if (!strategyId) {
                origCapitalEl.textContent = '--';
                origPositionEl.textContent = '--';
                return;
            }

            // Try to get original values from optimization run
            try {
                const strategy = window.validationStrategies?.find(s => s.id == strategyId);
                if (strategy && strategy.optimization_run_id) {
                    const runResponse = await fetch(`/api/db/runs`);
                    const runs = await runResponse.json();
                    const run = runs.find(r => r.id == strategy.optimization_run_id);
                    if (run) {
                        const origCapital = run.capital || 1000;
                        const origPosition = run.risk_percent || 75;
                        origCapitalEl.textContent = `£${origCapital}`;
                        origPositionEl.textContent = `${origPosition}%`;
                        // Pre-fill with original values
                        capitalInput.value = origCapital;
                        positionInput.value = origPosition;
                        return;
                    }
                }
            } catch (e) {
                console.error('Failed to get original settings:', e);
            }

            origCapitalEl.textContent = '£1000 (default)';
            origPositionEl.textContent = '75% (default)';
        }

        // Run Validation
        async function runValidation() {
            const strategyId = document.getElementById('validationStrategySelect').value;
            if (!strategyId) {
                alert('Please select a strategy to validate');
                return;
            }

            const btn = document.getElementById('validateBtn');
            const progress = document.getElementById('validationProgress');
            const progressBar = document.getElementById('validationProgressBar');
            const progressText = document.getElementById('validationProgressText');
            const results = document.getElementById('validationResults');

            btn.disabled = true;
            btn.textContent = 'Validating...';
            progress.style.display = 'block';
            progressBar.style.width = '10%';
            progressText.textContent = 'Fetching exchange rate...';
            results.style.display = 'none';

            try {
                // Fetch live exchange rate
                try {
                    const fxResponse = await fetch('/api/exchange-rate');
                    const fxData = await fxResponse.json();
                    window.liveExchangeRate = fxData.usd_to_gbp;
                    console.log(`Live USD/GBP rate: ${fxData.usd_to_gbp} (${fxData.source})`);
                } catch (e) {
                    console.warn('Failed to fetch exchange rate, using fallback');
                    window.liveExchangeRate = 0.79;
                }

                progressBar.style.width = '20%';
                progressText.textContent = 'Fetching strategy details...';
                progressBar.style.width = '40%';
                progressText.textContent = 'Running validation across all periods...';

                // Get capital and position size from inputs
                const capital = parseFloat(document.getElementById('validationCapital').value) || 1000;
                const positionSize = parseFloat(document.getElementById('validationPositionSize').value) || 75;

                const response = await fetch('/api/validate-strategy', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        strategy_id: parseInt(strategyId),
                        capital: capital,
                        position_size_pct: positionSize
                    })
                });

                if (!response.ok) {
                    throw new Error(`Validation failed: ${response.statusText}`);
                }

                const data = await response.json();

                progressBar.style.width = '100%';
                progressText.textContent = 'Validation complete!';

                // Render results
                renderValidationResults(data);
                results.style.display = 'block';

            } catch (e) {
                console.error('Validation failed:', e);
                progressText.textContent = `Error: ${e.message}`;
                progressBar.style.background = '#ff4444';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Validation (All Periods)';
                setTimeout(() => {
                    progress.style.display = 'none';
                    progressBar.style.width = '0%';
                    progressBar.style.background = 'linear-gradient(90deg, #00d4ff, #7b68ee)';
                }, 2000);
            }
        }

        // Render Validation Results
        function renderValidationResults(data) {
            // Update title
            document.getElementById('validationResultsTitle').textContent =
                `Validation: ${data.strategy.name}`;

            // Strategy info
            const infoDiv = document.getElementById('validationStrategyInfo');
            infoDiv.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Symbol</span>
                        <div style="font-weight: 600;">${data.strategy.symbol}</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Timeframe</span>
                        <div style="font-weight: 600;">${data.strategy.timeframe}</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Data Source</span>
                        <div style="font-weight: 600;">${data.strategy.data_source || 'Unknown'}</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Entry Rule</span>
                        <div style="font-weight: 600;">${data.strategy.entry_rule}</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Direction</span>
                        <div style="font-weight: 600; color: ${data.strategy.direction === 'long' ? '#4ade80' : '#f87171'};">
                            ${data.strategy.direction.toUpperCase()}
                        </div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">TP% / SL%</span>
                        <div style="font-weight: 600;">${data.strategy.tp_percent}% / ${data.strategy.sl_percent}%</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Capital</span>
                        <div style="font-weight: 600;">£${data.strategy.capital?.toLocaleString() || 1000}</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Position Size</span>
                        <div style="font-weight: 600;">${data.strategy.position_size_pct || 75}%</div>
                    </div>
                    <div>
                        <span style="color: #888; font-size: 0.8rem;">Optimized On</span>
                        <div style="font-weight: 600; color: #7b68ee;">${data.original.period || 'unknown'}</div>
                    </div>
                </div>
            `;

            // Build table rows
            const tbody = document.getElementById('validationTableBody');
            tbody.innerHTML = '';

            // Validation rows only (no original baseline row)
            data.validations.forEach(v => {
                const row = document.createElement('tr');

                let statusHtml = '';
                let statusColor = '';

                switch(v.status) {
                    case 'consistent':
                        statusHtml = '<span style="color: #4ade80;">✓ Consistent</span>';
                        break;
                    case 'minor_drop':
                        statusHtml = '<span style="color: #fbbf24;">⚠ Minor drop</span>';
                        break;
                    case 'degraded':
                        statusHtml = '<span style="color: #f87171;">✗ Degraded</span>';
                        break;
                    case 'no_trades':
                        statusHtml = '<span style="color: #888;">⚠ No trades</span>';
                        break;
                    case 'limit_exceeded':
                        statusHtml = '<span style="color: #666;">⛔ Limit exceeded</span>';
                        break;
                    case 'insufficient_data':
                        statusHtml = '<span style="color: #666;">⚠ Insufficient data</span>';
                        break;
                    case 'error':
                        statusHtml = '<span style="color: #f87171;">✗ Error</span>';
                        break;
                    default:
                        statusHtml = '<span style="color: #888;">Unknown</span>';
                }

                if (v.metrics) {
                    const pnlChange = calcChange(data.original.total_pnl, v.metrics.total_pnl);
                    const wrChange = calcChange(data.original.win_rate, v.metrics.win_rate);

                    // Determine currency from symbol and show both $ and £
                    const symbol = data.strategy.symbol.toUpperCase();
                    const isUSD = symbol.includes('USD') || symbol.includes('USDT');
                    const isGBP = symbol.includes('GBP');
                    const usdToGbp = window.liveExchangeRate || 0.79; // Live rate or fallback

                    const pnl = v.metrics.total_pnl;
                    const drawdown = v.metrics.max_drawdown || 0;
                    const pnlColor = pnl >= 0 ? '#4ade80' : '#f87171';

                    let pnlFormatted, drawdownFormatted;
                    const pnlSign = pnl >= 0 ? '+' : '-';
                    const ddSign = '-';

                    if (isUSD) {
                        const pnlGbp = Math.abs(pnl * usdToGbp);
                        const ddGbp = Math.abs(drawdown * usdToGbp);
                        pnlFormatted = `${pnlSign}$${Math.abs(pnl).toFixed(2)} / ${pnlSign}£${pnlGbp.toFixed(2)}`;
                        drawdownFormatted = `${ddSign}$${Math.abs(drawdown).toFixed(2)} / ${ddSign}£${ddGbp.toFixed(2)}`;
                    } else if (isGBP) {
                        const pnlUsd = Math.abs(pnl / usdToGbp);
                        const ddUsd = Math.abs(drawdown / usdToGbp);
                        pnlFormatted = `${pnlSign}£${Math.abs(pnl).toFixed(2)} / ${pnlSign}$${pnlUsd.toFixed(2)}`;
                        drawdownFormatted = `${ddSign}£${Math.abs(drawdown).toFixed(2)} / ${ddSign}$${ddUsd.toFixed(2)}`;
                    } else {
                        pnlFormatted = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
                        drawdownFormatted = `-${Math.abs(drawdown).toFixed(2)}`;
                    }

                    row.innerHTML = `
                        <td>${v.period}</td>
                        <td>${v.metrics.total_trades}</td>
                        <td>
                            ${v.metrics.win_rate.toFixed(1)}%
                            <span style="font-size: 0.75rem; color: ${wrChange >= 0 ? '#4ade80' : '#f87171'};">
                                (${wrChange >= 0 ? '+' : ''}${wrChange.toFixed(1)}%)
                            </span>
                        </td>
                        <td style="color: ${pnlColor};">
                            ${pnlFormatted}
                        </td>
                        <td style="color: #f87171;">
                            ${drawdownFormatted}
                        </td>
                        <td>${v.metrics.profit_factor.toFixed(2)}</td>
                        <td>${statusHtml}</td>
                    `;
                } else {
                    row.innerHTML = `
                        <td>${v.period}</td>
                        <td>--</td>
                        <td>--</td>
                        <td>--</td>
                        <td>--</td>
                        <td>--</td>
                        <td>${statusHtml}${v.message ? `<br><span style="font-size: 0.75rem; color: #666;">${v.message}</span>` : ''}</td>
                    `;
                }

                tbody.appendChild(row);
            });
        }

        // Calculate percentage change
        function calcChange(original, current) {
            if (original === 0) return current === 0 ? 0 : 100;
            return ((current - original) / Math.abs(original)) * 100;
        }

        // Validate from History - switch to validation tab and auto-run
        async function validateFromHistory(strategyId) {
            // Switch to validation tab
            showTab('validation');

            // Wait for tab to initialize if needed
            if (!validationInitialized) {
                await initValidationTab();
            }

            // Select the strategy in dropdown
            const select = document.getElementById('validationStrategySelect');
            select.value = strategyId;

            // Trigger change handler to pre-fill original capital/position size
            await onValidationStrategyChange();

            // Auto-run validation
            setTimeout(() => {
                runValidation();
            }, 100);
        }

        // =====================================================================
        // ELITE STRATEGIES TAB
        // =====================================================================

        let elitePolling = false;

        // Set up event delegation for elite table (only once)
        let eliteTableDelegationSetup = false;
        function setupEliteTableDelegation() {
            if (eliteTableDelegationSetup) return;
            eliteTableDelegationSetup = true;

            const tbody = document.getElementById('eliteTableBody');
            if (!tbody) return;

            tbody.addEventListener('click', (e) => {
                // Don't handle clicks on buttons
                if (e.target.closest('button')) return;

                const row = e.target.closest('.elite-row');
                if (!row) return;

                const hasVariants = row.dataset.hasVariants === 'true';
                if (!hasVariants) return;

                const groupIndex = parseInt(row.dataset.groupIndex, 10);

                // Find the group from stored grouped data
                if (eliteGroupedData && eliteGroupedData[groupIndex]) {
                    const group = eliteGroupedData[groupIndex];
                    // formatPnL helper matching the one used in renderEliteTable
                    const formatPnL = (periodData) => {
                        if (!periodData) return '<span style="color: var(--text-muted);">--</span>';
                        if (periodData.status === 'limit_exceeded') return '<span style="color: var(--text-muted);">N/A</span>';
                        if (periodData.status === 'no_trades') return '<span style="color: var(--text-muted);">0</span>';
                        if (periodData.status === 'error') return '<span style="color: var(--text-muted);">err</span>';
                        if (periodData.pnl === undefined) return '<span style="color: var(--text-muted);">--</span>';
                        const pnl = periodData.pnl;
                        const returnPct = periodData.return_pct !== undefined
                            ? periodData.return_pct
                            : Math.round((pnl / 1000) * 1000) / 10;
                        const sign = pnl >= 0 ? '+' : '';
                        const colorClass = pnl >= 0 ? 'success' : 'danger';
                        const pctSign = returnPct >= 0 ? '+' : '';
                        return `<span class="${colorClass}">£${sign}${pnl.toFixed(0)}<br><small style="opacity: 0.7;">(${pctSign}${returnPct.toFixed(1)}%)</small></span>`;
                    };
                    toggleEliteVariants(row, group, formatPnL);
                }
            });
        }

        async function initEliteTab() {
            eliteInitialized = true;
            // Set up event delegation before loading data
            setupEliteTableDelegation();
            // Initialize sort indicators
            updateEliteSortIndicators();
            await loadEliteData();
            // WebSocket provides real-time updates, no polling needed
        }

        // Track expanded elite group to preserve state on re-render
        let expandedEliteGroupKey = null;
        // Store grouped data for event delegation
        let eliteGroupedData = [];

        async function loadEliteData() {
            try {
                // Load via WebSocket (much faster than HTTP)
                const response = await loadEliteViaWs();
                const status = response.status;
                const strategies = response.strategies;

                // Update status counts (simplified model: validated + pending)
                document.getElementById('validatedCount').textContent = status.validated || 0;
                document.getElementById('pendingCount').textContent = status.pending || 0;

                // Update progress indicator
                updateEliteProgress(status);

                // Render table (show all validated strategies, not just elite)
                await renderEliteTable(strategies, status);

            } catch (error) {
                console.error('Failed to load elite data:', error);
            }
        }

        function updateEliteProgress(status) {
            // Delegate to updateEliteUI which handles both the badge and progress bar
            // Note: status fields are: running, processed, total, message, paused (not validation_* prefixed)
            updateEliteUI(status);
        }

        function startElitePolling() {
            // DISABLED - WebSocket provides all updates
            console.log('[Elite] Polling disabled - using WebSocket');
        }

        async function renderEliteTable(eliteStrategies, status) {
            const tbody = document.getElementById('eliteTableBody');
            const emptyMsg = document.getElementById('eliteEmpty');
            const table = document.getElementById('eliteTable');

            const validatedStrategies = eliteStrategies || [];

            if (validatedStrategies.length === 0) {
                table.style.display = 'none';
                emptyMsg.style.display = 'block';
                eliteStrategiesData = [];
                return;
            }

            // Store raw data for filtering/sorting
            eliteStrategiesData = validatedStrategies;

            // Populate symbol filter dropdown
            populateEliteSymbolFilter(validatedStrategies);

            // Apply current filters and sorting
            applyEliteSortAndFilter();
        }

        // Render elite table with filtered data
        function renderEliteTableFiltered(strategies) {
            const tbody = document.getElementById('eliteTableBody');
            const emptyMsg = document.getElementById('eliteEmpty');
            const table = document.getElementById('eliteTable');

            if (!strategies || strategies.length === 0) {
                table.style.display = 'none';
                emptyMsg.style.display = 'block';
                emptyMsg.textContent = eliteStrategiesData.length > 0
                    ? 'No strategies match your filters.'
                    : 'No elite strategies found. Run validation to identify consistent strategies.';
                return;
            }

            table.style.display = 'table';
            emptyMsg.style.display = 'none';
            tbody.innerHTML = '';
            // Reset expanded state since DOM is being rebuilt
            currentExpandedEliteRow = null;
            currentExpandedEliteId = null;

            // Helper to format P&L cell with validated date
            function formatPnL(periodData) {
                // Format the validated_at date compactly with time
                function formatDate(timestamp) {
                    if (!timestamp) return '';
                    try {
                        const date = new Date(timestamp);
                        const day = date.getDate();
                        const month = date.toLocaleString('en-GB', { month: 'short' });
                        const time = date.toLocaleTimeString('en-GB', { hour: 'numeric', minute: '2-digit', hour12: true });
                        return `${day} ${month} ${time}`;
                    } catch (e) {
                        return '';
                    }
                }

                const dateStr = formatDate(periodData?.validated_at);
                const dateHtml = dateStr ? `<br><small style="opacity: 0.4; font-size: 0.6rem;">${dateStr}</small>` : '';

                if (!periodData) return '<span style="color: var(--text-muted);">--</span>';
                if (periodData.status === 'limit_exceeded') return `<span style="color: var(--text-muted);">N/A${dateHtml}</span>`;
                if (periodData.status === 'no_trades') return `<span style="color: var(--text-muted);">0${dateHtml}</span>`;
                if (periodData.status === 'error') return `<span style="color: var(--text-muted);">err${dateHtml}</span>`;
                if (periodData.pnl === undefined) return '<span style="color: var(--text-muted);">--</span>';

                const pnl = periodData.pnl;
                // Use return_pct if available, otherwise calculate from PnL (£1000 capital)
                const returnPct = periodData.return_pct !== undefined
                    ? periodData.return_pct
                    : Math.round((pnl / 1000) * 1000) / 10;  // Calculate from PnL, round to 1 decimal
                const sign = pnl >= 0 ? '+' : '';
                const colorClass = pnl >= 0 ? 'success' : 'danger';

                // Always show both £ and %
                const pctSign = returnPct >= 0 ? '+' : '';
                return `<span class="${colorClass}">£${sign}${pnl.toFixed(0)}<br><small style="opacity: 0.7;">(${pctSign}${returnPct.toFixed(1)}%)</small>${dateHtml}</span>`;
            }

            // Helper to get period PnL value for sorting
            function getPeriodPnL(strategy, periodName) {
                try {
                    const validationResults = JSON.parse(strategy.elite_validation_data || '[]');
                    const period = validationResults.find(r => r.period === periodName);
                    return period?.pnl || 0;
                } catch (e) {
                    return 0;
                }
            }

            // Map sort column to period name
            const periodMap = {
                '1w': '1 week',
                '2w': '2 weeks',
                '1m': '1 month',
                '3m': '3 months',
                '6m': '6 months',
                '9m': '9 months',
                '1y': '1 year',
                '2y': '2 years',
                '3y': '3 years',
                '5y': '5 years'
            };

            // Debug: log first strategy to see structure
            if (strategies.length > 0) {
                console.log('First strategy from API:', strategies[0]);
                console.log('Strategy ID:', strategies[0].id);
            }

            // Group strategies by name+symbol+timeframe+direction
            const groups = {};
            strategies.forEach(s => {
                const direction = s.params?.direction || s.trade_mode || 'long';
                const key = `${s.strategy_name}|${s.symbol}|${s.timeframe}|${direction}`;
                if (!groups[key]) groups[key] = [];
                groups[key].push(s);
            });

            // Sort each group by elite_score descending (best first)
            Object.values(groups).forEach(group => {
                group.sort((a, b) => (b.elite_score || 0) - (a.elite_score || 0));
            });

            // Sort groups based on eliteSortBy column
            const sortedGroups = Object.values(groups).sort((a, b) => {
                const bestA = a[0];
                const bestB = b[0];
                let aVal, bVal;

                switch (eliteSortBy) {
                    case 'rank':
                        // Rank is determined by score, so same as score
                        aVal = bestA.elite_score || 0;
                        bVal = bestB.elite_score || 0;
                        break;
                    case 'score':
                        aVal = bestA.elite_score || 0;
                        bVal = bestB.elite_score || 0;
                        break;
                    case 'strategy':
                        aVal = bestA.strategy_name.toLowerCase();
                        bVal = bestB.strategy_name.toLowerCase();
                        break;
                    case 'status':
                        // Status sorting no longer needed - sort by score instead
                        aVal = bestA.elite_score || 0;
                        bVal = bestB.elite_score || 0;
                        break;
                    case '1w':
                    case '2w':
                    case '1m':
                    case '3m':
                    case '6m':
                    case '9m':
                    case '1y':
                    case '2y':
                    case '3y':
                    case '5y':
                        aVal = getPeriodPnL(bestA, periodMap[eliteSortBy]);
                        bVal = getPeriodPnL(bestB, periodMap[eliteSortBy]);
                        break;
                    default:
                        aVal = bestA.elite_score || 0;
                        bVal = bestB.elite_score || 0;
                }

                // Handle string comparison
                if (typeof aVal === 'string' && typeof bVal === 'string') {
                    return eliteSortOrder === 'asc'
                        ? aVal.localeCompare(bVal)
                        : bVal.localeCompare(aVal);
                }

                // Numeric comparison
                return eliteSortOrder === 'asc' ? aVal - bVal : bVal - aVal;
            });

            // Store grouped data for event delegation
            eliteGroupedData = sortedGroups;

            // Use DocumentFragment for batch DOM insert (single reflow instead of N)
            const fragment = document.createDocumentFragment();
            let rank = 0;
            let groupIndex = 0;
            sortedGroups.forEach(group => {
                const best = group[0];
                const hasVariants = group.length > 1;
                rank++;

                const direction = best.params?.direction || best.trade_mode || 'long';
                const dirClass = direction === 'short' ? 'danger' : 'success';
                const score = (best.elite_score || 0).toFixed(2);

                // Score color
                let scoreClass = 'danger';
                if (score >= 5) scoreClass = 'success';
                else if (score >= 3) scoreClass = 'warning';

                // Medal for top 3
                let medal = '';
                if (rank === 1) medal = '🥇';
                else if (rank === 2) medal = '🥈';
                else if (rank === 3) medal = '🥉';

                // Parse validation data for best strategy
                let periodData = {};
                try {
                    const validationResults = JSON.parse(best.elite_validation_data || '[]');
                    validationResults.forEach(r => { periodData[r.period] = r; });
                } catch (e) {}

                // TP/SL for best
                const tp = best.tp_percent || best.params?.tp_percent || '--';
                const sl = best.sl_percent || best.params?.sl_percent || '--';

                // Variant badge
                const variantBadge = hasVariants ? `<span style="background: var(--primary); color: white; padding: 0.1rem 0.4rem; border-radius: 10px; font-size: 0.7rem; margin-left: 0.3rem;">${group.length}</span>` : '';

                const row = document.createElement('tr');
                row.className = 'elite-row';
                row.dataset.groupKey = `${best.strategy_name}|${best.symbol}|${best.timeframe}|${direction}`;
                row.dataset.groupIndex = groupIndex;
                row.dataset.hasVariants = hasVariants ? 'true' : 'false';
                row.style.cursor = hasVariants ? 'pointer' : 'default';
                groupIndex++;

                row.innerHTML = `
                    <td><span class="expand-icon">${hasVariants ? '▶' : '•'}</span> ${medal}#${rank}</td>
                    <td class="${scoreClass}"><strong>${score}</strong></td>
                    <td style="text-align: left;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span class="status-badge ${dirClass}" style="font-size: 0.65rem; padding: 0.1rem 0.35rem;">
                                ${direction.toUpperCase()}
                            </span>
                            <strong>${best.strategy_name}</strong>${variantBadge}
                        </div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem;">
                            ${best.symbol || 'BTCUSDT'} · ${best.timeframe || '15m'} · TP ${tp}% / SL ${sl}%
                        </div>
                    </td>
                    <td>${formatPnL(periodData['1 week'])}</td>
                    <td>${formatPnL(periodData['2 weeks'])}</td>
                    <td>${formatPnL(periodData['1 month'])}</td>
                    <td>${formatPnL(periodData['3 months'])}</td>
                    <td>${formatPnL(periodData['6 months'])}</td>
                    <td>${formatPnL(periodData['9 months'])}</td>
                    <td>${formatPnL(periodData['1 year'])}</td>
                    <td>${formatPnL(periodData['2 years'])}</td>
                    <td>${formatPnL(periodData['3 years'])}</td>
                    <td>${formatPnL(periodData['5 years'])}</td>
                    <td>
                        <button class="btn btn-small" onclick="event.stopPropagation(); showPineScript(${best.id})" title="Pine Script">📋</button>
                        <button class="btn btn-small" onclick="event.stopPropagation(); showExportPeriodModal(${best.id}, '${best.strategy_name.replace(/'/g, "\\'")}')" title="Export Trades CSV">📊</button>
                        <button class="btn btn-small" onclick="event.stopPropagation(); validateFromHistory(${best.id})" title="Re-validate">🔬</button>
                    </td>
                `;

                // Event handling is done via delegation in setupEliteTableDelegation()
                fragment.appendChild(row);

                // Create variants container row (hidden by default)
                if (hasVariants) {
                    const variantsRow = document.createElement('tr');
                    variantsRow.className = 'elite-variants-row';
                    variantsRow.id = `elite-variants-${best.id}`;
                    variantsRow.style.display = 'none';
                    variantsRow.innerHTML = `<td colspan="14" style="padding: 0; background: var(--bg-secondary);"></td>`;
                    fragment.appendChild(variantsRow);

                    // Restore expanded state if this was the previously expanded group
                    const groupKey = `${best.strategy_name}|${best.symbol}|${best.timeframe}|${direction}`;
                    if (expandedEliteGroupKey === groupKey) {
                        // Re-expand this group
                        setTimeout(() => toggleEliteVariants(row, group, formatPnL), 0);
                    }
                }
            });

            // Single DOM operation: append all rows at once
            tbody.appendChild(fragment);
        }

        // Toggle elite variants dropdown
        function toggleEliteVariants(rowElement, group, formatPnL) {
            const best = group[0];
            const direction = best.params?.direction || best.trade_mode || 'long';
            const groupKey = `${best.strategy_name}|${best.symbol}|${best.timeframe}|${direction}`;
            const variantsRow = document.getElementById(`elite-variants-${best.id}`);
            const expandIcon = rowElement.querySelector('.expand-icon');

            if (!variantsRow) return;

            // Close previously expanded row (O(1) instead of O(n²))
            if (currentExpandedEliteRow && currentExpandedEliteRow !== rowElement) {
                const prevVariantsRow = document.getElementById(`elite-variants-${currentExpandedEliteId}`);
                if (prevVariantsRow) prevVariantsRow.style.display = 'none';
                currentExpandedEliteRow.classList.remove('expanded');
                const prevIcon = currentExpandedEliteRow.querySelector('.expand-icon');
                if (prevIcon) prevIcon.textContent = '▶';
            }

            // Toggle current row
            if (variantsRow.style.display === 'none') {
                variantsRow.style.display = 'table-row';
                rowElement.classList.add('expanded');
                expandIcon.textContent = '▼';
                expandedEliteGroupKey = groupKey;  // Track expanded state
                // Track expanded row for O(1) collapse
                currentExpandedEliteRow = rowElement;
                currentExpandedEliteId = best.id;

                // Build variants table
                const variantsCell = variantsRow.querySelector('td');
                let html = `<div style="padding: 0.5rem 1rem 0.5rem 2rem;">
                    <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                        <thead>
                            <tr style="background: var(--bg-card);">
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">TP%</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">SL%</th>
                                <th style="padding: 0.3rem; text-align: left; color: var(--text-muted);">Score</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">1W</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">2W</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">1M</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">3M</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">6M</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">9M</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">1Y</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">2Y</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">3Y</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">5Y</th>
                                <th style="padding: 0.3rem; text-align: center; color: var(--text-muted);">Actions</th>
                            </tr>
                        </thead>
                        <tbody>`;

                group.forEach((variant, idx) => {
                    const tp = variant.tp_percent || variant.params?.tp_percent || '--';
                    const sl = variant.sl_percent || variant.params?.sl_percent || '--';
                    const score = (variant.elite_score || 0).toFixed(2);
                    const isBest = idx === 0;

                    // Parse period data for this variant
                    let periodData = {};
                    try {
                        const validationResults = JSON.parse(variant.elite_validation_data || '[]');
                        validationResults.forEach(r => { periodData[r.period] = r; });
                    } catch (e) {}

                    html += `
                        <tr style="border-bottom: 1px solid var(--border-subtle); ${isBest ? 'background: rgba(34, 197, 94, 0.1);' : ''}">
                            <td style="padding: 0.4rem; color: var(--success);">${tp}${isBest ? ' <span style="color: var(--success); font-size: 0.7rem;">★ Best</span>' : ''}</td>
                            <td style="padding: 0.4rem; color: var(--danger);">${sl}</td>
                            <td style="padding: 0.4rem;">${score}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['1 week'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['2 weeks'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['1 month'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['3 months'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['6 months'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['9 months'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['1 year'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['2 years'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['3 years'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">${formatPnL(periodData['5 years'])}</td>
                            <td style="padding: 0.4rem; text-align: center;">
                                <button class="btn btn-small" onclick="event.stopPropagation(); showPineScript(${variant.id})" title="Pine Script">📋</button>
                                <button class="btn btn-small" onclick="event.stopPropagation(); showExportPeriodModal(${variant.id}, '${variant.strategy_name.replace(/'/g, "\\'")}')" title="Export Trades CSV">📊</button>
                                <button class="btn btn-small" onclick="event.stopPropagation(); validateFromHistory(${variant.id})" title="Re-validate">🔬</button>
                            </td>
                        </tr>`;
                });

                html += `</tbody></table></div>`;
                variantsCell.innerHTML = html;
            } else {
                variantsRow.style.display = 'none';
                rowElement.classList.remove('expanded');
                expandIcon.textContent = '▶';
                expandedEliteGroupKey = null;  // Clear expanded state
                // Clear expanded row tracking
                currentExpandedEliteRow = null;
                currentExpandedEliteId = null;
            }
        }


        // =====================================================
        // AUTONOMOUS OPTIMIZER
        // =====================================================

        async function initAutonomousTab() {
            autonomousInitialized = true;
            // WebSocket provides all updates - no polling needed
            console.log('[Autonomous] Tab initialized - WebSocket provides updates');
        }

        async function toggleAutonomousOptimizer() {
            const toggle = document.getElementById('autonomousToggle');
            const label = document.getElementById('autonomousToggleLabel');

            try {
                const response = await fetch('/api/autonomous/toggle', { method: 'POST' });
                const data = await response.json();

                if (data.status === 'enabled') {
                    label.textContent = 'ON';
                    label.style.color = 'var(--success)';
                    addLog('Autonomous optimizer enabled', 'normal');
                } else {
                    label.textContent = 'OFF';
                    label.style.color = 'var(--text-secondary)';
                    addLog('Autonomous optimizer disabled', 'normal');
                }

                updateAutonomousStatus();
            } catch (error) {
                console.error('Failed to toggle autonomous optimizer:', error);
                toggle.checked = !toggle.checked;  // Revert toggle
            }
        }

        function startAutonomousPolling() {
            // DISABLED - WebSocket provides all updates
            // HTTP polling is wasteful and drains resources
            console.log('[Autonomous] Polling disabled - using WebSocket');
        }

        function stopAutonomousPolling() {
            if (autonomousPollingInterval) {
                clearInterval(autonomousPollingInterval);
                autonomousPollingInterval = null;
            }
        }

        // Toggle collapsible sections
        function toggleCollapsible(element) {
            element.classList.toggle('expanded');
        }

        // Render task queue
        async function renderTaskQueue() {
            try {
                // Load via WebSocket (much faster than HTTP)
                const queue = await loadQueueViaWs();

                const queueBody = document.getElementById('taskQueueBody');
                if (!queueBody) return;

                // If no queue data yet
                if (queue.total === 0) {
                    queueBody.innerHTML = `
                        <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                            Enable optimizer to see queue
                        </div>
                    `;
                    return;
                }

                let html = '';

                // Completed items (most recent first, show up to 3)
                const completedToShow = queue.completed.slice(0, 3);
                completedToShow.reverse().forEach(item => {
                    const statusIcon = item.status === 'completed' ? '✓' : item.status === 'skipped' ? '⏭' : '✗';
                    const statusClass = item.status;
                    const strategiesFound = item.strategies_found ?? 0;
                    const resultClass = strategiesFound > 0 ? 'has-strategies' : '';
                    const resultText = item.status === 'completed'
                        ? `${strategiesFound} strategies`
                        : item.status === 'skipped' ? 'Skipped' : 'Error';

                    html += `
                        <div class="task-queue-item ${statusClass}">
                            <span class="task-status-icon ${statusClass}">${statusIcon}</span>
                            <span>${item.pair}</span>
                            <span>${item.period}</span>
                            <span>${item.timeframe}</span>
                            <span>${item.granularity}</span>
                            <span class="task-result ${resultClass}">${resultText}</span>
                        </div>
                    `;
                });

                // PARALLEL: Show all running items (multiple in parallel)
                const runningItems = queue.running || [];
                if (runningItems.length > 0) {
                    runningItems.forEach((item, idx) => {
                        const progress = item.progress || 0;
                        const message = item.message || 'Processing...';

                        html += `
                            <div class="task-queue-item in-progress">
                                <span class="task-status-icon"><span class="spinner"></span></span>
                                <span>${item.pair}</span>
                                <span>${item.period}</span>
                                <span>${item.timeframe}</span>
                                <span>${item.granularity}</span>
                                <div class="task-progress">
                                    <div class="task-progress-bar">
                                        <div class="task-progress-fill" style="width: ${progress}%"></div>
                                    </div>
                                    <span class="task-progress-text">${message}</span>
                                </div>
                            </div>
                        `;
                    });
                } else if (queue.current) {
                    // Fallback for legacy single item
                    const progress = queue.trial_total > 0
                        ? Math.round((queue.trial_current / queue.trial_total) * 100)
                        : 0;
                    const strategyName = queue.current_strategy || 'Starting...';

                    html += `
                        <div class="task-queue-item in-progress">
                            <span class="task-status-icon"><span class="spinner"></span></span>
                            <span>${queue.current.pair}</span>
                            <span>${queue.current.period}</span>
                            <span>${queue.current.timeframe}</span>
                            <span>${queue.current.granularity}</span>
                            <div class="task-progress">
                                <div class="task-progress-bar">
                                    <div class="task-progress-fill" style="width: ${progress}%"></div>
                                </div>
                                <span class="task-progress-text">${strategyName} - ${queue.trial_current.toLocaleString()}/${queue.trial_total.toLocaleString()}</span>
                            </div>
                        </div>
                    `;
                }

                // Pending items
                queue.pending.forEach(item => {
                    html += `
                        <div class="task-queue-item pending">
                            <span class="task-status-icon pending">○</span>
                            <span>${item.pair}</span>
                            <span>${item.period}</span>
                            <span>${item.timeframe}</span>
                            <span>${item.granularity}</span>
                            <span class="task-result">Pending</span>
                        </div>
                    `;
                });

                // Show remaining count and parallel info
                let footerText = '';
                if (queue.parallel_count > 0) {
                    footerText += `<span style="color: var(--success);">⚡ ${queue.parallel_count}/${queue.max_parallel} parallel</span>`;
                }
                if (queue.pending_remaining > 0) {
                    footerText += ` · +${queue.pending_remaining} more combinations`;
                }

                if (footerText) {
                    html += `
                        <div class="task-queue-more">
                            ${footerText}
                        </div>
                    `;
                }

                queueBody.innerHTML = html || `
                    <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                        Queue empty - all combinations processed
                    </div>
                `;

            } catch (error) {
                console.error('Failed to load queue:', error);
            }
        }

        // Render task queue from cached/pushed data (no network request)
        // Uses requestAnimationFrame to batch DOM updates and prevent layout thrashing
        function renderTaskQueueFromCache(queue) {
            if (!queue) return;

            // Store latest queue data and schedule render
            pendingQueueRender = queue;
            if (queueRenderScheduled) return;  // Already scheduled, will use latest data

            queueRenderScheduled = true;
            requestAnimationFrame(() => {
                queueRenderScheduled = false;
                const queueData = pendingQueueRender;
                if (!queueData) return;

                actuallyRenderTaskQueue(queueData);
            });
        }

        // Internal function that does the actual DOM updates
        function actuallyRenderTaskQueue(queue) {
            const queueBody = document.getElementById('taskQueueBody');
            if (!queueBody) return;

            // If no queue data yet
            if (queue.total === 0) {
                queueBody.innerHTML = `
                    <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                        Enable optimizer to see queue
                    </div>
                `;
                return;
            }

            let html = '';

            // Completed items (most recent first, show up to 3)
            const completedToShow = (queue.completed || []).slice(0, 3);
            completedToShow.reverse().forEach(item => {
                const statusIcon = item.status === 'completed' ? '✓' : item.status === 'skipped' ? '⏭' : '✗';
                const statusClass = item.status;
                const strategiesFound = item.strategies_found ?? 0;
                const resultClass = strategiesFound > 0 ? 'has-strategies' : '';
                const resultText = item.status === 'completed'
                    ? `${strategiesFound} strategies`
                    : item.status === 'skipped' ? 'Skipped' : 'Error';

                html += `
                    <div class="task-queue-item ${statusClass}">
                        <span class="task-status-icon ${statusClass}">${statusIcon}</span>
                        <span>${item.pair}</span>
                        <span>${item.period}</span>
                        <span>${item.timeframe}</span>
                        <span>${item.granularity}</span>
                        <span class="task-result ${resultClass}">${resultText}</span>
                    </div>
                `;
            });

            // PARALLEL: Show all running items with percentage and ETA
            const runningItems = queue.running || [];
            if (runningItems.length > 0) {
                runningItems.forEach((item, idx) => {
                    const progress = item.progress || 0;

                    // Build progress message with trial info, percentage, and ETA
                    let progressMsg = '';
                    if (item.trial_current > 0 && item.trial_total > 0) {
                        progressMsg = `${item.pair} - ${item.trial_current.toLocaleString()}/${item.trial_total.toLocaleString()}`;
                    } else {
                        progressMsg = item.message || 'Processing...';
                    }

                    // Add percentage and ETA
                    let etaText = `(${progress}%)`;
                    if (item.estimated_remaining !== null && item.estimated_remaining !== undefined) {
                        etaText += ` | Est: ${formatETA(item.estimated_remaining)}`;
                    }

                    html += `
                        <div class="task-queue-item in-progress">
                            <span class="task-status-icon"><span class="spinner"></span></span>
                            <span>${item.pair}</span>
                            <span>${item.period}</span>
                            <span>${item.timeframe}</span>
                            <span>${item.granularity}</span>
                            <div class="task-progress">
                                <div class="task-progress-bar">
                                    <div class="task-progress-fill" style="width: ${progress}%"></div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span class="task-progress-text">${progressMsg}</span>
                                    <span style="color: var(--text-muted); font-size: 0.75rem; white-space: nowrap;">${etaText}</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
            }

            // Pending items
            (queue.pending || []).forEach(item => {
                html += `
                    <div class="task-queue-item pending">
                        <span class="task-status-icon pending">○</span>
                        <span>${item.pair}</span>
                        <span>${item.period}</span>
                        <span>${item.timeframe}</span>
                        <span>${item.granularity}</span>
                        <span class="task-result">Pending</span>
                    </div>
                `;
            });

            // Show remaining count and parallel info
            let footerText = '';
            if (queue.parallel_count > 0) {
                footerText += `<span style="color: var(--success);">⚡ ${queue.parallel_count}/${queue.max_parallel} parallel</span>`;
            }
            if (queue.pending_remaining > 0) {
                footerText += ` · +${queue.pending_remaining} more combinations`;
            }

            if (footerText) {
                html += `
                    <div class="task-queue-more">
                        ${footerText}
                    </div>
                `;
            }

            queueBody.innerHTML = html || `
                <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                    Queue empty - all combinations processed
                </div>
            `;
        }

        async function updateAutonomousStatus() {
            // Skip HTTP polling if WebSocket is connected - we get push updates
            if (wsConnected) {
                return;
            }
            try {
                const response = await fetch('/api/autonomous/status');
                const status = await response.json();

                // Update badge - check parallel_count for parallel processing
                const badge = document.getElementById('autonomousStatusBadge');
                const message = document.getElementById('autonomousStatusMessage');
                const parallelCount = status.parallel_count || 0;
                const maxParallel = status.max_parallel || 1;

                if (parallelCount > 0) {
                    // Parallel processing active
                    badge.className = 'status-badge success';
                    badge.textContent = `Running ${parallelCount}`;
                } else if (status.running) {
                    badge.className = 'status-badge success';
                    badge.textContent = 'Running';
                } else if (status.paused) {
                    badge.className = 'status-badge warning';
                    badge.textContent = 'Paused';
                } else if (status.enabled && status.auto_running) {
                    badge.className = 'status-badge success';
                    badge.textContent = 'Active';
                } else if (status.enabled) {
                    badge.className = 'status-badge neutral';
                    badge.textContent = 'Waiting';
                } else {
                    badge.className = 'status-badge neutral';
                    badge.textContent = 'Off';
                }

                message.textContent = status.message || 'Not running';

                // Update cycle progress (inline in status bar)
                const cycleProgress = document.getElementById('autonomousCycleProgress');
                if (cycleProgress) {
                    if (status.running && status.total_combinations > 0) {
                        const completed = status.completed_count || 0;
                        const total = status.total_combinations || 0;
                        const pct = Math.round((completed / total) * 100);
                        cycleProgress.textContent = `${completed} / ${total} (${pct}%)`;
                    } else {
                        cycleProgress.textContent = '';
                    }
                }

                // Update task queue
                await renderTaskQueue();

                // Update summary stats
                document.getElementById('autoCompletedCount').textContent = status.completed_count || 0;
                document.getElementById('autoSkippedCount').textContent = status.skipped_count || 0;
                document.getElementById('autoErrorCount').textContent = status.error_count || 0;
                document.getElementById('autoTotalCombinations').textContent = status.total_combinations || 0;
                document.getElementById('autoLastCompleted').textContent = status.last_completed_at
                    ? new Date(status.last_completed_at).toLocaleTimeString()
                    : '-';

                // Update skipped validations table
                const skippedSection = document.getElementById('autoSkippedSection');
                const skippedTableBody = document.getElementById('autoSkippedTableBody');
                const skippedCountLabel = document.getElementById('skippedCountLabel');
                if (status.skipped_validations && status.skipped_validations.length > 0) {
                    skippedSection.style.display = 'block';
                    skippedCountLabel.textContent = status.skipped_validations.length;
                    skippedTableBody.innerHTML = status.skipped_validations.map(skip => `
                        <tr>
                            <td>${new Date(skip.timestamp).toLocaleTimeString()}</td>
                            <td>${skip.pair || '-'}</td>
                            <td>${skip.period || '-'}</td>
                            <td>${skip.timeframe || '-'}</td>
                            <td style="color: #f59e0b;">${skip.coverage_pct || 0}%</td>
                        </tr>
                    `).join('');
                } else {
                    skippedSection.style.display = 'none';
                }

                // Update best strategy
                const bestSection = document.getElementById('autoBestStrategySection');
                if (status.best_strategy_found) {
                    bestSection.style.display = 'block';
                    const best = status.best_strategy_found;
                    document.getElementById('autoBestStrategyDetails').innerHTML = `
                        <div class="stat-item">
                            <div class="stat-label">Pair</div>
                            <div class="stat-value">${best.pair || '-'}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Timeframe</div>
                            <div class="stat-value">${best.timeframe || '-'}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Period</div>
                            <div class="stat-value">${best.period || '-'}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Strategy</div>
                            <div class="stat-value">${best.strategy || '-'}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">PnL</div>
                            <div class="stat-value" style="color: var(--success);">£${(best.pnl || 0).toFixed(2)}</div>
                        </div>
                    `;
                } else {
                    bestSection.style.display = 'none';
                }

                // Update last result
                const lastSection = document.getElementById('autoLastResultSection');
                if (status.last_result) {
                    lastSection.style.display = 'block';
                    const last = status.last_result;
                    const pnlColor = (last.pnl || 0) >= 0 ? 'var(--success)' : 'var(--danger)';
                    document.getElementById('autoLastResultDetails').innerHTML = `
                        <span style="color: var(--text-primary);">${last.pair || '-'}</span>
                        <span style="color: var(--text-muted);"> ${last.timeframe || '-'} (${last.period || '-'})</span>
                        <span style="margin-left: 1rem;">→</span>
                        <span style="margin-left: 0.5rem; color: var(--accent-primary);">${last.strategy || '-'}</span>
                        <span style="margin-left: 1rem; color: ${pnlColor};">£${(last.pnl || 0).toFixed(2)}</span>
                        <span style="margin-left: 0.5rem; color: var(--text-muted);">(${(last.win_rate || 0).toFixed(1)}% WR)</span>
                    `;
                } else {
                    lastSection.style.display = 'none';
                }

                // Update toggle state
                const toggle = document.getElementById('autonomousToggle');
                const toggleLabel = document.getElementById('autonomousToggleLabel');
                if (toggle && toggleLabel) {
                    toggle.checked = status.enabled || false;
                    toggleLabel.textContent = status.enabled ? 'ON' : 'OFF';
                    toggleLabel.style.color = status.enabled ? 'var(--success)' : 'var(--text-secondary)';
                }

                // Update history table
                await updateAutonomousHistory();

            } catch (error) {
                console.error('Failed to update autonomous status:', error);
            }
        }

        async function updateAutonomousHistory() {
            try {
                const response = await fetch('/api/autonomous/history?limit=50');
                const data = await response.json();

                const tableBody = document.getElementById('autoHistoryTableBody');
                const emptyMessage = document.getElementById('autoHistoryEmpty');
                const table = document.getElementById('autoHistoryTable');

                if (!data.history || data.history.length === 0) {
                    tableBody.innerHTML = '';
                    emptyMessage.style.display = 'block';
                    table.style.display = 'none';
                    return;
                }

                emptyMessage.style.display = 'none';
                table.style.display = 'table';

                tableBody.innerHTML = data.history.map(run => {
                    const pnlColor = (run.best_pnl || 0) >= 0 ? 'var(--success)' : 'var(--danger)';

                    // Status-based coloring
                    let statusColor, statusIcon, statusTitle;
                    switch (run.status) {
                        case 'success':
                            statusColor = 'var(--success)';
                            statusIcon = '';
                            statusTitle = 'Completed successfully';
                            break;
                        case 'partial':
                            statusColor = 'var(--warning)';
                            statusIcon = '⚠️ ';
                            statusTitle = `Partial: ${run.skipped_combinations || 0} combinations skipped due to stalls`;
                            break;
                        case 'stalled':
                            statusColor = 'var(--danger)';
                            statusIcon = '❌ ';
                            statusTitle = `Stalled: ${run.skipped_combinations || 0} combinations skipped`;
                            break;
                        default:
                            statusColor = 'var(--text-muted)';
                            statusIcon = '';
                            statusTitle = 'No results';
                    }

                    const completedAt = new Date(run.completed_at);
                    const timeStr = completedAt.toLocaleString('en-GB', {
                        day: '2-digit',
                        month: 'short',
                        hour: '2-digit',
                        minute: '2-digit'
                    });

                    // Show skipped count if any
                    const skippedInfo = run.skipped_combinations > 0
                        ? ` <small style="color: var(--danger);">(${run.skipped_combinations} skipped)</small>`
                        : '';

                    return `
                        <tr title="${statusTitle}">
                            <td style="white-space: nowrap;">${timeStr}</td>
                            <td>${(run.source || '-').toUpperCase()}</td>
                            <td><strong>${run.pair || '-'}</strong></td>
                            <td>${run.period || '-'}</td>
                            <td>${run.timeframe || '-'}</td>
                            <td>${run.granularity || '-'}</td>
                            <td style="color: ${statusColor};">${statusIcon}${run.strategies_found || 0}${skippedInfo}</td>
                            <td style="color: ${pnlColor}; font-weight: 600;">£${(run.best_pnl || 0).toFixed(2)}</td>
                        </tr>
                    `;
                }).join('');

            } catch (error) {
                console.error('Failed to update autonomous history:', error);
            }
        }

        function showPineModal(title, script) {
            document.getElementById('pineModalTitle').textContent = title;
            // Normalize line endings and remove any invisible characters
            const cleanScript = script
                .replace(/\r\n/g, '\n')  // Windows -> Unix
                .replace(/\r/g, '\n')     // Old Mac -> Unix
                .replace(/\uFEFF/g, '')   // Remove BOM
                .replace(/[\u200B-\u200D\uFEFF]/g, '')  // Remove zero-width chars
                .replace(/\u00A0/g, ' '); // Replace non-breaking space with regular space
            document.getElementById('pineScriptContent').value = cleanScript;
            document.getElementById('pineScriptModal').style.display = 'flex';
            // Auto-select the content
            setTimeout(() => {
                const textarea = document.getElementById('pineScriptContent');
                textarea.focus();
                textarea.select();
            }, 100);
        }

        function closePineModal() {
            document.getElementById('pineScriptModal').style.display = 'none';
        }

        function selectAllPineScript() {
            const textarea = document.getElementById('pineScriptContent');
            textarea.focus();
            textarea.select();
        }

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closePineModal();
                closeExportModal();
            }
        });

        // Close modal when clicking outside
        document.getElementById('pineScriptModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'pineScriptModal') closePineModal();
        });
        document.getElementById('exportPeriodModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'exportPeriodModal') closeExportModal();
        });

        // Export Period Modal Functions
        let currentExportStrategyId = null;
        let currentExportStrategyName = null;

        const exportPeriods = [
            { name: '1w', label: '1 Week', months: 0.25 },
            { name: '2w', label: '2 Weeks', months: 0.5 },
            { name: '1m', label: '1 Month', months: 1 },
            { name: '3m', label: '3 Months', months: 3 },
            { name: '6m', label: '6 Months', months: 6 },
            { name: '9m', label: '9 Months', months: 9 },
            { name: '1y', label: '1 Year', months: 12 },
            { name: '2y', label: '2 Years', months: 24 },
            { name: '3y', label: '3 Years', months: 36 },
            { name: '5y', label: '5 Years', months: 60 }
        ];

        function showExportPeriodModal(strategyId, strategyName) {
            if (!strategyId) {
                showToast('Error: No strategy ID provided', 'error');
                console.error('showExportPeriodModal called with invalid strategyId:', strategyId);
                return;
            }
            currentExportStrategyId = strategyId;
            currentExportStrategyName = strategyName || 'Unknown';
            document.getElementById('exportModalTitle').textContent = `Export Trades: ${currentExportStrategyName}`;

            // Build period checkboxes
            const container = document.getElementById('exportPeriodOptions');
            container.innerHTML = exportPeriods.map(p => `
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.5rem; background: var(--bg-secondary); border-radius: 6px; color: var(--text-primary);">
                    <input type="checkbox" name="exportPeriod" value="${p.name}" data-months="${p.months}" style="width: 16px; height: 16px;">
                    <span>${p.label}</span>
                </label>
            `).join('');

            document.getElementById('exportSelectAll').checked = false;
            document.getElementById('exportPeriodModal').style.display = 'flex';
        }

        function closeExportModal() {
            document.getElementById('exportPeriodModal').style.display = 'none';
            currentExportStrategyId = null;
            currentExportStrategyName = null;
        }

        function toggleExportSelectAll() {
            const selectAll = document.getElementById('exportSelectAll').checked;
            document.querySelectorAll('input[name="exportPeriod"]').forEach(cb => {
                cb.checked = selectAll;
            });
        }

        async function executeExport() {
            const selected = Array.from(document.querySelectorAll('input[name="exportPeriod"]:checked'));
            if (selected.length === 0) {
                showToast('Please select at least one period', 'error');
                return;
            }

            if (!currentExportStrategyId) {
                showToast('Error: No strategy selected', 'error');
                console.error('executeExport: currentExportStrategyId is', currentExportStrategyId);
                return;
            }

            closeExportModal();
            showToast(`Exporting ${selected.length} period(s)...`);

            // Export each selected period
            for (const cb of selected) {
                const periodName = cb.value;
                const months = cb.dataset.months;
                try {
                    const url = `/api/export-trades/${currentExportStrategyId}?months=${months}&period_name=${periodName}`;
                    console.log('Fetching:', url);
                    const response = await fetch(url);
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                        throw new Error(errorData.detail || `HTTP ${response.status}`);
                    }

                    const disposition = response.headers.get('Content-Disposition');
                    let filename = `${currentExportStrategyName}_${periodName}_trades.csv`;
                    if (disposition) {
                        const match = disposition.match(/filename="?([^"]+)"?/);
                        if (match) filename = match[1];
                    }

                    const blob = await response.blob();
                    const blobUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = blobUrl;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(blobUrl);

                    // Small delay between downloads
                    if (selected.length > 1) {
                        await new Promise(r => setTimeout(r, 500));
                    }
                } catch (error) {
                    console.error(`Failed to export ${periodName}:`, error);
                    showToast(`Failed to export ${periodName}: ${error.message}`, 'error');
                }
            }

            showToast(`Exported ${selected.length} CSV file(s)!`);
        }
