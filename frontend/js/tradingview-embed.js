/**
 * TradingView Charting Library Integration
 * Embeds professional charts with strategy overlays
 *
 * NOTE: TradingView Charting Library files must be downloaded from TradingView
 * and placed in /frontend/charting_library/ directory.
 * See: https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/
 */

let tvWidget = null;

/**
 * Initialize the embedded TradingView chart widget
 * @param {string} containerId - DOM element ID to render the chart in
 * @param {string} symbol - Trading symbol (e.g., "BTCUSDT")
 * @param {string} resolution - Chart resolution (e.g., "60" for 1 hour)
 * @param {number} strategyId - Strategy ID for loading trade markers
 */
function initEmbeddedChart(containerId, symbol, resolution, strategyId) {
    // Clean up existing widget
    if (tvWidget) {
        try {
            tvWidget.remove();
        } catch (e) {
            console.warn('Error removing previous widget:', e);
        }
        tvWidget = null;
    }

    // Check if TradingView library is loaded
    if (typeof TradingView === 'undefined') {
        console.error('TradingView Charting Library not loaded');
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted);">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                        <div>TradingView Charting Library not available</div>
                        <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--text-secondary);">
                            The library files need to be installed in /charting_library/
                        </div>
                    </div>
                </div>
            `;
        }
        if (typeof showNotification === 'function') {
            showNotification('TradingView library not available', 'error');
        }
        return;
    }

    // Check if Datafeeds is available
    if (typeof Datafeeds === 'undefined' || typeof Datafeeds.UDFCompatibleDatafeed === 'undefined') {
        console.error('TradingView Datafeeds not loaded');
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted);">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                        <div>TradingView Datafeeds library not loaded</div>
                    </div>
                </div>
            `;
        }
        return;
    }

    // Create widget configuration
    const widgetOptions = {
        symbol: symbol,
        interval: resolution,
        container: containerId,
        datafeed: new Datafeeds.UDFCompatibleDatafeed('/api/udf'),
        library_path: '/static/charting_library/',
        locale: 'en',
        theme: 'Dark',
        autosize: true,
        timezone: 'Etc/UTC',
        toolbar_bg: '#1a1a2e',
        loading_screen: {
            backgroundColor: "#1a1a2e",
            foregroundColor: "#8b5cf6"
        },
        disabled_features: [
            'use_localstorage_for_settings',
            'header_symbol_search',
            'header_compare'
        ],
        enabled_features: [
            'study_templates',
            'hide_left_toolbar_by_default'
        ],
        overrides: {
            'mainSeriesProperties.style': 1,  // Candles
            'paneProperties.background': '#1a1a2e',
            'paneProperties.backgroundType': 'solid',
            'paneProperties.vertGridProperties.color': '#2a2a4e',
            'paneProperties.horzGridProperties.color': '#2a2a4e',
            'scalesProperties.textColor': '#AAA',
            'scalesProperties.backgroundColor': '#1a1a2e'
        },
        studies_overrides: {
            'volume.volume.color.0': '#ef4444',
            'volume.volume.color.1': '#22c55e'
        },
        custom_css_url: '/static/css/tradingview-overrides.css'
    };

    // Store strategy ID for marker loading
    if (strategyId) {
        widgetOptions.custom_indicators_getter = function(PineJS) {
            return Promise.resolve([]);
        };
    }

    try {
        tvWidget = new TradingView.widget(widgetOptions);

        tvWidget.onChartReady(() => {
            console.log('TradingView chart ready for', symbol);
            if (strategyId) {
                loadStrategyMarkers(strategyId);
            }
        });
    } catch (error) {
        console.error('Error creating TradingView widget:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--danger);">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">Error</div>
                        <div>Failed to initialize chart: ${error.message}</div>
                    </div>
                </div>
            `;
        }
    }
}

/**
 * Load strategy trade markers onto the chart
 * Markers are loaded via /api/udf/marks endpoint automatically by the datafeed
 * @param {number} strategyId - Strategy ID for loading markers
 */
async function loadStrategyMarkers(strategyId) {
    // The UDF compatible datafeed will automatically request marks from /api/udf/marks
    // We just need to ensure the strategyId is available for the request
    console.log('Strategy markers will be loaded for strategy:', strategyId);

    // Store the strategy ID globally for the datafeed to access
    window.currentTVStrategyId = strategyId;
}

/**
 * Open the TradingView chart modal with the specified symbol and timeframe
 * @param {string} symbol - Trading symbol (e.g., "BTCUSDT")
 * @param {number} timeframe - Timeframe in minutes (e.g., 60 for 1 hour)
 * @param {number} strategyId - Strategy ID for loading trade markers
 */
function openTVChart(symbol, timeframe, strategyId) {
    const container = document.getElementById('tv-chart-container');
    const backdrop = document.getElementById('tv-chart-backdrop');

    if (!container) {
        console.error('TradingView chart container not found');
        return;
    }

    // Show the container and backdrop
    container.classList.remove('hidden');
    if (backdrop) {
        backdrop.classList.remove('hidden');
    }

    // Convert timeframe (minutes) to TradingView resolution format
    const resolutionMap = {
        1: '1',
        5: '5',
        15: '15',
        30: '30',
        60: '60',
        120: '120',
        240: '240',
        360: '360',
        480: '480',
        720: '720',
        1440: 'D',      // Daily
        10080: 'W',     // Weekly
        43200: 'M'      // Monthly
    };

    const resolution = resolutionMap[timeframe] || '60';

    // Update the header title
    const headerTitle = container.querySelector('.tv-chart-header h3');
    if (headerTitle) {
        const tfLabel = timeframe >= 1440 ? (timeframe === 1440 ? 'Daily' : timeframe === 10080 ? 'Weekly' : 'Monthly')
                      : timeframe >= 60 ? `${timeframe / 60}H`
                      : `${timeframe}m`;
        headerTitle.textContent = `Strategy Visualization - ${symbol} (${tfLabel})`;
    }

    // Initialize the chart
    initEmbeddedChart('tv-chart-widget', symbol, resolution, strategyId);
}

/**
 * Close the TradingView chart modal and cleanup
 */
function closeTVChart() {
    const container = document.getElementById('tv-chart-container');
    const backdrop = document.getElementById('tv-chart-backdrop');

    if (container) {
        container.classList.add('hidden');
    }
    if (backdrop) {
        backdrop.classList.add('hidden');
    }

    // Clean up the widget
    if (tvWidget) {
        try {
            tvWidget.remove();
        } catch (e) {
            console.warn('Error removing widget:', e);
        }
        tvWidget = null;
    }

    // Clear the widget container
    const widgetContainer = document.getElementById('tv-chart-widget');
    if (widgetContainer) {
        widgetContainer.innerHTML = '';
    }

    // Clear the stored strategy ID
    window.currentTVStrategyId = null;
}

// Close chart on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const container = document.getElementById('tv-chart-container');
        if (container && !container.classList.contains('hidden')) {
            closeTVChart();
        }
    }
});
