# BTCGBP ML Optimizer

A Dockerized machine learning system for optimizing BTC/GBP sideways scalping strategies.

## Features

- **Kraken Data Fetching**: Automatically fetches historical 15-minute BTCGBP data
- **ML Optimization**: Uses Bayesian optimization (Optuna) to find optimal parameters
- **Web UI**: Clean, modern interface for controlling the optimization
- **Pine Script Output**: Generates ready-to-use TradingView Pine Script v6 code

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Internet connection (for fetching data from Kraken)

### Running the System

1. **Navigate to the project directory:**
   ```bash
   cd btcgbp-ml-optimizer
   ```

2. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

3. **Open the UI:**
   - Go to http://localhost:8080 in your browser

4. **Workflow:**
   1. Click "Fetch from Kraken" to download 12 months of data
   2. Wait for data to load (watch the status)
   3. Click "Start Optimization" to run ML parameter search
   4. Once complete, copy the generated Pine Script
   5. Paste into TradingView and test!

## How It Works

### Data Fetching
- Uses Kraken's public REST API
- Fetches BTCGBP 15-minute OHLCV candles
- Handles rate limiting automatically
- Stores data in `/data/btcgbp_15m.csv`

### ML Optimization
- **Algorithm**: TPE (Tree-structured Parzen Estimator) via Optuna
- **Train/Validation Split**: 70% training, 30% validation
- **Optimized Parameters**:
  - ADX Threshold (market regime detection)
  - Bollinger Band length and multiplier
  - RSI oversold/overbought levels
  - Stop loss distance
  - Take profit ratio

### Scoring Function
The optimizer maximizes a composite score considering:
- Profit Factor (35%)
- Win Rate (25%)
- Total PnL (25%)
- Trade Frequency (10%)
- Drawdown Penalty (-5%)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web UI |
| `/api/status` | GET | Get current system status |
| `/api/fetch-data` | POST | Start fetching data from Kraken |
| `/api/optimize` | POST | Start ML optimization |
| `/api/pinescript` | GET | Get generated Pine Script |
| `/api/backtest` | GET | Run single backtest with params |

## File Structure

```
btcgbp-ml-optimizer/
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Container build instructions
├── backend/
│   ├── main.py             # FastAPI application
│   ├── data_fetcher.py     # Kraken API client
│   ├── strategy.py         # Strategy backtesting logic
│   ├── optimizer.py        # Optuna ML optimizer
│   ├── pinescript_generator.py  # Pine Script output
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Web UI
├── data/                   # Downloaded data storage
└── output/                 # Generated Pine Scripts
```

## Troubleshooting

**Data fetch taking too long?**
- Kraken API has rate limits (~1 request/second)
- 12 months of data may take 2-3 minutes to fetch

**Optimization not improving?**
- Try increasing the number of trials (200-500)
- Check if the data has enough sideways periods

**Docker permission errors?**
- Ensure Docker is running
- Try `docker-compose down` then `docker-compose up --build`

## License

MIT License - Feel free to modify and use as needed.



