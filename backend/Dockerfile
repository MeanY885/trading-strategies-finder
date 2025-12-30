FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including build tools for TA-Lib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install TA-Lib C library from source (required for Python TA-Lib wrapper)
# This is the industry-standard technical analysis library that matches TradingView
# Note: We need to update config.guess and config.sub for ARM64 support
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.guess 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess' && \
    wget -O config.sub 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies including TA-Lib wrapper
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./backend/

# Create data and output directories
RUN mkdir -p /app/data /app/output

# Create Numba cache directory for persistent JIT caching
RUN mkdir -p /app/.numba_cache && chmod 777 /app/.numba_cache

# Expose port
EXPOSE 8080

# Run the application (auto-tune integrated into Python startup)
CMD ["python", "backend/main.py"]
