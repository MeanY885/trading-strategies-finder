"""
Logging configuration for BTCGBP ML Optimizer.

Provides structured, scannable CLI logging with:
- Color-coded output for Docker/terminal
- Log levels: DEBUG, INFO, WARNING, ERROR
- Component prefixes: [Autonomous Optimizer], [Elite Validation], etc.
- Backward-compatible log() function as print() replacement
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    WHITE = "\033[97m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    # Background colors (for prominent visibility like the screenshot)
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_ORANGE = "\033[48;5;208m"  # 256-color orange
    BG_PURPLE = "\033[48;5;93m"   # 256-color purple
    BG_PINK = "\033[48;5;205m"    # 256-color pink
    BG_DARK_GREEN = "\033[48;5;22m"
    BG_DARK_RED = "\033[48;5;52m"
    BG_DARK_BLUE = "\033[48;5;17m"
    BG_GRAY = "\033[100m"

    # Component styles (background + foreground for readability)
    AUTONOMOUS = f"\033[48;5;208m\033[30m"   # Orange bg, black text
    ELITE = f"\033[48;5;93m\033[97m"          # Purple bg, white text
    STARTUP = f"\033[42m\033[30m"             # Green bg, black text
    SHUTDOWN = f"\033[43m\033[30m"            # Yellow bg, black text
    DATA = f"\033[44m\033[97m"                # Blue bg, white text
    STRATEGY = f"\033[48;5;22m\033[97m"       # Dark green bg, white text
    VECTORBT = f"\033[46m\033[30m"            # Cyan bg, black text
    WEBSOCKET = f"\033[48;5;17m\033[97m"      # Dark blue bg, white text
    DEDUP = f"\033[48;5;205m\033[30m"         # Pink bg, black text
    PARALLEL = f"\033[48;5;208m\033[30m"      # Orange bg (same as autonomous)
    SMART = f"\033[48;5;205m\033[30m"         # Pink bg for smart dedup
    UDF = f"\033[48;5;33m\033[97m"            # Bright blue bg, white text

    # Level colors (for log level indication)
    DEBUG = "\033[90m"       # Gray
    INFO = ""                # Default
    WARNING = f"\033[43m\033[30m"  # Yellow bg, black text
    ERROR = f"\033[41m\033[97m"    # Red bg, white text
    SUCCESS = f"\033[42m\033[30m"  # Green bg, black text


class CLIFormatter(logging.Formatter):
    """Custom formatter for scannable CLI output with background colors."""

    # Component colors - background colors for high visibility
    COMPONENT_COLORS = {
        'autonomous': Colors.AUTONOMOUS,
        'elite': Colors.ELITE,
        'startup': Colors.STARTUP,
        'shutdown': Colors.SHUTDOWN,
        'data': Colors.DATA,
        'strategy': Colors.STRATEGY,
        'vectorbt': Colors.VECTORBT,
        'websocket': Colors.WEBSOCKET,
        'dedup': Colors.DEDUP,
        'parallel': Colors.PARALLEL,
        'smart': Colors.SMART,
        'udf': Colors.UDF,
        'app': '',
    }

    LEVEL_COLORS = {
        'DEBUG': Colors.DIM,
        'INFO': '',
        'WARNING': Colors.WARNING,
        'ERROR': Colors.ERROR,
    }

    def format(self, record):
        # Extract component from logger name
        component = record.name.split('.')[-1] if '.' in record.name else 'app'
        component_color = self.COMPONENT_COLORS.get(component, '')

        # Build prefix with short labels
        prefix_map = {
            'autonomous': ' AUTO ',
            'elite': ' ELITE ',
            'startup': ' START ',
            'shutdown': ' STOP ',
            'data': ' DATA ',
            'strategy': ' STRAT ',
            'vectorbt': ' VBT ',
            'websocket': ' WS ',
            'dedup': ' DEDUP ',
            'parallel': ' PARLL ',
            'smart': ' SMART ',
            'udf': ' UDF ',
            'app': ' APP ',
        }
        prefix = prefix_map.get(component, f' {component.upper()[:5]} ')

        # Time in HH:MM:SS format for scannability
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Level color - apply to whole line for warnings/errors
        level_color = self.LEVEL_COLORS.get(record.levelname, '')
        msg = record.getMessage()

        # For warnings/errors, apply background to whole message
        if record.levelname in ('WARNING', 'ERROR'):
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {component_color}{prefix}{Colors.RESET} {level_color} {msg} {Colors.RESET}"

        # Normal format: timestamp [colored prefix] message
        return f"{Colors.DIM}{timestamp}{Colors.RESET} {component_color}{prefix}{Colors.RESET} {msg}"


def setup_logging(level: str = None) -> logging.Logger:
    """
    Initialize application logging.

    Args:
        level: Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
               Defaults to LOG_LEVEL env var or 'DEBUG'

    Returns:
        Root application logger
    """
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'DEBUG')

    log_level = getattr(logging, level.upper(), logging.DEBUG)

    # Create root logger for our app
    root = logging.getLogger('mlopt')
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    # Add CLI handler with custom formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CLIFormatter())
    handler.setLevel(log_level)
    root.addHandler(handler)

    # Prevent propagation to root logger
    root.propagate = False

    return root


def get_logger(component: str) -> logging.Logger:
    """Get a logger for a specific component."""
    return logging.getLogger(f'mlopt.{component}')


# Pre-configured loggers for common components
autonomous_logger = get_logger('autonomous')
elite_logger = get_logger('elite')
startup_logger = get_logger('startup')
data_logger = get_logger('data')
strategy_logger = get_logger('strategy')
vectorbt_logger = get_logger('vectorbt')
websocket_logger = get_logger('websocket')
dedup_logger = get_logger('dedup')


def log(message: str, level: str = 'INFO', component: Optional[str] = None):
    """
    Log a message with automatic component detection from prefix.

    This is a drop-in replacement for print() that adds structured logging.
    Messages are color-coded by component for easy scanning in Docker logs.

    Components and their colors:
        - autonomous/parallel: Orange background
        - elite: Purple background
        - startup: Green background
        - shutdown: Yellow background
        - data: Blue background
        - strategy: Dark green background
        - vectorbt: Cyan background
        - websocket: Dark blue background
        - dedup/smart: Pink background
        - udf: Bright blue background

    Usage:
        log("[Autonomous Optimizer] Starting...")  # Auto-detects component
        log("Processing trial 45/200", component='autonomous')
        log("Warning: low coverage", level='WARNING')
    """
    # Auto-detect component from message prefix
    if component is None:
        if '[Autonomous' in message or '[Parallel' in message:
            component = 'autonomous'
            message = message.replace('[Autonomous Optimizer] ', '').replace('[Autonomous] ', '').replace('[Parallel Optimizer] ', '').replace('[Parallel] ', '')
        elif '[Elite' in message:
            component = 'elite'
            message = message.replace('[Elite Validation] ', '').replace('[Elite Auto-Validation] ', '').replace('[Elite] ', '')
        elif '[Startup]' in message:
            component = 'startup'
            message = message.replace('[Startup] ', '')
        elif '[Shutdown]' in message:
            component = 'shutdown'
            message = message.replace('[Shutdown] ', '')
        elif '[Data]' in message:
            component = 'data'
            message = message.replace('[Data] ', '')
        elif '[Strategy' in message or '[StrategyEngine]' in message:
            component = 'strategy'
            message = message.replace('[StrategyEngine] ', '').replace('[Strategy] ', '')
        elif '[VectorBT]' in message or '[VBT]' in message:
            component = 'vectorbt'
            message = message.replace('[VectorBT] ', '').replace('[VBT] ', '')
        elif '[WebSocket]' in message or '[WS]' in message:
            component = 'websocket'
            message = message.replace('[WebSocket] ', '').replace('[WS] ', '')
        elif '[Smart Dedup]' in message or '[Dedup]' in message:
            component = 'dedup'
            message = message.replace('[Smart Dedup] ', '').replace('[Dedup] ', '')
        elif '[UDF]' in message:
            component = 'udf'
            message = message.replace('[UDF] ', '')
        else:
            component = 'app'

    logger = get_logger(component)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)


# Uvicorn log config to suppress noisy access logs
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(levelname)s | %(message)s",
        },
        "access": {
            "format": "%(levelname)s | %(client_addr)s - %(request_line)s %(status_code)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "WARNING"},
        "uvicorn.error": {"level": "WARNING"},
        "uvicorn.access": {"handlers": ["access"], "level": "WARNING"},
    },
}


# Initialize logging on import
_root_logger = setup_logging()
