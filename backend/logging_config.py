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

    # Component colors
    AUTONOMOUS = "\033[36m"  # Cyan
    ELITE = "\033[35m"       # Magenta
    STARTUP = "\033[32m"     # Green
    SHUTDOWN = "\033[33m"    # Yellow
    DATA = "\033[34m"        # Blue

    # Level colors
    DEBUG = "\033[90m"       # Gray
    INFO = "\033[37m"        # White
    WARNING = "\033[33m"     # Yellow
    ERROR = "\033[31m"       # Red
    SUCCESS = "\033[32m"     # Green


class CLIFormatter(logging.Formatter):
    """Custom formatter for scannable CLI output."""

    COMPONENT_COLORS = {
        'autonomous': Colors.AUTONOMOUS,
        'elite': Colors.ELITE,
        'startup': Colors.STARTUP,
        'shutdown': Colors.SHUTDOWN,
        'data': Colors.DATA,
        'app': Colors.INFO,
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

        # Build prefix
        prefix_map = {
            'autonomous': '[Autonomous Optimizer]',
            'elite': '[Elite Validation]',
            'startup': '[Startup]',
            'shutdown': '[Shutdown]',
            'data': '[Data]',
            'app': '[App]',
        }
        prefix = prefix_map.get(component, f'[{component.title()}]')

        # Time in HH:MM:SS format for scannability
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Level color
        level_color = self.LEVEL_COLORS.get(record.levelname, '')

        # Format: HH:MM:SS [Component] Message
        return f"{Colors.DIM}{timestamp}{Colors.RESET} {component_color}{prefix}{Colors.RESET} {level_color}{record.getMessage()}{Colors.RESET}"


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


def log(message: str, level: str = 'INFO', component: Optional[str] = None):
    """
    Log a message with automatic component detection from prefix.

    This is a drop-in replacement for print() that adds structured logging.

    Usage:
        log("[Autonomous Optimizer] Starting...")  # Auto-detects component
        log("Processing trial 45/200", component='autonomous')
        log("Warning: low coverage", level='WARNING')
    """
    # Auto-detect component from message prefix
    if component is None:
        if '[Autonomous' in message:
            component = 'autonomous'
            message = message.replace('[Autonomous Optimizer] ', '').replace('[Autonomous] ', '')
        elif '[Elite' in message:
            component = 'elite'
            message = message.replace('[Elite Validation] ', '').replace('[Elite Auto-Validation] ', '')
        elif '[Startup]' in message:
            component = 'startup'
            message = message.replace('[Startup] ', '')
        elif '[Shutdown]' in message:
            component = 'shutdown'
            message = message.replace('[Shutdown] ', '')
        elif '[Data]' in message:
            component = 'data'
            message = message.replace('[Data] ', '')
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
