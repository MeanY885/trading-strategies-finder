"""
SERVICES PACKAGE
================
Business logic services for the BTCGBP ML Optimizer.
"""
from .websocket_manager import (
    ws_manager,
    WebSocketManager,
    broadcast_data_status,
    broadcast_optimization_status,
    broadcast_autonomous_status,
    broadcast_elite_status,
    broadcast_full_state,
    broadcast_strategy_result,
)
from .resource_monitor import resource_monitor, ResourceMonitor
from .autonomous_optimizer import (
    start_autonomous_optimizer,
    stop_autonomous_optimizer,
    build_optimization_combinations,
    has_period_boundary_crossed,
    autonomous_runs_history,
)
from .elite_validator import (
    start_auto_elite_validation,
    stop_elite_validation,
    validate_all_strategies,
    validate_strategy,
)

__all__ = [
    # WebSocket
    'ws_manager',
    'WebSocketManager',
    'broadcast_data_status',
    'broadcast_optimization_status',
    'broadcast_autonomous_status',
    'broadcast_elite_status',
    'broadcast_full_state',
    'broadcast_strategy_result',
    # Resource Monitor
    'resource_monitor',
    'ResourceMonitor',
    # Autonomous Optimizer
    'start_autonomous_optimizer',
    'stop_autonomous_optimizer',
    'build_optimization_combinations',
    'has_period_boundary_crossed',
    'autonomous_runs_history',
    # Elite Validator
    'start_auto_elite_validation',
    'stop_elite_validation',
    'validate_all_strategies',
    'validate_strategy',
]
