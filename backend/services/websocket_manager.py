"""
WEBSOCKET MANAGER
=================
Manages WebSocket connections for real-time UI updates.
Extracted from main.py for better modularity.
"""
import asyncio
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def serialize_for_json(data: Any) -> Any:
    """Recursively serialize data, converting datetime objects to ISO strings."""
    if isinstance(data, dict):
        return {k: serialize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_for_json(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    return data


class WebSocketManager:
    """
    Manages WebSocket connections for real-time UI updates.
    Replaces polling with push-based status updates.

    Thread-safe and supports broadcasting from both sync and async contexts.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._broadcast_queue: asyncio.Queue = None
        self._broadcast_task: asyncio.Task = None
        self._throttle_interval = 0.5  # Minimum seconds between broadcasts
        self._last_broadcast_time = 0
        self._main_loop = None  # Store reference to main event loop

    def set_main_loop(self, loop):
        """Store reference to main event loop for cross-thread broadcasts."""
        self._main_loop = loop
        logger.info(f"[WebSocket] Main event loop registered")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"[WebSocket] Client connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"[WebSocket] Client disconnected. Total: {len(self.active_connections)}")

    async def send_to_client(self, websocket: WebSocket, message: Dict) -> bool:
        """
        Send a message to a specific client.
        Returns True if successful, False otherwise.
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"[WebSocket] Error sending to client: {e}")
            return False

    async def broadcast(self, message_type: str, data: Dict) -> None:
        """
        Broadcast a message to all connected clients.
        Thread-safe - can be called from sync code via broadcast_sync.
        """
        if not self.active_connections:
            return

        # Serialize data to handle datetime objects
        message = serialize_for_json({"type": message_type, **data})

        async with self._lock:
            connections_copy = list(self.active_connections)

        disconnected = []
        for connection in connections_copy:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"[WebSocket] Error broadcasting {message_type} to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

    def broadcast_sync(self, message_type: str, data: Dict) -> None:
        """
        Thread-safe broadcast for use from synchronous code.
        Uses stored main event loop for reliable cross-thread communication.
        """
        try:
            # Check if we have clients to broadcast to
            if not self.active_connections:
                return  # No clients connected

            # Use stored main loop (set during startup)
            if self._main_loop and self._main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.broadcast(message_type, data),
                    self._main_loop
                )
                # Don't block waiting for result, just fire and forget
            else:
                logger.warning(f"[WebSocket] No main loop available for broadcast: {message_type}")
                # Fallback: try to get current loop
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast(message_type, data),
                        loop
                    )
                except RuntimeError:
                    logger.warning("[WebSocket] No event loop available for broadcast")
        except Exception as e:
            logger.warning(f"[WebSocket] broadcast_sync error: {e}")

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self.active_connections)

    def has_clients(self) -> bool:
        """Check if any clients are connected."""
        return len(self.active_connections) > 0


# Singleton instance
ws_manager = WebSocketManager()


# =============================================================================
# BROADCAST HELPER FUNCTIONS
# =============================================================================

def broadcast_data_status(data_status: Dict) -> None:
    """Broadcast data status update to all clients."""
    ws_manager.broadcast_sync("data_status", {"data": data_status})


def broadcast_optimization_status(unified_status: Dict) -> None:
    """Broadcast optimization status update to all clients."""
    ws_manager.broadcast_sync("optimization_status", {"optimization": unified_status})


def broadcast_autonomous_status(autonomous_status: Dict, queue_data: Optional[Dict] = None) -> None:
    """Broadcast autonomous optimizer status update to all clients, including queue data."""
    # Auto-generate queue data if not provided (like original code did)
    if queue_data is None:
        queue_data = _get_queue_data_from_status(autonomous_status)

    payload = {"autonomous": autonomous_status}
    if queue_data:
        payload["queue"] = queue_data

    # Debug: log broadcast attempt
    client_count = ws_manager.client_count
    if client_count > 0:
        logger.info(f"[WebSocket] Broadcasting autonomous_status to {client_count} clients")

    ws_manager.broadcast_sync("autonomous_status", payload)


def _get_queue_data_from_status(status: Dict) -> Dict:
    """Generate queue data from autonomous status for WebSocket broadcasts."""
    completed = status.get("queue_completed", [])[-10:]  # Last 10
    running = status.get("parallel_running", [])
    combinations = status.get("combinations_list", [])
    cycle_index = status.get("cycle_index", 0)
    total = len(combinations) if combinations else status.get("total_combinations", 0)
    pending_count = total - cycle_index

    # Get pending items from combinations list
    pending = []
    if combinations:
        running_indices = {r.get("index") for r in running}
        for i in range(cycle_index, min(cycle_index + 10, len(combinations))):
            if i not in running_indices:
                combo = combinations[i]
                pending.append({
                    "index": i,
                    "pair": combo.get("pair", ""),
                    "period": combo.get("period", ""),
                    "timeframe": combo.get("timeframe", ""),
                    "granularity": combo.get("granularity", ""),
                    "status": "pending"
                })
                if len(pending) >= 5:
                    break

    return {
        "total": total,
        "completed": completed,
        "running": running,
        "pending": pending,
        "pending_remaining": max(0, pending_count - len(pending)),
        "parallel_count": len(running),
        "max_parallel": status.get("max_parallel", 4),
        "cycle_index": cycle_index,
        "trial_current": status.get("trial_current", 0),
        "trial_total": status.get("trial_total", 0),
        "current_strategy": status.get("current_strategy", "")
    }


def broadcast_elite_status(elite_status: Dict) -> None:
    """Broadcast elite validation status update to all clients."""
    ws_manager.broadcast_sync("elite_status", {"elite": elite_status})


def broadcast_full_state(data: Dict, optimization: Dict, autonomous: Dict, elite: Dict) -> None:
    """Broadcast all status updates (useful after state changes)."""
    ws_manager.broadcast_sync("full_state", {
        "data": data,
        "optimization": optimization,
        "autonomous": autonomous,
        "elite": elite,
    })


def broadcast_strategy_result(result: Dict) -> None:
    """Broadcast a single strategy result (SSE replacement)."""
    ws_manager.broadcast_sync("strategy_result", {"strategy": result})
