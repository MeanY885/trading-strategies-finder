"""
WEBSOCKET MANAGER
=================
Manages WebSocket connections for real-time UI updates.
Extracted from main.py for better modularity.
"""
import asyncio
import json
import time
from datetime import datetime, date
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket

from logging_config import log
from utils.converters import get_pending_queue_items


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
        self.active_connections: Set[WebSocket] = set()  # Use Set for O(1) lookup/removal
        self._lock = asyncio.Lock()
        self._throttle_lock = asyncio.Lock()  # Separate lock for atomic throttle check
        self._broadcast_queue: asyncio.Queue = None
        self._broadcast_task: asyncio.Task = None
        self._throttle_interval = 1.0  # Minimum seconds between broadcasts per message type (increased from 0.5 for better UI performance at scale)
        self._last_broadcast_times: Dict[str, float] = {}  # Per-type throttling
        self._throttled_types = {"autonomous_status", "elite_status", "optimization_status"}  # Types to throttle
        self._main_loop = None  # Store reference to main event loop

    def set_main_loop(self, loop):
        """Store reference to main event loop for cross-thread broadcasts."""
        self._main_loop = loop
        log("[WebSocket] Main event loop registered")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)  # O(1) set add
        log(f"[WebSocket] Client connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        async with self._lock:
            self.active_connections.discard(websocket)  # O(1) set discard, no error if missing
        log(f"[WebSocket] Client disconnected. Total: {len(self.active_connections)}")

    async def send_to_client(self, websocket: WebSocket, message: Dict) -> bool:
        """
        Send a message to a specific client.
        Returns True if successful, False otherwise.
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            log(f"[WebSocket] Error sending to client: {e}", level='WARNING')
            return False

    async def broadcast(self, message_type: str, data: Dict) -> None:
        """
        Broadcast a message to all connected clients.
        Thread-safe - can be called from sync code via broadcast_sync.
        Throttles frequent message types to reduce network/UI load.
        """
        if not self.active_connections:
            return

        # Apply throttling for frequent message types with atomic check-and-update
        if message_type in self._throttled_types:
            async with self._throttle_lock:
                now = time.time()
                last_time = self._last_broadcast_times.get(message_type, 0)
                if now - last_time < self._throttle_interval:
                    return  # Skip this broadcast - too soon
                self._last_broadcast_times[message_type] = now

        # Serialize data to handle datetime objects
        message = serialize_for_json({"type": message_type, **data})

        async with self._lock:
            connections_copy = list(self.active_connections)

        disconnected = []
        sent_count = 0
        for connection in connections_copy:
            try:
                await connection.send_json(message)
                sent_count += 1
            except Exception as e:
                # Client disconnected mid-broadcast - this is normal
                error_msg = str(e) if str(e) else type(e).__name__
                log(f"[WebSocket] Client disconnected during {message_type}: {error_msg}", level='DEBUG')
                disconnected.append(connection)

        if sent_count > 0:
            log(f"[WebSocket] Broadcast {message_type} completed: {sent_count} clients")

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    self.active_connections.discard(conn)  # O(1) set discard

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
                log(f"[WebSocket] No main loop available for broadcast: {message_type}", level='WARNING')
                # Fallback: try to get current loop
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast(message_type, data),
                        loop
                    )
                except RuntimeError:
                    log("[WebSocket] No event loop available for broadcast", level='WARNING')
        except Exception as e:
            log(f"[WebSocket] broadcast_sync error: {e}", level='WARNING')

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

    # Remove large arrays from status before broadcasting (combinations_list can be 100KB+)
    status_for_broadcast = {k: v for k, v in autonomous_status.items()
                           if k not in ("combinations_list", "queue_completed")}

    payload = {"autonomous": status_for_broadcast}
    if queue_data:
        payload["queue"] = queue_data

    ws_manager.broadcast_sync("autonomous_status", payload)


def _get_queue_data_from_status(status: Dict) -> Dict:
    """Generate queue data from autonomous status for WebSocket broadcasts."""
    completed = status.get("queue_completed", [])[-10:]  # Last 10
    running = status.get("parallel_running", [])
    combinations = status.get("combinations_list", [])
    cycle_index = status.get("cycle_index", 0)
    total = len(combinations) if combinations else status.get("total_combinations", 0)
    pending_count = total - cycle_index

    # Get pending items using shared helper
    pending = get_pending_queue_items(
        combinations=combinations,
        cycle_index=cycle_index,
        running_items=running,
        max_items=5,
        lookahead=10
    )

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
