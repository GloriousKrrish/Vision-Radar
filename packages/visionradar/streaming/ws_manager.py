"""
VisionRadar WebSocket Connection Manager.

Manages connection registries per session and provides non-blocking multi-client broadcast
with drop-not-queue backpressure and client exception isolation (Section E & I compliance).
"""

import asyncio
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect

from visionradar.streaming.ws_protocol import create_stream_status_message

logger = logging.getLogger("visionradar.streaming")


class ConnectionManager:
    def __init__(self):
        self._active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accepts WebSocket connection and registers client under session_id."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._active_connections:
                self._active_connections[session_id] = set()
            self._active_connections[session_id].add(websocket)

        logger.info(f"WebSocket client connected to session {session_id}. Active: {len(self._active_connections[session_id])}")

        # Send initial stream_status handshake
        status_msg = create_stream_status_message(
            session_id=session_id,
            status="connected",
            details={"active_clients": len(self._active_connections[session_id])}
        )
        try:
            await websocket.send_json(status_msg)
        except Exception as e:
            logger.warning(f"Failed to send initial stream_status to WebSocket client: {e}")

    async def disconnect(self, websocket: WebSocket, session_id: str):
        """Removes WebSocket client from session registry."""
        async with self._lock:
            if session_id in self._active_connections:
                self._active_connections[session_id].discard(websocket)
                if not self._active_connections[session_id]:
                    del self._active_connections[session_id]
        logger.info(f"WebSocket client disconnected from session {session_id}")

    def get_client_count(self, session_id: str) -> int:
        """Returns current active client count for session_id."""
        return len(self._active_connections.get(session_id, set()))

    async def _send_client_isolated(self, websocket: WebSocket, session_id: str, message: Dict[str, Any]):
        """Sends message to a single client with strict timeout and drop-not-queue backpressure."""
        try:
            # 50ms send timeout: if client send buffer is full, drop frame for this client
            await asyncio.wait_for(websocket.send_json(message), timeout=0.05)
        except (asyncio.TimeoutError, WebSocketDisconnect, Exception) as e:
            logger.debug(f"Dropping frame for slow/disconnected WS client in session {session_id}: {e}")
            # Do not block other clients; remove dead socket
            try:
                await self.disconnect(websocket, session_id)
            except Exception:
                pass

    async def broadcast(self, session_id: str, message: Dict[str, Any]):
        """
        Broadcasts message to all connected clients in session_id.
        Uses asyncio.gather with return_exceptions=True to guarantee no client blocks another.
        """
        clients = list(self._active_connections.get(session_id, set()))
        if not clients:
            return

        tasks = [
            self._send_client_isolated(ws, session_id, message)
            for ws in clients
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


# Global singleton instance for API routes
ws_manager = ConnectionManager()
