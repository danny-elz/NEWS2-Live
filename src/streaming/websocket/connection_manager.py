"""
Connection Manager for Story 3.2
Manages WebSocket connections with pooling and load balancing
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    client_id: str
    connected_at: datetime
    last_activity: datetime
    state: ConnectionState
    reconnect_attempts: int = 0
    ward_subscriptions: Set[str] = field(default_factory=set)
    user_session: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0


class ConnectionManager:
    """
    Manages WebSocket connections with advanced features:
    - Connection pooling
    - Automatic reconnection with exponential backoff
    - Session preservation
    - Load balancing
    - Connection health monitoring
    """

    def __init__(self,
                 max_connections: int = 1000,
                 max_connections_per_ip: int = 10,
                 session_timeout: int = 28800,  # 8 hours in seconds
                 max_reconnect_attempts: int = 5):
        self.max_connections = max_connections
        self.max_connections_per_ip = max_connections_per_ip
        self.session_timeout = session_timeout
        self.max_reconnect_attempts = max_reconnect_attempts

        self.connections: Dict[str, ConnectionInfo] = {}
        self.sessions: Dict[str, str] = {}  # session_id -> client_id
        self.ip_connections: Dict[str, Set[str]] = {}  # ip -> set of client_ids

        self.reconnect_delays = [1, 2, 4, 8, 16, 30]  # Exponential backoff delays
        self._cleanup_task = None
        self._running = False

    async def start(self):
        """Start the connection manager"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection manager started")

    async def stop(self):
        """Stop the connection manager"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for client_id in list(self.connections.keys()):
            await self.disconnect(client_id)

        logger.info("Connection manager stopped")

    async def register_connection(self,
                                 client_id: str,
                                 ip_address: Optional[str] = None,
                                 user_agent: Optional[str] = None,
                                 session_id: Optional[str] = None) -> bool:
        """
        Register a new connection

        Args:
            client_id: Unique client identifier
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Optional session ID for reconnection

        Returns:
            True if registration successful, False otherwise
        """
        # Check connection limits
        if len(self.connections) >= self.max_connections:
            logger.warning(f"Max connections reached ({self.max_connections})")
            return False

        # Check per-IP limits
        if ip_address:
            if ip_address not in self.ip_connections:
                self.ip_connections[ip_address] = set()

            if len(self.ip_connections[ip_address]) >= self.max_connections_per_ip:
                logger.warning(f"Max connections per IP reached for {ip_address}")
                return False

            self.ip_connections[ip_address].add(client_id)

        # Check for existing session
        existing_client_id = None
        if session_id and session_id in self.sessions:
            existing_client_id = self.sessions[session_id]
            if existing_client_id in self.connections:
                # Restore session state
                old_conn = self.connections[existing_client_id]
                ward_subscriptions = old_conn.ward_subscriptions.copy()
                user_session = old_conn.user_session
            else:
                ward_subscriptions = set()
                user_session = session_id
        else:
            # Create new session
            session_id = self._generate_session_id()
            ward_subscriptions = set()
            user_session = session_id

        # Create connection info
        conn_info = ConnectionInfo(
            client_id=client_id,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            state=ConnectionState.CONNECTED,
            ward_subscriptions=ward_subscriptions,
            user_session=user_session,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.connections[client_id] = conn_info
        self.sessions[session_id] = client_id

        # Clean up old connection if session takeover
        if existing_client_id and existing_client_id != client_id:
            await self.disconnect(existing_client_id)

        logger.info(f"Connection registered: {client_id} (session: {session_id})")
        return True

    async def disconnect(self, client_id: str):
        """Disconnect and cleanup a client"""
        if client_id not in self.connections:
            return

        conn_info = self.connections[client_id]

        # Update state
        conn_info.state = ConnectionState.DISCONNECTED

        # Remove from IP tracking
        if conn_info.ip_address and conn_info.ip_address in self.ip_connections:
            self.ip_connections[conn_info.ip_address].discard(client_id)
            if not self.ip_connections[conn_info.ip_address]:
                del self.ip_connections[conn_info.ip_address]

        # Don't remove session immediately (allow reconnection)
        # Session cleanup happens in cleanup loop

        logger.info(f"Connection disconnected: {client_id}")

    async def mark_reconnecting(self, client_id: str) -> int:
        """
        Mark connection as reconnecting and return delay

        Returns:
            Reconnection delay in seconds
        """
        if client_id not in self.connections:
            return 0

        conn_info = self.connections[client_id]
        conn_info.state = ConnectionState.RECONNECTING
        conn_info.reconnect_attempts += 1

        # Calculate delay with exponential backoff
        delay_index = min(conn_info.reconnect_attempts - 1, len(self.reconnect_delays) - 1)
        delay = self.reconnect_delays[delay_index]

        logger.info(f"Connection {client_id} reconnecting (attempt {conn_info.reconnect_attempts}, delay {delay}s)")

        return delay

    async def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore session state for reconnection

        Returns:
            Session state including subscriptions and settings
        """
        if session_id not in self.sessions:
            return None

        client_id = self.sessions[session_id]
        if client_id not in self.connections:
            return None

        conn_info = self.connections[client_id]

        # Check session timeout
        session_age = (datetime.now() - conn_info.connected_at).total_seconds()
        if session_age > self.session_timeout:
            logger.info(f"Session {session_id} expired")
            return None

        return {
            "client_id": client_id,
            "session_id": session_id,
            "ward_subscriptions": list(conn_info.ward_subscriptions),
            "connected_at": conn_info.connected_at.isoformat(),
            "last_activity": conn_info.last_activity.isoformat()
        }

    def update_activity(self, client_id: str):
        """Update last activity timestamp for a connection"""
        if client_id in self.connections:
            self.connections[client_id].last_activity = datetime.now()

    def update_stats(self, client_id: str,
                    bytes_sent: int = 0,
                    bytes_received: int = 0,
                    messages_sent: int = 0,
                    messages_received: int = 0):
        """Update connection statistics"""
        if client_id in self.connections:
            conn = self.connections[client_id]
            conn.bytes_sent += bytes_sent
            conn.bytes_received += bytes_received
            conn.messages_sent += messages_sent
            conn.messages_received += messages_received

    def add_subscription(self, client_id: str, ward_id: str):
        """Add ward subscription for a client"""
        if client_id in self.connections:
            self.connections[client_id].ward_subscriptions.add(ward_id)

    def remove_subscription(self, client_id: str, ward_id: str):
        """Remove ward subscription for a client"""
        if client_id in self.connections:
            self.connections[client_id].ward_subscriptions.discard(ward_id)

    def get_connection_info(self, client_id: str) -> Optional[ConnectionInfo]:
        """Get connection information"""
        return self.connections.get(client_id)

    def get_subscribers(self, ward_id: str) -> List[str]:
        """Get all clients subscribed to a ward"""
        subscribers = []
        for client_id, conn_info in self.connections.items():
            if ward_id in conn_info.ward_subscriptions and conn_info.state == ConnectionState.CONNECTED:
                subscribers.append(client_id)
        return subscribers

    def get_statistics(self) -> Dict[str, Any]:
        """Get connection manager statistics"""
        total_connections = len(self.connections)
        active_connections = sum(
            1 for c in self.connections.values()
            if c.state == ConnectionState.CONNECTED
        )
        reconnecting = sum(
            1 for c in self.connections.values()
            if c.state == ConnectionState.RECONNECTING
        )

        total_bytes_sent = sum(c.bytes_sent for c in self.connections.values())
        total_bytes_received = sum(c.bytes_received for c in self.connections.values())
        total_messages_sent = sum(c.messages_sent for c in self.connections.values())
        total_messages_received = sum(c.messages_received for c in self.connections.values())

        return {
            "total_connections": total_connections,
            "active_connections": active_connections,
            "reconnecting_connections": reconnecting,
            "total_sessions": len(self.sessions),
            "unique_ips": len(self.ip_connections),
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "connection_limit": self.max_connections,
            "connections_available": self.max_connections - active_connections
        }

    def is_rate_limited(self, client_id: str, max_messages_per_minute: int = 60) -> bool:
        """Check if client is rate limited"""
        if client_id not in self.connections:
            return False

        conn = self.connections[client_id]
        # Simple rate limiting based on message count
        # In production, would use a sliding window or token bucket
        return conn.messages_received > max_messages_per_minute

    async def _cleanup_loop(self):
        """Periodic cleanup of stale connections and sessions"""
        cleanup_interval = 60  # seconds
        session_grace_period = 300  # 5 minutes after disconnect

        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)

                now = datetime.now()
                to_remove = []

                for client_id, conn_info in list(self.connections.items()):
                    # Remove failed connections
                    if conn_info.state == ConnectionState.FAILED:
                        if conn_info.reconnect_attempts >= self.max_reconnect_attempts:
                            to_remove.append(client_id)

                    # Remove disconnected connections after grace period
                    elif conn_info.state == ConnectionState.DISCONNECTED:
                        disconnect_age = (now - conn_info.last_activity).total_seconds()
                        if disconnect_age > session_grace_period:
                            to_remove.append(client_id)

                    # Remove expired sessions
                    elif conn_info.state == ConnectionState.CONNECTED:
                        session_age = (now - conn_info.connected_at).total_seconds()
                        inactivity = (now - conn_info.last_activity).total_seconds()

                        if session_age > self.session_timeout or inactivity > self.session_timeout:
                            to_remove.append(client_id)

                # Clean up connections
                for client_id in to_remove:
                    if client_id in self.connections:
                        conn = self.connections[client_id]

                        # Remove session
                        if conn.user_session in self.sessions:
                            del self.sessions[conn.user_session]

                        # Remove connection
                        del self.connections[client_id]

                        logger.info(f"Cleaned up connection: {client_id}")

                if to_remove:
                    logger.info(f"Cleanup removed {len(to_remove)} stale connections")

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)