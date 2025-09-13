"""
Unit tests for WebSocket Server (Story 3.2)
Tests real-time streaming infrastructure
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.streaming.websocket.websocket_server import (
    WebSocketServer,
    MessageType,
    ConnectionStatus
)
from src.streaming.websocket.connection_manager import (
    ConnectionManager,
    ConnectionState,
    ConnectionInfo
)


class TestWebSocketServer:
    """Test suite for WebSocketServer"""

    @pytest.fixture
    def server(self):
        """Create WebSocket server instance"""
        return WebSocketServer(host="localhost", port=8765)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection"""
        mock_ws = AsyncMock()
        mock_ws.remote_address = ("127.0.0.1", 12345)
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    @pytest.mark.asyncio
    async def test_server_initialization(self, server):
        """Test server initialization"""
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.heartbeat_interval == 30
        assert len(server.connections) == 0
        assert len(server.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_handle_client_connection(self, server, mock_websocket):
        """Test handling new client connection"""
        # Mock the message iteration
        messages = ['{"type": "heartbeat"}']
        mock_websocket.__aiter__.return_value = messages.__iter__()

        # Handle client connection
        await server.handle_client(mock_websocket, "/")

        # Verify initial connection message was sent
        mock_websocket.send.assert_called()
        call_args = mock_websocket.send.call_args[0][0]
        message = json.loads(call_args)

        assert message["type"] == ConnectionStatus.CONNECTION_STATUS.value
        assert message["status"] == ConnectionStatus.CONNECTED.value
        assert "client_id" in message

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, server):
        """Test handling subscription message"""
        client_id = "test-client-123"
        server.connections[client_id] = Mock()

        # Create subscription message
        data = {
            "type": "subscribe",
            "ward_ids": ["ward_a", "ward_b"]
        }

        await server.handle_message(client_id, json.dumps(data))

        # Verify subscriptions
        assert "ward_a" in server.subscriptions[client_id]
        assert "ward_b" in server.subscriptions[client_id]

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, server):
        """Test handling unsubscription message"""
        client_id = "test-client-123"
        server.connections[client_id] = Mock()
        server.subscriptions[client_id] = {"ward_a", "ward_b", "ward_c"}

        # Create unsubscription message
        data = {
            "type": "unsubscribe",
            "ward_ids": ["ward_b"]
        }

        await server.handle_message(client_id, json.dumps(data))

        # Verify unsubscription
        assert "ward_a" in server.subscriptions[client_id]
        assert "ward_b" not in server.subscriptions[client_id]
        assert "ward_c" in server.subscriptions[client_id]

    @pytest.mark.asyncio
    async def test_broadcast_to_ward(self, server, mock_websocket):
        """Test broadcasting message to ward subscribers"""
        # Setup clients with subscriptions
        client1 = "client-1"
        client2 = "client-2"
        client3 = "client-3"

        server.connections[client1] = mock_websocket
        server.connections[client2] = AsyncMock()
        server.connections[client3] = AsyncMock()

        server.subscriptions[client1] = {"ward_a", "ward_b"}
        server.subscriptions[client2] = {"ward_a"}
        server.subscriptions[client3] = {"ward_c"}

        # Broadcast to ward_a
        message = {"test": "data"}
        await server.broadcast_to_ward("ward_a", message)

        # Verify only ward_a subscribers received message
        server.connections[client1].send.assert_called_once()
        server.connections[client2].send.assert_called_once()
        server.connections[client3].send.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_patient_update(self, server):
        """Test broadcasting patient update"""
        client_id = "test-client"
        server.connections[client_id] = AsyncMock()
        server.subscriptions[client_id] = {"ward_a"}

        patient_data = {
            "patient_id": "P001",
            "ward_id": "ward_a",
            "news2_score": 5
        }

        await server.broadcast_patient_update(patient_data)

        # Verify message was sent
        server.connections[client_id].send.assert_called_once()
        sent_message = json.loads(server.connections[client_id].send.call_args[0][0])

        assert sent_message["type"] == MessageType.PATIENT_UPDATE.value
        assert sent_message["data"] == patient_data

    @pytest.mark.asyncio
    async def test_broadcast_alert_update(self, server):
        """Test broadcasting alert update"""
        client_id = "test-client"
        server.connections[client_id] = AsyncMock()
        server.subscriptions[client_id] = {"ward_a"}

        alert_data = {
            "alert_id": "A001",
            "ward_id": "ward_a",
            "priority": "critical",
            "message": "Patient deteriorating"
        }

        await server.broadcast_alert_update(alert_data)

        # Verify message was sent
        server.connections[client_id].send.assert_called_once()
        sent_message = json.loads(server.connections[client_id].send.call_args[0][0])

        assert sent_message["type"] == MessageType.ALERT_UPDATE.value
        assert sent_message["data"] == alert_data
        assert sent_message["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_send_batch_update(self, server):
        """Test sending batch updates"""
        client_id = "test-client"
        server.connections[client_id] = AsyncMock()
        server.subscriptions[client_id] = {"ward_a"}

        updates = [
            {"type": "patient", "id": "P001"},
            {"type": "patient", "id": "P002"},
            {"type": "alert", "id": "A001"}
        ]

        await server.send_batch_update("ward_a", updates)

        # Verify batch message was sent
        server.connections[client_id].send.assert_called_once()
        sent_message = json.loads(server.connections[client_id].send.call_args[0][0])

        assert sent_message["type"] == MessageType.BATCH_UPDATE.value
        assert sent_message["updates"] == updates
        assert sent_message["count"] == 3

    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, server):
        """Test heartbeat message handling"""
        client_id = "test-client"
        server.connections[client_id] = AsyncMock()

        await server.handle_heartbeat(client_id)

        # Verify heartbeat response
        server.connections[client_id].send.assert_called_once()
        sent_message = json.loads(server.connections[client_id].send.call_args[0][0])

        assert sent_message["type"] == MessageType.HEARTBEAT.value
        assert "timestamp" in sent_message

    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error message handling"""
        client_id = "test-client"
        server.connections[client_id] = AsyncMock()

        # Send invalid JSON
        await server.handle_message(client_id, "invalid json {")

        # Verify error message was sent
        server.connections[client_id].send.assert_called()
        sent_message = json.loads(server.connections[client_id].send.call_args[0][0])

        assert sent_message["type"] == MessageType.ERROR.value
        assert "Invalid JSON" in sent_message["error"]

    @pytest.mark.asyncio
    async def test_disconnect_client(self, server):
        """Test client disconnection"""
        client_id = "test-client"
        mock_ws = AsyncMock()
        server.connections[client_id] = mock_ws
        server.subscriptions[client_id] = {"ward_a"}
        initial_active = server._connection_stats["active_connections"] = 5

        await server.disconnect_client(client_id)

        # Verify cleanup
        assert client_id not in server.connections
        assert client_id not in server.subscriptions
        assert server._connection_stats["active_connections"] == initial_active - 1
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_stats(self, server):
        """Test connection statistics tracking"""
        # Add some connections
        server.connections["client1"] = Mock()
        server.connections["client2"] = Mock()
        server.subscriptions["client1"] = {"ward_a"}
        server.subscriptions["client2"] = {"ward_b", "ward_c"}

        server._connection_stats["messages_sent"] = 100
        server._connection_stats["messages_received"] = 50

        stats = server.get_connection_stats()

        assert stats["active_connections"] == 0  # Not incremented in this test
        assert stats["messages_sent"] == 100
        assert stats["messages_received"] == 50
        assert "subscriptions" in stats

    @pytest.mark.asyncio
    async def test_message_queuing(self, server):
        """Test message queuing for batch processing"""
        message = {
            "ward_id": "ward_a",
            "type": "update",
            "data": {"test": "data"}
        }

        await server.queue_message(message)

        # Verify message was queued
        queued_message = await asyncio.wait_for(server.message_queue.get(), timeout=1.0)
        assert queued_message == message


class TestConnectionManager:
    """Test suite for ConnectionManager"""

    @pytest.fixture
    def manager(self):
        """Create ConnectionManager instance"""
        return ConnectionManager(
            max_connections=100,
            max_connections_per_ip=5,
            session_timeout=3600
        )

    @pytest.mark.asyncio
    async def test_register_connection(self, manager):
        """Test registering new connection"""
        client_id = "client-123"
        ip_address = "192.168.1.1"

        success = await manager.register_connection(
            client_id=client_id,
            ip_address=ip_address,
            user_agent="TestAgent/1.0"
        )

        assert success is True
        assert client_id in manager.connections
        assert manager.connections[client_id].state == ConnectionState.CONNECTED
        assert ip_address in manager.ip_connections
        assert client_id in manager.ip_connections[ip_address]

    @pytest.mark.asyncio
    async def test_max_connections_limit(self, manager):
        """Test maximum connections limit"""
        manager.max_connections = 2

        # Register maximum connections
        assert await manager.register_connection("client1") is True
        assert await manager.register_connection("client2") is True

        # Should fail when limit reached
        assert await manager.register_connection("client3") is False

    @pytest.mark.asyncio
    async def test_max_connections_per_ip(self, manager):
        """Test maximum connections per IP limit"""
        manager.max_connections_per_ip = 2
        ip = "192.168.1.1"

        # Register maximum for IP
        assert await manager.register_connection("client1", ip) is True
        assert await manager.register_connection("client2", ip) is True

        # Should fail when IP limit reached
        assert await manager.register_connection("client3", ip) is False

        # Different IP should work
        assert await manager.register_connection("client4", "192.168.1.2") is True

    @pytest.mark.asyncio
    async def test_session_restoration(self, manager):
        """Test session restoration on reconnection"""
        # Register initial connection
        client_id = "client-123"
        await manager.register_connection(client_id)

        # Get session ID
        conn_info = manager.connections[client_id]
        session_id = conn_info.user_session

        # Add subscriptions
        manager.add_subscription(client_id, "ward_a")
        manager.add_subscription(client_id, "ward_b")

        # Restore session
        session_data = await manager.restore_session(session_id)

        assert session_data is not None
        assert session_data["client_id"] == client_id
        assert session_data["session_id"] == session_id
        assert "ward_a" in session_data["ward_subscriptions"]
        assert "ward_b" in session_data["ward_subscriptions"]

    @pytest.mark.asyncio
    async def test_reconnection_with_backoff(self, manager):
        """Test reconnection with exponential backoff"""
        client_id = "client-123"
        await manager.register_connection(client_id)

        # First reconnection attempt
        delay1 = await manager.mark_reconnecting(client_id)
        assert delay1 == 1  # First delay

        # Second reconnection attempt
        delay2 = await manager.mark_reconnecting(client_id)
        assert delay2 == 2  # Second delay

        # Third reconnection attempt
        delay3 = await manager.mark_reconnecting(client_id)
        assert delay3 == 4  # Exponential backoff

        conn_info = manager.connections[client_id]
        assert conn_info.state == ConnectionState.RECONNECTING
        assert conn_info.reconnect_attempts == 3

    @pytest.mark.asyncio
    async def test_activity_tracking(self, manager):
        """Test connection activity tracking"""
        client_id = "client-123"
        await manager.register_connection(client_id)

        initial_activity = manager.connections[client_id].last_activity

        # Wait a bit and update activity
        await asyncio.sleep(0.1)
        manager.update_activity(client_id)

        new_activity = manager.connections[client_id].last_activity
        assert new_activity > initial_activity

    @pytest.mark.asyncio
    async def test_subscription_management(self, manager):
        """Test ward subscription management"""
        client_id = "client-123"
        await manager.register_connection(client_id)

        # Add subscriptions
        manager.add_subscription(client_id, "ward_a")
        manager.add_subscription(client_id, "ward_b")

        conn_info = manager.connections[client_id]
        assert "ward_a" in conn_info.ward_subscriptions
        assert "ward_b" in conn_info.ward_subscriptions

        # Remove subscription
        manager.remove_subscription(client_id, "ward_a")
        assert "ward_a" not in conn_info.ward_subscriptions
        assert "ward_b" in conn_info.ward_subscriptions

    @pytest.mark.asyncio
    async def test_get_subscribers(self, manager):
        """Test getting subscribers for a ward"""
        # Register multiple clients
        await manager.register_connection("client1")
        await manager.register_connection("client2")
        await manager.register_connection("client3")

        # Add subscriptions
        manager.add_subscription("client1", "ward_a")
        manager.add_subscription("client2", "ward_a")
        manager.add_subscription("client3", "ward_b")

        # Get subscribers for ward_a
        subscribers = manager.get_subscribers("ward_a")

        assert len(subscribers) == 2
        assert "client1" in subscribers
        assert "client2" in subscribers
        assert "client3" not in subscribers

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, manager):
        """Test connection statistics tracking"""
        client_id = "client-123"
        await manager.register_connection(client_id)

        # Update statistics
        manager.update_stats(
            client_id,
            bytes_sent=1000,
            bytes_received=500,
            messages_sent=10,
            messages_received=5
        )

        conn_info = manager.connections[client_id]
        assert conn_info.bytes_sent == 1000
        assert conn_info.bytes_received == 500
        assert conn_info.messages_sent == 10
        assert conn_info.messages_received == 5

        # Get overall statistics
        stats = manager.get_statistics()
        assert stats["total_connections"] == 1
        assert stats["active_connections"] == 1
        assert stats["total_bytes_sent"] == 1000
        assert stats["total_messages_sent"] == 10

    @pytest.mark.asyncio
    async def test_rate_limiting(self, manager):
        """Test rate limiting functionality"""
        client_id = "client-123"
        await manager.register_connection(client_id)

        # Initially not rate limited
        assert manager.is_rate_limited(client_id, max_messages_per_minute=60) is False

        # Simulate excessive messages
        manager.update_stats(client_id, messages_received=100)

        # Should be rate limited
        assert manager.is_rate_limited(client_id, max_messages_per_minute=60) is True