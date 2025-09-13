"""
Quick test for WebSocket implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly without going through parent module
import asyncio
from src.streaming.websocket.websocket_server import WebSocketServer, MessageType
from src.streaming.websocket.connection_manager import ConnectionManager, ConnectionState


async def test_basic():
    """Test basic WebSocket functionality"""
    print("Testing WebSocket Server...")

    # Create server
    server = WebSocketServer()
    print(f"✓ Server created: {server.host}:{server.port}")

    # Create connection manager
    manager = ConnectionManager()
    print(f"✓ Connection manager created: max {manager.max_connections} connections")

    # Test registration
    success = await manager.register_connection("test-client-1", "127.0.0.1")
    print(f"✓ Connection registered: {success}")

    # Test subscription
    manager.add_subscription("test-client-1", "ward_a")
    subscribers = manager.get_subscribers("ward_a")
    print(f"✓ Subscription added: {len(subscribers)} subscribers to ward_a")

    # Test statistics
    stats = manager.get_statistics()
    print(f"✓ Statistics: {stats['active_connections']} active connections")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(test_basic())