"""
WebSocket Server for Story 3.2
Real-time data streaming infrastructure for clinical dashboards
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Set, Any, Optional, List
import websockets
from websockets.server import WebSocketServerProtocol
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    HEARTBEAT = "heartbeat"
    PATIENT_UPDATE = "patient_update"
    ALERT_UPDATE = "alert_update"
    CONNECTION_STATUS = "connection_status"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ERROR = "error"
    BATCH_UPDATE = "batch_update"


class ConnectionStatus(Enum):
    """Connection status states"""
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"


class WebSocketServer:
    """
    WebSocket server for real-time clinical dashboard updates
    Handles patient data streaming and alert notifications
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of ward_ids
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.heartbeat_interval = 30  # seconds
        self.server = None
        self._running = False
        self._message_counter = 0
        self._connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0
        }

    async def start(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                max_queue=100,
                compression=None  # Can enable compression if needed
            )
            self._running = True

            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._message_processor())

            logger.info(f"WebSocket server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self):
        """Stop the WebSocket server"""
        self._running = False

        # Close all connections
        for client_id, websocket in list(self.connections.items()):
            await self.disconnect_client(client_id)

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("WebSocket server stopped")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection"""
        client_id = str(uuid.uuid4())
        self.connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self._connection_stats["total_connections"] += 1
        self._connection_stats["active_connections"] += 1

        try:
            # Send initial connection status
            await self.send_message(client_id, {
                "type": MessageType.CONNECTION_STATUS.value,
                "status": ConnectionStatus.CONNECTED.value,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"Client {client_id} connected from {websocket.remote_address}")

            # Handle messages from client
            async for message in websocket:
                await self.handle_message(client_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self.disconnect_client(client_id)

    async def handle_message(self, client_id: str, message: str):
        """Process message from client"""
        try:
            data = json.loads(message)
            message_type = MessageType(data.get("type"))

            self._connection_stats["messages_received"] += 1

            if message_type == MessageType.SUBSCRIBE:
                await self.handle_subscribe(client_id, data)
            elif message_type == MessageType.UNSUBSCRIBE:
                await self.handle_unsubscribe(client_id, data)
            elif message_type == MessageType.HEARTBEAT:
                await self.handle_heartbeat(client_id)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")

        except json.JSONDecodeError:
            await self.send_error(client_id, "Invalid JSON message")
        except ValueError as e:
            await self.send_error(client_id, f"Invalid message type: {e}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_error(client_id, "Internal server error")

    async def handle_subscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        ward_ids = data.get("ward_ids", [])
        if not isinstance(ward_ids, list):
            ward_ids = [ward_ids]

        for ward_id in ward_ids:
            self.subscriptions[client_id].add(ward_id)

        logger.info(f"Client {client_id} subscribed to wards: {ward_ids}")

        # Send confirmation
        await self.send_message(client_id, {
            "type": "subscription_confirmed",
            "ward_ids": list(self.subscriptions[client_id]),
            "timestamp": datetime.now().isoformat()
        })

    async def handle_unsubscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        ward_ids = data.get("ward_ids", [])
        if not isinstance(ward_ids, list):
            ward_ids = [ward_ids]

        for ward_id in ward_ids:
            self.subscriptions[client_id].discard(ward_id)

        logger.info(f"Client {client_id} unsubscribed from wards: {ward_ids}")

    async def handle_heartbeat(self, client_id: str):
        """Handle heartbeat message"""
        await self.send_message(client_id, {
            "type": MessageType.HEARTBEAT.value,
            "timestamp": datetime.now().isoformat()
        })

    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client"""
        if client_id not in self.connections:
            return False

        websocket = self.connections[client_id]
        try:
            await websocket.send(json.dumps(message))
            self._connection_stats["messages_sent"] += 1
            return True
        except websockets.exceptions.ConnectionClosed:
            await self.disconnect_client(client_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            return False

    async def broadcast_to_ward(self, ward_id: str, message: Dict[str, Any]):
        """Broadcast message to all clients subscribed to a ward"""
        clients_to_notify = [
            client_id for client_id, wards in self.subscriptions.items()
            if ward_id in wards
        ]

        tasks = [
            self.send_message(client_id, message)
            for client_id in clients_to_notify
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)

        logger.debug(f"Broadcast to ward {ward_id}: {successful}/{len(clients_to_notify)} clients")

    async def broadcast_patient_update(self, patient_data: Dict[str, Any]):
        """Broadcast patient update to relevant subscribers"""
        ward_id = patient_data.get("ward_id")
        if not ward_id:
            return

        message = {
            "type": MessageType.PATIENT_UPDATE.value,
            "data": patient_data,
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_ward(ward_id, message)

    async def broadcast_alert_update(self, alert_data: Dict[str, Any]):
        """Broadcast alert update to relevant subscribers"""
        ward_id = alert_data.get("ward_id")
        if not ward_id:
            return

        message = {
            "type": MessageType.ALERT_UPDATE.value,
            "data": alert_data,
            "priority": alert_data.get("priority", "normal"),
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_ward(ward_id, message)

    async def send_batch_update(self, ward_id: str, updates: List[Dict[str, Any]]):
        """Send batch update to prevent UI flooding"""
        if not updates:
            return

        message = {
            "type": MessageType.BATCH_UPDATE.value,
            "updates": updates,
            "count": len(updates),
            "timestamp": datetime.now().isoformat()
        }

        await self.broadcast_to_ward(ward_id, message)

    async def send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self.send_message(client_id, {
            "type": MessageType.ERROR.value,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })

    async def disconnect_client(self, client_id: str):
        """Disconnect and cleanup client"""
        if client_id in self.connections:
            websocket = self.connections[client_id]
            try:
                await websocket.close()
            except Exception:
                pass

            del self.connections[client_id]
            del self.subscriptions[client_id]
            self._connection_stats["active_connections"] -= 1

            logger.info(f"Client {client_id} disconnected and cleaned up")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all connected clients"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                heartbeat_message = {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat()
                }

                # Send heartbeat to all connected clients
                disconnected = []
                for client_id in list(self.connections.keys()):
                    success = await self.send_message(client_id, heartbeat_message)
                    if not success:
                        disconnected.append(client_id)

                # Clean up disconnected clients
                for client_id in disconnected:
                    await self.disconnect_client(client_id)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _message_processor(self):
        """Process queued messages for batch sending"""
        batch_interval = 0.5  # seconds
        batch_size = 10

        while self._running:
            try:
                batches: Dict[str, List[Dict[str, Any]]] = {}
                deadline = asyncio.get_event_loop().time() + batch_interval

                # Collect messages for batching
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        timeout = deadline - asyncio.get_event_loop().time()
                        if timeout <= 0:
                            break

                        message = await asyncio.wait_for(
                            self.message_queue.get(),
                            timeout=timeout
                        )

                        ward_id = message.get("ward_id")
                        if ward_id:
                            if ward_id not in batches:
                                batches[ward_id] = []
                            batches[ward_id].append(message)

                            # Send immediately if batch is full
                            if len(batches[ward_id]) >= batch_size:
                                await self.send_batch_update(ward_id, batches[ward_id])
                                batches[ward_id] = []

                    except asyncio.TimeoutError:
                        break

                # Send remaining batches
                for ward_id, messages in batches.items():
                    if messages:
                        if len(messages) == 1:
                            # Send single message directly
                            await self.broadcast_to_ward(ward_id, messages[0])
                        else:
                            # Send as batch
                            await self.send_batch_update(ward_id, messages)

            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            **self._connection_stats,
            "subscriptions": {
                client_id: list(wards)
                for client_id, wards in self.subscriptions.items()
            }
        }

    def get_active_connections(self) -> int:
        """Get number of active connections"""
        return len(self.connections)

    async def queue_message(self, message: Dict[str, Any]):
        """Queue message for batch processing"""
        await self.message_queue.put(message)