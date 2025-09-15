#!/usr/bin/env python3
"""
Simple WebSocket Server for NEWS2 Frontend
"""

import asyncio
import websockets
import json
import logging
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store connected clients
clients = set()

async def handle_client(websocket):
    """Handle new WebSocket connection"""
    clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(clients)}")

    try:
        # Send initial connection message
        welcome_msg = {
            "type": "system_status",
            "data": {
                "status": "healthy",
                "message": "Connected to NEWS2 Live WebSocket server"
            }
        }
        await websocket.send(json.dumps(welcome_msg))

        # Keep connection alive and handle messages
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received: {data}")

                # Echo back for testing
                response = {
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(response))

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        clients.remove(websocket)
        logger.info(f"Remaining clients: {len(clients)}")

async def broadcast_updates():
    """Send periodic updates to all connected clients"""
    patient_ids = ["P001", "P002", "P003", "P004", "P005"]

    while True:
        await asyncio.sleep(5)  # Send updates every 5 seconds

        if clients:
            # Generate random vital signs update
            patient_id = random.choice(patient_ids)

            vitals_update = {
                "type": "vitals_update",
                "data": {
                    "patient_id": patient_id,
                    "ts": datetime.now().isoformat(),
                    "resp_rate": random.randint(12, 22),
                    "spo2": random.randint(92, 100),
                    "o2_supplemental": random.choice([True, False]),
                    "temp_c": round(random.uniform(36.0, 38.5), 1),
                    "sbp": random.randint(100, 140),
                    "hr": random.randint(60, 100),
                    "avpu": random.choice(["A", "V", "P", "U"]),
                    "source_seq": random.randint(1, 1000)
                }
            }

            # Generate NEWS2 score update
            news2_update = {
                "type": "news2_update",
                "data": {
                    "patient_id": patient_id,
                    "ts": datetime.now().isoformat(),
                    "total": random.randint(0, 12),
                    "component_scores": {
                        "resp": random.randint(0, 3),
                        "spo2": random.randint(0, 3),
                        "o2": random.randint(0, 2),
                        "temp": random.randint(0, 3),
                        "sbp": random.randint(0, 3),
                        "hr": random.randint(0, 3),
                        "avpu": random.randint(0, 3)
                    },
                    "hard_flag": random.choice([True, False]),
                    "single_param_eq3": random.choice([True, False])
                }
            }

            # Occasionally send alerts
            if random.random() > 0.7:
                alert_update = {
                    "type": "alert_created",
                    "data": {
                        "alert_id": f"ALT{random.randint(1000, 9999)}",
                        "patient_id": patient_id,
                        "ts": datetime.now().isoformat(),
                        "news2": random.randint(5, 12),
                        "reasons": ["High NEWS2 score", "Vital signs deterioration"],
                        "acknowledged": False
                    }
                }

                # Broadcast alert
                disconnected = set()
                for client in clients:
                    try:
                        await client.send(json.dumps(alert_update))
                    except:
                        disconnected.add(client)

                clients.difference_update(disconnected)

            # Broadcast vitals and NEWS2 updates
            disconnected = set()
            for client in clients:
                try:
                    await client.send(json.dumps(vitals_update))
                    await client.send(json.dumps(news2_update))
                except:
                    disconnected.add(client)

            # Remove disconnected clients
            clients.difference_update(disconnected)

            if len(clients) > 0:
                logger.info(f"Sent updates for patient {patient_id} to {len(clients)} clients")

async def main():
    """Start the WebSocket server"""
    logger.info("Starting WebSocket Server on ws://localhost:8765")

    # Create server
    async with websockets.serve(handle_client, "localhost", 8765):
        # Start broadcast task
        await broadcast_updates()

if __name__ == "__main__":
    asyncio.run(main())