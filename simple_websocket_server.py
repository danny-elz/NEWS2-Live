#!/usr/bin/env python3
"""
Simple WebSocket Server for Story 4.2 Demo
Basic WebSocket server to test frontend real-time connectivity
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

# Demo patient data
demo_patients = [
    {"id": "1", "name": "John Smith", "ward": "ICU", "bed": "A01", "currentNEWS2Score": 8, "riskLevel": "critical"},
    {"id": "2", "name": "Mary Johnson", "ward": "CCU", "bed": "B02", "currentNEWS2Score": 5, "riskLevel": "high"},
    {"id": "3", "name": "Robert Davis", "ward": "MEDICAL", "bed": "C03", "currentNEWS2Score": 3, "riskLevel": "medium"},
    {"id": "4", "name": "Sarah Wilson", "ward": "SURGICAL", "bed": "D04", "currentNEWS2Score": 1, "riskLevel": "low"},
]

async def handle_client(websocket, path):
    """Handle new WebSocket connection"""
    clients.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address}. Total clients: {len(clients)}")

    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection_status",
            "status": "connected",
            "message": "Connected to NEWS2 Live WebSocket server",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(welcome_msg))

        # Listen for client messages
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received message: {data}")

                # Handle subscription requests
                if data.get("type") == "subscribe":
                    response = {
                        "type": "subscription_confirmed",
                        "ward_ids": data.get("ward_ids", ["all"]),
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(response))
                    logger.info(f"Client subscribed to wards: {data.get('ward_ids', ['all'])}")

                # Handle heartbeat
                elif data.get("type") == "heartbeat":
                    response = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(response))

            except json.JSONDecodeError:
                error_msg = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_msg))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(clients)}")

async def broadcast_patient_updates():
    """Periodically broadcast patient updates to all connected clients"""
    while True:
        try:
            await asyncio.sleep(8)  # Send updates every 8 seconds

            if clients:
                # Pick a random patient to update
                patient = random.choice(demo_patients)

                # Simulate score changes
                score_change = random.choice([-1, 0, 0, 1])
                new_score = max(0, min(15, patient["currentNEWS2Score"] + score_change))

                # Update risk level
                if new_score >= 7:
                    risk_level = "critical"
                elif new_score >= 5:
                    risk_level = "high"
                elif new_score >= 3:
                    risk_level = "medium"
                else:
                    risk_level = "low"

                # Create patient update message
                patient_update = {
                    "type": "patient_update",
                    "payload": {
                        "data": {
                            **patient,
                            "currentNEWS2Score": new_score,
                            "riskLevel": risk_level,
                            "lastUpdated": datetime.now().isoformat(),
                            "status": "stable" if new_score < 5 else "monitoring" if new_score < 7 else "critical",
                            "alertCount": random.randint(0, 2) if new_score >= 5 else 0,
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                }

                # Broadcast to all clients
                if clients:
                    await asyncio.gather(
                        *[client.send(json.dumps(patient_update)) for client in clients.copy()],
                        return_exceptions=True
                    )
                    logger.info(f"Broadcasted update for {patient['name']} - NEWS2: {new_score}")

                    # Update the demo data
                    patient["currentNEWS2Score"] = new_score
                    patient["riskLevel"] = risk_level

        except Exception as e:
            logger.error(f"Error in broadcast: {e}")

async def main():
    """Start the WebSocket server"""
    logger.info("Starting Simple WebSocket Server on ws://localhost:8765")

    # Start the server
    start_server = websockets.serve(handle_client, "localhost", 8765)

    # Start background task for patient updates
    update_task = asyncio.create_task(broadcast_patient_updates())

    # Run the server
    await asyncio.gather(start_server, update_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")