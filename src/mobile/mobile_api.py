"""
Mobile API Controller for Story 3.4
RESTful API endpoints optimized for mobile device interactions
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
import base64

from src.mobile.mobile_interface_service import (
    MobileInterfaceService, ScreenSize, DeviceOrientation, TouchGesture
)
from src.mobile.mobile_forms_service import (
    MobileFormsService, VitalSignsFormData, FormValidationResult
)
from src.mobile.offline_manager import (
    OfflineManager, SyncStatus, CacheStrategy, NetworkState
)
from src.mobile.device_integration import (
    DeviceIntegrationService, ScanType, VoiceCommand, PhotoType
)

logger = logging.getLogger(__name__)


class MobileAPIController:
    """Mobile API controller for Epic 3 Story 3.4"""

    def __init__(self):
        self.router = APIRouter(prefix="/api/mobile/v1", tags=["mobile"])
        self.mobile_interface = MobileInterfaceService()
        self.mobile_forms = MobileFormsService()
        self.offline_manager = OfflineManager()
        self.device_integration = DeviceIntegrationService()
        self._setup_routes()

    def _setup_routes(self):
        """Setup mobile API routes"""

        @self.router.get("/ward/{ward_id}/overview")
        async def get_mobile_ward_overview(
            ward_id: str,
            screen_size: str = Query("tablet", regex="^(smartphone|tablet|desktop)$"),
            orientation: str = Query("portrait", regex="^(portrait|landscape)$")
        ):
            """Get mobile-optimized ward overview"""
            try:
                screen_enum = ScreenSize(screen_size)
                orientation_enum = DeviceOrientation(orientation)

                overview = await self.mobile_interface.get_mobile_ward_overview(
                    ward_id, screen_enum, orientation_enum
                )

                return JSONResponse(
                    content=overview,
                    headers={
                        "Cache-Control": "max-age=30",
                        "X-Mobile-Optimized": "true"
                    }
                )

            except Exception as e:
                logger.error(f"Error in mobile ward overview: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/patient/{patient_id}/detail")
        async def get_mobile_patient_detail(
            patient_id: str,
            screen_size: str = Query("tablet", regex="^(smartphone|tablet|desktop)$")
        ):
            """Get mobile-optimized patient detail view"""
            try:
                screen_enum = ScreenSize(screen_size)
                detail = await self.mobile_interface.get_mobile_patient_detail(
                    patient_id, screen_enum
                )

                return JSONResponse(
                    content=detail,
                    headers={
                        "Cache-Control": "max-age=60",
                        "X-Mobile-Optimized": "true"
                    }
                )

            except Exception as e:
                logger.error(f"Error in mobile patient detail: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/gestures/handle")
        async def handle_touch_gesture(
            gesture_data: Dict[str, Any] = Body(...)
        ):
            """Handle touch gestures for mobile interface"""
            try:
                gesture_type = gesture_data.get("gesture")
                context = gesture_data.get("context", {})

                if not gesture_type:
                    raise HTTPException(status_code=400, detail="Gesture type required")

                gesture_enum = TouchGesture(gesture_type)
                result = await self.mobile_interface.handle_touch_gesture(
                    gesture_enum, context
                )

                return JSONResponse(content=result)

            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid gesture type")
            except Exception as e:
                logger.error(f"Error handling gesture: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/forms/vital-signs/config")
        async def get_vital_signs_form_config(
            screen_size: str = Query("tablet", regex="^(smartphone|tablet|desktop)$")
        ):
            """Get mobile-optimized vital signs form configuration"""
            try:
                config = self.mobile_forms.get_vital_signs_form_config(screen_size)
                return JSONResponse(content=config)

            except Exception as e:
                logger.error(f"Error getting form config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/forms/vital-signs/validate")
        async def validate_vital_signs_form(
            form_data: Dict[str, Any] = Body(...)
        ):
            """Validate vital signs form data"""
            try:
                # Convert dict to VitalSignsFormData
                vitals_data = VitalSignsFormData(
                    patient_id=form_data.get("patient_id", ""),
                    timestamp=datetime.fromisoformat(form_data.get("timestamp", datetime.now().isoformat())),
                    respiratory_rate=form_data.get("respiratory_rate"),
                    spo2=form_data.get("spo2"),
                    on_oxygen=form_data.get("on_oxygen", False),
                    temperature=form_data.get("temperature"),
                    systolic_bp=form_data.get("systolic_bp"),
                    heart_rate=form_data.get("heart_rate"),
                    consciousness=form_data.get("consciousness", "A"),
                    pain_score=form_data.get("pain_score"),
                    notes=form_data.get("notes", ""),
                    entered_by=form_data.get("entered_by", "")
                )

                validation_result = await self.mobile_forms.validate_form_data(vitals_data)

                return JSONResponse(content={
                    "is_valid": validation_result.is_valid,
                    "overall_score": validation_result.overall_score,
                    "completion_percentage": validation_result.completion_percentage,
                    "critical_errors": validation_result.critical_errors,
                    "warnings": validation_result.warnings,
                    "field_results": {
                        field_id: {
                            "is_valid": result.is_valid,
                            "severity": result.severity.value,
                            "message": result.message,
                            "suggestions": result.suggestions
                        }
                        for field_id, result in validation_result.field_results.items()
                    }
                })

            except Exception as e:
                logger.error(f"Error validating form data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/forms/vital-signs/save-draft")
        async def save_vital_signs_draft(
            form_data: Dict[str, Any] = Body(...)
        ):
            """Save vital signs form as draft"""
            try:
                patient_id = form_data.get("patient_id")
                if not patient_id:
                    raise HTTPException(status_code=400, detail="Patient ID required")

                vitals_data = VitalSignsFormData(
                    patient_id=patient_id,
                    timestamp=datetime.fromisoformat(form_data.get("timestamp", datetime.now().isoformat())),
                    respiratory_rate=form_data.get("respiratory_rate"),
                    spo2=form_data.get("spo2"),
                    on_oxygen=form_data.get("on_oxygen", False),
                    temperature=form_data.get("temperature"),
                    systolic_bp=form_data.get("systolic_bp"),
                    heart_rate=form_data.get("heart_rate"),
                    consciousness=form_data.get("consciousness", "A"),
                    entered_by=form_data.get("entered_by", "")
                )

                result = await self.mobile_forms.save_draft(patient_id, vitals_data)
                return JSONResponse(content=result)

            except Exception as e:
                logger.error(f"Error saving draft: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/forms/vital-signs/draft/{patient_id}")
        async def load_vital_signs_draft(patient_id: str):
            """Load vital signs draft for patient"""
            try:
                draft = await self.mobile_forms.load_draft(patient_id)

                if draft:
                    return JSONResponse(content=draft.to_dict())
                else:
                    return JSONResponse(content={"message": "No draft found"}, status_code=404)

            except Exception as e:
                logger.error(f"Error loading draft: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/offline/queue-action")
        async def queue_offline_action(
            action_data: Dict[str, Any] = Body(...)
        ):
            """Queue action for offline execution"""
            try:
                action_type = action_data.get("action_type")
                patient_id = action_data.get("patient_id")
                data = action_data.get("data", {})
                priority = action_data.get("priority", 3)

                if not action_type or not patient_id:
                    raise HTTPException(status_code=400, detail="Action type and patient ID required")

                action_id = await self.offline_manager.queue_offline_action(
                    action_type, patient_id, data, priority
                )

                return JSONResponse(content={
                    "success": True,
                    "action_id": action_id,
                    "queued_at": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error queuing offline action: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/offline/sync")
        async def sync_offline_actions():
            """Synchronize queued offline actions"""
            try:
                sync_result = await self.offline_manager.sync_offline_actions()

                return JSONResponse(content={
                    "success": sync_result.success,
                    "synced_actions": sync_result.synced_actions,
                    "failed_actions": sync_result.failed_actions,
                    "conflicts": sync_result.conflicts,
                    "sync_duration_ms": sync_result.sync_duration_ms,
                    "errors": sync_result.errors
                })

            except Exception as e:
                logger.error(f"Error syncing offline actions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/offline/status")
        async def get_offline_status():
            """Get offline capabilities and status"""
            try:
                capabilities = await self.offline_manager.get_offline_capabilities()
                return JSONResponse(content=capabilities)

            except Exception as e:
                logger.error(f"Error getting offline status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/device/scan/barcode")
        async def scan_barcode(
            scan_data: Dict[str, Any] = Body(...)
        ):
            """Scan barcode/QR code"""
            try:
                scan_type_str = scan_data.get("scan_type", "patient_id")
                image_data = scan_data.get("image_data")

                scan_type = ScanType(scan_type_str)
                result = await self.device_integration.barcode_scanner.scan_barcode(
                    scan_type, image_data
                )

                return JSONResponse(content=result.to_dict())

            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid scan type")
            except Exception as e:
                logger.error(f"Error scanning barcode: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/device/voice/process")
        async def process_voice_input(
            voice_data: Dict[str, Any] = Body(...)
        ):
            """Process voice input for commands or transcription"""
            try:
                # In real implementation, audio_data would be base64 encoded audio
                audio_data_b64 = voice_data.get("audio_data", "")
                audio_data = base64.b64decode(audio_data_b64) if audio_data_b64 else b""

                result = await self.device_integration.voice_input.process_voice_input(audio_data)
                return JSONResponse(content=result.to_dict())

            except Exception as e:
                logger.error(f"Error processing voice input: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/device/camera/capture")
        async def capture_clinical_photo(
            photo_data: Dict[str, Any] = Body(...)
        ):
            """Capture clinical photo"""
            try:
                photo_type_str = photo_data.get("photo_type")
                patient_id = photo_data.get("patient_id")
                captured_by = photo_data.get("captured_by")
                annotations = photo_data.get("annotations", [])

                if not all([photo_type_str, patient_id, captured_by]):
                    raise HTTPException(
                        status_code=400,
                        detail="Photo type, patient ID, and captured_by required"
                    )

                photo_type = PhotoType(photo_type_str)
                photo = await self.device_integration.camera_service.capture_clinical_photo(
                    photo_type, patient_id, captured_by, annotations
                )

                return JSONResponse(content=photo.to_dict())

            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid photo type")
            except Exception as e:
                logger.error(f"Error capturing photo: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/device/capabilities")
        async def get_device_capabilities():
            """Get available device integration capabilities"""
            try:
                capabilities = self.device_integration.get_device_capabilities()
                return JSONResponse(content=capabilities)

            except Exception as e:
                logger.error(f"Error getting device capabilities: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/network/state")
        async def update_network_state(
            network_data: Dict[str, Any] = Body(...)
        ):
            """Update network state for optimizations"""
            try:
                result = await self.offline_manager.handle_network_change(network_data)
                optimizations = await self.mobile_interface.optimize_for_network(
                    network_data.get("type", "wifi")
                )

                return JSONResponse(content={
                    "network_state": result["network_state"],
                    "optimizations": optimizations["optimizations"]
                })

            except Exception as e:
                logger.error(f"Error updating network state: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/metrics/performance")
        async def get_mobile_performance_metrics():
            """Get mobile interface performance metrics"""
            try:
                metrics = await self.mobile_interface.get_mobile_metrics()
                return JSONResponse(content=metrics)

            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/accessibility/config")
        async def get_accessibility_config():
            """Get accessibility configuration"""
            try:
                config = self.mobile_forms.get_form_accessibility_config()
                return JSONResponse(content=config)

            except Exception as e:
                logger.error(f"Error getting accessibility config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def get_router(self) -> APIRouter:
        """Get the configured FastAPI router"""
        return self.router


# Create global instance
mobile_api_controller = MobileAPIController()
mobile_router = mobile_api_controller.get_router()