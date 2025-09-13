"""
Device Integration Service for Story 3.4
Integration with mobile device features: camera, voice input, barcode scanning
"""

import asyncio
import logging
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of barcode/QR code scanning"""
    PATIENT_ID = "patient_id"
    MEDICATION = "medication"
    EQUIPMENT = "equipment"
    BED_LOCATION = "bed_location"


class VoiceCommand(Enum):
    """Supported voice commands"""
    RECORD_VITALS = "record_vitals"
    ESCALATE_CARE = "escalate_care"
    SEARCH_PATIENT = "search_patient"
    NAVIGATE_TO = "navigate_to"
    SET_REMINDER = "set_reminder"


class PhotoType(Enum):
    """Types of clinical photography"""
    WOUND_CARE = "wound_care"
    MEDICATION_VERIFICATION = "medication_verification"
    EQUIPMENT_STATUS = "equipment_status"
    PATIENT_POSITION = "patient_position"
    DOCUMENTATION = "documentation"


@dataclass
class ScanResult:
    """Result of barcode/QR code scanning"""
    scan_type: ScanType
    data: str
    confidence: float
    timestamp: datetime
    location: Optional[Dict[str, float]] = None  # GPS coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_type": self.scan_type.value,
            "data": self.data,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "metadata": self.metadata
        }


@dataclass
class VoiceRecognitionResult:
    """Result of voice recognition processing"""
    command: Optional[VoiceCommand]
    transcribed_text: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime
    processing_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command.value if self.command else None,
            "transcribed_text": self.transcribed_text,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class PhotoCapture:
    """Clinical photo capture data"""
    photo_id: str
    photo_type: PhotoType
    patient_id: str
    image_data: str  # Base64 encoded
    metadata: Dict[str, Any]
    timestamp: datetime
    captured_by: str
    location: Optional[str] = None
    annotations: List[str] = field(default_factory=list)

    def get_file_size_mb(self) -> float:
        """Calculate approximate file size in MB"""
        return len(self.image_data) * 0.75 / (1024 * 1024)  # Base64 overhead

    def to_dict(self) -> Dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "photo_type": self.photo_type.value,
            "patient_id": self.patient_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "captured_by": self.captured_by,
            "location": self.location,
            "annotations": self.annotations,
            "file_size_mb": self.get_file_size_mb()
        }


class BarcodeScanner:
    """Barcode and QR code scanning functionality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._scan_history: List[ScanResult] = []

    async def scan_barcode(self, scan_type: ScanType,
                          image_data: Optional[str] = None) -> ScanResult:
        """Scan barcode/QR code from camera or provided image"""
        try:
            start_time = datetime.now()

            # Simulate scanning process
            await asyncio.sleep(0.5)  # Processing delay

            # Simulate different scan results based on type
            scan_data = await self._process_scan(scan_type)

            result = ScanResult(
                scan_type=scan_type,
                data=scan_data["data"],
                confidence=scan_data["confidence"],
                timestamp=start_time,
                metadata=scan_data.get("metadata", {})
            )

            self._scan_history.append(result)
            self.logger.info(f"Scanned {scan_type.value}: {result.data}")

            return result

        except Exception as e:
            self.logger.error(f"Error scanning barcode: {e}")
            raise

    async def _process_scan(self, scan_type: ScanType) -> Dict[str, Any]:
        """Process scan based on type (simulated)"""
        if scan_type == ScanType.PATIENT_ID:
            return {
                "data": "P002",
                "confidence": 0.95,
                "metadata": {
                    "format": "CODE128",
                    "ward": "A",
                    "bed": "A02"
                }
            }
        elif scan_type == ScanType.MEDICATION:
            return {
                "data": "MED123456",
                "confidence": 0.88,
                "metadata": {
                    "format": "DataMatrix",
                    "medication_name": "Acetaminophen",
                    "dosage": "500mg",
                    "lot_number": "LOT789"
                }
            }
        elif scan_type == ScanType.BED_LOCATION:
            return {
                "data": "BED_A02_ROOM_101",
                "confidence": 0.92,
                "metadata": {
                    "format": "QR",
                    "ward": "A",
                    "room": "101",
                    "bed": "A02"
                }
            }
        elif scan_type == ScanType.EQUIPMENT:
            return {
                "data": "EQP_MONITOR_12345",
                "confidence": 0.90,
                "metadata": {
                    "format": "QR",
                    "equipment_type": "Vital Signs Monitor",
                    "serial": "VSM12345",
                    "last_calibration": "2024-01-15"
                }
            }
        else:
            return {
                "data": "UNKNOWN_BARCODE",
                "confidence": 0.5,
                "metadata": {}
            }

    def get_scan_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scan history"""
        return [scan.to_dict() for scan in self._scan_history[-limit:]]


class VoiceInput:
    """Voice input and command recognition"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._command_patterns = self._initialize_command_patterns()

    def _initialize_command_patterns(self) -> Dict[VoiceCommand, List[str]]:
        """Initialize voice command patterns"""
        return {
            VoiceCommand.RECORD_VITALS: [
                "record vitals", "take vitals", "vital signs", "record vital signs"
            ],
            VoiceCommand.ESCALATE_CARE: [
                "escalate", "urgent", "emergency", "rapid response", "call doctor"
            ],
            VoiceCommand.SEARCH_PATIENT: [
                "find patient", "search patient", "locate patient", "patient lookup"
            ],
            VoiceCommand.NAVIGATE_TO: [
                "go to", "navigate to", "show me", "open"
            ],
            VoiceCommand.SET_REMINDER: [
                "remind me", "set reminder", "schedule", "alert me"
            ]
        }

    async def process_voice_input(self, audio_data: bytes) -> VoiceRecognitionResult:
        """Process voice input and extract commands/text"""
        try:
            start_time = datetime.now()

            # Simulate voice processing
            await asyncio.sleep(1.0)  # Processing delay

            # Simulate speech-to-text result
            transcribed_text = await self._simulate_transcription(audio_data)

            # Detect commands
            command, parameters = self._detect_command(transcribed_text)

            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            result = VoiceRecognitionResult(
                command=command,
                transcribed_text=transcribed_text,
                confidence=0.85,
                parameters=parameters,
                timestamp=start_time,
                processing_time_ms=processing_time
            )

            self.logger.info(f"Voice input processed: {transcribed_text}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing voice input: {e}")
            raise

    async def _simulate_transcription(self, audio_data: bytes) -> str:
        """Simulate speech-to-text transcription"""
        # Simulate different voice inputs
        sample_transcriptions = [
            "record vitals for patient P002",
            "respiratory rate eighteen",
            "SpO2 ninety five percent",
            "temperature thirty seven point two degrees",
            "escalate care for bed A02",
            "find patient Smith",
            "remind me to check on patient in thirty minutes"
        ]

        import random
        return random.choice(sample_transcriptions)

    def _detect_command(self, text: str) -> Tuple[Optional[VoiceCommand], Dict[str, Any]]:
        """Detect voice commands from transcribed text"""
        text_lower = text.lower()
        parameters = {}

        for command, patterns in self._command_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                # Extract parameters based on command type
                if command == VoiceCommand.RECORD_VITALS:
                    parameters = self._extract_vitals_parameters(text_lower)
                elif command == VoiceCommand.SEARCH_PATIENT:
                    parameters = self._extract_patient_parameters(text_lower)
                elif command == VoiceCommand.NAVIGATE_TO:
                    parameters = self._extract_navigation_parameters(text_lower)
                elif command == VoiceCommand.SET_REMINDER:
                    parameters = self._extract_reminder_parameters(text_lower)

                return command, parameters

        return None, {}

    def _extract_vitals_parameters(self, text: str) -> Dict[str, Any]:
        """Extract vital signs parameters from voice text"""
        parameters = {}

        # Extract patient ID
        import re
        patient_match = re.search(r'patient ([a-z0-9]+)', text)
        if patient_match:
            parameters["patient_id"] = patient_match.group(1).upper()

        # Extract vital values
        if "respiratory rate" in text or "respiration" in text:
            rate_match = re.search(r'(\d+)', text)
            if rate_match:
                parameters["respiratory_rate"] = int(rate_match.group(1))

        if "spo2" in text or "oxygen saturation" in text:
            spo2_match = re.search(r'(\d+)\s*percent', text)
            if spo2_match:
                parameters["spo2"] = int(spo2_match.group(1))

        if "temperature" in text:
            temp_match = re.search(r'(\d+(?:\.\d+)?)', text)
            if temp_match:
                parameters["temperature"] = float(temp_match.group(1))

        return parameters

    def _extract_patient_parameters(self, text: str) -> Dict[str, Any]:
        """Extract patient search parameters"""
        parameters = {}

        import re
        # Extract patient ID
        patient_match = re.search(r'patient ([a-z0-9]+)', text)
        if patient_match:
            parameters["patient_id"] = patient_match.group(1).upper()

        # Extract patient name
        name_match = re.search(r'patient (\w+)', text)
        if name_match and not patient_match:
            parameters["patient_name"] = name_match.group(1)

        return parameters

    def _extract_navigation_parameters(self, text: str) -> Dict[str, Any]:
        """Extract navigation parameters"""
        parameters = {}

        if "bed" in text:
            import re
            bed_match = re.search(r'bed ([a-z0-9]+)', text)
            if bed_match:
                parameters["target"] = f"bed_{bed_match.group(1)}"

        if "ward" in text:
            ward_match = re.search(r'ward ([a-z0-9]+)', text)
            if ward_match:
                parameters["target"] = f"ward_{ward_match.group(1)}"

        return parameters

    def _extract_reminder_parameters(self, text: str) -> Dict[str, Any]:
        """Extract reminder parameters"""
        parameters = {}

        import re
        # Extract time
        time_match = re.search(r'in (\d+) (minutes?|hours?)', text)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            if "hour" in unit:
                parameters["delay_minutes"] = value * 60
            else:
                parameters["delay_minutes"] = value

        return parameters


class CameraService:
    """Clinical photography and documentation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._photo_storage: List[PhotoCapture] = []

    async def capture_clinical_photo(self, photo_type: PhotoType, patient_id: str,
                                   captured_by: str, annotations: List[str] = None) -> PhotoCapture:
        """Capture clinical photo with metadata"""
        try:
            # Generate photo ID
            photo_id = self._generate_photo_id(patient_id, photo_type)

            # Simulate photo capture
            image_data = await self._simulate_photo_capture()

            # Create photo capture record
            photo = PhotoCapture(
                photo_id=photo_id,
                photo_type=photo_type,
                patient_id=patient_id,
                image_data=image_data,
                metadata=self._generate_photo_metadata(),
                timestamp=datetime.now(),
                captured_by=captured_by,
                annotations=annotations or []
            )

            self._photo_storage.append(photo)
            self.logger.info(f"Captured {photo_type.value} photo for patient {patient_id}")

            return photo

        except Exception as e:
            self.logger.error(f"Error capturing photo: {e}")
            raise

    async def _simulate_photo_capture(self) -> str:
        """Simulate photo capture returning base64 encoded image"""
        # Create a small sample base64 image data
        sample_image_data = """iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="""
        return sample_image_data

    def _generate_photo_id(self, patient_id: str, photo_type: PhotoType) -> str:
        """Generate unique photo ID"""
        timestamp = datetime.now().isoformat()
        content = f"{patient_id}_{photo_type.value}_{timestamp}"
        return f"PHOTO_{hashlib.md5(content.encode()).hexdigest()[:8].upper()}"

    def _generate_photo_metadata(self) -> Dict[str, Any]:
        """Generate photo metadata"""
        return {
            "resolution": "1920x1080",
            "file_format": "JPEG",
            "compression_quality": 85,
            "flash_used": False,
            "camera_settings": {
                "iso": 100,
                "aperture": "f/2.8",
                "shutter_speed": "1/60"
            },
            "device_info": {
                "model": "Healthcare Tablet Pro",
                "os": "iOS 16.0",
                "app_version": "2.1.0"
            }
        }

    def get_patient_photos(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all photos for a patient"""
        patient_photos = [photo for photo in self._photo_storage
                         if photo.patient_id == patient_id]
        return [photo.to_dict() for photo in patient_photos]

    async def annotate_photo(self, photo_id: str, annotation: str) -> bool:
        """Add annotation to existing photo"""
        try:
            for photo in self._photo_storage:
                if photo.photo_id == photo_id:
                    photo.annotations.append(annotation)
                    self.logger.info(f"Added annotation to photo {photo_id}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error annotating photo: {e}")
            return False


class DeviceIntegrationService:
    """Main service for device integration functionality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.barcode_scanner = BarcodeScanner()
        self.voice_input = VoiceInput()
        self.camera_service = CameraService()

    async def scan_patient_id(self) -> Dict[str, Any]:
        """Quick scan for patient identification"""
        try:
            scan_result = await self.barcode_scanner.scan_barcode(ScanType.PATIENT_ID)

            return {
                "success": True,
                "patient_id": scan_result.data,
                "confidence": scan_result.confidence,
                "metadata": scan_result.metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def voice_vital_entry(self, audio_data: bytes) -> Dict[str, Any]:
        """Process voice input for vital signs entry"""
        try:
            voice_result = await self.voice_input.process_voice_input(audio_data)

            if voice_result.command == VoiceCommand.RECORD_VITALS:
                return {
                    "success": True,
                    "command": "record_vitals",
                    "parameters": voice_result.parameters,
                    "transcription": voice_result.transcribed_text,
                    "confidence": voice_result.confidence
                }
            else:
                return {
                    "success": True,
                    "command": voice_result.command.value if voice_result.command else None,
                    "transcription": voice_result.transcribed_text,
                    "message": "Command recognized but not for vital signs entry"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def document_wound_care(self, patient_id: str, captured_by: str,
                                notes: List[str] = None) -> Dict[str, Any]:
        """Capture and document wound care photos"""
        try:
            photo = await self.camera_service.capture_clinical_photo(
                PhotoType.WOUND_CARE,
                patient_id,
                captured_by,
                notes or []
            )

            return {
                "success": True,
                "photo_id": photo.photo_id,
                "file_size_mb": photo.get_file_size_mb(),
                "timestamp": photo.timestamp.isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_device_capabilities(self) -> Dict[str, Any]:
        """Get available device integration capabilities"""
        return {
            "barcode_scanning": {
                "available": True,
                "supported_formats": ["CODE128", "QR", "DataMatrix"],
                "supported_types": [scan_type.value for scan_type in ScanType]
            },
            "voice_input": {
                "available": True,
                "supported_commands": [cmd.value for cmd in VoiceCommand],
                "languages": ["en-US", "en-GB"]
            },
            "camera": {
                "available": True,
                "supported_types": [photo_type.value for photo_type in PhotoType],
                "max_resolution": "1920x1080",
                "formats": ["JPEG", "PNG"]
            },
            "location_services": {
                "available": False,  # Simulated as not available
                "gps_accuracy": "N/A"
            }
        }