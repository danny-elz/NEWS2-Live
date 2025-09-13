"""
Unit tests for Device Integration Service (Story 3.4)
Tests mobile device features: camera, voice, barcode scanning
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.mobile.device_integration import (
    DeviceIntegrationService,
    BarcodeScanner,
    VoiceInput,
    CameraService,
    ScanType,
    VoiceCommand,
    PhotoType,
    ScanResult,
    VoiceRecognitionResult,
    PhotoCapture
)


class TestBarcodeScanner:
    """Test suite for BarcodeScanner"""

    @pytest.fixture
    def barcode_scanner(self):
        """Create BarcodeScanner instance"""
        return BarcodeScanner()

    @pytest.mark.asyncio
    async def test_scan_patient_id(self, barcode_scanner):
        """Test scanning patient ID barcode"""
        result = await barcode_scanner.scan_barcode(ScanType.PATIENT_ID)

        assert isinstance(result, ScanResult)
        assert result.scan_type == ScanType.PATIENT_ID
        assert result.data.startswith("P")  # Patient ID format
        assert result.confidence > 0.8
        assert "format" in result.metadata

    @pytest.mark.asyncio
    async def test_scan_medication(self, barcode_scanner):
        """Test scanning medication barcode"""
        result = await barcode_scanner.scan_barcode(ScanType.MEDICATION)

        assert result.scan_type == ScanType.MEDICATION
        assert result.data.startswith("MED")
        assert result.confidence > 0.8
        assert "medication_name" in result.metadata
        assert "dosage" in result.metadata
        assert "lot_number" in result.metadata

    @pytest.mark.asyncio
    async def test_scan_bed_location(self, barcode_scanner):
        """Test scanning bed location QR code"""
        result = await barcode_scanner.scan_barcode(ScanType.BED_LOCATION)

        assert result.scan_type == ScanType.BED_LOCATION
        assert "BED_" in result.data
        assert result.confidence > 0.8
        assert result.metadata["format"] == "QR"
        assert "ward" in result.metadata
        assert "bed" in result.metadata

    @pytest.mark.asyncio
    async def test_scan_equipment(self, barcode_scanner):
        """Test scanning equipment QR code"""
        result = await barcode_scanner.scan_barcode(ScanType.EQUIPMENT)

        assert result.scan_type == ScanType.EQUIPMENT
        assert "EQP_" in result.data
        assert "equipment_type" in result.metadata
        assert "serial" in result.metadata

    def test_scan_history(self, barcode_scanner):
        """Test scan history tracking"""
        # Initially empty
        history = barcode_scanner.get_scan_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_scan_history_tracking(self, barcode_scanner):
        """Test that scans are added to history"""
        # Perform multiple scans
        await barcode_scanner.scan_barcode(ScanType.PATIENT_ID)
        await barcode_scanner.scan_barcode(ScanType.MEDICATION)
        await barcode_scanner.scan_barcode(ScanType.BED_LOCATION)

        history = barcode_scanner.get_scan_history()
        assert len(history) == 3

        # Check history structure
        scan_entry = history[0]
        assert "scan_type" in scan_entry
        assert "data" in scan_entry
        assert "confidence" in scan_entry
        assert "timestamp" in scan_entry

    def test_scan_result_to_dict(self):
        """Test ScanResult dictionary conversion"""
        now = datetime.now()
        result = ScanResult(
            scan_type=ScanType.PATIENT_ID,
            data="P001",
            confidence=0.95,
            timestamp=now,
            location={"lat": 40.7128, "lng": -74.0060},
            metadata={"ward": "A", "bed": "A01"}
        )

        result_dict = result.to_dict()

        assert result_dict["scan_type"] == "patient_id"
        assert result_dict["data"] == "P001"
        assert result_dict["confidence"] == 0.95
        assert result_dict["timestamp"] == now.isoformat()
        assert result_dict["location"] == {"lat": 40.7128, "lng": -74.0060}
        assert result_dict["metadata"] == {"ward": "A", "bed": "A01"}


class TestVoiceInput:
    """Test suite for VoiceInput"""

    @pytest.fixture
    def voice_input(self):
        """Create VoiceInput instance"""
        return VoiceInput()

    @pytest.mark.asyncio
    async def test_process_voice_input_basic(self, voice_input):
        """Test basic voice input processing"""
        result = await voice_input.process_voice_input(b"dummy_audio_data")

        assert isinstance(result, VoiceRecognitionResult)
        assert result.transcribed_text != ""
        assert 0 <= result.confidence <= 1
        assert result.processing_time_ms > 0
        assert isinstance(result.parameters, dict)

    @pytest.mark.asyncio
    async def test_detect_record_vitals_command(self, voice_input):
        """Test detection of record vitals command"""
        # Mock transcription that should trigger record vitals
        with patch.object(voice_input, '_simulate_transcription',
                         return_value="record vitals for patient P002"):
            result = await voice_input.process_voice_input(b"audio")

            assert result.command == VoiceCommand.RECORD_VITALS
            assert result.transcribed_text == "record vitals for patient P002"
            assert "patient_id" in result.parameters
            assert result.parameters["patient_id"] == "P002"

    @pytest.mark.asyncio
    async def test_detect_escalate_care_command(self, voice_input):
        """Test detection of escalate care command"""
        with patch.object(voice_input, '_simulate_transcription',
                         return_value="escalate care for bed A02"):
            result = await voice_input.process_voice_input(b"audio")

            assert result.command == VoiceCommand.ESCALATE_CARE
            assert "escalate" in result.transcribed_text.lower()

    @pytest.mark.asyncio
    async def test_detect_search_patient_command(self, voice_input):
        """Test detection of search patient command"""
        with patch.object(voice_input, '_simulate_transcription',
                         return_value="find patient Smith"):
            result = await voice_input.process_voice_input(b"audio")

            assert result.command == VoiceCommand.SEARCH_PATIENT
            assert "patient_name" in result.parameters
            assert result.parameters["patient_name"] == "Smith"

    @pytest.mark.asyncio
    async def test_detect_navigation_command(self, voice_input):
        """Test detection of navigation command"""
        with patch.object(voice_input, '_simulate_transcription',
                         return_value="go to bed A05"):
            result = await voice_input.process_voice_input(b"audio")

            assert result.command == VoiceCommand.NAVIGATE_TO
            assert "target" in result.parameters
            assert "bed_a05" in result.parameters["target"]

    @pytest.mark.asyncio
    async def test_detect_reminder_command(self, voice_input):
        """Test detection of reminder command"""
        with patch.object(voice_input, '_simulate_transcription',
                         return_value="remind me to check on patient in 30 minutes"):
            result = await voice_input.process_voice_input(b"audio")

            assert result.command == VoiceCommand.SET_REMINDER
            assert "delay_minutes" in result.parameters
            assert result.parameters["delay_minutes"] == 30

    def test_extract_vitals_parameters(self, voice_input):
        """Test extraction of vitals parameters from speech"""
        # Test respiratory rate extraction
        params = voice_input._extract_vitals_parameters("patient p001 respiratory rate 18")
        assert params["patient_id"] == "P001"
        assert params["respiratory_rate"] == 18

        # Test SpO2 extraction
        params = voice_input._extract_vitals_parameters("spo2 95 percent")
        assert params["spo2"] == 95

        # Test temperature extraction
        params = voice_input._extract_vitals_parameters("temperature 37.5 degrees")
        assert params["temperature"] == 37.5

    def test_extract_patient_parameters(self, voice_input):
        """Test extraction of patient search parameters"""
        # Test patient ID extraction
        params = voice_input._extract_patient_parameters("patient p002 status")
        assert params["patient_id"] == "P002"

        # Test patient name extraction
        params = voice_input._extract_patient_parameters("find patient johnson")
        assert params["patient_name"] == "johnson"

    def test_extract_reminder_parameters(self, voice_input):
        """Test extraction of reminder parameters"""
        # Test minutes
        params = voice_input._extract_reminder_parameters("remind me in 15 minutes")
        assert params["delay_minutes"] == 15

        # Test hours
        params = voice_input._extract_reminder_parameters("remind me in 2 hours")
        assert params["delay_minutes"] == 120

    def test_voice_result_to_dict(self):
        """Test VoiceRecognitionResult dictionary conversion"""
        now = datetime.now()
        result = VoiceRecognitionResult(
            command=VoiceCommand.RECORD_VITALS,
            transcribed_text="record vitals",
            confidence=0.85,
            parameters={"patient_id": "P001"},
            timestamp=now,
            processing_time_ms=1500
        )

        result_dict = result.to_dict()

        assert result_dict["command"] == "record_vitals"
        assert result_dict["transcribed_text"] == "record vitals"
        assert result_dict["confidence"] == 0.85
        assert result_dict["parameters"] == {"patient_id": "P001"}
        assert result_dict["timestamp"] == now.isoformat()
        assert result_dict["processing_time_ms"] == 1500


class TestCameraService:
    """Test suite for CameraService"""

    @pytest.fixture
    def camera_service(self):
        """Create CameraService instance"""
        return CameraService()

    @pytest.mark.asyncio
    async def test_capture_wound_care_photo(self, camera_service):
        """Test capturing wound care photo"""
        photo = await camera_service.capture_clinical_photo(
            PhotoType.WOUND_CARE,
            "P001",
            "Nurse Johnson",
            ["Pre-treatment", "Left leg wound"]
        )

        assert isinstance(photo, PhotoCapture)
        assert photo.photo_type == PhotoType.WOUND_CARE
        assert photo.patient_id == "P001"
        assert photo.captured_by == "Nurse Johnson"
        assert len(photo.annotations) == 2
        assert photo.photo_id.startswith("PHOTO_")

    @pytest.mark.asyncio
    async def test_capture_medication_verification_photo(self, camera_service):
        """Test capturing medication verification photo"""
        photo = await camera_service.capture_clinical_photo(
            PhotoType.MEDICATION_VERIFICATION,
            "P002",
            "Pharmacist Smith"
        )

        assert photo.photo_type == PhotoType.MEDICATION_VERIFICATION
        assert photo.patient_id == "P002"
        assert photo.captured_by == "Pharmacist Smith"
        assert len(photo.annotations) == 0  # No annotations provided

    @pytest.mark.asyncio
    async def test_capture_equipment_status_photo(self, camera_service):
        """Test capturing equipment status photo"""
        photo = await camera_service.capture_clinical_photo(
            PhotoType.EQUIPMENT_STATUS,
            "P003",
            "Tech Support",
            ["Monitor calibration", "All readings normal"]
        )

        assert photo.photo_type == PhotoType.EQUIPMENT_STATUS
        assert "Monitor calibration" in photo.annotations

    def test_photo_metadata_structure(self, camera_service):
        """Test photo metadata structure"""
        metadata = camera_service._generate_photo_metadata()

        required_fields = ["resolution", "file_format", "compression_quality",
                          "flash_used", "camera_settings", "device_info"]

        for field in required_fields:
            assert field in metadata

        # Check camera settings structure
        camera_settings = metadata["camera_settings"]
        assert "iso" in camera_settings
        assert "aperture" in camera_settings
        assert "shutter_speed" in camera_settings

        # Check device info structure
        device_info = metadata["device_info"]
        assert "model" in device_info
        assert "os" in device_info
        assert "app_version" in device_info

    def test_get_patient_photos(self, camera_service):
        """Test getting photos for a patient"""
        # Initially no photos
        photos = camera_service.get_patient_photos("P001")
        assert len(photos) == 0

    @pytest.mark.asyncio
    async def test_get_patient_photos_after_capture(self, camera_service):
        """Test getting photos after capturing some"""
        # Capture photos for patient
        await camera_service.capture_clinical_photo(PhotoType.WOUND_CARE, "P001", "Nurse A")
        await camera_service.capture_clinical_photo(PhotoType.DOCUMENTATION, "P001", "Nurse B")
        await camera_service.capture_clinical_photo(PhotoType.WOUND_CARE, "P002", "Nurse C")  # Different patient

        # Get photos for P001
        p001_photos = camera_service.get_patient_photos("P001")
        assert len(p001_photos) == 2

        # Get photos for P002
        p002_photos = camera_service.get_patient_photos("P002")
        assert len(p002_photos) == 1

    @pytest.mark.asyncio
    async def test_annotate_photo(self, camera_service):
        """Test adding annotations to existing photo"""
        # Capture a photo
        photo = await camera_service.capture_clinical_photo(
            PhotoType.WOUND_CARE,
            "P001",
            "Nurse Johnson"
        )

        # Add annotation
        success = await camera_service.annotate_photo(photo.photo_id, "Healing progress noted")

        assert success
        assert "Healing progress noted" in photo.annotations

    @pytest.mark.asyncio
    async def test_annotate_nonexistent_photo(self, camera_service):
        """Test annotating non-existent photo"""
        success = await camera_service.annotate_photo("NONEXISTENT_ID", "Test annotation")
        assert not success

    def test_photo_capture_to_dict(self):
        """Test PhotoCapture dictionary conversion"""
        now = datetime.now()
        photo = PhotoCapture(
            photo_id="PHOTO_12345678",
            photo_type=PhotoType.WOUND_CARE,
            patient_id="P001",
            image_data="base64encodeddata",
            metadata={"test": "metadata"},
            timestamp=now,
            captured_by="Nurse Johnson",
            location="Room 101",
            annotations=["Test annotation"]
        )

        photo_dict = photo.to_dict()

        assert photo_dict["photo_id"] == "PHOTO_12345678"
        assert photo_dict["photo_type"] == "wound_care"
        assert photo_dict["patient_id"] == "P001"
        assert photo_dict["metadata"] == {"test": "metadata"}
        assert photo_dict["timestamp"] == now.isoformat()
        assert photo_dict["captured_by"] == "Nurse Johnson"
        assert photo_dict["location"] == "Room 101"
        assert photo_dict["annotations"] == ["Test annotation"]
        assert "file_size_mb" in photo_dict

    def test_photo_file_size_calculation(self):
        """Test photo file size calculation"""
        # Small photo
        small_photo = PhotoCapture(
            photo_id="TEST",
            photo_type=PhotoType.DOCUMENTATION,
            patient_id="P001",
            image_data="small",  # Very small base64 data
            metadata={},
            timestamp=datetime.now(),
            captured_by="Test"
        )

        assert small_photo.get_file_size_mb() < 0.001  # Very small

        # Larger photo (simulate)
        large_data = "x" * 10000  # 10KB of data
        large_photo = PhotoCapture(
            photo_id="TEST",
            photo_type=PhotoType.WOUND_CARE,
            patient_id="P001",
            image_data=large_data,
            metadata={},
            timestamp=datetime.now(),
            captured_by="Test"
        )

        assert large_photo.get_file_size_mb() > 0.005  # Larger


class TestDeviceIntegrationService:
    """Test suite for DeviceIntegrationService"""

    @pytest.fixture
    def device_service(self):
        """Create DeviceIntegrationService instance"""
        return DeviceIntegrationService()

    @pytest.mark.asyncio
    async def test_scan_patient_id_integration(self, device_service):
        """Test integrated patient ID scanning"""
        result = await device_service.scan_patient_id()

        assert result["success"]
        assert "patient_id" in result
        assert "confidence" in result
        assert "metadata" in result
        assert result["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_voice_vital_entry_integration(self, device_service):
        """Test integrated voice vital entry"""
        audio_data = b"dummy_audio_data"

        with patch.object(device_service.voice_input, 'process_voice_input') as mock_voice:
            mock_voice.return_value = VoiceRecognitionResult(
                command=VoiceCommand.RECORD_VITALS,
                transcribed_text="record vitals for patient P001",
                confidence=0.9,
                parameters={"patient_id": "P001", "respiratory_rate": 18},
                timestamp=datetime.now(),
                processing_time_ms=1000
            )

            result = await device_service.voice_vital_entry(audio_data)

            assert result["success"]
            assert result["command"] == "record_vitals"
            assert "patient_id" in result["parameters"]
            assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_voice_non_vitals_command(self, device_service):
        """Test voice command that's not for vitals entry"""
        audio_data = b"dummy_audio_data"

        with patch.object(device_service.voice_input, 'process_voice_input') as mock_voice:
            mock_voice.return_value = VoiceRecognitionResult(
                command=VoiceCommand.SEARCH_PATIENT,
                transcribed_text="find patient Smith",
                confidence=0.85,
                parameters={"patient_name": "Smith"},
                timestamp=datetime.now(),
                processing_time_ms=800
            )

            result = await device_service.voice_vital_entry(audio_data)

            assert result["success"]
            assert result["command"] == "search_patient"
            assert "not for vital signs entry" in result["message"]

    @pytest.mark.asyncio
    async def test_document_wound_care_integration(self, device_service):
        """Test integrated wound care documentation"""
        result = await device_service.document_wound_care(
            "P001",
            "Nurse Johnson",
            ["Pre-treatment photo", "Left leg"]
        )

        assert result["success"]
        assert "photo_id" in result
        assert "file_size_mb" in result
        assert "timestamp" in result

    def test_get_device_capabilities(self, device_service):
        """Test getting device capabilities"""
        capabilities = device_service.get_device_capabilities()

        assert "barcode_scanning" in capabilities
        assert "voice_input" in capabilities
        assert "camera" in capabilities
        assert "location_services" in capabilities

        # Check barcode scanning capabilities
        barcode_caps = capabilities["barcode_scanning"]
        assert barcode_caps["available"]
        assert len(barcode_caps["supported_formats"]) > 0
        assert len(barcode_caps["supported_types"]) > 0

        # Check voice input capabilities
        voice_caps = capabilities["voice_input"]
        assert voice_caps["available"]
        assert len(voice_caps["supported_commands"]) > 0

        # Check camera capabilities
        camera_caps = capabilities["camera"]
        assert camera_caps["available"]
        assert len(camera_caps["supported_types"]) > 0


class TestDeviceIntegrationErrorHandling:
    """Test suite for device integration error handling"""

    @pytest.fixture
    def device_service(self):
        """Create DeviceIntegrationService instance"""
        return DeviceIntegrationService()

    @pytest.mark.asyncio
    async def test_scan_patient_id_error(self, device_service):
        """Test error handling in patient ID scanning"""
        with patch.object(device_service.barcode_scanner, 'scan_barcode',
                         side_effect=Exception("Scanner error")):
            result = await device_service.scan_patient_id()

            assert not result["success"]
            assert "error" in result

    @pytest.mark.asyncio
    async def test_voice_processing_error(self, device_service):
        """Test error handling in voice processing"""
        with patch.object(device_service.voice_input, 'process_voice_input',
                         side_effect=Exception("Voice processing failed")):
            result = await device_service.voice_vital_entry(b"audio")

            assert not result["success"]
            assert "error" in result

    @pytest.mark.asyncio
    async def test_photo_capture_error(self, device_service):
        """Test error handling in photo capture"""
        with patch.object(device_service.camera_service, 'capture_clinical_photo',
                         side_effect=Exception("Camera error")):
            result = await device_service.document_wound_care("P001", "Nurse")

            assert not result["success"]
            assert "error" in result