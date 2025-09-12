import pytest
from datetime import datetime, timezone
from uuid import uuid4
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.validation import VitalSignsValidator, ValidationError, ValidationErrorCode


class TestVitalSignsValidator:
    """Unit tests for VitalSignsValidator service."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.validator = VitalSignsValidator()
    
    def create_valid_vitals(self):
        """Helper method to create valid vital signs for testing."""
        return VitalSigns(
            event_id=uuid4(),
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=98,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT,
            is_manual_entry=False,
            has_artifacts=False,
            confidence=1.0
        )
    
    def test_validate_valid_vitals(self):
        """Test validation passes for valid vital signs."""
        vitals = self.create_valid_vitals()
        errors = self.validator.validate_vital_signs(vitals)
        assert len(errors) == 0
    
    def test_validate_respiratory_rate_boundaries(self):
        """Test respiratory rate boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.respiratory_rate = 4  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.respiratory_rate = 50  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.respiratory_rate = 3  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "respiratory_rate"
        assert errors[0].code == ValidationErrorCode.RESPIRATORY_RATE_OUT_OF_RANGE
        assert "4 and 50" in errors[0].message
        
        vitals_above = self.create_valid_vitals()
        vitals_above.respiratory_rate = 51  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "respiratory_rate"
        assert errors[0].code == ValidationErrorCode.RESPIRATORY_RATE_OUT_OF_RANGE
    
    def test_validate_heart_rate_boundaries(self):
        """Test heart rate boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.heart_rate = 20  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.heart_rate = 220  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.heart_rate = 19  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "heart_rate"
        assert errors[0].code == ValidationErrorCode.HEART_RATE_OUT_OF_RANGE
        
        vitals_above = self.create_valid_vitals()
        vitals_above.heart_rate = 221  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "heart_rate"
        assert errors[0].code == ValidationErrorCode.HEART_RATE_OUT_OF_RANGE
    
    def test_validate_spo2_boundaries(self):
        """Test SpO2 boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.spo2 = 50  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.spo2 = 100  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.spo2 = 49  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "spo2"
        assert errors[0].code == ValidationErrorCode.SPO2_OUT_OF_RANGE
        
        vitals_above = self.create_valid_vitals()
        vitals_above.spo2 = 101  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "spo2"
        assert errors[0].code == ValidationErrorCode.SPO2_OUT_OF_RANGE
    
    def test_validate_temperature_boundaries(self):
        """Test temperature boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.temperature = 30.0  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.temperature = 45.0  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.temperature = 29.9  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "temperature"
        assert errors[0].code == ValidationErrorCode.TEMPERATURE_OUT_OF_RANGE
        
        vitals_above = self.create_valid_vitals()
        vitals_above.temperature = 45.1  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "temperature"
        assert errors[0].code == ValidationErrorCode.TEMPERATURE_OUT_OF_RANGE
    
    def test_validate_systolic_bp_boundaries(self):
        """Test systolic blood pressure boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.systolic_bp = 40  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.systolic_bp = 300  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.systolic_bp = 39  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "systolic_bp"
        assert errors[0].code == ValidationErrorCode.SYSTOLIC_BP_OUT_OF_RANGE
        
        vitals_above = self.create_valid_vitals()
        vitals_above.systolic_bp = 301  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "systolic_bp"
        assert errors[0].code == ValidationErrorCode.SYSTOLIC_BP_OUT_OF_RANGE
    
    def test_validate_confidence_boundaries(self):
        """Test confidence score boundary validation."""
        # Test valid boundaries
        vitals_min = self.create_valid_vitals()
        vitals_min.confidence = 0.0  # Minimum valid
        errors = self.validator.validate_vital_signs(vitals_min)
        assert len(errors) == 0
        
        vitals_max = self.create_valid_vitals()
        vitals_max.confidence = 1.0  # Maximum valid
        errors = self.validator.validate_vital_signs(vitals_max)
        assert len(errors) == 0
        
        # Test invalid boundaries
        vitals_below = self.create_valid_vitals()
        vitals_below.confidence = -0.1  # Below minimum
        errors = self.validator.validate_vital_signs(vitals_below)
        assert len(errors) == 1
        assert errors[0].field == "confidence"
        assert errors[0].code == ValidationErrorCode.CONFIDENCE_OUT_OF_RANGE
        
        vitals_above = self.create_valid_vitals()
        vitals_above.confidence = 1.1  # Above maximum
        errors = self.validator.validate_vital_signs(vitals_above)
        assert len(errors) == 1
        assert errors[0].field == "confidence"
        assert errors[0].code == ValidationErrorCode.CONFIDENCE_OUT_OF_RANGE
    
    def test_validate_multiple_errors(self):
        """Test validation with multiple invalid fields."""
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=3,  # Invalid - below minimum
            spo2=101,  # Invalid - above maximum
            on_oxygen=False,
            temperature=29.0,  # Invalid - below minimum
            systolic_bp=301,  # Invalid - above maximum
            heart_rate=19,  # Invalid - below minimum
            consciousness=ConsciousnessLevel.ALERT,
            confidence=1.5  # Invalid - above maximum
        )
        
        errors = self.validator.validate_vital_signs(vitals)
        
        # Should have 5 validation errors
        assert len(errors) == 5
        
        error_fields = {error.field for error in errors}
        expected_fields = {
            "respiratory_rate", "spo2", "temperature", 
            "systolic_bp", "heart_rate", "confidence"
        }
        
        assert error_fields == expected_fields
    
    def test_validate_consciousness_levels(self):
        """Test validation with different consciousness levels."""
        consciousness_levels = [
            ConsciousnessLevel.ALERT,
            ConsciousnessLevel.CONFUSION,
            ConsciousnessLevel.VOICE,
            ConsciousnessLevel.PAIN,
            ConsciousnessLevel.UNRESPONSIVE
        ]
        
        for consciousness in consciousness_levels:
            vitals = self.create_valid_vitals()
            vitals.consciousness = consciousness
            errors = self.validator.validate_vital_signs(vitals)
            assert len(errors) == 0, f"Valid consciousness level {consciousness} failed validation"
    
    def test_validate_partial_vitals(self):
        """Test validation of partial vitals dictionary."""
        # Test valid partial data
        partial_vitals = {
            "respiratory_rate": 18,
            "spo2": 98,
            "heart_rate": 75
        }
        
        errors = self.validator.validate_partial_vitals(partial_vitals)
        assert len(errors) == 0
        
        # Test invalid partial data
        invalid_partial = {
            "respiratory_rate": 3,  # Invalid
            "spo2": 98,  # Valid
            "heart_rate": 221  # Invalid
        }
        
        errors = self.validator.validate_partial_vitals(invalid_partial)
        assert len(errors) == 2
        
        error_fields = {error.field for error in errors}
        assert "respiratory_rate" in error_fields
        assert "heart_rate" in error_fields
        assert "spo2" not in error_fields  # This was valid
    
    def test_format_validation_errors_valid(self):
        """Test error formatting for valid data."""
        result = self.validator.format_validation_errors([])
        
        assert result["valid"] is True
        assert result["errors"] == []
    
    def test_format_validation_errors_invalid(self):
        """Test error formatting for invalid data."""
        errors = [
            ValidationError(
                field="respiratory_rate",
                code=ValidationErrorCode.RESPIRATORY_RATE_OUT_OF_RANGE,
                message="Respiratory rate must be between 4 and 50 breaths/min",
                received_value=3,
                valid_range="4-50 breaths/min"
            ),
            ValidationError(
                field="spo2",
                code=ValidationErrorCode.SPO2_OUT_OF_RANGE,
                message="SpO2 must be between 50 and 100%",
                received_value=101,
                valid_range="50-100%"
            )
        ]
        
        result = self.validator.format_validation_errors(errors)
        
        assert result["valid"] is False
        assert result["error_count"] == 2
        assert len(result["errors"]) == 2
        
        # Check first error
        first_error = result["errors"][0]
        assert first_error["field"] == "respiratory_rate"
        assert first_error["code"] == "RR_OUT_OF_RANGE"
        assert first_error["message"] == "Respiratory rate must be between 4 and 50 breaths/min"
        assert first_error["received_value"] == 3
        assert first_error["valid_range"] == "4-50 breaths/min"
        
        # Check second error
        second_error = result["errors"][1]
        assert second_error["field"] == "spo2"
        assert second_error["code"] == "SPO2_OUT_OF_RANGE"
        assert second_error["received_value"] == 101
    
    def test_hash_patient_id_consistency(self):
        """Test that patient ID hashing is consistent."""
        patient_id = "P001"
        
        hash_1 = self.validator._hash_patient_id(patient_id)
        hash_2 = self.validator._hash_patient_id(patient_id)
        
        # Should be consistent
        assert hash_1 == hash_2
        
        # Should be different for different patient IDs
        hash_different = self.validator._hash_patient_id("P002")
        assert hash_1 != hash_different
        
        # Should be limited length for logging
        assert len(hash_1) == 16
    
    def test_validation_error_logging(self, caplog):
        """Test that validation failures are logged with patient hash."""
        import logging
        
        # Set up logging to capture warnings
        caplog.set_level(logging.WARNING)
        
        vitals = self.create_valid_vitals()
        vitals.respiratory_rate = 3  # Invalid
        vitals.patient_id = "TEST_PATIENT_123"
        
        errors = self.validator.validate_vital_signs(vitals)
        
        assert len(errors) == 1
        
        # Check that warning was logged
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.levelname == "WARNING"
        assert "Vital signs validation failed" in log_record.message
        assert "TEST_PATIENT_123" not in log_record.message  # Patient ID should be hashed
        assert str(vitals.event_id) in log_record.message
        assert "Errors: 1" in log_record.message