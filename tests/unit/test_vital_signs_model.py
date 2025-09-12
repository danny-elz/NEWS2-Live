import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from src.models.vital_signs import VitalSigns, ConsciousnessLevel


class TestVitalSignsModel:
    """Unit tests for VitalSigns model validation and functionality."""
    
    def test_valid_vital_signs_creation(self):
        """Test creating valid vital signs with all required fields."""
        event_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        vitals = VitalSigns(
            event_id=event_id,
            patient_id="P001",
            timestamp=timestamp,
            respiratory_rate=18,
            spo2=98,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT,
            is_manual_entry=True,
            has_artifacts=False,
            confidence=0.95
        )
        
        assert vitals.event_id == event_id
        assert vitals.patient_id == "P001"
        assert vitals.timestamp == timestamp
        assert vitals.respiratory_rate == 18
        assert vitals.spo2 == 98
        assert vitals.on_oxygen is False
        assert vitals.temperature == 36.5
        assert vitals.systolic_bp == 120
        assert vitals.heart_rate == 75
        assert vitals.consciousness == ConsciousnessLevel.ALERT
        assert vitals.is_manual_entry is True
        assert vitals.has_artifacts is False
        assert vitals.confidence == 0.95
    
    def test_vital_signs_boundary_conditions(self):
        """Test vital signs with boundary values."""
        event_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        # Test minimum boundary values
        vitals_min = VitalSigns(
            event_id=event_id,
            patient_id="P001",
            timestamp=timestamp,
            respiratory_rate=4,  # Minimum
            spo2=50,  # Minimum
            on_oxygen=True,
            temperature=30.0,  # Minimum
            systolic_bp=40,  # Minimum
            heart_rate=20,  # Minimum
            consciousness=ConsciousnessLevel.UNRESPONSIVE,
            confidence=0.0  # Minimum
        )
        
        assert vitals_min.respiratory_rate == 4
        assert vitals_min.spo2 == 50
        assert vitals_min.temperature == 30.0
        assert vitals_min.systolic_bp == 40
        assert vitals_min.heart_rate == 20
        assert vitals_min.confidence == 0.0
        
        # Test maximum boundary values
        vitals_max = VitalSigns(
            event_id=uuid4(),
            patient_id="P002",
            timestamp=timestamp,
            respiratory_rate=50,  # Maximum
            spo2=100,  # Maximum
            on_oxygen=False,
            temperature=45.0,  # Maximum
            systolic_bp=300,  # Maximum
            heart_rate=220,  # Maximum
            consciousness=ConsciousnessLevel.ALERT,
            confidence=1.0  # Maximum
        )
        
        assert vitals_max.respiratory_rate == 50
        assert vitals_max.spo2 == 100
        assert vitals_max.temperature == 45.0
        assert vitals_max.systolic_bp == 300
        assert vitals_max.heart_rate == 220
        assert vitals_max.confidence == 1.0
    
    def test_vital_signs_validation_respiratory_rate_out_of_range(self):
        """Test validation fails for respiratory rate out of range."""
        with pytest.raises(ValueError, match="respiratory_rate must be between 4 and 50"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=3,  # Below minimum
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
        
        with pytest.raises(ValueError, match="respiratory_rate must be between 4 and 50"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=51,  # Above maximum
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
    
    def test_vital_signs_validation_spo2_out_of_range(self):
        """Test validation fails for SpO2 out of range."""
        with pytest.raises(ValueError, match="spo2 must be between 50 and 100"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=49,  # Below minimum
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
        
        with pytest.raises(ValueError, match="spo2 must be between 50 and 100"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=101,  # Above maximum
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
    
    def test_vital_signs_validation_temperature_out_of_range(self):
        """Test validation fails for temperature out of range."""
        with pytest.raises(ValueError, match="temperature must be between 30 and 45"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=29.9,  # Below minimum
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
        
        with pytest.raises(ValueError, match="temperature must be between 30 and 45"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=45.1,  # Above maximum
                systolic_bp=120,
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
    
    def test_vital_signs_validation_heart_rate_out_of_range(self):
        """Test validation fails for heart rate out of range."""
        with pytest.raises(ValueError, match="heart_rate must be between 20 and 220"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=19,  # Below minimum
                consciousness=ConsciousnessLevel.ALERT
            )
        
        with pytest.raises(ValueError, match="heart_rate must be between 20 and 220"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=221,  # Above maximum
                consciousness=ConsciousnessLevel.ALERT
            )
    
    def test_vital_signs_validation_systolic_bp_out_of_range(self):
        """Test validation fails for systolic blood pressure out of range."""
        with pytest.raises(ValueError, match="systolic_bp must be between 40 and 300"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=39,  # Below minimum
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
        
        with pytest.raises(ValueError, match="systolic_bp must be between 40 and 300"):
            VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=301,  # Above maximum
                heart_rate=75,
                consciousness=ConsciousnessLevel.ALERT
            )
    
    def test_vital_signs_validation_confidence_out_of_range(self):
        """Test validation fails for confidence out of range."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            VitalSigns(
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
                confidence=-0.1  # Below minimum
            )
        
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            VitalSigns(
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
                confidence=1.1  # Above maximum
            )
    
    def test_consciousness_levels(self):
        """Test all consciousness levels are valid."""
        event_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        consciousness_levels = [
            ConsciousnessLevel.ALERT,
            ConsciousnessLevel.CONFUSION,
            ConsciousnessLevel.VOICE,
            ConsciousnessLevel.PAIN,
            ConsciousnessLevel.UNRESPONSIVE
        ]
        
        for consciousness in consciousness_levels:
            vitals = VitalSigns(
                event_id=uuid4(),
                patient_id="P001",
                timestamp=timestamp,
                respiratory_rate=18,
                spo2=98,
                on_oxygen=False,
                temperature=36.5,
                systolic_bp=120,
                heart_rate=75,
                consciousness=consciousness
            )
            assert vitals.consciousness == consciousness
    
    def test_vital_signs_to_dict(self):
        """Test vital signs serialization to dictionary."""
        event_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        vitals = VitalSigns(
            event_id=event_id,
            patient_id="P001",
            timestamp=timestamp,
            respiratory_rate=18,
            spo2=98,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT,
            is_manual_entry=True,
            has_artifacts=False,
            confidence=0.95
        )
        
        vitals_dict = vitals.to_dict()
        
        assert vitals_dict["event_id"] == str(event_id)
        assert vitals_dict["patient_id"] == "P001"
        assert vitals_dict["timestamp"] == timestamp.isoformat()
        assert vitals_dict["vitals"]["respiratory_rate"] == 18
        assert vitals_dict["vitals"]["spo2"] == 98
        assert vitals_dict["vitals"]["on_oxygen"] is False
        assert vitals_dict["vitals"]["temperature"] == 36.5
        assert vitals_dict["vitals"]["systolic_bp"] == 120
        assert vitals_dict["vitals"]["heart_rate"] == 75
        assert vitals_dict["vitals"]["consciousness"] == "A"
        assert vitals_dict["quality_flags"]["is_manual_entry"] is True
        assert vitals_dict["quality_flags"]["has_artifacts"] is False
        assert vitals_dict["quality_flags"]["confidence"] == 0.95
    
    def test_vital_signs_from_dict(self):
        """Test vital signs deserialization from dictionary."""
        event_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        vitals_dict = {
            "event_id": str(event_id),
            "patient_id": "P001",
            "timestamp": timestamp.isoformat(),
            "vitals": {
                "respiratory_rate": 18,
                "spo2": 98,
                "on_oxygen": False,
                "temperature": 36.5,
                "systolic_bp": 120,
                "heart_rate": 75,
                "consciousness": "A"
            },
            "quality_flags": {
                "is_manual_entry": True,
                "has_artifacts": False,
                "confidence": 0.95
            }
        }
        
        vitals = VitalSigns.from_dict(vitals_dict)
        
        assert vitals.event_id == event_id
        assert vitals.patient_id == "P001"
        assert vitals.timestamp.replace(microsecond=0) == timestamp.replace(microsecond=0)
        assert vitals.respiratory_rate == 18
        assert vitals.spo2 == 98
        assert vitals.on_oxygen is False
        assert vitals.temperature == 36.5
        assert vitals.systolic_bp == 120
        assert vitals.heart_rate == 75
        assert vitals.consciousness == ConsciousnessLevel.ALERT
        assert vitals.is_manual_entry is True
        assert vitals.has_artifacts is False
        assert vitals.confidence == 0.95
    
    def test_generate_event_id(self):
        """Test event ID generation creates valid UUIDs."""
        event_id_1 = VitalSigns.generate_event_id()
        event_id_2 = VitalSigns.generate_event_id()
        
        # Should be valid UUIDs
        assert isinstance(event_id_1, UUID)
        assert isinstance(event_id_2, UUID)
        
        # Should be unique
        assert event_id_1 != event_id_2
    
    def test_vital_signs_defaults(self):
        """Test that quality flags have correct default values."""
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=98,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Quality flags should have correct defaults
        assert vitals.is_manual_entry is False
        assert vitals.has_artifacts is False
        assert vitals.confidence == 1.0