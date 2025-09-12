import pytest
from datetime import datetime, timezone
from src.models.patient import Patient


class TestPatientModel:
    """Unit tests for Patient model validation and functionality."""
    
    def test_valid_patient_creation(self):
        """Test creating a valid patient with all required fields."""
        patient = Patient(
            patient_id="P001",
            ward_id="ICU-1",
            bed_number="B-101",
            age=65,
            is_copd_patient=True,
            assigned_nurse_id="N001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            is_palliative=False,
            do_not_escalate=False,
            oxygen_dependent=True
        )
        
        assert patient.patient_id == "P001"
        assert patient.ward_id == "ICU-1"
        assert patient.bed_number == "B-101"
        assert patient.age == 65
        assert patient.is_copd_patient is True
        assert patient.oxygen_dependent is True
        assert patient.is_palliative is False
        assert patient.do_not_escalate is False
    
    def test_patient_validation_empty_patient_id(self):
        """Test validation fails for empty patient_id."""
        with pytest.raises(ValueError, match="patient_id is required"):
            Patient(
                patient_id="",
                ward_id="ICU-1",
                bed_number="B-101",
                age=65,
                is_copd_patient=False,
                assigned_nurse_id="N001",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_validation_empty_ward_id(self):
        """Test validation fails for empty ward_id."""
        with pytest.raises(ValueError, match="ward_id is required"):
            Patient(
                patient_id="P001",
                ward_id="",
                bed_number="B-101",
                age=65,
                is_copd_patient=False,
                assigned_nurse_id="N001",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_validation_invalid_age_negative(self):
        """Test validation fails for negative age."""
        with pytest.raises(ValueError, match="age must be between 0 and 150"):
            Patient(
                patient_id="P001",
                ward_id="ICU-1",
                bed_number="B-101",
                age=-1,
                is_copd_patient=False,
                assigned_nurse_id="N001",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_validation_invalid_age_too_high(self):
        """Test validation fails for age over 150."""
        with pytest.raises(ValueError, match="age must be between 0 and 150"):
            Patient(
                patient_id="P001",
                ward_id="ICU-1",
                bed_number="B-101",
                age=151,
                is_copd_patient=False,
                assigned_nurse_id="N001",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_validation_invalid_copd_type(self):
        """Test validation fails for non-boolean is_copd_patient."""
        with pytest.raises(ValueError, match="is_copd_patient must be boolean"):
            Patient(
                patient_id="P001",
                ward_id="ICU-1",
                bed_number="B-101",
                age=65,
                is_copd_patient="yes",  # Should be boolean
                assigned_nurse_id="N001",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_to_dict(self):
        """Test patient serialization to dictionary."""
        now = datetime.now(timezone.utc)
        patient = Patient(
            patient_id="P001",
            ward_id="ICU-1",
            bed_number="B-101",
            age=65,
            is_copd_patient=True,
            assigned_nurse_id="N001",
            admission_date=now,
            last_updated=now,
            is_palliative=True,
            do_not_escalate=False,
            oxygen_dependent=True
        )
        
        patient_dict = patient.to_dict()
        
        assert patient_dict["patient_id"] == "P001"
        assert patient_dict["ward_id"] == "ICU-1"
        assert patient_dict["bed_number"] == "B-101"
        assert patient_dict["age"] == 65
        assert patient_dict["is_copd_patient"] is True
        assert patient_dict["assigned_nurse_id"] == "N001"
        assert patient_dict["admission_date"] == now.isoformat()
        assert patient_dict["last_updated"] == now.isoformat()
        assert patient_dict["clinical_flags"]["is_palliative"] is True
        assert patient_dict["clinical_flags"]["do_not_escalate"] is False
        assert patient_dict["clinical_flags"]["oxygen_dependent"] is True
    
    def test_patient_from_dict(self):
        """Test patient deserialization from dictionary."""
        now = datetime.now(timezone.utc)
        patient_dict = {
            "patient_id": "P001",
            "ward_id": "ICU-1",
            "bed_number": "B-101",
            "age": 65,
            "is_copd_patient": True,
            "assigned_nurse_id": "N001",
            "admission_date": now.isoformat(),
            "last_updated": now.isoformat(),
            "clinical_flags": {
                "is_palliative": True,
                "do_not_escalate": False,
                "oxygen_dependent": True
            }
        }
        
        patient = Patient.from_dict(patient_dict)
        
        assert patient.patient_id == "P001"
        assert patient.ward_id == "ICU-1"
        assert patient.bed_number == "B-101"
        assert patient.age == 65
        assert patient.is_copd_patient is True
        assert patient.assigned_nurse_id == "N001"
        assert patient.admission_date.replace(microsecond=0) == now.replace(microsecond=0)
        assert patient.last_updated.replace(microsecond=0) == now.replace(microsecond=0)
        assert patient.is_palliative is True
        assert patient.do_not_escalate is False
        assert patient.oxygen_dependent is True
    
    def test_patient_boundary_conditions(self):
        """Test patient creation with boundary values."""
        # Test minimum age
        patient_min_age = Patient(
            patient_id="P001",
            ward_id="ICU-1",
            bed_number="B-101",
            age=0,
            is_copd_patient=False,
            assigned_nurse_id="N001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        assert patient_min_age.age == 0
        
        # Test maximum age
        patient_max_age = Patient(
            patient_id="P002",
            ward_id="ICU-1",
            bed_number="B-102",
            age=150,
            is_copd_patient=False,
            assigned_nurse_id="N001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        assert patient_max_age.age == 150
    
    def test_patient_clinical_flags_defaults(self):
        """Test that clinical flags have correct default values."""
        patient = Patient(
            patient_id="P001",
            ward_id="ICU-1",
            bed_number="B-101",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="N001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # All clinical flags should default to False
        assert patient.is_palliative is False
        assert patient.do_not_escalate is False
        assert patient.oxygen_dependent is False