import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.audit import AuditLogger, AuditEntry, AuditOperation, AuditService, AuditException


class TestAuditLogger:
    """Unit tests for AuditLogger service."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.audit_logger = AuditLogger()
    
    def create_test_patient(self):
        """Helper method to create test patient."""
        return Patient(
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
    
    def create_test_vital_signs(self):
        """Helper method to create test vital signs."""
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
            consciousness=ConsciousnessLevel.ALERT
        )
    
    def test_create_audit_entry_basic(self):
        """Test creating a basic audit entry."""
        audit_entry = self.audit_logger.create_audit_entry(
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id="P001"
        )
        
        assert isinstance(audit_entry, AuditEntry)
        assert isinstance(audit_entry.audit_id, UUID)
        assert audit_entry.table_name == "patients"
        assert audit_entry.operation == AuditOperation.INSERT
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert isinstance(audit_entry.timestamp, datetime)
        assert audit_entry.timestamp.tzinfo is not None  # Should have timezone
    
    def test_create_audit_entry_with_values(self):
        """Test creating audit entry with old and new values."""
        patient = self.create_test_patient()
        
        audit_entry = self.audit_logger.create_audit_entry(
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id=patient.patient_id,
            new_values=patient
        )
        
        assert audit_entry.new_values is not None
        assert isinstance(audit_entry.new_values, dict)
        assert audit_entry.new_values["patient_id"] == "P001"
        assert audit_entry.new_values["ward_id"] == "ICU-1"
        assert audit_entry.old_values is None
    
    def test_audit_patient_creation(self):
        """Test audit entry creation for patient creation."""
        patient = self.create_test_patient()
        
        audit_entry = self.audit_logger.audit_patient_creation(patient, "U001")
        
        assert audit_entry.table_name == "patients"
        assert audit_entry.operation == AuditOperation.INSERT
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.new_values is not None
        assert audit_entry.old_values is None
        assert audit_entry.new_values["patient_id"] == "P001"
    
    def test_audit_patient_update(self):
        """Test audit entry creation for patient update."""
        old_patient = self.create_test_patient()
        new_patient = self.create_test_patient()
        new_patient.age = 66  # Change age
        new_patient.is_palliative = True  # Change clinical flag
        
        audit_entry = self.audit_logger.audit_patient_update(old_patient, new_patient, "U001")
        
        assert audit_entry.table_name == "patients"
        assert audit_entry.operation == AuditOperation.UPDATE
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.old_values is not None
        assert audit_entry.new_values is not None
        
        # Verify the changes are captured
        assert audit_entry.old_values["age"] == 65
        assert audit_entry.new_values["age"] == 66
        assert audit_entry.old_values["clinical_flags"]["is_palliative"] is False
        assert audit_entry.new_values["clinical_flags"]["is_palliative"] is True
    
    def test_audit_patient_deletion(self):
        """Test audit entry creation for patient deletion."""
        patient = self.create_test_patient()
        
        audit_entry = self.audit_logger.audit_patient_deletion(patient, "U001")
        
        assert audit_entry.table_name == "patients"
        assert audit_entry.operation == AuditOperation.DELETE
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.old_values is not None
        assert audit_entry.new_values is None
    
    def test_audit_vital_signs_creation(self):
        """Test audit entry creation for vital signs creation."""
        vitals = self.create_test_vital_signs()
        
        audit_entry = self.audit_logger.audit_vital_signs_creation(vitals, "U001")
        
        assert audit_entry.table_name == "vital_signs"
        assert audit_entry.operation == AuditOperation.INSERT
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.new_values is not None
        assert audit_entry.old_values is None
        assert audit_entry.new_values["vitals"]["respiratory_rate"] == 18
    
    def test_audit_vital_signs_update(self):
        """Test audit entry creation for vital signs update."""
        old_vitals = self.create_test_vital_signs()
        new_vitals = self.create_test_vital_signs()
        new_vitals.respiratory_rate = 22  # Change RR
        new_vitals.consciousness = ConsciousnessLevel.CONFUSION  # Change consciousness
        
        audit_entry = self.audit_logger.audit_vital_signs_update(old_vitals, new_vitals, "U001")
        
        assert audit_entry.table_name == "vital_signs"
        assert audit_entry.operation == AuditOperation.UPDATE
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.old_values is not None
        assert audit_entry.new_values is not None
        
        # Verify the changes are captured
        assert audit_entry.old_values["vitals"]["respiratory_rate"] == 18
        assert audit_entry.new_values["vitals"]["respiratory_rate"] == 22
        assert audit_entry.old_values["vitals"]["consciousness"] == "A"
        assert audit_entry.new_values["vitals"]["consciousness"] == "C"
    
    def test_audit_vital_signs_deletion(self):
        """Test audit entry creation for vital signs deletion."""
        vitals = self.create_test_vital_signs()
        
        audit_entry = self.audit_logger.audit_vital_signs_deletion(vitals, "U001")
        
        assert audit_entry.table_name == "vital_signs"
        assert audit_entry.operation == AuditOperation.DELETE
        assert audit_entry.user_id == "U001"
        assert audit_entry.patient_id == "P001"
        assert audit_entry.old_values is not None
        assert audit_entry.new_values is None
    
    def test_serialize_audit_data_dict(self):
        """Test serialization of dictionary data."""
        data = {
            "field1": "value1",
            "field2": 123,
            "field3": True,
            "nested": {"inner": "value"}
        }
        
        result = self.audit_logger._serialize_audit_data(data)
        
        assert isinstance(result, dict)
        assert result["field1"] == "value1"
        assert result["field2"] == 123
        assert result["field3"] is True
        assert result["nested"]["inner"] == "value"
    
    def test_serialize_audit_data_patient_model(self):
        """Test serialization of Patient model."""
        patient = self.create_test_patient()
        
        result = self.audit_logger._serialize_audit_data(patient)
        
        assert isinstance(result, dict)
        assert result["patient_id"] == "P001"
        assert result["ward_id"] == "ICU-1"
        assert result["age"] == 65
        assert result["clinical_flags"]["is_palliative"] is False
    
    def test_serialize_audit_data_vital_signs_model(self):
        """Test serialization of VitalSigns model."""
        vitals = self.create_test_vital_signs()
        
        result = self.audit_logger._serialize_audit_data(vitals)
        
        assert isinstance(result, dict)
        assert result["patient_id"] == "P001"
        assert result["vitals"]["respiratory_rate"] == 18
        assert result["vitals"]["consciousness"] == "A"
        assert result["quality_flags"]["confidence"] == 1.0
    
    def test_ensure_json_serializable(self):
        """Test JSON serialization of complex objects."""
        now = datetime.now(timezone.utc)
        test_uuid = uuid4()
        
        complex_obj = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "none_value": None,
            "datetime": now,
            "uuid": test_uuid,
            "enum": ConsciousnessLevel.ALERT,
            "nested_dict": {
                "inner_datetime": now,
                "inner_uuid": test_uuid
            },
            "list": [now, test_uuid, ConsciousnessLevel.CONFUSION]
        }
        
        result = self.audit_logger._ensure_json_serializable(complex_obj)
        
        # Check basic types remain unchanged
        assert result["string"] == "test"
        assert result["integer"] == 123
        assert result["float"] == 45.67
        assert result["boolean"] is True
        assert result["none_value"] is None
        
        # Check complex types are serialized
        assert result["datetime"] == now.isoformat()
        assert result["uuid"] == str(test_uuid)
        assert result["enum"] == "A"
        
        # Check nested objects
        assert result["nested_dict"]["inner_datetime"] == now.isoformat()
        assert result["nested_dict"]["inner_uuid"] == str(test_uuid)
        
        # Check lists
        assert result["list"][0] == now.isoformat()
        assert result["list"][1] == str(test_uuid)
        assert result["list"][2] == "C"
    
    def test_hash_patient_id_consistency(self):
        """Test patient ID hashing is consistent and secure."""
        patient_id = "P001"
        
        hash_1 = self.audit_logger._hash_patient_id(patient_id)
        hash_2 = self.audit_logger._hash_patient_id(patient_id)
        
        # Should be consistent
        assert hash_1 == hash_2
        
        # Should be different for different IDs
        hash_different = self.audit_logger._hash_patient_id("P002")
        assert hash_1 != hash_different
        
        # Should be limited length
        assert len(hash_1) == 16
        
        # Should not contain original patient ID
        assert "P001" not in hash_1
    
    def test_validate_audit_entry_valid(self):
        """Test validation of valid audit entry."""
        audit_entry = AuditEntry(
            audit_id=uuid4(),
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            new_values={"field": "value"}
        )
        
        result = self.audit_logger.validate_audit_entry(audit_entry)
        assert result is True
    
    def test_validate_audit_entry_missing_audit_id(self):
        """Test validation fails for missing audit_id."""
        audit_entry = AuditEntry(
            audit_id=None,
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(AuditException, match="must have an audit_id"):
            self.audit_logger.validate_audit_entry(audit_entry)
    
    def test_validate_audit_entry_missing_table_name(self):
        """Test validation fails for missing table_name."""
        audit_entry = AuditEntry(
            audit_id=uuid4(),
            table_name=None,
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(AuditException, match="must specify table_name"):
            self.audit_logger.validate_audit_entry(audit_entry)
    
    def test_validate_audit_entry_insert_requires_new_values(self):
        """Test validation fails for INSERT without new_values."""
        audit_entry = AuditEntry(
            audit_id=uuid4(),
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            new_values=None
        )
        
        with pytest.raises(AuditException, match="INSERT operation must include new_values"):
            self.audit_logger.validate_audit_entry(audit_entry)
    
    def test_validate_audit_entry_delete_requires_old_values(self):
        """Test validation fails for DELETE without old_values."""
        audit_entry = AuditEntry(
            audit_id=uuid4(),
            table_name="patients",
            operation=AuditOperation.DELETE,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            old_values=None
        )
        
        with pytest.raises(AuditException, match="DELETE operation must include old_values"):
            self.audit_logger.validate_audit_entry(audit_entry)
    
    def test_validate_audit_entry_update_requires_both_values(self):
        """Test validation fails for UPDATE without both old and new values."""
        audit_entry = AuditEntry(
            audit_id=uuid4(),
            table_name="patients",
            operation=AuditOperation.UPDATE,
            user_id="U001",
            patient_id="P001",
            timestamp=datetime.now(timezone.utc),
            old_values={"field": "old_value"},
            new_values=None
        )
        
        with pytest.raises(AuditException, match="UPDATE operation must include both old_values and new_values"):
            self.audit_logger.validate_audit_entry(audit_entry)
    
    def test_audit_entry_to_dict(self):
        """Test AuditEntry serialization to dictionary."""
        audit_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        
        audit_entry = AuditEntry(
            audit_id=audit_id,
            table_name="patients",
            operation=AuditOperation.UPDATE,
            user_id="U001",
            patient_id="P001",
            timestamp=timestamp,
            old_values={"field": "old_value"},
            new_values={"field": "new_value"}
        )
        
        result = audit_entry.to_dict()
        
        assert result["audit_id"] == str(audit_id)
        assert result["table_name"] == "patients"
        assert result["operation"] == "UPDATE"
        assert result["user_id"] == "U001"
        assert result["patient_id"] == "P001"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["old_values"] == {"field": "old_value"}
        assert result["new_values"] == {"field": "new_value"}
    
    def test_audit_logging_with_patient_hash(self, caplog):
        """Test that audit creation is logged with hashed patient ID."""
        import logging
        
        caplog.set_level(logging.INFO)
        
        patient = self.create_test_patient()
        audit_entry = self.audit_logger.audit_patient_creation(patient, "U001")
        
        # Check that info log was created
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.levelname == "INFO"
        assert "Audit entry created" in log_record.message
        assert str(audit_entry.audit_id) in log_record.message
        assert "patients" in log_record.message
        assert "INSERT" in log_record.message
        assert "U001" in log_record.message
        
        # Patient ID should be hashed, not raw
        assert "P001" not in log_record.message
        # Should contain a hash
        assert len([part for part in log_record.message.split() if len(part) == 16]) >= 1


class TestAuditService:
    """Unit tests for AuditService high-level coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.audit_service = AuditService()
    
    def test_audit_service_initialization(self):
        """Test audit service initializes correctly."""
        assert self.audit_service.audit_logger is not None
        assert isinstance(self.audit_service.audit_logger, AuditLogger)
    
    def test_audit_service_with_custom_logger(self):
        """Test audit service with custom audit logger."""
        custom_logger = AuditLogger()
        audit_service = AuditService(custom_logger)
        
        assert audit_service.audit_logger is custom_logger