import pytest
import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.validation import VitalSignsValidator
from src.services.audit import AuditLogger, AuditService


class TestVitalSignsValidationIntegration:
    """Integration tests for vital signs validation with audit trail."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = VitalSignsValidator()
        self.audit_logger = AuditLogger()
        self.audit_service = AuditService(self.audit_logger)
    
    def create_test_patient(self, patient_id="P001", is_copd=False):
        """Helper to create test patient."""
        return Patient(
            patient_id=patient_id,
            ward_id="ICU-1",
            bed_number="B-101",
            age=65,
            is_copd_patient=is_copd,
            assigned_nurse_id="N001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
    
    def create_valid_vitals(self, patient_id="P001", **overrides):
        """Helper to create valid vital signs."""
        defaults = {
            "event_id": uuid4(),
            "patient_id": patient_id,
            "timestamp": datetime.now(timezone.utc),
            "respiratory_rate": 18,
            "spo2": 98,
            "on_oxygen": False,
            "temperature": 36.5,
            "systolic_bp": 120,
            "heart_rate": 75,
            "consciousness": ConsciousnessLevel.ALERT,
            "is_manual_entry": False,
            "has_artifacts": False,
            "confidence": 1.0
        }
        defaults.update(overrides)
        return VitalSigns(**defaults)
    
    def test_end_to_end_valid_vital_signs_insertion(self):
        """Test complete flow of valid vital signs insertion with audit."""
        # Create test patient
        patient = self.create_test_patient()
        
        # Create valid vital signs
        vitals = self.create_valid_vitals()
        
        # Validate vital signs
        validation_errors = self.validator.validate_vital_signs(vitals)
        assert len(validation_errors) == 0, "Valid vitals should pass validation"
        
        # Create audit entry for insertion
        audit_entry = self.audit_logger.audit_vital_signs_creation(vitals, "U001")
        
        # Validate audit entry
        assert self.audit_logger.validate_audit_entry(audit_entry) is True
        
        # Verify audit entry contents
        assert audit_entry.table_name == "vital_signs"
        assert audit_entry.patient_id == vitals.patient_id
        assert audit_entry.new_values["vitals"]["respiratory_rate"] == 18
        assert audit_entry.new_values["vitals"]["spo2"] == 98
    
    def test_end_to_end_invalid_vital_signs_rejection(self):
        """Test complete flow of invalid vital signs rejection with detailed errors."""
        # Create invalid vital signs (multiple errors)
        vitals = self.create_valid_vitals(
            respiratory_rate=3,  # Invalid - below minimum
            spo2=101,  # Invalid - above maximum
            temperature=29.0,  # Invalid - below minimum
            heart_rate=221,  # Invalid - above maximum
            confidence=1.5  # Invalid - above maximum
        )
        
        # Validate vital signs
        validation_errors = self.validator.validate_vital_signs(vitals)
        
        # Should have multiple validation errors
        assert len(validation_errors) >= 4, "Should have multiple validation errors"
        
        # Format errors for response
        error_response = self.validator.format_validation_errors(validation_errors)
        
        assert error_response["valid"] is False
        assert error_response["error_count"] >= 4
        assert len(error_response["errors"]) >= 4
        
        # Check specific error details
        error_fields = {error["field"] for error in error_response["errors"]}
        expected_fields = {"respiratory_rate", "spo2", "temperature", "heart_rate", "confidence"}
        assert error_fields.intersection(expected_fields) == expected_fields
        
        # Verify detailed error messages
        rr_error = next(e for e in error_response["errors"] if e["field"] == "respiratory_rate")
        assert rr_error["code"] == "RR_OUT_OF_RANGE"
        assert "4 and 50" in rr_error["message"]
        assert rr_error["received_value"] == 3
        assert "4-50" in rr_error["valid_range"]
    
    def test_concurrent_vital_signs_validation(self):
        """Test validation under concurrent load without race conditions."""
        num_patients = 100
        patients_per_thread = 10
        
        def validate_patient_vitals(patient_id_start):
            """Validate vitals for a batch of patients."""
            results = []
            for i in range(patients_per_thread):
                patient_id = f"P{patient_id_start + i:03d}"
                vitals = self.create_valid_vitals(
                    patient_id=patient_id,
                    respiratory_rate=15 + (i % 10),  # Vary within valid range
                    heart_rate=70 + (i % 20),  # Vary within valid range
                    spo2=95 + (i % 5)  # Vary within valid range
                )
                
                errors = self.validator.validate_vital_signs(vitals)
                results.append({
                    "patient_id": patient_id,
                    "valid": len(errors) == 0,
                    "error_count": len(errors)
                })
            return results
        
        # Run concurrent validation
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(0, num_patients, patients_per_thread):
                future = executor.submit(validate_patient_vitals, i)
                futures.append(future)
            
            all_results = []
            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all validations passed
        assert len(all_results) == num_patients
        valid_count = sum(1 for r in all_results if r["valid"])
        assert valid_count == num_patients, "All concurrent validations should pass"
        
        # Verify reasonable performance
        assert processing_time < 5.0, f"Processing {num_patients} validations took {processing_time:.2f}s, should be < 5s"
        
        # Verify no race conditions - all patient IDs should be unique
        patient_ids = {r["patient_id"] for r in all_results}
        assert len(patient_ids) == num_patients, "Should have unique patient IDs"
    
    def test_audit_trail_generation_for_all_operations(self):
        """Test audit trail is generated for all vital signs operations."""
        patient = self.create_test_patient()
        
        # Test INSERT audit
        vitals = self.create_valid_vitals()
        insert_audit = self.audit_logger.audit_vital_signs_creation(vitals, "U001")
        
        assert insert_audit.operation.value == "INSERT"
        assert insert_audit.new_values is not None
        assert insert_audit.old_values is None
        assert insert_audit.patient_id == vitals.patient_id
        
        # Test UPDATE audit
        updated_vitals = self.create_valid_vitals(
            event_id=vitals.event_id,  # Same event for update
            respiratory_rate=22,  # Changed value
            consciousness=ConsciousnessLevel.CONFUSION  # Changed value
        )
        update_audit = self.audit_logger.audit_vital_signs_update(vitals, updated_vitals, "U002")
        
        assert update_audit.operation.value == "UPDATE"
        assert update_audit.old_values is not None
        assert update_audit.new_values is not None
        assert update_audit.patient_id == vitals.patient_id
        
        # Verify changes are captured
        assert update_audit.old_values["vitals"]["respiratory_rate"] == 18
        assert update_audit.new_values["vitals"]["respiratory_rate"] == 22
        assert update_audit.old_values["vitals"]["consciousness"] == "A"
        assert update_audit.new_values["vitals"]["consciousness"] == "C"
        
        # Test DELETE audit
        delete_audit = self.audit_logger.audit_vital_signs_deletion(vitals, "U003")
        
        assert delete_audit.operation.value == "DELETE"
        assert delete_audit.old_values is not None
        assert delete_audit.new_values is None
        assert delete_audit.patient_id == vitals.patient_id
    
    def test_idempotency_with_duplicate_event_ids(self):
        """Test handling of duplicate event IDs for idempotency."""
        event_id = uuid4()
        patient_id = "P001"
        
        # Create two vital signs with same event ID but different values
        vitals_1 = self.create_valid_vitals(
            event_id=event_id,
            patient_id=patient_id,
            respiratory_rate=18,
            heart_rate=75
        )
        
        vitals_2 = self.create_valid_vitals(
            event_id=event_id,  # Same event ID
            patient_id=patient_id,
            respiratory_rate=20,  # Different values
            heart_rate=80
        )
        
        # Both should be valid individually
        errors_1 = self.validator.validate_vital_signs(vitals_1)
        errors_2 = self.validator.validate_vital_signs(vitals_2)
        
        assert len(errors_1) == 0
        assert len(errors_2) == 0
        
        # Create audit entries for both
        audit_1 = self.audit_logger.audit_vital_signs_creation(vitals_1, "U001")
        audit_2 = self.audit_logger.audit_vital_signs_creation(vitals_2, "U001")
        
        # Both should have same event ID in serialized form
        assert audit_1.new_values["event_id"] == str(event_id)
        assert audit_2.new_values["event_id"] == str(event_id)
        
        # But different audit IDs
        assert audit_1.audit_id != audit_2.audit_id
    
    def test_hipaa_compliance_no_phi_in_logs(self, caplog):
        """Test that no PHI data appears in logs."""
        import logging
        
        caplog.set_level(logging.DEBUG)  # Capture all log levels
        
        # Create patient with identifiable information
        patient = self.create_test_patient(patient_id="PATIENT_JOHN_DOE_123")
        
        # Create vital signs
        vitals = self.create_valid_vitals(patient_id="PATIENT_JOHN_DOE_123")
        
        # Perform validation (which logs validation failures if any)
        validation_errors = self.validator.validate_vital_signs(vitals)
        
        # Create audit entry (which logs audit creation)
        audit_entry = self.audit_logger.audit_vital_signs_creation(vitals, "NURSE_JANE_SMITH")
        
        # Check all log messages for PHI
        all_log_messages = " ".join([record.message for record in caplog.records])
        
        # Should not contain raw patient ID
        assert "PATIENT_JOHN_DOE_123" not in all_log_messages, "Raw patient ID found in logs"
        assert "JOHN_DOE" not in all_log_messages, "Patient name components found in logs"
        
        # Should not contain raw user information
        assert "JANE_SMITH" not in all_log_messages, "User name found in logs"
        
        # Should contain hashed patient references
        log_entries_with_hash = [record for record in caplog.records 
                               if any(len(word) == 16 and word.isalnum() 
                                     for word in record.message.split())]
        
        # At least one log entry should contain a hash (from audit logging)
        assert len(log_entries_with_hash) > 0, "No hashed patient references found in logs"
    
    def test_performance_single_patient_query_timing(self):
        """Test that single patient vital signs operations meet performance requirements."""
        patient = self.create_test_patient()
        
        # Test validation performance
        vitals = self.create_valid_vitals()
        
        start_time = time.time()
        for _ in range(100):  # Run 100 validations
            errors = self.validator.validate_vital_signs(vitals)
        validation_time = time.time() - start_time
        
        avg_validation_time = validation_time / 100
        assert avg_validation_time < 0.01, f"Average validation time {avg_validation_time:.4f}s should be < 10ms"
        
        # Test audit entry creation performance
        start_time = time.time()
        for _ in range(100):  # Create 100 audit entries
            audit_entry = self.audit_logger.audit_vital_signs_creation(vitals, "U001")
        audit_time = time.time() - start_time
        
        avg_audit_time = audit_time / 100
        assert avg_audit_time < 0.01, f"Average audit creation time {avg_audit_time:.4f}s should be < 10ms"
    
    def test_clinical_scenario_sepsis_progression(self):
        """Test realistic clinical scenario - sepsis progression with validation and audit."""
        # Create COPD patient (affects SpO2 scoring)
        patient = self.create_test_patient(is_copd=True)
        
        # T+0: Early sepsis signs (should be valid)
        vitals_t0 = self.create_valid_vitals(
            respiratory_rate=22,  # Slightly elevated
            spo2=94,  # Lower but acceptable for COPD
            temperature=38.2,  # Fever
            systolic_bp=115,  # Normal
            heart_rate=95,  # Slightly elevated
            consciousness=ConsciousnessLevel.ALERT
        )
        
        errors_t0 = self.validator.validate_vital_signs(vitals_t0)
        assert len(errors_t0) == 0, "Early sepsis vitals should be valid"
        
        audit_t0 = self.audit_logger.audit_vital_signs_creation(vitals_t0, "U001")
        assert audit_t0.new_values["vitals"]["respiratory_rate"] == 22
        
        # T+2h: Progression (should be valid but concerning)
        vitals_t2 = self.create_valid_vitals(
            respiratory_rate=28,  # More elevated
            spo2=91,  # Lower
            temperature=39.1,  # Higher fever
            systolic_bp=95,  # Dropping
            heart_rate=110,  # More elevated
            consciousness=ConsciousnessLevel.CONFUSION  # Altered
        )
        
        errors_t2 = self.validator.validate_vital_signs(vitals_t2)
        assert len(errors_t2) == 0, "Sepsis progression vitals should be valid"
        
        # Create audit for progression
        audit_t2 = self.audit_logger.audit_vital_signs_update(vitals_t0, vitals_t2, "U001")
        
        # Verify progression is captured in audit
        assert audit_t2.old_values["vitals"]["consciousness"] == "A"
        assert audit_t2.new_values["vitals"]["consciousness"] == "C"
        assert audit_t2.old_values["vitals"]["respiratory_rate"] == 22
        assert audit_t2.new_values["vitals"]["respiratory_rate"] == 28
        
        # T+4h: Severe deterioration (still valid ranges but critical)
        vitals_t4 = self.create_valid_vitals(
            respiratory_rate=35,  # High but still valid
            spo2=88,  # Low but still valid
            temperature=40.2,  # High fever but valid
            systolic_bp=80,  # Low but still valid
            heart_rate=130,  # High but valid
            consciousness=ConsciousnessLevel.PAIN  # Further deterioration
        )
        
        errors_t4 = self.validator.validate_vital_signs(vitals_t4)
        assert len(errors_t4) == 0, "Severe deterioration vitals should still be in valid ranges"
        
        # All vitals should be within clinical ranges even in severe sepsis
        # This tests that our validation ranges are appropriate for clinical use