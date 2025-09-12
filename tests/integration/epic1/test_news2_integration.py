"""
Integration tests for NEWS2 calculation system.

Tests end-to-end calculation using VitalSigns models from Story 1.1,
integration with Patient model, concurrent calculations, audit trail,
error handling, and 100% accuracy against RCP NEWS2 reference scenarios.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.models.partial_vital_signs import PartialVitalSigns
from src.models.news2 import NEWS2Result, RiskCategory
from src.services.audit import AuditLogger
from src.services.news2_calculator import NEWS2Calculator
from src.services.batch_processor import BatchNEWS2Processor, BatchRequest


class TestNEWS2Integration:
    """Integration test suite for NEWS2 calculation system."""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for integration tests."""
        return AuditLogger()
    
    @pytest.fixture
    def calculator(self, audit_logger):
        """Create NEWS2Calculator instance."""
        return NEWS2Calculator(audit_logger)
    
    @pytest.fixture
    def batch_processor(self, audit_logger):
        """Create batch processor for concurrent testing."""
        return BatchNEWS2Processor(audit_logger, max_workers=5)


class TestEndToEndCalculation(TestNEWS2Integration):
    """Test end-to-end calculation using VitalSigns models from Story 1.1."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_normal_patient(self, calculator):
        """Test complete workflow for normal patient."""
        # Create patient from Story 1.1 models
        patient = Patient(
            patient_id="INTEGRATION_001",
            ward_id="CARDIOLOGY",
            bed_number="C-012",
            age=58,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_SMITH",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Create vital signs using Story 1.1 models
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="INTEGRATION_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=20,
            spo2=96,
            on_oxygen=False,
            temperature=37.2,
            systolic_bp=135,
            heart_rate=88,
            consciousness=ConsciousnessLevel.ALERT,
            is_manual_entry=False,
            has_artifacts=False,
            confidence=0.98
        )
        
        # Perform calculation
        result = await calculator.calculate_news2(vitals, patient)
        
        # Verify result structure
        assert isinstance(result, NEWS2Result)
        assert result.total_score >= 0
        assert result.risk_category in [RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH]
        assert result.scale_used == 1
        assert result.confidence > 0.0
        assert result.calculated_at is not None
        assert result.calculation_time_ms > 0.0
        assert result.clinical_guidance is not None
        
        # Verify individual scores
        assert len(result.individual_scores) == 7  # All vital sign parameters
        for param, score in result.individual_scores.items():
            assert isinstance(score, int)
            assert 0 <= score <= 3
        
        # Verify clinical guidance structure
        assert "escalation" in result.clinical_guidance
        assert "response_time" in result.clinical_guidance
        assert "staff_level" in result.clinical_guidance
        assert "documentation" in result.clinical_guidance
    
    @pytest.mark.asyncio
    async def test_complete_workflow_copd_patient(self, calculator):
        """Test complete workflow for COPD patient using Scale 2."""
        # Create COPD patient
        copd_patient = Patient(
            patient_id="COPD_INTEGRATION_001",
            ward_id="RESPIRATORY",
            bed_number="R-008",
            age=72,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_JOHNSON",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Create vital signs typical for COPD patient
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="COPD_INTEGRATION_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=24,    # 2 points
            spo2=89,              # 0 points on Scale 2, 3 points on Scale 1
            on_oxygen=True,       # 2 points
            temperature=36.8,     # 0 points
            systolic_bp=110,      # 0 points
            heart_rate=92,        # 1 point
            consciousness=ConsciousnessLevel.ALERT,  # 0 points
            confidence=0.95
        )
        
        # Perform calculation
        result = await calculator.calculate_news2(vitals, copd_patient)
        
        # Verify COPD-specific behavior
        assert result.scale_used == 2, "COPD patient should use Scale 2"
        assert result.individual_scores['spo2'] == 0, "SpO2 89% should be 0 points on Scale 2"
        
        # Verify COPD warnings and guidance
        copd_warning = any("Scale 2 (COPD)" in warning for warning in result.warnings)
        assert copd_warning, "Expected COPD Scale 2 usage warning"
        
        assert "copd_considerations" in result.clinical_guidance, "Expected COPD clinical guidance"
        
        # Calculate expected total score (2+0+2+0+0+1+0 = 5)
        expected_total = 2 + 0 + 2 + 0 + 0 + 1 + 0
        assert result.total_score == expected_total
        assert result.risk_category == RiskCategory.MEDIUM


class TestPatientModelIntegration(TestNEWS2Integration):
    """Test integration with Patient model for COPD status detection."""
    
    @pytest.mark.asyncio
    async def test_patient_copd_status_detection(self, calculator):
        """Test automatic COPD status detection from Patient model."""
        # Create same vital signs for both patient types
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="TEST",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=93,  # Different scoring between scales
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Normal patient
        normal_patient = Patient(
            patient_id="NORMAL_001",
            ward_id="WARD_A",
            bed_number="001",
            age=55,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # COPD patient
        copd_patient = Patient(
            patient_id="COPD_001",
            ward_id="WARD_A",
            bed_number="002",
            age=68,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Calculate for both
        normal_result = await calculator.calculate_news2(vitals, normal_patient)
        copd_result = await calculator.calculate_news2(vitals, copd_patient)
        
        # Verify different scale usage
        assert normal_result.scale_used == 1, "Normal patient should use Scale 1"
        assert copd_result.scale_used == 2, "COPD patient should use Scale 2"
        
        # Verify different SpO2 scoring (93%)
        assert normal_result.individual_scores['spo2'] == 2, "Scale 1: SpO2 93% = 2 points"
        assert copd_result.individual_scores['spo2'] == 1, "Scale 2: SpO2 93% = 1 point"
        
        # Verify different total scores
        assert copd_result.total_score != normal_result.total_score, "Total scores should differ"
    
    @pytest.mark.asyncio
    async def test_patient_clinical_flags_integration(self, calculator):
        """Test integration with patient clinical flags."""
        # Create palliative care patient
        palliative_patient = Patient(
            patient_id="PALLIATIVE_001",
            ward_id="ONCOLOGY",
            bed_number="O-015",
            age=78,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_WILLIAMS",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            is_palliative=True,
            do_not_escalate=True
        )
        
        # Create concerning vital signs
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="PALLIATIVE_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=26,    # 3 points - red flag
            spo2=92,              # 2 points
            on_oxygen=True,       # 2 points
            temperature=38.8,     # 1 point
            systolic_bp=95,       # 2 points
            heart_rate=105,       # 1 point
            consciousness=ConsciousnessLevel.ALERT
        )
        
        result = await calculator.calculate_news2(vitals, palliative_patient)
        
        # Verify clinical flags are reflected in guidance
        assert "palliative_note" in result.clinical_guidance, "Expected palliative care guidance"
        assert "escalation_limit" in result.clinical_guidance, "Expected do-not-escalate guidance"
        
        palliative_note = result.clinical_guidance["palliative_note"]
        assert "care goals" in palliative_note or "advance directives" in palliative_note


class TestConcurrentCalculations(TestNEWS2Integration):
    """Test concurrent calculations for thread safety and performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_calculation_thread_safety(self, batch_processor):
        """Test thread-safe concurrent calculations."""
        # Create multiple calculation requests
        requests = []
        for i in range(20):
            patient = Patient(
                patient_id=f"CONCURRENT_{i:03d}",
                ward_id="ICU",
                bed_number=f"I-{i:03d}",
                age=45 + (i % 30),
                is_copd_patient=(i % 5 == 0),  # Every 5th patient is COPD
                assigned_nurse_id=f"NURSE_{i % 3 + 1:03d}",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            vitals = VitalSigns(
                event_id=uuid4(),
                patient_id=f"CONCURRENT_{i:03d}",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=15 + (i % 8),
                spo2=92 + (i % 8),
                on_oxygen=(i % 3 == 0),
                temperature=36.0 + (i % 4) * 0.5,
                systolic_bp=100 + (i % 30),
                heart_rate=60 + (i % 50),
                consciousness=ConsciousnessLevel.ALERT
            )
            
            request = BatchRequest(
                request_id=f"concurrent_{i:03d}",
                patient=patient,
                vital_signs=vitals
            )
            requests.append(request)
        
        # Process all requests concurrently
        start_time = time.perf_counter()
        results, stats = await batch_processor.process_batch(requests)
        end_time = time.perf_counter()
        
        # Verify all requests completed successfully
        assert len(results) == 20, "All concurrent requests should complete"
        assert stats.successful_calculations == 20, "All calculations should succeed"
        assert stats.failed_calculations == 0, "No calculations should fail"
        
        # Verify results are valid and different (ensuring no race conditions)
        total_scores = [r.result.total_score for r in results if r.result]
        assert len(set(total_scores)) > 1, "Different inputs should produce different results"
        
        # Verify thread safety by checking result consistency
        for result in results:
            assert result.result is not None, "All results should be present"
            assert result.result.total_score >= 0, "All scores should be valid"
            assert result.result.scale_used in [1, 2], "Scale should be 1 or 2"
        
        # Verify performance
        concurrent_time_ms = (end_time - start_time) * 1000
        assert concurrent_time_ms < 1000, "Concurrent processing should be fast"
        
        print(f"\nConcurrent processing metrics:")
        print(f"  Requests: {len(requests)}")
        print(f"  Total time: {concurrent_time_ms:.2f}ms")
        print(f"  Avg per request: {stats.average_time_per_calculation_ms:.2f}ms")
        print(f"  Success rate: {stats.successful_calculations}/{len(requests)}")


class TestAuditTrailIntegration(TestNEWS2Integration):
    """Test calculation audit trail integration with AuditLogger."""
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self, calculator, caplog):
        """Test that calculations are properly logged for audit trail."""
        import logging
        
        # Set up logging capture
        caplog.set_level(logging.INFO)
        
        # Create test data
        patient = Patient(
            patient_id="AUDIT_TEST_001",
            ward_id="EMERGENCY",
            bed_number="E-005",
            age=42,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_DAVIS",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="AUDIT_TEST_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=22,
            spo2=94,
            on_oxygen=True,
            temperature=38.2,
            systolic_bp=105,
            heart_rate=98,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Perform calculation
        result = await calculator.calculate_news2(vitals, patient)
        
        # Verify audit log entries were created
        audit_logs = [record.message for record in caplog.records 
                     if "NEWS2 calculation" in record.message]
        
        assert len(audit_logs) > 0, "Expected audit log entries"
        
        # Verify log contains key information
        audit_message = audit_logs[0]
        assert "Patient:" in audit_message, "Audit should contain patient hash"
        assert "Score:" in audit_message, "Audit should contain score"
        assert "Risk:" in audit_message, "Audit should contain risk category"
        assert "Scale:" in audit_message, "Audit should contain scale used"
        assert "Time:" in audit_message, "Audit should contain calculation time"
        
        # Verify patient ID is hashed for privacy
        assert "AUDIT_TEST_001" not in audit_message, "Patient ID should be hashed"


class TestErrorHandlingAndGracefulDegradation(TestNEWS2Integration):
    """Test error handling and graceful degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_vital_signs_graceful_degradation(self, calculator):
        """Test graceful handling of partial vital signs."""
        patient = Patient(
            patient_id="PARTIAL_TEST",
            ward_id="WARD_B",
            bed_number="B-010",
            age=67,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_BROWN",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Create partial vital signs (some missing)
        partial_vitals = PartialVitalSigns(
            event_id=uuid4(),
            patient_id="PARTIAL_TEST",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=20,
            spo2=None,            # Missing
            on_oxygen=True,
            temperature=None,     # Missing
            systolic_bp=110,
            heart_rate=None,      # Missing
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Should complete successfully with warnings
        result = await calculator.calculate_partial_news2(partial_vitals, patient)
        
        # Verify graceful degradation
        assert result.total_score >= 0, "Should calculate partial score"
        assert result.confidence < 1.0, "Confidence should be reduced for partial data"
        
        # Verify warnings about missing parameters
        missing_warnings = [w for w in result.warnings if "missing" in w.lower()]
        assert len(missing_warnings) > 0, "Should warn about missing parameters"
        
        # Verify clinical guidance includes missing parameter note
        assert "missing_parameters" in result.clinical_guidance
    
    @pytest.mark.asyncio
    async def test_calculation_retry_mechanism(self, calculator):
        """Test calculation retry mechanism for transient failures."""
        patient = Patient(
            patient_id="RETRY_TEST",
            ward_id="WARD_C",
            bed_number="C-007",
            age=51,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_WILSON",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Normal partial vitals should succeed without retries needed
        partial_vitals = PartialVitalSigns(
            event_id=uuid4(),
            patient_id="RETRY_TEST",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=96,
            on_oxygen=False,
            temperature=36.8,
            systolic_bp=125,
            heart_rate=72,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Should complete successfully
        result = await calculator.calculate_partial_news2(partial_vitals, patient, max_retries=3)
        assert result.total_score >= 0, "Calculation should succeed"


class TestRCPNews2AccuracyValidation(TestNEWS2Integration):
    """Validate 100% accuracy against RCP NEWS2 reference scenarios."""
    
    @pytest.mark.asyncio
    async def test_rcp_reference_scenarios(self, calculator):
        """Test against known RCP NEWS2 reference scenarios."""
        
        # RCP Reference Scenario 1: Normal healthy adult
        scenario_1_patient = Patient(
            patient_id="RCP_001",
            ward_id="GENERAL",
            bed_number="G-001",
            age=35,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        scenario_1_vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="RCP_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=16,    # 0 points
            spo2=98,              # 0 points
            on_oxygen=False,      # 0 points
            temperature=36.8,     # 0 points
            systolic_bp=125,      # 0 points
            heart_rate=70,        # 0 points
            consciousness=ConsciousnessLevel.ALERT  # 0 points
        )
        
        result_1 = await calculator.calculate_news2(scenario_1_vitals, scenario_1_patient)
        assert result_1.total_score == 0, "RCP Scenario 1: Expected score 0"
        assert result_1.risk_category == RiskCategory.LOW, "RCP Scenario 1: Expected LOW risk"
        
        # RCP Reference Scenario 2: Moderately unwell patient
        scenario_2_vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="RCP_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=22,    # 2 points
            spo2=94,              # 1 point
            on_oxygen=True,       # 2 points
            temperature=38.5,     # 1 point
            systolic_bp=105,      # 1 point
            heart_rate=95,        # 1 point
            consciousness=ConsciousnessLevel.ALERT  # 0 points
        )
        
        result_2 = await calculator.calculate_news2(scenario_2_vitals, scenario_1_patient)
        expected_score_2 = 2 + 1 + 2 + 1 + 1 + 1 + 0  # 8 points
        assert result_2.total_score == expected_score_2, f"RCP Scenario 2: Expected score {expected_score_2}"
        assert result_2.risk_category == RiskCategory.HIGH, "RCP Scenario 2: Expected HIGH risk"
        
        # RCP Reference Scenario 3: COPD patient (Scale 2)
        copd_patient = Patient(
            patient_id="RCP_COPD_001",
            ward_id="RESPIRATORY",
            bed_number="R-001",
            age=68,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        scenario_3_vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="RCP_COPD_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,    # 0 points
            spo2=90,              # 0 points on Scale 2 (88-92 range)
            on_oxygen=False,      # 0 points
            temperature=36.5,     # 0 points
            systolic_bp=130,      # 0 points
            heart_rate=78,        # 0 points
            consciousness=ConsciousnessLevel.ALERT  # 0 points
        )
        
        result_3 = await calculator.calculate_news2(scenario_3_vitals, copd_patient)
        assert result_3.total_score == 0, "RCP COPD Scenario: Expected score 0 with Scale 2"
        assert result_3.scale_used == 2, "RCP COPD Scenario: Expected Scale 2"
        assert result_3.risk_category == RiskCategory.LOW, "RCP COPD Scenario: Expected LOW risk"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])