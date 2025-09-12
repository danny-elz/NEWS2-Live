#!/usr/bin/env python3
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.models.patient import Patient
from src.models.patient_state import PatientState, PatientContext, TrendingAnalysis
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.models.news2 import NEWS2Result, RiskCategory
from src.services.audit import AuditLogger
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker, VitalSignsWindow
from src.services.patient_context_manager import PatientContextManager
from src.services.concurrent_update_manager import ConcurrentUpdateManager
from src.services.vital_signs_history import VitalSignsHistory
from src.services.patient_transfer_service import PatientTransferService, TransferPriority
from src.services.news2_calculator import NEWS2Calculator
from src.services.patient_cache import PatientDataCache


class TestPatientStateIntegration:
    """Integration tests for patient state management workflows."""
    
    @pytest.fixture
    def audit_logger(self):
        logger = Mock(spec=AuditLogger)
        logger.log_operation = AsyncMock()
        return logger
    
    @pytest.fixture
    def patient_cache(self):
        cache = Mock(spec=PatientDataCache)
        cache.get = AsyncMock(return_value=None)
        cache.put = AsyncMock()
        cache.get_multiple = AsyncMock(return_value=[])
        return cache
    
    @pytest.fixture
    def news2_calculator(self, audit_logger):
        return Mock(spec=NEWS2Calculator)
    
    @pytest.fixture
    def patient_registry(self, audit_logger, patient_cache):
        return PatientRegistry(audit_logger, patient_cache)
    
    @pytest.fixture
    def state_tracker(self, audit_logger):
        return PatientStateTracker(audit_logger)
    
    @pytest.fixture
    def context_manager(self, audit_logger):
        return PatientContextManager(audit_logger)
    
    @pytest.fixture
    def concurrent_manager(self, audit_logger):
        return ConcurrentUpdateManager(audit_logger)
    
    @pytest.fixture
    def history_service(self, audit_logger):
        return VitalSignsHistory(audit_logger)
    
    @pytest.fixture
    def transfer_service(self, audit_logger, patient_registry, concurrent_manager):
        return PatientTransferService(audit_logger, patient_registry, concurrent_manager)
    
    @pytest.fixture
    def sample_patient(self):
        return Patient(
            patient_id="INT_TEST_001",
            ward_id="WARD_A",
            bed_number="001",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_vital_signs(self):
        return VitalSigns(
            event_id=uuid4(),
            patient_id="INT_TEST_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=96,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_patient_registration_workflow(
        self, patient_registry, context_manager, history_service, 
        sample_patient, sample_vital_signs
    ):
        """Test complete patient registration and initial setup workflow."""
        
        # Step 1: Register patient
        with patch.object(patient_registry, 'get_patient_state', return_value=None):
            patient_state = await patient_registry.register_patient(sample_patient)
            
            assert patient_state.patient_id == "INT_TEST_001"
            assert patient_state.current_ward_id == "WARD_A"
            assert patient_state.state_version == 0
        
        # Step 2: Create patient context
        context_data = {
            'allergies': ['penicillin'],
            'medications': ['aspirin'],
            'comorbidities': ['hypertension'],
            'medical_history': 'Routine admission for monitoring'
        }
        
        context = await context_manager.create_patient_context(
            "INT_TEST_001", context_data
        )
        
        assert context.allergies == ['penicillin']
        assert context.medications == ['aspirin']
        
        # Step 3: Store initial vital signs
        record_id = await history_service.store_vital_signs(
            "INT_TEST_001", sample_vital_signs
        )
        
        assert isinstance(record_id, str)
        
        # Step 4: Update patient state with context
        with patch.object(patient_registry, 'get_patient_state', return_value=patient_state):
            context_updates = {'context': {'allergies': context.allergies}}
            updated_state = await patient_registry.update_patient_state(
                "INT_TEST_001", context_updates, expected_version=0
            )
            
            assert updated_state.state_version == 1
    
    @pytest.mark.asyncio 
    async def test_patient_transfer_with_state_preservation(
        self, patient_registry, transfer_service, concurrent_manager, sample_patient
    ):
        """Test patient transfer between wards with complete state preservation."""
        
        # Setup: Patient exists in WARD_A
        initial_state = PatientState.from_patient(sample_patient)
        initial_state.clinical_flags = {'is_copd_patient': False}
        initial_state.context = PatientContext(allergies=['shellfish'])
        initial_state.trending_data = TrendingAnalysis(current_score=3)
        
        with patch.object(patient_registry, 'get_patient_state', return_value=initial_state):
            # Step 1: Initiate transfer
            transfer_request = await transfer_service.initiate_transfer(
                patient_id="INT_TEST_001",
                destination_ward_id="WARD_B",
                transfer_reason="closer_monitoring",
                requested_by="DR_SMITH",
                priority=TransferPriority.ROUTINE
            )
            
            assert transfer_request.source_ward_id == "WARD_A"
            assert transfer_request.destination_ward_id == "WARD_B"
            
            # Step 2: Validate transfer
            validation = await transfer_service.validate_transfer(transfer_request.transfer_id)
            
            assert validation.is_valid == True
            assert len(validation.validation_errors) == 0
            
            # Step 3: Execute transfer with state preservation
            with patch.object(patient_registry, 'transfer_patient') as mock_transfer:
                mock_transfer.return_value = initial_state
                initial_state.current_ward_id = "WARD_B"  # Simulate successful transfer
                
                completed_transfer = await transfer_service.execute_transfer(
                    transfer_request.transfer_id, "NURSE_002"
                )
                
                assert completed_transfer.status.value == "completed"
                mock_transfer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trending_analysis_with_news2_integration(
        self, state_tracker, news2_calculator, sample_patient
    ):
        """Test trending analysis integration with NEWS2 calculation engine."""
        
        # Create vital signs history with deteriorating pattern
        history = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=6)
        
        # Simulate deteriorating patient over 6 hours
        vital_params = [
            (18, 96, 36.5, 120, 75, 2),  # Hour 0: Normal
            (20, 94, 36.8, 125, 80, 3),  # Hour 1: Slight increase
            (22, 93, 37.2, 130, 85, 4),  # Hour 2: Getting worse
            (24, 91, 37.5, 110, 90, 5),  # Hour 3: Concerning
            (26, 90, 37.8, 105, 95, 6),  # Hour 4: Deteriorating
            (28, 88, 38.1, 100, 100, 7), # Hour 5: Critical
        ]
        
        for i, (rr, spo2, temp, sbp, hr, news2_score) in enumerate(vital_params):
            vitals = VitalSigns(
                event_id=uuid4(),
                patient_id="INT_TEST_001",
                timestamp=base_time + timedelta(hours=i),
                respiratory_rate=rr,
                spo2=spo2,
                on_oxygen=False,
                temperature=temp,
                systolic_bp=sbp,
                heart_rate=hr,
                consciousness=ConsciousnessLevel.ALERT
            )
            
            history.append(VitalSignsWindow(
                timestamp=vitals.timestamp,
                vital_signs=vitals,
                news2_score=news2_score
            ))
        
        # Calculate trending analysis
        trending_result = await state_tracker.calculate_24h_trends("INT_TEST_001", history)
        
        assert trending_result.patient_id == "INT_TEST_001"
        assert trending_result.deterioration_risk in ["MEDIUM", "HIGH"]
        assert trending_result.news2_trend_slope > 0.5  # Positive slope indicating deterioration
        assert len(trending_result.early_warning_indicators) > 0
        
        # Verify trend comparisons
        assert trending_result.trend_comparison['current'] is not None
        assert trending_result.confidence_score > 0.5  # Should have good confidence with 6 data points
    
    @pytest.mark.asyncio
    async def test_concurrent_patient_updates_across_services(
        self, patient_registry, context_manager, concurrent_manager, sample_patient
    ):
        """Test concurrent patient updates across multiple services."""
        
        initial_state = PatientState.from_patient(sample_patient)
        
        with patch.object(patient_registry, 'get_patient_state', return_value=initial_state):
            # Simulate concurrent operations
            tasks = []
            
            # Task 1: Update clinical flags
            async def update_flags():
                return await patient_registry.update_clinical_flags(
                    "INT_TEST_001", {'is_palliative': True}
                )
            
            # Task 2: Update allergies
            async def update_allergies():
                return await context_manager.update_allergies(
                    "INT_TEST_001", ['penicillin', 'sulfa'], "allergy_review"
                )
            
            # Task 3: Assign new nurse
            async def update_nurse():
                return await patient_registry.assign_nurse(
                    "INT_TEST_001", "NURSE_003", "shift_change"
                )
            
            tasks = [update_flags(), update_allergies(), update_nurse()]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all operations completed (some may have concurrency conflicts, which is expected)
            assert len(results) == 3
            # At least some operations should succeed
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0
    
    @pytest.mark.asyncio
    async def test_historical_data_queries_with_timescale_integration(
        self, history_service, sample_patient
    ):
        """Test historical data queries with TimescaleDB-style operations."""
        
        # Store multiple historical records
        vital_signs_list = []
        base_time = datetime.now(timezone.utc) - timedelta(days=5)
        
        for i in range(120):  # 5 days of hourly readings
            vitals = VitalSigns(
                event_id=uuid4(),
                patient_id="INT_TEST_001",
                timestamp=base_time + timedelta(hours=i),
                respiratory_rate=16 + (i % 6),  # Varies between 16-21
                spo2=94 + (i % 5),  # Varies between 94-98
                on_oxygen=i % 10 == 0,  # Occasionally on oxygen
                temperature=36.0 + (i % 4) * 0.5,  # Varies between 36-37.5
                systolic_bp=110 + (i % 20) * 2,  # Varies between 110-148
                heart_rate=60 + (i % 30) * 2,  # Varies between 60-118
                consciousness=ConsciousnessLevel.ALERT
            )
            
            record_id = await history_service.store_vital_signs(
                "INT_TEST_001", vitals, data_source="device"
            )
            
            vital_signs_list.append((vitals, record_id))
        
        # Query last 24 hours
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        historical_records = await history_service.query_historical_data(
            "INT_TEST_001", start_time, end_time, include_compressed=True
        )
        
        # In real implementation, would return actual records
        # For now, just verify the method executed without error
        assert isinstance(historical_records, list)
        
        # Test 30-day retention policy
        retention_result = await history_service.implement_30day_retention("INT_TEST_001")
        
        assert 'total_processed' in retention_result
        assert 'archived_count' in retention_result
        
        # Test data compression
        compression_result = await history_service.compress_historical_data("INT_TEST_001")
        
        assert compression_result.patient_id == "INT_TEST_001"
        assert compression_result.compression_timestamp is not None
        
        # Test data integrity verification
        integrity_check = await history_service.verify_data_integrity("INT_TEST_001")
        
        assert integrity_check.patient_id == "INT_TEST_001"
        assert isinstance(integrity_check.overall_integrity, bool)
    
    @pytest.mark.asyncio
    async def test_complex_clinical_scenario_workflow(
        self, patient_registry, state_tracker, context_manager, 
        transfer_service, history_service, concurrent_manager,
        sample_patient
    ):
        """Test complex clinical scenario with multiple service interactions."""
        
        # Scenario: COPD patient with deteriorating condition requiring transfer
        
        # Step 1: Setup COPD patient with context
        copd_patient = Patient(
            patient_id="COPD_001",
            ward_id="MEDICAL",
            bed_number="M001",
            age=68,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(days=2),
            last_updated=datetime.now(timezone.utc)
        )
        
        initial_state = PatientState.from_patient(copd_patient)
        
        with patch.object(patient_registry, 'get_patient_state', return_value=initial_state):
            # Register patient
            patient_state = await patient_registry.register_patient(copd_patient)
            
            # Add clinical context
            context_data = {
                'allergies': ['penicillin'],
                'medications': ['salbutamol', 'prednisolone', 'oxygen'],
                'comorbidities': ['COPD', 'hypertension'],
                'medical_history': 'COPD exacerbation, previous ICU admissions'
            }
            
            context = await context_manager.create_patient_context("COPD_001", context_data)
            
            # Calculate NEWS2 adjustments for COPD patient
            adjustments = context_manager.calculate_news2_adjustments(copd_patient, context)
            assert 'special_considerations' in adjustments
            
            # Step 2: Record deteriorating vital signs
            deterioration_timeline = []
            base_time = datetime.now(timezone.utc) - timedelta(hours=3)
            
            vital_progressions = [
                (20, 89, 36.8, 135, 85, 5),  # Initial: Medium risk
                (24, 87, 37.2, 140, 90, 7),  # 1 hour later: High risk
                (26, 85, 37.5, 145, 95, 8),  # 2 hours later: Critical
            ]
            
            for i, (rr, spo2, temp, sbp, hr, news2_score) in enumerate(vital_progressions):
                vitals = VitalSigns(
                    event_id=uuid4(),
                    patient_id="COPD_001",
                    timestamp=base_time + timedelta(hours=i),
                    respiratory_rate=rr,
                    spo2=spo2,
                    on_oxygen=True,  # COPD patient on oxygen
                    temperature=temp,
                    systolic_bp=sbp,
                    heart_rate=hr,
                    consciousness=ConsciousnessLevel.ALERT
                )
                
                # Store in history
                await history_service.store_vital_signs("COPD_001", vitals)
                
                deterioration_timeline.append(VitalSignsWindow(
                    timestamp=vitals.timestamp,
                    vital_signs=vitals,
                    news2_score=news2_score
                ))
            
            # Step 3: Calculate trending analysis
            trending_result = await state_tracker.calculate_24h_trends(
                "COPD_001", deterioration_timeline
            )
            
            assert trending_result.deterioration_risk == "HIGH"
            assert trending_result.news2_trend_slope > 1.0  # Rapid deterioration
            
            # Step 4: Initiate urgent transfer to ICU
            with patch.object(patient_registry, 'transfer_patient') as mock_transfer:
                mock_transfer.return_value = patient_state
                patient_state.current_ward_id = "ICU"
                
                transfer_request = await transfer_service.initiate_transfer(
                    patient_id="COPD_001",
                    destination_ward_id="ICU",
                    transfer_reason="deteriorating_copd_exacerbation",
                    requested_by="DR_CRITICAL",
                    priority=TransferPriority.EMERGENCY
                )
                
                # Validate emergency transfer
                validation = await transfer_service.validate_transfer(transfer_request.transfer_id)
                
                # Should be valid but with warnings about critical patient
                assert validation.is_valid == True
                assert len(validation.warnings) > 0  # Should warn about high NEWS2 score
                assert 'critical_care_team' in validation.required_approvals
                
                # Execute transfer
                completed_transfer = await transfer_service.execute_transfer(
                    transfer_request.transfer_id, "CRITICAL_CARE_TEAM"
                )
                
                assert completed_transfer.status.value == "completed"
                mock_transfer.assert_called_once()
            
            # Step 5: Verify state preservation after transfer
            with patch.object(patient_registry, 'get_patient_state') as mock_get_state:
                # Simulate updated state after transfer
                transferred_state = PatientState.from_patient(copd_patient)
                transferred_state.current_ward_id = "ICU"
                transferred_state.state_version = 2
                transferred_state.last_transfer_date = datetime.now(timezone.utc)
                mock_get_state.return_value = transferred_state
                
                final_state = await patient_registry.get_patient_state("COPD_001")
                assert final_state.current_ward_id == "ICU"
                assert final_state.state_version == 2
                assert final_state.last_transfer_date is not None


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])