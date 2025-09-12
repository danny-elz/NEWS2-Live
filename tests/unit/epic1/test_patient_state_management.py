#!/usr/bin/env python3
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.models.patient import Patient
from src.models.patient_state import (
    PatientState, PatientContext, TrendingAnalysis,
    PatientStateError, PatientTransferError, ConcurrentUpdateError
)
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.audit import AuditLogger
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker, VitalSignsWindow, TrendingResult
from src.services.patient_context_manager import PatientContextManager, AgeRiskProfile
from src.services.concurrent_update_manager import ConcurrentUpdateManager, RetryConfig
from src.services.vital_signs_history import VitalSignsHistory, HistoricalVitalSigns
from src.services.patient_transfer_service import PatientTransferService, TransferPriority


class TestPatientState:
    """Test PatientState data model."""
    
    def test_patient_state_from_patient(self):
        """Test creating PatientState from Patient model."""
        patient = Patient(
            patient_id="TEST_001",
            ward_id="WARD_A",
            bed_number="001",
            age=65,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            is_palliative=False,
            do_not_escalate=True
        )
        
        state = PatientState.from_patient(patient)
        
        assert state.patient_id == "TEST_001"
        assert state.current_ward_id == "WARD_A"
        assert state.clinical_flags['is_copd_patient'] == True
        assert state.clinical_flags['do_not_escalate'] == True
        assert state.state_version == 0
        assert isinstance(state.context, PatientContext)
        assert isinstance(state.trending_data, TrendingAnalysis)
    
    def test_patient_state_validation(self):
        """Test PatientState field validation."""
        with pytest.raises(ValueError, match="patient_id is required"):
            PatientState(
                patient_id="",
                current_ward_id="WARD_A",
                bed_number="001",
                clinical_flags={},
                assigned_nurse_id="NURSE_001",
                admission_date=datetime.now(timezone.utc),
                last_transfer_date=None,
                context=PatientContext(),
                trending_data=TrendingAnalysis(),
                state_version=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    def test_patient_state_to_dict(self):
        """Test PatientState serialization."""
        state = PatientState(
            patient_id="TEST_001",
            current_ward_id="WARD_A",
            bed_number="001",
            clinical_flags={'is_copd_patient': True},
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_transfer_date=None,
            context=PatientContext(allergies=['penicillin']),
            trending_data=TrendingAnalysis(current_score=3),
            state_version=1,
            last_updated=datetime.now(timezone.utc)
        )
        
        state_dict = state.to_dict()
        
        assert state_dict['patient_id'] == "TEST_001"
        assert state_dict['context']['allergies'] == ['penicillin']
        assert state_dict['trending_data']['current_score'] == 3
        assert state_dict['state_version'] == 1
    
    def test_patient_state_from_dict(self):
        """Test PatientState deserialization."""
        state_dict = {
            'patient_id': 'TEST_001',
            'current_ward_id': 'WARD_A',
            'bed_number': '001',
            'clinical_flags': {'is_copd_patient': True},
            'assigned_nurse_id': 'NURSE_001',
            'admission_date': '2024-01-01T12:00:00+00:00',
            'last_transfer_date': None,
            'context': {'allergies': ['penicillin']},
            'trending_data': {'current_score': 3},
            'state_version': 1,
            'last_updated': '2024-01-01T12:00:00+00:00'
        }
        
        state = PatientState.from_dict(state_dict)
        
        assert state.patient_id == "TEST_001"
        assert state.context.allergies == ['penicillin']
        assert state.trending_data.current_score == 3


class TestPatientRegistry:
    """Test PatientRegistry service."""
    
    @pytest.fixture
    def audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def patient_cache(self):
        cache = Mock()
        cache.get = AsyncMock()
        cache.put = AsyncMock()
        return cache
    
    @pytest.fixture
    def registry(self, audit_logger, patient_cache):
        return PatientRegistry(audit_logger, patient_cache)
    
    @pytest.fixture
    def sample_patient(self):
        return Patient(
            patient_id="TEST_001",
            ward_id="WARD_A",
            bed_number="001",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_register_patient(self, registry, sample_patient, audit_logger):
        """Test patient registration."""
        audit_logger.log_operation = AsyncMock()
        
        with patch.object(registry, 'get_patient_state', return_value=None):
            state = await registry.register_patient(sample_patient)
            
            assert state.patient_id == "TEST_001"
            assert state.current_ward_id == "WARD_A"
            assert state.state_version == 0
            audit_logger.log_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_existing_patient_error(self, registry, sample_patient):
        """Test error when registering existing patient."""
        existing_state = PatientState.from_patient(sample_patient)
        
        with patch.object(registry, 'get_patient_state', return_value=existing_state):
            with pytest.raises(PatientStateError, match="already registered"):
                await registry.register_patient(sample_patient)
    
    @pytest.mark.asyncio
    async def test_update_patient_state(self, registry, audit_logger):
        """Test patient state updates with optimistic locking."""
        audit_logger.log_operation = AsyncMock()
        current_state = PatientState(
            patient_id="TEST_001",
            current_ward_id="WARD_A",
            bed_number="001",
            clinical_flags={},
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_transfer_date=None,
            context=PatientContext(),
            trending_data=TrendingAnalysis(),
            state_version=1,
            last_updated=datetime.now(timezone.utc)
        )
        
        with patch.object(registry, 'get_patient_state', return_value=current_state):
            updates = {'current_ward_id': 'WARD_B', 'bed_number': '002'}
            
            updated_state = await registry.update_patient_state("TEST_001", updates, expected_version=1)
            
            assert updated_state.current_ward_id == "WARD_B"
            assert updated_state.bed_number == "002"
            assert updated_state.state_version == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_update_error(self, registry):
        """Test concurrent update detection."""
        current_state = PatientState(
            patient_id="TEST_001",
            current_ward_id="WARD_A",
            bed_number="001",
            clinical_flags={},
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_transfer_date=None,
            context=PatientContext(),
            trending_data=TrendingAnalysis(),
            state_version=2,  # Different version
            last_updated=datetime.now(timezone.utc)
        )
        
        with patch.object(registry, 'get_patient_state', return_value=current_state):
            with pytest.raises(ConcurrentUpdateError, match="State version mismatch"):
                await registry.update_patient_state("TEST_001", {}, expected_version=1)
    
    @pytest.mark.asyncio
    async def test_transfer_patient(self, registry, audit_logger):
        """Test patient transfer workflow."""
        audit_logger.log_operation = AsyncMock()
        current_state = PatientState(
            patient_id="TEST_001",
            current_ward_id="WARD_A",
            bed_number="001",
            clinical_flags={},
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_transfer_date=None,
            context=PatientContext(),
            trending_data=TrendingAnalysis(),
            state_version=1,
            last_updated=datetime.now(timezone.utc)
        )
        
        with patch.object(registry, 'get_patient_state', return_value=current_state):
            with patch.object(registry, 'update_patient_state') as mock_update:
                mock_update.return_value = current_state
                
                result = await registry.transfer_patient(
                    "TEST_001", "WARD_B", "routine_transfer"
                )
                
                mock_update.assert_called_once()
                assert audit_logger.log_operation.call_count >= 2  # Transfer + update logs
    
    @pytest.mark.asyncio
    async def test_transfer_do_not_escalate_patient(self, registry):
        """Test transfer validation for do_not_escalate patients."""
        current_state = PatientState(
            patient_id="TEST_001",
            current_ward_id="WARD_A",
            bed_number="001",
            clinical_flags={'do_not_escalate': True},
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_transfer_date=None,
            context=PatientContext(),
            trending_data=TrendingAnalysis(),
            state_version=1,
            last_updated=datetime.now(timezone.utc)
        )
        
        with patch.object(registry, 'get_patient_state', return_value=current_state):
            with pytest.raises(PatientTransferError, match="do_not_escalate flag"):
                await registry.transfer_patient("TEST_001", "WARD_B", "transfer")


class TestPatientStateTracker:
    """Test PatientStateTracker service."""
    
    @pytest.fixture
    def audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def tracker(self, audit_logger):
        return PatientStateTracker(audit_logger)
    
    @pytest.fixture
    def sample_vital_signs_history(self):
        """Create sample vital signs history for testing."""
        history = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=12)
        
        for i in range(12):  # 12 hours of hourly readings
            vitals = VitalSigns(
                event_id=uuid4(),
                patient_id="TEST_001",
                timestamp=base_time + timedelta(hours=i),
                respiratory_rate=18 + (i % 3),  # Slight variation
                spo2=96 + (i % 3),
                on_oxygen=False,
                temperature=36.5 + (i * 0.1),
                systolic_bp=120 + (i * 2),
                heart_rate=75 + (i * 3),
                consciousness=ConsciousnessLevel.ALERT
            )
            
            history.append(VitalSignsWindow(
                timestamp=vitals.timestamp,
                vital_signs=vitals,
                news2_score=2 + (i % 3)  # Scores between 2-4
            ))
        
        return history
    
    @pytest.mark.asyncio
    async def test_calculate_24h_trends(self, tracker, sample_vital_signs_history, audit_logger):
        """Test 24-hour trending analysis."""
        audit_logger.log_operation = AsyncMock()
        
        result = await tracker.calculate_24h_trends("TEST_001", sample_vital_signs_history)
        
        assert isinstance(result, TrendingResult)
        assert result.patient_id == "TEST_001"
        assert 'heart_rate' in result.rolling_stats
        assert 'news2_score' in result.rolling_stats
        assert result.deterioration_risk in ['LOW', 'MEDIUM', 'HIGH']
        assert isinstance(result.confidence_score, float)
        assert 0.0 <= result.confidence_score <= 1.0
        audit_logger.log_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_empty_history_error(self, tracker):
        """Test error handling for empty vital signs history."""
        with pytest.raises(Exception):  # TrendingCalculationError
            await tracker.calculate_24h_trends("TEST_001", [])
    
    def test_calculate_rolling_statistics(self, tracker, sample_vital_signs_history):
        """Test rolling window statistics calculation."""
        stats = tracker._calculate_rolling_statistics(sample_vital_signs_history)
        
        assert 'heart_rate' in stats
        assert 'respiratory_rate' in stats
        assert 'news2_score' in stats
        
        hr_stats = stats['heart_rate']
        assert 'min' in hr_stats
        assert 'max' in hr_stats
        assert 'avg' in hr_stats
        assert 'std' in hr_stats
        assert hr_stats['min'] <= hr_stats['avg'] <= hr_stats['max']
    
    def test_calculate_trend_slope(self, tracker, sample_vital_signs_history):
        """Test NEWS2 trend slope calculation."""
        slope = tracker._calculate_trend_slope(sample_vital_signs_history)
        
        assert isinstance(slope, float)
        # With our test data (scores 2-4 cyclical), slope should be small
        assert -1.0 <= slope <= 1.0
    
    def test_assess_deterioration_risk(self, tracker):
        """Test deterioration risk assessment."""
        # High risk scenario
        high_risk_data = [
            VitalSignsWindow(
                timestamp=datetime.now(timezone.utc),
                vital_signs=Mock(),
                news2_score=8  # Critical score
            )
        ]
        
        risk = tracker._assess_deterioration_risk(high_risk_data, 0.1)
        assert risk == "HIGH"
        
        # Low risk scenario
        low_risk_data = [
            VitalSignsWindow(
                timestamp=datetime.now(timezone.utc),
                vital_signs=Mock(),
                news2_score=1  # Low score
            )
        ]
        
        risk = tracker._assess_deterioration_risk(low_risk_data, 0.0)
        assert risk == "LOW"


class TestPatientContextManager:
    """Test PatientContextManager service."""
    
    @pytest.fixture
    def audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def context_manager(self, audit_logger):
        return PatientContextManager(audit_logger)
    
    @pytest.fixture
    def sample_patient(self):
        return Patient(
            patient_id="TEST_001",
            ward_id="WARD_A",
            bed_number="001",
            age=75,  # Elderly patient
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_create_patient_context(self, context_manager, audit_logger):
        """Test patient context creation."""
        audit_logger.log_operation = AsyncMock()
        
        context_data = {
            'allergies': ['penicillin', 'shellfish'],
            'medications': ['aspirin', 'metformin'],
            'comorbidities': ['diabetes', 'hypertension'],
            'medical_history': 'Previous MI in 2020'
        }
        
        context = await context_manager.create_patient_context("TEST_001", context_data)
        
        assert context.allergies == ['penicillin', 'shellfish']
        assert context.medications == ['aspirin', 'metformin']
        assert context.medical_history == 'Previous MI in 2020'
        audit_logger.log_operation.assert_called_once()
    
    def test_calculate_age_risk_profile(self, context_manager, sample_patient):
        """Test age-based risk factor calculation."""
        profile = context_manager.calculate_age_risk_profile(sample_patient)
        
        assert isinstance(profile, AgeRiskProfile)
        assert profile.age == 75
        assert profile.risk_category == "GERIATRIC"
        assert profile.monitoring_frequency_modifier > 1.0  # More frequent monitoring
        assert len(profile.special_considerations) > 0
    
    def test_pediatric_risk_profile(self, context_manager):
        """Test pediatric risk profile calculation."""
        pediatric_patient = Patient(
            patient_id="CHILD_001",
            ward_id="PEDIATRIC",
            bed_number="001",
            age=10,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        profile = context_manager.calculate_age_risk_profile(pediatric_patient)
        
        assert profile.risk_category == "PEDIATRIC"
        assert profile.monitoring_frequency_modifier > 1.0
        assert "Pediatric vital sign ranges" in str(profile.special_considerations)
    
    def test_calculate_news2_adjustments(self, context_manager, sample_patient):
        """Test context-aware NEWS2 adjustments."""
        context = PatientContext(
            allergies=['penicillin'],
            medications=['beta-blocker', 'steroid'],
            comorbidities=['heart_failure', 'diabetes']
        )
        
        adjustments = context_manager.calculate_news2_adjustments(sample_patient, context)
        
        assert 'baseline_adjustment' in adjustments
        assert 'threshold_modifications' in adjustments
        assert 'monitoring_adjustments' in adjustments
        assert 'special_considerations' in adjustments
        
        # Beta-blockers should trigger heart rate masking warning
        considerations = adjustments['special_considerations']
        assert any('beta-blocker' in str(c).lower() for c in considerations)
        assert any('steroid' in str(c).lower() for c in considerations)
    
    def test_validate_context_completeness(self, context_manager):
        """Test context completeness validation."""
        # Complete context
        complete_context = PatientContext(
            allergies=['penicillin'],
            medications=['aspirin'],
            comorbidities=['diabetes'],
            medical_history='Complete history',
            special_instructions='Monitor blood sugar'
        )
        
        result = context_manager.validate_context_completeness(complete_context)
        
        assert result['completeness_score'] == 1.0
        assert result['is_sufficient_for_clinical_decision'] == True
        assert len(result['missing_elements']) == 0
        
        # Incomplete context
        incomplete_context = PatientContext()
        
        result = context_manager.validate_context_completeness(incomplete_context)
        
        assert result['completeness_score'] == 0.0
        assert result['is_sufficient_for_clinical_decision'] == False
        assert len(result['missing_elements']) > 0


class TestConcurrentUpdateManager:
    """Test ConcurrentUpdateManager service."""
    
    @pytest.fixture
    def audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def update_manager(self, audit_logger):
        return ConcurrentUpdateManager(audit_logger)
    
    @pytest.mark.asyncio
    async def test_distributed_lock(self, update_manager, audit_logger):
        """Test distributed locking mechanism."""
        audit_logger.log_operation = AsyncMock()
        
        async with update_manager.distributed_lock("TEST_001", "test_operation") as lock_id:
            assert isinstance(lock_id, str)
            
            # Verify lock is held
            lock_status = update_manager.get_lock_status("TEST_001")
            assert lock_status is not None
            assert lock_status['operation_type'] == "test_operation"
            assert not lock_status['is_expired']
        
        # Verify lock is released
        lock_status = update_manager.get_lock_status("TEST_001")
        assert lock_status is None
    
    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self, update_manager, audit_logger):
        """Test that concurrent lock acquisition fails properly."""
        audit_logger.log_operation = AsyncMock()
        
        async with update_manager.distributed_lock("TEST_001", "operation_1"):
            # Try to acquire same patient lock concurrently
            with pytest.raises(ConcurrentUpdateError, match="Failed to acquire lock"):
                async with update_manager.distributed_lock("TEST_001", "operation_2", timeout_seconds=1):
                    pass
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, update_manager, audit_logger):
        """Test retry mechanism with exponential backoff."""
        audit_logger.log_operation = AsyncMock()
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        
        result = await update_manager.retry_with_backoff(
            failing_operation, "TEST_001", "test_operation", retry_config
        )
        
        assert result == "success"
        assert call_count == 3
        assert audit_logger.log_operation.call_count >= 2  # Retry attempts + success
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, update_manager, audit_logger):
        """Test behavior when retries are exhausted."""
        audit_logger.log_operation = AsyncMock()
        
        async def always_failing_operation():
            raise ConnectionError("Persistent failure")
        
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        
        with pytest.raises(ConnectionError, match="Persistent failure"):
            await update_manager.retry_with_backoff(
                always_failing_operation, "TEST_001", "test_operation", retry_config
            )
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_locks(self, update_manager, audit_logger):
        """Test cleanup of expired locks."""
        audit_logger.log_operation = AsyncMock()
        
        # Acquire lock with very short timeout
        async with update_manager.distributed_lock("TEST_001", "test", timeout_seconds=0.01):
            pass
        
        # Wait for expiration
        await asyncio.sleep(0.02)
        
        # Cleanup should remove expired lock
        await update_manager.cleanup_expired_locks()
        
        lock_status = update_manager.get_lock_status("TEST_001")
        assert lock_status is None


class TestVitalSignsHistory:
    """Test VitalSignsHistory service."""
    
    @pytest.fixture
    def audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def history_service(self, audit_logger):
        return VitalSignsHistory(audit_logger)
    
    @pytest.fixture
    def sample_vitals(self):
        return VitalSigns(
            event_id=uuid4(),
            patient_id="TEST_001",
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
    async def test_store_vital_signs(self, history_service, sample_vitals, audit_logger):
        """Test storing vital signs with historical preservation."""
        audit_logger.log_operation = AsyncMock()
        
        record_id = await history_service.store_vital_signs(
            "TEST_001", sample_vitals, data_source="device"
        )
        
        assert isinstance(record_id, str)
        assert len(record_id) > 0
        audit_logger.log_operation.assert_called_once()
    
    def test_calculate_quality_score(self, history_service, sample_vitals):
        """Test data quality score calculation."""
        # Complete vital signs
        quality_score = history_service._calculate_quality_score(sample_vitals)
        assert quality_score == 1.0
        
        # Incomplete vital signs
        incomplete_vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="TEST_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=96,
            on_oxygen=None,  # Missing
            temperature=None,  # Missing
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        quality_score = history_service._calculate_quality_score(incomplete_vitals)
        assert 0.0 < quality_score < 1.0
    
    @pytest.mark.asyncio
    async def test_30day_retention_policy(self, history_service, audit_logger):
        """Test 30-day retention policy implementation."""
        audit_logger.log_operation = AsyncMock()
        
        result = await history_service.implement_30day_retention("TEST_001")
        
        assert 'total_processed' in result
        assert 'archived_count' in result
        assert 'retained_count' in result
        audit_logger.log_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_integrity_check(self, history_service, audit_logger):
        """Test data integrity verification."""
        audit_logger.log_operation = AsyncMock()
        
        integrity_result = await history_service.verify_data_integrity("TEST_001")
        
        assert hasattr(integrity_result, 'patient_id')
        assert hasattr(integrity_result, 'overall_integrity')
        assert hasattr(integrity_result, 'records_checked')
        assert isinstance(integrity_result.integrity_violations, list)
        assert isinstance(integrity_result.missing_records, list)
        audit_logger.log_operation.assert_called_once()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not slow"  # Exclude slow tests for quick runs
    ])