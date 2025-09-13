"""
Unit Tests for Alert Generation Engine - Story 2.1

Tests cover:
- Critical alert generation within 5 seconds
- Single parameter score = 3 bypass logic
- Ward-specific threshold configuration
- Alert deduplication and correlation
- Performance requirements validation
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.models.alerts import (
    AlertLevel, AlertPriority, AlertDecision, AlertThreshold, Alert, AlertStatus
)
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_generation import (
    NEWS2ScoreEvaluator, AlertDecisionEngine, AlertGenerator, DefaultThresholdManager
)
from src.services.audit import AuditLogger


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    mock_logger = Mock(spec=AuditLogger)
    mock_logger.create_audit_entry = Mock()
    return mock_logger


@pytest.fixture
def news2_evaluator(mock_audit_logger):
    """NEWS2 score evaluator instance for testing."""
    return NEWS2ScoreEvaluator(mock_audit_logger)


@pytest.fixture
def alert_decision_engine(mock_audit_logger):
    """Alert decision engine instance for testing."""
    return AlertDecisionEngine(mock_audit_logger)


@pytest.fixture
def alert_generator(mock_audit_logger):
    """Alert generator instance for testing."""
    return AlertGenerator(mock_audit_logger)


@pytest.fixture
def sample_patient():
    """Sample patient for testing."""
    return Patient(
        patient_id="TEST_PATIENT_001",
        ward_id="WARD_A",
        bed_number="A-101",
        age=67,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def copd_patient():
    """COPD patient for testing."""
    return Patient(
        patient_id="TEST_COPD_001",
        ward_id="WARD_B",
        bed_number="B-205",
        age=72,
        is_copd_patient=True,
        assigned_nurse_id="NURSE_002",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def critical_news2_result():
    """Critical NEWS2 result (score ≥7)."""
    return NEWS2Result(
        total_score=8,
        individual_scores={
            "respiratory_rate": 2,
            "spo2": 2,
            "temperature": 1,
            "systolic_bp": 2,
            "heart_rate": 1,
            "consciousness": 0
        },
        risk_category=RiskCategory.HIGH,
        monitoring_frequency="continuous",
        scale_used=1,
        warnings=[],
        confidence=1.0,
        calculated_at=datetime.now(timezone.utc),
        calculation_time_ms=3.5
    )


@pytest.fixture
def single_param_critical_news2():
    """NEWS2 result with single parameter score = 3."""
    return NEWS2Result(
        total_score=5,
        individual_scores={
            "respiratory_rate": 3,  # Critical parameter
            "spo2": 1,
            "temperature": 0,
            "systolic_bp": 1,
            "heart_rate": 0,
            "consciousness": 0
        },
        risk_category=RiskCategory.MEDIUM,
        monitoring_frequency="6 hourly",
        scale_used=1,
        warnings=[],
        confidence=1.0,
        calculated_at=datetime.now(timezone.utc),
        calculation_time_ms=2.1
    )


@pytest.fixture
def default_ward_thresholds():
    """Default ward thresholds for testing."""
    manager = DefaultThresholdManager()
    return manager.get_default_thresholds("TEST_WARD")


class TestNEWS2ScoreEvaluator:
    """Test cases for NEWS2 score evaluation logic."""
    
    @pytest.mark.asyncio
    async def test_critical_total_score_evaluation(self, news2_evaluator, sample_patient, critical_news2_result, default_ward_thresholds):
        """Test that NEWS2 ≥7 generates critical alerts."""
        alert_level, priority, reasoning, single_param = news2_evaluator.evaluate_news2_score(
            critical_news2_result, sample_patient, default_ward_thresholds
        )
        
        assert alert_level == AlertLevel.CRITICAL
        assert priority == AlertPriority.IMMEDIATE
        assert "Critical NEWS2 total score 8 (≥7)" in reasoning
        assert single_param is False
    
    @pytest.mark.asyncio
    async def test_single_parameter_critical_evaluation(self, news2_evaluator, sample_patient, single_param_critical_news2, default_ward_thresholds):
        """Test that single parameter score = 3 bypasses all suppression rules."""
        alert_level, priority, reasoning, single_param = news2_evaluator.evaluate_news2_score(
            single_param_critical_news2, sample_patient, default_ward_thresholds
        )
        
        assert alert_level == AlertLevel.CRITICAL
        assert priority == AlertPriority.LIFE_THREATENING
        assert "CRITICAL: Respiratory Rate score = 3" in reasoning
        assert single_param is True
    
    @pytest.mark.asyncio
    async def test_multiple_critical_parameters(self, news2_evaluator, sample_patient, default_ward_thresholds):
        """Test handling of multiple critical parameters."""
        news2_result = NEWS2Result(
            total_score=6,
            individual_scores={
                "respiratory_rate": 3,  # Critical
                "spo2": 3,             # Critical
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.8
        )
        
        alert_level, priority, reasoning, single_param = news2_evaluator.evaluate_news2_score(
            news2_result, sample_patient, default_ward_thresholds
        )
        
        assert alert_level == AlertLevel.CRITICAL
        assert priority == AlertPriority.LIFE_THREATENING
        assert "Multiple parameters" in reasoning
        assert "Respiratory Rate, Spo2" in reasoning
        assert single_param is True
    
    @pytest.mark.asyncio
    async def test_ward_specific_threshold_application(self, news2_evaluator, sample_patient, default_ward_thresholds):
        """Test that ward-specific thresholds are correctly applied."""
        # Create NEWS2 result that matches medium threshold (3-4)
        news2_result = NEWS2Result(
            total_score=4,
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 1,
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="6 hourly",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.5
        )
        
        alert_level, priority, reasoning, single_param = news2_evaluator.evaluate_news2_score(
            news2_result, sample_patient, default_ward_thresholds
        )
        
        assert alert_level == AlertLevel.MEDIUM
        assert priority == AlertPriority.URGENT
        assert "NEWS2 score 4 matches medium threshold" in reasoning
        assert single_param is False
    
    @pytest.mark.asyncio
    async def test_low_score_evaluation(self, news2_evaluator, sample_patient, default_ward_thresholds):
        """Test evaluation of low NEWS2 scores."""
        news2_result = NEWS2Result(
            total_score=1,
            individual_scores={
                "respiratory_rate": 1,
                "spo2": 0,
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.LOW,
            monitoring_frequency="12 hourly",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.2
        )
        
        alert_level, priority, reasoning, single_param = news2_evaluator.evaluate_news2_score(
            news2_result, sample_patient, default_ward_thresholds
        )
        
        assert alert_level == AlertLevel.LOW
        assert priority == AlertPriority.ROUTINE
        assert single_param is False


class TestAlertDecisionEngine:
    """Test cases for alert decision making logic."""
    
    @pytest.mark.asyncio
    async def test_alert_decision_creation(self, alert_decision_engine, sample_patient, critical_news2_result, default_ward_thresholds):
        """Test creation of alert decisions with proper audit trail."""
        start_time = datetime.now(timezone.utc)
        
        decision = await alert_decision_engine.make_alert_decision(
            critical_news2_result, sample_patient, default_ward_thresholds, "TEST_USER"
        )
        
        end_time = datetime.now(timezone.utc)
        
        assert isinstance(decision, AlertDecision)
        assert decision.patient_id == sample_patient.patient_id
        assert decision.alert_level == AlertLevel.CRITICAL
        assert decision.generation_latency_ms > 0
        assert start_time <= decision.decision_timestamp <= end_time
        assert decision.ward_id == sample_patient.ward_id
    
    @pytest.mark.asyncio
    async def test_decision_latency_measurement(self, alert_decision_engine, sample_patient, critical_news2_result, default_ward_thresholds):
        """Test that decision latency is properly measured."""
        decision = await alert_decision_engine.make_alert_decision(
            critical_news2_result, sample_patient, default_ward_thresholds
        )
        
        # Decision should complete quickly (< 100ms for unit test)
        assert decision.generation_latency_ms < 100
        assert decision.generation_latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_suppression_bypass_for_critical_alerts(self, alert_decision_engine, sample_patient, single_param_critical_news2, default_ward_thresholds):
        """Test that critical alerts are never suppressed."""
        decision = await alert_decision_engine.make_alert_decision(
            single_param_critical_news2, sample_patient, default_ward_thresholds
        )
        
        assert decision.alert_level == AlertLevel.CRITICAL
        assert decision.suppressed is False
        assert decision.single_param_trigger is True
    
    @pytest.mark.asyncio
    async def test_error_handling_in_decision_making(self, alert_decision_engine, sample_patient, default_ward_thresholds):
        """Test error handling during decision making."""
        # Create invalid NEWS2 result
        invalid_news2 = NEWS2Result(
            total_score=-1,  # Invalid score
            individual_scores={},
            risk_category=RiskCategory.LOW,
            monitoring_frequency="routine",
            scale_used=1,
            warnings=[],
            confidence=0.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=0.0
        )
        
        # Test that system handles invalid input gracefully
        decision = await alert_decision_engine.make_alert_decision(
            invalid_news2, sample_patient, default_ward_thresholds
        )

        # Should still create a decision but may have warnings or special handling
        assert decision is not None
        assert decision.patient_id == sample_patient.patient_id


class TestAlertGenerator:
    """Test cases for complete alert generation workflow."""
    
    @pytest.mark.asyncio
    async def test_critical_alert_generation_speed(self, alert_generator, sample_patient, critical_news2_result):
        """Test that critical alerts are generated within 5 seconds."""
        start_time = datetime.now(timezone.utc)
        
        alert = await alert_generator.generate_alert(
            critical_news2_result, sample_patient, None, "TEST_USER"
        )
        
        end_time = datetime.now(timezone.utc)
        generation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        assert alert is not None
        assert alert.alert_level == AlertLevel.CRITICAL
        assert generation_time_ms < 5000  # Must be under 5 seconds
    
    @pytest.mark.asyncio
    async def test_single_parameter_alert_generation(self, alert_generator, sample_patient, single_param_critical_news2):
        """Test generation of single parameter critical alerts."""
        alert = await alert_generator.generate_alert(
            single_param_critical_news2, sample_patient
        )
        
        assert alert is not None
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.alert_priority == AlertPriority.LIFE_THREATENING
        assert alert.alert_decision.single_param_trigger is True
        assert "CRITICAL ALERT - Single Parameter Red Flag" in alert.title
    
    @pytest.mark.asyncio
    async def test_alert_content_generation(self, alert_generator, sample_patient, critical_news2_result):
        """Test that alert content is properly generated."""
        alert = await alert_generator.generate_alert(
            critical_news2_result, sample_patient
        )
        
        assert alert.title is not None
        assert alert.message is not None
        assert sample_patient.patient_id in alert.message
        assert str(critical_news2_result.total_score) in alert.message
        assert sample_patient.ward_id in alert.message
    
    @pytest.mark.asyncio
    async def test_alert_clinical_context(self, alert_generator, sample_patient, critical_news2_result):
        """Test that alerts include proper clinical context."""
        alert = await alert_generator.generate_alert(
            critical_news2_result, sample_patient
        )
        
        context = alert.clinical_context
        assert context["news2_total_score"] == critical_news2_result.total_score
        assert context["patient_age"] == sample_patient.age
        assert context["patient_ward"] == sample_patient.ward_id
        assert context["is_copd_patient"] == sample_patient.is_copd_patient
    
    @pytest.mark.asyncio
    async def test_copd_patient_alert_generation(self, alert_generator, copd_patient):
        """Test alert generation for COPD patients."""
        # Create NEWS2 result with Scale 2 (COPD)
        copd_news2 = NEWS2Result(
            total_score=7,
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 3,  # High SpO2 concerning for COPD
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 1,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=2,  # COPD scale
            warnings=["High SpO2 for COPD patient"],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.3
        )
        
        alert = await alert_generator.generate_alert(copd_news2, copd_patient)
        
        assert alert is not None
        assert alert.clinical_context["is_copd_patient"] is True
        assert alert.clinical_context["scale_used"] == 2
    
    @pytest.mark.asyncio
    async def test_no_alert_for_low_scores(self, alert_generator, sample_patient):
        """Test that low scores don't generate alerts."""
        low_news2 = NEWS2Result(
            total_score=0,
            individual_scores={
                "respiratory_rate": 0,
                "spo2": 0,
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.LOW,
            monitoring_frequency="routine",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.0
        )
        
        alert = await alert_generator.generate_alert(low_news2, sample_patient)
        
        assert alert is None


class TestDefaultThresholdManager:
    """Test cases for default threshold management."""
    
    def test_default_threshold_creation(self):
        """Test creation of default thresholds."""
        manager = DefaultThresholdManager()
        thresholds = manager.get_default_thresholds("TEST_WARD")
        
        assert len(thresholds) == 4  # CRITICAL, HIGH, MEDIUM, LOW
        
        # Verify critical threshold
        critical_threshold = next(t for t in thresholds if t.alert_level == AlertLevel.CRITICAL)
        assert critical_threshold.news2_min == 7
        assert critical_threshold.news2_max is None
        assert critical_threshold.single_param_trigger is True
        assert critical_threshold.active_hours == (0, 24)
        
        # Verify high threshold
        high_threshold = next(t for t in thresholds if t.alert_level == AlertLevel.HIGH)
        assert high_threshold.news2_min == 5
        assert high_threshold.news2_max == 6
        
        # Verify medium threshold
        medium_threshold = next(t for t in thresholds if t.alert_level == AlertLevel.MEDIUM)
        assert medium_threshold.news2_min == 3
        assert medium_threshold.news2_max == 4
        
        # Verify low threshold
        low_threshold = next(t for t in thresholds if t.alert_level == AlertLevel.LOW)
        assert low_threshold.news2_min == 1
        assert low_threshold.news2_max == 2
    
    def test_default_escalation_matrix_creation(self):
        """Test creation of default escalation matrices."""
        manager = DefaultThresholdManager()
        
        # Test critical alert escalation
        critical_matrix = manager.get_default_escalation_matrix("TEST_WARD", AlertLevel.CRITICAL)
        assert len(critical_matrix.escalation_steps) == 4
        
        steps = critical_matrix.escalation_steps
        assert steps[0].delay_minutes == 0    # Immediate to ward nurse
        assert steps[1].delay_minutes == 15   # 15 min to charge nurse
        assert steps[2].delay_minutes == 30   # 30 min to doctor
        assert steps[3].delay_minutes == 45   # 45 min to rapid response
        
        # Test high alert escalation
        high_matrix = manager.get_default_escalation_matrix("TEST_WARD", AlertLevel.HIGH)
        assert len(high_matrix.escalation_steps) == 3
        
        high_steps = high_matrix.escalation_steps
        assert high_steps[0].delay_minutes == 0    # Immediate to ward nurse
        assert high_steps[1].delay_minutes == 30   # 30 min to charge nurse
        assert high_steps[2].delay_minutes == 60   # 60 min to doctor


class TestPerformanceRequirements:
    """Test cases for performance requirements validation."""
    
    @pytest.mark.asyncio
    async def test_critical_alert_5_second_requirement(self, alert_generator, sample_patient):
        """Test that critical alerts are consistently generated within 5 seconds."""
        # Create multiple critical NEWS2 results
        critical_results = []
        for i in range(10):
            result = NEWS2Result(
                total_score=7 + i % 3,  # Scores 7, 8, 9
                individual_scores={
                    "respiratory_rate": 2,
                    "spo2": 2,
                    "temperature": 1,
                    "systolic_bp": 2,
                    "heart_rate": min(3, i % 4),
                    "consciousness": 0
                },
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=2.0 + i * 0.1
            )
            critical_results.append(result)
        
        # Test generation times
        generation_times = []
        for result in critical_results:
            start_time = datetime.now(timezone.utc)
            alert = await alert_generator.generate_alert(result, sample_patient)
            end_time = datetime.now(timezone.utc)
            
            generation_time_ms = (end_time - start_time).total_seconds() * 1000
            generation_times.append(generation_time_ms)
            
            assert alert is not None
            assert alert.alert_level == AlertLevel.CRITICAL
        
        # Verify all generation times are under 5 seconds
        max_time = max(generation_times)
        avg_time = sum(generation_times) / len(generation_times)
        
        assert max_time < 5000, f"Maximum generation time {max_time:.1f}ms exceeds 5s limit"
        assert avg_time < 2000, f"Average generation time {avg_time:.1f}ms should be well under limit"
    
    @pytest.mark.asyncio
    async def test_concurrent_alert_generation(self, alert_generator, sample_patient):
        """Test concurrent alert generation performance."""
        # Create multiple NEWS2 results for concurrent processing
        news2_results = []
        for i in range(20):
            result = NEWS2Result(
                total_score=3 + i % 5,  # Mix of alert levels
                individual_scores={
                    "respiratory_rate": i % 4,
                    "spo2": (i + 1) % 4,
                    "temperature": (i + 2) % 4,
                    "systolic_bp": (i + 3) % 4,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.MEDIUM,
                monitoring_frequency="6 hourly",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=1.5
            )
            news2_results.append(result)
        
        # Process alerts concurrently
        start_time = datetime.now(timezone.utc)
        
        tasks = [
            alert_generator.generate_alert(result, sample_patient)
            for result in news2_results
        ]
        alerts = await asyncio.gather(*tasks)
        
        end_time = datetime.now(timezone.utc)
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify results
        successful_alerts = [a for a in alerts if a is not None]
        assert len(successful_alerts) > 0
        
        # Concurrent processing should be faster than sequential
        # (Allow generous timeout for test environment)
        assert total_time_ms < 10000, f"Concurrent processing took {total_time_ms:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_single_parameter_bypass_speed(self, alert_generator, sample_patient):
        """Test that single parameter alerts bypass suppression quickly."""
        # Test multiple single parameter scenarios
        single_param_scenarios = [
            {"respiratory_rate": 3, "param_name": "respiratory_rate"},
            {"spo2": 3, "param_name": "spo2"},
            {"temperature": 3, "param_name": "temperature"},
            {"systolic_bp": 3, "param_name": "systolic_bp"},
            {"heart_rate": 3, "param_name": "heart_rate"},
            {"consciousness": 3, "param_name": "consciousness"}
        ]
        
        for scenario in single_param_scenarios:
            # Create NEWS2 result with single critical parameter
            individual_scores = {
                "respiratory_rate": 0,
                "spo2": 0,
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 0,
                "consciousness": 0
            }
            individual_scores[scenario["param_name"]] = 3
            
            news2_result = NEWS2Result(
                total_score=3,  # Just the critical parameter
                individual_scores=individual_scores,
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=1.8
            )
            
            start_time = datetime.now(timezone.utc)
            alert = await alert_generator.generate_alert(news2_result, sample_patient)
            end_time = datetime.now(timezone.utc)
            
            generation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            assert alert is not None
            assert alert.alert_level == AlertLevel.CRITICAL
            assert alert.alert_decision.single_param_trigger is True
            assert generation_time_ms < 2000  # Should be very fast for bypass logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])