"""
Integration tests for alert suppression functionality.

Tests the complete alert suppression system including time-based suppression,
pattern recognition, manual overrides, and 50%+ volume reduction without
critical misses.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from uuid import uuid4

import redis.asyncio as redis
from prometheus_client import CollectorRegistry

from src.models.alerts import Alert, AlertLevel, AlertPriority, AlertStatus
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_suppression import (
    SuppressionEngine, PatternDetector, ManualOverrideManager, 
    SuppressionDecisionLogger, SuppressionMetrics,
    SuppressionDecision, AlertAcknowledgment, SuppressionOverride
)
from src.services.audit import AuditLogger


@pytest.fixture
async def redis_client():
    """Redis client for testing."""
    client = redis.Redis.from_url("redis://localhost:6379/1", decode_responses=True)
    yield client
    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture
def audit_logger():
    """Mock audit logger for testing."""
    return AuditLogger()


@pytest.fixture
def prometheus_registry():
    """Fresh Prometheus registry for each test."""
    return CollectorRegistry()


@pytest.fixture
async def suppression_engine(redis_client, audit_logger):
    """Configured suppression engine for testing."""
    return SuppressionEngine(redis_client, audit_logger)


@pytest.fixture
async def pattern_detector(redis_client):
    """Pattern detector for testing."""
    return PatternDetector(redis_client)


@pytest.fixture
async def override_manager(redis_client, audit_logger):
    """Manual override manager for testing."""
    return ManualOverrideManager(redis_client, audit_logger)


@pytest.fixture
async def suppression_logger(redis_client, audit_logger):
    """Suppression decision logger for testing."""
    return SuppressionDecisionLogger(redis_client, audit_logger)


@pytest.fixture
async def suppression_metrics(prometheus_registry):
    """Suppression metrics collector for testing."""
    return SuppressionMetrics(prometheus_registry)


@pytest.fixture
def sample_patient():
    """Sample patient for testing."""
    return Patient(
        patient_id="TEST_001",
        age=65,
        ward_id="WARD_A",
        is_copd_patient=False
    )


@pytest.fixture
def sample_news2_result():
    """Sample NEWS2 result for testing."""
    return NEWS2Result(
        patient_id="TEST_001",
        total_score=5,
        individual_scores={
            "respiratory_rate": 1,
            "spo2": 1,
            "temperature": 0,
            "systolic_bp": 2,
            "heart_rate": 1,
            "consciousness": 0
        },
        risk_category=RiskCategory.MEDIUM,
        scale_used="Scale 1",
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_alert(sample_patient, sample_news2_result):
    """Sample alert for testing."""
    from src.models.alerts import AlertDecision
    
    alert_decision = AlertDecision(
        decision_id=uuid4(),
        patient_id=sample_patient.patient_id,
        news2_result=sample_news2_result,
        alert_level=AlertLevel.MEDIUM,
        alert_priority=AlertPriority.URGENT,
        threshold_applied=None,
        reasoning="Test alert for suppression testing",
        decision_timestamp=datetime.now(timezone.utc),
        generation_latency_ms=10.0,
        single_param_trigger=False,
        suppressed=False,
        ward_id=sample_patient.ward_id
    )
    
    return Alert(
        alert_id=uuid4(),
        patient_id=sample_patient.patient_id,
        patient=sample_patient,
        alert_decision=alert_decision,
        alert_level=AlertLevel.MEDIUM,
        alert_priority=AlertPriority.URGENT,
        title="MEDIUM ALERT - NEWS2 Score 5",
        message="Patient requires medical review",
        clinical_context={
            "news2_total_score": 5,
            "patient_age": 65,
            "patient_ward": "WARD_A"
        },
        created_at=datetime.now(timezone.utc),
        status=AlertStatus.PENDING,
        assigned_to=None,
        acknowledged_at=None,
        acknowledged_by=None,
        escalation_step=0,
        max_escalation_step=2,
        next_escalation_at=None,
        resolved_at=None,
        resolved_by=None
    )


class TestCriticalAlertSafety:
    """Test critical alert safety - never suppress critical alerts."""
    
    async def test_never_suppress_critical_alerts(self, suppression_engine, sample_alert):
        """Test that critical alerts are never suppressed."""
        # Make alert critical
        sample_alert.alert_level = AlertLevel.CRITICAL
        sample_alert.alert_decision.alert_level = AlertLevel.CRITICAL
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        assert decision.suppressed is False
        assert decision.reason == "NEVER_SUPPRESS_CRITICAL"
        assert decision.confidence_score == 1.0
    
    async def test_never_suppress_single_param_trigger(self, suppression_engine, sample_alert):
        """Test that single parameter triggers are never suppressed."""
        # Make alert have single parameter trigger
        sample_alert.alert_decision.single_param_trigger = True
        sample_alert.alert_level = AlertLevel.CRITICAL
        sample_alert.alert_decision.alert_level = AlertLevel.CRITICAL
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        assert decision.suppressed is False
        assert decision.reason == "NEVER_SUPPRESS_CRITICAL"


class TestTimeBasedSuppression:
    """Test time-based suppression with 30-minute windows."""
    
    async def test_time_based_suppression_within_window(self, suppression_engine, sample_alert):
        """Test suppression within 30-minute window."""
        # Record recent acknowledgment
        ack_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        await suppression_engine.record_acknowledgment(sample_alert, "NURSE_001")
        
        # Create acknowledgment in Redis manually for testing
        ack = AlertAcknowledgment(
            ack_id=uuid4(),
            alert_id=sample_alert.alert_id,
            patient_id=sample_alert.patient_id,
            news2_score=3,  # Lower than current score
            acknowledged_by="NURSE_001",
            acknowledged_at=ack_time,
            alert_level=AlertLevel.MEDIUM
        )
        
        ack_key = f"patient_acknowledgments:{sample_alert.patient_id}"
        await suppression_engine.redis.zadd(
            ack_key,
            {json.dumps(ack.to_dict()): ack_time.timestamp()}
        )
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        assert decision.suppressed is True
        assert decision.reason == "TIME_BASED_SUPPRESSION"
    
    async def test_bypass_suppression_on_score_increase(self, suppression_engine, sample_alert):
        """Test suppression bypass when score increases by 2+ points."""
        # Record recent acknowledgment with lower score
        ack_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        ack = AlertAcknowledgment(
            ack_id=uuid4(),
            alert_id=uuid4(),
            patient_id=sample_alert.patient_id,
            news2_score=3,  # Current alert has score 5, delta = 2
            acknowledged_by="NURSE_001",
            acknowledged_at=ack_time,
            alert_level=AlertLevel.LOW
        )
        
        ack_key = f"patient_acknowledgments:{sample_alert.patient_id}"
        await suppression_engine.redis.zadd(
            ack_key,
            {json.dumps(ack.to_dict()): ack_time.timestamp()}
        )
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        # Should not suppress due to score increase
        assert decision.suppressed is False
        assert decision.reason == "NO_SUPPRESSION_APPLIES"
    
    async def test_no_suppression_outside_window(self, suppression_engine, sample_alert):
        """Test no suppression outside 30-minute window."""
        # Record old acknowledgment (45 minutes ago)
        ack_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        ack = AlertAcknowledgment(
            ack_id=uuid4(),
            alert_id=uuid4(),
            patient_id=sample_alert.patient_id,
            news2_score=4,
            acknowledged_by="NURSE_001",
            acknowledged_at=ack_time,
            alert_level=AlertLevel.MEDIUM
        )
        
        ack_key = f"patient_acknowledgments:{sample_alert.patient_id}"
        await suppression_engine.redis.zadd(
            ack_key,
            {json.dumps(ack.to_dict()): ack_time.timestamp()}
        )
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        assert decision.suppressed is False
        assert decision.reason == "NO_SUPPRESSION_APPLIES"


class TestPatternDetection:
    """Test pattern-based suppression for stable high scores."""
    
    async def test_stable_pattern_detection(self, pattern_detector, sample_alert):
        """Test detection of stable score patterns."""
        # Create stable score history
        patient_id = sample_alert.patient_id
        history_key = f"patient_score_history:{patient_id}"
        
        # Add 6 entries with stable scores (variance ≤ 1)
        base_time = datetime.now(timezone.utc)
        stable_scores = [5, 5, 6, 5, 5, 6]  # Variance ≤ 1
        
        for i, score in enumerate(stable_scores):
            entry = {
                "total_score": score,
                "timestamp": (base_time - timedelta(hours=3-i*0.5)).isoformat(),
                "patient_id": patient_id
            }
            await pattern_detector.redis.zadd(
                history_key,
                {json.dumps(entry): (base_time - timedelta(hours=3-i*0.5)).timestamp()}
            )
        
        is_stable = await pattern_detector.is_stable_pattern(sample_alert)
        assert is_stable is True
    
    async def test_unstable_pattern_not_detected(self, pattern_detector, sample_alert):
        """Test that unstable patterns are not flagged for suppression."""
        # Create unstable score history
        patient_id = sample_alert.patient_id
        history_key = f"patient_score_history:{patient_id}"
        
        # Add entries with high variance
        base_time = datetime.now(timezone.utc)
        unstable_scores = [3, 7, 2, 8, 4, 9]  # High variance
        
        for i, score in enumerate(unstable_scores):
            entry = {
                "total_score": score,
                "timestamp": (base_time - timedelta(hours=3-i*0.5)).isoformat(),
                "patient_id": patient_id
            }
            await pattern_detector.redis.zadd(
                history_key,
                {json.dumps(entry): (base_time - timedelta(hours=3-i*0.5)).timestamp()}
            )
        
        is_stable = await pattern_detector.is_stable_pattern(sample_alert)
        assert is_stable is False


class TestManualOverrides:
    """Test manual suppression overrides by clinical staff."""
    
    async def test_create_manual_override(self, override_manager):
        """Test creation of manual suppression override."""
        override = await override_manager.create_override(
            patient_id="TEST_001",
            nurse_id="NURSE_001",
            justification="Patient stable, family visiting, no clinical concern despite elevated vitals",
            duration_minutes=120
        )
        
        assert override.patient_id == "TEST_001"
        assert override.nurse_id == "NURSE_001"
        assert len(override.justification) >= 20
        assert override.is_active is True
    
    async def test_override_prevents_alerts(self, suppression_engine, override_manager, sample_alert):
        """Test that active overrides prevent alert generation."""
        # Create manual override
        await override_manager.create_override(
            patient_id=sample_alert.patient_id,
            nurse_id="NURSE_001",
            justification="Patient stable, family visiting, monitoring closely",
            duration_minutes=60
        )
        
        decision = await suppression_engine.should_suppress(sample_alert)
        
        assert decision.suppressed is True
        assert decision.reason == "MANUAL_OVERRIDE"
        assert "NURSE_001" in decision.metadata["nurse_id"]
    
    async def test_expired_override_no_suppression(self, override_manager, sample_alert):
        """Test that expired overrides don't suppress alerts."""
        # Create override that expires immediately
        override = await override_manager.create_override(
            patient_id=sample_alert.patient_id,
            nurse_id="NURSE_001",
            justification="Short term override for testing",
            duration_minutes=0  # Expires immediately
        )
        
        # Wait a moment to ensure expiration
        await asyncio.sleep(0.1)
        
        active_override = await override_manager._save_override(override)
        retrieved_override = await override_manager.get_active_override(sample_alert.patient_id)
        
        assert retrieved_override is None


class TestVolumeReduction:
    """Test 50%+ volume reduction without critical misses."""
    
    async def test_volume_reduction_target(self, suppression_engine, suppression_logger):
        """Test that suppression achieves 50%+ volume reduction."""
        # Simulate 100 alerts over 24 hours
        alerts_generated = 100
        alerts_suppressed = 0
        critical_alerts_suppressed = 0
        
        for i in range(alerts_generated):
            # Create mix of alert levels
            if i < 10:  # 10% critical alerts
                alert_level = AlertLevel.CRITICAL
            elif i < 30:  # 20% high alerts  
                alert_level = AlertLevel.HIGH
            elif i < 60:  # 30% medium alerts
                alert_level = AlertLevel.MEDIUM
            else:  # 40% low alerts
                alert_level = AlertLevel.LOW
            
            # Create test alert
            test_alert = self._create_test_alert(f"TEST_{i:03d}", alert_level)
            
            # Test suppression decision
            decision = await suppression_engine.should_suppress(test_alert)
            
            if decision.suppressed:
                alerts_suppressed += 1
                
                # CRITICAL SAFETY CHECK
                if alert_level == AlertLevel.CRITICAL:
                    critical_alerts_suppressed += 1
        
        # Calculate metrics
        suppression_rate = (alerts_suppressed / alerts_generated) * 100
        
        # Assertions
        assert critical_alerts_suppressed == 0, "CRITICAL: No critical alerts should be suppressed"
        assert suppression_rate >= 50.0, f"Volume reduction {suppression_rate:.1f}% below 50% target"
        
        print(f"Volume reduction achieved: {suppression_rate:.1f}%")
        print(f"Critical alerts suppressed: {critical_alerts_suppressed} (should be 0)")
    
    def _create_test_alert(self, patient_id: str, alert_level: AlertLevel) -> Alert:
        """Create test alert for volume reduction testing."""
        from src.models.alerts import AlertDecision
        
        # Vary NEWS2 scores based on alert level
        score_map = {
            AlertLevel.CRITICAL: 8,
            AlertLevel.HIGH: 6,
            AlertLevel.MEDIUM: 4,
            AlertLevel.LOW: 2
        }
        
        news2_result = NEWS2Result(
            patient_id=patient_id,
            total_score=score_map[alert_level],
            individual_scores={"respiratory_rate": 1, "spo2": 1, "temperature": 0, "systolic_bp": 1, "heart_rate": 1, "consciousness": 0},
            risk_category=RiskCategory.MEDIUM,
            scale_used="Scale 1",
            timestamp=datetime.now(timezone.utc)
        )
        
        patient = Patient(
            patient_id=patient_id,
            age=70,
            ward_id="WARD_A",
            is_copd_patient=False
        )
        
        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient_id,
            news2_result=news2_result,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            threshold_applied=None,
            reasoning=f"Test alert level {alert_level.value}",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=5.0,
            single_param_trigger=(alert_level == AlertLevel.CRITICAL),
            suppressed=False,
            ward_id="WARD_A"
        )
        
        return Alert(
            alert_id=uuid4(),
            patient_id=patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            title=f"{alert_level.value.upper()} ALERT",
            message="Test alert",
            clinical_context={"news2_total_score": score_map[alert_level]},
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=2,
            next_escalation_at=None,
            resolved_at=None,
            resolved_by=None
        )


class TestSuppressionLogging:
    """Test comprehensive suppression decision logging."""
    
    async def test_suppression_decision_logging(self, suppression_logger, sample_alert):
        """Test that all suppression decisions are logged with full context."""
        decision = SuppressionDecision(
            decision_id=uuid4(),
            alert_id=sample_alert.alert_id,
            patient_id=sample_alert.patient_id,
            suppressed=True,
            reason="TEST_SUPPRESSION",
            confidence_score=0.85,
            decision_timestamp=datetime.now(timezone.utc),
            suppression_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            metadata={"test": "logging"}
        )
        
        await suppression_logger.log_suppression_decision(decision, sample_alert)
        
        # Verify logging
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        log_key = f"suppression_log:{today}"
        
        logged_entries = await suppression_logger.redis.zrange(log_key, 0, -1)
        assert len(logged_entries) == 1
        
        log_entry = json.loads(logged_entries[0])
        assert log_entry["decision_id"] == str(decision.decision_id)
        assert log_entry["suppressed"] is True
        assert log_entry["reason"] == "TEST_SUPPRESSION"
    
    async def test_suppression_analytics_generation(self, suppression_logger, sample_alert):
        """Test generation of suppression analytics."""
        # Log multiple decisions
        for i in range(10):
            decision = SuppressionDecision(
                decision_id=uuid4(),
                alert_id=uuid4(),
                patient_id=f"TEST_{i:03d}",
                suppressed=(i % 2 == 0),  # 50% suppression rate
                reason="TIME_BASED_SUPPRESSION" if i % 2 == 0 else "NO_SUPPRESSION_APPLIES",
                confidence_score=0.8,
                decision_timestamp=datetime.now(timezone.utc),
                suppression_expires_at=None,
                metadata={}
            )
            
            test_alert = sample_alert
            test_alert.patient_id = f"TEST_{i:03d}"
            test_alert.alert_id = decision.alert_id
            
            await suppression_logger.log_suppression_decision(decision, test_alert)
        
        # Generate analytics
        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)
        
        analytics = await suppression_logger.get_suppression_analytics(start_date, end_date)
        
        assert analytics["total_decisions"] == 10
        assert analytics["suppressed_count"] == 5
        assert analytics["suppression_rate"] == 0.5
        assert analytics["critical_miss_count"] == 0


class TestPrometheusMetrics:
    """Test Prometheus metrics collection for suppression system."""
    
    async def test_metrics_tracking(self, suppression_metrics, sample_alert):
        """Test that suppression decisions are tracked in Prometheus metrics."""
        decision = SuppressionDecision(
            decision_id=uuid4(),
            alert_id=sample_alert.alert_id,
            patient_id=sample_alert.patient_id,
            suppressed=True,
            reason="TIME_BASED_SUPPRESSION",
            confidence_score=0.9,
            decision_timestamp=datetime.now(timezone.utc),
            suppression_expires_at=None,
            metadata={}
        )
        
        # Track the decision
        suppression_metrics.track_suppression_decision(decision, sample_alert, 0.05)
        
        # Verify metrics are recorded
        decision_metric = suppression_metrics.suppression_decisions_total.labels(
            reason="TIME_BASED_SUPPRESSION",
            ward_id="WARD_A", 
            alert_level="medium"
        )
        assert decision_metric._value._value == 1
        
        suppressed_metric = suppression_metrics.alerts_suppressed_total.labels(
            reason="TIME_BASED_SUPPRESSION",
            ward_id="WARD_A",
            alert_level="medium"
        )
        assert suppressed_metric._value._value == 1
    
    async def test_critical_alert_safety_metrics(self, suppression_metrics, sample_alert):
        """Test that critical alert suppression errors are tracked."""
        # Make alert critical
        sample_alert.alert_level = AlertLevel.CRITICAL
        
        # Create erroneous suppression decision (should never happen)
        decision = SuppressionDecision(
            decision_id=uuid4(),
            alert_id=sample_alert.alert_id,
            patient_id=sample_alert.patient_id,
            suppressed=True,  # This is wrong for critical alerts!
            reason="ERROR_SUPPRESSION",
            confidence_score=0.5,
            decision_timestamp=datetime.now(timezone.utc),
            suppression_expires_at=None,
            metadata={}
        )
        
        # Track the decision
        suppression_metrics.track_suppression_decision(decision, sample_alert, 0.05)
        
        # Verify critical error is tracked
        error_metric = suppression_metrics.critical_alerts_suppressed_error.labels(ward_id="WARD_A")
        assert error_metric._value._value == 1


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for real-world suppression scenarios."""
    
    async def test_complete_suppression_workflow(
        self, 
        suppression_engine,
        override_manager,
        suppression_logger,
        suppression_metrics,
        sample_alert
    ):
        """Test complete workflow from alert generation to suppression decision."""
        # 1. First alert - should not be suppressed
        decision1 = await suppression_engine.should_suppress(sample_alert)
        assert decision1.suppressed is False
        
        # 2. Acknowledge the alert
        await suppression_engine.record_acknowledgment(sample_alert, "NURSE_001")
        
        # 3. Second alert within 30 minutes - should be suppressed
        sample_alert.alert_id = uuid4()
        decision2 = await suppression_engine.should_suppress(sample_alert)
        assert decision2.suppressed is True
        assert decision2.reason == "TIME_BASED_SUPPRESSION"
        
        # 4. Log decisions
        await suppression_logger.log_suppression_decision(decision1, sample_alert)
        await suppression_logger.log_suppression_decision(decision2, sample_alert)
        
        # 5. Track metrics
        suppression_metrics.track_suppression_decision(decision1, sample_alert, 0.01)
        suppression_metrics.track_suppression_decision(decision2, sample_alert, 0.008)
        
        # 6. Create manual override
        override = await override_manager.create_override(
            patient_id=sample_alert.patient_id,
            nurse_id="NURSE_002", 
            justification="Patient being closely monitored by clinical team",
            duration_minutes=60
        )
        
        # 7. Third alert with override - should be suppressed
        sample_alert.alert_id = uuid4()
        decision3 = await suppression_engine.should_suppress(sample_alert)
        assert decision3.suppressed is True
        assert decision3.reason == "MANUAL_OVERRIDE"
        
        print("✅ Complete suppression workflow test passed")