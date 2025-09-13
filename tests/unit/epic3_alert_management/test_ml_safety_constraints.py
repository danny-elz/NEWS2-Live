"""
Epic 3: ML Alert Suppression Safety Constraint Tests
CRITICAL: These tests validate absolute safety constraints that must NEVER fail.

This module implements property-based testing and safety validation for the ML-powered
alert suppression system, ensuring critical alerts are never suppressed.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from typing import List, Dict, Any

# Hypothesis for property-based testing
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle

# Core models and services
from src.models.alerts import Alert, AlertLevel, AlertStatus, AlertDecision, AlertPriority
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_suppression import SuppressionEngine, SuppressionDecision


class TestCriticalSafetyConstraints:
    """Test ABSOLUTE safety constraints - these tests must NEVER fail."""

    @pytest.fixture
    async def suppression_engine(self):
        """Create suppression engine for testing."""
        mock_redis = AsyncMock()
        mock_audit_logger = Mock()
        engine = SuppressionEngine(mock_redis, mock_audit_logger)
        return engine

    @pytest.fixture
    def critical_alert(self):
        """Create a critical alert for safety testing."""
        return self._create_critical_alert()

    @pytest.fixture
    def red_flag_alert(self):
        """Create a red flag alert (single parameter = 3)."""
        return self._create_red_flag_alert()

    @pytest.mark.critical_safety
    @pytest.mark.asyncio
    async def test_critical_alerts_never_suppressed_basic(self, suppression_engine, critical_alert):
        """BASIC: Critical alerts must NEVER be suppressed."""

        # Test basic critical alert (NEWS2 ≥ 7)
        decision = await suppression_engine.should_suppress(critical_alert)

        # ABSOLUTE SAFETY CONSTRAINT
        assert decision.suppressed is False, \
            f"SAFETY VIOLATION: Critical alert {critical_alert.alert_id} was suppressed"

        assert decision.reason == "NEVER_SUPPRESS_CRITICAL", \
            f"Wrong suppression reason for critical alert: {decision.reason}"

    @pytest.mark.critical_safety
    @pytest.mark.asyncio
    async def test_red_flag_alerts_never_suppressed_basic(self, suppression_engine, red_flag_alert):
        """BASIC: Red flag alerts (single param = 3) must NEVER be suppressed."""

        decision = await suppression_engine.should_suppress(red_flag_alert)

        # ABSOLUTE SAFETY CONSTRAINT
        assert decision.suppressed is False, \
            f"SAFETY VIOLATION: Red flag alert {red_flag_alert.alert_id} was suppressed"

    @pytest.mark.critical_safety
    @pytest.mark.asyncio
    @given(
        news2_score=st.integers(min_value=7, max_value=20),
        patient_age=st.integers(min_value=18, max_value=100),
        time_since_ack=st.integers(min_value=0, max_value=1440)  # 0-24 hours
    )
    @settings(
        max_examples=3,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_critical_score_safety_property(self, suppression_engine, news2_score, patient_age, time_since_ack):
        """PROPERTY: ANY NEWS2 ≥7 must never be suppressed regardless of context."""

        # Create alert with critical score
        alert = self._create_alert_with_context(
            news2_score=news2_score,
            patient_age=patient_age,
            time_since_ack=time_since_ack
        )

        decision = await suppression_engine.should_suppress(alert)

        # PROPERTY: Critical scores are NEVER suppressed
        assert decision.suppressed is False, \
            f"Property violation: NEWS2 {news2_score} suppressed (age={patient_age}, time_since_ack={time_since_ack})"

    @pytest.mark.critical_safety
    @pytest.mark.asyncio
    @given(
        param_name=st.sampled_from([
            "respiratory_rate", "spo2", "temperature",
            "systolic_bp", "heart_rate", "consciousness"
        ]),
        other_scores=st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=5)
    )
    @settings(
        max_examples=3,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_single_parameter_red_flag_property(self, suppression_engine, param_name, other_scores):
        """PROPERTY: ANY single parameter = 3 must never be suppressed."""

        # Create alert with one parameter = 3
        alert = self._create_single_red_flag_alert(param_name, other_scores)

        decision = await suppression_engine.should_suppress(alert)

        # PROPERTY: Red flags are NEVER suppressed
        assert decision.suppressed is False, \
            f"Property violation: Red flag {param_name}=3 was suppressed"

    @pytest.mark.critical_safety
    @pytest.mark.asyncio
    async def test_multiple_critical_scenarios_batch(self, suppression_engine):
        """BATCH: Test 100 different critical scenarios to ensure none are suppressed."""

        critical_scenarios = []

        # Generate diverse critical scenarios
        for i in range(100):
            scenario = self._create_diverse_critical_scenario(i)
            critical_scenarios.append(scenario)

        # Test all scenarios
        results = []
        for scenario in critical_scenarios:
            decision = await suppression_engine.should_suppress(scenario)
            results.append((scenario, decision))

        # Verify ALL are not suppressed
        suppressed_criticals = [
            (scenario, decision) for scenario, decision in results
            if decision.suppressed
        ]

        assert len(suppressed_criticals) == 0, \
            f"SAFETY VIOLATION: {len(suppressed_criticals)} critical alerts were suppressed: {suppressed_criticals}"

    # Helper methods for creating test alerts
    def _create_critical_alert(self) -> Alert:
        """Create critical alert for testing."""
        news2_result = NEWS2Result(
            total_score=8,  # Critical score
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 2,
                "temperature": 1,
                "systolic_bp": 3,  # Critical parameter
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["Critical hypotension detected"],
            confidence=0.95,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.1
        )

        patient = Patient(
            patient_id="CRITICAL_TEST_001",
            ward_id="ICU",
            bed_number="ICU-01",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(days=1),
            last_updated=datetime.now(timezone.utc)
        )

        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient.patient_id,
            news2_result=news2_result,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            threshold_applied=None,
            reasoning="Critical NEWS2 score requires immediate attention",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=5.2,
            single_param_trigger=True,
            suppressed=False,
            ward_id=patient.ward_id
        )

        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            title="Critical Patient Deterioration",
            message="Patient shows critical deterioration - immediate intervention required",
            clinical_context={
                "news2_total_score": news2_result.total_score,
                "critical_parameters": ["systolic_bp"],
                "trend": "deteriorating",
                "confidence": news2_result.confidence
            },
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=patient.assigned_nurse_id,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=4,
            next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            resolved_at=None,
            resolved_by=None
        )

    def _create_red_flag_alert(self) -> Alert:
        """Create red flag alert (single parameter = 3)."""
        news2_result = NEWS2Result(
            total_score=5,  # May not be critical total, but has red flag
            individual_scores={
                "respiratory_rate": 3,  # RED FLAG
                "spo2": 1,
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="1 hourly",
            scale_used=1,
            warnings=["Critical respiratory rate detected"],
            confidence=0.92,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.3
        )

        patient = Patient(
            patient_id="RED_FLAG_TEST_001",
            ward_id="MEDICAL_A",
            bed_number="A-15",
            age=72,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_002",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=12),
            last_updated=datetime.now(timezone.utc)
        )

        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient.patient_id,
            news2_result=news2_result,
            alert_level=AlertLevel.HIGH,
            alert_priority=AlertPriority.URGENT,
            threshold_applied=None,
            reasoning="Single parameter red flag detected",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=3.1,
            single_param_trigger=True,
            suppressed=False,
            ward_id=patient.ward_id
        )

        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=AlertLevel.HIGH,
            alert_priority=AlertPriority.URGENT,
            title="Red Flag Alert: Critical Respiratory Rate",
            message="Patient has critically abnormal respiratory rate requiring immediate assessment",
            clinical_context={
                "news2_total_score": news2_result.total_score,
                "red_flag_parameter": "respiratory_rate",
                "red_flag_score": 3,
                "confidence": news2_result.confidence
            },
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=patient.assigned_nurse_id,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=3,
            next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            resolved_at=None,
            resolved_by=None
        )

    def _create_alert_with_context(self, news2_score: int, patient_age: int, time_since_ack: int) -> Alert:
        """Create alert with specific context for property testing."""

        # Distribute score across parameters to reach target
        base_scores = [0, 1, 1, 2, 1, 0]  # Base distribution
        remaining = news2_score - sum(base_scores)

        # Add remaining points randomly
        import random
        while remaining > 0:
            param_idx = random.randint(0, 5)
            if base_scores[param_idx] < 3:
                addition = min(remaining, 3 - base_scores[param_idx])
                base_scores[param_idx] += addition
                remaining -= addition

        param_names = ["respiratory_rate", "spo2", "temperature", "systolic_bp", "heart_rate", "consciousness"]
        individual_scores = dict(zip(param_names, base_scores))

        news2_result = NEWS2Result(
            total_score=news2_score,
            individual_scores=individual_scores,
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["Generated for property testing"],
            confidence=0.90,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.0
        )

        patient = Patient(
            patient_id=f"PROP_TEST_{news2_score}_{patient_age}",
            ward_id="TEST_WARD",
            bed_number="T-01",
            age=patient_age,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_PROP_TEST",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=24),
            last_updated=datetime.now(timezone.utc)
        )

        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient.patient_id,
            news2_result=news2_result,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            threshold_applied=None,
            reasoning="Property test critical alert",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=2.0,
            single_param_trigger=max(individual_scores.values()) >= 3,
            suppressed=False,
            ward_id=patient.ward_id
        )

        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            title="Property Test Critical Alert",
            message=f"Property test alert - NEWS2: {news2_score}",
            clinical_context={
                "news2_total_score": news2_score,
                "patient_age": patient_age,
                "time_since_ack_minutes": time_since_ack,
                "test_context": True
            },
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=patient.assigned_nurse_id,
            acknowledged_at=datetime.now(timezone.utc) - timedelta(minutes=time_since_ack) if time_since_ack < 1440 else None,
            acknowledged_by="NURSE_TEST" if time_since_ack < 1440 else None,
            escalation_step=0,
            max_escalation_step=4,
            next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            resolved_at=None,
            resolved_by=None
        )

    def _create_single_red_flag_alert(self, param_name: str, other_scores: List[int]) -> Alert:
        """Create alert with single red flag parameter for property testing."""

        # Create scores dict with one parameter = 3
        param_names = ["respiratory_rate", "spo2", "temperature", "systolic_bp", "heart_rate", "consciousness"]
        individual_scores = {}

        score_idx = 0
        for param in param_names:
            if param == param_name:
                individual_scores[param] = 3  # Red flag
            else:
                individual_scores[param] = other_scores[score_idx] if score_idx < len(other_scores) else 0
                score_idx += 1

        total_score = sum(individual_scores.values())

        news2_result = NEWS2Result(
            total_score=total_score,
            individual_scores=individual_scores,
            risk_category=RiskCategory.HIGH if total_score >= 7 else RiskCategory.MEDIUM,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=[f"Red flag detected: {param_name}"],
            confidence=0.94,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.8
        )

        patient = Patient(
            patient_id=f"RED_FLAG_{param_name}_{total_score}",
            ward_id="TEST_WARD",
            bed_number="T-02",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_RED_FLAG_TEST",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=6),
            last_updated=datetime.now(timezone.utc)
        )

        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient.patient_id,
            news2_result=news2_result,
            alert_level=AlertLevel.HIGH,
            alert_priority=AlertPriority.URGENT,
            threshold_applied=None,
            reasoning=f"Red flag parameter detected: {param_name}=3",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=2.1,
            single_param_trigger=True,
            suppressed=False,
            ward_id=patient.ward_id
        )

        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=AlertLevel.HIGH,
            alert_priority=AlertPriority.URGENT,
            title=f"Red Flag Alert: {param_name.replace('_', ' ').title()}",
            message=f"Critical {param_name.replace('_', ' ')} detected - immediate assessment required",
            clinical_context={
                "news2_total_score": total_score,
                "red_flag_parameter": param_name,
                "red_flag_score": 3,
                "other_scores": other_scores
            },
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=patient.assigned_nurse_id,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=3,
            next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            resolved_at=None,
            resolved_by=None
        )

    def _create_diverse_critical_scenario(self, scenario_id: int) -> Alert:
        """Create diverse critical scenario for batch testing."""

        # Vary patient demographics and clinical context
        ages = [25, 45, 65, 85]
        wards = ["ICU", "MEDICAL_A", "SURGICAL_B", "RESPIRATORY"]

        age = ages[scenario_id % len(ages)]
        ward = wards[scenario_id % len(wards)]

        # Create different patterns of critical scores
        patterns = [
            {"respiratory_rate": 3, "spo2": 2, "temperature": 1, "systolic_bp": 2, "heart_rate": 1, "consciousness": 0},
            {"respiratory_rate": 2, "spo2": 3, "temperature": 0, "systolic_bp": 1, "heart_rate": 2, "consciousness": 1},
            {"respiratory_rate": 1, "spo2": 1, "temperature": 1, "systolic_bp": 3, "heart_rate": 2, "consciousness": 0},
            {"respiratory_rate": 0, "spo2": 0, "temperature": 0, "systolic_bp": 0, "heart_rate": 0, "consciousness": 3}
        ]

        pattern = patterns[scenario_id % len(patterns)]
        total_score = sum(pattern.values())

        # Ensure score is critical (≥7)
        if total_score < 7:
            pattern["respiratory_rate"] += (7 - total_score)
            total_score = sum(pattern.values())

        news2_result = NEWS2Result(
            total_score=total_score,
            individual_scores=pattern,
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=[f"Critical scenario {scenario_id}"],
            confidence=0.93 + (scenario_id % 7) * 0.01,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.5 + (scenario_id % 5) * 0.1
        )

        patient = Patient(
            patient_id=f"BATCH_CRITICAL_{scenario_id:03d}",
            ward_id=ward,
            bed_number=f"{ward[0]}-{(scenario_id % 30) + 1:02d}",
            age=age,
            is_copd_patient=(scenario_id % 10 == 0),  # 10% COPD
            assigned_nurse_id=f"NURSE_{(scenario_id % 20) + 1:03d}",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=scenario_id % 72),
            last_updated=datetime.now(timezone.utc)
        )

        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient.patient_id,
            news2_result=news2_result,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            threshold_applied=None,
            reasoning=f"Batch test critical scenario {scenario_id}",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=2.0,
            single_param_trigger=max(pattern.values()) >= 3,
            suppressed=False,
            ward_id=patient.ward_id
        )

        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            title=f"Batch Critical Alert {scenario_id}",
            message=f"Critical patient deterioration - scenario {scenario_id}",
            clinical_context={
                "news2_total_score": total_score,
                "scenario_id": scenario_id,
                "patient_age": age,
                "ward_type": ward
            },
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=patient.assigned_nurse_id,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=4,
            next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            resolved_at=None,
            resolved_by=None
        )


if __name__ == "__main__":
    # Run safety tests
    pytest.main([__file__, "-v", "-m", "critical_safety", "--tb=short"])