"""
Epic 3: ML Alert Suppression Testing Framework
Advanced ML testing for intelligent alert suppression with clinical safety validation.

This refactors existing predictive model patterns for alert-specific ML testing.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any

from src.models.alerts import Alert, AlertLevel, AlertStatus
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.ml_alert_suppression import (
    MLSuppressionEngine,
    SuppressionPrediction,
    AlertFeatures,
    ClinicalContext,
    SafetyConstraints,
    ModelConfidence
)


class TestMLSuppressionSafetyConstraints:
    """Test ABSOLUTE safety constraints - these tests must NEVER fail in production."""

    @pytest.fixture
    def safety_validator(self):
        """Safety constraint validator."""
        return SafetyConstraints()

    @pytest.fixture
    def ml_suppression_engine(self):
        """ML suppression engine for testing."""
        return MLSuppressionEngine()

    @pytest.mark.critical_safety
    async def test_critical_alerts_never_suppressed(self, ml_suppression_engine, safety_validator):
        """CRITICAL: ML models must NEVER suppress critical alerts."""

        # Generate 1000 critical alert scenarios
        critical_scenarios = []
        for i in range(1000):
            alert = self._create_critical_alert(
                news2_score=7 + (i % 14),  # Scores 7-20
                patient_age=30 + (i % 70),
                single_param_score=3 if i % 3 == 0 else None
            )
            critical_scenarios.append(alert)

        # Test ML suppression decisions
        for scenario in critical_scenarios:
            prediction = await ml_suppression_engine.predict_suppression(scenario)

            # ABSOLUTE SAFETY CONSTRAINT
            assert prediction.should_suppress is False, \
                f"SAFETY VIOLATION: Critical alert {scenario.alert_id} was marked for suppression"

            assert "CRITICAL_NEVER_SUPPRESS" in prediction.reasoning, \
                f"Missing safety reasoning for critical alert {scenario.alert_id}"

            # Validate safety constraints
            safety_result = safety_validator.validate_prediction(scenario, prediction)
            assert safety_result.safe is True, \
                f"Safety constraint failed: {safety_result.violation_reason}"

    @pytest.mark.critical_safety
    async def test_red_flag_alerts_never_suppressed(self, ml_suppression_engine):
        """CRITICAL: Single parameter red flags (score=3) must never be suppressed."""

        red_flag_params = [
            "respiratory_rate", "spo2", "temperature",
            "systolic_bp", "heart_rate", "consciousness"
        ]

        for param in red_flag_params:
            # Create alert with single parameter = 3
            alert = self._create_red_flag_alert(param, score=3)

            prediction = await ml_suppression_engine.predict_suppression(alert)

            # ABSOLUTE SAFETY CONSTRAINT
            assert prediction.should_suppress is False, \
                f"SAFETY VIOLATION: Red flag alert ({param}=3) was marked for suppression"

            assert f"RED_FLAG_{param.upper()}" in prediction.reasoning

    @property_based_test
    @given(
        news2_score=st.integers(min_value=7, max_value=20),
        patient_age=st.integers(min_value=18, max_value=100),
        time_since_last_alert=st.integers(min_value=0, max_value=1440)
    )
    @settings(max_examples=5000, deadline=10000)  # Extensive property testing
    async def test_critical_score_safety_property(self, ml_suppression_engine, news2_score, patient_age, time_since_last_alert):
        """Property-based test: ANY NEWS2 ≥7 must never be suppressed regardless of context."""

        alert = self._create_alert_with_context(
            news2_score=news2_score,
            patient_age=patient_age,
            time_since_last_alert=time_since_last_alert
        )

        prediction = await ml_suppression_engine.predict_suppression(alert)

        # PROPERTY: Critical scores are NEVER suppressed
        assert prediction.should_suppress is False, \
            f"Property violation: NEWS2 {news2_score} suppressed (age={patient_age}, time={time_since_last_alert})"

    def _create_critical_alert(self, news2_score: int, patient_age: int, single_param_score: int = None) -> Alert:
        """Create critical alert for testing."""
        individual_scores = {
            "respiratory_rate": 1, "spo2": 1, "temperature": 1,
            "systolic_bp": 1, "heart_rate": 1, "consciousness": 1
        }

        # Add single parameter red flag if specified
        if single_param_score:
            individual_scores["respiratory_rate"] = single_param_score

        news2_result = NEWS2Result(
            total_score=news2_score,
            individual_scores=individual_scores,
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["Critical deterioration detected"],
            confidence=0.95,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.5
        )

        patient = Patient(
            patient_id=f"CRITICAL_{news2_score}_{patient_age}",
            ward_id="ICU",
            bed_number="ICU-01",
            age=patient_age,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(days=1),
            last_updated=datetime.now(timezone.utc)
        )

        return self._build_alert(news2_result, patient, AlertLevel.CRITICAL)


class TestMLSuppressionPerformance:
    """Test ML suppression performance and accuracy."""

    @pytest.fixture
    def ml_engine(self):
        """ML suppression engine."""
        return MLSuppressionEngine()

    @pytest.fixture
    def clinical_scenarios(self):
        """Clinical scenario generator."""
        return ClinicalScenarioGenerator()

    async def test_suppression_precision_target(self, ml_engine, clinical_scenarios):
        """Test that suppression precision meets 90% target."""

        # Generate diverse clinical scenarios with ground truth
        test_scenarios = clinical_scenarios.generate_labeled_scenarios(1000)

        predictions = []
        ground_truth = []

        for scenario in test_scenarios:
            prediction = await ml_engine.predict_suppression(scenario.alert)
            predictions.append(prediction.should_suppress)
            ground_truth.append(scenario.should_suppress_ground_truth)

        # Calculate precision
        true_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
        false_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Must meet 90% precision target
        assert precision >= 0.90, f"Suppression precision {precision:.3f} below 90% target"

    async def test_suppression_recall_target(self, ml_engine, clinical_scenarios):
        """Test that suppression recall meets 95% target."""

        # Generate scenarios with known suppressible alerts
        suppressible_scenarios = clinical_scenarios.generate_suppressible_scenarios(500)

        predictions = []
        ground_truth = []

        for scenario in suppressible_scenarios:
            prediction = await ml_engine.predict_suppression(scenario.alert)
            predictions.append(prediction.should_suppress)
            ground_truth.append(True)  # All should be suppressible

        # Calculate recall
        true_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
        false_negatives = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # Must meet 95% recall target
        assert recall >= 0.95, f"Suppression recall {recall:.3f} below 95% target"

    async def test_volume_reduction_target(self, ml_engine, clinical_scenarios):
        """Test that 60% alert volume reduction is achieved."""

        # Simulate 24-hour hospital alert volume
        daily_alerts = clinical_scenarios.generate_realistic_daily_alerts(500)

        suppressed_count = 0
        critical_alerts_processed = 0

        for alert in daily_alerts:
            prediction = await ml_engine.predict_suppression(alert)

            # Count suppressions (excluding critical alerts)
            if alert.alert_level != AlertLevel.CRITICAL:
                if prediction.should_suppress:
                    suppressed_count += 1
            else:
                critical_alerts_processed += 1
                # Verify critical alerts are never suppressed
                assert prediction.should_suppress is False

        non_critical_alerts = len(daily_alerts) - critical_alerts_processed
        suppression_rate = suppressed_count / non_critical_alerts if non_critical_alerts > 0 else 0

        # Must achieve 60% volume reduction on non-critical alerts
        assert suppression_rate >= 0.60, f"Volume reduction {suppression_rate:.3f} below 60% target"


class TestMLModelInterpretability:
    """Test ML model interpretability and clinical reasoning."""

    @pytest.fixture
    def ml_engine(self):
        return MLSuppressionEngine()

    async def test_clinical_reasoning_generation(self, ml_engine):
        """Test that all ML decisions include clinical reasoning."""

        test_alerts = [
            self._create_stable_patient_alert(),
            self._create_deteriorating_patient_alert(),
            self._create_copd_exacerbation_alert(),
            self._create_post_operative_alert()
        ]

        for alert in test_alerts:
            prediction = await ml_engine.predict_suppression(alert)

            # Must include clinical reasoning
            assert prediction.clinical_reasoning is not None
            assert len(prediction.clinical_reasoning) >= 20  # Meaningful explanation

            # Must include confidence score
            assert 0.0 <= prediction.confidence <= 1.0

            # Must include feature importance
            assert prediction.feature_importance is not None

            # Key clinical features must be present
            expected_features = ["patient_stability", "news2_trend", "time_since_acknowledgment"]
            for feature in expected_features:
                assert feature in prediction.feature_importance

    async def test_model_bias_detection(self, ml_engine):
        """Test for demographic bias in suppression decisions."""

        # Generate identical clinical scenarios with different demographics
        base_scenario = self._create_base_clinical_scenario()

        demographic_variations = [
            {"age": 25, "gender": "female"},
            {"age": 75, "gender": "male"},
            {"age": 45, "gender": "non-binary"},
            {"age": 65, "gender": "female"}
        ]

        predictions = []
        for demographics in demographic_variations:
            alert = self._create_alert_with_demographics(base_scenario, demographics)
            prediction = await ml_engine.predict_suppression(alert)
            predictions.append(prediction)

        # Check for consistent decisions across demographics
        suppression_decisions = [p.should_suppress for p in predictions]
        confidence_scores = [p.confidence for p in predictions]

        # Decisions should be consistent (same clinical scenario)
        assert len(set(suppression_decisions)) <= 1, "Inconsistent decisions across demographics"

        # Confidence scores should be similar (within 10%)
        confidence_range = max(confidence_scores) - min(confidence_scores)
        assert confidence_range <= 0.1, f"Confidence variance {confidence_range} suggests bias"


class TestMLContinuousLearning:
    """Test ML model continuous learning and improvement."""

    @pytest.fixture
    def learning_engine(self):
        return MLSuppressionEngine(learning_enabled=True)

    async def test_outcome_feedback_integration(self, learning_engine):
        """Test that clinical outcomes improve ML model decisions."""

        # Generate initial predictions
        test_alert = self._create_test_alert()
        initial_prediction = await learning_engine.predict_suppression(test_alert)

        # Simulate negative clinical outcome
        outcome_feedback = {
            "alert_id": test_alert.alert_id,
            "suppressed": initial_prediction.should_suppress,
            "clinical_outcome": "patient_deteriorated",
            "nurse_feedback": "should_not_have_suppressed",
            "confidence_in_feedback": 0.9
        }

        # Update model with outcome
        await learning_engine.learn_from_outcome(outcome_feedback)

        # Generate new prediction for similar scenario
        similar_alert = self._create_similar_alert(test_alert)
        updated_prediction = await learning_engine.predict_suppression(similar_alert)

        # Model should learn from negative outcome
        if initial_prediction.should_suppress and outcome_feedback["nurse_feedback"] == "should_not_have_suppressed":
            assert updated_prediction.should_suppress is False or updated_prediction.confidence < initial_prediction.confidence

    async def test_model_performance_monitoring(self, learning_engine):
        """Test continuous monitoring of model performance."""

        # Simulate model performance over time
        performance_metrics = await learning_engine.get_performance_metrics(days=30)

        # Must track key metrics
        required_metrics = [
            "precision", "recall", "f1_score",
            "clinical_outcome_correlation", "nurse_satisfaction"
        ]

        for metric in required_metrics:
            assert metric in performance_metrics
            assert performance_metrics[metric] is not None

        # Performance should meet minimum thresholds
        assert performance_metrics["precision"] >= 0.85
        assert performance_metrics["recall"] >= 0.90
        assert performance_metrics["clinical_outcome_correlation"] >= 0.7


# Helper methods and utilities
class ClinicalScenarioGenerator:
    """Generate diverse clinical scenarios for ML testing."""

    def generate_labeled_scenarios(self, count: int) -> List[Dict]:
        """Generate scenarios with ground truth labels."""
        scenarios = []

        for i in range(count):
            scenario = {
                "alert": self._generate_alert(i),
                "should_suppress_ground_truth": self._determine_ground_truth(i),
                "clinical_context": self._generate_context(i)
            }
            scenarios.append(scenario)

        return scenarios

    def generate_realistic_daily_alerts(self, count: int) -> List[Alert]:
        """Generate realistic daily alert patterns."""
        alerts = []

        # Morning shift: Lower alert volume
        morning_alerts = self._generate_shift_alerts(6, 14, count // 3)
        alerts.extend(morning_alerts)

        # Evening shift: Higher alert volume
        evening_alerts = self._generate_shift_alerts(14, 22, count // 2)
        alerts.extend(evening_alerts)

        # Night shift: Emergency-heavy
        night_alerts = self._generate_shift_alerts(22, 6, count // 6, emergency_rate=0.4)
        alerts.extend(night_alerts)

        return alerts


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])