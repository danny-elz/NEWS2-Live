"""
Unit tests for Clinical Metrics Service (Story 3.5)
Tests specialized clinical outcome analytics and metrics
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analytics.clinical_metrics import (
    ClinicalMetricsService,
    NEWS2TrendAnalysis,
    InterventionOutcome,
    RiskStratification,
    TrendDirection,
    OutcomeType
)


class TestClinicalMetricsService:
    """Test suite for ClinicalMetricsService"""

    @pytest.fixture
    def clinical_service(self):
        """Create ClinicalMetricsService instance"""
        return ClinicalMetricsService()

    @pytest.fixture
    def sample_patient_ids(self):
        """Sample patient IDs for testing"""
        return ["P001", "P002", "P003"]

    @pytest.mark.asyncio
    async def test_analyze_news2_trends(self, clinical_service, sample_patient_ids):
        """Test NEWS2 trend analysis"""
        trends = await clinical_service.analyze_news2_trends(
            patient_ids=sample_patient_ids,
            time_period="24h"
        )

        assert len(trends) > 0

        for trend in trends:
            assert isinstance(trend, NEWS2TrendAnalysis)
            assert trend.patient_id in sample_patient_ids
            assert trend.time_period == "24h"
            assert isinstance(trend.trend_direction, TrendDirection)
            assert isinstance(trend.slope, float)
            assert -1 <= trend.correlation_coefficient <= 1
            assert trend.average_score >= 0
            assert trend.data_points > 0
            assert 0 <= trend.confidence_level <= 1

    @pytest.mark.asyncio
    async def test_news2_trend_analysis_insufficient_data(self, clinical_service):
        """Test trend analysis with insufficient data points"""
        # Mock patient data with only 2 data points (less than minimum 3)
        with patch.object(clinical_service, '_get_patient_news2_data') as mock_data:
            mock_data.return_value = {
                "P001": [(datetime.now(), 3.0), (datetime.now(), 3.2)]  # Only 2 points
            }

            trends = await clinical_service.analyze_news2_trends(["P001"])

            # Should return empty list due to insufficient data
            assert len(trends) == 0

    @pytest.mark.asyncio
    async def test_trend_direction_classification(self, clinical_service):
        """Test trend direction classification"""
        # Mock data with different trend patterns
        improving_data = {
            "P001": [
                (datetime.now() - timedelta(hours=4), 5.0),
                (datetime.now() - timedelta(hours=2), 4.0),
                (datetime.now(), 3.0)
            ]
        }

        stable_data = {
            "P002": [
                (datetime.now() - timedelta(hours=4), 3.0),
                (datetime.now() - timedelta(hours=2), 3.1),
                (datetime.now(), 3.0)
            ]
        }

        deteriorating_data = {
            "P003": [
                (datetime.now() - timedelta(hours=4), 2.0),
                (datetime.now() - timedelta(hours=2), 4.0),
                (datetime.now(), 6.0)
            ]
        }

        # Test improving trend
        with patch.object(clinical_service, '_get_patient_news2_data', return_value=improving_data):
            trends = await clinical_service.analyze_news2_trends(["P001"])
            assert trends[0].trend_direction == TrendDirection.IMPROVING

        # Test stable trend
        with patch.object(clinical_service, '_get_patient_news2_data', return_value=stable_data):
            trends = await clinical_service.analyze_news2_trends(["P002"])
            assert trends[0].trend_direction == TrendDirection.STABLE

        # Test deteriorating trend
        with patch.object(clinical_service, '_get_patient_news2_data', return_value=deteriorating_data):
            trends = await clinical_service.analyze_news2_trends(["P003"])
            assert trends[0].trend_direction == TrendDirection.DETERIORATING

    @pytest.mark.asyncio
    async def test_analyze_intervention_outcomes(self, clinical_service):
        """Test intervention outcome analysis"""
        interventions = await clinical_service.analyze_intervention_outcomes(
            intervention_types=["oxygen_therapy", "medication_adjustment"],
            time_range=(datetime.now() - timedelta(days=7), datetime.now())
        )

        assert len(interventions) > 0

        for intervention in interventions:
            assert isinstance(intervention, InterventionOutcome)
            assert intervention.intervention_type in ["oxygen_therapy", "medication_adjustment",
                                                    "fluid_therapy", "position_change", "respiratory_support"]
            assert intervention.news2_before >= 0
            assert intervention.news2_after >= 0
            assert intervention.score_improvement == intervention.news2_before - intervention.news2_after
            assert 0 <= intervention.effectiveness_score <= 1
            assert intervention.time_to_improvement_hours > 0

    @pytest.mark.asyncio
    async def test_intervention_effectiveness_calculation(self, clinical_service):
        """Test intervention effectiveness score calculation"""
        # Mock intervention data with known values
        mock_intervention = {
            "intervention_id": "INT001",
            "intervention_type": "oxygen_therapy",
            "patient_id": "P001",
            "timestamp": datetime.now(),
            "news2_before": 8.0,
            "news2_after": 5.0,  # 3-point improvement
            "time_to_improvement": 2.0,
            "notes": "Test intervention",
            "staff_id": "staff_01"
        }

        outcome = await clinical_service._analyze_single_intervention(mock_intervention)

        assert outcome.news2_before == 8.0
        assert outcome.news2_after == 5.0
        assert outcome.score_improvement == 3.0
        assert outcome.effectiveness_score > 0  # Should have positive effectiveness

        # Test no improvement scenario
        mock_no_improvement = {
            **mock_intervention,
            "news2_before": 5.0,
            "news2_after": 5.0  # No change
        }

        outcome_no_improvement = await clinical_service._analyze_single_intervention(mock_no_improvement)
        assert outcome_no_improvement.effectiveness_score == 0

    @pytest.mark.asyncio
    async def test_stratify_patient_risk(self, clinical_service):
        """Test patient risk stratification"""
        stratifications = await clinical_service.stratify_patient_risk(
            ward_ids=["ward_a", "ward_b"]
        )

        assert len(stratifications) > 0

        # Should have stratifications for different risk levels
        risk_categories = [s.risk_category for s in stratifications]
        expected_categories = ["low", "medium", "high"]

        for category in expected_categories:
            if category in risk_categories:
                stratification = next(s for s in stratifications if s.risk_category == category)

                assert isinstance(stratification, RiskStratification)
                assert stratification.patient_count >= 0
                assert stratification.avg_news2_score >= 0
                assert 0 <= stratification.mortality_rate <= 100
                assert stratification.avg_length_of_stay > 0
                assert 0 <= stratification.readmission_rate_30_days <= 100
                assert isinstance(stratification.common_complications, list)
                assert stratification.recommended_monitoring_frequency in [
                    "Every 12 hours", "Every 4-6 hours", "Continuous monitoring"
                ]

    @pytest.mark.asyncio
    async def test_risk_stratification_categories(self, clinical_service):
        """Test risk stratification category assignment"""
        # Mock patient data with known NEWS2 scores
        mock_patients = [
            {"patient_id": "P001", "avg_news2_score": 2.0, "outcome": "discharged_home",
             "length_of_stay": 3, "readmitted_30_days": False},
            {"patient_id": "P002", "avg_news2_score": 6.0, "outcome": "discharged_home",
             "length_of_stay": 5, "readmitted_30_days": False},
            {"patient_id": "P003", "avg_news2_score": 9.0, "outcome": "deceased",
             "length_of_stay": 12, "readmitted_30_days": False}
        ]

        with patch.object(clinical_service, '_get_patient_outcome_data', return_value=mock_patients):
            stratifications = await clinical_service.stratify_patient_risk()

            # Should have different risk categories
            categories = {s.risk_category for s in stratifications}
            assert "low" in categories
            assert "medium" in categories
            assert "high" in categories

            # High risk should have higher mortality rate
            high_risk = next(s for s in stratifications if s.risk_category == "high")
            low_risk = next(s for s in stratifications if s.risk_category == "low")
            assert high_risk.mortality_rate > low_risk.mortality_rate

    @pytest.mark.asyncio
    async def test_get_clinical_metrics_summary(self, clinical_service):
        """Test clinical metrics summary generation"""
        summary = await clinical_service.get_clinical_metrics_summary(ward_ids=["ward_a"])

        assert "news2_trends" in summary
        assert "risk_stratification" in summary
        assert "overall_metrics" in summary
        assert "generated_at" in summary

        # Verify NEWS2 trends summary
        trends_summary = summary["news2_trends"]
        assert "total_patients_analyzed" in trends_summary
        assert "deteriorating_patients" in trends_summary
        assert "improving_patients" in trends_summary
        assert "stable_patients" in trends_summary

        # Verify overall metrics
        overall_metrics = summary["overall_metrics"]
        assert "avg_patient_acuity" in overall_metrics
        assert "high_risk_percentage" in overall_metrics

    def test_news2_trend_analysis_serialization(self):
        """Test NEWS2TrendAnalysis serialization"""
        trend = NEWS2TrendAnalysis(
            patient_id="P001",
            time_period="24h",
            trend_direction=TrendDirection.IMPROVING,
            slope=-0.5,
            correlation_coefficient=0.85,
            average_score=3.2,
            score_variance=0.8,
            data_points=24,
            confidence_level=0.9,
            clinical_significance="Improving trend - continue monitoring"
        )

        trend_dict = trend.to_dict()

        assert trend_dict["patient_id"] == "P001"
        assert trend_dict["trend_direction"] == "improving"
        assert trend_dict["slope"] == -0.5
        assert trend_dict["correlation_coefficient"] == 0.85
        assert trend_dict["clinical_significance"] == "Improving trend - continue monitoring"

    def test_intervention_outcome_serialization(self):
        """Test InterventionOutcome serialization"""
        outcome = InterventionOutcome(
            intervention_id="INT001",
            intervention_type="oxygen_therapy",
            patient_id="P001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            news2_before=6.0,
            news2_after=4.0,
            score_improvement=2.0,
            time_to_improvement_hours=3.5,
            effectiveness_score=0.8,
            clinical_notes="Oxygen therapy effective",
            staff_id="staff_01"
        )

        outcome_dict = outcome.to_dict()

        assert outcome_dict["intervention_id"] == "INT001"
        assert outcome_dict["intervention_type"] == "oxygen_therapy"
        assert outcome_dict["score_improvement"] == 2.0
        assert outcome_dict["effectiveness_score"] == 0.8
        assert "2024-01-01T12:00:00" in outcome_dict["timestamp"]

    def test_risk_stratification_serialization(self):
        """Test RiskStratification serialization"""
        stratification = RiskStratification(
            risk_category="high",
            patient_count=15,
            avg_news2_score=8.2,
            mortality_rate=12.5,
            avg_length_of_stay=9.3,
            readmission_rate_30_days=18.7,
            common_complications=["Sepsis", "Respiratory failure"],
            recommended_monitoring_frequency="Continuous monitoring"
        )

        strat_dict = stratification.to_dict()

        assert strat_dict["risk_category"] == "high"
        assert strat_dict["patient_count"] == 15
        assert strat_dict["mortality_rate"] == 12.5
        assert strat_dict["common_complications"] == ["Sepsis", "Respiratory failure"]


class TestClinicalMetricsDataGeneration:
    """Test clinical metrics data generation methods"""

    @pytest.fixture
    def clinical_service(self):
        """Create ClinicalMetricsService instance"""
        return ClinicalMetricsService()

    @pytest.mark.asyncio
    async def test_get_patient_news2_data(self, clinical_service):
        """Test patient NEWS2 data generation"""
        data = await clinical_service._get_patient_news2_data(
            patient_ids=["P001", "P002"],
            time_period="24h"
        )

        assert len(data) == 2
        assert "P001" in data
        assert "P002" in data

        for patient_id, scores in data.items():
            assert len(scores) > 0
            for timestamp, score in scores:
                assert isinstance(timestamp, datetime)
                assert 0 <= score <= 15  # Valid NEWS2 range

    @pytest.mark.asyncio
    async def test_get_interventions_data(self, clinical_service):
        """Test interventions data generation"""
        now = datetime.now()
        time_range = (now - timedelta(days=1), now)

        interventions = await clinical_service._get_interventions_data(
            intervention_types=["oxygen_therapy"],
            time_range=time_range
        )

        assert len(interventions) > 0

        for intervention in interventions:
            assert intervention["intervention_type"] == "oxygen_therapy"
            assert intervention["news2_before"] >= intervention["news2_after"]
            assert time_range[0] <= intervention["timestamp"] <= time_range[1]

    @pytest.mark.asyncio
    async def test_get_patient_outcome_data(self, clinical_service):
        """Test patient outcome data generation"""
        now = datetime.now()
        time_range = (now - timedelta(days=30), now)

        patients = await clinical_service._get_patient_outcome_data(
            ward_ids=["ward_a"],
            time_range=time_range
        )

        assert len(patients) > 0

        for patient in patients:
            assert "patient_id" in patient
            assert "ward_id" in patient
            assert patient["ward_id"] == "ward_a"
            assert patient["avg_news2_score"] >= 1
            assert patient["length_of_stay"] >= 2
            assert patient["outcome"] in ["discharged_home", "transferred", "deceased"]


class TestTrendDirectionEnum:
    """Test TrendDirection enum"""

    def test_trend_direction_values(self):
        """Test TrendDirection enum values"""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DETERIORATING.value == "deteriorating"
        assert TrendDirection.UNKNOWN.value == "unknown"


class TestOutcomeTypeEnum:
    """Test OutcomeType enum"""

    def test_outcome_type_values(self):
        """Test OutcomeType enum values"""
        assert OutcomeType.DISCHARGED_HOME.value == "discharged_home"
        assert OutcomeType.TRANSFERRED.value == "transferred"
        assert OutcomeType.DECEASED.value == "deceased"
        assert OutcomeType.STILL_ADMITTED.value == "still_admitted"


class TestClinicalMetricsErrorHandling:
    """Test error handling in clinical metrics service"""

    @pytest.fixture
    def clinical_service(self):
        """Create ClinicalMetricsService instance"""
        return ClinicalMetricsService()

    @pytest.mark.asyncio
    async def test_news2_analysis_exception_handling(self, clinical_service):
        """Test exception handling in NEWS2 analysis"""
        with patch.object(clinical_service, '_get_patient_news2_data',
                         side_effect=Exception("Data access error")):
            with pytest.raises(Exception):
                await clinical_service.analyze_news2_trends(["P001"])

    @pytest.mark.asyncio
    async def test_intervention_analysis_exception_handling(self, clinical_service):
        """Test exception handling in intervention analysis"""
        with patch.object(clinical_service, '_get_interventions_data',
                         side_effect=Exception("Database error")):
            with pytest.raises(Exception):
                await clinical_service.analyze_intervention_outcomes()

    @pytest.mark.asyncio
    async def test_risk_stratification_exception_handling(self, clinical_service):
        """Test exception handling in risk stratification"""
        with patch.object(clinical_service, '_get_patient_outcome_data',
                         side_effect=Exception("Connection timeout")):
            with pytest.raises(Exception):
                await clinical_service.stratify_patient_risk()

    @pytest.mark.asyncio
    async def test_clinical_significance_assessment(self, clinical_service):
        """Test clinical significance assessment for different score ranges"""
        # Test high-risk significance
        high_risk_data = {
            "P001": [
                (datetime.now() - timedelta(hours=2), 8.0),
                (datetime.now() - timedelta(hours=1), 8.5),
                (datetime.now(), 9.0)
            ]
        }

        with patch.object(clinical_service, '_get_patient_news2_data', return_value=high_risk_data):
            trends = await clinical_service.analyze_news2_trends(["P001"])
            assert "High risk" in trends[0].clinical_significance

        # Test medium-risk significance
        medium_risk_data = {
            "P002": [
                (datetime.now() - timedelta(hours=2), 5.5),
                (datetime.now() - timedelta(hours=1), 6.0),
                (datetime.now(), 6.5)
            ]
        }

        with patch.object(clinical_service, '_get_patient_news2_data', return_value=medium_risk_data):
            trends = await clinical_service.analyze_news2_trends(["P002"])
            assert "Medium risk" in trends[0].clinical_significance