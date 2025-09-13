"""
Clinical Metrics Service for Story 3.5
Provides specialized clinical outcome analytics and metrics
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend analysis directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    UNKNOWN = "unknown"


class OutcomeType(Enum):
    """Patient outcome categories"""
    DISCHARGED_HOME = "discharged_home"
    TRANSFERRED = "transferred"
    DECEASED = "deceased"
    STILL_ADMITTED = "still_admitted"


@dataclass
class NEWS2TrendAnalysis:
    """NEWS2 score trend analysis result"""
    patient_id: str
    time_period: str
    trend_direction: TrendDirection
    slope: float
    correlation_coefficient: float
    average_score: float
    score_variance: float
    data_points: int
    confidence_level: float
    clinical_significance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "time_period": self.time_period,
            "trend_direction": self.trend_direction.value,
            "slope": self.slope,
            "correlation_coefficient": self.correlation_coefficient,
            "average_score": self.average_score,
            "score_variance": self.score_variance,
            "data_points": self.data_points,
            "confidence_level": self.confidence_level,
            "clinical_significance": self.clinical_significance
        }


@dataclass
class InterventionOutcome:
    """Clinical intervention outcome analysis"""
    intervention_id: str
    intervention_type: str
    patient_id: str
    timestamp: datetime
    news2_before: float
    news2_after: float
    score_improvement: float
    time_to_improvement_hours: float
    effectiveness_score: float
    clinical_notes: str
    staff_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "news2_before": self.news2_before,
            "news2_after": self.news2_after,
            "score_improvement": self.score_improvement,
            "time_to_improvement_hours": self.time_to_improvement_hours,
            "effectiveness_score": self.effectiveness_score,
            "clinical_notes": self.clinical_notes,
            "staff_id": self.staff_id
        }


@dataclass
class RiskStratification:
    """Patient risk stratification analysis"""
    risk_category: str
    patient_count: int
    avg_news2_score: float
    mortality_rate: float
    avg_length_of_stay: float
    readmission_rate_30_days: float
    common_complications: List[str]
    recommended_monitoring_frequency: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_category": self.risk_category,
            "patient_count": self.patient_count,
            "avg_news2_score": self.avg_news2_score,
            "mortality_rate": self.mortality_rate,
            "avg_length_of_stay": self.avg_length_of_stay,
            "readmission_rate_30_days": self.readmission_rate_30_days,
            "common_complications": self.common_complications,
            "recommended_monitoring_frequency": self.recommended_monitoring_frequency
        }


class ClinicalMetricsService:
    """Service for clinical outcome analytics and metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_ttl = 600  # 10 minutes

    async def analyze_news2_trends(self, patient_ids: List[str] = None,
                                 time_period: str = "24h") -> List[NEWS2TrendAnalysis]:
        """Analyze NEWS2 score trends for patients"""
        try:
            # Get patient data (simulated)
            patient_data = await self._get_patient_news2_data(patient_ids, time_period)

            trends = []
            for patient_id, scores in patient_data.items():
                if len(scores) < 3:  # Need minimum 3 points for trend analysis
                    continue

                trend_analysis = await self._calculate_news2_trend(patient_id, scores, time_period)
                trends.append(trend_analysis)

            return trends

        except Exception as e:
            self.logger.error(f"Error analyzing NEWS2 trends: {e}")
            raise

    async def _calculate_news2_trend(self, patient_id: str,
                                   scores: List[Tuple[datetime, float]],
                                   time_period: str) -> NEWS2TrendAnalysis:
        """Calculate trend analysis for individual patient"""
        # Extract values and timestamps
        timestamps = [score[0] for score in scores]
        values = [score[1] for score in scores]

        # Convert timestamps to hours since first measurement
        first_time = timestamps[0]
        hours = [(ts - first_time).total_seconds() / 3600 for ts in timestamps]

        # Calculate linear regression (simplified)
        n = len(values)
        sum_x = sum(hours)
        sum_y = sum(values)
        sum_xy = sum(h * v for h, v in zip(hours, values))
        sum_x2 = sum(h * h for h in hours)

        # Slope calculation
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope = 0
            intercept = statistics.mean(values)

        # Calculate correlation coefficient
        if len(values) > 1:
            try:
                mean_x = statistics.mean(hours)
                mean_y = statistics.mean(values)

                numerator = sum((h - mean_x) * (v - mean_y) for h, v in zip(hours, values))

                sum_sq_x = sum((h - mean_x) ** 2 for h in hours)
                sum_sq_y = sum((v - mean_y) ** 2 for v in values)

                if sum_sq_x > 0 and sum_sq_y > 0:
                    correlation = numerator / (sum_sq_x * sum_sq_y) ** 0.5
                else:
                    correlation = 0
            except:
                correlation = 0
        else:
            correlation = 0

        # Determine trend direction
        if abs(slope) < 0.1:
            trend_direction = TrendDirection.STABLE
        elif slope > 0.1:
            trend_direction = TrendDirection.DETERIORATING
        else:
            trend_direction = TrendDirection.IMPROVING

        # Clinical significance assessment
        avg_score = statistics.mean(values)
        if avg_score > 7:
            clinical_significance = "High risk - requires immediate attention"
        elif avg_score > 5:
            clinical_significance = "Medium risk - increased monitoring needed"
        elif trend_direction == TrendDirection.DETERIORATING:
            clinical_significance = "Deteriorating trend - consider intervention"
        else:
            clinical_significance = "Normal monitoring required"

        return NEWS2TrendAnalysis(
            patient_id=patient_id,
            time_period=time_period,
            trend_direction=trend_direction,
            slope=slope,
            correlation_coefficient=correlation,
            average_score=avg_score,
            score_variance=statistics.variance(values) if len(values) > 1 else 0,
            data_points=len(values),
            confidence_level=min(0.95, 0.5 + abs(correlation) * 0.5),
            clinical_significance=clinical_significance
        )

    async def analyze_intervention_outcomes(self,
                                          intervention_types: List[str] = None,
                                          time_range: Tuple[datetime, datetime] = None) -> List[InterventionOutcome]:
        """Analyze clinical intervention effectiveness"""
        try:
            # Get intervention data (simulated)
            interventions_data = await self._get_interventions_data(intervention_types, time_range)

            outcomes = []
            for intervention in interventions_data:
                outcome = await self._analyze_single_intervention(intervention)
                outcomes.append(outcome)

            return outcomes

        except Exception as e:
            self.logger.error(f"Error analyzing intervention outcomes: {e}")
            raise

    async def _analyze_single_intervention(self, intervention_data: Dict[str, Any]) -> InterventionOutcome:
        """Analyze outcome of single intervention"""
        # Calculate improvement metrics
        news2_before = intervention_data["news2_before"]
        news2_after = intervention_data["news2_after"]
        score_improvement = news2_before - news2_after

        # Calculate effectiveness score (0-1)
        max_possible_improvement = max(news2_before - 0, 1)  # Can't go below 0
        effectiveness_score = min(1.0, max(0, score_improvement / max_possible_improvement))

        return InterventionOutcome(
            intervention_id=intervention_data["intervention_id"],
            intervention_type=intervention_data["intervention_type"],
            patient_id=intervention_data["patient_id"],
            timestamp=intervention_data["timestamp"],
            news2_before=news2_before,
            news2_after=news2_after,
            score_improvement=score_improvement,
            time_to_improvement_hours=intervention_data.get("time_to_improvement", 2.0),
            effectiveness_score=effectiveness_score,
            clinical_notes=intervention_data.get("notes", ""),
            staff_id=intervention_data.get("staff_id", "unknown")
        )

    async def stratify_patient_risk(self, ward_ids: List[str] = None,
                                  time_range: Tuple[datetime, datetime] = None) -> List[RiskStratification]:
        """Perform patient risk stratification analysis"""
        try:
            # Get patient outcome data (simulated)
            patient_data = await self._get_patient_outcome_data(ward_ids, time_range)

            # Group patients by risk categories
            risk_groups = {
                "low": [],     # NEWS2 < 5
                "medium": [],  # NEWS2 5-6
                "high": []     # NEWS2 >= 7
            }

            for patient in patient_data:
                avg_news2 = patient.get("avg_news2_score", 0)
                if avg_news2 < 5:
                    risk_groups["low"].append(patient)
                elif avg_news2 < 7:
                    risk_groups["medium"].append(patient)
                else:
                    risk_groups["high"].append(patient)

            stratifications = []
            for risk_category, patients in risk_groups.items():
                if patients:
                    stratification = await self._calculate_risk_stratification(risk_category, patients)
                    stratifications.append(stratification)

            return stratifications

        except Exception as e:
            self.logger.error(f"Error in risk stratification: {e}")
            raise

    async def _calculate_risk_stratification(self, risk_category: str,
                                           patients: List[Dict[str, Any]]) -> RiskStratification:
        """Calculate risk stratification metrics for patient group"""
        if not patients:
            return RiskStratification(
                risk_category=risk_category,
                patient_count=0,
                avg_news2_score=0,
                mortality_rate=0,
                avg_length_of_stay=0,
                readmission_rate_30_days=0,
                common_complications=[],
                recommended_monitoring_frequency="standard"
            )

        # Calculate metrics
        avg_news2 = statistics.mean([p.get("avg_news2_score", 0) for p in patients])
        mortality_rate = len([p for p in patients if p.get("outcome") == "deceased"]) / len(patients)
        avg_los = statistics.mean([p.get("length_of_stay", 5) for p in patients])
        readmission_rate = len([p for p in patients if p.get("readmitted_30_days", False)]) / len(patients)

        # Common complications (simulated based on risk category)
        complications_map = {
            "low": ["Minor respiratory issues", "Dehydration"],
            "medium": ["Pneumonia", "UTI", "Falls", "Medication errors"],
            "high": ["Sepsis", "Cardiac events", "Respiratory failure", "Multi-organ failure"]
        }

        monitoring_frequency = {
            "low": "Every 12 hours",
            "medium": "Every 4-6 hours",
            "high": "Continuous monitoring"
        }

        return RiskStratification(
            risk_category=risk_category,
            patient_count=len(patients),
            avg_news2_score=round(avg_news2, 2),
            mortality_rate=round(mortality_rate * 100, 2),  # Convert to percentage
            avg_length_of_stay=round(avg_los, 1),
            readmission_rate_30_days=round(readmission_rate * 100, 2),  # Convert to percentage
            common_complications=complications_map.get(risk_category, []),
            recommended_monitoring_frequency=monitoring_frequency.get(risk_category, "standard")
        )

    async def _get_patient_news2_data(self, patient_ids: List[str] = None,
                                    time_period: str = "24h") -> Dict[str, List[Tuple[datetime, float]]]:
        """Get simulated patient NEWS2 data"""
        # Simulate NEWS2 score data for patients
        now = datetime.now()
        hours_back = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}.get(time_period, 24)

        patient_data = {}
        patients = patient_ids or [f"P{i:03d}" for i in range(1, 11)]  # 10 patients

        for patient_id in patients:
            scores = []
            current_time = now - timedelta(hours=hours_back)

            # Generate trend-based scores
            base_score = 2 + (hash(patient_id) % 4)  # 2-5 base score
            trend = (hash(patient_id + "trend") % 3) - 1  # -1, 0, or 1

            while current_time <= now:
                # Add some random variation
                variation = (hash(current_time.isoformat() + patient_id) % 3) - 1
                score = max(0, min(15, base_score + (trend * 0.1) + variation))
                scores.append((current_time, score))
                current_time += timedelta(hours=1)

            patient_data[patient_id] = scores

        return patient_data

    async def _get_interventions_data(self, intervention_types: List[str] = None,
                                    time_range: Tuple[datetime, datetime] = None) -> List[Dict[str, Any]]:
        """Get simulated intervention data"""
        now = datetime.now()
        start_time = time_range[0] if time_range else now - timedelta(days=7)
        end_time = time_range[1] if time_range else now

        intervention_types = intervention_types or [
            "oxygen_therapy", "medication_adjustment", "fluid_therapy",
            "position_change", "respiratory_support"
        ]

        interventions = []
        for i in range(50):  # 50 interventions
            intervention_time = start_time + timedelta(
                seconds=(end_time - start_time).total_seconds() * (i / 50)
            )

            intervention_type = intervention_types[i % len(intervention_types)]
            patient_id = f"P{(i % 20):03d}"

            # Simulate before/after scores
            news2_before = 3 + (i % 8)  # 3-10
            improvement = max(0, (hash(f"{i}{intervention_type}") % 4))  # 0-3 improvement
            news2_after = max(0, news2_before - improvement)

            interventions.append({
                "intervention_id": f"INT{i:04d}",
                "intervention_type": intervention_type,
                "patient_id": patient_id,
                "timestamp": intervention_time,
                "news2_before": news2_before,
                "news2_after": news2_after,
                "time_to_improvement": 1 + (i % 6),  # 1-6 hours
                "notes": f"Standard {intervention_type} protocol applied",
                "staff_id": f"staff_{(i % 10):02d}"
            })

        return interventions

    async def _get_patient_outcome_data(self, ward_ids: List[str] = None,
                                      time_range: Tuple[datetime, datetime] = None) -> List[Dict[str, Any]]:
        """Get simulated patient outcome data"""
        now = datetime.now()
        start_time = time_range[0] if time_range else now - timedelta(days=30)

        ward_ids = ward_ids or ["ward_a", "ward_b", "ward_c"]

        patients = []
        for i in range(100):  # 100 patients
            avg_news2 = 1 + (i % 10)  # 1-10 score range
            length_of_stay = 2 + (i % 15)  # 2-16 days

            # Outcome based on NEWS2 score (higher scores = worse outcomes)
            outcome_rand = hash(f"outcome_{i}") % 100
            if avg_news2 >= 7:
                if outcome_rand < 15:  # 15% mortality for high risk
                    outcome = "deceased"
                elif outcome_rand < 35:  # 20% transfer
                    outcome = "transferred"
                else:
                    outcome = "discharged_home"
            elif avg_news2 >= 5:
                if outcome_rand < 5:  # 5% mortality for medium risk
                    outcome = "deceased"
                elif outcome_rand < 20:  # 15% transfer
                    outcome = "transferred"
                else:
                    outcome = "discharged_home"
            else:
                if outcome_rand < 2:  # 2% mortality for low risk
                    outcome = "deceased"
                elif outcome_rand < 10:  # 8% transfer
                    outcome = "transferred"
                else:
                    outcome = "discharged_home"

            patients.append({
                "patient_id": f"P{i:03d}",
                "ward_id": ward_ids[i % len(ward_ids)],
                "avg_news2_score": avg_news2,
                "length_of_stay": length_of_stay,
                "outcome": outcome,
                "readmitted_30_days": (i % 15) == 0,  # ~6.7% readmission rate
                "admission_date": start_time + timedelta(days=i % 30)
            })

        return patients

    async def get_clinical_metrics_summary(self, ward_ids: List[str] = None) -> Dict[str, Any]:
        """Get summary of all clinical metrics"""
        try:
            # Run all analyses
            trends = await self.analyze_news2_trends(time_period="24h")
            risk_stratification = await self.stratify_patient_risk(ward_ids)

            # Calculate summary metrics
            deteriorating_patients = len([t for t in trends if t.trend_direction == TrendDirection.DETERIORATING])
            improving_patients = len([t for t in trends if t.trend_direction == TrendDirection.IMPROVING])

            return {
                "news2_trends": {
                    "total_patients_analyzed": len(trends),
                    "deteriorating_patients": deteriorating_patients,
                    "improving_patients": improving_patients,
                    "stable_patients": len(trends) - deteriorating_patients - improving_patients
                },
                "risk_stratification": [rs.to_dict() for rs in risk_stratification],
                "overall_metrics": {
                    "avg_patient_acuity": statistics.mean([rs.avg_news2_score for rs in risk_stratification]),
                    "high_risk_percentage": next(
                        (rs.patient_count for rs in risk_stratification if rs.risk_category == "high"), 0
                    ) * 100 / sum(rs.patient_count for rs in risk_stratification) if risk_stratification else 0
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating clinical metrics summary: {e}")
            raise