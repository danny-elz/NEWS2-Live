"""
Patient Detail Service for Story 3.3
Comprehensive patient data aggregation and detail views
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass, field

from src.models.patient import Patient
from src.models.patient_state import PatientState
from src.models.vital_signs import VitalSigns
from src.models.news2 import NEWS2Result, RiskCategory
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker
from src.services.news2_calculator import NEWS2Calculator
from src.services.vital_signs_history import VitalSignsHistory

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Source of vital signs data"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    DEVICE = "device"
    IMPORT = "import"


class InterventionOutcome(Enum):
    """Outcome of clinical intervention"""
    IMPROVED = "improved"
    STABLE = "stable"
    WORSENED = "worsened"
    TOO_EARLY = "too_early_to_determine"


@dataclass
class ClinicalIntervention:
    """Clinical intervention record"""
    intervention_id: str
    patient_id: str
    timestamp: datetime
    intervention_type: str
    description: str
    performed_by: str
    outcome: Optional[InterventionOutcome] = None
    notes: Optional[str] = None
    news2_before: Optional[int] = None
    news2_after: Optional[int] = None


@dataclass
class PatientDetailData:
    """Comprehensive patient detail data"""
    # Patient information
    patient_id: str
    patient_name: str
    age: int
    admission_date: datetime
    ward_id: str
    bed_number: str

    # Current status
    current_news2: Optional[int]
    risk_level: str
    last_vitals: Optional[VitalSigns]
    last_update: datetime
    data_source: DataSource

    # Clinical team
    attending_physician: str
    assigned_nurse: str
    contact_info: Dict[str, str]

    # Alerts
    active_alerts: int
    suppressed_alerts: int
    total_alerts: int

    # Timeline data
    timeline_data: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[ClinicalIntervention] = field(default_factory=list)

    # Statistics
    avg_news2_24h: Optional[float] = None
    max_news2_24h: Optional[int] = None
    min_news2_24h: Optional[int] = None
    trend_direction: Optional[str] = None


class PatientDetailService:
    """Service for managing patient detail views and clinical timelines"""

    def __init__(self,
                 patient_registry: PatientRegistry,
                 state_tracker: PatientStateTracker,
                 news2_calculator: NEWS2Calculator,
                 history_service: Optional[VitalSignsHistory] = None):
        self.patient_registry = patient_registry
        self.state_tracker = state_tracker
        self.news2_calculator = news2_calculator
        self.history_service = history_service or VitalSignsHistory()
        self._intervention_cache: Dict[str, List[ClinicalIntervention]] = {}

    async def get_patient_detail(self, patient_id: str) -> Optional[PatientDetailData]:
        """
        Get comprehensive patient detail data

        Args:
            patient_id: Patient identifier

        Returns:
            Complete patient detail data or None if not found
        """
        try:
            # Get patient base information
            patient = self.patient_registry.get_patient(patient_id)
            if not patient:
                logger.warning(f"Patient {patient_id} not found")
                return None

            # Get current state
            state = self.state_tracker.get_patient_state(patient_id)
            if not state:
                logger.warning(f"No state found for patient {patient_id}")
                return None

            # Calculate current NEWS2
            current_news2 = None
            risk_level = "unknown"
            if state.current_vitals:
                news2_result = self.news2_calculator.calculate_news2(state.current_vitals)
                if news2_result:
                    current_news2 = news2_result.total_score
                    risk_level = news2_result.risk_level.value

            # Get 24-hour timeline data
            timeline_data = await self._get_timeline_data(patient_id)

            # Get interventions
            interventions = self._get_interventions(patient_id)

            # Calculate statistics
            stats = self._calculate_statistics(timeline_data)

            # Determine data source
            data_source = DataSource.MANUAL
            if state.current_vitals and hasattr(state.current_vitals, 'is_manual_entry'):
                data_source = DataSource.MANUAL if state.current_vitals.is_manual_entry else DataSource.AUTOMATIC

            # Build patient detail data
            detail_data = PatientDetailData(
                patient_id=patient_id,
                patient_name=f"Patient {patient_id}",
                age=patient.age,
                admission_date=patient.admission_date,
                ward_id=patient.ward_id,
                bed_number=patient.bed_number,
                current_news2=current_news2,
                risk_level=risk_level,
                last_vitals=state.current_vitals,
                last_update=state.last_update or datetime.now(),
                data_source=data_source,
                attending_physician="Dr. Unknown",
                assigned_nurse=patient.assigned_nurse_id,
                contact_info={
                    "physician": "ext-1234",
                    "nurse": "ext-5678",
                    "charge_nurse": "ext-9012"
                },
                active_alerts=0,  # Would integrate with Epic 2
                suppressed_alerts=0,
                total_alerts=0,
                timeline_data=timeline_data,
                interventions=interventions,
                avg_news2_24h=stats.get("avg_news2"),
                max_news2_24h=stats.get("max_news2"),
                min_news2_24h=stats.get("min_news2"),
                trend_direction=stats.get("trend")
            )

            return detail_data

        except Exception as e:
            logger.error(f"Error getting patient detail for {patient_id}: {e}")
            return None

    async def _get_timeline_data(self, patient_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get timeline data for the specified period

        Args:
            patient_id: Patient identifier
            hours: Number of hours to retrieve (default 24)

        Returns:
            List of timeline data points
        """
        timeline_data = []

        try:
            # Get historical vital signs
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Get history from service
            history = await self.history_service.query_historical_data(
                patient_id=patient_id,
                start_time=start_time,
                end_time=end_time
            )

            for vitals in history:
                # Calculate NEWS2 for each vital signs reading
                news2_result = self.news2_calculator.calculate_news2(vitals)

                data_point = {
                    "timestamp": vitals.timestamp.isoformat() if vitals.timestamp else None,
                    "news2_score": news2_result.total_score if news2_result else None,
                    "vital_signs": {
                        "respiratory_rate": vitals.respiratory_rate,
                        "spo2": vitals.sp_o2 if hasattr(vitals, 'sp_o2') else None,
                        "on_oxygen": vitals.on_oxygen if hasattr(vitals, 'on_oxygen') else False,
                        "temperature": vitals.temperature,
                        "systolic_bp": vitals.systolic_bp if hasattr(vitals, 'systolic_bp') else None,
                        "heart_rate": vitals.heart_rate,
                        "consciousness": str(vitals.consciousness) if hasattr(vitals, 'consciousness') else "A"
                    },
                    "is_manual": getattr(vitals, 'is_manual_entry', True),
                    "confidence": getattr(vitals, 'confidence', 1.0)
                }

                timeline_data.append(data_point)

            # Sort by timestamp
            timeline_data.sort(key=lambda x: x["timestamp"] or "")

        except Exception as e:
            logger.error(f"Error getting timeline data for {patient_id}: {e}")

        return timeline_data

    def _get_interventions(self, patient_id: str) -> List[ClinicalIntervention]:
        """
        Get clinical interventions for patient

        Args:
            patient_id: Patient identifier

        Returns:
            List of clinical interventions
        """
        # Check cache first
        if patient_id in self._intervention_cache:
            return self._intervention_cache[patient_id]

        # In production, would fetch from database
        # For now, return sample interventions
        interventions = []

        # Sample intervention data
        sample_interventions = [
            ClinicalIntervention(
                intervention_id="INT001",
                patient_id=patient_id,
                timestamp=datetime.now() - timedelta(hours=6),
                intervention_type="oxygen_therapy",
                description="Started supplemental oxygen at 2L/min",
                performed_by="Nurse Johnson",
                outcome=InterventionOutcome.IMPROVED,
                notes="SpO2 improved from 92% to 96%",
                news2_before=6,
                news2_after=4
            ),
            ClinicalIntervention(
                intervention_id="INT002",
                patient_id=patient_id,
                timestamp=datetime.now() - timedelta(hours=3),
                intervention_type="medication",
                description="Administered antipyretic for fever",
                performed_by="Dr. Smith",
                outcome=InterventionOutcome.IMPROVED,
                notes="Temperature reduced from 39.5°C to 37.8°C",
                news2_before=5,
                news2_after=3
            )
        ]

        # Cache and return
        self._intervention_cache[patient_id] = sample_interventions
        return sample_interventions

    def _calculate_statistics(self, timeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from timeline data

        Args:
            timeline_data: List of timeline data points

        Returns:
            Statistics dictionary
        """
        if not timeline_data:
            return {}

        news2_scores = [
            point["news2_score"]
            for point in timeline_data
            if point.get("news2_score") is not None
        ]

        if not news2_scores:
            return {}

        # Calculate basic statistics
        avg_news2 = sum(news2_scores) / len(news2_scores)
        max_news2 = max(news2_scores)
        min_news2 = min(news2_scores)

        # Determine trend (simple linear trend)
        trend = "stable"
        if len(news2_scores) >= 2:
            recent_avg = sum(news2_scores[-3:]) / len(news2_scores[-3:])
            older_avg = sum(news2_scores[:3]) / min(3, len(news2_scores))

            if recent_avg > older_avg + 1:
                trend = "worsening"
            elif recent_avg < older_avg - 1:
                trend = "improving"

        return {
            "avg_news2": round(avg_news2, 1),
            "max_news2": max_news2,
            "min_news2": min_news2,
            "trend": trend,
            "data_points": len(news2_scores)
        }

    async def add_intervention(self, intervention: ClinicalIntervention) -> bool:
        """
        Add a new clinical intervention

        Args:
            intervention: Clinical intervention to add

        Returns:
            True if successful, False otherwise
        """
        try:
            patient_id = intervention.patient_id

            # Add to cache
            if patient_id not in self._intervention_cache:
                self._intervention_cache[patient_id] = []

            self._intervention_cache[patient_id].append(intervention)

            # In production, would also persist to database
            logger.info(f"Added intervention {intervention.intervention_id} for patient {patient_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding intervention: {e}")
            return False

    async def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient summary for handoff or documentation

        Args:
            patient_id: Patient identifier

        Returns:
            Patient summary dictionary
        """
        detail_data = await self.get_patient_detail(patient_id)
        if not detail_data:
            return {}

        # Build summary
        summary = {
            "patient_id": patient_id,
            "patient_name": detail_data.patient_name,
            "age": detail_data.age,
            "admission_date": detail_data.admission_date.isoformat(),
            "current_status": {
                "news2_score": detail_data.current_news2,
                "risk_level": detail_data.risk_level,
                "trend": detail_data.trend_direction
            },
            "24h_statistics": {
                "average_news2": detail_data.avg_news2_24h,
                "max_news2": detail_data.max_news2_24h,
                "min_news2": detail_data.min_news2_24h
            },
            "recent_interventions": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "type": i.intervention_type,
                    "description": i.description,
                    "outcome": i.outcome.value if i.outcome else None
                }
                for i in detail_data.interventions[-5:]  # Last 5 interventions
            ],
            "clinical_team": {
                "physician": detail_data.attending_physician,
                "nurse": detail_data.assigned_nurse
            },
            "generated_at": datetime.now().isoformat()
        }

        return summary

    def get_zoom_levels(self) -> List[Tuple[str, int]]:
        """
        Get available timeline zoom levels

        Returns:
            List of (label, hours) tuples
        """
        return [
            ("4 hours", 4),
            ("8 hours", 8),
            ("12 hours", 12),
            ("24 hours", 24),
            ("48 hours", 48),
            ("7 days", 168)
        ]

    def get_vital_sign_parameters(self) -> List[Dict[str, Any]]:
        """
        Get available vital sign parameters for display

        Returns:
            List of parameter configurations
        """
        return [
            {"key": "respiratory_rate", "label": "RR", "unit": "bpm", "color": "#FF6B6B"},
            {"key": "heart_rate", "label": "HR", "unit": "bpm", "color": "#4ECDC4"},
            {"key": "systolic_bp", "label": "BP", "unit": "mmHg", "color": "#45B7D1"},
            {"key": "temperature", "label": "Temp", "unit": "°C", "color": "#FFA07A"},
            {"key": "spo2", "label": "SpO2", "unit": "%", "color": "#98D8C8"},
            {"key": "news2_score", "label": "NEWS2", "unit": "", "color": "#F7DC6F"}
        ]