"""
Patient Detail View module for Epic 3
Comprehensive patient views with clinical timelines
"""

from src.dashboard.patient_detail.patient_detail_service import (
    PatientDetailService,
    PatientDetailData,
    ClinicalIntervention,
    InterventionOutcome,
    DataSource
)
from src.dashboard.patient_detail.timeline_service import (
    TimelineService,
    TimeRange,
    TrendDirection
)
from src.dashboard.patient_detail.clinical_actions import (
    ClinicalActionsService,
    QuickAction,
    ActionType,
    EscalationLevel
)

__all__ = [
    "PatientDetailService",
    "PatientDetailData",
    "ClinicalIntervention",
    "InterventionOutcome",
    "DataSource",
    "TimelineService",
    "TimeRange",
    "TrendDirection",
    "ClinicalActionsService",
    "QuickAction",
    "ActionType",
    "EscalationLevel"
]