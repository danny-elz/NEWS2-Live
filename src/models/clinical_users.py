"""
Clinical User Management Models for Epic 3 Alert Management
Defines user roles, contact information, and preferences for alert delivery.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID


class UserRole(Enum):
    """Clinical user roles for alert targeting."""
    ICU_NURSE = "icu_nurse"
    WARD_NURSE = "ward_nurse"
    CHARGE_NURSE = "charge_nurse"
    DOCTOR = "doctor"
    RESIDENT = "resident"
    CONSULTANT = "consultant"
    RESPIRATORY_THERAPIST = "respiratory_therapist"
    CLINICAL_PHARMACIST = "clinical_pharmacist"
    NURSING_SUPERVISOR = "nursing_supervisor"
    UNIT_COORDINATOR = "unit_coordinator"


@dataclass
class ClinicalUser:
    """Clinical user profile for alert delivery."""
    user_id: str
    name: str
    role: UserRole
    contact_info: Dict[str, str]
    shift_schedule: Dict[str, Any]
    preferences: Dict[str, Any]
    ward_assignments: Optional[List[str]] = None
    is_active: bool = True
    last_alert_ack: Optional[datetime] = None
    alert_load: int = 0

    def __post_init__(self):
        if self.ward_assignments is None:
            self.ward_assignments = []

        # Ensure contact_info has required fields
        required_fields = {
            "websocket_session": "",
            "mobile_push_token": "",
            "phone_number": "",
            "pager_number": "",
            "email": ""
        }

        for field, default in required_fields.items():
            if field not in self.contact_info:
                self.contact_info[field] = default

    def can_receive_alerts(self) -> bool:
        """Check if user can currently receive alerts."""
        return self.is_active and bool(self.contact_info)

    def get_preferred_channels(self, alert_priority: str) -> List[str]:
        """Get preferred delivery channels based on alert priority."""
        if alert_priority == "critical":
            return self.preferences.get("critical_alert_channels", ["all"])
        elif alert_priority == "urgent":
            return self.preferences.get("urgent_alert_channels", ["websocket", "push"])
        else:
            return self.preferences.get("normal_alert_channels", ["websocket"])

    def update_alert_load(self, increment: int = 1):
        """Update current alert load for load balancing."""
        self.alert_load += increment
        if self.alert_load < 0:
            self.alert_load = 0


@dataclass
class ContactPreferences:
    """User contact preferences for different alert types."""
    critical_alerts: List[str]
    urgent_alerts: List[str]
    normal_alerts: List[str]
    do_not_disturb_hours: Dict[str, str]
    escalation_timeout_minutes: int = 15
    require_confirmation: bool = True


@dataclass
class UserSession:
    """Active user session information."""
    session_id: str
    user_id: str
    connection_type: str  # websocket, mobile, etc.
    connected_at: datetime
    last_activity: datetime
    device_info: Dict[str, Any]

    def is_active(self) -> bool:
        """Check if session is currently active."""
        return (datetime.now(timezone.utc) - self.last_activity).seconds < 300  # 5 minute timeout