"""
Alert Generation Models for NEWS2 Live System

This module defines the core data models for the alert generation system,
including alert levels, alert decisions, escalation matrices, and audit trails.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4

from .news2 import NEWS2Result
from .patient import Patient


class AlertLevel(Enum):
    """Alert severity levels based on NEWS2 scores and clinical parameters."""
    LOW = "low"           # NEWS2 0-2: Routine monitoring
    MEDIUM = "medium"     # NEWS2 3-4: Increased monitoring
    HIGH = "high"         # NEWS2 5-6: Urgent medical review
    CRITICAL = "critical" # NEWS2 7+ or single parameter = 3: Immediate response


class AlertPriority(Enum):
    """Alert priority for escalation and delivery."""
    ROUTINE = "routine"
    URGENT = "urgent"
    IMMEDIATE = "immediate"
    LIFE_THREATENING = "life_threatening"


class EscalationRole(Enum):
    """Healthcare roles in escalation matrix."""
    WARD_NURSE = "ward_nurse"
    CHARGE_NURSE = "charge_nurse"
    DOCTOR = "doctor"
    RAPID_RESPONSE = "rapid_response"
    CONSULTANT = "consultant"


class AlertStatus(Enum):
    """Current status of an alert."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertThreshold:
    """Configuration for ward-specific alert thresholds."""
    threshold_id: UUID
    ward_id: str
    alert_level: AlertLevel
    news2_min: int
    news2_max: Optional[int]
    single_param_trigger: bool
    active_hours: Tuple[int, int]  # (start_hour, end_hour) in 24h format
    enabled: bool
    created_at: datetime
    updated_at: datetime
    
    def is_active_now(self) -> bool:
        """Check if threshold is active for current time."""
        if not self.enabled:
            return False
        
        current_hour = datetime.now().hour
        start_hour, end_hour = self.active_hours
        
        if start_hour <= end_hour:
            # Normal range (e.g., 8-17)
            return start_hour <= current_hour < end_hour
        else:
            # Overnight range (e.g., 22-6)
            return current_hour >= start_hour or current_hour < end_hour
    
    def matches_news2_score(self, news2_score: int) -> bool:
        """Check if NEWS2 score matches this threshold."""
        if self.news2_max is None:
            return news2_score >= self.news2_min
        else:
            return self.news2_min <= news2_score <= self.news2_max
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "threshold_id": str(self.threshold_id),
            "ward_id": self.ward_id,
            "alert_level": self.alert_level.value,
            "news2_min": self.news2_min,
            "news2_max": self.news2_max,
            "single_param_trigger": self.single_param_trigger,
            "active_hours": self.active_hours,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class EscalationStep:
    """Single step in escalation matrix."""
    role: EscalationRole
    delay_minutes: int
    max_attempts: int = 3
    retry_interval_minutes: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "role": self.role.value,
            "delay_minutes": self.delay_minutes,
            "max_attempts": self.max_attempts,
            "retry_interval_minutes": self.retry_interval_minutes
        }


@dataclass
class EscalationMatrix:
    """Complete escalation matrix for a ward and alert level."""
    matrix_id: UUID
    ward_id: str
    alert_level: AlertLevel
    escalation_steps: List[EscalationStep]
    enabled: bool
    created_at: datetime
    updated_at: datetime
    
    def get_next_escalation(self, current_step: int) -> Optional[EscalationStep]:
        """Get next escalation step if available."""
        if current_step < len(self.escalation_steps):
            return self.escalation_steps[current_step]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "matrix_id": str(self.matrix_id),
            "ward_id": self.ward_id,
            "alert_level": self.alert_level.value,
            "escalation_steps": [step.to_dict() for step in self.escalation_steps],
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class AlertDecision:
    """Decision made by alert generation engine."""
    decision_id: UUID
    patient_id: str
    news2_result: NEWS2Result
    alert_level: AlertLevel
    alert_priority: AlertPriority
    threshold_applied: Optional[AlertThreshold]
    reasoning: str
    decision_timestamp: datetime
    generation_latency_ms: float
    single_param_trigger: bool
    suppressed: bool
    ward_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "decision_id": str(self.decision_id),
            "patient_id": self.patient_id,
            "news2_result": self.news2_result.to_dict(),
            "alert_level": self.alert_level.value,
            "alert_priority": self.alert_priority.value,
            "threshold_applied": self.threshold_applied.to_dict() if self.threshold_applied else None,
            "reasoning": self.reasoning,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "generation_latency_ms": self.generation_latency_ms,
            "single_param_trigger": self.single_param_trigger,
            "suppressed": self.suppressed,
            "ward_id": self.ward_id
        }


@dataclass
class Alert:
    """Complete alert with all metadata for delivery."""
    alert_id: UUID
    patient_id: str
    patient: Patient
    alert_decision: AlertDecision
    alert_level: AlertLevel
    alert_priority: AlertPriority
    title: str
    message: str
    clinical_context: Dict[str, Any]
    created_at: datetime
    status: AlertStatus
    assigned_to: Optional[str]  # User ID of assigned healthcare worker
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]  # User ID
    escalation_step: int
    max_escalation_step: int
    next_escalation_at: Optional[datetime]
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]  # User ID
    
    def is_overdue_for_escalation(self) -> bool:
        """Check if alert should be escalated."""
        if self.status != AlertStatus.PENDING or not self.next_escalation_at:
            return False
        return datetime.now(timezone.utc) >= self.next_escalation_at
    
    def is_critical(self) -> bool:
        """Check if alert is critical level."""
        return self.alert_level == AlertLevel.CRITICAL
    
    def time_since_created(self) -> timedelta:
        """Get time elapsed since alert creation."""
        return datetime.now(timezone.utc) - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "alert_id": str(self.alert_id),
            "patient_id": self.patient_id,
            "patient": self.patient.to_dict(),
            "alert_decision": self.alert_decision.to_dict(),
            "alert_level": self.alert_level.value,
            "alert_priority": self.alert_priority.value,
            "title": self.title,
            "message": self.message,
            "clinical_context": self.clinical_context,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "escalation_step": self.escalation_step,
            "max_escalation_step": self.max_escalation_step,
            "next_escalation_at": self.next_escalation_at.isoformat() if self.next_escalation_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by
        }


@dataclass
class AlertDeliveryAttempt:
    """Record of alert delivery attempt."""
    attempt_id: UUID
    alert_id: UUID
    delivery_channel: str  # 'email', 'sms', 'push', 'pager'
    recipient: str
    attempted_at: datetime
    success: bool
    error_message: Optional[str]
    delivery_latency_ms: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "attempt_id": str(self.attempt_id),
            "alert_id": str(self.alert_id),
            "delivery_channel": self.delivery_channel,
            "recipient": self.recipient,
            "attempted_at": self.attempted_at.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
            "delivery_latency_ms": self.delivery_latency_ms
        }


class AlertGenerationException(Exception):
    """Exception raised during alert generation."""
    pass


class ThresholdConfigurationException(Exception):
    """Exception raised for invalid threshold configuration."""
    pass


class EscalationMatrixException(Exception):
    """Exception raised for escalation matrix issues."""
    pass


@dataclass
class EscalationEvent:
    """Event representing an escalation action."""
    event_id: UUID
    alert_id: UUID
    escalation_step: int
    target_role: EscalationRole
    scheduled_at: datetime
    executed_at: Optional[datetime]
    success: bool
    error_message: Optional[str]
    delivery_attempts: List[AlertDeliveryAttempt]
    retry_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": str(self.event_id),
            "alert_id": str(self.alert_id),
            "escalation_step": self.escalation_step,
            "target_role": self.target_role.value,
            "scheduled_at": self.scheduled_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "success": self.success,
            "error_message": self.error_message,
            "delivery_attempts": [attempt.to_dict() for attempt in self.delivery_attempts],
            "retry_count": self.retry_count
        }