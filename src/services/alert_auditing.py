"""
Alert Decision Auditing Service for NEWS2 Live System

This module implements comprehensive auditing for all alert generation decisions,
including threshold evaluations, rule applications, performance metrics, and
compliance reporting for clinical governance requirements.
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.alerts import (
    AlertDecision, AlertLevel, AlertThreshold, Alert, AlertStatus,
    EscalationEvent, AlertDeliveryAttempt
)
from ..models.news2 import NEWS2Result
from ..models.patient import Patient
from ..services.audit import AuditLogger, AuditOperation


class AlertAuditCategory(Enum):
    """Categories of alert audit events."""
    ALERT_GENERATION = "alert_generation"
    THRESHOLD_EVALUATION = "threshold_evaluation"
    ESCALATION_EXECUTION = "escalation_execution"
    ALERT_ACKNOWLEDGMENT = "alert_acknowledgment"
    ALERT_RESOLUTION = "alert_resolution"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_ACTION = "system_action"


class AlertAuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertAuditEntry:
    """Comprehensive audit entry for alert-related activities."""
    audit_id: UUID
    category: AlertAuditCategory
    severity: AlertAuditSeverity
    event_type: str
    alert_id: Optional[UUID]
    patient_id: Optional[str]
    ward_id: Optional[str]
    user_id: str
    timestamp: datetime
    event_data: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]]
    reasoning: Optional[str]
    clinical_context: Optional[Dict[str, Any]]
    compliance_flags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "audit_id": str(self.audit_id),
            "category": self.category.value,
            "severity": self.severity.value,
            "event_type": self.event_type,
            "alert_id": str(self.alert_id) if self.alert_id else None,
            "patient_id": self.patient_id,
            "ward_id": self.ward_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "event_data": self.event_data,
            "performance_metrics": self.performance_metrics,
            "reasoning": self.reasoning,
            "clinical_context": self.clinical_context,
            "compliance_flags": self.compliance_flags
        }


@dataclass
class AlertPerformanceMetrics:
    """Performance metrics for alert generation and processing."""
    alert_id: UUID
    decision_latency_ms: float
    threshold_evaluation_time_ms: float
    escalation_scheduling_time_ms: float
    total_processing_time_ms: float
    delivery_attempts: int
    successful_deliveries: int
    average_delivery_latency_ms: float
    escalation_events_processed: int
    compliance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "alert_id": str(self.alert_id),
            "decision_latency_ms": self.decision_latency_ms,
            "threshold_evaluation_time_ms": self.threshold_evaluation_time_ms,
            "escalation_scheduling_time_ms": self.escalation_scheduling_time_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "delivery_attempts": self.delivery_attempts,
            "successful_deliveries": self.successful_deliveries,
            "average_delivery_latency_ms": self.average_delivery_latency_ms,
            "escalation_events_processed": self.escalation_events_processed,
            "compliance_score": self.compliance_score
        }


class AlertDecisionAuditor:
    """
    Audits alert generation decisions with comprehensive logging.
    
    Responsibilities:
    - Track all alert generation decisions with full reasoning
    - Log threshold evaluations and rule applications
    - Monitor performance metrics for compliance
    - Provide audit query interface for governance
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        # In-memory storage for audit entries (replace with database in production)
        self._audit_entries: List[AlertAuditEntry] = []
        self._performance_metrics: List[AlertPerformanceMetrics] = []
    
    async def audit_alert_decision(
        self,
        alert_decision: AlertDecision,
        patient: Patient,
        threshold_applied: Optional[AlertThreshold],
        evaluation_time_ms: float,
        user_id: str = "SYSTEM"
    ) -> AlertAuditEntry:
        """
        Audit alert generation decision with full context.
        
        Args:
            alert_decision: Generated alert decision
            patient: Patient information
            threshold_applied: Threshold that was applied (if any)
            evaluation_time_ms: Time taken for threshold evaluation
            user_id: ID of user/system generating alert
            
        Returns:
            Created AlertAuditEntry
        """
        try:
            # Determine severity based on alert level
            severity = self._map_alert_level_to_severity(alert_decision.alert_level)
            
            # Build event data
            event_data = {
                "alert_decision": alert_decision.to_dict(),
                "threshold_applied": threshold_applied.to_dict() if threshold_applied else None,
                "evaluation_time_ms": evaluation_time_ms,
                "news2_score": alert_decision.news2_result.total_score,
                "individual_scores": alert_decision.news2_result.individual_scores,
                "scale_used": alert_decision.news2_result.scale_used,
                "suppressed": alert_decision.suppressed,
                "single_param_trigger": alert_decision.single_param_trigger
            }
            
            # Build clinical context
            clinical_context = {
                "patient_age": patient.age,
                "patient_ward": patient.ward_id,
                "is_copd_patient": patient.is_copd_patient,
                "risk_category": alert_decision.news2_result.risk_category.value,
                "monitoring_frequency": alert_decision.news2_result.monitoring_frequency,
                "calculation_confidence": alert_decision.news2_result.confidence
            }
            
            # Check compliance flags
            compliance_flags = self._evaluate_compliance_flags(alert_decision, evaluation_time_ms)
            
            # Create audit entry
            audit_entry = AlertAuditEntry(
                audit_id=uuid4(),
                category=AlertAuditCategory.ALERT_GENERATION,
                severity=severity,
                event_type="alert_decision_made",
                alert_id=None,  # Will be set when alert is created
                patient_id=patient.patient_id,
                ward_id=patient.ward_id,
                user_id=user_id,
                timestamp=alert_decision.decision_timestamp,
                event_data=event_data,
                performance_metrics={
                    "decision_latency_ms": alert_decision.generation_latency_ms,
                    "evaluation_time_ms": evaluation_time_ms
                },
                reasoning=alert_decision.reasoning,
                clinical_context=clinical_context,
                compliance_flags=compliance_flags
            )
            
            # Store audit entry
            self._audit_entries.append(audit_entry)
            
            # Create underlying audit trail
            await self._create_underlying_audit(audit_entry, user_id)
            
            self.logger.info(
                f"Audited alert decision for patient {patient.patient_id}: "
                f"{alert_decision.alert_level.value} level, latency {alert_decision.generation_latency_ms:.1f}ms"
            )
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to audit alert decision: {str(e)}")
            raise
    
    async def audit_threshold_evaluation(
        self,
        patient_id: str,
        ward_id: str,
        news2_score: int,
        thresholds_evaluated: List[AlertThreshold],
        matched_thresholds: List[AlertThreshold],
        evaluation_details: Dict[str, Any],
        user_id: str = "SYSTEM"
    ) -> AlertAuditEntry:
        """
        Audit threshold evaluation process.
        
        Args:
            patient_id: Patient identifier
            ward_id: Ward identifier
            news2_score: NEWS2 score being evaluated
            thresholds_evaluated: All thresholds considered
            matched_thresholds: Thresholds that matched the score
            evaluation_details: Detailed evaluation information
            user_id: ID of user/system performing evaluation
            
        Returns:
            Created AlertAuditEntry
        """
        try:
            event_data = {
                "news2_score": news2_score,
                "thresholds_evaluated": [t.to_dict() for t in thresholds_evaluated],
                "matched_thresholds": [t.to_dict() for t in matched_thresholds],
                "evaluation_details": evaluation_details,
                "threshold_count": len(thresholds_evaluated),
                "matches_found": len(matched_thresholds)
            }
            
            # Determine severity
            severity = AlertAuditSeverity.WARNING if not matched_thresholds else AlertAuditSeverity.INFO
            
            audit_entry = AlertAuditEntry(
                audit_id=uuid4(),
                category=AlertAuditCategory.THRESHOLD_EVALUATION,
                severity=severity,
                event_type="threshold_evaluation",
                alert_id=None,
                patient_id=patient_id,
                ward_id=ward_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                event_data=event_data,
                performance_metrics=evaluation_details.get("performance_metrics"),
                reasoning=f"Evaluated {len(thresholds_evaluated)} thresholds for NEWS2 score {news2_score}, found {len(matched_thresholds)} matches",
                clinical_context=evaluation_details.get("clinical_context"),
                compliance_flags=[]
            )
            
            self._audit_entries.append(audit_entry)
            await self._create_underlying_audit(audit_entry, user_id)
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to audit threshold evaluation: {str(e)}")
            raise
    
    async def audit_escalation_execution(
        self,
        escalation_event: EscalationEvent,
        alert: Alert,
        execution_success: bool,
        delivery_attempts: List[AlertDeliveryAttempt],
        user_id: str = "SYSTEM"
    ) -> AlertAuditEntry:
        """
        Audit escalation event execution.
        
        Args:
            escalation_event: Escalation event that was executed
            alert: Alert being escalated
            execution_success: Whether escalation was successful
            delivery_attempts: List of delivery attempts made
            user_id: ID of user/system executing escalation
            
        Returns:
            Created AlertAuditEntry
        """
        try:
            # Calculate delivery metrics
            successful_deliveries = sum(1 for attempt in delivery_attempts if attempt.success)
            avg_latency = (
                sum(attempt.delivery_latency_ms or 0 for attempt in delivery_attempts if attempt.delivery_latency_ms) /
                max(len([a for a in delivery_attempts if a.delivery_latency_ms]), 1)
            )
            
            event_data = {
                "escalation_event": escalation_event.to_dict(),
                "execution_success": execution_success,
                "delivery_attempts": [attempt.to_dict() for attempt in delivery_attempts],
                "successful_deliveries": successful_deliveries,
                "total_attempts": len(delivery_attempts),
                "average_delivery_latency_ms": avg_latency
            }
            
            severity = AlertAuditSeverity.ERROR if not execution_success else AlertAuditSeverity.INFO
            if alert.alert_level == AlertLevel.CRITICAL and not execution_success:
                severity = AlertAuditSeverity.CRITICAL
            
            audit_entry = AlertAuditEntry(
                audit_id=uuid4(),
                category=AlertAuditCategory.ESCALATION_EXECUTION,
                severity=severity,
                event_type="escalation_executed",
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                ward_id=alert.patient.ward_id,
                user_id=user_id,
                timestamp=escalation_event.executed_at or datetime.now(timezone.utc),
                event_data=event_data,
                performance_metrics={
                    "delivery_success_rate": successful_deliveries / len(delivery_attempts) if delivery_attempts else 0,
                    "average_delivery_latency_ms": avg_latency,
                    "escalation_step": escalation_event.escalation_step
                },
                reasoning=f"Escalation to {escalation_event.target_role.value} {'succeeded' if execution_success else 'failed'}",
                clinical_context={
                    "alert_level": alert.alert_level.value,
                    "escalation_step": escalation_event.escalation_step,
                    "target_role": escalation_event.target_role.value
                },
                compliance_flags=self._evaluate_escalation_compliance_flags(escalation_event, execution_success, alert)
            )
            
            self._audit_entries.append(audit_entry)
            await self._create_underlying_audit(audit_entry, user_id)
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to audit escalation execution: {str(e)}")
            raise
    
    async def audit_alert_acknowledgment(
        self,
        alert: Alert,
        acknowledged_by: str,
        acknowledgment_time: datetime,
        acknowledgment_note: Optional[str] = None
    ) -> AlertAuditEntry:
        """
        Audit alert acknowledgment.
        
        Args:
            alert: Alert being acknowledged
            acknowledged_by: ID of user acknowledging alert
            acknowledgment_time: Time of acknowledgment
            acknowledgment_note: Optional acknowledgment note
            
        Returns:
            Created AlertAuditEntry
        """
        try:
            # Calculate response time
            response_time_minutes = (acknowledgment_time - alert.created_at).total_seconds() / 60
            
            event_data = {
                "alert_id": str(alert.alert_id),
                "acknowledged_by": acknowledged_by,
                "acknowledgment_time": acknowledgment_time.isoformat(),
                "acknowledgment_note": acknowledgment_note,
                "response_time_minutes": response_time_minutes,
                "alert_level": alert.alert_level.value,
                "escalation_step_when_acknowledged": alert.escalation_step
            }
            
            # Check if acknowledgment was within acceptable timeframes
            compliance_flags = []
            if alert.alert_level == AlertLevel.CRITICAL and response_time_minutes > 15:
                compliance_flags.append("CRITICAL_ALERT_LATE_ACKNOWLEDGMENT")
            elif alert.alert_level == AlertLevel.HIGH and response_time_minutes > 30:
                compliance_flags.append("HIGH_ALERT_LATE_ACKNOWLEDGMENT")
            
            audit_entry = AlertAuditEntry(
                audit_id=uuid4(),
                category=AlertAuditCategory.ALERT_ACKNOWLEDGMENT,
                severity=AlertAuditSeverity.WARNING if compliance_flags else AlertAuditSeverity.INFO,
                event_type="alert_acknowledged",
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                ward_id=alert.patient.ward_id,
                user_id=acknowledged_by,
                timestamp=acknowledgment_time,
                event_data=event_data,
                performance_metrics={
                    "response_time_minutes": response_time_minutes,
                    "escalation_step_at_acknowledgment": alert.escalation_step
                },
                reasoning=f"Alert acknowledged by {acknowledged_by} after {response_time_minutes:.1f} minutes",
                clinical_context={
                    "alert_level": alert.alert_level.value,
                    "news2_score": alert.alert_decision.news2_result.total_score,
                    "patient_age": alert.patient.age,
                    "ward_id": alert.patient.ward_id
                },
                compliance_flags=compliance_flags
            )
            
            self._audit_entries.append(audit_entry)
            await self._create_underlying_audit(audit_entry, acknowledged_by)
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to audit alert acknowledgment: {str(e)}")
            raise
    
    async def audit_performance_metrics(
        self,
        metrics: AlertPerformanceMetrics,
        user_id: str = "SYSTEM"
    ) -> AlertAuditEntry:
        """
        Audit performance metrics for alert processing.
        
        Args:
            metrics: Performance metrics to audit
            user_id: ID of user/system recording metrics
            
        Returns:
            Created AlertAuditEntry
        """
        try:
            # Evaluate performance compliance
            compliance_flags = []
            if metrics.decision_latency_ms > 5000:  # > 5 seconds
                compliance_flags.append("DECISION_LATENCY_EXCEEDED")
            if metrics.total_processing_time_ms > 10000:  # > 10 seconds
                compliance_flags.append("PROCESSING_TIME_EXCEEDED")
            if metrics.successful_deliveries < metrics.delivery_attempts * 0.9:  # < 90% success rate
                compliance_flags.append("LOW_DELIVERY_SUCCESS_RATE")
            
            severity = AlertAuditSeverity.WARNING if compliance_flags else AlertAuditSeverity.INFO
            
            audit_entry = AlertAuditEntry(
                audit_id=uuid4(),
                category=AlertAuditCategory.PERFORMANCE_METRIC,
                severity=severity,
                event_type="performance_metrics_recorded",
                alert_id=metrics.alert_id,
                patient_id=None,
                ward_id=None,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                event_data=metrics.to_dict(),
                performance_metrics=metrics.to_dict(),
                reasoning=f"Performance metrics recorded with compliance score {metrics.compliance_score:.2f}",
                clinical_context=None,
                compliance_flags=compliance_flags
            )
            
            self._audit_entries.append(audit_entry)
            self._performance_metrics.append(metrics)
            await self._create_underlying_audit(audit_entry, user_id)
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to audit performance metrics: {str(e)}")
            raise
    
    async def query_audit_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[AlertAuditCategory] = None,
        severity: Optional[AlertAuditSeverity] = None,
        patient_id: Optional[str] = None,
        ward_id: Optional[str] = None,
        alert_id: Optional[UUID] = None,
        limit: int = 1000
    ) -> List[AlertAuditEntry]:
        """
        Query audit entries with filtering.
        
        Args:
            start_time: Start time for query (inclusive)
            end_time: End time for query (inclusive)
            category: Filter by audit category
            severity: Filter by severity level
            patient_id: Filter by patient ID
            ward_id: Filter by ward ID
            alert_id: Filter by alert ID
            limit: Maximum number of entries to return
            
        Returns:
            List of matching AlertAuditEntry objects
        """
        try:
            filtered_entries = []
            
            for entry in self._audit_entries:
                # Apply filters
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                if category and entry.category != category:
                    continue
                if severity and entry.severity != severity:
                    continue
                if patient_id and entry.patient_id != patient_id:
                    continue
                if ward_id and entry.ward_id != ward_id:
                    continue
                if alert_id and entry.alert_id != alert_id:
                    continue
                
                filtered_entries.append(entry)
                
                if len(filtered_entries) >= limit:
                    break
            
            # Sort by timestamp (most recent first)
            filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
            
            return filtered_entries
            
        except Exception as e:
            self.logger.error(f"Failed to query audit entries: {str(e)}")
            raise
    
    async def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        ward_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for alert processing.
        
        Args:
            start_time: Start time for report period
            end_time: End time for report period
            ward_id: Optional ward filter
            
        Returns:
            Compliance report dictionary
        """
        try:
            # Query relevant audit entries
            entries = await self.query_audit_entries(
                start_time=start_time,
                end_time=end_time,
                ward_id=ward_id,
                limit=10000
            )
            
            # Calculate compliance metrics
            total_alerts = len([e for e in entries if e.category == AlertAuditCategory.ALERT_GENERATION])
            critical_alerts = len([
                e for e in entries 
                if e.category == AlertAuditCategory.ALERT_GENERATION and 
                e.event_data.get("alert_decision", {}).get("alert_level") == "critical"
            ])
            
            # Performance compliance
            decision_latency_compliant = len([
                e for e in entries 
                if e.category == AlertAuditCategory.ALERT_GENERATION and
                e.performance_metrics and
                e.performance_metrics.get("decision_latency_ms", 0) <= 5000
            ])
            
            # Escalation compliance
            escalation_entries = [e for e in entries if e.category == AlertAuditCategory.ESCALATION_EXECUTION]
            successful_escalations = len([e for e in escalation_entries if e.event_data.get("execution_success")])
            
            # Acknowledgment compliance
            acknowledgment_entries = [e for e in entries if e.category == AlertAuditCategory.ALERT_ACKNOWLEDGMENT]
            timely_acknowledgments = len([
                e for e in acknowledgment_entries 
                if not e.compliance_flags or not any("LATE" in flag for flag in e.compliance_flags)
            ])
            
            # Compliance flags analysis
            all_compliance_flags = []
            for entry in entries:
                all_compliance_flags.extend(entry.compliance_flags)
            
            flag_summary = {}
            for flag in set(all_compliance_flags):
                flag_summary[flag] = all_compliance_flags.count(flag)
            
            report = {
                "report_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "ward_id": ward_id
                },
                "alert_metrics": {
                    "total_alerts": total_alerts,
                    "critical_alerts": critical_alerts,
                    "critical_alert_percentage": (critical_alerts / max(total_alerts, 1)) * 100
                },
                "performance_compliance": {
                    "decision_latency_compliant": decision_latency_compliant,
                    "decision_latency_compliance_rate": (decision_latency_compliant / max(total_alerts, 1)) * 100,
                    "target_decision_latency_ms": 5000
                },
                "escalation_compliance": {
                    "total_escalations": len(escalation_entries),
                    "successful_escalations": successful_escalations,
                    "escalation_success_rate": (successful_escalations / max(len(escalation_entries), 1)) * 100
                },
                "acknowledgment_compliance": {
                    "total_acknowledgments": len(acknowledgment_entries),
                    "timely_acknowledgments": timely_acknowledgments,
                    "timely_acknowledgment_rate": (timely_acknowledgments / max(len(acknowledgment_entries), 1)) * 100
                },
                "compliance_flags": flag_summary,
                "overall_compliance_score": self._calculate_overall_compliance_score(entries),
                "recommendations": self._generate_compliance_recommendations(flag_summary)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {str(e)}")
            raise
    
    def _map_alert_level_to_severity(self, alert_level: AlertLevel) -> AlertAuditSeverity:
        """Map alert level to audit severity."""
        severity_map = {
            AlertLevel.CRITICAL: AlertAuditSeverity.CRITICAL,
            AlertLevel.HIGH: AlertAuditSeverity.WARNING,
            AlertLevel.MEDIUM: AlertAuditSeverity.INFO,
            AlertLevel.LOW: AlertAuditSeverity.INFO
        }
        return severity_map.get(alert_level, AlertAuditSeverity.INFO)
    
    def _evaluate_compliance_flags(
        self, 
        alert_decision: AlertDecision, 
        evaluation_time_ms: float
    ) -> List[str]:
        """Evaluate compliance flags for alert decision."""
        flags = []
        
        # Performance compliance
        if alert_decision.generation_latency_ms > 5000:
            flags.append("DECISION_LATENCY_EXCEEDED")
        if evaluation_time_ms > 1000:  # Threshold evaluation should be fast
            flags.append("THRESHOLD_EVALUATION_SLOW")
        
        # Clinical compliance
        if alert_decision.alert_level == AlertLevel.CRITICAL and alert_decision.suppressed:
            flags.append("CRITICAL_ALERT_SUPPRESSED")  # This should never happen
        
        # Data quality compliance
        if alert_decision.news2_result.confidence < 0.8:
            flags.append("LOW_CONFIDENCE_CALCULATION")
        
        return flags
    
    def _evaluate_escalation_compliance_flags(
        self,
        escalation_event: EscalationEvent,
        execution_success: bool,
        alert: Alert
    ) -> List[str]:
        """Evaluate compliance flags for escalation execution."""
        flags = []
        
        # Execution compliance
        if not execution_success:
            if alert.alert_level == AlertLevel.CRITICAL:
                flags.append("CRITICAL_ESCALATION_FAILED")
            else:
                flags.append("ESCALATION_FAILED")
        
        # Timing compliance
        expected_time = alert.created_at + timedelta(minutes=15 * escalation_event.escalation_step)
        actual_time = escalation_event.executed_at or datetime.now(timezone.utc)
        if actual_time > expected_time + timedelta(minutes=5):  # 5 minute tolerance
            flags.append("ESCALATION_TIMING_DELAYED")
        
        # Retry compliance
        if escalation_event.retry_count > 3:
            flags.append("EXCESSIVE_ESCALATION_RETRIES")
        
        return flags
    
    def _calculate_overall_compliance_score(self, entries: List[AlertAuditEntry]) -> float:
        """Calculate overall compliance score from audit entries."""
        if not entries:
            return 1.0
        
        total_score = 0.0
        scored_entries = 0
        
        for entry in entries:
            # Weight different categories
            category_weight = {
                AlertAuditCategory.ALERT_GENERATION: 1.0,
                AlertAuditCategory.ESCALATION_EXECUTION: 0.8,
                AlertAuditCategory.ALERT_ACKNOWLEDGMENT: 0.6,
                AlertAuditCategory.PERFORMANCE_METRIC: 0.4
            }.get(entry.category, 0.2)
            
            # Score based on compliance flags (fewer flags = higher score)
            entry_score = max(0.0, 1.0 - (len(entry.compliance_flags) * 0.2))
            
            total_score += entry_score * category_weight
            scored_entries += category_weight
        
        return total_score / max(scored_entries, 1)
    
    def _generate_compliance_recommendations(self, flag_summary: Dict[str, int]) -> List[str]:
        """Generate compliance recommendations based on flag analysis."""
        recommendations = []
        
        if flag_summary.get("DECISION_LATENCY_EXCEEDED", 0) > 0:
            recommendations.append("Optimize alert generation algorithms to meet 5-second latency target")
        
        if flag_summary.get("CRITICAL_ESCALATION_FAILED", 0) > 0:
            recommendations.append("Review escalation delivery channels and backup notification methods")
        
        if flag_summary.get("CRITICAL_ALERT_LATE_ACKNOWLEDGMENT", 0) > 0:
            recommendations.append("Improve critical alert visibility and staff notification procedures")
        
        if flag_summary.get("LOW_DELIVERY_SUCCESS_RATE", 0) > 0:
            recommendations.append("Investigate and improve alert delivery reliability")
        
        if flag_summary.get("LOW_CONFIDENCE_CALCULATION", 0) > 0:
            recommendations.append("Review vital signs data quality and validation procedures")
        
        return recommendations
    
    async def _create_underlying_audit(self, audit_entry: AlertAuditEntry, user_id: str):
        """Create underlying audit trail entry."""
        try:
            underlying_audit = self.audit_logger.create_audit_entry(
                table_name="alert_audit_entries",
                operation=AuditOperation.INSERT,
                user_id=user_id,
                patient_id=audit_entry.patient_id,
                new_values=audit_entry.to_dict()
            )
            
            self.logger.debug(f"Created underlying audit for alert audit entry {audit_entry.audit_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create underlying audit: {str(e)}")
            # Don't fail the alert auditing due to underlying audit issues


class AlertAuditingService:
    """
    High-level service for alert auditing and compliance reporting.
    
    Provides unified interface for all alert-related auditing activities.
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.decision_auditor = AlertDecisionAuditor(audit_logger)
        self.logger = logging.getLogger(__name__)
    
    async def get_alert_audit_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        ward_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive audit summary for a time period.
        
        Args:
            start_time: Start time for summary period
            end_time: End time for summary period
            ward_id: Optional ward filter
            
        Returns:
            Comprehensive audit summary
        """
        try:
            # Get audit entries
            entries = await self.decision_auditor.query_audit_entries(
                start_time=start_time,
                end_time=end_time,
                ward_id=ward_id,
                limit=10000
            )
            
            # Generate compliance report
            compliance_report = await self.decision_auditor.generate_compliance_report(
                start_time, end_time, ward_id
            )
            
            # Calculate category breakdown
            category_breakdown = {}
            for category in AlertAuditCategory:
                category_entries = [e for e in entries if e.category == category]
                category_breakdown[category.value] = {
                    "count": len(category_entries),
                    "severity_breakdown": self._get_severity_breakdown(category_entries)
                }
            
            # Calculate performance statistics
            performance_entries = [
                e for e in entries 
                if e.performance_metrics
            ]
            
            avg_decision_latency = (
                sum(e.performance_metrics.get("decision_latency_ms", 0) for e in performance_entries) /
                max(len(performance_entries), 1)
            )
            
            summary = {
                "summary_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "ward_id": ward_id,
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                },
                "total_audit_entries": len(entries),
                "category_breakdown": category_breakdown,
                "compliance_report": compliance_report,
                "performance_statistics": {
                    "average_decision_latency_ms": avg_decision_latency,
                    "entries_with_performance_data": len(performance_entries)
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get alert audit summary: {str(e)}")
            raise
    
    def _get_severity_breakdown(self, entries: List[AlertAuditEntry]) -> Dict[str, int]:
        """Get breakdown of entries by severity."""
        breakdown = {}
        for severity in AlertAuditSeverity:
            breakdown[severity.value] = len([e for e in entries if e.severity == severity])
        return breakdown