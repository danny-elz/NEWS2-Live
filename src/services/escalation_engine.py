"""
Alert Escalation Engine for NEWS2 Live System

This module implements the escalation matrix infrastructure including:
- Escalation scheduling and timing management
- Unacknowledged alert tracking
- Role-based escalation routing (Ward Nurse → Charge Nurse → Doctor → Rapid Response)
- Escalation decision logging and audit trail
- Automatic escalation after 15 minutes for critical alerts
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from uuid import uuid4, UUID
from dataclasses import dataclass, replace
from enum import Enum

from ..models.alerts import (
    Alert, AlertLevel, AlertStatus, EscalationMatrix, EscalationStep, EscalationRole,
    AlertDeliveryAttempt, EscalationMatrixException, EscalationEvent
)
from ..models.patient import Patient
from ..services.audit import AuditLogger, AuditOperation
from ..services.alert_configuration import EscalationMatrixManager



@dataclass
class EscalationSchedule:
    """Schedule tracking escalation timing for an alert."""
    alert_id: UUID
    escalation_matrix: EscalationMatrix
    current_step: int
    pending_events: List[EscalationEvent]
    completed_events: List[EscalationEvent]
    escalation_paused: bool
    paused_reason: Optional[str]
    created_at: datetime
    last_updated: datetime
    
    def get_next_scheduled_event(self) -> Optional[EscalationEvent]:
        """Get the next pending escalation event."""
        if self.escalation_paused:
            return None
        
        pending_sorted = sorted(self.pending_events, key=lambda e: e.scheduled_at)
        now = datetime.now(timezone.utc)
        
        for event in pending_sorted:
            if event.scheduled_at <= now:
                return event
        
        return None
    
    def has_overdue_events(self) -> bool:
        """Check if there are overdue escalation events."""
        if self.escalation_paused:
            return False
        
        now = datetime.now(timezone.utc)
        return any(event.scheduled_at <= now for event in self.pending_events)


class EscalationScheduler:
    """
    Manages escalation scheduling and timing for alerts.
    
    Responsibilities:
    - Create escalation schedules based on escalation matrices
    - Track timing for automatic escalations
    - Handle escalation retries and failures
    - Manage escalation pause/resume functionality
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        # In-memory storage for escalation schedules
        self._schedules: Dict[UUID, EscalationSchedule] = {}
        self._pending_events: List[EscalationEvent] = []
    
    async def create_escalation_schedule(
        self,
        alert: Alert,
        escalation_matrix: EscalationMatrix
    ) -> EscalationSchedule:
        """
        Create escalation schedule for a new alert.
        
        Args:
            alert: Alert to create schedule for
            escalation_matrix: Escalation matrix to use
            
        Returns:
            Created EscalationSchedule object
        """
        try:
            # Create escalation events based on matrix steps
            pending_events = []
            base_time = alert.created_at
            
            for step_index, step in enumerate(escalation_matrix.escalation_steps):
                scheduled_time = base_time + timedelta(minutes=step.delay_minutes)
                
                event = EscalationEvent(
                    event_id=uuid4(),
                    alert_id=alert.alert_id,
                    escalation_step=step_index,
                    target_role=step.role,
                    scheduled_at=scheduled_time,
                    executed_at=None,
                    success=False,
                    error_message=None,
                    delivery_attempts=[],
                    retry_count=0
                )
                pending_events.append(event)
            
            # Create schedule
            schedule = EscalationSchedule(
                alert_id=alert.alert_id,
                escalation_matrix=escalation_matrix,
                current_step=0,
                pending_events=pending_events,
                completed_events=[],
                escalation_paused=False,
                paused_reason=None,
                created_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Store schedule
            self._schedules[alert.alert_id] = schedule
            
            # Add events to pending list for processing
            self._pending_events.extend(pending_events)
            
            self.logger.info(
                f"Created escalation schedule for alert {alert.alert_id} with "
                f"{len(pending_events)} escalation events"
            )
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Failed to create escalation schedule for alert {alert.alert_id}: {str(e)}")
            raise EscalationMatrixException(f"Escalation schedule creation failed: {str(e)}")
    
    async def get_overdue_escalations(self) -> List[EscalationEvent]:
        """Get all overdue escalation events that need processing."""
        now = datetime.now(timezone.utc)
        overdue_events = []
        
        for event in self._pending_events:
            if event.scheduled_at <= now:
                overdue_events.append(event)
        
        return overdue_events
    
    async def mark_escalation_executed(
        self,
        event_id: UUID,
        success: bool,
        error_message: Optional[str] = None,
        delivery_attempts: Optional[List[AlertDeliveryAttempt]] = None
    ):
        """
        Mark escalation event as executed.
        
        Args:
            event_id: ID of escalation event
            success: Whether escalation was successful
            error_message: Error message if failed
            delivery_attempts: List of delivery attempts made
        """
        try:
            # Find and update event
            event = None
            for pending_event in self._pending_events:
                if pending_event.event_id == event_id:
                    event = pending_event
                    break
            
            if not event:
                self.logger.warning(f"Escalation event {event_id} not found in pending events")
                return
            
            # Update event
            event.executed_at = datetime.now(timezone.utc)
            event.success = success
            event.error_message = error_message
            if delivery_attempts:
                event.delivery_attempts = delivery_attempts
            
            # Move from pending to completed
            self._pending_events.remove(event)
            
            # Update schedule
            schedule = self._schedules.get(event.alert_id)
            if schedule:
                schedule.completed_events.append(event)
                if success:
                    schedule.current_step = event.escalation_step + 1
                schedule.last_updated = datetime.now(timezone.utc)
            
            self.logger.info(
                f"Marked escalation event {event_id} as executed "
                f"(success: {success}, step: {event.escalation_step})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to mark escalation event {event_id} as executed: {str(e)}")
    
    async def pause_escalation(
        self,
        alert_id: UUID,
        reason: str,
        user_id: str = "SYSTEM"
    ):
        """
        Pause escalation for an alert.
        
        Args:
            alert_id: ID of alert to pause escalation for
            reason: Reason for pausing
            user_id: ID of user pausing escalation
        """
        try:
            schedule = self._schedules.get(alert_id)
            if not schedule:
                self.logger.warning(f"Escalation schedule not found for alert {alert_id}")
                return
            
            schedule.escalation_paused = True
            schedule.paused_reason = reason
            schedule.last_updated = datetime.now(timezone.utc)
            
            # Audit escalation pause
            await self._audit_escalation_action(
                "PAUSE", alert_id, user_id, {"reason": reason}
            )
            
            self.logger.info(f"Paused escalation for alert {alert_id}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to pause escalation for alert {alert_id}: {str(e)}")
    
    async def resume_escalation(
        self,
        alert_id: UUID,
        user_id: str = "SYSTEM"
    ):
        """
        Resume escalation for an alert.
        
        Args:
            alert_id: ID of alert to resume escalation for
            user_id: ID of user resuming escalation
        """
        try:
            schedule = self._schedules.get(alert_id)
            if not schedule:
                self.logger.warning(f"Escalation schedule not found for alert {alert_id}")
                return
            
            schedule.escalation_paused = False
            schedule.paused_reason = None
            schedule.last_updated = datetime.now(timezone.utc)
            
            # Audit escalation resume
            await self._audit_escalation_action("RESUME", alert_id, user_id)
            
            self.logger.info(f"Resumed escalation for alert {alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to resume escalation for alert {alert_id}: {str(e)}")
    
    async def get_escalation_schedule(self, alert_id: UUID) -> Optional[EscalationSchedule]:
        """Get escalation schedule for an alert."""
        return self._schedules.get(alert_id)
    
    async def cancel_escalation(
        self,
        alert_id: UUID,
        reason: str,
        user_id: str = "SYSTEM"
    ):
        """
        Cancel escalation for an alert (e.g., when alert is acknowledged/resolved).
        
        Args:
            alert_id: ID of alert to cancel escalation for
            reason: Reason for cancellation
            user_id: ID of user cancelling escalation
        """
        try:
            schedule = self._schedules.get(alert_id)
            if not schedule:
                return
            
            # Remove pending events
            self._pending_events = [
                event for event in self._pending_events 
                if event.alert_id != alert_id
            ]
            
            # Remove schedule
            del self._schedules[alert_id]
            
            # Audit escalation cancellation
            await self._audit_escalation_action(
                "CANCEL", alert_id, user_id, {"reason": reason}
            )
            
            self.logger.info(f"Cancelled escalation for alert {alert_id}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel escalation for alert {alert_id}: {str(e)}")
    
    async def _audit_escalation_action(
        self,
        action: str,
        alert_id: UUID,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Audit escalation management actions."""
        try:
            audit_data = {
                "action": action,
                "alert_id": str(alert_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            audit_entry = self.audit_logger.create_audit_entry(
                table_name="escalation_actions",
                operation=AuditOperation.INSERT,
                user_id=user_id,
                patient_id=None,  # Alert-specific, not patient-specific
                new_values=audit_data
            )
            
            self.logger.debug(f"Escalation {action} action audited for alert {alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to audit escalation {action}: {str(e)}")


class EscalationExecutor:
    """
    Executes escalation events and manages delivery attempts.
    
    Responsibilities:
    - Execute escalation events at scheduled times
    - Handle delivery retries and failures
    - Track delivery attempt history
    - Interface with notification delivery systems
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        # Callback function for alert delivery (set by delivery service)
        self._delivery_callback: Optional[Callable] = None
    
    def set_delivery_callback(self, callback: Callable):
        """Set callback function for alert delivery."""
        self._delivery_callback = callback
    
    async def execute_escalation_event(
        self,
        event: EscalationEvent,
        alert: Alert,
        escalation_step: EscalationStep
    ) -> Tuple[bool, Optional[str], List[AlertDeliveryAttempt]]:
        """
        Execute escalation event with retry logic.
        
        Args:
            event: Escalation event to execute
            alert: Alert being escalated
            escalation_step: Escalation step configuration
            
        Returns:
            Tuple of (success, error_message, delivery_attempts)
        """
        try:
            delivery_attempts = []
            success = False
            error_message = None
            
            # Attempt delivery with retries
            for attempt in range(escalation_step.max_attempts):
                self.logger.info(
                    f"Executing escalation event {event.event_id} "
                    f"(attempt {attempt + 1}/{escalation_step.max_attempts})"
                )
                
                attempt_success, attempt_error, attempt_record = await self._attempt_delivery(
                    event, alert, escalation_step.role
                )
                
                delivery_attempts.append(attempt_record)
                
                if attempt_success:
                    success = True
                    break
                else:
                    error_message = attempt_error
                    if attempt < escalation_step.max_attempts - 1:
                        # Wait before retry
                        await asyncio.sleep(escalation_step.retry_interval_minutes * 60)
            
            # Log execution result
            if success:
                self.logger.info(
                    f"Escalation event {event.event_id} executed successfully "
                    f"after {len(delivery_attempts)} attempts"
                )
            else:
                self.logger.warning(
                    f"Escalation event {event.event_id} failed after "
                    f"{escalation_step.max_attempts} attempts: {error_message}"
                )
            
            return success, error_message, delivery_attempts
            
        except Exception as e:
            error_msg = f"Escalation execution failed: {str(e)}"
            self.logger.error(f"Failed to execute escalation event {event.event_id}: {error_msg}")
            
            # Create failed delivery attempt record
            failed_attempt = AlertDeliveryAttempt(
                attempt_id=uuid4(),
                alert_id=event.alert_id,
                delivery_channel="system_error",
                recipient="unknown",
                attempted_at=datetime.now(timezone.utc),
                success=False,
                error_message=error_msg,
                delivery_latency_ms=None
            )
            
            return False, error_msg, [failed_attempt]
    
    async def _attempt_delivery(
        self,
        event: EscalationEvent,
        alert: Alert,
        target_role: EscalationRole
    ) -> Tuple[bool, Optional[str], AlertDeliveryAttempt]:
        """
        Attempt single delivery of escalation alert.
        
        Returns:
            Tuple of (success, error_message, delivery_attempt_record)
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Determine delivery targets for role
            delivery_targets = await self._get_delivery_targets_for_role(
                alert.patient.ward_id, target_role
            )
            
            if not delivery_targets:
                error_msg = f"No delivery targets found for role {target_role.value} in ward {alert.patient.ward_id}"
                return False, error_msg, self._create_delivery_attempt_record(
                    event.alert_id, "no_targets", "unknown", start_time, False, error_msg
                )
            
            # Use callback for actual delivery if available
            if self._delivery_callback:
                success, error_message = await self._delivery_callback(
                    alert, target_role, delivery_targets
                )
            else:
                # Simulate delivery for testing
                success, error_message = await self._simulate_delivery(
                    alert, target_role, delivery_targets
                )
            
            # Create delivery attempt record
            delivery_channel = self._get_preferred_delivery_channel(target_role)
            primary_target = delivery_targets[0] if delivery_targets else "unknown"
            
            attempt_record = self._create_delivery_attempt_record(
                event.alert_id, delivery_channel, primary_target, start_time, success, error_message
            )
            
            return success, error_message, attempt_record
            
        except Exception as e:
            error_msg = f"Delivery attempt failed: {str(e)}"
            attempt_record = self._create_delivery_attempt_record(
                event.alert_id, "error", "unknown", start_time, False, error_msg
            )
            return False, error_msg, attempt_record
    
    def _create_delivery_attempt_record(
        self,
        alert_id: UUID,
        delivery_channel: str,
        recipient: str,
        start_time: datetime,
        success: bool,
        error_message: Optional[str] = None
    ) -> AlertDeliveryAttempt:
        """Create delivery attempt record."""
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return AlertDeliveryAttempt(
            attempt_id=uuid4(),
            alert_id=alert_id,
            delivery_channel=delivery_channel,
            recipient=recipient,
            attempted_at=start_time,
            success=success,
            error_message=error_message,
            delivery_latency_ms=latency_ms if success else None
        )
    
    async def _get_delivery_targets_for_role(
        self, 
        ward_id: str, 
        role: EscalationRole
    ) -> List[str]:
        """
        Get delivery targets (user IDs) for a role in a ward.
        
        In production, this would query staff assignment database.
        For now, return mock data.
        """
        # Mock delivery targets based on role
        mock_targets = {
            EscalationRole.WARD_NURSE: [f"nurse_{ward_id}_001", f"nurse_{ward_id}_002"],
            EscalationRole.CHARGE_NURSE: [f"charge_nurse_{ward_id}"],
            EscalationRole.DOCTOR: [f"doctor_{ward_id}_001", f"doctor_on_call"],
            EscalationRole.RAPID_RESPONSE: ["rapid_response_team"],
            EscalationRole.CONSULTANT: [f"consultant_{ward_id}"]
        }
        
        return mock_targets.get(role, [])
    
    def _get_preferred_delivery_channel(self, role: EscalationRole) -> str:
        """Get preferred delivery channel for role."""
        channel_preferences = {
            EscalationRole.WARD_NURSE: "mobile_app",
            EscalationRole.CHARGE_NURSE: "mobile_app",
            EscalationRole.DOCTOR: "pager",
            EscalationRole.RAPID_RESPONSE: "pager",
            EscalationRole.CONSULTANT: "pager"
        }
        return channel_preferences.get(role, "email")
    
    async def _simulate_delivery(
        self,
        alert: Alert,
        target_role: EscalationRole,
        delivery_targets: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Simulate alert delivery for testing purposes."""
        # Simulate delivery delay
        await asyncio.sleep(0.1)
        
        # Simulate 95% success rate
        import random
        if random.random() < 0.95:
            self.logger.debug(
                f"Simulated successful delivery to {target_role.value} "
                f"for alert {alert.alert_id}"
            )
            return True, None
        else:
            error_msg = f"Simulated delivery failure to {target_role.value}"
            self.logger.debug(f"Simulated delivery failure for alert {alert.alert_id}")
            return False, error_msg


class EscalationEngine:
    """
    Main escalation engine that coordinates all escalation activities.
    
    Responsibilities:
    - Manage escalation schedules and execution
    - Monitor for overdue escalations
    - Handle acknowledgment and resolution workflows
    - Provide escalation status and reporting
    """
    
    def __init__(
        self, 
        audit_logger: AuditLogger,
        escalation_matrix_manager: EscalationMatrixManager
    ):
        self.audit_logger = audit_logger
        self.escalation_matrix_manager = escalation_matrix_manager
        self.scheduler = EscalationScheduler(audit_logger)
        self.executor = EscalationExecutor(audit_logger)
        self.logger = logging.getLogger(__name__)
        
        # Background task for processing escalations
        self._escalation_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_escalation_processing(self):
        """Start background escalation processing."""
        if self._running:
            self.logger.warning("Escalation processing already running")
            return
        
        self._running = True
        self._escalation_task = asyncio.create_task(self._escalation_processing_loop())
        self.logger.info("Started escalation processing engine")
    
    async def stop_escalation_processing(self):
        """Stop background escalation processing."""
        if not self._running:
            return
        
        self._running = False
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped escalation processing engine")
    
    async def create_alert_escalation(self, alert: Alert) -> Optional[EscalationSchedule]:
        """
        Create escalation schedule for a new alert.
        
        Args:
            alert: Alert to create escalation for
            
        Returns:
            Created EscalationSchedule or None if no escalation needed
        """
        try:
            # Get escalation matrix for ward and alert level
            escalation_matrix = await self.escalation_matrix_manager.get_escalation_matrix(
                alert.patient.ward_id, alert.alert_level
            )
            
            if not escalation_matrix:
                self.logger.info(
                    f"No escalation matrix found for ward {alert.patient.ward_id}, "
                    f"alert level {alert.alert_level.value}"
                )
                return None
            
            # Create escalation schedule
            schedule = await self.scheduler.create_escalation_schedule(alert, escalation_matrix)
            
            self.logger.info(
                f"Created escalation for alert {alert.alert_id} "
                f"({alert.alert_level.value} level in ward {alert.patient.ward_id})"
            )
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Failed to create alert escalation: {str(e)}")
            raise EscalationMatrixException(f"Alert escalation creation failed: {str(e)}")
    
    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: str,
        acknowledgment_note: Optional[str] = None
    ):
        """
        Acknowledge an alert and pause its escalation.
        
        Args:
            alert_id: ID of alert being acknowledged
            acknowledged_by: ID of user acknowledging alert
            acknowledgment_note: Optional note from acknowledging user
        """
        try:
            # Pause escalation
            reason = f"Alert acknowledged by {acknowledged_by}"
            if acknowledgment_note:
                reason += f": {acknowledgment_note}"
            
            await self.scheduler.pause_escalation(alert_id, reason, acknowledged_by)
            
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
    
    async def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ):
        """
        Resolve an alert and cancel its escalation.
        
        Args:
            alert_id: ID of alert being resolved
            resolved_by: ID of user resolving alert
            resolution_note: Optional note from resolving user
        """
        try:
            # Cancel escalation
            reason = f"Alert resolved by {resolved_by}"
            if resolution_note:
                reason += f": {resolution_note}"
            
            await self.scheduler.cancel_escalation(alert_id, reason, resolved_by)
            
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
    
    async def get_escalation_status(self, alert_id: UUID) -> Dict[str, Any]:
        """
        Get escalation status for an alert.
        
        Args:
            alert_id: ID of alert
            
        Returns:
            Dictionary with escalation status information
        """
        try:
            schedule = await self.scheduler.get_escalation_schedule(alert_id)
            
            if not schedule:
                return {
                    "alert_id": str(alert_id),
                    "has_escalation": False,
                    "status": "no_escalation"
                }
            
            next_event = schedule.get_next_scheduled_event()
            
            return {
                "alert_id": str(alert_id),
                "has_escalation": True,
                "current_step": schedule.current_step,
                "total_steps": len(schedule.escalation_matrix.escalation_steps),
                "is_paused": schedule.escalation_paused,
                "pause_reason": schedule.paused_reason,
                "next_escalation_at": next_event.scheduled_at.isoformat() if next_event else None,
                "next_escalation_role": next_event.target_role.value if next_event else None,
                "has_overdue_events": schedule.has_overdue_events(),
                "completed_events": len(schedule.completed_events),
                "pending_events": len(schedule.pending_events),
                "last_updated": schedule.last_updated.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get escalation status for alert {alert_id}: {str(e)}")
            return {
                "alert_id": str(alert_id),
                "has_escalation": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _escalation_processing_loop(self):
        """Background loop for processing escalation events."""
        self.logger.info("Starting escalation processing loop")
        
        while self._running:
            try:
                # Get overdue escalation events
                overdue_events = await self.scheduler.get_overdue_escalations()
                
                if overdue_events:
                    self.logger.info(f"Processing {len(overdue_events)} overdue escalation events")
                
                # Process each overdue event
                for event in overdue_events:
                    if not self._running:
                        break
                    
                    await self._process_escalation_event(event)
                
                # Sleep before next check (adjust based on requirements)
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in escalation processing loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
        
        self.logger.info("Escalation processing loop stopped")
    
    async def _process_escalation_event(self, event: EscalationEvent):
        """Process a single escalation event."""
        try:
            # Get escalation schedule
            schedule = await self.scheduler.get_escalation_schedule(event.alert_id)
            if not schedule or schedule.escalation_paused:
                return
            
            # Get escalation step configuration
            if event.escalation_step >= len(schedule.escalation_matrix.escalation_steps):
                self.logger.warning(
                    f"Escalation step {event.escalation_step} out of range for alert {event.alert_id}"
                )
                return
            
            escalation_step = schedule.escalation_matrix.escalation_steps[event.escalation_step]
            
            # TODO: Get alert details (in production, this would fetch from database)
            # For now, create mock alert for processing
            mock_alert = self._create_mock_alert_for_event(event)
            
            # Execute escalation
            success, error_message, delivery_attempts = await self.executor.execute_escalation_event(
                event, mock_alert, escalation_step
            )
            
            # Mark event as executed
            await self.scheduler.mark_escalation_executed(
                event.event_id, success, error_message, delivery_attempts
            )
            
            self.logger.info(
                f"Processed escalation event {event.event_id} "
                f"for alert {event.alert_id} (success: {success})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process escalation event {event.event_id}: {str(e)}")
            
            # Mark event as failed
            await self.scheduler.mark_escalation_executed(
                event.event_id, False, str(e)
            )
    
    def _create_mock_alert_for_event(self, event: EscalationEvent) -> Alert:
        """Create mock alert for escalation processing (temporary)."""
        from ..models.news2 import NEWS2Result, RiskCategory
        
        # In production, this would fetch the actual alert from database
        mock_patient = Patient(
            patient_id="MOCK_PATIENT",
            ward_id="MOCK_WARD",
            bed_number="MOCK_BED",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="MOCK_NURSE",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        mock_news2_result = NEWS2Result(
            total_score=7,
            individual_scores={"test": 3},
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=5.0
        )
        
        from .alert_generation import AlertDecisionEngine, AlertDecision
        
        mock_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=mock_patient.patient_id,
            news2_result=mock_news2_result,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            threshold_applied=None,
            reasoning="Mock escalation event processing",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=1.0,
            single_param_trigger=False,
            suppressed=False,
            ward_id=mock_patient.ward_id
        )
        
        return Alert(
            alert_id=event.alert_id,
            patient_id=mock_patient.patient_id,
            patient=mock_patient,
            alert_decision=mock_decision,
            alert_level=AlertLevel.CRITICAL,
            alert_priority=AlertPriority.IMMEDIATE,
            title="Mock Alert for Escalation",
            message="Mock alert message",
            clinical_context={},
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=3,
            next_escalation_at=None,
            resolved_at=None,
            resolved_by=None
        )