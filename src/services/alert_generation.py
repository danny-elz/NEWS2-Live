"""
Alert Generation Engine for NEWS2 Live System

This module implements the core alert generation logic, including NEWS2 score evaluation,
threshold matching, escalation matrix management, and alert decision auditing.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4, UUID

from ..models.alerts import (
    AlertLevel, AlertPriority, AlertDecision, AlertThreshold, 
    EscalationMatrix, EscalationStep, EscalationRole, Alert, AlertStatus,
    AlertGenerationException, ThresholdConfigurationException
)
from ..models.news2 import NEWS2Result, RiskCategory
from ..models.patient import Patient
from ..services.audit import AuditLogger, AuditOperation
from ..services.alert_suppression import SuppressionEngine, SuppressionDecisionLogger


class NEWS2ScoreEvaluator:
    """
    Evaluates NEWS2 scores against clinical thresholds to determine alert levels.
    
    Implements the core logic for:
    - Critical parameter detection (single parameter score = 3)
    - NEWS2 total score evaluation
    - Ward-specific threshold application
    - Alert level determination with clinical reasoning
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    def evaluate_news2_score(
        self, 
        news2_result: NEWS2Result, 
        patient: Patient,
        ward_thresholds: List[AlertThreshold]
    ) -> Tuple[AlertLevel, AlertPriority, str, bool]:
        """
        Evaluate NEWS2 score and determine alert level.
        
        Args:
            news2_result: Calculated NEWS2 result
            patient: Patient information
            ward_thresholds: Active thresholds for patient's ward
            
        Returns:
            Tuple of (alert_level, alert_priority, reasoning, single_param_trigger)
        """
        try:
            # Check for critical single parameter (score = 3) - HIGHEST PRIORITY
            single_param_trigger = self._has_critical_single_parameter(news2_result)
            if single_param_trigger:
                reasoning = self._build_single_param_reasoning(news2_result)
                self.logger.info(f"Critical single parameter detected for patient {patient.patient_id}: {reasoning}")
                return AlertLevel.CRITICAL, AlertPriority.LIFE_THREATENING, reasoning, True
            
            # Check for critical total score (NEWS2 ≥7) - SECOND PRIORITY
            if news2_result.total_score >= 7:
                reasoning = f"Critical NEWS2 total score {news2_result.total_score} (≥7) requires immediate clinical review"
                self.logger.info(f"Critical NEWS2 score for patient {patient.patient_id}: {news2_result.total_score}")
                return AlertLevel.CRITICAL, AlertPriority.IMMEDIATE, reasoning, False
            
            # Apply ward-specific thresholds for non-critical scores
            alert_level, priority, reasoning = self._evaluate_ward_thresholds(
                news2_result, patient, ward_thresholds
            )
            
            return alert_level, priority, reasoning, False
            
        except Exception as e:
            self.logger.error(f"Error evaluating NEWS2 score for patient {patient.patient_id}: {str(e)}")
            raise AlertGenerationException(f"NEWS2 score evaluation failed: {str(e)}")
    
    def _has_critical_single_parameter(self, news2_result: NEWS2Result) -> bool:
        """
        Check if any single parameter has score = 3 (immediate alert).
        
        Critical parameters requiring immediate response:
        - Respiratory rate ≤8 or ≥25
        - SpO2 ≤91% (Scale 1) or ≤83% (Scale 2)
        - Temperature ≤35.0°C or ≥39.1°C
        - Systolic BP ≤90 or ≥220
        - Heart rate ≤40 or ≥131
        - Consciousness: Confusion/Voice/Pain/Unresponsive
        """
        critical_params = [
            news2_result.individual_scores.get("respiratory_rate", 0) == 3,
            news2_result.individual_scores.get("spo2", 0) == 3,
            news2_result.individual_scores.get("temperature", 0) == 3,
            news2_result.individual_scores.get("systolic_bp", 0) == 3,
            news2_result.individual_scores.get("heart_rate", 0) == 3,
            news2_result.individual_scores.get("consciousness", 0) == 3
        ]
        
        return any(critical_params)
    
    def _build_single_param_reasoning(self, news2_result: NEWS2Result) -> str:
        """Build reasoning text for single parameter alerts."""
        critical_parameters = []
        
        for param, score in news2_result.individual_scores.items():
            if score == 3:
                param_name = param.replace("_", " ").title()
                critical_parameters.append(param_name)
        
        if len(critical_parameters) == 1:
            return f"CRITICAL: {critical_parameters[0]} score = 3 requires immediate clinical assessment"
        else:
            params_str = ", ".join(critical_parameters)
            return f"CRITICAL: Multiple parameters ({params_str}) score = 3 require immediate clinical assessment"
    
    def _evaluate_ward_thresholds(
        self, 
        news2_result: NEWS2Result, 
        patient: Patient, 
        ward_thresholds: List[AlertThreshold]
    ) -> Tuple[AlertLevel, AlertPriority, str]:
        """Evaluate NEWS2 score against ward-specific thresholds."""
        
        # Filter active thresholds for current time
        active_thresholds = [t for t in ward_thresholds if t.is_active_now()]
        
        if not active_thresholds:
            # Fall back to default thresholds if no ward-specific config
            return self._apply_default_thresholds(news2_result)
        
        # Find matching threshold (highest priority first)
        matching_thresholds = []
        for threshold in active_thresholds:
            if threshold.matches_news2_score(news2_result.total_score):
                matching_thresholds.append(threshold)
        
        if not matching_thresholds:
            # Score doesn't match any threshold - treat as LOW
            return AlertLevel.LOW, AlertPriority.ROUTINE, f"NEWS2 score {news2_result.total_score} below configured thresholds"
        
        # Select highest priority threshold
        highest_threshold = max(matching_thresholds, key=lambda t: self._get_threshold_priority(t.alert_level))
        
        priority = self._map_alert_level_to_priority(highest_threshold.alert_level)
        reasoning = f"NEWS2 score {news2_result.total_score} matches {highest_threshold.alert_level.value} threshold (≥{highest_threshold.news2_min}) for ward {patient.ward_id}"
        
        return highest_threshold.alert_level, priority, reasoning
    
    def _apply_default_thresholds(self, news2_result: NEWS2Result) -> Tuple[AlertLevel, AlertPriority, str]:
        """Apply default NEWS2 thresholds when no ward-specific config exists."""
        score = news2_result.total_score
        
        if score >= 5:  # High risk
            return AlertLevel.HIGH, AlertPriority.URGENT, f"NEWS2 score {score} indicates high risk (≥5)"
        elif score >= 3:  # Medium risk
            return AlertLevel.MEDIUM, AlertPriority.URGENT, f"NEWS2 score {score} indicates medium risk (3-4)"
        elif score >= 1:  # Low risk with monitoring
            return AlertLevel.LOW, AlertPriority.ROUTINE, f"NEWS2 score {score} indicates low risk with monitoring (1-2)"
        else:  # score = 0
            return AlertLevel.LOW, AlertPriority.ROUTINE, f"NEWS2 score {score} indicates routine monitoring"
    
    def _get_threshold_priority(self, alert_level: AlertLevel) -> int:
        """Get numeric priority for threshold selection."""
        priority_map = {
            AlertLevel.CRITICAL: 4,
            AlertLevel.HIGH: 3,
            AlertLevel.MEDIUM: 2,
            AlertLevel.LOW: 1
        }
        return priority_map.get(alert_level, 0)
    
    def _map_alert_level_to_priority(self, alert_level: AlertLevel) -> AlertPriority:
        """Map alert level to delivery priority."""
        priority_map = {
            AlertLevel.CRITICAL: AlertPriority.IMMEDIATE,
            AlertLevel.HIGH: AlertPriority.URGENT,
            AlertLevel.MEDIUM: AlertPriority.URGENT,
            AlertLevel.LOW: AlertPriority.ROUTINE
        }
        return priority_map.get(alert_level, AlertPriority.ROUTINE)


class AlertDecisionEngine:
    """
    Makes alert generation decisions based on NEWS2 evaluation and clinical rules.
    
    Responsibilities:
    - Coordinate NEWS2 score evaluation
    - Apply alert suppression rules
    - Create alert decisions with audit trail
    - Measure and log decision latency
    """
    
    def __init__(self, audit_logger: AuditLogger, suppression_engine: Optional[SuppressionEngine] = None):
        self.audit_logger = audit_logger
        self.score_evaluator = NEWS2ScoreEvaluator(audit_logger)
        self.suppression_engine = suppression_engine
        self.logger = logging.getLogger(__name__)
    
    async def make_alert_decision(
        self,
        news2_result: NEWS2Result,
        patient: Patient,
        ward_thresholds: List[AlertThreshold],
        user_id: str = "SYSTEM"
    ) -> AlertDecision:
        """
        Make comprehensive alert decision for NEWS2 result.
        
        Args:
            news2_result: Calculated NEWS2 result
            patient: Patient information
            ward_thresholds: Active thresholds for patient's ward
            user_id: ID of user triggering alert evaluation
            
        Returns:
            Complete alert decision with reasoning and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Evaluate NEWS2 score against thresholds
            alert_level, alert_priority, reasoning, single_param_trigger = (
                self.score_evaluator.evaluate_news2_score(
                    news2_result, patient, ward_thresholds
                )
            )
            
            # Check for alert suppression using smart suppression engine
            suppressed = False
            suppression_reasoning = ""
            
            if self.suppression_engine and alert_level != AlertLevel.CRITICAL and not single_param_trigger:
                # Create a temporary alert for suppression evaluation
                temp_alert = await self._create_temp_alert_for_suppression(
                    alert_decision, patient, alert_level, alert_priority
                )
                
                suppression_decision = await self.suppression_engine.should_suppress(temp_alert)
                suppressed = suppression_decision.suppressed
                
                if suppressed:
                    suppression_reasoning = f" [SUPPRESSED: {suppression_decision.reason} - Confidence: {suppression_decision.confidence_score:.2f}]"
                    reasoning += suppression_reasoning
            
            # Calculate decision latency
            decision_timestamp = datetime.now(timezone.utc)
            latency_ms = (decision_timestamp - start_time).total_seconds() * 1000
            
            # Find applied threshold
            applied_threshold = self._find_applied_threshold(
                news2_result.total_score, ward_thresholds
            ) if not single_param_trigger else None
            
            # Create alert decision
            decision = AlertDecision(
                decision_id=uuid4(),
                patient_id=patient.patient_id,
                news2_result=news2_result,
                alert_level=alert_level,
                alert_priority=alert_priority,
                threshold_applied=applied_threshold,
                reasoning=reasoning,
                decision_timestamp=decision_timestamp,
                generation_latency_ms=latency_ms,
                single_param_trigger=single_param_trigger,
                suppressed=suppressed,
                ward_id=patient.ward_id
            )
            
            # Log decision for audit
            await self._audit_alert_decision(decision, user_id)
            
            # Log performance metrics
            if latency_ms > 5000:  # Alert if decision takes >5 seconds
                self.logger.warning(
                    f"Alert decision latency {latency_ms:.1f}ms exceeded 5s target for patient {patient.patient_id}"
                )
            
            self.logger.info(
                f"Alert decision: {alert_level.value} alert for patient {patient.patient_id} "
                f"(NEWS2: {news2_result.total_score}, latency: {latency_ms:.1f}ms)"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Alert decision failed for patient {patient.patient_id}: {str(e)}")
            raise AlertGenerationException(f"Alert decision generation failed: {str(e)}")
    
    async def _create_temp_alert_for_suppression(
        self,
        alert_decision: AlertDecision,
        patient: Patient,
        alert_level: AlertLevel,
        alert_priority: AlertPriority
    ) -> Alert:
        """
        Create a temporary alert object for suppression evaluation.
        
        The suppression engine needs an Alert object to evaluate, but we haven't
        created the full alert yet. This creates a minimal Alert for evaluation.
        """
        from ..models.alerts import Alert, AlertStatus
        
        # Build minimal alert title and message for suppression evaluation
        title = f"{alert_level.value.upper()} ALERT - NEWS2 Score {alert_decision.news2_result.total_score}"
        message = f"Patient {patient.patient_id} requires {alert_level.value} level response."
        
        # Build minimal clinical context
        clinical_context = {
            "news2_total_score": alert_decision.news2_result.total_score,
            "news2_individual_scores": alert_decision.news2_result.individual_scores,
            "patient_age": patient.age,
            "patient_ward": patient.ward_id,
            "single_param_trigger": alert_decision.single_param_trigger
        }
        
        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=alert_level,
            alert_priority=alert_priority,
            title=title,
            message=message,
            clinical_context=clinical_context,
            created_at=alert_decision.decision_timestamp,
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=3,  # Default escalation steps
            next_escalation_at=None,
            resolved_at=None,
            resolved_by=None
        )
    
    def _find_applied_threshold(
        self, 
        news2_score: int, 
        ward_thresholds: List[AlertThreshold]
    ) -> Optional[AlertThreshold]:
        """Find the threshold that was applied for this score."""
        active_thresholds = [t for t in ward_thresholds if t.is_active_now()]
        
        for threshold in active_thresholds:
            if threshold.matches_news2_score(news2_score):
                return threshold
        
        return None
    
    async def _audit_alert_decision(self, decision: AlertDecision, user_id: str):
        """Create audit trail for alert decision."""
        try:
            audit_entry = self.audit_logger.create_audit_entry(
                table_name="alert_decisions",
                operation=AuditOperation.INSERT,
                user_id=user_id,
                patient_id=decision.patient_id,
                new_values=decision.to_dict()
            )
            
            # TODO: Store audit entry in database when persistence layer is ready
            self.logger.debug(f"Alert decision audited: {decision.decision_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to audit alert decision: {str(e)}")
            # Don't fail the alert generation due to audit issues
            pass


class DefaultThresholdManager:
    """
    Manages default alert thresholds and escalation matrices.
    
    Provides standard configurations when ward-specific settings are not available.
    """
    
    @staticmethod
    def get_default_thresholds(ward_id: str) -> List[AlertThreshold]:
        """Get default alert thresholds for a ward."""
        base_time = datetime.now(timezone.utc)
        
        return [
            # Critical threshold - NEWS2 ≥7
            AlertThreshold(
                threshold_id=uuid4(),
                ward_id=ward_id,
                alert_level=AlertLevel.CRITICAL,
                news2_min=7,
                news2_max=None,
                single_param_trigger=True,
                active_hours=(0, 24),  # 24/7
                enabled=True,
                created_at=base_time,
                updated_at=base_time
            ),
            
            # High threshold - NEWS2 5-6
            AlertThreshold(
                threshold_id=uuid4(),
                ward_id=ward_id,
                alert_level=AlertLevel.HIGH,
                news2_min=5,
                news2_max=6,
                single_param_trigger=False,
                active_hours=(0, 24),  # 24/7
                enabled=True,
                created_at=base_time,
                updated_at=base_time
            ),
            
            # Medium threshold - NEWS2 3-4
            AlertThreshold(
                threshold_id=uuid4(),
                ward_id=ward_id,
                alert_level=AlertLevel.MEDIUM,
                news2_min=3,
                news2_max=4,
                single_param_trigger=False,
                active_hours=(0, 24),  # 24/7
                enabled=True,
                created_at=base_time,
                updated_at=base_time
            ),
            
            # Low threshold - NEWS2 1-2
            AlertThreshold(
                threshold_id=uuid4(),
                ward_id=ward_id,
                alert_level=AlertLevel.LOW,
                news2_min=1,
                news2_max=2,
                single_param_trigger=False,
                active_hours=(0, 24),  # 24/7
                enabled=True,
                created_at=base_time,
                updated_at=base_time
            )
        ]
    
    @staticmethod
    def get_default_escalation_matrix(ward_id: str, alert_level: AlertLevel) -> EscalationMatrix:
        """Get default escalation matrix for ward and alert level."""
        base_time = datetime.now(timezone.utc)
        
        # Define escalation steps based on alert level
        if alert_level == AlertLevel.CRITICAL:
            steps = [
                EscalationStep(EscalationRole.WARD_NURSE, 0),      # Immediate
                EscalationStep(EscalationRole.CHARGE_NURSE, 15),   # After 15 min
                EscalationStep(EscalationRole.DOCTOR, 30),         # After 30 min
                EscalationStep(EscalationRole.RAPID_RESPONSE, 45)  # After 45 min
            ]
        elif alert_level == AlertLevel.HIGH:
            steps = [
                EscalationStep(EscalationRole.WARD_NURSE, 0),      # Immediate
                EscalationStep(EscalationRole.CHARGE_NURSE, 30),   # After 30 min
                EscalationStep(EscalationRole.DOCTOR, 60)          # After 60 min
            ]
        elif alert_level == AlertLevel.MEDIUM:
            steps = [
                EscalationStep(EscalationRole.WARD_NURSE, 0),      # Immediate
                EscalationStep(EscalationRole.CHARGE_NURSE, 60)    # After 60 min
            ]
        else:  # LOW
            steps = [
                EscalationStep(EscalationRole.WARD_NURSE, 0)       # Immediate only
            ]
        
        return EscalationMatrix(
            matrix_id=uuid4(),
            ward_id=ward_id,
            alert_level=alert_level,
            escalation_steps=steps,
            enabled=True,
            created_at=base_time,
            updated_at=base_time
        )


class AlertGenerator:
    """
    High-level alert generation service that coordinates all alert creation activities.
    
    Main entry point for alert generation workflow.
    """
    
    def __init__(self, audit_logger: AuditLogger, suppression_engine: Optional[SuppressionEngine] = None):
        self.audit_logger = audit_logger
        self.decision_engine = AlertDecisionEngine(audit_logger, suppression_engine)
        self.threshold_manager = DefaultThresholdManager()
        self.suppression_engine = suppression_engine
        self.logger = logging.getLogger(__name__)
    
    async def generate_alert(
        self,
        news2_result: NEWS2Result,
        patient: Patient,
        ward_thresholds: Optional[List[AlertThreshold]] = None,
        user_id: str = "SYSTEM"
    ) -> Optional[Alert]:
        """
        Generate alert from NEWS2 result if thresholds are met.
        
        Args:
            news2_result: Calculated NEWS2 result
            patient: Patient information
            ward_thresholds: Optional ward-specific thresholds
            user_id: ID of user triggering alert generation
            
        Returns:
            Generated alert or None if no alert needed
        """
        try:
            # Use ward-specific thresholds or fall back to defaults
            if ward_thresholds is None:
                ward_thresholds = self.threshold_manager.get_default_thresholds(patient.ward_id)
            
            # Make alert decision
            alert_decision = await self.decision_engine.make_alert_decision(
                news2_result, patient, ward_thresholds, user_id
            )
            
            # Skip alert creation if suppressed or LOW priority routine monitoring
            if alert_decision.suppressed or (
                alert_decision.alert_level == AlertLevel.LOW and 
                alert_decision.alert_priority == AlertPriority.ROUTINE and
                news2_result.total_score == 0
            ):
                self.logger.info(f"Alert decision made but not creating alert: {alert_decision.reasoning}")
                return None
            
            # Create alert object
            alert = await self._create_alert(alert_decision, patient)
            
            self.logger.info(f"Alert generated: {alert.alert_level.value} for patient {patient.patient_id}")
            return alert
            
        except Exception as e:
            self.logger.error(f"Alert generation failed for patient {patient.patient_id}: {str(e)}")
            raise AlertGenerationException(f"Alert generation failed: {str(e)}")
    
    async def _create_alert(self, alert_decision: AlertDecision, patient: Patient) -> Alert:
        """Create complete alert object from decision."""
        
        # Build alert title and message
        title, message = self._build_alert_content(alert_decision, patient)
        
        # Get escalation matrix
        escalation_matrix = self.threshold_manager.get_default_escalation_matrix(
            patient.ward_id, alert_decision.alert_level
        )
        
        # Calculate next escalation time
        next_escalation_at = None
        if escalation_matrix.escalation_steps:
            first_step = escalation_matrix.escalation_steps[0]
            if first_step.delay_minutes > 0:
                next_escalation_at = alert_decision.decision_timestamp + timedelta(minutes=first_step.delay_minutes)
            elif len(escalation_matrix.escalation_steps) > 1:
                # First step is immediate, calculate for second step
                second_step = escalation_matrix.escalation_steps[1]
                next_escalation_at = alert_decision.decision_timestamp + timedelta(minutes=second_step.delay_minutes)
        
        # Build clinical context
        clinical_context = {
            "news2_total_score": alert_decision.news2_result.total_score,
            "news2_individual_scores": alert_decision.news2_result.individual_scores,
            "risk_category": alert_decision.news2_result.risk_category.value,
            "scale_used": alert_decision.news2_result.scale_used,
            "patient_age": patient.age,
            "patient_ward": patient.ward_id,
            "is_copd_patient": patient.is_copd_patient,
            "single_param_trigger": alert_decision.single_param_trigger
        }
        
        return Alert(
            alert_id=uuid4(),
            patient_id=patient.patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=alert_decision.alert_level,
            alert_priority=alert_decision.alert_priority,
            title=title,
            message=message,
            clinical_context=clinical_context,
            created_at=alert_decision.decision_timestamp,
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=len(escalation_matrix.escalation_steps) - 1,
            next_escalation_at=next_escalation_at,
            resolved_at=None,
            resolved_by=None
        )
    
    def _build_alert_content(self, alert_decision: AlertDecision, patient: Patient) -> Tuple[str, str]:
        """Build alert title and message content."""
        news2_result = alert_decision.news2_result
        
        if alert_decision.alert_level == AlertLevel.CRITICAL:
            if alert_decision.single_param_trigger:
                title = f"CRITICAL ALERT - Single Parameter Red Flag"
                message = (
                    f"Patient {patient.patient_id} ({patient.age}y) in {patient.ward_id} "
                    f"has critical vital sign parameter requiring IMMEDIATE assessment.\n\n"
                    f"NEWS2 Score: {news2_result.total_score}\n"
                    f"Critical Parameter Alert: {alert_decision.reasoning}\n"
                    f"Scale Used: {news2_result.scale_used}"
                )
            else:
                title = f"CRITICAL ALERT - NEWS2 Score {news2_result.total_score}"
                message = (
                    f"Patient {patient.patient_id} ({patient.age}y) in {patient.ward_id} "
                    f"has critical NEWS2 score requiring IMMEDIATE clinical response.\n\n"
                    f"NEWS2 Score: {news2_result.total_score}/20\n"
                    f"Risk Category: {news2_result.risk_category.value}\n"
                    f"Reasoning: {alert_decision.reasoning}"
                )
        elif alert_decision.alert_level == AlertLevel.HIGH:
            title = f"HIGH ALERT - NEWS2 Score {news2_result.total_score}"
            message = (
                f"Patient {patient.patient_id} ({patient.age}y) in {patient.ward_id} "
                f"requires urgent medical review.\n\n"
                f"NEWS2 Score: {news2_result.total_score}/20\n"
                f"Risk Category: {news2_result.risk_category.value}\n"
                f"Reasoning: {alert_decision.reasoning}"
            )
        elif alert_decision.alert_level == AlertLevel.MEDIUM:
            title = f"MEDIUM ALERT - NEWS2 Score {news2_result.total_score}"
            message = (
                f"Patient {patient.patient_id} ({patient.age}y) in {patient.ward_id} "
                f"requires increased monitoring and medical review.\n\n"
                f"NEWS2 Score: {news2_result.total_score}/20\n"
                f"Risk Category: {news2_result.risk_category.value}\n"
                f"Reasoning: {alert_decision.reasoning}"
            )
        else:  # LOW
            title = f"LOW ALERT - NEWS2 Score {news2_result.total_score}"
            message = (
                f"Patient {patient.patient_id} ({patient.age}y) in {patient.ward_id} "
                f"requires routine monitoring.\n\n"
                f"NEWS2 Score: {news2_result.total_score}/20\n"
                f"Risk Category: {news2_result.risk_category.value}\n"
                f"Reasoning: {alert_decision.reasoning}"
            )
        
        return title, message