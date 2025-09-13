"""
Alert Suppression Engine for NEWS2 Live System

This module implements intelligent alert suppression logic to reduce alert fatigue
while maintaining 100% critical alert detection. Includes time-based suppression,
pattern recognition, manual overrides, and comprehensive audit logging.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from ..models.alerts import Alert, AlertLevel, AlertStatus
from ..models.news2 import NEWS2Result
from ..models.patient import Patient
from ..services.audit import AuditLogger, AuditOperation


@dataclass
class SuppressionDecision:
    """Decision made by suppression engine."""
    decision_id: UUID
    alert_id: UUID
    patient_id: str
    suppressed: bool
    reason: str
    confidence_score: float
    decision_timestamp: datetime
    suppression_expires_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "alert_id": str(self.alert_id),
            "patient_id": self.patient_id,
            "suppressed": self.suppressed,
            "reason": self.reason,
            "confidence_score": self.confidence_score,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "suppression_expires_at": self.suppression_expires_at.isoformat() if self.suppression_expires_at else None,
            "metadata": self.metadata
        }


@dataclass
class AlertAcknowledgment:
    """Record of alert acknowledgment for suppression tracking."""
    ack_id: UUID
    alert_id: UUID
    patient_id: str
    news2_score: int
    acknowledged_by: str
    acknowledged_at: datetime
    alert_level: AlertLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ack_id": str(self.ack_id),
            "alert_id": str(self.alert_id),
            "patient_id": self.patient_id,
            "news2_score": self.news2_score,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat(),
            "alert_level": self.alert_level.value
        }


@dataclass
class SuppressionOverride:
    """Manual suppression override by clinical staff."""
    override_id: UUID
    patient_id: str
    nurse_id: str
    justification: str
    expires_at: datetime
    created_at: datetime
    is_active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_id": str(self.override_id),
            "patient_id": self.patient_id,
            "nurse_id": self.nurse_id,
            "justification": self.justification,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }


class PatternDetector:
    """
    Detects stable score patterns to identify false positive scenarios.
    
    Uses statistical analysis to identify patients with stable high scores
    that generate repetitive alerts without clinical deterioration.
    Includes ML-ready framework for learning from suppression outcomes.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.stability_threshold = timedelta(hours=2)
        self.score_variance_limit = 1
        self.logger = logging.getLogger(__name__)
        
        # ML-ready framework configuration
        self.ml_confidence_threshold = 0.7
        self.pattern_learning_enabled = True
    
    async def is_stable_pattern(self, alert: Alert) -> bool:
        """
        Detect if patient has stable high score pattern.
        
        Args:
            alert: Alert to evaluate for pattern stability
            
        Returns:
            True if stable pattern detected, False otherwise
        """
        try:
            history = await self.get_score_history(alert.patient_id, hours=4)
            
            if len(history) < 6:
                return False
            
            scores = [entry['total_score'] for entry in history]
            variance = statistics.variance(scores)
            
            if variance <= self.score_variance_limit:
                await self._log_pattern_suppression(alert, "stable_high_score")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting stable pattern for patient {alert.patient_id}: {str(e)}")
            return False
    
    async def get_score_history(self, patient_id: str, hours: int = 4) -> List[Dict[str, Any]]:
        """Get recent NEWS2 score history for pattern analysis."""
        try:
            history_key = f"patient_score_history:{patient_id}"
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            raw_history = await self.redis.zrevrangebyscore(
                history_key,
                '+inf',
                cutoff_time.timestamp(),
                withscores=False
            )
            
            history = []
            for entry_json in raw_history:
                entry = json.loads(entry_json)
                history.append(entry)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error retrieving score history for {patient_id}: {str(e)}")
            return []
    
    async def detect_deterioration_trend(self, patient_id: str) -> bool:
        """Detect if patient is actually deteriorating vs stable."""
        try:
            history = await self.get_score_history(patient_id, hours=2)
            
            if len(history) < 3:
                return False
            
            times = [(datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) - 
                     datetime.fromisoformat(history[0]['timestamp'].replace('Z', '+00:00'))).total_seconds()
                    for h in history]
            scores = [h['total_score'] for h in history]
            
            slope = self._calculate_trend_slope(times, scores)
            return slope > 0.5
            
        except Exception as e:
            self.logger.error(f"Error detecting deterioration trend for {patient_id}: {str(e)}")
            return False
    
    def _calculate_trend_slope(self, times: List[float], scores: List[int]) -> float:
        """Calculate linear regression slope for trend analysis."""
        if len(times) != len(scores) or len(times) < 2:
            return 0.0
        
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(times, scores))
        sum_x2 = sum(x * x for x in times)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    async def _log_pattern_suppression(self, alert: Alert, pattern_type: str):
        """Log pattern-based suppression decision."""
        pattern_key = f"pattern_suppression:{alert.patient_id}:{pattern_type}"
        pattern_data = {
            "alert_id": str(alert.alert_id),
            "pattern_type": pattern_type,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "news2_score": alert.alert_decision.news2_result.total_score
        }
        
        await self.redis.setex(
            pattern_key,
            timedelta(hours=1),
            json.dumps(pattern_data)
        )
    
    async def calculate_confidence_score(self, alert: Alert, pattern_type: str) -> float:
        """
        Calculate confidence score for suppression decision.
        
        Uses statistical analysis and historical outcomes to score confidence
        in suppression decisions for ML learning framework.
        """
        try:
            base_confidence = 0.5
            
            # Historical pattern success rate
            pattern_success = await self._get_pattern_success_rate(alert.patient_id, pattern_type)
            base_confidence += pattern_success * 0.3
            
            # Score stability factor
            history = await self.get_score_history(alert.patient_id, hours=6)
            if len(history) >= 4:
                scores = [entry['total_score'] for entry in history]
                variance = statistics.variance(scores)
                stability_factor = max(0, 1 - (variance / 3))  # Normalize variance
                base_confidence += stability_factor * 0.2
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    async def learn_from_outcome(self, suppression_decision_id: UUID, outcome: str, clinical_notes: Optional[str] = None):
        """
        ML-ready framework for learning from suppression outcomes.
        
        Records suppression decision outcomes for pattern learning and
        false positive reduction improvement over time.
        """
        try:
            if not self.pattern_learning_enabled:
                return
            
            learning_data = {
                "suppression_decision_id": str(suppression_decision_id),
                "outcome": outcome,  # 'correct_suppression', 'false_positive', 'missed_deterioration'
                "clinical_notes": clinical_notes,
                "recorded_at": datetime.now(timezone.utc).isoformat()
            }
            
            learning_key = f"ml_learning:{suppression_decision_id}"
            await self.redis.setex(
                learning_key,
                timedelta(days=30),  # Keep learning data for 30 days
                json.dumps(learning_data)
            )
            
            # Update pattern success rates
            await self._update_pattern_statistics(suppression_decision_id, outcome)
            
            self.logger.info(f"Recorded learning outcome: {outcome} for decision {suppression_decision_id}")
            
        except Exception as e:
            self.logger.error(f"Error recording learning outcome: {str(e)}")
    
    async def _get_pattern_success_rate(self, patient_id: str, pattern_type: str) -> float:
        """Get historical success rate for pattern type."""
        try:
            stats_key = f"pattern_stats:{patient_id}:{pattern_type}"
            stats_data = await self.redis.get(stats_key)
            
            if not stats_data:
                return 0.5  # Default neutral confidence
            
            stats = json.loads(stats_data)
            total_decisions = stats.get('total_decisions', 0)
            correct_decisions = stats.get('correct_decisions', 0)
            
            if total_decisions == 0:
                return 0.5
            
            return correct_decisions / total_decisions
            
        except Exception as e:
            self.logger.error(f"Error getting pattern success rate: {str(e)}")
            return 0.5
    
    async def _update_pattern_statistics(self, suppression_decision_id: UUID, outcome: str):
        """Update pattern statistics for ML learning."""
        try:
            # Retrieve original decision to get pattern info
            decision_key = f"suppression_decision:{suppression_decision_id}"
            decision_data = await self.redis.get(decision_key)
            
            if not decision_data:
                return
            
            decision = json.loads(decision_data)
            pattern_type = decision.get('reason', 'unknown')
            patient_id = decision.get('patient_id')
            
            if not patient_id:
                return
            
            # Update statistics
            stats_key = f"pattern_stats:{patient_id}:{pattern_type}"
            stats_data = await self.redis.get(stats_key)
            
            if stats_data:
                stats = json.loads(stats_data)
            else:
                stats = {"total_decisions": 0, "correct_decisions": 0}
            
            stats["total_decisions"] += 1
            if outcome == 'correct_suppression':
                stats["correct_decisions"] += 1
            
            await self.redis.setex(
                stats_key,
                timedelta(days=90),  # Keep stats for 90 days
                json.dumps(stats)
            )
            
        except Exception as e:
            self.logger.error(f"Error updating pattern statistics: {str(e)}")


class SuppressionEngine:
    """
    Main suppression decision engine that coordinates all suppression rules.
    
    Implements smart alert suppression logic that reduces alert volume by 50%+
    while maintaining 100% critical alert detection.
    """
    
    def __init__(self, redis_client: redis.Redis, audit_logger: AuditLogger):
        self.redis = redis_client
        self.audit_logger = audit_logger
        self.pattern_detector = PatternDetector(redis_client)
        self.logger = logging.getLogger(__name__)
        
        # Suppression configuration
        self.time_suppression_window = timedelta(minutes=30)
        self.score_increase_threshold = 2
    
    async def should_suppress(self, alert: Alert) -> SuppressionDecision:
        """
        Main suppression decision logic.
        
        Args:
            alert: Alert to evaluate for suppression
            
        Returns:
            SuppressionDecision with suppression determination
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # CRITICAL RULE: Never suppress critical alerts
            if alert.alert_level == AlertLevel.CRITICAL:
                decision = SuppressionDecision(
                    decision_id=uuid4(),
                    alert_id=alert.alert_id,
                    patient_id=alert.patient_id,
                    suppressed=False,
                    reason="NEVER_SUPPRESS_CRITICAL",
                    confidence_score=1.0,
                    decision_timestamp=start_time,
                    suppression_expires_at=None,
                    metadata={"alert_level": alert.alert_level.value}
                )
                await self._log_suppression_decision(decision)
                return decision
            
            # Check time-based suppression
            if await self.is_time_suppressed(alert):
                decision = SuppressionDecision(
                    decision_id=uuid4(),
                    alert_id=alert.alert_id,
                    patient_id=alert.patient_id,
                    suppressed=True,
                    reason="TIME_BASED_SUPPRESSION",
                    confidence_score=0.9,
                    decision_timestamp=start_time,
                    suppression_expires_at=start_time + self.time_suppression_window,
                    metadata={"suppression_window_minutes": self.time_suppression_window.total_seconds() / 60}
                )
                await self._log_suppression_decision(decision)
                return decision
            
            # Check pattern-based suppression
            if await self.pattern_detector.is_stable_pattern(alert):
                decision = SuppressionDecision(
                    decision_id=uuid4(),
                    alert_id=alert.alert_id,
                    patient_id=alert.patient_id,
                    suppressed=True,
                    reason="STABLE_PATTERN_SUPPRESSION",
                    confidence_score=0.8,
                    decision_timestamp=start_time,
                    suppression_expires_at=start_time + timedelta(hours=1),
                    metadata={"pattern_type": "stable_high_score"}
                )
                await self._log_suppression_decision(decision)
                return decision
            
            # Check manual overrides
            override = await self.get_active_override(alert.patient_id)
            if override:
                decision = SuppressionDecision(
                    decision_id=uuid4(),
                    alert_id=alert.alert_id,
                    patient_id=alert.patient_id,
                    suppressed=True,
                    reason="MANUAL_OVERRIDE",
                    confidence_score=1.0,
                    decision_timestamp=start_time,
                    suppression_expires_at=override.expires_at,
                    metadata={
                        "override_id": str(override.override_id),
                        "nurse_id": override.nurse_id,
                        "justification": override.justification
                    }
                )
                await self._log_suppression_decision(decision)
                return decision
            
            # No suppression applies
            decision = SuppressionDecision(
                decision_id=uuid4(),
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                suppressed=False,
                reason="NO_SUPPRESSION_APPLIES",
                confidence_score=1.0,
                decision_timestamp=start_time,
                suppression_expires_at=None,
                metadata={}
            )
            await self._log_suppression_decision(decision)
            return decision
            
        except Exception as e:
            self.logger.error(f"Suppression decision failed for alert {alert.alert_id}: {str(e)}")
            # Fail-safe: don't suppress on errors
            decision = SuppressionDecision(
                decision_id=uuid4(),
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                suppressed=False,
                reason="ERROR_NO_SUPPRESSION",
                confidence_score=0.0,
                decision_timestamp=start_time,
                suppression_expires_at=None,
                metadata={"error": str(e)}
            )
            await self._log_suppression_decision(decision)
            return decision
    
    async def is_time_suppressed(self, alert: Alert) -> bool:
        """
        Check if alert is within 30-minute suppression window.
        
        Suppression rules:
        - Suppress if recent acknowledgment within 30 minutes
        - Bypass suppression if score increased by 2+ points
        """
        try:
            last_ack = await self.get_last_acknowledgment(alert.patient_id)
            if not last_ack:
                return False
            
            time_since_ack = datetime.now(timezone.utc) - last_ack.acknowledged_at
            if time_since_ack < self.time_suppression_window:
                # Check if score increased by 2+ points
                score_delta = alert.alert_decision.news2_result.total_score - last_ack.news2_score
                if score_delta >= self.score_increase_threshold:
                    await self._log_suppression_bypass(alert, "score_increase", score_delta)
                    return False
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking time suppression for {alert.patient_id}: {str(e)}")
            return False
    
    async def get_last_acknowledgment(self, patient_id: str) -> Optional[AlertAcknowledgment]:
        """Get most recent alert acknowledgment for patient."""
        try:
            ack_key = f"patient_acknowledgments:{patient_id}"
            latest_ack = await self.redis.zrevrange(ack_key, 0, 0, withscores=False)
            
            if not latest_ack:
                return None
            
            ack_data = json.loads(latest_ack[0])
            return AlertAcknowledgment(
                ack_id=UUID(ack_data['ack_id']),
                alert_id=UUID(ack_data['alert_id']),
                patient_id=ack_data['patient_id'],
                news2_score=ack_data['news2_score'],
                acknowledged_by=ack_data['acknowledged_by'],
                acknowledged_at=datetime.fromisoformat(ack_data['acknowledged_at'].replace('Z', '+00:00')),
                alert_level=AlertLevel(ack_data['alert_level'])
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving last acknowledgment for {patient_id}: {str(e)}")
            return None
    
    async def get_active_override(self, patient_id: str) -> Optional[SuppressionOverride]:
        """Get active manual suppression override for patient."""
        try:
            override_key = f"suppression_override:{patient_id}"
            override_data = await self.redis.get(override_key)
            
            if not override_data:
                return None
            
            override_dict = json.loads(override_data)
            override = SuppressionOverride(
                override_id=UUID(override_dict['override_id']),
                patient_id=override_dict['patient_id'],
                nurse_id=override_dict['nurse_id'],
                justification=override_dict['justification'],
                expires_at=datetime.fromisoformat(override_dict['expires_at'].replace('Z', '+00:00')),
                created_at=datetime.fromisoformat(override_dict['created_at'].replace('Z', '+00:00')),
                is_active=override_dict['is_active']
            )
            
            # Check if override is still active
            if override.expires_at <= datetime.now(timezone.utc) or not override.is_active:
                await self.redis.delete(override_key)
                return None
            
            return override
            
        except Exception as e:
            self.logger.error(f"Error retrieving active override for {patient_id}: {str(e)}")
            return None
    
    async def record_acknowledgment(self, alert: Alert, acknowledged_by: str):
        """Record alert acknowledgment for suppression tracking."""
        try:
            acknowledgment = AlertAcknowledgment(
                ack_id=uuid4(),
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                news2_score=alert.alert_decision.news2_result.total_score,
                acknowledged_by=acknowledged_by,
                acknowledged_at=datetime.now(timezone.utc),
                alert_level=alert.alert_level
            )
            
            # Store in sorted set for time-based queries
            ack_key = f"patient_acknowledgments:{alert.patient_id}"
            await self.redis.zadd(
                ack_key,
                {json.dumps(acknowledgment.to_dict()): acknowledgment.acknowledged_at.timestamp()}
            )
            
            # Set expiration for data cleanup
            await self.redis.expire(ack_key, timedelta(hours=24))
            
            self.logger.info(f"Recorded acknowledgment for alert {alert.alert_id} by {acknowledged_by}")
            
        except Exception as e:
            self.logger.error(f"Error recording acknowledgment: {str(e)}")
    
    async def _log_suppression_decision(self, decision: SuppressionDecision):
        """Log suppression decision for audit and analytics."""
        try:
            # Store in Redis for quick access
            decision_key = f"suppression_decision:{decision.decision_id}"
            await self.redis.setex(
                decision_key,
                timedelta(hours=24),
                json.dumps(decision.to_dict())
            )
            
            # Create audit entry
            audit_entry = self.audit_logger.create_audit_entry(
                table_name="suppression_decisions",
                operation=AuditOperation.INSERT,
                user_id="SUPPRESSION_ENGINE",
                patient_id=decision.patient_id,
                new_values=decision.to_dict()
            )
            
            self.logger.info(
                f"Suppression decision: {decision.reason} for alert {decision.alert_id} "
                f"(suppressed: {decision.suppressed})"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging suppression decision: {str(e)}")
    
    async def _log_suppression_bypass(self, alert: Alert, reason: str, score_delta: int):
        """Log when suppression is bypassed due to score increase."""
        try:
            bypass_data = {
                "alert_id": str(alert.alert_id),
                "patient_id": alert.patient_id,
                "reason": reason,
                "score_delta": score_delta,
                "bypassed_at": datetime.now(timezone.utc).isoformat()
            }
            
            bypass_key = f"suppression_bypass:{alert.patient_id}:{reason}"
            await self.redis.setex(
                bypass_key,
                timedelta(hours=1),
                json.dumps(bypass_data)
            )
            
            self.logger.info(
                f"Suppression bypassed for alert {alert.alert_id}: {reason} (delta: {score_delta})"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging suppression bypass: {str(e)}")


class ManualOverrideManager:
    """
    Manages manual suppression overrides by clinical staff.
    
    Provides authentication, validation, and audit trail for clinical
    staff to manually override alert suppression when needed.
    """
    
    def __init__(self, redis_client: redis.Redis, audit_logger: AuditLogger):
        self.redis = redis_client
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def create_override(
        self,
        patient_id: str,
        nurse_id: str,
        justification: str,
        duration_minutes: int = 60
    ) -> SuppressionOverride:
        """
        Create manual suppression override with authentication.
        
        Args:
            patient_id: Patient to create override for
            nurse_id: ID of nurse creating override
            justification: Clinical justification for override
            duration_minutes: Override duration in minutes
            
        Returns:
            Created suppression override
        """
        try:
            # Verify nurse authentication
            nurse = await self._authenticate_nurse(nurse_id)
            if not nurse.get('can_override_alerts', False):
                raise PermissionError("Nurse lacks override permissions")
            
            # Validate clinical justification
            if len(justification.strip()) < 20:
                raise ValueError("Clinical justification required (min 20 chars)")
            
            override = SuppressionOverride(
                override_id=uuid4(),
                patient_id=patient_id,
                nurse_id=nurse_id,
                justification=justification,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=duration_minutes),
                created_at=datetime.now(timezone.utc),
                is_active=True
            )
            
            # Store override
            await self._save_override(override)
            await self._audit_override_creation(override, nurse)
            
            self.logger.info(f"Created suppression override {override.override_id} for patient {patient_id}")
            return override
            
        except Exception as e:
            self.logger.error(f"Error creating override for patient {patient_id}: {str(e)}")
            raise
    
    async def _authenticate_nurse(self, nurse_id: str) -> Dict[str, Any]:
        """Authenticate nurse and verify permissions."""
        # TODO: Implement actual nurse authentication
        # For now, return mock data for basic validation
        return {
            "nurse_id": nurse_id,
            "can_override_alerts": True,
            "name": f"Nurse {nurse_id}",
            "role": "registered_nurse"
        }
    
    async def _save_override(self, override: SuppressionOverride):
        """Save override to Redis."""
        override_key = f"suppression_override:{override.patient_id}"
        await self.redis.setex(
            override_key,
            override.expires_at - datetime.now(timezone.utc),
            json.dumps(override.to_dict())
        )
    
    async def _audit_override_creation(self, override: SuppressionOverride, nurse: Dict[str, Any]):
        """Create audit trail for override creation."""
        audit_entry = self.audit_logger.create_audit_entry(
            table_name="suppression_overrides",
            operation=AuditOperation.INSERT,
            user_id=override.nurse_id,
            patient_id=override.patient_id,
            new_values=override.to_dict()
        )
        
        self.logger.info(f"Override creation audited: {override.override_id}")
    
    async def review_expired_overrides(self) -> List[SuppressionOverride]:
        """
        Review workflow for expired overrides.
        
        Returns list of overrides that have expired and should be reviewed
        for effectiveness and clinical outcomes.
        """
        try:
            expired_overrides = []
            
            # Scan for override patterns in Redis
            override_keys = []
            async for key in self.redis.scan_iter(match="suppression_override:*"):
                override_keys.append(key)
            
            for key in override_keys:
                try:
                    override_data = await self.redis.get(key)
                    if not override_data:
                        continue
                        
                    override_dict = json.loads(override_data)
                    
                    # Check if override has expired
                    expires_at = datetime.fromisoformat(override_dict['expires_at'].replace('Z', '+00:00'))
                    if expires_at <= datetime.now(timezone.utc):
                        override = SuppressionOverride(
                            override_id=UUID(override_dict['override_id']),
                            patient_id=override_dict['patient_id'],
                            nurse_id=override_dict['nurse_id'],
                            justification=override_dict['justification'],
                            expires_at=expires_at,
                            created_at=datetime.fromisoformat(override_dict['created_at'].replace('Z', '+00:00')),
                            is_active=False
                        )
                        expired_overrides.append(override)
                        
                        # Clean up expired override
                        await self.redis.delete(key)
                        
                except Exception as e:
                    self.logger.error(f"Error processing override key {key}: {str(e)}")
                    continue
            
            if expired_overrides:
                self.logger.info(f"Found {len(expired_overrides)} expired overrides for review")
            
            return expired_overrides
            
        except Exception as e:
            self.logger.error(f"Error reviewing expired overrides: {str(e)}")
            return []


class SuppressionDecisionLogger:
    """
    Comprehensive logging and analytics for suppression decisions.
    
    Tracks all suppression decisions with full reasoning, effectiveness metrics,
    and performance analytics for system optimization.
    """
    
    def __init__(self, redis_client: redis.Redis, audit_logger: AuditLogger):
        self.redis = redis_client
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def log_suppression_decision(self, decision: SuppressionDecision, alert: Alert):
        """
        Log comprehensive suppression decision with full context.
        
        Args:
            decision: The suppression decision made
            alert: The alert that was evaluated
        """
        try:
            # Create comprehensive log entry
            log_entry = {
                "decision_id": str(decision.decision_id),
                "alert_id": str(decision.alert_id),
                "patient_id": decision.patient_id,
                "ward_id": alert.patient.ward_id,
                "suppressed": decision.suppressed,
                "reason": decision.reason,
                "confidence_score": decision.confidence_score,
                "decision_timestamp": decision.decision_timestamp.isoformat(),
                "suppression_expires_at": decision.suppression_expires_at.isoformat() if decision.suppression_expires_at else None,
                "metadata": decision.metadata,
                
                # Alert context
                "alert_level": alert.alert_level.value,
                "alert_priority": alert.alert_priority.value,
                "news2_score": alert.alert_decision.news2_result.total_score,
                "news2_individual_scores": alert.alert_decision.news2_result.individual_scores,
                "single_param_trigger": alert.alert_decision.single_param_trigger,
                "patient_age": alert.patient.age,
                "is_copd_patient": alert.patient.is_copd_patient,
                
                # Performance metrics
                "generation_latency_ms": alert.alert_decision.generation_latency_ms
            }
            
            # Store in time-series for analytics
            log_key = f"suppression_log:{decision.decision_timestamp.strftime('%Y-%m-%d')}"
            await self.redis.zadd(
                log_key,
                {json.dumps(log_entry): decision.decision_timestamp.timestamp()}
            )
            
            # Set expiration for data retention
            await self.redis.expire(log_key, timedelta(days=90))
            
            # Store by patient for patient-specific analytics
            patient_log_key = f"patient_suppression_log:{decision.patient_id}"
            await self.redis.zadd(
                patient_log_key,
                {json.dumps(log_entry): decision.decision_timestamp.timestamp()}
            )
            await self.redis.expire(patient_log_key, timedelta(days=30))
            
            # Update effectiveness metrics
            await self._update_effectiveness_metrics(decision, alert)
            
            self.logger.info(f"Logged suppression decision {decision.decision_id}: {decision.reason}")
            
        except Exception as e:
            self.logger.error(f"Error logging suppression decision: {str(e)}")
    
    async def track_suppression_effectiveness(self, decision_id: UUID, outcome: str, clinical_feedback: Optional[str] = None):
        """
        Track the effectiveness of a suppression decision.
        
        Args:
            decision_id: ID of the suppression decision
            outcome: 'effective', 'false_positive', 'missed_deterioration'
            clinical_feedback: Optional feedback from clinical staff
        """
        try:
            effectiveness_entry = {
                "decision_id": str(decision_id),
                "outcome": outcome,
                "clinical_feedback": clinical_feedback,
                "tracked_at": datetime.now(timezone.utc).isoformat()
            }
            
            effectiveness_key = f"suppression_effectiveness:{decision_id}"
            await self.redis.setex(
                effectiveness_key,
                timedelta(days=30),
                json.dumps(effectiveness_entry)
            )
            
            # Update system-wide effectiveness metrics
            await self._update_system_effectiveness_metrics(outcome)
            
            self.logger.info(f"Tracked effectiveness for decision {decision_id}: {outcome}")
            
        except Exception as e:
            self.logger.error(f"Error tracking effectiveness: {str(e)}")
    
    async def get_suppression_analytics(self, start_date: datetime, end_date: datetime, ward_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive suppression analytics for date range.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            ward_id: Optional ward filter
            
        Returns:
            Dictionary containing suppression analytics
        """
        try:
            analytics = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "ward_id": ward_id
                },
                "total_decisions": 0,
                "suppressed_count": 0,
                "suppression_rate": 0.0,
                "reasons": {},
                "effectiveness": {},
                "confidence_distribution": [],
                "volume_reduction": 0.0,
                "critical_miss_count": 0
            }
            
            # Collect decisions for the period
            current_date = start_date.date()
            end_date_obj = end_date.date()
            
            all_decisions = []
            
            while current_date <= end_date_obj:
                log_key = f"suppression_log:{current_date.strftime('%Y-%m-%d')}"
                
                # Get decisions for this day
                day_decisions = await self.redis.zrangebyscore(
                    log_key,
                    start_date.timestamp(),
                    end_date.timestamp(),
                    withscores=False
                )
                
                for decision_json in day_decisions:
                    decision_data = json.loads(decision_json)
                    
                    # Apply ward filter if specified
                    if ward_id and decision_data.get('ward_id') != ward_id:
                        continue
                    
                    all_decisions.append(decision_data)
                
                current_date += timedelta(days=1)
            
            # Analyze decisions
            analytics["total_decisions"] = len(all_decisions)
            
            if analytics["total_decisions"] > 0:
                suppressed_decisions = [d for d in all_decisions if d.get('suppressed', False)]
                analytics["suppressed_count"] = len(suppressed_decisions)
                analytics["suppression_rate"] = len(suppressed_decisions) / analytics["total_decisions"]
                
                # Analyze reasons
                for decision in all_decisions:
                    reason = decision.get('reason', 'unknown')
                    analytics["reasons"][reason] = analytics["reasons"].get(reason, 0) + 1
                
                # Confidence distribution
                confidence_scores = [d.get('confidence_score', 0.5) for d in all_decisions]
                analytics["confidence_distribution"] = {
                    "mean": sum(confidence_scores) / len(confidence_scores),
                    "min": min(confidence_scores),
                    "max": max(confidence_scores)
                }
                
                # Check for critical misses (should be 0)
                critical_alerts = [d for d in all_decisions if d.get('alert_level') == 'critical']
                critical_suppressed = [d for d in critical_alerts if d.get('suppressed', False)]
                analytics["critical_miss_count"] = len(critical_suppressed)
                
                # Calculate volume reduction
                total_alerts = analytics["total_decisions"]
                suppressed_alerts = analytics["suppressed_count"]
                analytics["volume_reduction"] = (suppressed_alerts / total_alerts) * 100 if total_alerts > 0 else 0
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error generating suppression analytics: {str(e)}")
            return {}
    
    async def _update_effectiveness_metrics(self, decision: SuppressionDecision, alert: Alert):
        """Update real-time effectiveness metrics."""
        try:
            metrics_key = "suppression_metrics:realtime"
            
            # Increment total decisions
            await self.redis.hincrby(metrics_key, "total_decisions", 1)
            
            if decision.suppressed:
                await self.redis.hincrby(metrics_key, "total_suppressed", 1)
            
            # Track by reason
            reason_key = f"suppression_reason:{decision.reason}"
            await self.redis.hincrby(reason_key, "count", 1)
            await self.redis.expire(reason_key, timedelta(days=7))
            
            # Set expiration for metrics
            await self.redis.expire(metrics_key, timedelta(days=7))
            
        except Exception as e:
            self.logger.error(f"Error updating effectiveness metrics: {str(e)}")
    
    async def _update_system_effectiveness_metrics(self, outcome: str):
        """Update system-wide effectiveness tracking."""
        try:
            effectiveness_key = "suppression_effectiveness:system"
            await self.redis.hincrby(effectiveness_key, f"outcome_{outcome}", 1)
            await self.redis.expire(effectiveness_key, timedelta(days=30))
            
        except Exception as e:
            self.logger.error(f"Error updating system effectiveness metrics: {str(e)}")
    
    async def query_suppression_decisions(
        self,
        patient_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        reason: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query suppression decisions with flexible filtering.
        
        Args:
            patient_id: Filter by patient ID
            start_time: Filter by start time
            end_time: Filter by end time
            reason: Filter by suppression reason
            limit: Maximum number of results
            
        Returns:
            List of matching suppression decisions
        """
        try:
            results = []
            
            if patient_id:
                # Query patient-specific log
                patient_log_key = f"patient_suppression_log:{patient_id}"
                
                if start_time and end_time:
                    decision_data = await self.redis.zrangebyscore(
                        patient_log_key,
                        start_time.timestamp(),
                        end_time.timestamp(),
                        withscores=False
                    )
                else:
                    decision_data = await self.redis.zrevrange(
                        patient_log_key,
                        0,
                        limit - 1,
                        withscores=False
                    )
                
                for decision_json in decision_data:
                    decision = json.loads(decision_json)
                    
                    # Apply reason filter if specified
                    if reason and decision.get('reason') != reason:
                        continue
                    
                    results.append(decision)
            
            else:
                # Query system-wide logs
                if not start_time:
                    start_time = datetime.now(timezone.utc) - timedelta(days=7)
                if not end_time:
                    end_time = datetime.now(timezone.utc)
                
                current_date = start_time.date()
                end_date_obj = end_time.date()
                
                while current_date <= end_date_obj and len(results) < limit:
                    log_key = f"suppression_log:{current_date.strftime('%Y-%m-%d')}"
                    
                    day_decisions = await self.redis.zrangebyscore(
                        log_key,
                        start_time.timestamp(),
                        end_time.timestamp(),
                        withscores=False
                    )
                    
                    for decision_json in day_decisions:
                        if len(results) >= limit:
                            break
                        
                        decision = json.loads(decision_json)
                        
                        # Apply filters
                        if reason and decision.get('reason') != reason:
                            continue
                        
                        results.append(decision)
                    
                    current_date += timedelta(days=1)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error querying suppression decisions: {str(e)}")
            return []


class SuppressionMetrics:
    """
    Prometheus metrics collection for alert suppression system.
    
    Tracks suppression effectiveness, performance, and volume reduction
    for system monitoring and optimization.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Setup Prometheus metrics
        self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics for suppression system."""
        
        # Suppression decision counters
        self.suppression_decisions_total = Counter(
            'alert_suppression_decisions_total',
            'Total number of suppression decisions made',
            ['reason', 'ward_id', 'alert_level'],
            registry=self.registry
        )
        
        self.alerts_suppressed_total = Counter(
            'alerts_suppressed_total',
            'Total number of alerts suppressed',
            ['reason', 'ward_id', 'alert_level'],
            registry=self.registry
        )
        
        self.alerts_not_suppressed_total = Counter(
            'alerts_not_suppressed_total',
            'Total number of alerts not suppressed',
            ['reason', 'ward_id', 'alert_level'],
            registry=self.registry
        )
        
        # Volume reduction metrics
        self.alert_volume_reduction_percent = Gauge(
            'alert_volume_reduction_percent',
            'Percentage of alert volume reduced by suppression',
            ['ward_id'],
            registry=self.registry
        )
        
        # Critical safety metrics
        self.critical_alerts_suppressed_error = Counter(
            'critical_alerts_suppressed_error_total',
            'Critical alerts incorrectly suppressed (should be 0)',
            ['ward_id'],
            registry=self.registry
        )
        
        # Effectiveness metrics
        self.suppression_effectiveness = Counter(
            'suppression_effectiveness_total',
            'Suppression effectiveness outcomes',
            ['outcome', 'reason'],  # outcome: effective, false_positive, missed_deterioration
            registry=self.registry
        )
        
        # Performance metrics
        self.suppression_decision_duration_seconds = Histogram(
            'suppression_decision_duration_seconds',
            'Time taken to make suppression decisions',
            ['reason'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        # Confidence score distribution
        self.suppression_confidence_score = Histogram(
            'suppression_confidence_score',
            'Distribution of suppression confidence scores',
            ['reason'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Manual override metrics
        self.manual_overrides_created_total = Counter(
            'manual_overrides_created_total',
            'Total manual suppression overrides created',
            ['nurse_id', 'ward_id'],
            registry=self.registry
        )
        
        self.manual_overrides_expired_total = Counter(
            'manual_overrides_expired_total',
            'Total manual suppression overrides expired',
            ['ward_id'],
            registry=self.registry
        )
        
        # Pattern detection metrics
        self.pattern_detections_total = Counter(
            'pattern_detections_total',
            'Total pattern-based suppressions',
            ['pattern_type', 'ward_id'],
            registry=self.registry
        )
        
        # Real-time gauges
        self.active_suppressions = Gauge(
            'active_suppressions_current',
            'Current number of active suppressions',
            ['ward_id'],
            registry=self.registry
        )
        
        self.false_positive_rate = Gauge(
            'suppression_false_positive_rate',
            'Current false positive rate for suppressions',
            ['ward_id'],
            registry=self.registry
        )
    
    def track_suppression_decision(self, decision: SuppressionDecision, alert: Alert, duration_seconds: float):
        """
        Track a suppression decision with comprehensive metrics.
        
        Args:
            decision: The suppression decision made
            alert: The alert that was evaluated
            duration_seconds: Time taken to make the decision
        """
        try:
            ward_id = alert.patient.ward_id
            alert_level = alert.alert_level.value
            reason = decision.reason
            
            # Track decision counter
            self.suppression_decisions_total.labels(
                reason=reason,
                ward_id=ward_id,
                alert_level=alert_level
            ).inc()
            
            # Track suppression outcome
            if decision.suppressed:
                self.alerts_suppressed_total.labels(
                    reason=reason,
                    ward_id=ward_id,
                    alert_level=alert_level
                ).inc()
                
                # CRITICAL SAFETY CHECK: Should never suppress critical alerts
                if alert.alert_level == AlertLevel.CRITICAL:
                    self.critical_alerts_suppressed_error.labels(ward_id=ward_id).inc()
                    self.logger.critical(f"CRITICAL SAFETY VIOLATION: Critical alert {alert.alert_id} was suppressed!")
            else:
                self.alerts_not_suppressed_total.labels(
                    reason=reason,
                    ward_id=ward_id,
                    alert_level=alert_level
                ).inc()
            
            # Track performance
            self.suppression_decision_duration_seconds.labels(reason=reason).observe(duration_seconds)
            
            # Track confidence distribution
            self.suppression_confidence_score.labels(reason=reason).observe(decision.confidence_score)
            
            # Track pattern-based suppressions
            if 'PATTERN' in reason:
                pattern_type = reason.lower().replace('_suppression', '')
                self.pattern_detections_total.labels(
                    pattern_type=pattern_type,
                    ward_id=ward_id
                ).inc()
            
        except Exception as e:
            self.logger.error(f"Error tracking suppression decision metrics: {str(e)}")
    
    def track_manual_override(self, override: SuppressionOverride):
        """Track manual override creation."""
        try:
            self.manual_overrides_created_total.labels(
                nurse_id=override.nurse_id,
                ward_id="unknown"  # TODO: Get ward from patient context
            ).inc()
            
        except Exception as e:
            self.logger.error(f"Error tracking manual override metrics: {str(e)}")
    
    def track_override_expiration(self, override: SuppressionOverride, ward_id: str):
        """Track manual override expiration."""
        try:
            self.manual_overrides_expired_total.labels(ward_id=ward_id).inc()
            
        except Exception as e:
            self.logger.error(f"Error tracking override expiration metrics: {str(e)}")
    
    def track_suppression_effectiveness(self, outcome: str, reason: str):
        """Track the effectiveness of suppression decisions."""
        try:
            self.suppression_effectiveness.labels(
                outcome=outcome,
                reason=reason
            ).inc()
            
        except Exception as e:
            self.logger.error(f"Error tracking suppression effectiveness: {str(e)}")
    
    def update_volume_reduction_metrics(self, ward_id: str, reduction_percentage: float):
        """Update real-time volume reduction metrics."""
        try:
            self.alert_volume_reduction_percent.labels(ward_id=ward_id).set(reduction_percentage)
            
        except Exception as e:
            self.logger.error(f"Error updating volume reduction metrics: {str(e)}")
    
    def update_false_positive_rate(self, ward_id: str, fp_rate: float):
        """Update real-time false positive rate."""
        try:
            self.false_positive_rate.labels(ward_id=ward_id).set(fp_rate)
            
        except Exception as e:
            self.logger.error(f"Error updating false positive rate: {str(e)}")
    
    def update_active_suppressions_count(self, ward_id: str, count: int):
        """Update count of currently active suppressions."""
        try:
            self.active_suppressions.labels(ward_id=ward_id).set(count)
            
        except Exception as e:
            self.logger.error(f"Error updating active suppressions count: {str(e)}")
    
    def generate_suppression_performance_alert(self, ward_id: str, metric: str, value: float, threshold: float):
        """Generate performance alert when metrics exceed thresholds."""
        try:
            if metric == "critical_suppression" and value > 0:
                self.logger.critical(
                    f"CRITICAL SAFETY ALERT: {value} critical alerts suppressed in ward {ward_id}. "
                    f"This should NEVER happen!"
                )
            elif metric == "false_positive_rate" and value > threshold:
                self.logger.warning(
                    f"High false positive rate in ward {ward_id}: {value:.2%} (threshold: {threshold:.2%})"
                )
            elif metric == "volume_reduction" and value < threshold:
                self.logger.warning(
                    f"Low volume reduction in ward {ward_id}: {value:.2%} (target: {threshold:.2%})"
                )
                
        except Exception as e:
            self.logger.error(f"Error generating performance alert: {str(e)}")