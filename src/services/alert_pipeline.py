"""
Alert Processing Pipeline for NEWS2 Live System

This module implements the complete alert processing pipeline including:
- Kafka consumer for NEWS2 score events
- Async alert evaluation workflow
- Alert deduplication and correlation logic
- Alert enrichment with patient context
- Alert publishing to delivery system
- Escalation scheduling for unacknowledged alerts
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from ..models.alerts import (
    Alert, AlertLevel, AlertStatus, AlertDecision, AlertThreshold,
    AlertGenerationException
)
from ..models.news2 import NEWS2Result
from ..models.patient import Patient
from ..services.alert_generation import AlertGenerator, AlertDecisionEngine
from ..services.alert_configuration import AlertConfigurationService
from ..services.escalation_engine import EscalationEngine
from ..services.alert_auditing import AlertAuditingService
from ..services.audit import AuditLogger


class PipelineStatus(Enum):
    """Status of alert processing pipeline."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AlertPipelineMetrics:
    """Metrics for alert processing pipeline."""
    events_processed: int
    alerts_generated: int
    alerts_suppressed: int
    escalations_scheduled: int
    processing_errors: int
    average_processing_time_ms: float
    pipeline_uptime_seconds: float
    last_processed_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring."""
        return {
            "events_processed": self.events_processed,
            "alerts_generated": self.alerts_generated,
            "alerts_suppressed": self.alerts_suppressed,
            "escalations_scheduled": self.escalations_scheduled,
            "processing_errors": self.processing_errors,
            "average_processing_time_ms": self.average_processing_time_ms,
            "pipeline_uptime_seconds": self.pipeline_uptime_seconds,
            "last_processed_timestamp": self.last_processed_timestamp.isoformat()
        }


@dataclass
class AlertDeduplicationEntry:
    """Entry for alert deduplication tracking."""
    patient_id: str
    alert_level: AlertLevel
    news2_score: int
    dedup_key: str
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    suppressed: bool
    
    def should_suppress(self, suppression_window_minutes: int = 30) -> bool:
        """Check if new alert should be suppressed based on recent occurrence."""
        time_since_last = datetime.now(timezone.utc) - self.last_occurrence
        return time_since_last.total_seconds() < (suppression_window_minutes * 60)


class AlertDeduplicationManager:
    """
    Manages alert deduplication to prevent alert fatigue.
    
    Responsibilities:
    - Track recent alerts for deduplication
    - Generate deduplication keys
    - Apply suppression rules
    - Clean up expired deduplication entries
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # In-memory storage for deduplication (replace with Redis in production)
        self._dedup_entries: Dict[str, AlertDeduplicationEntry] = {}
        self._cleanup_interval = 3600  # Clean up every hour
        self._last_cleanup = datetime.now(timezone.utc)
    
    def generate_deduplication_key(
        self,
        patient_id: str,
        alert_level: AlertLevel,
        news2_score: int,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deduplication key for alert.
        
        Args:
            patient_id: Patient identifier
            alert_level: Alert level
            news2_score: NEWS2 score
            additional_context: Optional additional context for key generation
            
        Returns:
            Deduplication key string
        """
        # Create base key components
        key_components = [
            patient_id,
            alert_level.value,
            str(news2_score)
        ]
        
        # Add additional context if provided
        if additional_context:
            sorted_context = sorted(additional_context.items())
            context_str = json.dumps(sorted_context, sort_keys=True)
            key_components.append(context_str)
        
        # Generate hash-based key
        key_string = "|".join(key_components)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{patient_id}:{alert_level.value}:{key_hash}"
    
    async def check_and_update_deduplication(
        self,
        patient_id: str,
        alert_level: AlertLevel,
        news2_score: int,
        additional_context: Optional[Dict[str, Any]] = None,
        suppression_window_minutes: int = 30
    ) -> Tuple[bool, str]:
        """
        Check if alert should be suppressed and update deduplication tracking.
        
        Args:
            patient_id: Patient identifier
            alert_level: Alert level
            news2_score: NEWS2 score
            additional_context: Optional additional context
            suppression_window_minutes: Suppression window in minutes
            
        Returns:
            Tuple of (should_suppress, deduplication_key)
        """
        try:
            # Generate deduplication key
            dedup_key = self.generate_deduplication_key(
                patient_id, alert_level, news2_score, additional_context
            )
            
            current_time = datetime.now(timezone.utc)
            
            # Check existing entry
            existing_entry = self._dedup_entries.get(dedup_key)
            
            if existing_entry:
                # Update existing entry
                existing_entry.last_occurrence = current_time
                existing_entry.occurrence_count += 1
                
                # Check if should suppress (never suppress CRITICAL alerts)
                should_suppress = (
                    alert_level != AlertLevel.CRITICAL and
                    existing_entry.should_suppress(suppression_window_minutes)
                )
                
                existing_entry.suppressed = should_suppress
                
                self.logger.debug(
                    f"Updated deduplication entry for {dedup_key}: "
                    f"count={existing_entry.occurrence_count}, suppress={should_suppress}"
                )
                
                return should_suppress, dedup_key
            else:
                # Create new entry
                new_entry = AlertDeduplicationEntry(
                    patient_id=patient_id,
                    alert_level=alert_level,
                    news2_score=news2_score,
                    dedup_key=dedup_key,
                    first_occurrence=current_time,
                    last_occurrence=current_time,
                    occurrence_count=1,
                    suppressed=False
                )
                
                self._dedup_entries[dedup_key] = new_entry
                
                self.logger.debug(f"Created new deduplication entry for {dedup_key}")
                
                # First occurrence is never suppressed
                return False, dedup_key
            
        except Exception as e:
            self.logger.error(f"Error in deduplication check: {str(e)}")
            # On error, don't suppress to ensure alerts are not lost
            return False, "error_fallback"
        finally:
            # Periodic cleanup
            await self._cleanup_expired_entries()
    
    async def _cleanup_expired_entries(self, max_age_hours: int = 24):
        """Clean up expired deduplication entries."""
        current_time = datetime.now(timezone.utc)
        
        # Only clean up periodically
        if (current_time - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return
        
        try:
            cutoff_time = current_time - timedelta(hours=max_age_hours)
            expired_keys = []
            
            for key, entry in self._dedup_entries.items():
                if entry.last_occurrence < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._dedup_entries[key]
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired deduplication entries")
            
            self._last_cleanup = current_time
            
        except Exception as e:
            self.logger.error(f"Error cleaning up deduplication entries: {str(e)}")
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        total_entries = len(self._dedup_entries)
        suppressed_entries = len([e for e in self._dedup_entries.values() if e.suppressed])
        total_occurrences = sum(e.occurrence_count for e in self._dedup_entries.values())
        
        return {
            "total_entries": total_entries,
            "suppressed_entries": suppressed_entries,
            "total_occurrences": total_occurrences,
            "suppression_rate": (suppressed_entries / max(total_entries, 1)) * 100,
            "last_cleanup": self._last_cleanup.isoformat()
        }


class AlertEnrichmentService:
    """
    Enriches alerts with additional patient and clinical context.
    
    Responsibilities:
    - Add patient demographic information
    - Include relevant medical history
    - Add ward-specific context
    - Include trending information
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def enrich_alert(
        self,
        alert: Alert,
        patient: Patient,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Enrich alert with additional context information.
        
        Args:
            alert: Alert to enrich
            patient: Patient information
            additional_context: Optional additional context
            
        Returns:
            Enriched alert
        """
        try:
            # Start with existing clinical context
            enriched_context = alert.clinical_context.copy()
            
            # Add patient demographics
            enriched_context.update({
                "patient_demographics": {
                    "age": patient.age,
                    "age_category": self._categorize_age(patient.age),
                    "ward_id": patient.ward_id,
                    "bed_number": patient.bed_number,
                    "admission_date": patient.admission_date.isoformat() if patient.admission_date else None
                }
            })
            
            # Add medical flags
            enriched_context.update({
                "medical_flags": {
                    "is_copd_patient": patient.is_copd_patient,
                    "requires_special_monitoring": self._requires_special_monitoring(patient),
                    "high_risk_patient": self._is_high_risk_patient(patient, alert)
                }
            })
            
            # Add alert timing context
            enriched_context.update({
                "timing_context": {
                    "alert_created_at": alert.created_at.isoformat(),
                    "time_of_day": alert.created_at.hour,
                    "day_of_week": alert.created_at.weekday(),
                    "is_night_shift": self._is_night_shift(alert.created_at),
                    "is_weekend": alert.created_at.weekday() >= 5
                }
            })
            
            # Add NEWS2 context
            enriched_context.update({
                "news2_context": {
                    "score_trend": await self._get_score_trend(patient.patient_id),
                    "scale_explanation": self._explain_scale_used(alert.alert_decision.news2_result.scale_used, patient),
                    "parameter_analysis": self._analyze_individual_parameters(alert.alert_decision.news2_result.individual_scores)
                }
            })
            
            # Add any additional context provided
            if additional_context:
                enriched_context.update({"additional_context": additional_context})
            
            # Update alert with enriched context
            alert.clinical_context = enriched_context
            
            self.logger.debug(f"Enriched alert {alert.alert_id} with additional context")
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Failed to enrich alert {alert.alert_id}: {str(e)}")
            # Return original alert if enrichment fails
            return alert
    
    def _categorize_age(self, age: int) -> str:
        """Categorize patient age for clinical context."""
        if age < 18:
            return "pediatric"
        elif age < 65:
            return "adult"
        elif age < 80:
            return "elderly"
        else:
            return "very_elderly"
    
    def _requires_special_monitoring(self, patient: Patient) -> bool:
        """Determine if patient requires special monitoring."""
        # Check various risk factors
        risk_factors = [
            patient.is_copd_patient,
            patient.age >= 80,  # Very elderly
            # Add other risk factors as needed
        ]
        return any(risk_factors)
    
    def _is_high_risk_patient(self, patient: Patient, alert: Alert) -> bool:
        """Determine if patient is high risk."""
        risk_factors = [
            patient.age >= 75,
            patient.is_copd_patient,
            alert.alert_decision.news2_result.total_score >= 7,
            alert.alert_decision.single_param_trigger
        ]
        return sum(risk_factors) >= 2  # High risk if 2+ risk factors
    
    def _is_night_shift(self, timestamp: datetime) -> bool:
        """Check if timestamp falls during night shift (22:00 - 06:00)."""
        hour = timestamp.hour
        return hour >= 22 or hour < 6
    
    async def _get_score_trend(self, patient_id: str) -> str:
        """Get NEWS2 score trend for patient (mock implementation)."""
        # In production, this would query recent NEWS2 scores
        # For now, return mock trend
        trends = ["improving", "stable", "deteriorating", "unknown"]
        import random
        return random.choice(trends)
    
    def _explain_scale_used(self, scale_used: int, patient: Patient) -> str:
        """Explain which NEWS2 scale was used and why."""
        if scale_used == 2:
            return f"Scale 2 (COPD) used because patient is flagged as COPD patient"
        else:
            return f"Scale 1 (Standard) used for general patient population"
    
    def _analyze_individual_parameters(self, individual_scores: Dict[str, int]) -> Dict[str, str]:
        """Analyze individual parameter scores for clinical insights."""
        analysis = {}
        
        for param, score in individual_scores.items():
            if score == 3:
                analysis[param] = "CRITICAL - Immediate attention required"
            elif score == 2:
                analysis[param] = "HIGH - Concerning value"
            elif score == 1:
                analysis[param] = "MODERATE - Monitor closely"
            else:
                analysis[param] = "NORMAL - Within acceptable range"
        
        return analysis


class AlertProcessingPipeline:
    """
    Main alert processing pipeline that coordinates all alert generation activities.
    
    Responsibilities:
    - Consume NEWS2 events from Kafka
    - Process alerts through generation pipeline
    - Handle deduplication and enrichment
    - Publish alerts to delivery system
    - Schedule escalations
    - Monitor pipeline health and metrics
    """
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        alert_generator: AlertGenerator,
        escalation_engine: EscalationEngine,
        auditing_service: AlertAuditingService,
        config_service: AlertConfigurationService
    ):
        self.audit_logger = audit_logger
        self.alert_generator = alert_generator
        self.escalation_engine = escalation_engine
        self.auditing_service = auditing_service
        self.config_service = config_service
        
        self.deduplication_manager = AlertDeduplicationManager()
        self.enrichment_service = AlertEnrichmentService()
        
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self._status = PipelineStatus.STOPPED
        self._processing_task: Optional[asyncio.Task] = None
        self._metrics = AlertPipelineMetrics(
            events_processed=0,
            alerts_generated=0,
            alerts_suppressed=0,
            escalations_scheduled=0,
            processing_errors=0,
            average_processing_time_ms=0.0,
            pipeline_uptime_seconds=0.0,
            last_processed_timestamp=datetime.now(timezone.utc)
        )
        self._start_time: Optional[datetime] = None
        
        # Mock event queue (replace with actual Kafka consumer)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
        # Alert delivery callback
        self._alert_delivery_callback: Optional[Callable] = None
    
    def set_alert_delivery_callback(self, callback: Callable):
        """Set callback for alert delivery."""
        self._alert_delivery_callback = callback
    
    async def start_pipeline(self):
        """Start the alert processing pipeline."""
        if self._status != PipelineStatus.STOPPED:
            self.logger.warning("Pipeline is already running or starting")
            return
        
        try:
            self._status = PipelineStatus.STARTING
            self.logger.info("Starting alert processing pipeline")
            
            # Start escalation engine
            await self.escalation_engine.start_escalation_processing()
            
            # Start processing task
            self._start_time = datetime.now(timezone.utc)
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            self._status = PipelineStatus.RUNNING
            self.logger.info("Alert processing pipeline started successfully")
            
        except Exception as e:
            self._status = PipelineStatus.ERROR
            self.logger.error(f"Failed to start alert processing pipeline: {str(e)}")
            raise
    
    async def stop_pipeline(self):
        """Stop the alert processing pipeline."""
        if self._status in [PipelineStatus.STOPPED, PipelineStatus.STOPPING]:
            return
        
        try:
            self._status = PipelineStatus.STOPPING
            self.logger.info("Stopping alert processing pipeline")
            
            # Stop processing task
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            # Stop escalation engine
            await self.escalation_engine.stop_escalation_processing()
            
            self._status = PipelineStatus.STOPPED
            self.logger.info("Alert processing pipeline stopped")
            
        except Exception as e:
            self._status = PipelineStatus.ERROR
            self.logger.error(f"Error stopping alert processing pipeline: {str(e)}")
    
    async def process_news2_event(
        self,
        news2_result: NEWS2Result,
        patient: Patient,
        user_id: str = "SYSTEM"
    ) -> Optional[Alert]:
        """
        Process a single NEWS2 event through the alert pipeline.
        
        Args:
            news2_result: NEWS2 calculation result
            patient: Patient information
            user_id: ID of user/system triggering processing
            
        Returns:
            Generated alert or None if no alert was created
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(
                f"Processing NEWS2 event for patient {patient.patient_id}: "
                f"score={news2_result.total_score}, level={news2_result.risk_category.value}"
            )
            
            # Get ward-specific thresholds
            ward_thresholds = await self.config_service.threshold_manager.get_ward_thresholds(
                patient.ward_id, active_only=True
            )
            
            # Check deduplication
            should_suppress, dedup_key = await self.deduplication_manager.check_and_update_deduplication(
                patient.patient_id,
                self._determine_alert_level_from_news2(news2_result),
                news2_result.total_score,
                {"ward_id": patient.ward_id}
            )
            
            # Generate alert decision
            alert = await self.alert_generator.generate_alert(
                news2_result, patient, ward_thresholds, user_id
            )
            
            if not alert:
                self.logger.debug(f"No alert generated for patient {patient.patient_id}")
                self._metrics.events_processed += 1
                return None
            
            # Apply deduplication suppression
            if should_suppress and alert.alert_level != AlertLevel.CRITICAL:
                self.logger.info(
                    f"Suppressing duplicate alert for patient {patient.patient_id} "
                    f"(dedup_key: {dedup_key})"
                )
                self._metrics.alerts_suppressed += 1
                self._metrics.events_processed += 1
                return None
            
            # Enrich alert with additional context
            enriched_alert = await self.enrichment_service.enrich_alert(
                alert, patient, {"dedup_key": dedup_key}
            )
            
            # Audit alert decision
            await self.auditing_service.decision_auditor.audit_alert_decision(
                enriched_alert.alert_decision,
                patient,
                enriched_alert.alert_decision.threshold_applied,
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                user_id
            )
            
            # Schedule escalation
            escalation_schedule = await self.escalation_engine.create_alert_escalation(enriched_alert)
            if escalation_schedule:
                self.logger.info(f"Scheduled escalation for alert {enriched_alert.alert_id}")
                self._metrics.escalations_scheduled += 1
            
            # Deliver alert
            if self._alert_delivery_callback:
                try:
                    await self._alert_delivery_callback(enriched_alert)
                    self.logger.info(f"Delivered alert {enriched_alert.alert_id}")
                except Exception as e:
                    self.logger.error(f"Failed to deliver alert {enriched_alert.alert_id}: {str(e)}")
            
            # Update metrics
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_metrics(processing_time_ms, success=True)
            
            self.logger.info(
                f"Successfully processed NEWS2 event for patient {patient.patient_id}: "
                f"alert_id={enriched_alert.alert_id}, level={enriched_alert.alert_level.value}, "
                f"processing_time={processing_time_ms:.1f}ms"
            )
            
            return enriched_alert
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_metrics(processing_time_ms, success=False)
            
            self.logger.error(
                f"Failed to process NEWS2 event for patient {patient.patient_id}: {str(e)}"
            )
            raise AlertGenerationException(f"Alert processing failed: {str(e)}")
    
    async def _processing_loop(self):
        """Main processing loop for consuming events."""
        self.logger.info("Starting alert processing loop")
        
        while self._status == PipelineStatus.RUNNING:
            try:
                # Mock event consumption (replace with actual Kafka consumer)
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    
                    # Process event
                    await self._process_event(event)
                    
                except asyncio.TimeoutError:
                    # No events available, continue loop
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                self._metrics.processing_errors += 1
                await asyncio.sleep(5)  # Brief pause on error
        
        self.logger.info("Alert processing loop stopped")
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process a single event from the queue."""
        try:
            # Extract NEWS2 result and patient from event
            news2_data = event.get("news2_result")
            patient_data = event.get("patient")
            user_id = event.get("user_id", "SYSTEM")
            
            if not news2_data or not patient_data:
                self.logger.warning("Received malformed event, skipping")
                return
            
            # Reconstruct objects (in production, these would be properly serialized)
            news2_result = self._reconstruct_news2_result(news2_data)
            patient = self._reconstruct_patient(patient_data)
            
            # Process through pipeline
            await self.process_news2_event(news2_result, patient, user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to process event: {str(e)}")
            self._metrics.processing_errors += 1
    
    def _determine_alert_level_from_news2(self, news2_result: NEWS2Result) -> AlertLevel:
        """Determine alert level from NEWS2 result for deduplication."""
        if news2_result.total_score >= 7:
            return AlertLevel.CRITICAL
        elif news2_result.total_score >= 5:
            return AlertLevel.HIGH
        elif news2_result.total_score >= 3:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _update_metrics(self, processing_time_ms: float, success: bool):
        """Update pipeline metrics."""
        self._metrics.events_processed += 1
        if success:
            self._metrics.alerts_generated += 1
        else:
            self._metrics.processing_errors += 1
        
        # Update average processing time
        total_time = (self._metrics.average_processing_time_ms * (self._metrics.events_processed - 1) +
                     processing_time_ms)
        self._metrics.average_processing_time_ms = total_time / self._metrics.events_processed
        
        # Update uptime
        if self._start_time:
            self._metrics.pipeline_uptime_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()
        
        self._metrics.last_processed_timestamp = datetime.now(timezone.utc)
    
    def _reconstruct_news2_result(self, data: Dict[str, Any]) -> NEWS2Result:
        """Reconstruct NEWS2Result from serialized data."""
        from ..models.news2 import RiskCategory
        
        return NEWS2Result(
            total_score=data["total_score"],
            individual_scores=data["individual_scores"],
            risk_category=RiskCategory(data["risk_category"]),
            monitoring_frequency=data["monitoring_frequency"],
            scale_used=data["scale_used"],
            warnings=data.get("warnings", []),
            confidence=data.get("confidence", 1.0),
            calculated_at=datetime.fromisoformat(data["calculated_at"]),
            calculation_time_ms=data.get("calculation_time_ms", 0.0)
        )
    
    def _reconstruct_patient(self, data: Dict[str, Any]) -> Patient:
        """Reconstruct Patient from serialized data."""
        return Patient(
            patient_id=data["patient_id"],
            ward_id=data["ward_id"],
            bed_number=data.get("bed_number"),
            age=data["age"],
            is_copd_patient=data.get("is_copd_patient", False),
            assigned_nurse_id=data.get("assigned_nurse_id"),
            admission_date=datetime.fromisoformat(data["admission_date"]) if data.get("admission_date") else None,
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
    
    async def add_mock_event(self, news2_result: NEWS2Result, patient: Patient, user_id: str = "SYSTEM"):
        """Add mock event to processing queue (for testing)."""
        event = {
            "news2_result": {
                "total_score": news2_result.total_score,
                "individual_scores": news2_result.individual_scores,
                "risk_category": news2_result.risk_category.value,
                "monitoring_frequency": news2_result.monitoring_frequency,
                "scale_used": news2_result.scale_used,
                "warnings": news2_result.warnings,
                "confidence": news2_result.confidence,
                "calculated_at": news2_result.calculated_at.isoformat(),
                "calculation_time_ms": news2_result.calculation_time_ms
            },
            "patient": {
                "patient_id": patient.patient_id,
                "ward_id": patient.ward_id,
                "bed_number": patient.bed_number,
                "age": patient.age,
                "is_copd_patient": patient.is_copd_patient,
                "assigned_nurse_id": patient.assigned_nurse_id,
                "admission_date": patient.admission_date.isoformat() if patient.admission_date else None,
                "last_updated": patient.last_updated.isoformat()
            },
            "user_id": user_id
        }
        
        await self._event_queue.put(event)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "status": self._status.value,
            "metrics": self._metrics.to_dict(),
            "deduplication_stats": self.deduplication_manager.get_deduplication_stats(),
            "queue_size": self._event_queue.qsize(),
            "start_time": self._start_time.isoformat() if self._start_time else None
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information for monitoring."""
        is_healthy = (
            self._status == PipelineStatus.RUNNING and
            self._metrics.processing_errors < self._metrics.events_processed * 0.1  # < 10% error rate
        )
        
        return {
            "healthy": is_healthy,
            "status": self._status.value,
            "uptime_seconds": self._metrics.pipeline_uptime_seconds,
            "events_processed": self._metrics.events_processed,
            "error_rate": (self._metrics.processing_errors / max(self._metrics.events_processed, 1)) * 100,
            "average_processing_time_ms": self._metrics.average_processing_time_ms,
            "last_processed": self._metrics.last_processed_timestamp.isoformat()
        }