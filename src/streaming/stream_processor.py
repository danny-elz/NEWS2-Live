"""
Faust-based stream processor for real-time NEWS2 calculation.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import faust
from faust import StreamT
import redis.asyncio as redis

from ..models.vital_signs import VitalSigns, ConsciousnessLevel
from ..models.news2 import NEWS2Result, RiskCategory
from ..models.patient import Patient
from ..models.patient_state import PatientState, PatientStateError
from ..services.news2_calculator import NEWS2Calculator
from ..services.patient_registry import PatientRegistry
from ..services.patient_state_tracker import PatientStateTracker
from ..services.audit import AuditLogger, AuditOperation
from .idempotency_manager import IdempotencyManager
from .error_handler import StreamErrorHandler

logger = logging.getLogger(__name__)


class VitalSignsRecord(faust.Record, serializer='json'):
    """Faust record for vital signs input."""
    event_id: str
    patient_id: str
    timestamp: str  # ISO format timestamp
    respiratory_rate: Optional[int]
    spo2: Optional[int] 
    on_oxygen: Optional[bool]
    temperature: Optional[float]
    systolic_bp: Optional[int]
    heart_rate: Optional[int]
    consciousness: str  # ConsciousnessLevel enum value
    data_source: str = "device"
    quality_flags: Dict[str, Any] = {}


class NEWS2ResultRecord(faust.Record, serializer='json'):
    """Faust record for NEWS2 calculation results."""
    event_id: str
    patient_id: str
    calculation_timestamp: str
    vital_signs_timestamp: str
    news2_score: int
    individual_scores: Dict[str, int]
    risk_category: str
    scale_used: int
    monitoring_frequency: str
    red_flags: List[str]
    trending_data: Optional[Dict[str, Any]]
    processing_latency_ms: float


class StreamProcessor:
    """Main stream processing application for NEWS2 calculation."""
    
    def __init__(self, 
                 broker_url: str = 'kafka://localhost:9092',
                 redis_url: str = 'redis://localhost:6379',
                 app_id: str = 'news2_stream_processor'):
        
        self.app_id = app_id
        self.broker_url = broker_url
        self.redis_url = redis_url
        
        # Initialize Faust app
        self.app = faust.App(
            app_id,
            broker=broker_url,
            value_serializer='json',
            stream_wait_empty=False,
            broker_commit_every=1,  # Commit frequently for exactly-once semantics
        )
        
        # Define topics
        self.vital_signs_topic = self.app.topic(
            'vital_signs_input', 
            value_type=VitalSignsRecord,
            partitions=10
        )
        
        self.news2_results_topic = self.app.topic(
            'news2_results',
            value_type=NEWS2ResultRecord,
            partitions=10
        )
        
        self.alerts_topic = self.app.topic(
            'news2_alerts',
            value_type=NEWS2ResultRecord,
            partitions=5
        )
        
        self.dead_letter_topic = self.app.topic(
            'dead_letter_queue',
            partitions=3
        )
        
        # Initialize services
        self._redis_client = None
        self._idempotency_manager = None
        self._error_handler = None
        self._news2_calculator = None
        self._patient_registry = None
        self._state_tracker = None
        self._audit_logger = None
        
        # Processing metrics
        self._processed_count = 0
        self._error_count = 0
        self._start_time = datetime.now(timezone.utc)
    
    async def initialize_services(self) -> None:
        """Initialize all required services."""
        try:
            # Initialize Redis client
            self._redis_client = redis.from_url(self.redis_url)
            await self._redis_client.ping()
            
            # Initialize managers
            self._idempotency_manager = IdempotencyManager(self._redis_client)
            self._error_handler = StreamErrorHandler(
                dead_letter_topic=self.dead_letter_topic,
                max_retries=3
            )
            
            # Initialize NEWS2 services
            self._audit_logger = AuditLogger()
            self._news2_calculator = NEWS2Calculator(self._audit_logger)
            self._patient_registry = PatientRegistry(self._audit_logger)
            self._state_tracker = PatientStateTracker(self._audit_logger)
            
            logger.info("Stream processor services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    @property
    def vital_signs_processor(self):
        """Main stream processing agent."""
        @self.app.agent(self.vital_signs_topic)
        async def process_vital_signs(stream: StreamT[VitalSignsRecord]) -> None:
            async for vital_signs_record in stream:
                processing_start = datetime.now(timezone.utc)
                
                try:
                    # Check for duplicate processing
                    is_duplicate = await self._idempotency_manager.is_duplicate(
                        vital_signs_record.event_id,
                        vital_signs_record.patient_id
                    )
                    
                    if is_duplicate:
                        logger.debug(f"Skipping duplicate event: {vital_signs_record.event_id}")
                        continue
                    
                    # Convert to internal vital signs model
                    vital_signs = await self._convert_to_vital_signs(vital_signs_record)
                    
                    # Calculate NEWS2 score
                    news2_result = await self._calculate_news2_with_context(vital_signs)
                    
                    # Calculate processing latency
                    processing_end = datetime.now(timezone.utc)
                    latency_ms = (processing_end - processing_start).total_seconds() * 1000
                    
                    # Create result record
                    result_record = NEWS2ResultRecord(
                        event_id=vital_signs_record.event_id,
                        patient_id=vital_signs_record.patient_id,
                        calculation_timestamp=processing_end.isoformat(),
                        vital_signs_timestamp=vital_signs_record.timestamp,
                        news2_score=news2_result.total_score,
                        individual_scores=news2_result.individual_scores,
                        risk_category=news2_result.risk_category.value,
                        scale_used=news2_result.scale_used,
                        monitoring_frequency=news2_result.monitoring_frequency,
                        red_flags=news2_result.red_flags,
                        trending_data=await self._get_trending_data(vital_signs.patient_id),
                        processing_latency_ms=latency_ms
                    )
                    
                    # Publish to results topic
                    await self.news2_results_topic.send(value=result_record)
                    
                    # Check if alert should be sent
                    if await self._should_send_alert(news2_result):
                        await self.alerts_topic.send(value=result_record)
                    
                    # Mark as processed
                    await self._idempotency_manager.mark_processed(
                        vital_signs_record.event_id,
                        vital_signs_record.patient_id,
                        metadata={
                            'news2_score': news2_result.total_score,
                            'risk_category': news2_result.risk_category.value,
                            'processing_latency_ms': latency_ms
                        }
                    )
                    
                    self._processed_count += 1
                    
                    # Log successful processing
                    logger.debug(f"Processed event {vital_signs_record.event_id} "
                               f"for patient {vital_signs_record.patient_id} "
                               f"in {latency_ms:.2f}ms")
                    
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Error processing event {vital_signs_record.event_id}: {e}")
                    
                    # Handle error with retry/dead letter queue
                    await self._error_handler.handle_processing_error(
                        vital_signs_record, e, attempt=1
                    )
        
        return process_vital_signs
    
    async def _convert_to_vital_signs(self, record: VitalSignsRecord) -> VitalSigns:
        """Convert Faust record to internal VitalSigns model."""
        try:
            return VitalSigns(
                event_id=UUID(record.event_id),
                patient_id=record.patient_id,
                timestamp=datetime.fromisoformat(record.timestamp),
                respiratory_rate=record.respiratory_rate,
                spo2=record.spo2,
                on_oxygen=record.on_oxygen,
                temperature=record.temperature,
                systolic_bp=record.systolic_bp,
                heart_rate=record.heart_rate,
                consciousness=ConsciousnessLevel(record.consciousness),
                data_source=record.data_source,
                quality_flags=record.quality_flags
            )
        except Exception as e:
            logger.error(f"Error converting record to VitalSigns: {e}")
            raise ValueError(f"Invalid vital signs data: {e}")
    
    async def _calculate_news2_with_context(self, vital_signs: VitalSigns) -> NEWS2Result:
        """Calculate NEWS2 with patient context and COPD handling."""
        try:
            # Get patient state for COPD detection
            patient_state = await self._patient_registry.get_patient_state(vital_signs.patient_id)
            
            # Determine if COPD patient
            is_copd_patient = False
            if patient_state:
                is_copd_patient = patient_state.clinical_flags.get('is_copd_patient', False)
            
            # Calculate NEWS2
            news2_result = await self._news2_calculator.calculate_news2(
                vital_signs, is_copd_patient=is_copd_patient
            )
            
            return news2_result
            
        except Exception as e:
            logger.error(f"Error calculating NEWS2 for patient {vital_signs.patient_id}: {e}")
            raise
    
    async def _get_trending_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get trending analysis data for the patient."""
        try:
            # This would integrate with the trending service
            # For now, return basic placeholder
            return {
                'trend_available': False,
                'message': 'Trending data integration pending'
            }
        except Exception as e:
            logger.warning(f"Could not retrieve trending data for patient {patient_id}: {e}")
            return None
    
    async def _should_send_alert(self, news2_result: NEWS2Result) -> bool:
        """Determine if an alert should be sent based on NEWS2 result."""
        # Send alerts for high risk scores or red flags
        return (
            news2_result.total_score >= 7 or  # High risk
            len(news2_result.red_flags) > 0 or  # Any red flags
            news2_result.risk_category == RiskCategory.HIGH
        )
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        uptime = datetime.now(timezone.utc) - self._start_time
        uptime_seconds = uptime.total_seconds()
        
        return {
            'processed_count': self._processed_count,
            'error_count': self._error_count,
            'uptime_seconds': uptime_seconds,
            'processing_rate': self._processed_count / uptime_seconds if uptime_seconds > 0 else 0,
            'error_rate': self._error_count / max(self._processed_count, 1),
            'start_time': self._start_time.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the stream processor."""
        try:
            health_status = {
                'status': 'healthy',
                'app_id': self.app_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check Redis connection
            if self._redis_client:
                await self._redis_client.ping()
                health_status['redis'] = 'connected'
            else:
                health_status['redis'] = 'not_initialized'
            
            # Check idempotency manager
            if self._idempotency_manager:
                idempotency_health = await self._idempotency_manager.health_check()
                health_status['idempotency'] = idempotency_health['status']
            
            # Get processing stats
            stats = await self.get_processing_stats()
            health_status['processing_stats'] = stats
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the stream processor."""
        try:
            logger.info("Shutting down stream processor...")
            
            # Close Redis connection
            if self._redis_client:
                await self._redis_client.close()
            
            # Close idempotency manager
            if self._idempotency_manager:
                await self._idempotency_manager.close()
            
            logger.info("Stream processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create the stream processor instance
NEWS2StreamProcessor = StreamProcessor()


# Web endpoint for health checks
@NEWS2StreamProcessor.app.page('/health')
async def health_endpoint(web, request):
    """Health check endpoint."""
    health_data = await NEWS2StreamProcessor.health_check()
    return web.json(health_data)


@NEWS2StreamProcessor.app.page('/stats')  
async def stats_endpoint(web, request):
    """Processing statistics endpoint."""
    stats = await NEWS2StreamProcessor.get_processing_stats()
    return web.json(stats)


# Initialize services when app starts
@NEWS2StreamProcessor.app.on_startup
async def startup():
    """Initialize services on app startup."""
    await NEWS2StreamProcessor.initialize_services()
    # Register the processing agent
    NEWS2StreamProcessor.vital_signs_processor


if __name__ == '__main__':
    # Run the Faust app
    NEWS2StreamProcessor.app.main()