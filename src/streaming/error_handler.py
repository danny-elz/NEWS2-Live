"""
Error handling and retry mechanisms for stream processing.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import faust

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of processing errors."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for processing errors."""
    event_id: str
    patient_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    error_message: str
    stack_trace: Optional[str]
    timestamp: datetime
    attempt_number: int
    original_event: Dict[str, Any]
    processing_metadata: Optional[Dict[str, Any]] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for protecting downstream services."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.seconds >= self.timeout_seconds
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class StreamErrorHandler:
    """Handles errors in stream processing with retry and dead letter queue."""
    
    def __init__(self, 
                 dead_letter_topic: faust.Topic,
                 max_retries: int = 3,
                 base_delay_seconds: float = 1.0,
                 max_delay_seconds: float = 60.0):
        
        self.dead_letter_topic = dead_letter_topic
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        
        # Circuit breakers for different service types
        self.circuit_breakers = {
            'database': CircuitBreaker(failure_threshold=5, timeout_seconds=30),
            'news2_calculation': CircuitBreaker(failure_threshold=10, timeout_seconds=15),
            'patient_registry': CircuitBreaker(failure_threshold=5, timeout_seconds=30)
        }
        
        # Error statistics
        self._error_stats = {
            'total_errors': 0,
            'retry_attempts': 0,
            'dead_letter_sent': 0,
            'errors_by_type': {},
            'last_reset': datetime.now(timezone.utc)
        }
    
    async def handle_processing_error(self, 
                                    original_event: Any,
                                    error: Exception,
                                    attempt: int = 1,
                                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle processing error with retry logic and dead letter queue.
        
        Args:
            original_event: The original event that failed processing
            error: The exception that occurred
            attempt: Current attempt number (1-based)
            context: Additional context information
        """
        try:
            # Extract event information
            event_id = getattr(original_event, 'event_id', 'unknown')
            patient_id = getattr(original_event, 'patient_id', 'unknown')
            
            # Classify error
            error_type = self._classify_error(error)
            severity = self._determine_severity(error_type, attempt)
            
            # Create error context
            error_context = ErrorContext(
                event_id=event_id,
                patient_id=patient_id,
                error_type=error_type,
                severity=severity,
                error_message=str(error),
                stack_trace=self._get_stack_trace(error),
                timestamp=datetime.now(timezone.utc),
                attempt_number=attempt,
                original_event=self._serialize_event(original_event),
                processing_metadata=context
            )
            
            # Update error statistics
            self._update_error_stats(error_type)
            
            # Log error
            self._log_error(error_context)
            
            # Decide whether to retry or send to dead letter queue
            if attempt <= self.max_retries and self._should_retry(error_type, attempt):
                await self._schedule_retry(original_event, error_context, attempt)
            else:
                await self._send_to_dead_letter_queue(error_context)
                
        except Exception as handler_error:
            logger.error(f"Error in error handler: {handler_error}")
            # Fallback: try to send to dead letter queue
            await self._send_fallback_error(original_event, error, handler_error)
    
    async def handle_with_circuit_breaker(self, 
                                        service_name: str,
                                        func,
                                        *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if service_name not in self.circuit_breakers:
            raise ValueError(f"Unknown service: {service_name}")
        
        circuit_breaker = self.circuit_breakers[service_name]
        return await circuit_breaker.call(func, *args, **kwargs)
    
    async def _schedule_retry(self, 
                            original_event: Any,
                            error_context: ErrorContext,
                            attempt: int) -> None:
        """Schedule retry with exponential backoff."""
        delay = min(
            self.base_delay_seconds * (2 ** (attempt - 1)),
            self.max_delay_seconds
        )
        
        logger.warning(f"Scheduling retry #{attempt} for event {error_context.event_id} "
                      f"in {delay:.2f} seconds")
        
        self._error_stats['retry_attempts'] += 1
        
        # In a production system, you might use a task queue like Celery
        # For now, we'll use asyncio.sleep (not recommended for production)
        await asyncio.sleep(delay)
        
        # TODO: Implement proper retry scheduling with task queue
        logger.info(f"Retry scheduled for event {error_context.event_id}")
    
    async def _send_to_dead_letter_queue(self, error_context: ErrorContext) -> None:
        """Send failed event to dead letter queue."""
        try:
            dead_letter_record = {
                'error_context': asdict(error_context),
                'failed_at': datetime.now(timezone.utc).isoformat(),
                'requires_manual_intervention': True
            }
            
            await self.dead_letter_topic.send(value=dead_letter_record)
            self._error_stats['dead_letter_sent'] += 1
            
            logger.error(f"Sent event {error_context.event_id} to dead letter queue "
                        f"after {error_context.attempt_number} attempts")
                        
        except Exception as e:
            logger.critical(f"Failed to send event {error_context.event_id} "
                           f"to dead letter queue: {e}")
    
    async def _send_fallback_error(self, 
                                 original_event: Any,
                                 original_error: Exception,
                                 handler_error: Exception) -> None:
        """Fallback error handling when main error handler fails."""
        try:
            fallback_record = {
                'type': 'fallback_error',
                'original_event': str(original_event),
                'original_error': str(original_error),
                'handler_error': str(handler_error),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'requires_immediate_attention': True
            }
            
            await self.dead_letter_topic.send(value=fallback_record)
            logger.critical("Sent fallback error record to dead letter queue")
            
        except Exception as e:
            logger.critical(f"Complete error handling failure: {e}")
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type based on exception."""
        error_name = error.__class__.__name__.lower()
        error_message = str(error).lower()
        
        if 'validation' in error_message or 'invalid' in error_message:
            return ErrorType.VALIDATION_ERROR
        elif 'calculation' in error_message or 'news2' in error_message:
            return ErrorType.CALCULATION_ERROR
        elif 'database' in error_message or 'connection' in error_message:
            return ErrorType.DATABASE_ERROR
        elif 'network' in error_message or 'timeout' in error_message:
            return ErrorType.NETWORK_ERROR
        elif 'timeout' in error_name:
            return ErrorType.TIMEOUT_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: ErrorType, attempt: int) -> ErrorSeverity:
        """Determine error severity based on type and attempt number."""
        if error_type == ErrorType.VALIDATION_ERROR:
            return ErrorSeverity.LOW if attempt <= 2 else ErrorSeverity.MEDIUM
        elif error_type == ErrorType.CALCULATION_ERROR:
            return ErrorSeverity.HIGH  # NEWS2 calculation errors are serious
        elif error_type == ErrorType.DATABASE_ERROR:
            return ErrorSeverity.CRITICAL if attempt > 2 else ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if error should be retried."""
        # Don't retry validation errors after first attempt
        if error_type == ErrorType.VALIDATION_ERROR and attempt > 1:
            return False
        
        # Always retry network and timeout errors
        if error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
            return True
        
        # Retry database errors
        if error_type == ErrorType.DATABASE_ERROR:
            return True
        
        # Don't retry calculation errors (likely data issue)
        if error_type == ErrorType.CALCULATION_ERROR:
            return False
        
        # Retry unknown errors conservatively
        return attempt <= 2
    
    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """Serialize event for storage in error context."""
        try:
            if hasattr(event, 'to_dict'):
                return event.to_dict()
            elif hasattr(event, '__dict__'):
                return dict(event.__dict__)
            else:
                return {'serialized': str(event)}
        except Exception as e:
            logger.warning(f"Could not serialize event: {e}")
            return {'error': 'serialization_failed', 'type': str(type(event))}
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Get stack trace from exception."""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity."""
        message = (f"Processing error for event {error_context.event_id}: "
                  f"{error_context.error_message}")
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)
    
    def _update_error_stats(self, error_type: ErrorType) -> None:
        """Update error statistics."""
        self._error_stats['total_errors'] += 1
        
        error_type_str = error_type.value
        if error_type_str not in self._error_stats['errors_by_type']:
            self._error_stats['errors_by_type'][error_type_str] = 0
        self._error_stats['errors_by_type'][error_type_str] += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get current error statistics."""
        uptime = datetime.now(timezone.utc) - self._error_stats['last_reset']
        
        return {
            **self._error_stats,
            'uptime_seconds': uptime.total_seconds(),
            'circuit_breaker_states': {
                name: cb.state.value 
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def reset_error_stats(self) -> None:
        """Reset error statistics."""
        self._error_stats = {
            'total_errors': 0,
            'retry_attempts': 0,
            'dead_letter_sent': 0,
            'errors_by_type': {},
            'last_reset': datetime.now(timezone.utc)
        }