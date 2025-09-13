"""
Failure Recovery and Resilience Mechanisms for Alert Suppression System

Implements circuit breakers, retry logic, fallback mechanisms, and health monitoring
to ensure the suppression system remains available and fails safely.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import time
import json

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying recovery
    success_threshold: int = 3   # Successes needed to close
    timeout_seconds: float = 30.0  # Request timeout


@dataclass
class HealthStatus:
    """Health status for system components."""
    component_name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time_ms: float
    error_count: int
    details: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    
    Prevents system overload by failing fast when dependencies are unavailable
    and automatically recovering when they become healthy again.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Metrics
        self.calls_total = Counter(f'circuit_breaker_calls_total', 'Total circuit breaker calls', ['name', 'result'])
        self.failures_total = Counter(f'circuit_breaker_failures_total', 'Total circuit breaker failures', ['name'])
        self.state_transitions = Counter(f'circuit_breaker_state_transitions_total', 'Circuit breaker state transitions', ['name', 'from_state', 'to_state'])
    
    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions when circuit is closed
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.calls_total.labels(name=self.name, result='rejected').inc()
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.timeout_seconds
            )
            
            self._on_success()
            self.calls_total.labels(name=self.name, result='success').inc()
            return result
            
        except Exception as e:
            self._on_failure()
            self.calls_total.labels(name=self.name, result='failure').inc()
            self.failures_total.labels(name=self.name).inc()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN to HALF_OPEN."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned from {old_state.value} to HALF_OPEN")
        self.state_transitions.labels(name=self.name, from_state=old_state.value, to_state='half_open').inc()
    
    def _on_success(self):
        """Handle successful function execution."""
        self.last_success_time = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success in CLOSED state
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed function execution."""
        self.last_failure_time = datetime.now(timezone.utc)
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in HALF_OPEN goes back to OPEN
            self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned from {old_state.value} to CLOSED")
        self.state_transitions.labels(name=self.name, from_state=old_state.value, to_state='closed').inc()
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.logger.warning(f"Circuit breaker {self.name} transitioned from {old_state.value} to OPEN after {self.failure_count} failures")
        self.state_transitions.labels(name=self.name, from_state=old_state.value, to_state='open').inc()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """
    Retry manager with exponential backoff and jitter.
    
    Provides intelligent retry logic for transient failures while
    preventing thundering herd problems.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.retry_attempts = Counter('retry_attempts_total', 'Total retry attempts', ['operation', 'attempt'])
        self.retry_success = Counter('retry_success_total', 'Successful retries', ['operation'])
        self.retry_exhausted = Counter('retry_exhausted_total', 'Retries exhausted', ['operation'])
    
    async def retry_with_backoff(
        self,
        func: Callable[[], Awaitable[Any]],
        operation_name: str,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: tuple = (Exception,),
        *args, **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Async function to retry
            operation_name: Name for metrics/logging
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Exponential backoff multiplier
            jitter: Add random jitter to prevent thundering herd
            retry_exceptions: Exceptions that trigger retry
            
        Returns:
            Function result
            
        Raises:
            Exception: When all retries exhausted
        """
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                self.retry_attempts.labels(operation=operation_name, attempt=str(attempt + 1)).inc()
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:  # Only log if we actually retried
                    self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                    self.retry_success.labels(operation=operation_name).inc()
                
                return result
                
            except retry_exceptions as e:
                last_exception = e
                
                if attempt == max_attempts - 1:  # Last attempt
                    self.retry_exhausted.labels(operation=operation_name).inc()
                    self.logger.error(f"Operation {operation_name} failed after {max_attempts} attempts: {str(e)}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                
                # Add jitter to prevent thundering herd
                if jitter:
                    import random
                    delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                
                self.logger.warning(f"Operation {operation_name} failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {str(e)}")
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception


class HealthMonitor:
    """
    Health monitoring system for suppression components.
    
    Continuously monitors system health and provides early warning
    of degradation before complete failures occur.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        self.health_checks = {}
        self.health_status = {}
        
        # Metrics
        self.health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['component'])
        self.health_check_status = Gauge('health_check_status', 'Health check status (1=healthy, 0=unhealthy)', ['component'])
        self.component_errors = Counter('component_errors_total', 'Component error count', ['component', 'error_type'])
    
    def register_health_check(self, component_name: str, check_func: Callable[[], Awaitable[bool]], interval_seconds: int = 30):
        """
        Register a health check for a component.
        
        Args:
            component_name: Name of component to monitor
            check_func: Async function that returns True if healthy
            interval_seconds: Check interval in seconds
        """
        self.health_checks[component_name] = {
            "check_func": check_func,
            "interval_seconds": interval_seconds,
            "last_check": None,
            "next_check": datetime.now(timezone.utc)
        }
        
        self.health_status[component_name] = HealthStatus(
            component_name=component_name,
            status="unknown",
            last_check=datetime.now(timezone.utc),
            response_time_ms=0.0,
            error_count=0
        )
    
    async def check_all_components(self) -> Dict[str, HealthStatus]:
        """
        Check health of all registered components.
        
        Returns:
            Dictionary of component health statuses
        """
        current_time = datetime.now(timezone.utc)
        
        for component_name, check_config in self.health_checks.items():
            if current_time >= check_config["next_check"]:
                await self._check_component_health(component_name, check_config)
        
        return self.health_status.copy()
    
    async def _check_component_health(self, component_name: str, check_config: Dict[str, Any]):
        """Check health of a single component."""
        start_time = time.time()
        
        try:
            is_healthy = await check_config["check_func"]()
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            # Update health status
            self.health_status[component_name] = HealthStatus(
                component_name=component_name,
                status="healthy" if is_healthy else "degraded",
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time_ms,
                error_count=self.health_status[component_name].error_count if not is_healthy else 0,
                details={"last_response_time_ms": response_time_ms}
            )
            
            # Update metrics
            self.health_check_duration.labels(component=component_name).observe(response_time_ms / 1000)
            self.health_check_status.labels(component=component_name).set(1 if is_healthy else 0)
            
            # Schedule next check
            check_config["next_check"] = datetime.now(timezone.utc) + timedelta(seconds=check_config["interval_seconds"])
            
            if not is_healthy:
                self.logger.warning(f"Health check failed for {component_name}")
                self.component_errors.labels(component=component_name, error_type='health_check_failed').inc()
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Update health status with error
            current_status = self.health_status[component_name]
            self.health_status[component_name] = HealthStatus(
                component_name=component_name,
                status="unhealthy",
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time_ms,
                error_count=current_status.error_count + 1,
                details={"error": str(e), "last_response_time_ms": response_time_ms}
            )
            
            # Update metrics
            self.health_check_duration.labels(component=component_name).observe(response_time_ms / 1000)
            self.health_check_status.labels(component=component_name).set(0)
            self.component_errors.labels(component=component_name, error_type='health_check_exception').inc()
            
            # Schedule next check (shorter interval when unhealthy)
            check_config["next_check"] = datetime.now(timezone.utc) + timedelta(seconds=min(check_config["interval_seconds"], 10))
            
            self.logger.error(f"Health check exception for {component_name}: {str(e)}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health summary.
        
        Returns:
            System health summary
        """
        await self.check_all_components()
        
        healthy_count = sum(1 for status in self.health_status.values() if status.status == "healthy")
        degraded_count = sum(1 for status in self.health_status.values() if status.status == "degraded")
        unhealthy_count = sum(1 for status in self.health_status.values() if status.status == "unhealthy")
        total_count = len(self.health_status)
        
        overall_status = "healthy"
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "total_components": total_count,
            "component_details": {name: status.__dict__ for name, status in self.health_status.items()},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class FallbackManager:
    """
    Fallback manager for graceful degradation when components fail.
    
    Provides alternative behaviors when primary systems are unavailable
    while maintaining safety and basic functionality.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_strategies = {}
        
        # Metrics
        self.fallback_activations = Counter('fallback_activations_total', 'Fallback strategy activations', ['strategy'])
        self.fallback_success = Counter('fallback_success_total', 'Successful fallback executions', ['strategy'])
    
    def register_fallback(self, operation_name: str, fallback_func: Callable[[], Awaitable[Any]]):
        """
        Register a fallback strategy for an operation.
        
        Args:
            operation_name: Name of the operation
            fallback_func: Async fallback function
        """
        self.fallback_strategies[operation_name] = fallback_func
    
    async def execute_with_fallback(self, operation_name: str, primary_func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute primary function with fallback on failure.
        
        Args:
            operation_name: Name of operation for fallback lookup
            primary_func: Primary function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary operation {operation_name} failed, attempting fallback: {str(e)}")
            
            if operation_name in self.fallback_strategies:
                try:
                    self.fallback_activations.labels(strategy=operation_name).inc()
                    result = await self.fallback_strategies[operation_name](*args, **kwargs)
                    self.fallback_success.labels(strategy=operation_name).inc()
                    self.logger.info(f"Fallback for {operation_name} succeeded")
                    return result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback for {operation_name} also failed: {str(fallback_error)}")
                    raise fallback_error
            else:
                self.logger.error(f"No fallback strategy registered for {operation_name}")
                raise e


class ResilientSuppressionEngine:
    """
    Resilient wrapper for the suppression engine with failure recovery.
    
    Provides circuit breakers, retries, health monitoring, and fallbacks
    to ensure the suppression system remains available under adverse conditions.
    """
    
    def __init__(self, suppression_engine, redis_client: redis.Redis):
        self.suppression_engine = suppression_engine
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize resilience components
        self.circuit_breaker = CircuitBreaker(
            "suppression_engine",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
        )
        self.retry_manager = RetryManager()
        self.health_monitor = HealthMonitor(redis_client)
        self.fallback_manager = FallbackManager()
        
        # Register health checks
        self._register_health_checks()
        
        # Register fallback strategies
        self._register_fallback_strategies()
    
    def _register_health_checks(self):
        """Register health checks for system components."""
        
        async def redis_health_check() -> bool:
            """Check Redis connectivity."""
            try:
                await self.redis.ping()
                return True
            except Exception:
                return False
        
        async def suppression_engine_health_check() -> bool:
            """Check suppression engine basic functionality."""
            try:
                # Create minimal test alert
                from ..models.alerts import Alert, AlertLevel, AlertPriority, AlertStatus, AlertDecision
                from ..models.news2 import NEWS2Result, RiskCategory
                from ..models.patient import Patient
                
                news2_result = NEWS2Result(
                    patient_id="HEALTH_CHECK",
                    total_score=1,
                    individual_scores={},
                    risk_category=RiskCategory.LOW,
                    scale_used="Scale 1",
                    timestamp=datetime.now(timezone.utc)
                )
                
                patient = Patient(
                    patient_id="HEALTH_CHECK",
                    age=50,
                    ward_id="TEST_WARD",
                    is_copd_patient=False
                )
                
                alert_decision = AlertDecision(
                    decision_id=uuid4(),
                    patient_id="HEALTH_CHECK",
                    news2_result=news2_result,
                    alert_level=AlertLevel.LOW,
                    alert_priority=AlertPriority.ROUTINE,
                    threshold_applied=None,
                    reasoning="Health check",
                    decision_timestamp=datetime.now(timezone.utc),
                    generation_latency_ms=1.0,
                    single_param_trigger=False,
                    suppressed=False,
                    ward_id="TEST_WARD"
                )
                
                alert = Alert(
                    alert_id=uuid4(),
                    patient_id="HEALTH_CHECK",
                    patient=patient,
                    alert_decision=alert_decision,
                    alert_level=AlertLevel.LOW,
                    alert_priority=AlertPriority.ROUTINE,
                    title="Health Check",
                    message="Health check alert",
                    clinical_context={},
                    created_at=datetime.now(timezone.utc),
                    status=AlertStatus.PENDING,
                    assigned_to=None,
                    acknowledged_at=None,
                    acknowledged_by=None,
                    escalation_step=0,
                    max_escalation_step=1,
                    next_escalation_at=None,
                    resolved_at=None,
                    resolved_by=None
                )
                
                # Test basic suppression decision
                decision = await self.suppression_engine.should_suppress(alert)
                return decision is not None
                
            except Exception:
                return False
        
        self.health_monitor.register_health_check("redis", redis_health_check, 30)
        self.health_monitor.register_health_check("suppression_engine", suppression_engine_health_check, 60)
    
    def _register_fallback_strategies(self):
        """Register fallback strategies for critical operations."""
        
        async def safe_suppression_fallback(*args, **kwargs):
            """
            Safe fallback for suppression decisions.
            
            When the main suppression engine fails, default to NOT suppressing
            to ensure no critical alerts are missed.
            """
            from ..services.alert_suppression import SuppressionDecision
            
            alert = args[0] if args else None
            if not alert:
                raise ValueError("No alert provided for fallback suppression")
            
            # ALWAYS fail safe - do not suppress when system is degraded
            decision = SuppressionDecision(
                decision_id=uuid4(),
                alert_id=alert.alert_id,
                patient_id=alert.patient_id,
                suppressed=False,
                reason="FALLBACK_NO_SUPPRESSION",
                confidence_score=0.0,
                decision_timestamp=datetime.now(timezone.utc),
                suppression_expires_at=None,
                metadata={"fallback": True, "reason": "System degraded, failing safe"}
            )
            
            self.logger.warning(f"Using fallback suppression decision for alert {alert.alert_id}")
            return decision
        
        self.fallback_manager.register_fallback("should_suppress", safe_suppression_fallback)
    
    async def should_suppress(self, alert) -> Any:
        """
        Resilient suppression decision with failure recovery.
        
        Args:
            alert: Alert to evaluate for suppression
            
        Returns:
            SuppressionDecision
        """
        async def suppression_operation():
            return await self.suppression_engine.should_suppress(alert)
        
        # Execute with circuit breaker protection
        try:
            return await self.circuit_breaker.call(
                lambda: self.retry_manager.retry_with_backoff(
                    suppression_operation,
                    "suppression_decision",
                    max_attempts=3,
                    base_delay=0.1,
                    max_delay=2.0
                )
            )
        except Exception as e:
            self.logger.error(f"All suppression attempts failed for alert {alert.alert_id}: {str(e)}")
            
            # Use fallback strategy
            return await self.fallback_manager.execute_with_fallback(
                "should_suppress",
                suppression_operation,
                alert
            )
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = await self.health_monitor.get_system_health()
        
        # Add circuit breaker status
        health_status["circuit_breaker"] = self.circuit_breaker.get_state()
        
        return health_status
    
    async def start_monitoring(self):
        """Start background health monitoring."""
        self.logger.info("Starting resilient suppression engine monitoring")
        
        # Start background task for continuous health monitoring
        asyncio.create_task(self._continuous_health_monitoring())
    
    async def _continuous_health_monitoring(self):
        """Continuous health monitoring background task."""
        while True:
            try:
                await self.health_monitor.check_all_components()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in continuous health monitoring: {str(e)}")
                await asyncio.sleep(30)  # Back off on errors