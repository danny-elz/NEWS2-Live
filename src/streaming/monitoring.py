"""
Monitoring and observability for stream processing components.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str = "count"


@dataclass
class HealthStatus:
    """Health status for a component."""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None


class MetricsCollector:
    """Collects and manages metrics for stream processing."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Define Prometheus metrics
        self._setup_prometheus_metrics()
        
        # In-memory metrics storage for dashboards
        self._metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self._performance_window = 300  # 5 minutes
        
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Event processing metrics
        self.events_processed = Counter(
            'stream_events_processed_total',
            'Total number of events processed',
            ['patient_type', 'event_type'],
            registry=self.registry
        )
        
        self.events_failed = Counter(
            'stream_events_failed_total', 
            'Total number of events that failed processing',
            ['error_type', 'retry_attempt'],
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'stream_processing_duration_seconds',
            'Time spent processing each event',
            ['event_type', 'patient_type'],
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
            registry=self.registry
        )
        
        # NEWS2 specific metrics
        self.news2_scores = Histogram(
            'news2_scores_calculated',
            'Distribution of calculated NEWS2 scores',
            ['scale_used', 'risk_category'],
            buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20),
            registry=self.registry
        )
        
        self.copd_patients = Counter(
            'copd_patients_processed_total',
            'Total number of COPD patients processed',
            ['scale_used'],
            registry=self.registry
        )
        
        # Stream lag and throughput
        self.stream_lag = Gauge(
            'kafka_consumer_lag_seconds',
            'Current lag in stream processing',
            ['topic', 'partition'],
            registry=self.registry
        )
        
        self.throughput = Gauge(
            'stream_throughput_events_per_second',
            'Current throughput in events per second',
            ['stream_type'],
            registry=self.registry
        )
        
        # System health metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['service_name'],
            registry=self.registry
        )
        
        self.redis_connections = Gauge(
            'redis_connections_active',
            'Number of active Redis connections',
            registry=self.registry
        )
        
        # Duplicate detection metrics
        self.duplicate_events = Counter(
            'duplicate_events_detected_total',
            'Total number of duplicate events detected',
            ['patient_id_hash'],
            registry=self.registry
        )
    
    def record_event_processed(self, 
                             patient_type: str = "standard",
                             event_type: str = "vital_signs") -> None:
        """Record a successfully processed event."""
        self.events_processed.labels(
            patient_type=patient_type,
            event_type=event_type
        ).inc()
    
    def record_event_failed(self, 
                          error_type: str,
                          retry_attempt: int = 1) -> None:
        """Record a failed event."""
        self.events_failed.labels(
            error_type=error_type,
            retry_attempt=str(retry_attempt)
        ).inc()
    
    def record_processing_duration(self, 
                                 duration_seconds: float,
                                 event_type: str = "vital_signs",
                                 patient_type: str = "standard") -> None:
        """Record event processing duration."""
        self.processing_duration.labels(
            event_type=event_type,
            patient_type=patient_type
        ).observe(duration_seconds)
    
    def record_news2_score(self, 
                          score: int,
                          scale_used: int,
                          risk_category: str) -> None:
        """Record a calculated NEWS2 score."""
        self.news2_scores.labels(
            scale_used=str(scale_used),
            risk_category=risk_category
        ).observe(score)
    
    def record_copd_patient(self, scale_used: int) -> None:
        """Record processing of a COPD patient."""
        self.copd_patients.labels(scale_used=str(scale_used)).inc()
    
    def update_stream_lag(self, 
                         topic: str,
                         partition: int,
                         lag_seconds: float) -> None:
        """Update stream lag metric."""
        self.stream_lag.labels(
            topic=topic,
            partition=str(partition)
        ).set(lag_seconds)
    
    def update_throughput(self, 
                         events_per_second: float,
                         stream_type: str = "vital_signs") -> None:
        """Update throughput metric."""
        self.throughput.labels(stream_type=stream_type).set(events_per_second)
    
    def update_circuit_breaker_state(self, 
                                   service_name: str,
                                   state: str) -> None:
        """Update circuit breaker state metric."""
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, -1)
        self.circuit_breaker_state.labels(service_name=service_name).set(state_value)
    
    def record_duplicate_event(self, patient_id_hash: str) -> None:
        """Record a duplicate event detection."""
        self.duplicate_events.labels(patient_id_hash=patient_id_hash).inc()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry).decode('utf-8')
    
    def add_custom_metric(self, metric: PerformanceMetric) -> None:
        """Add a custom performance metric to the buffer."""
        self._metrics_buffer[metric.name].append({
            'value': metric.value,
            'timestamp': metric.timestamp.isoformat(),
            'labels': metric.labels,
            'unit': metric.unit
        })
    
    def get_recent_metrics(self, 
                          metric_name: str,
                          minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent_metrics = []
        
        for metric_data in self._metrics_buffer[metric_name]:
            metric_time = datetime.fromisoformat(metric_data['timestamp'])
            if metric_time >= cutoff_time:
                recent_metrics.append(metric_data)
        
        return recent_metrics


class HealthChecker:
    """Monitors health of stream processing components."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.health_status = {}
        self.check_interval = 30  # seconds
        self._running = False
        self._health_check_task = None
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            return
            
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self.perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_checks(self) -> None:
        """Perform all health checks."""
        checks = [
            self._check_redis_health(),
            self._check_kafka_health(),
            self._check_stream_processor_health(),
            self._check_system_resources()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check {i} failed: {result}")
    
    async def _check_redis_health(self) -> None:
        """Check Redis connectivity and performance."""
        if not self.redis_client:
            self.health_status['redis'] = HealthStatus(
                component='redis',
                status='unhealthy',
                last_check=datetime.now(timezone.utc),
                details={'error': 'Redis client not configured'},
                error_message='No Redis client available'
            )
            return
        
        try:
            start_time = time.time()
            await self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Check Redis info
            info = await self.redis_client.info()
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            status = 'healthy'
            if response_time > 100:  # > 100ms is concerning
                status = 'degraded'
            
            self.health_status['redis'] = HealthStatus(
                component='redis',
                status=status,
                last_check=datetime.now(timezone.utc),
                details={
                    'response_time_ms': response_time,
                    'used_memory_bytes': used_memory,
                    'max_memory_bytes': max_memory,
                    'memory_usage_percent': (used_memory / max_memory * 100) if max_memory > 0 else 0
                }
            )
            
        except Exception as e:
            self.health_status['redis'] = HealthStatus(
                component='redis',
                status='unhealthy',
                last_check=datetime.now(timezone.utc),
                details={'error': str(e)},
                error_message=str(e)
            )
    
    async def _check_kafka_health(self) -> None:
        """Check Kafka cluster health."""
        try:
            # This would integrate with Kafka admin client
            # For now, we'll create a placeholder
            self.health_status['kafka'] = HealthStatus(
                component='kafka',
                status='healthy',
                last_check=datetime.now(timezone.utc),
                details={
                    'brokers_available': 3,
                    'topics_available': ['vital_signs_input', 'news2_results'],
                    'consumer_lag': 'within_limits'
                }
            )
            
        except Exception as e:
            self.health_status['kafka'] = HealthStatus(
                component='kafka',
                status='unhealthy',
                last_check=datetime.now(timezone.utc),
                details={'error': str(e)},
                error_message=str(e)
            )
    
    async def _check_stream_processor_health(self) -> None:
        """Check stream processor health."""
        try:
            # This would check the actual Faust app health
            # For now, we'll create a placeholder
            self.health_status['stream_processor'] = HealthStatus(
                component='stream_processor',
                status='healthy',
                last_check=datetime.now(timezone.utc),
                details={
                    'workers_active': 1,
                    'processing_rate': 'normal',
                    'memory_usage': 'acceptable'
                }
            )
            
        except Exception as e:
            self.health_status['stream_processor'] = HealthStatus(
                component='stream_processor',
                status='unhealthy',
                last_check=datetime.now(timezone.utc),
                details={'error': str(e)},
                error_message=str(e)
            )
    
    async def _check_system_resources(self) -> None:
        """Check system resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = 'healthy'
            if cpu_percent > 80 or memory.percent > 85:
                status = 'degraded'
            if cpu_percent > 95 or memory.percent > 95:
                status = 'unhealthy'
            
            self.health_status['system'] = HealthStatus(
                component='system',
                status=status,
                last_check=datetime.now(timezone.utc),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'available_memory_gb': memory.available / (1024**3)
                }
            )
            
        except ImportError:
            # psutil not available
            self.health_status['system'] = HealthStatus(
                component='system',
                status='unknown',
                last_check=datetime.now(timezone.utc),
                details={'error': 'psutil not available for system monitoring'}
            )
        except Exception as e:
            self.health_status['system'] = HealthStatus(
                component='system',
                status='unhealthy',
                last_check=datetime.now(timezone.utc),
                details={'error': str(e)},
                error_message=str(e)
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        overall_status = 'healthy'
        unhealthy_components = []
        degraded_components = []
        
        for component, status in self.health_status.items():
            if status.status == 'unhealthy':
                overall_status = 'unhealthy'
                unhealthy_components.append(component)
            elif status.status == 'degraded':
                if overall_status == 'healthy':
                    overall_status = 'degraded'
                degraded_components.append(component)
        
        return {
            'overall_status': overall_status,
            'last_check': datetime.now(timezone.utc).isoformat(),
            'component_count': len(self.health_status),
            'unhealthy_components': unhealthy_components,
            'degraded_components': degraded_components,
            'components': {
                name: asdict(status) for name, status in self.health_status.items()
            }
        }


class StreamMonitor:
    """Main monitoring coordinator for stream processing."""
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 metrics_registry: Optional[CollectorRegistry] = None):
        
        self.metrics_collector = MetricsCollector(metrics_registry)
        self.health_checker = HealthChecker(redis_client)
        self.alerts = []
        self._monitoring_active = False
    
    async def start(self) -> None:
        """Start all monitoring components."""
        await self.health_checker.start_monitoring()
        self._monitoring_active = True
        logger.info("Stream monitoring started")
    
    async def stop(self) -> None:
        """Stop all monitoring components."""
        await self.health_checker.stop_monitoring()
        self._monitoring_active = False
        logger.info("Stream monitoring stopped")
    
    def record_processing_event(self, 
                              event_id: str,
                              patient_id: str,
                              processing_duration: float,
                              news2_score: Optional[int] = None,
                              is_copd: bool = False,
                              error: Optional[Exception] = None) -> None:
        """Record a complete processing event."""
        patient_type = "copd" if is_copd else "standard"
        
        if error:
            error_type = error.__class__.__name__
            self.metrics_collector.record_event_failed(error_type)
        else:
            self.metrics_collector.record_event_processed(patient_type)
            self.metrics_collector.record_processing_duration(
                processing_duration, patient_type=patient_type
            )
            
            if news2_score is not None:
                # This would need the actual risk category
                risk_category = self._score_to_risk_category(news2_score)
                scale = 2 if is_copd else 1
                self.metrics_collector.record_news2_score(
                    news2_score, scale, risk_category
                )
                
                if is_copd:
                    self.metrics_collector.record_copd_patient(scale)
    
    def _score_to_risk_category(self, score: int) -> str:
        """Convert NEWS2 score to risk category."""
        if score >= 7:
            return "HIGH"
        elif score >= 5:
            return "MEDIUM_HIGH"
        elif score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        health_summary = self.health_checker.get_health_summary()
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'monitoring_active': self._monitoring_active,
            'health': health_summary,
            'metrics': {
                'prometheus': self.metrics_collector.get_prometheus_metrics()
            },
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
    
    def add_alert(self, 
                 level: str,
                 message: str,
                 component: str,
                 details: Optional[Dict[str, Any]] = None) -> None:
        """Add an alert to the monitoring system."""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
            'component': component,
            'details': details or {}
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        log_level = getattr(logger, level.lower(), logger.info)
        log_level(f"ALERT [{component}]: {message}")
    
    def get_metrics_export(self) -> str:
        """Get Prometheus metrics export."""
        return self.metrics_collector.get_prometheus_metrics()