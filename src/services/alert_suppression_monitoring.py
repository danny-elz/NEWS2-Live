"""
Enhanced monitoring and alerting system for NEWS2 Alert Suppression.
Provides comprehensive observability and alerting capabilities.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import UUID, uuid4
import json
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MonitoringEventType(Enum):
    """Types of monitoring events."""
    SUPPRESSION_APPLIED = "suppression_applied"
    SUPPRESSION_LIFTED = "suppression_lifted"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_CHANGED = "configuration_changed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: MonitoringEventType = MonitoringEventType.SUPPRESSION_APPLIED
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = "alert_suppression"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source_component": self.source_component
        }


@dataclass
class MetricThreshold:
    """Defines threshold for metric alerting."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    window_minutes: int = 5
    consecutive_violations: int = 3
    alert_severity: AlertSeverity = AlertSeverity.MEDIUM
    alert_message: str = ""
    
    def is_violated(self, value: float) -> bool:
        """Check if threshold is violated."""
        if self.comparison_operator == ">":
            return value > self.threshold_value
        elif self.comparison_operator == "<":
            return value < self.threshold_value
        elif self.comparison_operator == ">=":
            return value >= self.threshold_value
        elif self.comparison_operator == "<=":
            return value <= self.threshold_value
        elif self.comparison_operator == "==":
            return value == self.threshold_value
        elif self.comparison_operator == "!=":
            return value != self.threshold_value
        return False


class AlertChannel(ABC):
    """Abstract base class for alert channels."""
    
    @abstractmethod
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Send alert through this channel."""
        pass


class EmailAlertChannel(AlertChannel):
    """Email alert channel implementation."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, recipients: List[str], use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls
        self.logger = logging.getLogger(__name__)
    
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Send email alert."""
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = f"[{event.severity.value.upper()}] NEWS2 Alert: {event.event_type.value}"
            
            body = self._format_email_body(event)
            msg.attach(MimeText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for event {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _format_email_body(self, event: MonitoringEvent) -> str:
        """Format email body with event details."""
        return f"""
        <html>
        <body>
        <h2>NEWS2 Alert Suppression Monitoring Alert</h2>
        <p><strong>Event Type:</strong> {event.event_type.value}</p>
        <p><strong>Severity:</strong> {event.severity.value}</p>
        <p><strong>Message:</strong> {event.message}</p>
        <p><strong>Timestamp:</strong> {event.timestamp.isoformat()}</p>
        <p><strong>Source:</strong> {event.source_component}</p>
        
        <h3>Metadata</h3>
        <pre>{json.dumps(event.metadata, indent=2)}</pre>
        </body>
        </html>
        """


class SlackAlertChannel(AlertChannel):
    """Slack alert channel implementation."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Send Slack alert."""
        try:
            import aiohttp
            
            color_map = {
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.HIGH: "#ff8800",
                AlertSeverity.MEDIUM: "#ffaa00",
                AlertSeverity.LOW: "#00aa00",
                AlertSeverity.INFO: "#0088cc"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(event.severity, "#808080"),
                    "title": f"NEWS2 Alert: {event.event_type.value}",
                    "text": event.message,
                    "fields": [
                        {"title": "Severity", "value": event.severity.value, "short": True},
                        {"title": "Source", "value": event.source_component, "short": True},
                        {"title": "Timestamp", "value": event.timestamp.isoformat(), "short": False}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack alert sent for event {event.event_id}")
                        return True
                    else:
                        self.logger.error(f"Slack alert failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class PrometheusMetrics:
    """Prometheus metrics collector for alert suppression."""
    
    def __init__(self, registry: Optional[Any] = None):
        self.registry = registry or (CollectorRegistry() if PROMETHEUS_AVAILABLE else None)
        self.enabled = PROMETHEUS_AVAILABLE and self.registry is not None
        
        if self.enabled:
            self._init_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.suppression_decisions = Counter(
            'news2_suppression_decisions_total',
            'Total number of suppression decisions',
            ['decision_type', 'patient_id'],
            registry=self.registry
        )
        
        self.suppression_duration = Histogram(
            'news2_suppression_duration_seconds',
            'Duration of alert suppressions',
            ['suppression_type'],
            registry=self.registry
        )
        
        self.performance_metrics = Histogram(
            'news2_suppression_processing_time_seconds',
            'Time taken to process suppression logic',
            ['operation'],
            registry=self.registry
        )
        
        self.active_suppressions = Gauge(
            'news2_active_suppressions',
            'Number of currently active suppressions',
            registry=self.registry
        )
        
        self.health_status = Gauge(
            'news2_suppression_health_status',
            'Health status of suppression components',
            ['component'],
            registry=self.registry
        )
    
    def record_suppression_decision(self, decision_type: str, patient_id: str):
        """Record a suppression decision."""
        if self.enabled:
            self.suppression_decisions.labels(
                decision_type=decision_type, 
                patient_id=patient_id
            ).inc()
    
    def record_suppression_duration(self, suppression_type: str, duration_seconds: float):
        """Record suppression duration."""
        if self.enabled:
            self.suppression_duration.labels(suppression_type=suppression_type).observe(duration_seconds)
    
    def record_processing_time(self, operation: str, duration_seconds: float):
        """Record processing time for operations."""
        if self.enabled:
            self.performance_metrics.labels(operation=operation).observe(duration_seconds)
    
    def set_active_suppressions(self, count: int):
        """Set the number of active suppressions."""
        if self.enabled:
            self.active_suppressions.set(count)
    
    def set_component_health(self, component: str, is_healthy: bool):
        """Set component health status."""
        if self.enabled:
            self.health_status.labels(component=component).set(1 if is_healthy else 0)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if self.enabled:
            return generate_latest(self.registry).decode('utf-8')
        return ""


class SuppressionMonitor:
    """Comprehensive monitoring system for alert suppression."""
    
    def __init__(self, 
                 alert_channels: List[AlertChannel] = None,
                 prometheus_registry: Optional[Any] = None,
                 redis_client: Optional[Any] = None):
        self.alert_channels = alert_channels or []
        self.prometheus = PrometheusMetrics(prometheus_registry)
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        self.thresholds: List[MetricThreshold] = []
        self.metric_history: Dict[str, List[Dict[str, Any]]] = {}
        self.violation_counts: Dict[str, int] = {}
        
        self._setup_default_thresholds()
        self._monitoring_tasks: Set[asyncio.Task] = set()
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds."""
        self.thresholds = [
            MetricThreshold(
                metric_name="suppression_processing_time",
                threshold_value=5.0,
                comparison_operator=">",
                window_minutes=5,
                consecutive_violations=3,
                alert_severity=AlertSeverity.HIGH,
                alert_message="Suppression processing time exceeding 5 seconds"
            ),
            MetricThreshold(
                metric_name="active_suppressions",
                threshold_value=1000,
                comparison_operator=">",
                window_minutes=10,
                consecutive_violations=2,
                alert_severity=AlertSeverity.MEDIUM,
                alert_message="High number of active suppressions detected"
            ),
            MetricThreshold(
                metric_name="circuit_breaker_failures",
                threshold_value=5,
                comparison_operator=">=",
                window_minutes=1,
                consecutive_violations=1,
                alert_severity=AlertSeverity.CRITICAL,
                alert_message="Circuit breaker failure threshold exceeded"
            )
        ]
    
    def add_alert_channel(self, channel: AlertChannel):
        """Add an alert channel."""
        self.alert_channels.append(channel)
    
    def add_threshold(self, threshold: MetricThreshold):
        """Add a metric threshold for monitoring."""
        self.thresholds.append(threshold)
    
    async def emit_event(self, event: MonitoringEvent):
        """Emit a monitoring event."""
        self.logger.info(f"Monitoring event: {event.event_type.value} - {event.message}")
        
        # Store event in Redis if available
        if self.redis_client:
            try:
                await self._store_event_in_redis(event)
            except Exception as e:
                self.logger.error(f"Failed to store event in Redis: {e}")
        
        # Send alerts if severity warrants it
        if event.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM]:
            await self._send_alerts(event)
    
    async def _store_event_in_redis(self, event: MonitoringEvent):
        """Store event in Redis for persistence."""
        key = f"news2:monitoring:events:{event.timestamp.strftime('%Y-%m-%d')}"
        await self.redis_client.lpush(key, json.dumps(event.to_dict()))
        await self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    async def _send_alerts(self, event: MonitoringEvent):
        """Send alerts through all configured channels."""
        tasks = []
        for channel in self.alert_channels:
            task = asyncio.create_task(channel.send_alert(event))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for result in results if result is True)
            self.logger.info(f"Sent alert to {successful}/{len(tasks)} channels")
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric value and check thresholds."""
        timestamp = time.time()
        
        # Store metric in history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append({
            "value": value,
            "timestamp": timestamp,
            "metadata": metadata or {}
        })
        
        # Keep only recent history (1 hour)
        cutoff_time = timestamp - 3600
        self.metric_history[metric_name] = [
            entry for entry in self.metric_history[metric_name]
            if entry["timestamp"] > cutoff_time
        ]
        
        # Check thresholds
        asyncio.create_task(self._check_thresholds(metric_name, value))
    
    async def _check_thresholds(self, metric_name: str, value: float):
        """Check if metric value violates any thresholds."""
        for threshold in self.thresholds:
            if threshold.metric_name == metric_name:
                if threshold.is_violated(value):
                    violation_key = f"{metric_name}:{threshold.threshold_value}"
                    self.violation_counts[violation_key] = self.violation_counts.get(violation_key, 0) + 1
                    
                    if self.violation_counts[violation_key] >= threshold.consecutive_violations:
                        await self._trigger_threshold_alert(threshold, value)
                        self.violation_counts[violation_key] = 0
                else:
                    # Reset violation count if threshold not violated
                    violation_key = f"{metric_name}:{threshold.threshold_value}"
                    self.violation_counts[violation_key] = 0
    
    async def _trigger_threshold_alert(self, threshold: MetricThreshold, current_value: float):
        """Trigger alert for threshold violation."""
        event = MonitoringEvent(
            event_type=MonitoringEventType.THRESHOLD_EXCEEDED,
            severity=threshold.alert_severity,
            message=f"{threshold.alert_message}. Current value: {current_value}",
            metadata={
                "metric_name": threshold.metric_name,
                "threshold_value": threshold.threshold_value,
                "current_value": current_value,
                "comparison": threshold.comparison_operator,
                "consecutive_violations": threshold.consecutive_violations
            }
        )
        
        await self.emit_event(event)
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring tasks."""
        # Health check monitoring
        health_task = asyncio.create_task(self._health_check_loop())
        self._monitoring_tasks.add(health_task)
        
        # Performance monitoring
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        self._monitoring_tasks.add(perf_task)
        
        # Cleanup old metrics
        cleanup_task = asyncio.create_task(self._cleanup_metrics_loop())
        self._monitoring_tasks.add(cleanup_task)
        
        self.logger.info("Started continuous monitoring tasks")
    
    async def _health_check_loop(self):
        """Continuous health checking."""
        while True:
            try:
                # Check Redis connection
                if self.redis_client:
                    await self.redis_client.ping()
                    self.prometheus.set_component_health("redis", True)
                
                # Check Prometheus metrics
                if self.prometheus.enabled:
                    self.prometheus.set_component_health("prometheus", True)
                
                # Check alert channels
                for i, channel in enumerate(self.alert_channels):
                    try:
                        # Basic connectivity check could be added here
                        self.prometheus.set_component_health(f"alert_channel_{i}", True)
                    except Exception:
                        self.prometheus.set_component_health(f"alert_channel_{i}", False)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait longer if error
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance continuously."""
        while True:
            try:
                # Monitor active suppressions count
                if self.redis_client:
                    active_count = await self.redis_client.scard("news2:active_suppressions")
                    self.prometheus.set_active_suppressions(active_count)
                    self.record_metric("active_suppressions", active_count)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_metrics_loop(self):
        """Clean up old metrics periodically."""
        while True:
            try:
                cutoff_time = time.time() - 3600  # Keep 1 hour of history
                
                for metric_name in list(self.metric_history.keys()):
                    self.metric_history[metric_name] = [
                        entry for entry in self.metric_history[metric_name]
                        if entry["timestamp"] > cutoff_time
                    ]
                    
                    if not self.metric_history[metric_name]:
                        del self.metric_history[metric_name]
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(600)
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        for task in self._monitoring_tasks:
            task.cancel()
        
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        self.logger.info("Stopped all monitoring tasks")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        status = {
            "overall_health": "healthy",
            "components": {},
            "active_alerts": 0,
            "monitoring_enabled": True,
            "last_check": datetime.utcnow().isoformat()
        }
        
        # Check component health
        components = ["redis", "prometheus"] + [f"alert_channel_{i}" for i in range(len(self.alert_channels))]
        
        for component in components:
            try:
                if component == "redis" and self.redis_client:
                    await self.redis_client.ping()
                    status["components"][component] = "healthy"
                elif component == "prometheus" and self.prometheus.enabled:
                    status["components"][component] = "healthy"
                elif component.startswith("alert_channel"):
                    status["components"][component] = "healthy"
                else:
                    status["components"][component] = "disabled"
            except Exception:
                status["components"][component] = "unhealthy"
                status["overall_health"] = "degraded"
        
        # Count recent critical alerts
        if self.redis_client:
            try:
                today_key = f"news2:monitoring:events:{datetime.utcnow().strftime('%Y-%m-%d')}"
                events = await self.redis_client.lrange(today_key, 0, -1)
                critical_count = 0
                
                for event_json in events:
                    event_data = json.loads(event_json)
                    if event_data.get("severity") == "critical":
                        critical_count += 1
                
                status["active_alerts"] = critical_count
                
            except Exception as e:
                self.logger.error(f"Failed to count alerts: {e}")
        
        return status


# Integration helper functions
def create_monitoring_system(config: Dict[str, Any]) -> SuppressionMonitor:
    """Create a fully configured monitoring system."""
    alert_channels = []
    
    # Setup email alerts if configured
    if config.get("email_alerts", {}).get("enabled", False):
        email_config = config["email_alerts"]
        email_channel = EmailAlertChannel(
            smtp_host=email_config["smtp_host"],
            smtp_port=email_config["smtp_port"],
            username=email_config["username"],
            password=email_config["password"],
            recipients=email_config["recipients"],
            use_tls=email_config.get("use_tls", True)
        )
        alert_channels.append(email_channel)
    
    # Setup Slack alerts if configured
    if config.get("slack_alerts", {}).get("enabled", False):
        slack_config = config["slack_alerts"]
        slack_channel = SlackAlertChannel(slack_config["webhook_url"])
        alert_channels.append(slack_channel)
    
    # Setup Redis client if configured
    redis_client = None
    if config.get("redis", {}).get("enabled", False) and REDIS_AVAILABLE:
        redis_config = config["redis"]
        redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            decode_responses=True
        )
    
    # Setup Prometheus registry if configured
    prometheus_registry = None
    if config.get("prometheus", {}).get("enabled", False) and PROMETHEUS_AVAILABLE:
        prometheus_registry = CollectorRegistry()
    
    monitor = SuppressionMonitor(
        alert_channels=alert_channels,
        prometheus_registry=prometheus_registry,
        redis_client=redis_client
    )
    
    # Add custom thresholds from config
    for threshold_config in config.get("custom_thresholds", []):
        threshold = MetricThreshold(
            metric_name=threshold_config["metric_name"],
            threshold_value=threshold_config["threshold_value"],
            comparison_operator=threshold_config["comparison_operator"],
            window_minutes=threshold_config.get("window_minutes", 5),
            consecutive_violations=threshold_config.get("consecutive_violations", 3),
            alert_severity=AlertSeverity(threshold_config.get("alert_severity", "medium")),
            alert_message=threshold_config.get("alert_message", "")
        )
        monitor.add_threshold(threshold)
    
    return monitor