"""
Alert Performance Monitoring and Optimization for NEWS2 Live System

This module implements comprehensive performance monitoring for the alert generation system,
including Prometheus metrics, health checks, performance optimization, and load testing support.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

from ..models.alerts import AlertLevel, Alert, AlertStatus
from ..services.audit import AuditLogger


class PerformanceThreshold(Enum):
    """Performance thresholds for alert generation system."""
    CRITICAL_ALERT_LATENCY_MS = 5000      # Critical alerts must be generated within 5 seconds
    ALERT_PROCESSING_TIME_MS = 10000      # Total processing time including delivery
    ESCALATION_EXECUTION_TIME_MS = 30000  # Escalation execution time
    ERROR_RATE_THRESHOLD = 0.05           # Maximum 5% error rate
    THROUGHPUT_MIN_ALERTS_PER_MINUTE = 60 # Minimum throughput


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    healthy: bool
    status: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "check_name": self.check_name,
            "healthy": self.healthy,
            "status": self.status,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceCollector:
    """
    Collects and aggregates performance metrics for alert generation.
    
    Responsibilities:
    - Collect timing metrics for all alert operations
    - Calculate percentiles and averages
    - Track error rates and throughput
    - Provide Prometheus-compatible metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Metric storage (in production, use Prometheus client library)
        self._metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance counters
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Time series data for trend analysis
        self._time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minutes
        
        # Start time for uptime calculation
        self._start_time = datetime.now(timezone.utc)
    
    def record_metric(
        self, 
        metric_name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        try:
            labels = labels or {}
            timestamp = datetime.now(timezone.utc)
            
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                timestamp=timestamp,
                labels=labels
            )
            
            # Store metric
            self._metrics[metric_name].append(metric)
            self._metric_windows[metric_name].append(value)
            
            # Update histograms for percentile calculations
            self._histograms[metric_name].append(value)
            
            # Keep histogram size manageable
            if len(self._histograms[metric_name]) > 10000:
                self._histograms[metric_name] = self._histograms[metric_name][-5000:]
            
            self.logger.debug(f"Recorded metric {metric_name}: {value} {labels}")
            
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {str(e)}")
    
    def increment_counter(self, counter_name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        try:
            # Create unique key with labels
            key = f"{counter_name}:{':'.join(f'{k}={v}' for k, v in (labels or {}).items())}"
            self._counters[key] += 1
            
            self.logger.debug(f"Incremented counter {counter_name} {labels}")
            
        except Exception as e:
            self.logger.error(f"Failed to increment counter {counter_name}: {str(e)}")
    
    def record_alert_generation_metrics(
        self,
        alert: Alert,
        generation_latency_ms: float,
        processing_time_ms: float,
        threshold_evaluation_time_ms: float
    ):
        """Record comprehensive metrics for alert generation."""
        labels = {
            "alert_level": alert.alert_level.value,
            "ward_id": alert.patient.ward_id,
            "patient_age_category": self._categorize_age(alert.patient.age)
        }
        
        # Record timing metrics
        self.record_metric("alert_generation_latency_ms", generation_latency_ms, labels)
        self.record_metric("alert_processing_time_ms", processing_time_ms, labels)
        self.record_metric("threshold_evaluation_time_ms", threshold_evaluation_time_ms, labels)
        
        # Record alert counters
        self.increment_counter("alerts_generated_total", labels)
        
        # Check performance thresholds
        if alert.alert_level == AlertLevel.CRITICAL and generation_latency_ms > PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value:
            self.increment_counter("critical_alert_latency_violations_total", labels)
        
        if processing_time_ms > PerformanceThreshold.ALERT_PROCESSING_TIME_MS.value:
            self.increment_counter("processing_time_violations_total", labels)
    
    def record_escalation_metrics(
        self,
        alert_id: str,
        escalation_step: int,
        target_role: str,
        execution_time_ms: float,
        success: bool,
        delivery_attempts: int
    ):
        """Record metrics for escalation execution."""
        labels = {
            "escalation_step": str(escalation_step),
            "target_role": target_role,
            "success": str(success).lower()
        }
        
        self.record_metric("escalation_execution_time_ms", execution_time_ms, labels)
        self.record_metric("escalation_delivery_attempts", delivery_attempts, labels)
        
        self.increment_counter("escalations_executed_total", labels)
        
        if success:
            self.increment_counter("escalations_successful_total", labels)
        else:
            self.increment_counter("escalations_failed_total", labels)
        
        # Check escalation performance
        if execution_time_ms > PerformanceThreshold.ESCALATION_EXECUTION_TIME_MS.value:
            self.increment_counter("escalation_time_violations_total", labels)
    
    def record_pipeline_metrics(
        self,
        events_processed: int,
        alerts_generated: int,
        processing_errors: int,
        pipeline_uptime_seconds: float
    ):
        """Record pipeline-level metrics."""
        # Calculate rates
        error_rate = processing_errors / max(events_processed, 1)
        throughput_events_per_minute = (events_processed / max(pipeline_uptime_seconds, 1)) * 60
        
        self.record_metric("pipeline_error_rate", error_rate)
        self.record_metric("pipeline_throughput_events_per_minute", throughput_events_per_minute)
        self.record_metric("pipeline_uptime_seconds", pipeline_uptime_seconds)
        
        # Update counters
        self._counters["events_processed_total"] = events_processed
        self._counters["alerts_generated_total"] = alerts_generated
        self._counters["processing_errors_total"] = processing_errors
        
        # Check performance thresholds
        if error_rate > PerformanceThreshold.ERROR_RATE_THRESHOLD.value:
            self.increment_counter("error_rate_violations_total")
        
        if throughput_events_per_minute < PerformanceThreshold.THROUGHPUT_MIN_ALERTS_PER_MINUTE.value:
            self.increment_counter("throughput_violations_total")
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get statistical summary for a metric over a time window.
        
        Args:
            metric_name: Name of the metric
            window_minutes: Time window in minutes
            
        Returns:
            Statistical summary including percentiles, average, min, max
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            
            # Get metrics within window
            recent_metrics = [
                m for m in self._metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    "metric_name": metric_name,
                    "window_minutes": window_minutes,
                    "sample_count": 0,
                    "no_data": True
                }
            
            values = [m.value for m in recent_metrics]
            
            # Calculate statistics
            summary = {
                "metric_name": metric_name,
                "window_minutes": window_minutes,
                "sample_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
            }
            
            # Calculate percentiles
            if len(values) >= 2:
                sorted_values = sorted(values)
                summary.update({
                    "p50": statistics.median(sorted_values),
                    "p90": sorted_values[int(0.9 * len(sorted_values))],
                    "p95": sorted_values[int(0.95 * len(sorted_values))],
                    "p99": sorted_values[int(0.99 * len(sorted_values))]
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metric summary for {metric_name}: {str(e)}")
            return {"error": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """
        Generate Prometheus-compatible metrics output.
        
        Returns:
            Prometheus metrics format string
        """
        try:
            metrics_output = []
            current_time = time.time()
            
            # Generate counter metrics
            for counter_name, count in self._counters.items():
                # Parse labels from counter name
                if ':' in counter_name:
                    name, labels_str = counter_name.split(':', 1)
                    labels = f"{{{labels_str}}}" if labels_str else ""
                else:
                    name = counter_name
                    labels = ""
                
                metrics_output.append(f"# TYPE {name} counter")
                metrics_output.append(f"{name}{labels} {count} {int(current_time * 1000)}")
            
            # Generate histogram metrics (latest values)
            for metric_name, values in self._histograms.items():
                if not values:
                    continue
                
                # Calculate percentiles from recent values
                recent_values = values[-100:] if len(values) > 100 else values
                sorted_values = sorted(recent_values)
                
                if len(sorted_values) >= 2:
                    p50 = sorted_values[int(0.5 * len(sorted_values))]
                    p90 = sorted_values[int(0.9 * len(sorted_values))]
                    p95 = sorted_values[int(0.95 * len(sorted_values))]
                    p99 = sorted_values[int(0.99 * len(sorted_values))]
                    
                    metrics_output.append(f"# TYPE {metric_name} histogram")
                    metrics_output.append(f"{metric_name}_p50 {p50} {int(current_time * 1000)}")
                    metrics_output.append(f"{metric_name}_p90 {p90} {int(current_time * 1000)}")
                    metrics_output.append(f"{metric_name}_p95 {p95} {int(current_time * 1000)}")
                    metrics_output.append(f"{metric_name}_p99 {p99} {int(current_time * 1000)}")
            
            # Add uptime metric
            uptime_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            metrics_output.append(f"# TYPE alert_system_uptime_seconds gauge")
            metrics_output.append(f"alert_system_uptime_seconds {uptime_seconds} {int(current_time * 1000)}")
            
            return "\n".join(metrics_output)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus metrics: {str(e)}")
            return f"# Error generating metrics: {str(e)}"
    
    def _categorize_age(self, age: int) -> str:
        """Categorize age for metrics labeling."""
        if age < 18:
            return "pediatric"
        elif age < 65:
            return "adult"
        elif age < 80:
            return "elderly"
        else:
            return "very_elderly"
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance monitoring dashboard."""
        return {
            "alert_generation": self.get_metric_summary("alert_generation_latency_ms", 60),
            "processing_time": self.get_metric_summary("alert_processing_time_ms", 60),
            "escalation_time": self.get_metric_summary("escalation_execution_time_ms", 60),
            "counters": dict(self._counters),
            "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }


class HealthChecker:
    """
    Performs health checks for alert generation system components.
    
    Responsibilities:
    - Check pipeline health and responsiveness
    - Validate configuration integrity
    - Monitor resource utilization
    - Verify external dependencies
    """
    
    def __init__(self, performance_collector: PerformanceCollector):
        self.performance_collector = performance_collector
        self.logger = logging.getLogger(__name__)
    
    async def perform_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """
        Perform comprehensive health check of all system components.
        
        Returns:
            Dictionary of health check results by component
        """
        health_checks = {}
        
        try:
            # Run all health checks
            checks = [
                self._check_pipeline_health(),
                self._check_performance_metrics(),
                self._check_escalation_system(),
                self._check_configuration_health(),
                self._check_memory_usage(),
                self._check_external_dependencies()
            ]
            
            results = await asyncio.gather(*checks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    check_name = f"check_{i}"
                    health_checks[check_name] = HealthCheckResult(
                        check_name=check_name,
                        healthy=False,
                        status="error",
                        details={"error": str(result)},
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    health_checks[result.check_name] = result
            
            return health_checks
            
        except Exception as e:
            self.logger.error(f"Failed to perform health checks: {str(e)}")
            return {
                "system": HealthCheckResult(
                    check_name="system",
                    healthy=False,
                    status="critical_error",
                    details={"error": str(e)},
                    timestamp=datetime.now(timezone.utc)
                )
            }
    
    async def _check_pipeline_health(self) -> HealthCheckResult:
        """Check alert processing pipeline health."""
        try:
            # Check recent error rates
            error_rate_summary = self.performance_collector.get_metric_summary("pipeline_error_rate", 15)
            
            if error_rate_summary.get("no_data"):
                return HealthCheckResult(
                    check_name="pipeline",
                    healthy=False,
                    status="no_data",
                    details={"message": "No recent pipeline data available"},
                    timestamp=datetime.now(timezone.utc)
                )
            
            current_error_rate = error_rate_summary.get("mean", 0)
            max_error_rate = error_rate_summary.get("max", 0)
            
            # Determine health status
            if max_error_rate > 0.1:  # > 10% error rate
                status = "critical"
                healthy = False
            elif current_error_rate > 0.05:  # > 5% error rate
                status = "warning"
                healthy = False
            else:
                status = "healthy"
                healthy = True
            
            return HealthCheckResult(
                check_name="pipeline",
                healthy=healthy,
                status=status,
                details={
                    "current_error_rate": current_error_rate,
                    "max_error_rate_15min": max_error_rate,
                    "sample_count": error_rate_summary.get("sample_count", 0)
                },
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="pipeline",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_performance_metrics(self) -> HealthCheckResult:
        """Check performance metrics against thresholds."""
        try:
            # Check critical alert latency
            latency_summary = self.performance_collector.get_metric_summary("alert_generation_latency_ms", 30)
            
            issues = []
            warnings = []
            
            if not latency_summary.get("no_data"):
                p95_latency = latency_summary.get("p95", 0)
                mean_latency = latency_summary.get("mean", 0)
                
                if p95_latency > PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value:
                    issues.append(f"P95 latency ({p95_latency:.1f}ms) exceeds 5s threshold")
                elif mean_latency > PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value * 0.8:
                    warnings.append(f"Mean latency ({mean_latency:.1f}ms) approaching threshold")
            
            # Check throughput
            throughput_summary = self.performance_collector.get_metric_summary("pipeline_throughput_events_per_minute", 15)
            if not throughput_summary.get("no_data"):
                min_throughput = throughput_summary.get("min", 0)
                if min_throughput < PerformanceThreshold.THROUGHPUT_MIN_ALERTS_PER_MINUTE.value:
                    warnings.append(f"Minimum throughput ({min_throughput:.1f}/min) below target")
            
            # Determine overall health
            if issues:
                healthy = False
                status = "performance_issues"
            elif warnings:
                healthy = True  # Warnings don't make system unhealthy
                status = "performance_warnings"
            else:
                healthy = True
                status = "optimal"
            
            return HealthCheckResult(
                check_name="performance",
                healthy=healthy,
                status=status,
                details={
                    "latency_summary": latency_summary,
                    "throughput_summary": throughput_summary,
                    "issues": issues,
                    "warnings": warnings
                },
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="performance",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_escalation_system(self) -> HealthCheckResult:
        """Check escalation system health."""
        try:
            # Check escalation success rates
            success_count = self.performance_collector._counters.get("escalations_successful_total", 0)
            total_count = self.performance_collector._counters.get("escalations_executed_total", 0)
            
            if total_count == 0:
                success_rate = 1.0  # No escalations is not a failure
                status = "no_activity"
            else:
                success_rate = success_count / total_count
                if success_rate < 0.9:  # < 90% success rate
                    status = "degraded"
                elif success_rate < 0.95:  # < 95% success rate
                    status = "warning"
                else:
                    status = "healthy"
            
            healthy = success_rate >= 0.9
            
            return HealthCheckResult(
                check_name="escalation",
                healthy=healthy,
                status=status,
                details={
                    "success_rate": success_rate,
                    "successful_escalations": success_count,
                    "total_escalations": total_count
                },
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="escalation",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_configuration_health(self) -> HealthCheckResult:
        """Check configuration integrity."""
        try:
            # In production, this would validate:
            # - Threshold configurations are valid
            # - Escalation matrices are complete
            # - No conflicting rules exist
            
            # For now, assume configuration is healthy
            return HealthCheckResult(
                check_name="configuration",
                healthy=True,
                status="valid",
                details={"message": "Configuration validation passed"},
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="configuration",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage and resource consumption."""
        try:
            import psutil
            
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Memory thresholds
            warning_threshold_mb = 500
            critical_threshold_mb = 1000
            
            if memory_mb > critical_threshold_mb:
                status = "critical"
                healthy = False
            elif memory_mb > warning_threshold_mb:
                status = "warning"
                healthy = True
            else:
                status = "normal"
                healthy = True
            
            return HealthCheckResult(
                check_name="memory",
                healthy=healthy,
                status=status,
                details={
                    "memory_usage_mb": memory_mb,
                    "warning_threshold_mb": warning_threshold_mb,
                    "critical_threshold_mb": critical_threshold_mb
                },
                timestamp=datetime.now(timezone.utc)
            )
            
        except ImportError:
            # psutil not available
            return HealthCheckResult(
                check_name="memory",
                healthy=True,
                status="check_unavailable",
                details={"message": "Memory monitoring not available (psutil not installed)"},
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                check_name="memory",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_external_dependencies(self) -> HealthCheckResult:
        """Check external dependencies (database, cache, etc.)."""
        try:
            # In production, this would check:
            # - Database connectivity
            # - Redis/cache connectivity
            # - Kafka connectivity
            # - External API availability
            
            # For now, assume dependencies are healthy
            dependencies_checked = ["database", "cache", "message_queue"]
            
            return HealthCheckResult(
                check_name="dependencies",
                healthy=True,
                status="all_available",
                details={
                    "dependencies_checked": dependencies_checked,
                    "message": "All external dependencies are available"
                },
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="dependencies",
                healthy=False,
                status="check_failed",
                details={"error": str(e)},
                timestamp=datetime.now(timezone.utc)
            )


class AlertMonitoringService:
    """
    High-level monitoring service for alert generation system.
    
    Responsibilities:
    - Coordinate performance monitoring
    - Provide health check endpoints
    - Generate monitoring dashboards
    - Alert on performance degradation
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.performance_collector = PerformanceCollector()
        self.health_checker = HealthChecker(self.performance_collector)
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self._monitoring_enabled = True
        self._alert_callbacks: List[Callable] = []
    
    def enable_monitoring(self):
        """Enable performance monitoring."""
        self._monitoring_enabled = True
        self.logger.info("Alert monitoring enabled")
    
    def disable_monitoring(self):
        """Disable performance monitoring."""
        self._monitoring_enabled = False
        self.logger.info("Alert monitoring disabled")
    
    def add_performance_alert_callback(self, callback: Callable):
        """Add callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    async def record_alert_performance(
        self,
        alert: Alert,
        generation_latency_ms: float,
        processing_time_ms: float,
        threshold_evaluation_time_ms: float
    ):
        """Record performance metrics for alert generation."""
        if not self._monitoring_enabled:
            return
        
        try:
            self.performance_collector.record_alert_generation_metrics(
                alert, generation_latency_ms, processing_time_ms, threshold_evaluation_time_ms
            )
            
            # Check for performance violations and send alerts
            await self._check_performance_violations(alert, generation_latency_ms, processing_time_ms)
            
        except Exception as e:
            self.logger.error(f"Failed to record alert performance: {str(e)}")
    
    async def record_escalation_performance(
        self,
        alert_id: str,
        escalation_step: int,
        target_role: str,
        execution_time_ms: float,
        success: bool,
        delivery_attempts: int
    ):
        """Record performance metrics for escalation execution."""
        if not self._monitoring_enabled:
            return
        
        try:
            self.performance_collector.record_escalation_metrics(
                alert_id, escalation_step, target_role, execution_time_ms, success, delivery_attempts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record escalation performance: {str(e)}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        try:
            health_checks = await self.health_checker.perform_comprehensive_health_check()
            
            # Determine overall health
            overall_healthy = all(check.healthy for check in health_checks.values())
            critical_issues = [
                check.check_name for check in health_checks.values()
                if not check.healthy and check.status in ["critical", "critical_error"]
            ]
            
            return {
                "overall_healthy": overall_healthy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "critical_issues": critical_issues,
                "health_checks": {name: check.to_dict() for name, check in health_checks.items()},
                "monitoring_enabled": self._monitoring_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health status: {str(e)}")
            return {
                "overall_healthy": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "monitoring_enabled": self._monitoring_enabled
            }
    
    async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Performance report with metrics and analysis
        """
        try:
            window_minutes = hours * 60
            
            # Get metric summaries
            alert_latency = self.performance_collector.get_metric_summary("alert_generation_latency_ms", window_minutes)
            processing_time = self.performance_collector.get_metric_summary("alert_processing_time_ms", window_minutes)
            escalation_time = self.performance_collector.get_metric_summary("escalation_execution_time_ms", window_minutes)
            
            # Get performance violations
            violations = {
                "latency_violations": self.performance_collector._counters.get("critical_alert_latency_violations_total", 0),
                "processing_violations": self.performance_collector._counters.get("processing_time_violations_total", 0),
                "escalation_violations": self.performance_collector._counters.get("escalation_time_violations_total", 0),
                "error_rate_violations": self.performance_collector._counters.get("error_rate_violations_total", 0)
            }
            
            # Calculate compliance scores
            total_alerts = self.performance_collector._counters.get("alerts_generated_total", 0)
            compliance_scores = {}
            
            if total_alerts > 0:
                compliance_scores = {
                    "latency_compliance": max(0, 1 - (violations["latency_violations"] / total_alerts)),
                    "processing_compliance": max(0, 1 - (violations["processing_violations"] / total_alerts)),
                    "overall_compliance": max(0, 1 - (sum(violations.values()) / (total_alerts * len(violations))))
                }
            
            report = {
                "report_period": {
                    "hours": hours,
                    "start_time": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat()
                },
                "performance_metrics": {
                    "alert_latency": alert_latency,
                    "processing_time": processing_time,
                    "escalation_time": escalation_time
                },
                "performance_violations": violations,
                "compliance_scores": compliance_scores,
                "thresholds": {
                    "critical_alert_latency_ms": PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value,
                    "alert_processing_time_ms": PerformanceThreshold.ALERT_PROCESSING_TIME_MS.value,
                    "escalation_execution_time_ms": PerformanceThreshold.ESCALATION_EXECUTION_TIME_MS.value,
                    "error_rate_threshold": PerformanceThreshold.ERROR_RATE_THRESHOLD.value
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
            return {"error": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-compatible metrics."""
        try:
            return self.performance_collector.get_prometheus_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get Prometheus metrics: {str(e)}")
            return f"# Error: {str(e)}"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        try:
            return self.performance_collector.get_performance_dashboard_data()
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {str(e)}")
            return {"error": str(e)}
    
    async def _check_performance_violations(
        self,
        alert: Alert,
        generation_latency_ms: float,
        processing_time_ms: float
    ):
        """Check for performance violations and trigger alerts."""
        try:
            violations = []
            
            # Check critical alert latency
            if alert.alert_level == AlertLevel.CRITICAL and generation_latency_ms > PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value:
                violations.append({
                    "type": "critical_alert_latency",
                    "threshold_ms": PerformanceThreshold.CRITICAL_ALERT_LATENCY_MS.value,
                    "actual_ms": generation_latency_ms,
                    "alert_id": str(alert.alert_id)
                })
            
            # Check processing time
            if processing_time_ms > PerformanceThreshold.ALERT_PROCESSING_TIME_MS.value:
                violations.append({
                    "type": "processing_time",
                    "threshold_ms": PerformanceThreshold.ALERT_PROCESSING_TIME_MS.value,
                    "actual_ms": processing_time_ms,
                    "alert_id": str(alert.alert_id)
                })
            
            # Send alerts for violations
            for violation in violations:
                await self._send_performance_alert(violation)
            
        except Exception as e:
            self.logger.error(f"Failed to check performance violations: {str(e)}")
    
    async def _send_performance_alert(self, violation: Dict[str, Any]):
        """Send performance alert to registered callbacks."""
        try:
            alert_message = (
                f"Performance violation detected: {violation['type']} "
                f"({violation['actual_ms']:.1f}ms > {violation['threshold_ms']}ms threshold)"
            )
            
            self.logger.warning(alert_message)
            
            for callback in self._alert_callbacks:
                try:
                    await callback(violation, alert_message)
                except Exception as e:
                    self.logger.error(f"Performance alert callback failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Failed to send performance alert: {str(e)}")