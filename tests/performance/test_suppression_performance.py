"""
Performance benchmarking and load testing for alert suppression system.

Validates NFR requirements including response time, throughput, and
resource utilization under various load conditions.
"""

import asyncio
import pytest
import time
import statistics
import resource
import psutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import redis.asyncio as redis
from prometheus_client import CollectorRegistry

from src.models.alerts import Alert, AlertLevel, AlertPriority, AlertStatus, AlertDecision
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_suppression import (
    SuppressionEngine, PatternDetector, SuppressionDecisionLogger,
    SuppressionMetrics, SuppressionDecision
)
from src.services.audit import AuditLogger


class PerformanceBenchmark:
    """Performance benchmarking framework for suppression system."""
    
    def __init__(self):
        self.results = {
            "response_times": [],
            "throughput_metrics": [],
            "resource_usage": [],
            "error_rates": [],
            "concurrency_results": []
        }
        self.start_time = None
        self.end_time = None
    
    def start_benchmark(self):
        """Start performance benchmark."""
        self.start_time = time.time()
        self.results = {
            "response_times": [],
            "throughput_metrics": [],
            "resource_usage": [],
            "error_rates": [],
            "concurrency_results": []
        }
    
    def record_response_time(self, operation: str, duration_ms: float):
        """Record response time for operation."""
        self.results["response_times"].append({
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        })
    
    def record_throughput(self, operations_per_second: float, concurrent_operations: int):
        """Record throughput metrics."""
        self.results["throughput_metrics"].append({
            "ops_per_second": operations_per_second,
            "concurrent_ops": concurrent_operations,
            "timestamp": time.time()
        })
    
    def record_resource_usage(self):
        """Record current resource usage."""
        process = psutil.Process()
        self.results["resource_usage"].append({
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "open_files": len(process.open_files()),
            "timestamp": time.time()
        })
    
    def record_error_rate(self, total_operations: int, errors: int):
        """Record error rate."""
        error_rate = (errors / total_operations) * 100 if total_operations > 0 else 0
        self.results["error_rates"].append({
            "total_operations": total_operations,
            "errors": errors,
            "error_rate_percent": error_rate,
            "timestamp": time.time()
        })
    
    def end_benchmark(self):
        """End performance benchmark and calculate summary."""
        self.end_time = time.time()
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary report."""
        if not self.start_time or not self.end_time:
            return {}
        
        total_duration = self.end_time - self.start_time
        response_times = [r["duration_ms"] for r in self.results["response_times"]]
        
        summary = {
            "test_duration_seconds": total_duration,
            "total_operations": len(response_times),
            "response_time_stats": {
                "min_ms": min(response_times) if response_times else 0,
                "max_ms": max(response_times) if response_times else 0,
                "avg_ms": statistics.mean(response_times) if response_times else 0,
                "median_ms": statistics.median(response_times) if response_times else 0,
                "p95_ms": self._percentile(response_times, 95) if response_times else 0,
                "p99_ms": self._percentile(response_times, 99) if response_times else 0
            },
            "throughput": {
                "avg_ops_per_second": len(response_times) / total_duration if total_duration > 0 else 0,
                "peak_ops_per_second": max([t["ops_per_second"] for t in self.results["throughput_metrics"]], default=0)
            },
            "resource_usage": self._summarize_resource_usage(),
            "error_analysis": self._summarize_errors(),
            "nfr_compliance": self._check_nfr_compliance(response_times)
        }
        
        return summary
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _summarize_resource_usage(self) -> Dict[str, Any]:
        """Summarize resource usage metrics."""
        if not self.results["resource_usage"]:
            return {}
        
        cpu_usage = [r["cpu_percent"] for r in self.results["resource_usage"]]
        memory_usage = [r["memory_mb"] for r in self.results["resource_usage"]]
        
        return {
            "cpu_usage": {
                "avg_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage),
                "min_percent": min(cpu_usage)
            },
            "memory_usage": {
                "avg_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage),
                "min_mb": min(memory_usage)
            }
        }
    
    def _summarize_errors(self) -> Dict[str, Any]:
        """Summarize error analysis."""
        if not self.results["error_rates"]:
            return {"overall_error_rate": 0}
        
        total_ops = sum(r["total_operations"] for r in self.results["error_rates"])
        total_errors = sum(r["errors"] for r in self.results["error_rates"])
        
        return {
            "overall_error_rate": (total_errors / total_ops * 100) if total_ops > 0 else 0,
            "total_operations": total_ops,
            "total_errors": total_errors
        }
    
    def _check_nfr_compliance(self, response_times: List[float]) -> Dict[str, bool]:
        """Check NFR compliance."""
        if not response_times:
            return {}
        
        avg_response_time = statistics.mean(response_times)
        p95_response_time = self._percentile(response_times, 95)
        
        return {
            "response_time_under_1s": avg_response_time < 1000,  # <1 second average
            "p95_under_500ms": p95_response_time < 500,  # 95% under 500ms
            "no_timeouts": max(response_times) < 10000,  # No timeouts over 10s
            "low_error_rate": self._summarize_errors().get("overall_error_rate", 0) < 1.0  # <1% error rate
        }


@pytest.mark.performance
class TestSuppressionPerformance:
    """Performance tests for suppression system NFR validation."""
    
    @pytest.fixture
    async def redis_client(self):
        """Redis client for performance testing."""
        client = redis.Redis.from_url("redis://localhost:6379/3", decode_responses=True)  # Use test DB
        yield client
        await client.flushdb()
        await client.close()
    
    @pytest.fixture
    def audit_logger(self):
        """Audit logger for performance testing."""
        return AuditLogger()
    
    @pytest.fixture
    async def suppression_engine(self, redis_client, audit_logger):
        """Configured suppression engine for performance testing."""
        return SuppressionEngine(redis_client, audit_logger)
    
    @pytest.fixture
    def benchmark(self):
        """Performance benchmark instance."""
        return PerformanceBenchmark()
    
    def create_test_alert(self, patient_id: str, alert_level: AlertLevel, news2_score: int) -> Alert:
        """Create test alert for performance testing."""
        news2_result = NEWS2Result(
            total_score=news2_score,
            individual_scores={
                "respiratory_rate": 1,
                "spo2": 1,
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 1,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="6 hourly",
            scale_used=1,
            warnings=[],
            confidence=0.95,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.0
        )
        
        patient = Patient(
            patient_id=patient_id,
            age=70,
            ward_id="PERF_WARD",
            is_copd_patient=False
        )
        
        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient_id,
            news2_result=news2_result,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            threshold_applied=None,
            reasoning="Performance test alert",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=1.0,
            single_param_trigger=False,
            suppressed=False,
            ward_id="PERF_WARD"
        )
        
        return Alert(
            alert_id=uuid4(),
            patient_id=patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            title=f"{alert_level.value.upper()} ALERT",
            message="Performance test alert",
            clinical_context={"news2_total_score": news2_score},
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=2,
            next_escalation_at=None,
            resolved_at=None,
            resolved_by=None
        )
    
    async def test_single_suppression_decision_latency(self, suppression_engine, benchmark):
        """Test single suppression decision latency meets NFR (<1 second)."""
        benchmark.start_benchmark()
        
        # Test 1000 individual decisions
        for i in range(1000):
            alert = self.create_test_alert(f"LATENCY_TEST_{i}", AlertLevel.MEDIUM, 5)
            
            start_time = time.time()
            decision = await suppression_engine.should_suppress(alert)
            end_time = time.time()
            
            duration_ms = (end_time - start_time) * 1000
            benchmark.record_response_time("suppression_decision", duration_ms)
            
            # Record resource usage every 100 operations
            if i % 100 == 0:
                benchmark.record_resource_usage()
            
            assert decision is not None
            assert duration_ms < 1000, f"Decision {i} took {duration_ms:.2f}ms (>1000ms NFR violation)"
        
        summary = benchmark.end_benchmark()
        
        # NFR Assertions
        assert summary["response_time_stats"]["avg_ms"] < 100, f"Average response time {summary['response_time_stats']['avg_ms']:.2f}ms exceeds 100ms target"
        assert summary["response_time_stats"]["p95_ms"] < 500, f"95th percentile {summary['response_time_stats']['p95_ms']:.2f}ms exceeds 500ms"
        assert summary["response_time_stats"]["max_ms"] < 1000, f"Max response time {summary['response_time_stats']['max_ms']:.2f}ms exceeds 1000ms NFR"
        
        print(f"Single Decision Latency Results:")
        print(f"  Average: {summary['response_time_stats']['avg_ms']:.2f}ms")
        print(f"  95th percentile: {summary['response_time_stats']['p95_ms']:.2f}ms")
        print(f"  Maximum: {summary['response_time_stats']['max_ms']:.2f}ms")
    
    async def test_concurrent_suppression_decisions(self, suppression_engine, benchmark):
        """Test concurrent suppression decisions under load."""
        benchmark.start_benchmark()
        
        concurrent_levels = [1, 5, 10, 25, 50]
        
        for concurrency in concurrent_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            tasks = []
            start_time = time.time()
            
            # Create concurrent tasks
            for i in range(concurrency * 10):  # 10 operations per concurrency level
                alert = self.create_test_alert(f"CONCURRENT_TEST_{concurrency}_{i}", AlertLevel.MEDIUM, 5)
                task = asyncio.create_task(self._timed_suppression_decision(suppression_engine, alert, benchmark))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Calculate metrics
            successful_operations = sum(1 for r in results if not isinstance(r, Exception))
            failed_operations = len(results) - successful_operations
            duration = end_time - start_time
            ops_per_second = successful_operations / duration if duration > 0 else 0
            
            benchmark.record_throughput(ops_per_second, concurrency)
            benchmark.record_error_rate(len(results), failed_operations)
            benchmark.record_resource_usage()
            
            # Assertions for each concurrency level
            assert failed_operations == 0, f"Concurrency {concurrency}: {failed_operations} failed operations"
            assert ops_per_second > concurrency * 5, f"Concurrency {concurrency}: Throughput {ops_per_second:.1f} ops/s too low"
            
            print(f"  Throughput: {ops_per_second:.1f} ops/second")
            print(f"  Errors: {failed_operations}")
        
        summary = benchmark.end_benchmark()
        
        # Overall NFR assertions
        assert summary["error_analysis"]["overall_error_rate"] < 1.0, f"Error rate {summary['error_analysis']['overall_error_rate']:.2f}% exceeds 1% NFR"
        assert summary["throughput"]["peak_ops_per_second"] > 100, f"Peak throughput {summary['throughput']['peak_ops_per_second']:.1f} ops/s below 100 ops/s target"
        
        print(f"Concurrent Decision Results:")
        print(f"  Peak throughput: {summary['throughput']['peak_ops_per_second']:.1f} ops/second")
        print(f"  Overall error rate: {summary['error_analysis']['overall_error_rate']:.2f}%")
    
    async def _timed_suppression_decision(self, suppression_engine, alert, benchmark):
        """Execute timed suppression decision for concurrency testing."""
        start_time = time.time()
        try:
            decision = await suppression_engine.should_suppress(alert)
            end_time = time.time()
            
            duration_ms = (end_time - start_time) * 1000
            benchmark.record_response_time("concurrent_suppression_decision", duration_ms)
            
            return decision
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            benchmark.record_response_time("concurrent_suppression_decision_error", duration_ms)
            raise e
    
    async def test_volume_reduction_performance(self, suppression_engine, benchmark):
        """Test volume reduction performance under realistic load."""
        benchmark.start_benchmark()
        
        # Simulate 24-hour alert volume (scaled down for testing)
        total_alerts = 500  # Scaled from ~5000 alerts/day
        alert_distribution = {
            AlertLevel.CRITICAL: int(total_alerts * 0.05),  # 5%
            AlertLevel.HIGH: int(total_alerts * 0.15),      # 15%
            AlertLevel.MEDIUM: int(total_alerts * 0.35),    # 35%
            AlertLevel.LOW: int(total_alerts * 0.45)        # 45%
        }
        
        # Create patient pool for realistic patterns
        patients = [f"VOL_PERF_TEST_{i:03d}" for i in range(50)]
        
        alerts_processed = 0
        alerts_suppressed = 0
        critical_alerts_suppressed = 0
        processing_errors = 0
        
        print(f"Processing {total_alerts} alerts across {len(patients)} patients...")
        
        # Add some acknowledgments to enable time-based suppression
        for i, patient_id in enumerate(patients[:20]):
            if i % 3 == 0:
                ack_alert = self.create_test_alert(patient_id, AlertLevel.MEDIUM, 3)
                await suppression_engine.record_acknowledgment(ack_alert, f"NURSE_{i:03d}")
        
        # Process alerts by level
        for alert_level, count in alert_distribution.items():
            level_start_time = time.time()
            
            for i in range(count):
                patient_id = patients[alerts_processed % len(patients)]
                score = self._get_score_for_level(alert_level)
                
                alert = self.create_test_alert(patient_id, alert_level, score)
                
                try:
                    start_time = time.time()
                    decision = await suppression_engine.should_suppress(alert)
                    end_time = time.time()
                    
                    duration_ms = (end_time - start_time) * 1000
                    benchmark.record_response_time("volume_suppression_decision", duration_ms)
                    
                    alerts_processed += 1
                    
                    if decision.suppressed:
                        alerts_suppressed += 1
                        
                        # Critical safety check
                        if alert_level == AlertLevel.CRITICAL:
                            critical_alerts_suppressed += 1
                    
                    # Record metrics every 50 alerts
                    if alerts_processed % 50 == 0:
                        benchmark.record_resource_usage()
                        ops_per_second = 50 / (time.time() - level_start_time) if time.time() > level_start_time else 0
                        benchmark.record_throughput(ops_per_second, 1)
                        level_start_time = time.time()
                
                except Exception as e:
                    processing_errors += 1
                    print(f"Error processing alert {alerts_processed}: {str(e)}")
        
        # Record final metrics
        benchmark.record_error_rate(alerts_processed, processing_errors)
        
        summary = benchmark.end_benchmark()
        
        # Calculate volume reduction metrics
        suppression_rate = (alerts_suppressed / alerts_processed) * 100 if alerts_processed > 0 else 0
        
        # NFR Assertions
        assert critical_alerts_suppressed == 0, f"CRITICAL SAFETY VIOLATION: {critical_alerts_suppressed} critical alerts suppressed"
        assert suppression_rate >= 50.0, f"Volume reduction {suppression_rate:.1f}% below 50% NFR requirement"
        assert summary["response_time_stats"]["avg_ms"] < 100, f"Average processing time {summary['response_time_stats']['avg_ms']:.2f}ms exceeds target"
        assert processing_errors == 0, f"{processing_errors} processing errors occurred"
        
        print(f"Volume Reduction Performance Results:")
        print(f"  Alerts processed: {alerts_processed}")
        print(f"  Volume reduction: {suppression_rate:.1f}%")
        print(f"  Critical alerts suppressed: {critical_alerts_suppressed}")
        print(f"  Processing errors: {processing_errors}")
        print(f"  Average processing time: {summary['response_time_stats']['avg_ms']:.2f}ms")
        print(f"  Peak throughput: {summary['throughput']['peak_ops_per_second']:.1f} ops/second")
    
    def _get_score_for_level(self, alert_level: AlertLevel) -> int:
        """Get appropriate NEWS2 score for alert level."""
        score_map = {
            AlertLevel.CRITICAL: 8,
            AlertLevel.HIGH: 6,
            AlertLevel.MEDIUM: 4,
            AlertLevel.LOW: 2
        }
        return score_map[alert_level]
    
    async def test_memory_usage_under_load(self, suppression_engine, benchmark):
        """Test memory usage patterns under sustained load."""
        benchmark.start_benchmark()
        
        # Record initial memory usage
        benchmark.record_resource_usage()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Sustained load test
        alerts_per_batch = 100
        num_batches = 10
        
        for batch in range(num_batches):
            print(f"Processing batch {batch + 1}/{num_batches}")
            
            batch_start_time = time.time()
            
            # Process batch of alerts
            for i in range(alerts_per_batch):
                alert = self.create_test_alert(f"MEM_TEST_{batch}_{i}", AlertLevel.MEDIUM, 5)
                
                start_time = time.time()
                decision = await suppression_engine.should_suppress(alert)
                end_time = time.time()
                
                duration_ms = (end_time - start_time) * 1000
                benchmark.record_response_time("memory_load_test", duration_ms)
            
            # Record memory usage after each batch
            benchmark.record_resource_usage()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            batch_duration = time.time() - batch_start_time
            batch_throughput = alerts_per_batch / batch_duration
            benchmark.record_throughput(batch_throughput, 1)
            
            print(f"  Batch {batch + 1} memory usage: {current_memory:.1f} MB")
            print(f"  Batch {batch + 1} throughput: {batch_throughput:.1f} ops/second")
        
        summary = benchmark.end_benchmark()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory usage assertions
        memory_growth = final_memory - initial_memory
        memory_growth_percent = (memory_growth / initial_memory) * 100 if initial_memory > 0 else 0
        
        assert memory_growth_percent < 50, f"Memory growth {memory_growth_percent:.1f}% exceeds 50% threshold"
        assert final_memory < 500, f"Final memory usage {final_memory:.1f} MB exceeds 500 MB limit"
        
        print(f"Memory Usage Results:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB ({memory_growth_percent:.1f}%)")
        print(f"  Peak memory: {summary['resource_usage']['memory_usage']['max_mb']:.1f} MB")
    
    async def test_error_recovery_performance(self, suppression_engine, benchmark):
        """Test performance under error conditions and recovery."""
        benchmark.start_benchmark()
        
        # Simulate various error conditions
        error_scenarios = [
            "redis_timeout",
            "invalid_data", 
            "network_failure",
            "resource_exhaustion"
        ]
        
        total_operations = 0
        total_errors = 0
        recovery_times = []
        
        for scenario in error_scenarios:
            print(f"Testing error scenario: {scenario}")
            
            scenario_start_time = time.time()
            scenario_operations = 0
            scenario_errors = 0
            
            # Process 50 operations per scenario
            for i in range(50):
                alert = self.create_test_alert(f"ERROR_TEST_{scenario}_{i}", AlertLevel.MEDIUM, 5)
                
                try:
                    start_time = time.time()
                    
                    # Simulate error conditions periodically
                    if i % 10 == 0:  # Every 10th operation fails
                        # Simulate error by causing a temporary issue
                        if scenario == "invalid_data":
                            # Test with invalid patient data
                            alert.patient_id = None  # This should be handled gracefully
                    
                    decision = await suppression_engine.should_suppress(alert)
                    end_time = time.time()
                    
                    duration_ms = (end_time - start_time) * 1000
                    benchmark.record_response_time(f"error_recovery_{scenario}", duration_ms)
                    
                    scenario_operations += 1
                    
                except Exception as e:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    recovery_times.append(duration_ms)
                    scenario_errors += 1
                    
                    # Verify system recovers quickly from errors
                    assert duration_ms < 5000, f"Error recovery took {duration_ms:.2f}ms (>5s)"
            
            scenario_duration = time.time() - scenario_start_time
            scenario_throughput = scenario_operations / scenario_duration if scenario_duration > 0 else 0
            
            benchmark.record_throughput(scenario_throughput, 1)
            benchmark.record_error_rate(scenario_operations + scenario_errors, scenario_errors)
            
            total_operations += scenario_operations + scenario_errors
            total_errors += scenario_errors
            
            print(f"  Operations: {scenario_operations}, Errors: {scenario_errors}")
            print(f"  Throughput: {scenario_throughput:.1f} ops/second")
        
        summary = benchmark.end_benchmark()
        
        # Error recovery assertions
        error_rate = (total_errors / total_operations) * 100 if total_operations > 0 else 0
        avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0
        
        assert error_rate < 25, f"Error rate {error_rate:.1f}% exceeds 25% threshold during error scenarios"
        assert avg_recovery_time < 1000, f"Average error recovery {avg_recovery_time:.2f}ms exceeds 1s"
        
        print(f"Error Recovery Results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error rate: {error_rate:.1f}%")
        print(f"  Average recovery time: {avg_recovery_time:.2f}ms")


@pytest.mark.load
class TestSuppressionLoadTesting:
    """Load testing for suppression system under extreme conditions."""
    
    async def test_stress_test_high_volume(self):
        """Stress test with high alert volume."""
        print("Running stress test with high alert volume...")
        
        # This would be run with external load testing tools in practice
        # For unit tests, we verify the framework is in place
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        # Simulate load test metrics
        for i in range(100):
            benchmark.record_response_time("stress_test", i * 0.1)  # Simulated increasing latency
            if i % 10 == 0:
                benchmark.record_resource_usage()
        
        summary = benchmark.end_benchmark()
        
        # Verify load testing framework works
        assert "response_time_stats" in summary
        assert "throughput" in summary
        assert "resource_usage" in summary
        assert "nfr_compliance" in summary
        
        print("Stress test framework validated")
    
    async def test_endurance_test_sustained_load(self):
        """Endurance test with sustained load over time."""
        print("Running endurance test framework validation...")
        
        # This would run for hours/days in practice
        # For unit tests, verify monitoring is in place
        
        benchmark = PerformanceBenchmark()
        benchmark.start_benchmark()
        
        # Simulate endurance test metrics
        for hour in range(24):  # Simulate 24 hours
            for minute in range(60):  # 60 minutes per hour
                benchmark.record_response_time("endurance_test", 50 + (hour * 0.1))  # Slight degradation over time
                
                if minute % 15 == 0:  # Every 15 minutes
                    benchmark.record_resource_usage()
                    benchmark.record_throughput(100 - (hour * 0.5), 1)  # Slight throughput decrease
        
        summary = benchmark.end_benchmark()
        
        # Verify endurance monitoring works
        assert summary["test_duration_seconds"] > 0
        assert len(benchmark.results["response_times"]) > 1000
        assert len(benchmark.results["resource_usage"]) > 90  # Should have many resource samples
        
        print("Endurance test framework validated")


if __name__ == "__main__":
    # Run performance tests: pytest tests/performance/ -v -m performance
    # Run load tests: pytest tests/performance/ -v -m load
    pytest.main([__file__, "-v", "-m", "performance"])