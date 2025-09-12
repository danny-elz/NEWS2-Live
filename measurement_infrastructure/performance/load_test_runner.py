#!/usr/bin/env python3
"""
Load testing framework for Epic 1 performance validation.

This framework validates:
- 1,000+ events/second sustained throughput
- <100ms P95 latency under load
- System behavior under stress conditions
- Memory and CPU usage patterns
- Error handling under high load
"""

import asyncio
import json
import time
import statistics
import psutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from uuid import uuid4
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import redis
    from kafka import KafkaProducer
    import prometheus_client
    from prometheus_client import CollectorRegistry, Gauge, Histogram, Counter
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install kafka-python redis prometheus-client psutil")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    target_throughput: int = 1000  # events/second
    test_duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 30
    kafka_servers: List[str] = None
    redis_url: str = "redis://localhost:6379"
    topic_name: str = "vital_signs_input"
    patient_count: int = 100
    copd_patient_percentage: float = 0.15  # 15% COPD patients
    
    def __post_init__(self):
        if self.kafka_servers is None:
            self.kafka_servers = ['localhost:9092']


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    timestamp: datetime
    events_sent: int
    events_processed: int
    throughput_achieved: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    kafka_lag_ms: float


@dataclass
class LoadTestResult:
    """Complete load test results."""
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    metrics_history: List[PerformanceMetrics]
    summary_stats: Dict[str, Any]
    success: bool
    issues_found: List[str]


class VitalSignsGenerator:
    """Generates realistic vital signs data for load testing."""
    
    def __init__(self, patient_count: int, copd_percentage: float):
        self.patient_count = patient_count
        self.copd_percentage = copd_percentage
        self._generate_patient_profiles()
    
    def _generate_patient_profiles(self):
        """Generate patient profiles with COPD distribution."""
        import random
        
        self.patients = []
        copd_count = int(self.patient_count * self.copd_percentage)
        
        for i in range(self.patient_count):
            is_copd = i < copd_count
            patient = {
                'patient_id': f"LOAD_TEST_{i:04d}",
                'is_copd': is_copd,
                'age': random.randint(45, 85) if is_copd else random.randint(25, 80),
                'baseline_vitals': self._generate_baseline_vitals(is_copd)
            }
            self.patients.append(patient)
    
    def _generate_baseline_vitals(self, is_copd: bool) -> Dict[str, Any]:
        """Generate baseline vital signs for a patient."""
        import random
        
        if is_copd:
            # COPD patients tend to have lower SpO2 baselines
            return {
                'rr_base': random.randint(16, 22),
                'spo2_base': random.randint(88, 94),
                'temp_base': random.uniform(36.2, 36.8),
                'hr_base': random.randint(70, 90),
                'sbp_base': random.randint(120, 150)
            }
        else:
            return {
                'rr_base': random.randint(12, 18),
                'spo2_base': random.randint(96, 99),
                'temp_base': random.uniform(36.1, 36.7),
                'hr_base': random.randint(60, 80),
                'sbp_base': random.randint(110, 130)
            }
    
    def generate_vital_signs_event(self) -> Dict[str, Any]:
        """Generate a realistic vital signs event."""
        import random
        
        patient = random.choice(self.patients)
        baseline = patient['baseline_vitals']
        
        # Add some variation to baseline values
        vital_signs = {
            'event_id': str(uuid4()),
            'patient_id': patient['patient_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'respiratory_rate': max(8, min(35, baseline['rr_base'] + random.randint(-3, 3))),
            'spo2': max(75, min(100, baseline['spo2_base'] + random.randint(-2, 2))),
            'on_oxygen': random.random() < 0.1,  # 10% on oxygen
            'temperature': max(35.0, min(42.0, baseline['temp_base'] + random.uniform(-1.0, 1.5))),
            'systolic_bp': max(60, min(220, baseline['sbp_base'] + random.randint(-15, 20))),
            'heart_rate': max(30, min(180, baseline['hr_base'] + random.randint(-10, 15))),
            'consciousness': random.choices(['ALERT', 'CONFUSED', 'VOICE', 'PAIN', 'UNRESPONSIVE'], 
                                          weights=[85, 8, 4, 2, 1])[0],
            'is_copd_patient': patient['is_copd'],
            'data_source': 'load_test',
            'quality_flags': {
                'is_manual_entry': random.random() < 0.05,
                'has_artifacts': random.random() < 0.02,
                'confidence': random.uniform(0.9, 1.0)
            }
        }
        
        return vital_signs


class PerformanceMonitor:
    """Monitors system performance during load testing."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self.monitoring_active = False
        self.metrics_history = []
        
    def _setup_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        self.events_sent_total = Counter(
            'load_test_events_sent_total',
            'Total events sent during load test',
            registry=self.registry
        )
        
        self.events_processed_total = Counter(
            'load_test_events_processed_total', 
            'Total events processed during load test',
            registry=self.registry
        )
        
        self.processing_latency = Histogram(
            'load_test_processing_latency_seconds',
            'Processing latency in seconds',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'load_test_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'load_test_memory_usage_mb',
            'Memory usage in MB', 
            registry=self.registry
        )
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous performance monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=10)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Update Prometheus metrics
                self.system_cpu_usage.set(metrics.cpu_usage_percent)
                self.system_memory_usage.set(metrics.memory_usage_mb)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            events_sent=0,  # Will be updated by load tester
            events_processed=0,  # Will be updated by load tester  
            throughput_achieved=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            error_rate=0.0,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / (1024 * 1024),
            kafka_lag_ms=0.0
        )
    
    def record_event_sent(self):
        """Record an event sent."""
        self.events_sent_total.inc()
    
    def record_event_processed(self, latency_seconds: float):
        """Record an event processed with latency."""
        self.events_processed_total.inc()
        self.processing_latency.observe(latency_seconds)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_usage_percent for m in self.metrics_history]
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'monitoring_duration_seconds': len(self.metrics_history) * 5,
            'total_metrics_collected': len(self.metrics_history)
        }


class LoadTester:
    """Main load testing orchestrator."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.producer = None
        self.redis_client = None
        self.monitor = PerformanceMonitor()
        self.generator = VitalSignsGenerator(
            config.patient_count,
            config.copd_patient_percentage
        )
        
        # Test tracking
        self.events_sent = 0
        self.events_failed = 0
        self.latency_measurements = []
        self.start_time = None
        self.end_time = None
    
    async def setup(self):
        """Setup connections and resources."""
        try:
            # Setup Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_servers,
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=5,
                compression_type='gzip'
            )
            
            # Setup Redis client
            self.redis_client = redis.Redis.from_url(self.config.redis_url)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            
            logger.info("Load tester setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup load tester: {e}")
            raise
    
    async def run_load_test(self) -> LoadTestResult:
        """Run complete load test with ramp-up and sustained load."""
        logger.info(f"Starting load test: {self.config.target_throughput} events/second for {self.config.test_duration_seconds}s")
        
        self.start_time = datetime.now(timezone.utc)
        self.monitor.start_monitoring()
        
        try:
            # Phase 1: Ramp up
            await self._ramp_up_phase()
            
            # Phase 2: Sustained load
            await self._sustained_load_phase()
            
            # Phase 3: Cool down
            await self._cool_down_phase()
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            raise
        
        finally:
            self.end_time = datetime.now(timezone.utc)
            self.monitor.stop_monitoring()
            await self.cleanup()
        
        return self._generate_results()
    
    async def _ramp_up_phase(self):
        """Gradually increase load to target throughput."""
        logger.info(f"Ramp-up phase: {self.config.ramp_up_seconds} seconds")
        
        ramp_steps = 10
        step_duration = self.config.ramp_up_seconds / ramp_steps
        
        for step in range(ramp_steps):
            target_rate = self.config.target_throughput * (step + 1) / ramp_steps
            await self._send_events_at_rate(target_rate, step_duration)
            
            logger.info(f"Ramp-up step {step + 1}/{ramp_steps}: {target_rate:.0f} events/second")
    
    async def _sustained_load_phase(self):
        """Run sustained load at target throughput."""
        logger.info(f"Sustained load phase: {self.config.target_throughput} events/second for {self.config.test_duration_seconds}s")
        
        await self._send_events_at_rate(
            self.config.target_throughput,
            self.config.test_duration_seconds
        )
    
    async def _cool_down_phase(self):
        """Cool down phase to let system process remaining events."""
        logger.info("Cool down phase: 30 seconds")
        
        # Gradually reduce load
        for step in range(5, 0, -1):
            rate = self.config.target_throughput * step / 5
            await self._send_events_at_rate(rate, 6)  # 6 seconds per step
    
    async def _send_events_at_rate(self, events_per_second: float, duration_seconds: float):
        """Send events at specified rate for given duration."""
        target_interval = 1.0 / events_per_second if events_per_second > 0 else 1.0
        end_time = time.time() + duration_seconds
        last_sent_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            while time.time() < end_time:
                current_time = time.time()
                
                if current_time - last_sent_time >= target_interval:
                    # Send event
                    event = self.generator.generate_vital_signs_event()
                    
                    try:
                        future = executor.submit(self._send_event, event)
                        # Don't wait for completion to maintain throughput
                        
                        self.monitor.record_event_sent()
                        self.events_sent += 1
                        last_sent_time = current_time
                        
                    except Exception as e:
                        self.events_failed += 1
                        logger.warning(f"Failed to send event: {e}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)
    
    def _send_event(self, event: Dict[str, Any]) -> bool:
        """Send single event to Kafka."""
        try:
            send_time = time.time()
            
            future = self.producer.send(
                self.config.topic_name,
                key=event['patient_id'],
                value=event
            )
            
            # Wait for send completion to measure latency
            result = future.get(timeout=10)
            
            latency = (time.time() - send_time) * 1000  # ms
            self.latency_measurements.append(latency)
            self.monitor.record_event_processed(latency / 1000)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send event to Kafka: {e}")
            return False
    
    def _generate_results(self) -> LoadTestResult:
        """Generate comprehensive load test results."""
        duration = (self.end_time - self.start_time).total_seconds()
        throughput_achieved = self.events_sent / duration if duration > 0 else 0
        error_rate = self.events_failed / max(self.events_sent, 1)
        
        # Calculate latency percentiles
        if self.latency_measurements:
            latencies = sorted(self.latency_measurements)
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0
        
        # Determine success criteria
        success = (
            throughput_achieved >= self.config.target_throughput * 0.95 and  # Within 5% of target
            p95 < 100.0 and  # P95 latency < 100ms
            error_rate < 0.01  # Error rate < 1%
        )
        
        issues_found = []
        if throughput_achieved < self.config.target_throughput * 0.95:
            issues_found.append(f"Throughput below target: {throughput_achieved:.1f} < {self.config.target_throughput * 0.95:.1f}")
        
        if p95 >= 100.0:
            issues_found.append(f"P95 latency above 100ms: {p95:.1f}ms")
        
        if error_rate >= 0.01:
            issues_found.append(f"Error rate too high: {error_rate:.2%}")
        
        summary_stats = {
            'throughput_achieved': throughput_achieved,
            'target_throughput': self.config.target_throughput,
            'throughput_percentage': (throughput_achieved / self.config.target_throughput * 100) if self.config.target_throughput > 0 else 0,
            'total_events_sent': self.events_sent,
            'total_events_failed': self.events_failed,
            'error_rate': error_rate,
            'latency_p50_ms': p50,
            'latency_p95_ms': p95,
            'latency_p99_ms': p99,
            'test_duration_seconds': duration,
            'system_metrics': self.monitor.get_metrics_summary()
        }
        
        return LoadTestResult(
            config=self.config,
            start_time=self.start_time,
            end_time=self.end_time,
            total_duration_seconds=duration,
            metrics_history=self.monitor.metrics_history,
            summary_stats=summary_stats,
            success=success,
            issues_found=issues_found
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.producer:
            self.producer.close()
        
        if self.redis_client:
            self.redis_client.close()


async def run_performance_validation():
    """Run comprehensive performance validation for Epic 1."""
    print("=" * 60)
    print("NEWS2 Live - Epic 1 Performance Validation")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        LoadTestConfig(
            target_throughput=500,
            test_duration_seconds=120,
            ramp_up_seconds=20
        ),
        LoadTestConfig(
            target_throughput=1000,
            test_duration_seconds=300,
            ramp_up_seconds=30
        ),
        LoadTestConfig(
            target_throughput=1500,
            test_duration_seconds=180,
            ramp_up_seconds=30
        )
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\nRunning Test {i}/{len(test_configs)}: {config.target_throughput} events/second")
        print("-" * 40)
        
        tester = LoadTester(config)
        
        try:
            await tester.setup()
            result = await tester.run_load_test()
            all_results.append(result)
            
            # Print immediate results
            print(f"✅ Test {i} {'PASSED' if result.success else 'FAILED'}")
            print(f"   Throughput: {result.summary_stats['throughput_achieved']:.1f} events/second")
            print(f"   P95 Latency: {result.summary_stats['latency_p95_ms']:.1f}ms")
            print(f"   Error Rate: {result.summary_stats['error_rate']:.2%}")
            
            if result.issues_found:
                print("   Issues:")
                for issue in result.issues_found:
                    print(f"   - {issue}")
            
        except Exception as e:
            print(f"❌ Test {i} FAILED with error: {e}")
            
        # Wait between tests
        if i < len(test_configs):
            print("   Waiting 30 seconds before next test...")
            await asyncio.sleep(30)
    
    # Generate final report
    generate_performance_report(all_results)
    
    return all_results


def generate_performance_report(results: List[LoadTestResult]):
    """Generate comprehensive performance test report."""
    print("\n" + "=" * 60)
    print("PERFORMANCE VALIDATION REPORT")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PERFORMANCE TESTS PASSED!")
        print("✅ Epic 1 meets all performance requirements")
    else:
        print("⚠️  Some performance tests failed")
        print("❌ Epic 1 performance needs optimization")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for i, result in enumerate(results, 1):
        stats = result.summary_stats
        print(f"\nTest {i}: {result.config.target_throughput} events/second")
        print(f"  Status: {'✅ PASSED' if result.success else '❌ FAILED'}")
        print(f"  Throughput: {stats['throughput_achieved']:.1f} events/second ({stats['throughput_percentage']:.1f}% of target)")
        print(f"  Latency P50: {stats['latency_p50_ms']:.1f}ms")
        print(f"  Latency P95: {stats['latency_p95_ms']:.1f}ms")
        print(f"  Latency P99: {stats['latency_p99_ms']:.1f}ms")
        print(f"  Error Rate: {stats['error_rate']:.2%}")
        print(f"  Duration: {stats['test_duration_seconds']:.1f}s")
        
        if result.issues_found:
            print("  Issues Found:")
            for issue in result.issues_found:
                print(f"    - {issue}")
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0,
        'results': [asdict(r) for r in results]
    }
    
    with open('measurement_infrastructure/performance/load_test_results.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: measurement_infrastructure/performance/load_test_results.json")


if __name__ == "__main__":
    try:
        asyncio.run(run_performance_validation())
    except KeyboardInterrupt:
        print("\nPerformance validation interrupted by user")
    except Exception as e:
        print(f"\nPerformance validation failed: {e}")
        import traceback
        traceback.print_exc()