#!/usr/bin/env python3
"""
Test script for Performance Optimization functionality validation.
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.models.partial_vital_signs import PartialVitalSigns
from src.services.audit import AuditLogger
from src.services.batch_processor import BatchNEWS2Processor, MemoryOptimizedBatchProcessor, BatchRequest
from src.services.patient_cache import PatientDataCache, ConnectionPool, PerformanceMonitor


def create_test_patient(patient_id: str, is_copd: bool = False) -> Patient:
    """Create a test patient."""
    return Patient(
        patient_id=patient_id,
        ward_id="WARD_A",
        bed_number="001",
        age=65,
        is_copd_patient=is_copd,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )


def create_test_vitals(patient_id: str, score_target: int = 0) -> VitalSigns:
    """Create test vital signs."""
    # Create normal vitals by default
    base_vitals = {
        'event_id': uuid4(),
        'patient_id': patient_id,
        'timestamp': datetime.now(timezone.utc),
        'respiratory_rate': 18,
        'spo2': 98,
        'on_oxygen': False,
        'temperature': 36.5,
        'systolic_bp': 120,
        'heart_rate': 75,
        'consciousness': ConsciousnessLevel.ALERT
    }
    
    # Adjust for target score if needed
    if score_target >= 2:
        base_vitals['on_oxygen'] = True  # 2 points
    if score_target >= 3:
        base_vitals['heart_rate'] = 95  # 1 additional point
    
    return VitalSigns(**base_vitals)


async def test_calculation_performance():
    """Test individual calculation performance meets <10ms requirement."""
    print("Testing calculation performance...")
    
    audit_logger = AuditLogger()
    processor = BatchNEWS2Processor(audit_logger)
    
    # Create test data
    patient = create_test_patient("PERF_001")
    vitals = create_test_vitals("PERF_001")
    
    # Test single calculation performance
    times = []
    for i in range(100):
        start_time = time.perf_counter()
        result = await processor.calculator.calculate_news2(vitals, patient)
        end_time = time.perf_counter()
        
        calc_time_ms = (end_time - start_time) * 1000
        times.append(calc_time_ms)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    p95_time = sorted(times)[94]  # 95th percentile
    
    # Verify performance requirements
    assert avg_time < 10.0, f"Average time {avg_time:.2f}ms exceeds 10ms requirement"
    assert max_time < 50.0, f"Maximum time {max_time:.2f}ms is excessive"
    
    print(f"   Average calculation time: {avg_time:.2f}ms")
    print(f"   95th percentile time: {p95_time:.2f}ms")
    print(f"   Maximum time: {max_time:.2f}ms")
    print("Individual calculation performance validated")
    
    return True


async def test_batch_processing():
    """Test memory-efficient batch processing for multiple patients."""
    print("\nTesting batch processing...")
    
    audit_logger = AuditLogger()
    processor = BatchNEWS2Processor(audit_logger, max_workers=10)
    
    # Create batch requests
    batch_size = 50
    requests = []
    
    for i in range(batch_size):
        patient = create_test_patient(f"BATCH_{i:03d}")
        vitals = create_test_vitals(f"BATCH_{i:03d}", score_target=i % 4)  # Vary scores
        
        request = BatchRequest(
            request_id=f"req_{i:03d}",
            patient=patient,
            vital_signs=vitals,
            priority=1 if i < 40 else 2  # Some high priority
        )
        requests.append(request)
    
    # Process batch
    start_time = time.perf_counter()
    results, stats = await processor.process_batch(requests)
    end_time = time.perf_counter()
    
    # Verify results
    assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
    assert stats.successful_calculations == batch_size, f"Expected {batch_size} successful calculations"
    assert stats.failed_calculations == 0, "Expected no failed calculations"
    
    # Verify performance
    batch_time_ms = (end_time - start_time) * 1000
    throughput = batch_size / (batch_time_ms / 1000)
    
    assert stats.average_time_per_calculation_ms < 10.0, "Individual calculations too slow in batch"
    assert throughput > 100, f"Batch throughput {throughput:.1f} calculations/sec is too low"
    
    print(f"   Batch size: {batch_size}")
    print(f"   Total batch time: {batch_time_ms:.2f}ms")
    print(f"   Average per calculation: {stats.average_time_per_calculation_ms:.2f}ms")
    print(f"   Throughput: {throughput:.1f} calculations/sec")
    print(f"   Peak memory: {stats.peak_memory_mb:.1f} MB")
    print(f"   Concurrent workers: {stats.concurrent_workers}")
    print("Batch processing validated")
    
    return True


async def test_memory_optimized_batch_processing():
    """Test memory-optimized batch processing for very large batches."""
    print("\nTesting memory-optimized batch processing...")
    
    audit_logger = AuditLogger()
    processor = MemoryOptimizedBatchProcessor(audit_logger, chunk_size=20)
    
    # Create large batch
    large_batch_size = 100
    requests = []
    
    for i in range(large_batch_size):
        patient = create_test_patient(f"LARGE_{i:03d}")
        vitals = create_test_vitals(f"LARGE_{i:03d}")
        
        request = BatchRequest(
            request_id=f"large_req_{i:03d}",
            patient=patient,
            vital_signs=vitals
        )
        requests.append(request)
    
    # Process large batch in chunks
    start_time = time.perf_counter()
    results, chunk_stats = await processor.process_large_batch(requests)
    end_time = time.perf_counter()
    
    # Verify results
    assert len(results) == large_batch_size, f"Expected {large_batch_size} results"
    
    total_successful = sum(stats.successful_calculations for stats in chunk_stats)
    assert total_successful == large_batch_size, "Expected all calculations to succeed"
    
    # Calculate overall stats
    batch_time_ms = (end_time - start_time) * 1000
    throughput = large_batch_size / (batch_time_ms / 1000)
    avg_chunk_time = sum(stats.average_time_per_calculation_ms for stats in chunk_stats) / len(chunk_stats)
    
    print(f"   Large batch size: {large_batch_size}")
    print(f"   Number of chunks: {len(chunk_stats)}")
    print(f"   Total processing time: {batch_time_ms:.2f}ms")
    print(f"   Average chunk calculation time: {avg_chunk_time:.2f}ms")
    print(f"   Overall throughput: {throughput:.1f} calculations/sec")
    print("Memory-optimized batch processing validated")
    
    return True


async def test_patient_data_caching():
    """Test caching for frequently accessed patient data."""
    print("\nTesting patient data caching...")
    
    cache = PatientDataCache(max_size=100, default_ttl=300)
    
    # Create test patients
    patients = [create_test_patient(f"CACHE_{i:03d}") for i in range(10)]
    
    # Test cache misses
    for patient in patients:
        result = await cache.get(patient.patient_id)
        assert result is None, "Expected cache miss for new patient"
    
    # Store patients in cache
    for patient in patients:
        await cache.put(patient)
    
    # Test cache hits
    hit_count = 0
    for patient in patients:
        result = await cache.get(patient.patient_id)
        if result is not None:
            hit_count += 1
            assert result.patient_id == patient.patient_id, "Cached patient data mismatch"
    
    assert hit_count == len(patients), f"Expected {len(patients)} cache hits, got {hit_count}"
    
    # Test cache statistics
    stats = cache.get_stats()
    assert stats.hits == len(patients), "Cache hit count incorrect"
    assert stats.size == len(patients), "Cache size incorrect"
    assert stats.hit_rate > 0.5, "Cache hit rate too low"
    
    # Test multiple get
    patient_ids = [p.patient_id for p in patients[:5]]
    multi_results = await cache.get_multiple(patient_ids)
    assert len(multi_results) == 5, "Multi-get returned wrong number of results"
    
    # Test cache eviction (fill beyond capacity)
    for i in range(100, 150):  # Add 50 more to trigger eviction
        extra_patient = create_test_patient(f"EXTRA_{i}")
        await cache.put(extra_patient)
    
    final_stats = cache.get_stats()
    assert final_stats.size <= 100, "Cache exceeded maximum size"
    assert final_stats.evictions > 0, "Expected some evictions"
    
    cache_info = cache.get_cache_info()
    print(f"   Cache hit rate: {cache_info['hit_rate']:.2%}")
    print(f"   Cache size: {cache_info['size']}/{cache_info['max_size']}")
    print(f"   Memory usage: {cache_info['memory_usage_mb']:.2f} MB")
    print(f"   Total evictions: {cache_info['evictions']}")
    print("Patient data caching validated")
    
    return True


async def test_connection_pooling():
    """Test connection pooling for database access optimization."""
    print("\nTesting connection pooling...")
    
    pool = ConnectionPool(pool_size=5)
    
    # Test basic acquire/release
    conn_id = await pool.acquire()
    assert isinstance(conn_id, int), "Connection ID should be integer"
    await pool.release(conn_id)
    
    # Test concurrent connection usage
    async def use_connection(duration_ms: int):
        conn_id = await pool.acquire(timeout=2.0)
        await asyncio.sleep(duration_ms / 1000)
        await pool.release(conn_id)
        return conn_id
    
    # Start multiple concurrent operations
    tasks = [use_connection(10) for _ in range(8)]  # More than pool size
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    
    assert len(results) == 8, "Expected all connection requests to complete"
    
    # Check pool stats
    pool_stats = pool.get_stats()
    assert pool_stats['total_requests'] >= 8, "Pool should have handled all requests"
    assert pool_stats['available'] == pool_stats['pool_size'], "All connections should be returned"
    
    concurrent_time_ms = (end_time - start_time) * 1000
    print(f"   Pool size: {pool_stats['pool_size']}")
    print(f"   Peak usage: {pool_stats['peak_usage']}")
    print(f"   Total requests handled: {pool_stats['total_requests']}")
    print(f"   Average wait time: {pool_stats['average_wait_time_ms']:.2f}ms")
    print(f"   Concurrent operations time: {concurrent_time_ms:.2f}ms")
    print("Connection pooling validated")
    
    return True


async def test_performance_monitoring():
    """Test performance monitoring and logging for calculation times."""
    print("\nTesting performance monitoring...")
    
    monitor = PerformanceMonitor(window_size=100)
    
    # Simulate various calculation times
    calculation_times = [0.5, 1.2, 0.8, 2.1, 0.9, 1.5, 0.7, 3.2, 1.1, 0.6]
    memory_usage = [45.2, 46.1, 44.8, 47.3, 45.9, 46.5, 44.2, 48.1, 46.0, 44.5]
    
    for calc_time, memory in zip(calculation_times, memory_usage):
        await monitor.record_calculation(calc_time, memory)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    
    assert summary['total_calculations'] == 10, "Expected 10 recorded calculations"
    assert summary['avg_calculation_time_ms'] > 0, "Average calculation time should be positive"
    assert summary['max_calculation_time_ms'] == max(calculation_times), "Max time incorrect"
    assert summary['throughput_per_second'] > 0, "Throughput should be positive"
    
    expected_avg_time = sum(calculation_times) / len(calculation_times)
    actual_avg_time = summary['avg_calculation_time_ms']
    assert abs(actual_avg_time - expected_avg_time) < 0.1, "Average time calculation incorrect"
    
    print(f"   Total calculations monitored: {summary['total_calculations']}")
    print(f"   Average calculation time: {summary['avg_calculation_time_ms']:.2f}ms")
    print(f"   95th percentile time: {summary['p95_calculation_time_ms']:.2f}ms")
    print(f"   Maximum calculation time: {summary['max_calculation_time_ms']:.2f}ms")
    print(f"   Throughput: {summary['throughput_per_second']:.1f} calculations/sec")
    print(f"   Average memory usage: {summary['avg_memory_mb']:.1f} MB")
    print(f"   Peak memory usage: {summary['peak_memory_mb']:.1f} MB")
    print("Performance monitoring validated")
    
    return True


async def test_concurrent_calculation_capability():
    """Test concurrent calculation capability using asyncio for scalability."""
    print("\nTesting concurrent calculation capability...")
    
    audit_logger = AuditLogger()
    processor = BatchNEWS2Processor(audit_logger, max_workers=20)
    
    # Create concurrent calculation requests
    concurrent_count = 100
    requests = []
    
    for i in range(concurrent_count):
        patient = create_test_patient(f"CONCURRENT_{i:03d}")
        vitals = create_test_vitals(f"CONCURRENT_{i:03d}")
        
        request = BatchRequest(
            request_id=f"concurrent_{i:03d}",
            patient=patient,
            vital_signs=vitals
        )
        requests.append(request)
    
    # Process all requests concurrently
    start_time = time.perf_counter()
    results, stats = await processor.process_batch(requests)
    end_time = time.perf_counter()
    
    # Verify concurrent processing
    concurrent_time_ms = (end_time - start_time) * 1000
    theoretical_sequential_time = stats.average_time_per_calculation_ms * concurrent_count
    speedup_factor = theoretical_sequential_time / concurrent_time_ms
    
    assert len(results) == concurrent_count, "All concurrent requests should complete"
    assert stats.successful_calculations == concurrent_count, "All calculations should succeed"
    assert speedup_factor > 5, f"Concurrent speedup {speedup_factor:.1f}x is too low"
    
    throughput = concurrent_count / (concurrent_time_ms / 1000)
    
    print(f"   Concurrent requests: {concurrent_count}")
    print(f"   Concurrent processing time: {concurrent_time_ms:.2f}ms")
    print(f"   Theoretical sequential time: {theoretical_sequential_time:.2f}ms")
    print(f"   Speedup factor: {speedup_factor:.1f}x")
    print(f"   Concurrent throughput: {throughput:.1f} calculations/sec")
    print(f"   Peak memory usage: {stats.peak_memory_mb:.1f} MB")
    print("Concurrent calculation capability validated")
    
    return True


async def main():
    """Run all performance optimization tests."""
    print("=== Performance Optimization Validation Tests ===\n")
    
    tests = [
        test_calculation_performance,
        test_batch_processing,
        test_memory_optimized_batch_processing,
        test_patient_data_caching,
        test_connection_pooling,
        test_performance_monitoring,
        test_concurrent_calculation_capability,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"FAILED: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAILED: {test.__name__} with error: {str(e)}")
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("All Performance Optimization tests passed!")
        print("Individual calculation algorithms optimized for <10ms")
        print("Memory-efficient batch processing implemented")
        print("Connection pooling and caching for patient data working")
        print("Concurrent calculation capability using asyncio validated")
        print("Performance monitoring and logging implemented")
    else:
        print("Some performance optimization tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)