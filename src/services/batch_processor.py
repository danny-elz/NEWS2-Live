import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

from ..models.patient import Patient
from ..models.vital_signs import VitalSigns
from ..models.partial_vital_signs import PartialVitalSigns
from ..models.news2 import NEWS2Result, CalculationError
from ..services.audit import AuditLogger
from ..services.news2_calculator import NEWS2Calculator


@dataclass
class BatchRequest:
    """Single calculation request in a batch."""
    request_id: str
    patient: Patient
    vital_signs: Union[VitalSigns, PartialVitalSigns]
    priority: int = 1  # 1 = normal, 2 = high, 3 = urgent


@dataclass
class BatchResult:
    """Result of batch processing."""
    request_id: str
    result: Optional[NEWS2Result] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class BatchStats:
    """Statistics from batch processing."""
    total_requests: int
    successful_calculations: int
    failed_calculations: int
    total_processing_time_ms: float
    average_time_per_calculation_ms: float
    peak_memory_mb: float
    concurrent_workers: int


class BatchNEWS2Processor:
    """
    High-performance batch processor for NEWS2 calculations.
    
    Optimized for memory efficiency and concurrent processing capability
    with performance monitoring and intelligent resource management.
    """
    
    def __init__(self, audit_logger: AuditLogger, max_workers: int = 10, max_batch_size: int = 100):
        self.audit_logger = audit_logger
        self.calculator = NEWS2Calculator(audit_logger)
        self.max_workers = max_workers
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.stats = {
            'total_processed': 0,
            'total_time_ms': 0.0,
            'peak_concurrent': 0,
            'memory_peak_mb': 0.0
        }
    
    async def process_batch(self, requests: List[BatchRequest]) -> Tuple[List[BatchResult], BatchStats]:
        """
        Process a batch of NEWS2 calculation requests concurrently.
        
        Args:
            requests: List of BatchRequest objects
            
        Returns:
            Tuple of (results, batch_stats)
        """
        if not requests:
            return [], BatchStats(0, 0, 0, 0.0, 0.0, 0.0, 0)
        
        if len(requests) > self.max_batch_size:
            raise ValueError(f"Batch size {len(requests)} exceeds maximum {self.max_batch_size}")
        
        start_time = time.perf_counter()
        
        # Sort requests by priority (urgent first)
        sorted_requests = sorted(requests, key=lambda x: x.priority, reverse=True)
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Process requests concurrently with semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        results = []
        
        async def process_single_request(request: BatchRequest) -> BatchResult:
            nonlocal peak_memory
            
            async with semaphore:
                request_start = time.perf_counter()
                try:
                    # Check memory usage
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Perform calculation based on vital signs type
                    if isinstance(request.vital_signs, PartialVitalSigns):
                        result = await self.calculator.calculate_partial_news2(
                            request.vital_signs, request.patient
                        )
                    else:
                        result = await self.calculator.calculate_news2(
                            request.vital_signs, request.patient
                        )
                    
                    request_time = (time.perf_counter() - request_start) * 1000
                    
                    return BatchResult(
                        request_id=request.request_id,
                        result=result,
                        processing_time_ms=request_time
                    )
                    
                except Exception as e:
                    request_time = (time.perf_counter() - request_start) * 1000
                    error_msg = f"Calculation failed: {str(e)}"
                    
                    self.logger.error(f"Batch request {request.request_id} failed: {error_msg}")
                    
                    return BatchResult(
                        request_id=request.request_id,
                        error=error_msg,
                        processing_time_ms=request_time
                    )
        
        # Execute all requests concurrently
        tasks = [process_single_request(req) for req in sorted_requests]
        results = await asyncio.gather(*tasks)
        
        # Calculate batch statistics
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        
        successful = sum(1 for r in results if r.result is not None)
        failed = sum(1 for r in results if r.error is not None)
        avg_time = sum(r.processing_time_ms for r in results) / len(results) if results else 0.0
        
        batch_stats = BatchStats(
            total_requests=len(requests),
            successful_calculations=successful,
            failed_calculations=failed,
            total_processing_time_ms=total_time,
            average_time_per_calculation_ms=avg_time,
            peak_memory_mb=peak_memory,
            concurrent_workers=min(self.max_workers, len(requests))
        )
        
        # Update global stats
        self.stats['total_processed'] += len(requests)
        self.stats['total_time_ms'] += total_time
        self.stats['peak_concurrent'] = max(self.stats['peak_concurrent'], len(tasks))
        self.stats['memory_peak_mb'] = max(self.stats['memory_peak_mb'], peak_memory)
        
        # Log batch completion
        self.logger.info(
            f"Batch processed: {len(requests)} requests, {successful} successful, "
            f"{failed} failed, {total_time:.2f}ms total, {avg_time:.2f}ms average"
        )
        
        return results, batch_stats
    
    async def process_streaming_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """
        Process requests with streaming results as they complete.
        Useful for real-time processing where results are needed as soon as available.
        
        Args:
            requests: List of BatchRequest objects
            
        Returns:
            List of BatchResult objects in completion order
        """
        if not requests:
            return []
        
        semaphore = asyncio.Semaphore(self.max_workers)
        completed_results = []
        
        async def process_and_yield(request: BatchRequest):
            async with semaphore:
                request_start = time.perf_counter()
                try:
                    if isinstance(request.vital_signs, PartialVitalSigns):
                        result = await self.calculator.calculate_partial_news2(
                            request.vital_signs, request.patient
                        )
                    else:
                        result = await self.calculator.calculate_news2(
                            request.vital_signs, request.patient
                        )
                    
                    request_time = (time.perf_counter() - request_start) * 1000
                    
                    batch_result = BatchResult(
                        request_id=request.request_id,
                        result=result,
                        processing_time_ms=request_time
                    )
                    
                    completed_results.append(batch_result)
                    return batch_result
                    
                except Exception as e:
                    request_time = (time.perf_counter() - request_start) * 1000
                    error_msg = f"Calculation failed: {str(e)}"
                    
                    batch_result = BatchResult(
                        request_id=request.request_id,
                        error=error_msg,
                        processing_time_ms=request_time
                    )
                    
                    completed_results.append(batch_result)
                    return batch_result
        
        # Start all tasks
        tasks = [process_and_yield(req) for req in requests]
        await asyncio.gather(*tasks)
        
        return completed_results
    
    def get_performance_stats(self) -> Dict:
        """Get overall performance statistics."""
        if self.stats['total_processed'] == 0:
            return {
                'total_processed': 0,
                'average_processing_time_ms': 0.0,
                'peak_concurrent_operations': 0,
                'peak_memory_usage_mb': 0.0,
                'throughput_per_second': 0.0
            }
        
        avg_time = self.stats['total_time_ms'] / self.stats['total_processed']
        throughput = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'total_processed': self.stats['total_processed'],
            'average_processing_time_ms': avg_time,
            'peak_concurrent_operations': self.stats['peak_concurrent'],
            'peak_memory_usage_mb': self.stats['memory_peak_mb'],
            'throughput_per_second': throughput
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_processed': 0,
            'total_time_ms': 0.0,
            'peak_concurrent': 0,
            'memory_peak_mb': 0.0
        }


class MemoryOptimizedBatchProcessor(BatchNEWS2Processor):
    """
    Memory-optimized version of batch processor for very large batches.
    
    Processes requests in smaller chunks to maintain low memory footprint
    while still providing high throughput.
    """
    
    def __init__(self, audit_logger: AuditLogger, chunk_size: int = 50, max_workers: int = 5):
        super().__init__(audit_logger, max_workers, chunk_size * 2)
        self.chunk_size = chunk_size
    
    async def process_large_batch(self, requests: List[BatchRequest]) -> Tuple[List[BatchResult], List[BatchStats]]:
        """
        Process a large batch by breaking it into memory-efficient chunks.
        
        Args:
            requests: List of BatchRequest objects (can be very large)
            
        Returns:
            Tuple of (all_results, chunk_stats_list)
        """
        if not requests:
            return [], []
        
        all_results = []
        chunk_stats = []
        
        # Process in chunks to maintain memory efficiency
        for i in range(0, len(requests), self.chunk_size):
            chunk = requests[i:i + self.chunk_size]
            
            self.logger.info(f"Processing chunk {i // self.chunk_size + 1}/{(len(requests) + self.chunk_size - 1) // self.chunk_size}")
            
            chunk_results, stats = await self.process_batch(chunk)
            all_results.extend(chunk_results)
            chunk_stats.append(stats)
            
            # Allow brief pause between chunks for garbage collection
            await asyncio.sleep(0.001)
        
        self.logger.info(f"Completed large batch processing: {len(requests)} total requests")
        
        return all_results, chunk_stats