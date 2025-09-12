#!/usr/bin/env python3
import pytest
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.streaming.kafka_config import KafkaConfig, TopicConfig
from src.streaming.idempotency_manager import IdempotencyManager
from src.streaming.error_handler import StreamErrorHandler
from src.streaming.monitoring import StreamMonitor
from src.streaming.stream_processor import StreamProcessor, VitalSignsRecord
from src.models.vital_signs import VitalSigns, ConsciousnessLevel


class TestStreamProcessingIntegration:
    """Integration tests for stream processing components."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock()
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.info = AsyncMock(return_value={
            'used_memory': 1024000,
            'maxmemory': 10240000,
            'connected_clients': 5
        })
        redis_mock.close = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def mock_kafka_config(self):
        """Mock Kafka configuration."""
        kafka_mock = Mock()
        kafka_mock.initialize = AsyncMock()
        kafka_mock.create_topics = AsyncMock()
        kafka_mock.list_topics = AsyncMock(return_value=[
            'vital_signs_input',
            'news2_results',
            'news2_alerts',
            'stream_errors',
            'dead_letter_queue'
        ])
        kafka_mock.health_check = AsyncMock(return_value=True)
        kafka_mock.close = Mock()
        return kafka_mock
    
    @pytest.fixture
    def sample_vital_signs_record(self):
        """Sample vital signs record for testing."""
        return VitalSignsRecord(
            event_id=str(uuid4()),
            patient_id="INT_TEST_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            respiratory_rate=20,
            spo2=94,
            on_oxygen=False,
            temperature=37.2,
            systolic_bp=130,
            heart_rate=85,
            consciousness=ConsciousnessLevel.ALERT.value,
            data_source="device",
            quality_flags={}
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_kafka_redis_integration(self, mock_redis_client, mock_kafka_config):
        """Test end-to-end integration between Kafka and Redis components."""
        
        # Initialize idempotency manager with Redis
        idempotency_manager = IdempotencyManager(mock_redis_client)
        
        # Test idempotency workflow
        event_id = str(uuid4())
        patient_id = "INT_TEST_001"
        
        # Check initial state (not duplicate)
        is_duplicate_1 = await idempotency_manager.is_duplicate(event_id, patient_id)
        assert is_duplicate_1 == False
        
        # Mark as processed
        await idempotency_manager.mark_processed(event_id, patient_id, metadata={
            'news2_score': 4,
            'processing_time': 0.05
        })
        
        # Verify Redis interactions
        mock_redis_client.exists.assert_called()
        mock_redis_client.setex.assert_called()
        
        # Check duplicate detection
        mock_redis_client.exists.return_value = True
        is_duplicate_2 = await idempotency_manager.is_duplicate(event_id, patient_id)
        assert is_duplicate_2 == True
    
    @pytest.mark.asyncio
    async def test_stream_processor_service_initialization(self, mock_redis_client):
        """Test stream processor service initialization."""
        
        processor = StreamProcessor(
            broker_url='kafka://localhost:9092',
            redis_url='redis://localhost:6379',
            app_id='test_integration_processor'
        )
        
        # Mock the Redis client creation
        with patch('redis.asyncio.from_url', return_value=mock_redis_client):
            await processor.initialize_services()
            
            assert processor._redis_client is not None
            assert processor._idempotency_manager is not None
            assert processor._error_handler is not None
            assert processor._news2_calculator is not None
            
            # Verify Redis ping was called during initialization
            mock_redis_client.ping.assert_called()
    
    @pytest.mark.asyncio
    async def test_vital_signs_to_news2_processing_pipeline(self, sample_vital_signs_record):
        """Test complete vital signs to NEWS2 processing pipeline."""
        
        processor = StreamProcessor()
        
        # Mock dependencies
        with patch.object(processor, '_redis_client'), \
             patch.object(processor, '_idempotency_manager') as mock_idempotency, \
             patch.object(processor, '_news2_calculator') as mock_calculator, \
             patch.object(processor, '_patient_registry') as mock_registry:
            
            # Configure mocks
            mock_idempotency.is_duplicate = AsyncMock(return_value=False)
            mock_idempotency.mark_processed = AsyncMock()
            
            mock_registry.get_patient_state = AsyncMock(return_value=Mock(
                clinical_flags={'is_copd_patient': False}
            ))
            
            mock_calculator.calculate_news2 = AsyncMock(return_value=Mock(
                total_score=4,
                individual_scores={'rr': 2, 'spo2': 1, 'temp': 1, 'hr': 0, 'sbp': 0, 'consciousness': 0, 'oxygen': 0},
                risk_category=Mock(value='MEDIUM'),
                scale_used=1,
                monitoring_frequency='4-6 hourly',
                red_flags=[]
            ))
            
            # Test vital signs conversion
            vital_signs = await processor._convert_to_vital_signs(sample_vital_signs_record)
            
            assert isinstance(vital_signs, VitalSigns)
            assert vital_signs.patient_id == "INT_TEST_001"
            assert vital_signs.respiratory_rate == 20
            assert vital_signs.spo2 == 94
            
            # Test NEWS2 calculation with context
            news2_result = await processor._calculate_news2_with_context(vital_signs)
            
            assert news2_result.total_score == 4
            assert news2_result.risk_category.value == 'MEDIUM'
            
            # Verify service interactions
            mock_registry.get_patient_state.assert_called_once_with("INT_TEST_001")
            mock_calculator.calculate_news2.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_copd_patient_processing_integration(self, sample_vital_signs_record):
        """Test COPD patient processing integration."""
        
        processor = StreamProcessor()
        
        # Mock dependencies for COPD patient
        with patch.object(processor, '_patient_registry') as mock_registry, \
             patch.object(processor, '_news2_calculator') as mock_calculator:
            
            # Configure COPD patient state
            mock_registry.get_patient_state = AsyncMock(return_value=Mock(
                clinical_flags={'is_copd_patient': True}
            ))
            
            mock_calculator.calculate_news2 = AsyncMock(return_value=Mock(
                total_score=2,  # Different score due to COPD Scale 2
                individual_scores={'rr': 2, 'spo2': 0, 'temp': 1, 'hr': 0, 'sbp': 0, 'consciousness': 0, 'oxygen': 0},
                risk_category=Mock(value='LOW'),
                scale_used=2,  # COPD scale
                monitoring_frequency='routine',
                red_flags=[]
            ))
            
            # Convert and process
            vital_signs = await processor._convert_to_vital_signs(sample_vital_signs_record)
            news2_result = await processor._calculate_news2_with_context(vital_signs)
            
            # Verify COPD-specific processing
            assert news2_result.scale_used == 2
            assert news2_result.total_score == 2  # Lower score due to COPD SpO2 scaling
            
            # Verify calculator was called with is_copd_patient=True
            mock_calculator.calculate_news2.assert_called_once()
            call_args = mock_calculator.calculate_news2.call_args
            assert call_args[1]['is_copd_patient'] == True
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_redis_client, sample_vital_signs_record):
        """Test error handling integration across components."""
        
        # Create error handler with mock dead letter topic
        mock_dead_letter_topic = Mock()
        mock_dead_letter_topic.send = AsyncMock()
        
        error_handler = StreamErrorHandler(
            dead_letter_topic=mock_dead_letter_topic,
            max_retries=2
        )
        
        # Test error processing
        test_error = ValueError("Invalid vital signs: temperature out of range")
        
        await error_handler.handle_processing_error(
            original_event=sample_vital_signs_record,
            error=test_error,
            attempt=1
        )
        
        # Verify error was classified correctly
        error_type = error_handler._classify_error(test_error)
        assert error_type.value == "validation_error"
        
        # Check error statistics
        stats = error_handler.get_error_stats()
        assert stats['total_errors'] == 1
        assert 'validation_error' in stats['errors_by_type']
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, mock_redis_client):
        """Test monitoring integration with all components."""
        
        # Initialize stream monitor
        monitor = StreamMonitor(mock_redis_client)
        await monitor.start()
        
        # Record various processing events
        monitor.record_processing_event(
            event_id=str(uuid4()),
            patient_id="MONITOR_TEST_001",
            processing_duration=0.045,
            news2_score=3,
            is_copd=False
        )
        
        monitor.record_processing_event(
            event_id=str(uuid4()),
            patient_id="MONITOR_TEST_002",
            processing_duration=0.078,
            news2_score=7,
            is_copd=True
        )
        
        # Record an error event
        monitor.record_processing_event(
            event_id=str(uuid4()),
            patient_id="MONITOR_TEST_003",
            processing_duration=0.0,
            error=Exception("Processing failed")
        )
        
        # Get dashboard data
        dashboard_data = await monitor.get_dashboard_data()
        
        assert dashboard_data['monitoring_active'] == True
        assert 'health' in dashboard_data
        assert 'metrics' in dashboard_data
        
        # Add an alert
        monitor.add_alert(
            level='warning',
            message='High processing latency detected',
            component='stream_processor',
            details={'avg_latency_ms': 85.5}
        )
        
        # Verify alert was added
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]['level'] == 'warning'
        assert monitor.alerts[0]['component'] == 'stream_processor'
        
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_across_restarts(self, mock_redis_client):
        """Test duplicate detection persists across processor restarts."""
        
        # First processor instance
        manager_1 = IdempotencyManager(mock_redis_client)
        
        event_id = str(uuid4())
        patient_id = "RESTART_TEST_001"
        
        # Process event with first instance
        await manager_1.mark_processed(event_id, patient_id)
        
        # Simulate processor restart with new instance
        manager_2 = IdempotencyManager(mock_redis_client)
        
        # Configure Redis to return that the event exists
        mock_redis_client.exists.return_value = True
        
        # Check duplicate detection with second instance
        is_duplicate = await manager_2.is_duplicate(event_id, patient_id)
        
        assert is_duplicate == True
        
        # Verify persistence across instances
        mock_redis_client.exists.assert_called()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self, mock_redis_client):
        """Test performance metrics collection integration."""
        
        monitor = StreamMonitor(mock_redis_client)
        
        # Simulate high-frequency processing
        start_time = time.time()
        event_count = 100
        
        for i in range(event_count):
            processing_duration = 0.01 + (i * 0.001)  # Gradually increasing duration
            news2_score = (i % 10) + 1  # Scores 1-10
            is_copd = i % 5 == 0  # Every 5th patient is COPD
            
            monitor.record_processing_event(
                event_id=str(uuid4()),
                patient_id=f"PERF_TEST_{i:03d}",
                processing_duration=processing_duration,
                news2_score=news2_score,
                is_copd=is_copd
            )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Update throughput metrics
        throughput = event_count / total_duration
        monitor.metrics_collector.update_throughput(throughput, "vital_signs")
        
        # Get Prometheus metrics export
        prometheus_data = monitor.get_metrics_export()
        
        assert isinstance(prometheus_data, str)
        assert "stream_events_processed_total" in prometheus_data
        assert "stream_processing_duration_seconds" in prometheus_data
        assert "news2_scores_calculated" in prometheus_data
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handling."""
        
        mock_dead_letter_topic = Mock()
        mock_dead_letter_topic.send = AsyncMock()
        
        error_handler = StreamErrorHandler(
            dead_letter_topic=mock_dead_letter_topic,
            max_retries=3
        )
        
        # Test circuit breaker functionality
        service_name = "news2_calculation"
        circuit_breaker = error_handler.circuit_breakers[service_name]
        
        # Function that always fails
        async def failing_service():
            raise Exception("Service unavailable")
        
        # Trip the circuit breaker
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await error_handler.handle_with_circuit_breaker(
                    service_name, failing_service
                )
        
        # Verify circuit is now open
        assert circuit_breaker.state.value == "open"
        
        # Next call should be rejected immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await error_handler.handle_with_circuit_breaker(
                service_name, failing_service
            )
    
    @pytest.mark.asyncio
    async def test_stream_processor_health_check_integration(self, mock_redis_client):
        """Test stream processor health check integration."""
        
        processor = StreamProcessor()
        
        # Mock services
        with patch.object(processor, '_redis_client', mock_redis_client), \
             patch.object(processor, '_idempotency_manager') as mock_idempotency:
            
            # Configure health check responses
            mock_idempotency.health_check = AsyncMock(return_value={
                'status': 'healthy',
                'redis_connected': True
            })
            
            # Perform health check
            health_data = await processor.health_check()
            
            assert health_data['status'] == 'healthy'
            assert health_data['app_id'] == processor.app_id
            assert 'timestamp' in health_data
            assert 'processing_stats' in health_data
            
            # Verify Redis ping was called
            mock_redis_client.ping.assert_called()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_integration(self, mock_redis_client):
        """Test graceful shutdown integration across components."""
        
        # Initialize components
        processor = StreamProcessor()
        monitor = StreamMonitor(mock_redis_client)
        idempotency_manager = IdempotencyManager(mock_redis_client)
        
        # Start monitoring
        await monitor.start()
        
        # Simulate some processing
        processor._processed_count = 500
        processor._error_count = 5
        
        # Perform graceful shutdown
        await processor.shutdown()
        await monitor.stop()
        await idempotency_manager.close()
        
        # Verify cleanup was performed
        mock_redis_client.close.assert_called()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--asyncio-mode=auto"
    ])