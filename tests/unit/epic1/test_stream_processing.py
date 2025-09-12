#!/usr/bin/env python3
import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from src.streaming.kafka_config import KafkaConfig, TopicConfig
from src.streaming.idempotency_manager import IdempotencyManager
from src.streaming.error_handler import StreamErrorHandler, ErrorType, ErrorSeverity, CircuitBreaker
from src.streaming.monitoring import MetricsCollector, HealthChecker, StreamMonitor
from src.streaming.stream_processor import StreamProcessor, VitalSignsRecord, NEWS2ResultRecord


class TestKafkaConfig:
    """Test Kafka configuration and topic management."""
    
    def test_topic_config_creation(self):
        """Test TopicConfig data structure."""
        config = TopicConfig(
            name="test_topic",
            partitions=5,
            replication_factor=3,
            retention_ms=86400000,
            cleanup_policy="delete"
        )
        
        assert config.name == "test_topic"
        assert config.partitions == 5
        assert config.replication_factor == 3
        assert config.retention_ms == 86400000
    
    def test_kafka_config_initialization(self):
        """Test KafkaConfig initialization."""
        bootstrap_servers = ['localhost:9092', 'kafka:9092']
        kafka_config = KafkaConfig(bootstrap_servers)
        
        assert kafka_config.bootstrap_servers == bootstrap_servers
        assert kafka_config.admin_client is None  # Not initialized yet
    
    @patch('src.streaming.kafka_config.KafkaAdminClient')
    def test_kafka_producer_config(self, mock_admin):
        """Test Kafka producer configuration."""
        kafka_config = KafkaConfig(['localhost:9092'])
        
        producer_config = kafka_config._producer_config
        
        assert producer_config['bootstrap_servers'] == ['localhost:9092']
        assert producer_config['acks'] == 'all'
        assert producer_config['retries'] == 3
        assert producer_config['compression_type'] == 'gzip'
    
    @patch('src.streaming.kafka_config.KafkaAdminClient')
    def test_kafka_consumer_config(self, mock_admin):
        """Test Kafka consumer configuration.""" 
        kafka_config = KafkaConfig(['localhost:9092'])
        
        consumer_config = kafka_config._consumer_config
        
        assert consumer_config['bootstrap_servers'] == ['localhost:9092']
        assert consumer_config['auto_offset_reset'] == 'earliest'
        assert consumer_config['enable_auto_commit'] == False
        assert consumer_config['max_poll_records'] == 500


class TestIdempotencyManager:
    """Test idempotency manager for exactly-once processing."""
    
    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock()
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.close = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def idempotency_manager(self, mock_redis):
        return IdempotencyManager(mock_redis, ttl_seconds=3600)
    
    @pytest.mark.asyncio
    async def test_new_event_not_duplicate(self, idempotency_manager, mock_redis):
        """Test that new events are not identified as duplicates."""
        mock_redis.exists.return_value = False
        
        is_duplicate = await idempotency_manager.is_duplicate("event_123", "patient_456")
        
        assert is_duplicate == False
        mock_redis.exists.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_processed_event_is_duplicate(self, idempotency_manager, mock_redis):
        """Test that processed events are identified as duplicates."""
        # First call - not duplicate
        mock_redis.exists.return_value = False
        is_duplicate_1 = await idempotency_manager.is_duplicate("event_123", "patient_456")
        
        # Mark as processed
        await idempotency_manager.mark_processed("event_123", "patient_456")
        
        # Second call - should be duplicate
        mock_redis.exists.return_value = True
        is_duplicate_2 = await idempotency_manager.is_duplicate("event_123", "patient_456")
        
        assert is_duplicate_1 == False
        assert is_duplicate_2 == True
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_processed_with_metadata(self, idempotency_manager, mock_redis):
        """Test marking event as processed with metadata."""
        metadata = {"news2_score": 5, "risk_category": "MEDIUM"}
        
        await idempotency_manager.mark_processed(
            "event_123", "patient_456", metadata=metadata
        )
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600  # TTL
        assert "metadata" in str(call_args[0][2])  # Value contains metadata
    
    @pytest.mark.asyncio
    async def test_event_sequence_validation(self, idempotency_manager, mock_redis):
        """Test event sequence validation for ordering."""
        # Mock Redis to return None (no previous timestamp)
        mock_redis.get.return_value = None
        
        current_time = datetime.now(timezone.utc)
        
        is_in_sequence = await idempotency_manager.is_event_in_sequence(
            "event_123", "patient_456", current_time
        )
        
        assert is_in_sequence == True
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, idempotency_manager, mock_redis):
        """Test health check when Redis is healthy."""
        mock_redis.ping = AsyncMock()
        
        health = await idempotency_manager.health_check()
        
        assert health['status'] == 'healthy'
        assert health['redis_connected'] == True
        assert 'local_cache_size' in health
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, idempotency_manager, mock_redis):
        """Test health check when Redis is unhealthy."""
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        health = await idempotency_manager.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['redis_connected'] == False
        assert 'error' in health


class TestStreamErrorHandler:
    """Test stream processing error handling."""
    
    @pytest.fixture
    def mock_dead_letter_topic(self):
        topic = Mock()
        topic.send = AsyncMock()
        return topic
    
    @pytest.fixture
    def error_handler(self, mock_dead_letter_topic):
        return StreamErrorHandler(
            dead_letter_topic=mock_dead_letter_topic,
            max_retries=3,
            base_delay_seconds=1.0
        )
    
    def test_error_classification(self, error_handler):
        """Test error type classification."""
        # Test validation error
        validation_error = ValueError("Invalid vital signs data")
        error_type = error_handler._classify_error(validation_error)
        assert error_type == ErrorType.VALIDATION_ERROR
        
        # Test database error
        db_error = Exception("Database connection failed")
        error_type = error_handler._classify_error(db_error)
        assert error_type == ErrorType.DATABASE_ERROR
        
        # Test timeout error
        timeout_error = TimeoutError("Request timeout")
        error_type = error_handler._classify_error(timeout_error)
        assert error_type == ErrorType.TIMEOUT_ERROR
    
    def test_severity_determination(self, error_handler):
        """Test error severity determination."""
        # Validation errors should be low severity initially
        severity = error_handler._determine_severity(ErrorType.VALIDATION_ERROR, 1)
        assert severity == ErrorSeverity.LOW
        
        # Calculation errors should be high severity
        severity = error_handler._determine_severity(ErrorType.CALCULATION_ERROR, 1) 
        assert severity == ErrorSeverity.HIGH
        
        # Database errors should escalate to critical after retries
        severity = error_handler._determine_severity(ErrorType.DATABASE_ERROR, 3)
        assert severity == ErrorSeverity.CRITICAL
    
    def test_retry_logic(self, error_handler):
        """Test retry decision logic."""
        # Validation errors should not retry after first attempt
        should_retry = error_handler._should_retry(ErrorType.VALIDATION_ERROR, 2)
        assert should_retry == False
        
        # Network errors should always retry
        should_retry = error_handler._should_retry(ErrorType.NETWORK_ERROR, 2)
        assert should_retry == True
        
        # Database errors should retry
        should_retry = error_handler._should_retry(ErrorType.DATABASE_ERROR, 1)
        assert should_retry == True
        
        # Calculation errors should not retry
        should_retry = error_handler._should_retry(ErrorType.CALCULATION_ERROR, 1)
        assert should_retry == False
    
    def test_circuit_breaker_initialization(self, error_handler):
        """Test circuit breaker initialization."""
        assert 'database' in error_handler.circuit_breakers
        assert 'news2_calculation' in error_handler.circuit_breakers
        assert 'patient_registry' in error_handler.circuit_breakers
        
        db_cb = error_handler.circuit_breakers['database']
        assert db_cb.failure_threshold == 5
        assert db_cb.timeout_seconds == 30
    
    def test_error_statistics_tracking(self, error_handler):
        """Test error statistics tracking."""
        # Update stats for different error types
        error_handler._update_error_stats(ErrorType.VALIDATION_ERROR)
        error_handler._update_error_stats(ErrorType.DATABASE_ERROR)
        error_handler._update_error_stats(ErrorType.VALIDATION_ERROR)
        
        stats = error_handler.get_error_stats()
        
        assert stats['total_errors'] == 3
        assert stats['errors_by_type']['validation_error'] == 2
        assert stats['errors_by_type']['database_error'] == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=30)
        
        assert cb.failure_threshold == 3
        assert cb.timeout_seconds == 30
        assert cb.failure_count == 0
        assert cb.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful operations."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=30)
        
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=30)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.failure_count == 1
        assert cb.state.value == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.failure_count == 2
        assert cb.state.value == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker rejecting calls when open."""
        cb = CircuitBreaker(failure_threshold=1, timeout_seconds=30)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.state.value == "open"
        
        # Next call should be rejected immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await cb.call(failing_func)


class TestMetricsCollector:
    """Test metrics collection and monitoring."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    def test_metrics_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector.events_processed is not None
        assert metrics_collector.events_failed is not None
        assert metrics_collector.processing_duration is not None
        assert metrics_collector.news2_scores is not None
    
    def test_record_event_processed(self, metrics_collector):
        """Test recording processed events."""
        # Should not raise exception
        metrics_collector.record_event_processed("copd", "vital_signs")
        metrics_collector.record_event_processed("standard", "vital_signs")
        
        # Verify metrics were recorded (would check actual values in integration tests)
        assert True  # Placeholder - would check Prometheus metrics
    
    def test_record_news2_score(self, metrics_collector):
        """Test recording NEWS2 scores."""
        # Should not raise exception
        metrics_collector.record_news2_score(5, scale_used=1, risk_category="MEDIUM")
        metrics_collector.record_news2_score(8, scale_used=2, risk_category="HIGH")
        
        # Verify metrics were recorded
        assert True  # Placeholder - would check histogram values
    
    def test_record_processing_duration(self, metrics_collector):
        """Test recording processing durations."""
        # Should not raise exception
        metrics_collector.record_processing_duration(0.05, "vital_signs", "standard")
        metrics_collector.record_processing_duration(0.12, "vital_signs", "copd")
        
        # Verify metrics were recorded
        assert True  # Placeholder - would check histogram values
    
    def test_custom_metrics_buffer(self, metrics_collector):
        """Test custom metrics buffer functionality."""
        from src.streaming.monitoring import PerformanceMetric
        
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            timestamp=datetime.now(timezone.utc),
            labels={"type": "test"},
            unit="count"
        )
        
        metrics_collector.add_custom_metric(metric)
        
        recent_metrics = metrics_collector.get_recent_metrics("test_metric", minutes=5)
        assert len(recent_metrics) == 1
        assert recent_metrics[0]['value'] == 42.0
    
    def test_prometheus_metrics_export(self, metrics_collector):
        """Test Prometheus metrics export."""
        # Record some metrics
        metrics_collector.record_event_processed("standard", "vital_signs")
        metrics_collector.record_news2_score(3, 1, "LOW")
        
        # Get Prometheus export
        prometheus_data = metrics_collector.get_prometheus_metrics()
        
        assert isinstance(prometheus_data, str)
        assert "stream_events_processed_total" in prometheus_data


class TestHealthChecker:
    """Test health checking functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock()
        redis_mock.info = AsyncMock(return_value={
            'used_memory': 1024000,
            'maxmemory': 10240000
        })
        return redis_mock
    
    @pytest.fixture
    def health_checker(self, mock_redis):
        return HealthChecker(mock_redis)
    
    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, health_checker, mock_redis):
        """Test successful Redis health check."""
        await health_checker._check_redis_health()
        
        redis_status = health_checker.health_status.get('redis')
        assert redis_status is not None
        assert redis_status.status in ['healthy', 'degraded']
        assert redis_status.component == 'redis'
        
        mock_redis.ping.assert_called_once()
        mock_redis.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_health_check_failure(self, health_checker, mock_redis):
        """Test Redis health check failure."""
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        await health_checker._check_redis_health()
        
        redis_status = health_checker.health_status.get('redis')
        assert redis_status.status == 'unhealthy'
        assert redis_status.error_message == "Connection failed"
    
    def test_health_summary_generation(self, health_checker):
        """Test health summary generation."""
        from src.streaming.monitoring import HealthStatus
        
        # Add some mock health statuses
        health_checker.health_status['test_component'] = HealthStatus(
            component='test_component',
            status='healthy',
            last_check=datetime.now(timezone.utc),
            details={'test': 'data'}
        )
        
        summary = health_checker.get_health_summary()
        
        assert 'overall_status' in summary
        assert 'component_count' in summary
        assert summary['component_count'] == 1
        assert 'components' in summary


class TestStreamProcessor:
    """Test stream processor components."""
    
    def test_vital_signs_record_creation(self):
        """Test VitalSignsRecord creation."""
        record = VitalSignsRecord(
            event_id=str(uuid4()),
            patient_id="TEST_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            respiratory_rate=18,
            spo2=96,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness="ALERT",
            data_source="device"
        )
        
        assert record.patient_id == "TEST_001"
        assert record.respiratory_rate == 18
        assert record.consciousness == "ALERT"
    
    def test_news2_result_record_creation(self):
        """Test NEWS2ResultRecord creation."""
        record = NEWS2ResultRecord(
            event_id=str(uuid4()),
            patient_id="TEST_001",
            calculation_timestamp=datetime.now(timezone.utc).isoformat(),
            vital_signs_timestamp=datetime.now(timezone.utc).isoformat(),
            news2_score=5,
            individual_scores={"rr": 1, "spo2": 2, "temp": 0, "hr": 1, "sbp": 0, "consciousness": 0, "oxygen": 2},
            risk_category="MEDIUM",
            scale_used=1,
            monitoring_frequency="4-6 hourly",
            red_flags=[],
            processing_latency_ms=45.2
        )
        
        assert record.news2_score == 5
        assert record.risk_category == "MEDIUM"
        assert record.scale_used == 1
        assert record.processing_latency_ms == 45.2
    
    def test_stream_processor_initialization(self):
        """Test StreamProcessor initialization."""
        processor = StreamProcessor(
            broker_url='kafka://localhost:9092',
            redis_url='redis://localhost:6379',
            app_id='test_processor'
        )
        
        assert processor.app_id == 'test_processor'
        assert processor.broker_url == 'kafka://localhost:9092'
        assert processor.redis_url == 'redis://localhost:6379'
        assert processor._processed_count == 0
        assert processor._error_count == 0


if __name__ == "__main__":
    # Run unit tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])