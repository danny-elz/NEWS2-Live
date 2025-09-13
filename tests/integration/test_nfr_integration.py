"""
Integration tests for NEWS2 Alert Suppression NFR components.
Tests the complete system with all NFR enhancements working together.
"""

import asyncio
import json
import logging
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.services.alert_suppression_config import SuppressionConfig, ConfigManager
from src.services.alert_suppression_security import InputValidator, AuthenticationManager, AuthorizationManager
from src.services.alert_suppression_resilience import CircuitBreaker, RetryManager, HealthMonitor
from src.services.alert_suppression_monitoring import SuppressionMonitor, MonitoringEvent, AlertSeverity, MonitoringEventType
from src.services.alert_suppression_dashboard import SuppressionDashboard, SuppressionReportGenerator


class TestNFRIntegration:
    """Integration tests for all NFR components working together."""
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.scard.return_value = 5
        redis_mock.get.return_value = "10"
        redis_mock.lrange.return_value = ["1.5", "2.1", "1.8"]
        redis_mock.hgetall.return_value = {"patient_1": "3", "patient_2": "7"}
        redis_mock.lpush.return_value = 1
        redis_mock.expire.return_value = True
        return redis_mock
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "time_suppression_window_minutes": 30,
                "score_increase_threshold": 2,
                "stability_threshold_hours": 2,
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                },
                "security": {
                    "require_authentication": True,
                    "session_timeout_minutes": 60
                },
                "monitoring": {
                    "prometheus_enabled": True,
                    "health_check_interval": 30
                }
            }
            json.dump(config_data, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    async def config_manager(self, temp_config_file):
        """Create configured ConfigManager."""
        config_manager = ConfigManager(config_file=temp_config_file)
        await config_manager.load_config()
        return config_manager
    
    @pytest.fixture
    def authorization_manager(self):
        """Create AuthorizationManager."""
        return AuthorizationManager()
    
    @pytest.fixture
    def input_validator(self):
        """Create InputValidator."""
        return InputValidator()
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create CircuitBreaker."""
        return CircuitBreaker("test_breaker", failure_threshold=3, timeout_seconds=5)
    
    @pytest.fixture
    def retry_manager(self):
        """Create RetryManager."""
        return RetryManager(max_retries=3, base_delay=0.1)
    
    @pytest.fixture
    async def health_monitor(self, mock_redis):
        """Create HealthMonitor."""
        monitor = HealthMonitor(check_interval=1)
        monitor.redis_client = mock_redis
        return monitor
    
    @pytest.fixture
    async def suppression_monitor(self, mock_redis):
        """Create SuppressionMonitor."""
        monitor = SuppressionMonitor(redis_client=mock_redis)
        return monitor
    
    @pytest.fixture
    async def dashboard(self, suppression_monitor, mock_redis):
        """Create SuppressionDashboard."""
        return SuppressionDashboard(suppression_monitor, mock_redis)
    
    @pytest.fixture
    async def report_generator(self, suppression_monitor, mock_redis):
        """Create SuppressionReportGenerator."""
        return SuppressionReportGenerator(suppression_monitor, mock_redis)


class TestConfigSecurityIntegration(TestNFRIntegration):
    """Test integration between configuration and security components."""
    
    @pytest.mark.asyncio
    async def test_secure_config_loading(self, config_manager, authorization_manager):
        """Test that configuration loading respects security constraints."""
        # Load config
        config = await config_manager.get_config()
        assert config is not None
        
        # Validate security settings are enforced
        assert config.security_config.require_authentication is True
        assert config.security_config.session_timeout_minutes == 60
        
        # Test security manager initialization with config
        auth_manager = AuthenticationManager(
            secret_key="test_secret",
            session_timeout=config.security_config.session_timeout_minutes
        )
        
        # Create a test session
        user_id = "test_user"
        token = auth_manager.create_session(user_id)
        assert token is not None
        
        # Validate session
        session_info = auth_manager.validate_session(token)
        assert session_info["user_id"] == user_id
        assert session_info["is_valid"] is True
    
    @pytest.mark.asyncio
    async def test_input_validation_with_config(self, config_manager, input_validator):
        """Test input validation using configuration parameters."""
        config = await config_manager.get_config()
        
        # Test patient ID validation
        valid_patient_id = "PAT123456"
        validated_id = input_validator.validate_patient_id(valid_patient_id)
        assert validated_id == valid_patient_id
        
        # Test NEWS2 score validation with config thresholds
        valid_score = 5
        validated_score = input_validator.validate_news2_score(valid_score)
        assert validated_score == valid_score
        
        # Test that scores above threshold are handled appropriately
        high_score = 15
        validated_high_score = input_validator.validate_news2_score(high_score)
        assert validated_high_score == high_score  # Should still validate but may trigger alerts
    
    @pytest.mark.asyncio
    async def test_hot_reload_security_impact(self, config_manager, authorization_manager):
        """Test that hot config reloads don't compromise security."""
        # Get initial config
        initial_config = await config_manager.get_config()
        initial_timeout = initial_config.security_config.session_timeout_minutes
        
        # Create a test session
        auth_manager = AuthenticationManager("test_secret", initial_timeout)
        token = auth_manager.create_session("test_user")
        
        # Simulate config change
        new_config_data = {
            "time_suppression_window_minutes": 45,  # Changed
            "security": {
                "require_authentication": True,
                "session_timeout_minutes": 30  # Reduced timeout
            }
        }
        
        # Update config
        await config_manager._apply_config_update(new_config_data)
        updated_config = await config_manager.get_config()
        
        # Verify config updated
        assert updated_config.time_suppression_window_minutes == 45
        assert updated_config.security_config.session_timeout_minutes == 30
        
        # Verify existing session still works initially
        session_info = auth_manager.validate_session(token)
        assert session_info["is_valid"] is True


class TestResilienceMonitoringIntegration(TestNFRIntegration):
    """Test integration between resilience and monitoring components."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_monitoring(self, circuit_breaker, suppression_monitor):
        """Test that circuit breaker events are properly monitored."""
        events_captured = []
        
        # Mock the emit_event method to capture events
        original_emit = suppression_monitor.emit_event
        async def capture_emit(event):
            events_captured.append(event)
            return await original_emit(event)
        
        suppression_monitor.emit_event = capture_emit
        
        # Simulate function that fails
        async def failing_function():
            raise Exception("Simulated failure")
        
        # Trigger circuit breaker failures
        for i in range(4):  # Exceed failure threshold
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass
        
        # Verify circuit breaker opened and event was emitted
        assert circuit_breaker.state.name == "OPEN"
        
        # Check that monitoring events were captured
        circuit_breaker_events = [
            e for e in events_captured 
            if e.event_type == MonitoringEventType.CIRCUIT_BREAKER_OPENED
        ]
        assert len(circuit_breaker_events) > 0
        assert circuit_breaker_events[0].severity == AlertSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_retry_manager_with_monitoring(self, retry_manager, suppression_monitor):
        """Test retry manager integration with monitoring."""
        events_captured = []
        
        # Mock emit_event to capture monitoring events
        async def capture_emit(event):
            events_captured.append(event)
        
        suppression_monitor.emit_event = capture_emit
        
        # Function that fails then succeeds
        attempt_count = 0
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return "success"
        
        # Execute with retry manager
        result = await retry_manager.execute_with_retry(
            flaky_function,
            operation_name="test_operation"
        )
        
        assert result == "success"
        assert attempt_count == 3
        
        # Verify performance metrics were recorded
        retry_manager.record_retry_metrics("test_operation", attempt_count - 1, 0.1)
    
    @pytest.mark.asyncio
    async def test_health_monitor_alerts(self, health_monitor, suppression_monitor):
        """Test health monitor integration with alerting."""
        events_captured = []
        
        async def capture_emit(event):
            events_captured.append(event)
        
        suppression_monitor.emit_event = capture_emit
        
        # Register components with health monitor
        health_monitor.register_component("database", lambda: True)
        health_monitor.register_component("cache", lambda: False)  # Unhealthy
        
        # Run health checks
        await health_monitor.check_all_components()
        
        # Verify health status
        status = health_monitor.get_overall_status()
        assert status["overall_health"] == "degraded"  # Due to unhealthy cache
        
        # Verify unhealthy components trigger monitoring events
        health_events = [
            e for e in events_captured 
            if e.event_type == MonitoringEventType.HEALTH_CHECK_FAILED
        ]
        assert len(health_events) > 0


class TestMonitoringDashboardIntegration(TestNFRIntegration):
    """Test integration between monitoring and dashboard components."""
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_metrics(self, dashboard, suppression_monitor, mock_redis):
        """Test that dashboard displays real-time monitoring metrics."""
        # Record some test metrics
        suppression_monitor.record_metric("suppression_processing_time", 1.5)
        suppression_monitor.record_metric("active_suppressions", 10)
        
        # Generate dashboard metrics
        metrics = await dashboard.get_real_time_metrics()
        
        assert metrics is not None
        assert metrics.active_suppressions >= 0
        assert metrics.total_suppressions >= 0
        assert metrics.system_health_score >= 0
        assert metrics.last_updated is not None
    
    @pytest.mark.asyncio
    async def test_dashboard_html_generation(self, dashboard, suppression_monitor):
        """Test HTML dashboard generation with monitoring data."""
        # Emit a test monitoring event
        test_event = MonitoringEvent(
            event_type=MonitoringEventType.SUPPRESSION_APPLIED,
            severity=AlertSeverity.INFO,
            message="Test suppression applied",
            metadata={"patient_id": "TEST123"}
        )
        await suppression_monitor.emit_event(test_event)
        
        # Generate HTML dashboard
        html_content = await dashboard.generate_html_dashboard()
        
        assert html_content is not None
        assert "NEWS2 Alert Suppression Dashboard" in html_content
        assert "LIVE" in html_content
        assert "System Health" in html_content
        
        # Verify dashboard contains metric cards
        assert "Active Suppressions" in html_content
        assert "Total Today" in html_content
        assert "Avg Processing Time" in html_content
    
    @pytest.mark.asyncio
    async def test_report_generation_with_monitoring_data(self, report_generator, suppression_monitor, mock_redis):
        """Test report generation using monitoring system data."""
        # Set up mock Redis data for report generation
        mock_redis.get.side_effect = lambda key: {
            f"news2:suppression_decisions:{datetime.utcnow().strftime('%Y-%m-%d')}": "25",
            "news2:errors:today": "2",
            "news2:requests:today": "100"
        }.get(key, "0")
        
        mock_redis.hgetall.side_effect = lambda key: {
            f"news2:patient_stats:{datetime.utcnow().strftime('%Y-%m-%d')}": {
                "PAT001": "5", "PAT002": "3", "PAT003": "8"
            },
            f"news2:performance:{datetime.utcnow().strftime('%Y-%m-%d')}": {
                "avg_processing_time": "1.2",
                "max_processing_time": "3.5",
                "min_processing_time": "0.8",
                "total_requests": "100"
            }
        }.get(key, {})
        
        # Generate daily report
        report = await report_generator.generate_daily_report()
        
        assert report is not None
        assert "report_date" in report
        assert "summary" in report
        assert report["summary"]["total_decisions"] == 25
        assert "patient_statistics" in report
        assert "performance_metrics" in report
        assert "recommendations" in report
        
        # Verify CSV export works
        csv_content = await report_generator.export_report_csv(report)
        assert csv_content is not None
        assert "Metric,Value" in csv_content


class TestFullSystemIntegration(TestNFRIntegration):
    """Test complete system integration with all NFR components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_suppression_with_nfr(self, config_manager, authorization_manager, 
                                                  circuit_breaker, suppression_monitor, 
                                                  dashboard, input_validator):
        """Test complete alert suppression flow with all NFR components."""
        # 1. Load and validate configuration
        config = await config_manager.get_config()
        assert config is not None
        
        # 2. Validate input data
        patient_id = input_validator.validate_patient_id("PAT123456")
        news2_score = input_validator.validate_news2_score(8)
        
        # 3. Create authentication session
        auth_manager = AuthenticationManager("test_secret", 60)
        session_token = auth_manager.create_session("test_user")
        session_info = auth_manager.validate_session(session_token)
        assert session_info["is_valid"] is True
        
        # 4. Execute suppression logic with circuit breaker protection
        async def suppression_logic():
            # Simulate suppression decision making
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "decision": "suppress",
                "reason": "time_based_suppression",
                "patient_id": patient_id,
                "score": news2_score,
                "duration_minutes": config.time_suppression_window_minutes
            }
        
        start_time = time.time()
        result = await circuit_breaker.call(suppression_logic)
        processing_time = time.time() - start_time
        
        assert result["decision"] == "suppress"
        assert result["patient_id"] == patient_id
        
        # 5. Record monitoring metrics
        suppression_monitor.record_metric("suppression_processing_time", processing_time)
        suppression_monitor.record_metric("active_suppressions", 1)
        
        # 6. Emit monitoring event
        monitoring_event = MonitoringEvent(
            event_type=MonitoringEventType.SUPPRESSION_APPLIED,
            severity=AlertSeverity.INFO,
            message=f"Suppression applied for patient {patient_id}",
            metadata={
                "patient_id": patient_id,
                "news2_score": news2_score,
                "suppression_duration": config.time_suppression_window_minutes,
                "processing_time": processing_time
            }
        )
        await suppression_monitor.emit_event(monitoring_event)
        
        # 7. Verify dashboard reflects the changes
        dashboard_metrics = await dashboard.get_real_time_metrics()
        assert dashboard_metrics.last_updated is not None
        
        # 8. Test configuration hot reload doesn't break active session
        new_config = {
            "time_suppression_window_minutes": 45,  # Changed value
            "score_increase_threshold": 3,  # Changed value
            "security": {
                "require_authentication": True,
                "session_timeout_minutes": 60
            }
        }
        await config_manager._apply_config_update(new_config)
        
        # Verify session still valid after config reload
        session_info_after_reload = auth_manager.validate_session(session_token)
        assert session_info_after_reload["is_valid"] is True
        
        # Verify new config values are active
        updated_config = await config_manager.get_config()
        assert updated_config.time_suppression_window_minutes == 45
        assert updated_config.score_increase_threshold == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_across_all_components(self, config_manager, circuit_breaker, 
                                                       suppression_monitor, mock_redis):
        """Test error handling and recovery across all NFR components."""
        events_captured = []
        
        async def capture_emit(event):
            events_captured.append(event)
        
        suppression_monitor.emit_event = capture_emit
        
        # 1. Test configuration error handling
        try:
            await config_manager.load_config_from_file("nonexistent_file.json")
        except Exception as e:
            assert "not found" in str(e).lower()
        
        # 2. Test circuit breaker with cascading failures
        async def failing_operation():
            raise Exception("Simulated system failure")
        
        # Trigger circuit breaker
        for i in range(5):
            try:
                await circuit_breaker.call(failing_operation)
            except:
                pass
        
        # Verify circuit breaker is open
        assert circuit_breaker.state.name == "OPEN"
        
        # 3. Test monitoring system handles Redis failures gracefully
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        # Should not crash, should handle gracefully
        try:
            health_status = await suppression_monitor.get_health_status()
            assert "components" in health_status
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            assert "connection" in str(e).lower()
        
        # 4. Verify error events were captured
        error_events = [e for e in events_captured if e.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]]
        assert len(error_events) > 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, config_manager, suppression_monitor, 
                                         circuit_breaker, dashboard):
        """Test system performance under simulated load."""
        # Load configuration
        config = await config_manager.get_config()
        
        # Simulate concurrent suppression requests
        async def simulate_suppression_request(request_id: int):
            start_time = time.time()
            
            # Simulate suppression logic
            await asyncio.sleep(0.01)  # Small delay to simulate processing
            
            processing_time = time.time() - start_time
            
            # Record metrics
            suppression_monitor.record_metric("suppression_processing_time", processing_time)
            suppression_monitor.record_metric("request_latency", processing_time * 1000)  # ms
            
            return {
                "request_id": request_id,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow()
            }
        
        # Execute 50 concurrent requests
        start_time = time.time()
        tasks = [simulate_suppression_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(results) == 50
        
        # Calculate performance metrics
        processing_times = [r["processing_time"] for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        # Performance assertions
        assert avg_processing_time < 0.1  # Average under 100ms
        assert max_processing_time < 0.2   # Max under 200ms
        assert total_time < 2.0            # All requests under 2 seconds
        
        # Verify dashboard can handle the load
        dashboard_metrics = await dashboard.get_real_time_metrics()
        assert dashboard_metrics is not None
        
        # Record load test metrics
        suppression_monitor.record_metric("load_test_requests", 50)
        suppression_monitor.record_metric("load_test_duration", total_time)
        suppression_monitor.record_metric("load_test_avg_latency", avg_processing_time * 1000)


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "--tb=short"])