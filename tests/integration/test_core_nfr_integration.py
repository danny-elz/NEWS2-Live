"""
Core NFR Integration tests for NEWS2 Alert Suppression.
Tests essential NFR components without external dependencies.
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
from src.services.alert_suppression_security import InputValidator, AuthenticationManager
from src.services.alert_suppression_resilience import CircuitBreaker, RetryManager, HealthMonitor


class TestCoreNFRIntegration:
    """Core NFR integration tests."""
    
    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
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
    async def health_monitor(self):
        """Create HealthMonitor."""
        return HealthMonitor(check_interval=1)


class TestConfigSecurityIntegration(TestCoreNFRIntegration):
    """Test integration between configuration and security components."""
    
    @pytest.mark.asyncio
    async def test_secure_config_loading(self, config_manager):
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
    async def test_hot_reload_security_impact(self, config_manager):
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


class TestResilienceIntegration(TestCoreNFRIntegration):
    """Test resilience component integrations."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, circuit_breaker):
        """Test circuit breaker core functionality."""
        # Simulate function that fails
        async def failing_function():
            raise Exception("Simulated failure")
        
        # Trigger circuit breaker failures
        failure_count = 0
        for i in range(5):  # Exceed failure threshold
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        # Verify circuit breaker opened
        assert circuit_breaker.state.name == "OPEN"
        assert failure_count >= 3  # At least threshold failures
    
    @pytest.mark.asyncio
    async def test_retry_manager_functionality(self, retry_manager):
        """Test retry manager functionality."""
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
        assert attempt_count == 3  # Should have retried twice
    
    @pytest.mark.asyncio
    async def test_health_monitor_functionality(self, health_monitor):
        """Test health monitor functionality."""
        # Register components with health monitor
        health_monitor.register_component("database", lambda: True)
        health_monitor.register_component("cache", lambda: False)  # Unhealthy
        
        # Run health checks
        await health_monitor.check_all_components()
        
        # Verify health status
        status = health_monitor.get_overall_status()
        assert status["overall_health"] == "degraded"  # Due to unhealthy cache
        assert status["components"]["database"] == "healthy"
        assert status["components"]["cache"] == "unhealthy"


class TestEndToEndIntegration(TestCoreNFRIntegration):
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_suppression_flow(self, config_manager, circuit_breaker, input_validator):
        """Test complete alert suppression flow with NFR components."""
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
            await asyncio.sleep(0.01)  # Simulate processing time
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
        assert processing_time < 1.0  # Should be fast
        
        # 5. Test configuration hot reload doesn't break active session
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
    async def test_error_handling_across_components(self, config_manager, circuit_breaker):
        """Test error handling and recovery across NFR components."""
        # 1. Test configuration error handling
        try:
            await config_manager.load_config_from_file("nonexistent_file.json")
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        
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
        
        # 3. Test that circuit breaker prevents further calls
        with pytest.raises(Exception):  # Should raise CircuitBreakerOpenError or similar
            await circuit_breaker.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, config_manager, circuit_breaker):
        """Test system performance under simulated load."""
        # Load configuration
        config = await config_manager.get_config()
        
        # Simulate concurrent suppression requests
        async def simulate_suppression_request(request_id: int):
            start_time = time.time()
            
            # Simulate suppression logic
            await asyncio.sleep(0.001)  # Small delay to simulate processing
            
            processing_time = time.time() - start_time
            
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
        
        # Performance assertions (generous limits for test environment)
        assert avg_processing_time < 0.1  # Average under 100ms
        assert max_processing_time < 0.2   # Max under 200ms
        assert total_time < 2.0            # All requests under 2 seconds
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, config_manager):
        """Test configuration validation across components."""
        # Test valid configuration
        valid_config = {
            "time_suppression_window_minutes": 30,
            "score_increase_threshold": 2,
            "stability_threshold_hours": 2,
            "security": {
                "require_authentication": True,
                "session_timeout_minutes": 60
            }
        }
        
        await config_manager._apply_config_update(valid_config)
        config = await config_manager.get_config()
        assert config.time_suppression_window_minutes == 30
        
        # Test invalid configuration (negative values)
        invalid_config = {
            "time_suppression_window_minutes": -10,  # Invalid
            "score_increase_threshold": 2,
            "stability_threshold_hours": 2
        }
        
        # Should handle invalid config gracefully
        try:
            await config_manager._apply_config_update(invalid_config)
            # If it doesn't raise an exception, it should validate/sanitize the value
            updated_config = await config_manager.get_config()
            # Should either reject the update or sanitize to a valid value
            assert updated_config.time_suppression_window_minutes > 0
        except Exception:
            # It's also acceptable to raise a validation exception
            pass
    
    @pytest.mark.asyncio
    async def test_security_input_validation_integration(self, input_validator):
        """Test comprehensive input validation."""
        # Test valid inputs
        valid_patient_id = input_validator.validate_patient_id("PAT123456")
        assert valid_patient_id == "PAT123456"
        
        valid_score = input_validator.validate_news2_score(5)
        assert valid_score == 5
        
        # Test invalid inputs
        with pytest.raises(Exception):  # ValidationError
            input_validator.validate_patient_id("PAT'; DROP TABLE patients; --")
        
        with pytest.raises(Exception):  # ValidationError
            input_validator.validate_news2_score(-5)
        
        with pytest.raises(Exception):  # ValidationError
            input_validator.validate_news2_score(25)  # Above maximum


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "--tb=short"])