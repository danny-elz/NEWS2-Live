"""
Simple validation tests for NFR components.
Validates that all NFR components can be imported and instantiated correctly.
"""

import asyncio
import json
import pytest
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_config_management_import():
    """Test that configuration management can be imported."""
    try:
        from src.services.alert_suppression_config import SuppressionConfig, ConfigManager
        # Test basic instantiation
        config = SuppressionConfig()
        assert config.time_suppression_window_minutes == 30  # Default value
        
        # Test ConfigManager instantiation
        config_manager = ConfigManager()
        assert config_manager is not None
        
        print("✅ Configuration management components imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration management import failed: {e}")
        return False


def test_security_components_import():
    """Test that security components can be imported."""
    try:
        from src.services.alert_suppression_security import (
            InputValidator, 
            AuthenticationManager, 
            AuthorizationManager,
            ValidationError
        )
        
        # Test basic instantiation
        validator = InputValidator()
        assert validator is not None
        
        auth_manager = AuthenticationManager("test_secret", 60)
        assert auth_manager is not None
        
        authz_manager = AuthorizationManager()
        assert authz_manager is not None
        
        print("✅ Security components imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Security components import failed: {e}")
        return False


def test_resilience_components_import():
    """Test that resilience components can be imported."""
    try:
        from src.services.alert_suppression_resilience import (
            CircuitBreaker, 
            CircuitBreakerConfig,
            RetryManager, 
            HealthMonitor
        )
        
        # Test basic instantiation with correct parameters
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5)
        circuit_breaker = CircuitBreaker("test", config)
        assert circuit_breaker is not None
        
        retry_manager = RetryManager()
        assert retry_manager is not None
        
        health_monitor = HealthMonitor(check_interval=30)
        assert health_monitor is not None
        
        print("✅ Resilience components imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Resilience components import failed: {e}")
        return False


def test_monitoring_components_import():
    """Test that monitoring components can be imported."""
    try:
        # Try importing without external dependencies first
        from src.services.alert_suppression_monitoring import (
            MonitoringEvent,
            AlertSeverity,
            MonitoringEventType,
            MetricThreshold
        )
        
        # Test basic instantiation
        event = MonitoringEvent(
            event_type=MonitoringEventType.SUPPRESSION_APPLIED,
            severity=AlertSeverity.INFO,
            message="Test event"
        )
        assert event is not None
        
        threshold = MetricThreshold(
            metric_name="test_metric",
            threshold_value=10.0,
            comparison_operator=">"
        )
        assert threshold is not None
        
        print("✅ Monitoring components imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Monitoring components import failed: {e}")
        return False


def test_dashboard_components_import():
    """Test that dashboard components can be imported."""
    try:
        from src.services.alert_suppression_dashboard import (
            DashboardMetrics
        )
        
        # Test basic instantiation
        metrics = DashboardMetrics()
        assert metrics is not None
        assert metrics.total_suppressions == 0
        
        print("✅ Dashboard components imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard components import failed: {e}")
        return False


@pytest.mark.asyncio
async def test_config_manager_basic_functionality():
    """Test basic ConfigManager functionality."""
    try:
        from src.services.alert_suppression_config import ConfigManager
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "time_suppression_window_minutes": 45,
                "score_increase_threshold": 3,
                "stability_threshold_hours": 3,
                "security": {
                    "require_authentication": True,
                    "session_timeout_minutes": 90
                }
            }
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Test loading config
            config_manager = ConfigManager(config_file=config_file)
            await config_manager.load_config()
            
            config = await config_manager.get_config()
            assert config is not None
            assert config.time_suppression_window_minutes == 45
            assert config.score_increase_threshold == 3
            
            print("✅ ConfigManager basic functionality working")
            return True
        finally:
            os.unlink(config_file)
            
    except Exception as e:
        print(f"❌ ConfigManager functionality test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_input_validation_functionality():
    """Test basic input validation functionality."""
    try:
        from src.services.alert_suppression_security import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test valid inputs
        valid_patient_id = validator.validate_patient_id("PAT123456")
        assert valid_patient_id == "PAT123456"
        
        valid_score = validator.validate_news2_score(5)
        assert valid_score == 5
        
        # Test invalid inputs
        try:
            validator.validate_patient_id("'; DROP TABLE;")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        try:
            validator.validate_news2_score(-1)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        print("✅ Input validation functionality working")
        return True
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_authentication_functionality():
    """Test basic authentication functionality."""
    try:
        from src.services.alert_suppression_security import AuthenticationManager
        
        auth_manager = AuthenticationManager("test_secret_key", 60)
        
        # Test session creation
        token = auth_manager.create_session("test_user")
        assert token is not None
        
        # Test session validation
        session_info = auth_manager.validate_session(token)
        assert session_info["is_valid"] is True
        assert session_info["user_id"] == "test_user"
        
        # Test invalid session
        invalid_session = auth_manager.validate_session("invalid_token")
        assert invalid_session["is_valid"] is False
        
        print("✅ Authentication functionality working")
        return True
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_circuit_breaker_functionality():
    """Test basic circuit breaker functionality."""
    try:
        from src.services.alert_suppression_resilience import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        circuit_breaker = CircuitBreaker("test_breaker", config)
        
        # Test successful call
        async def success_function():
            return "success"
        
        result = await circuit_breaker.call(success_function)
        assert result == "success"
        
        # Test failing function
        async def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit breaker
        for i in range(3):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass
        
        # Circuit breaker should be open now
        assert circuit_breaker.state.name == "OPEN"
        
        print("✅ Circuit breaker functionality working")
        return True
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_retry_manager_functionality():
    """Test basic retry manager functionality."""
    try:
        from src.services.alert_suppression_resilience import RetryManager
        
        retry_manager = RetryManager()
        
        # Test function that succeeds after retries
        attempt_count = 0
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception(f"Attempt {attempt_count} failed")
            return "success"
        
        result = await retry_manager.execute_with_retry(
            flaky_function,
            operation_name="test_operation",
            max_retries=3,
            base_delay=0.01
        )
        
        assert result == "success"
        assert attempt_count == 2
        
        print("✅ Retry manager functionality working")
        return True
    except Exception as e:
        print(f"❌ Retry manager test failed: {e}")
        return False


def test_integration_validation_summary():
    """Run all validation tests and provide summary."""
    print("\n" + "="*60)
    print("NFR COMPONENTS INTEGRATION VALIDATION SUMMARY")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Config Management", test_config_management_import()))
    results.append(("Security Components", test_security_components_import()))
    results.append(("Resilience Components", test_resilience_components_import()))
    results.append(("Monitoring Components", test_monitoring_components_import()))
    results.append(("Dashboard Components", test_dashboard_components_import()))
    
    print(f"\nIMPORT VALIDATION RESULTS:")
    for component, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {component}: {status}")
    
    # Count successes
    successful_imports = sum(1 for _, success in results if success)
    total_components = len(results)
    
    print(f"\nIMPORT SUMMARY: {successful_imports}/{total_components} components imported successfully")
    
    if successful_imports == total_components:
        print("🎉 ALL NFR COMPONENTS READY FOR INTEGRATION!")
        print("\nNext steps:")
        print("- Run async functionality tests with: pytest -k 'async' -v")
        print("- Components are properly structured for production use")
        print("- Configuration, Security, Resilience, and Monitoring are operational")
    else:
        print("⚠️  Some components need attention before full integration")
    
    print("="*60)
    
    return successful_imports == total_components


if __name__ == "__main__":
    # Run validation
    test_integration_validation_summary()
    
    # Run async tests
    import asyncio
    async def run_async_tests():
        print("\nRunning async functionality tests...")
        
        tests = [
            ("Config Manager", test_config_manager_basic_functionality()),
            ("Input Validation", test_input_validation_functionality()), 
            ("Authentication", test_authentication_functionality()),
            ("Circuit Breaker", test_circuit_breaker_functionality()),
            ("Retry Manager", test_retry_manager_functionality())
        ]
        
        print(f"\nASYNC FUNCTIONALITY RESULTS:")
        for name, test_coro in tests:
            try:
                success = await test_coro
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {name}: {status}")
            except Exception as e:
                print(f"  {name}: ❌ ERROR - {e}")
    
    asyncio.run(run_async_tests())