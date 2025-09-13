"""
Global pytest configuration for NEWS2-Live project.

This file configures pytest for the entire project, including async support,
fixtures, and test environment setup.
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import Mock, AsyncMock

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure asyncio event loop policy for testing
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Register custom pytest markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "critical_safety: Critical safety constraint tests that must NEVER fail"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.delete.return_value = 1
    mock.zadd.return_value = 1
    mock.zrevrange.return_value = []
    mock.zrevrangebyscore.return_value = []
    mock.expire.return_value = True
    return mock


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    mock = Mock()
    mock.create_audit_entry = Mock()
    mock.log_alert_generation = Mock()
    mock.log_suppression_decision = Mock()
    return mock


@pytest.fixture
def test_timestamp():
    """Standard test timestamp."""
    return datetime.now(timezone.utc)


@pytest.fixture
def test_uuid():
    """Generate test UUID."""
    return uuid4()


# Performance test configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set asyncio mode
    config.option.asyncio_mode = "auto"

    # Configure markers
    config.addinivalue_line(
        "markers", "performance: Performance and load testing"
    )
    config.addinivalue_line(
        "markers", "security: Security validation tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring multiple components"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 5 seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add performance marker for performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        # Add security marker for security tests
        if "security" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)

        # Add integration marker for integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add unit marker for unit tests
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Async test timeout configuration
@pytest.fixture(scope="session", autouse=True)
def configure_async_timeouts():
    """Configure async test timeouts."""
    # Set default asyncio timeout for tests
    import asyncio
    asyncio.get_event_loop().set_debug(True)


# Skip tests based on environment
def pytest_runtest_setup(item):
    """Setup function called for each test."""
    # Skip performance tests in CI unless specifically requested
    if item.get_closest_marker("performance"):
        if os.environ.get("SKIP_PERFORMANCE_TESTS", "false").lower() == "true":
            pytest.skip("Performance tests skipped in this environment")

    # Skip integration tests if required services unavailable
    if item.get_closest_marker("integration"):
        if os.environ.get("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
            pytest.skip("Integration tests skipped - external services unavailable")