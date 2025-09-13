"""
Epic 3: Multi-Channel Alert Delivery Testing Framework
Comprehensive testing for redundant alert delivery with fault injection and failover validation.

This framework tests the critical communication infrastructure for alert management.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from enum import Enum

from src.models.alerts import Alert, AlertLevel, AlertPriority
from src.services.multi_channel_delivery import (
    MultiChannelDeliveryService,
    DeliveryChannel,
    DeliveryResult,
    DeliveryAttempt,
    ChannelFailure,
    FaultTolerantDelivery
)
from src.models.clinical_users import ClinicalUser, UserRole


class DeliveryChannelType(Enum):
    """Delivery channel types for testing."""
    WEBSOCKET = "websocket"
    PUSH = "push"
    SMS = "sms"
    VOICE = "voice"
    PAGER = "pager"
    EMAIL = "email"
    OVERHEAD_PAGING = "overhead_paging"


class FaultInjector:
    """Inject faults for testing delivery resilience."""

    def __init__(self):
        self.active_faults = {}

    def inject_channel_failure(self, channel: str, duration_seconds: int = 30):
        """Inject failure for specific channel."""
        self.active_faults[channel] = {
            "start_time": time.time(),
            "duration": duration_seconds,
            "fault_type": "channel_failure"
        }

    def inject_network_partition(self, duration_seconds: int = 60):
        """Inject network partition affecting all channels."""
        self.active_faults["network"] = {
            "start_time": time.time(),
            "duration": duration_seconds,
            "fault_type": "network_partition"
        }

    def inject_latency(self, channel: str, latency_ms: int = 5000):
        """Inject high latency for specific channel."""
        self.active_faults[f"{channel}_latency"] = {
            "start_time": time.time(),
            "latency_ms": latency_ms,
            "fault_type": "high_latency"
        }

    def is_channel_failed(self, channel: str) -> bool:
        """Check if channel is currently failed."""
        fault_key = channel
        if fault_key in self.active_faults:
            fault = self.active_faults[fault_key]
            elapsed = time.time() - fault["start_time"]
            return elapsed < fault["duration"]
        return False

    def get_channel_latency(self, channel: str) -> int:
        """Get injected latency for channel."""
        fault_key = f"{channel}_latency"
        if fault_key in self.active_faults:
            return self.active_faults[fault_key]["latency_ms"]
        return 0

    def clear_all_faults(self):
        """Clear all injected faults."""
        self.active_faults.clear()


class TestCriticalAlertDelivery:
    """Test critical alert delivery requirements."""

    @pytest.fixture
    def delivery_service(self):
        """Multi-channel delivery service."""
        return MultiChannelDeliveryService()

    @pytest.fixture
    def fault_injector(self):
        """Fault injection service."""
        return FaultInjector()

    @pytest.fixture
    def critical_user(self):
        """Critical care user for testing."""
        return ClinicalUser(
            user_id="ICU_NURSE_001",
            name="Sarah Johnson",
            role=UserRole.ICU_NURSE,
            contact_info={
                "websocket_session": "ws_session_001",
                "mobile_push_token": "push_token_001",
                "phone_number": "+1234567890",
                "pager_number": "12345",
                "email": "sarah.johnson@hospital.org"
            },
            shift_schedule={"current_shift": "day"},
            preferences={"critical_alert_channels": ["all"]}
        )

    @pytest.mark.critical_safety
    async def test_critical_alert_broadcast_all_channels(self, delivery_service, critical_user):
        """CRITICAL: Critical alerts must be delivered through ALL available channels."""

        critical_alert = self._create_critical_alert()

        # Mock all channel delivery methods
        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(return_value=True),
            _deliver_via_push=AsyncMock(return_value=True),
            _deliver_via_sms=AsyncMock(return_value=True),
            _deliver_via_voice=AsyncMock(return_value=True),
            _deliver_via_pager=AsyncMock(return_value=True),
            _deliver_via_email=AsyncMock(return_value=True)
        ) as mocked_channels:

            delivery_result = await delivery_service.deliver_critical_alert(
                critical_alert, critical_user
            )

            # Verify ALL channels were attempted
            assert delivery_result.total_channels_attempted == 6
            assert delivery_result.broadcast_mode is True

            # Verify each channel was called
            for method_name, mock_method in mocked_channels.items():
                mock_method.assert_called_once()

            # At least one channel must succeed
            assert delivery_result.successful_channels >= 1
            assert delivery_result.delivery_success is True

    @pytest.mark.critical_safety
    async def test_critical_alert_delivery_under_failure(self, delivery_service, critical_user, fault_injector):
        """CRITICAL: Critical alerts must be delivered even with channel failures."""

        critical_alert = self._create_critical_alert()

        # Inject failures in primary channels
        fault_injector.inject_channel_failure("websocket", 30)
        fault_injector.inject_channel_failure("push", 30)

        with patch.object(delivery_service, '_is_channel_available') as mock_available:
            def channel_availability(channel):
                return not fault_injector.is_channel_failed(channel)
            mock_available.side_effect = channel_availability

            with patch.multiple(
                delivery_service,
                _deliver_via_websocket=AsyncMock(side_effect=ChannelFailure("WebSocket failed")),
                _deliver_via_push=AsyncMock(side_effect=ChannelFailure("Push failed")),
                _deliver_via_sms=AsyncMock(return_value=True),
                _deliver_via_voice=AsyncMock(return_value=True),
                _deliver_via_pager=AsyncMock(return_value=True)
            ):

                delivery_result = await delivery_service.deliver_critical_alert(
                    critical_alert, critical_user
                )

                # Must still achieve delivery despite failures
                assert delivery_result.delivery_success is True
                assert delivery_result.successful_channels >= 1

                # Should have attempted all channels
                assert delivery_result.total_channels_attempted == 6
                assert delivery_result.failed_channels == 2

    async def test_critical_alert_delivery_timing(self, delivery_service, critical_user):
        """Test critical alert delivery meets <15 second SLA."""

        critical_alert = self._create_critical_alert()

        start_time = time.time()

        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(return_value=True),
            _deliver_via_push=AsyncMock(return_value=True),
            _deliver_via_sms=AsyncMock(return_value=True)
        ):

            delivery_result = await delivery_service.deliver_critical_alert(
                critical_alert, critical_user
            )

            end_time = time.time()
            delivery_time = (end_time - start_time) * 1000  # Convert to ms

            # Must deliver within 15 seconds
            assert delivery_time < 15000, f"Critical alert delivery took {delivery_time}ms (>15s limit)"
            assert delivery_result.delivery_latency_ms < 15000

    async def test_critical_alert_confirmation_tracking(self, delivery_service, critical_user):
        """Test critical alert delivery confirmation tracking."""

        critical_alert = self._create_critical_alert()

        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(return_value=True),
            _deliver_via_push=AsyncMock(return_value=True)
        ):

            delivery_result = await delivery_service.deliver_critical_alert(
                critical_alert, critical_user
            )

            # Critical alerts must require confirmation
            assert delivery_result.confirmation_required is True
            assert delivery_result.confirmation_timeout_minutes == 5

            # Simulate user confirmation
            confirmation = await delivery_service.confirm_alert_receipt(
                critical_alert.alert_id,
                critical_user.user_id,
                "acknowledged_critical_alert"
            )

            assert confirmation.confirmed is True
            assert confirmation.confirmed_at is not None
            assert confirmation.confirmation_latency_ms < 300000  # <5 minutes

    def _create_critical_alert(self) -> Alert:
        """Create critical alert for testing."""
        return _create_critical_alert()


class TestFailoverCascade:
    """Test failover cascade for normal alerts."""

    @pytest.fixture
    def delivery_service(self):
        return MultiChannelDeliveryService()

    @pytest.fixture
    def fault_injector(self):
        return FaultInjector()

    @pytest.fixture
    def regular_user(self):
        return ClinicalUser(
            user_id="WARD_NURSE_001",
            name="Linda Smith",
            role=UserRole.WARD_NURSE,
            contact_info={
                "websocket_session": "ws_session_002",
                "mobile_push_token": "push_token_002",
                "phone_number": "+1234567891"
            }
        )

    async def test_normal_alert_failover_cascade(self, delivery_service, regular_user, fault_injector):
        """Test normal alerts use failover cascade (not broadcast)."""

        normal_alert = self._create_normal_alert()

        # Inject failures in primary channels
        fault_injector.inject_channel_failure("websocket", 30)
        fault_injector.inject_channel_failure("push", 30)

        delivery_attempts = []

        async def track_delivery_attempt(channel, *args, **kwargs):
            delivery_attempts.append(channel)
            if fault_injector.is_channel_failed(channel):
                raise ChannelFailure(f"{channel} failed")
            return True

        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(side_effect=lambda *args, **kwargs: track_delivery_attempt("websocket", *args, **kwargs)),
            _deliver_via_push=AsyncMock(side_effect=lambda *args, **kwargs: track_delivery_attempt("push", *args, **kwargs)),
            _deliver_via_sms=AsyncMock(side_effect=lambda *args, **kwargs: track_delivery_attempt("sms", *args, **kwargs))
        ):

            delivery_result = await delivery_service.deliver_alert(normal_alert, regular_user)

            # Should follow cascade order
            assert delivery_attempts == ["websocket", "push", "sms"]
            assert delivery_result.final_successful_channel == "sms"
            assert delivery_result.delivery_success is True

            # Should stop after first success (not broadcast)
            assert delivery_result.broadcast_mode is False

    async def test_failover_timing_requirements(self, delivery_service, regular_user, fault_injector):
        """Test failover activation meets <30 second requirement."""

        normal_alert = self._create_normal_alert()

        # Inject failure in primary channel
        fault_injector.inject_channel_failure("websocket", 30)

        start_time = time.time()

        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(side_effect=ChannelFailure("WebSocket failed")),
            _deliver_via_push=AsyncMock(return_value=True)
        ):

            delivery_result = await delivery_service.deliver_alert(normal_alert, regular_user)

            failover_time = (time.time() - start_time) * 1000

            # Failover must activate within 30 seconds
            assert failover_time < 30000, f"Failover took {failover_time}ms (>30s limit)"
            assert delivery_result.failover_latency_ms < 30000
            assert delivery_result.delivery_success is True

    async def test_exhausted_channels_escalation(self, delivery_service, regular_user, fault_injector):
        """Test escalation when all channels fail."""

        normal_alert = self._create_normal_alert()

        # Inject failures in all channels
        for channel in ["websocket", "push", "sms", "voice", "pager"]:
            fault_injector.inject_channel_failure(channel, 60)

        with patch.multiple(
            delivery_service,
            _deliver_via_websocket=AsyncMock(side_effect=ChannelFailure("WebSocket failed")),
            _deliver_via_push=AsyncMock(side_effect=ChannelFailure("Push failed")),
            _deliver_via_sms=AsyncMock(side_effect=ChannelFailure("SMS failed")),
            _deliver_via_voice=AsyncMock(side_effect=ChannelFailure("Voice failed")),
            _deliver_via_pager=AsyncMock(side_effect=ChannelFailure("Pager failed"))
        ) as mocked_channels:

            delivery_result = await delivery_service.deliver_alert(normal_alert, regular_user)

            # Should attempt all configured channels
            assert delivery_result.total_channels_attempted == 5
            assert delivery_result.successful_channels == 0
            assert delivery_result.delivery_success is False

            # Should trigger escalation
            assert delivery_result.escalation_triggered is True
            assert delivery_result.escalation_reason == "all_channels_failed"


class TestNetworkResilienceAndQueueing:
    """Test network resilience and message queueing."""

    @pytest.fixture
    def delivery_service(self):
        return MultiChannelDeliveryService(queue_enabled=True)

    @pytest.fixture
    def fault_injector(self):
        return FaultInjector()

    async def test_network_partition_queueing(self, delivery_service, fault_injector):
        """Test alert queueing during network partitions."""

        alert = self._create_normal_alert()
        user = self._create_test_user()

        # Simulate network partition
        fault_injector.inject_network_partition(60)

        with patch.object(delivery_service, '_is_network_available', return_value=False):

            delivery_result = await delivery_service.deliver_alert(alert, user)

            # Should queue the alert
            assert delivery_result.queued is True
            assert delivery_result.queue_position is not None
            assert delivery_result.estimated_retry_time is not None

            # Verify alert is in queue
            queue_status = await delivery_service.get_queue_status()
            assert queue_status.total_queued_alerts >= 1

    async def test_queue_persistence_during_failure(self, delivery_service, fault_injector):
        """Test that queued messages persist during system failures."""

        alerts = [self._create_test_alert(f"QUEUE_TEST_{i}") for i in range(10)]
        user = self._create_test_user()

        # Queue multiple alerts during network partition
        fault_injector.inject_network_partition(30)

        with patch.object(delivery_service, '_is_network_available', return_value=False):

            for alert in alerts:
                result = await delivery_service.deliver_alert(alert, user)
                assert result.queued is True

        # Simulate system restart
        await delivery_service.restart_service()

        # Verify queue persistence
        queue_status = await delivery_service.get_queue_status()
        assert queue_status.total_queued_alerts == 10

        # Network recovery
        fault_injector.clear_all_faults()

        with patch.object(delivery_service, '_is_network_available', return_value=True):
            # Process queued alerts
            await delivery_service.process_queued_alerts()

            # Verify all alerts were delivered
            final_queue_status = await delivery_service.get_queue_status()
            assert final_queue_status.total_queued_alerts == 0

    async def test_priority_queue_ordering(self, delivery_service, fault_injector):
        """Test that critical alerts are prioritized in queue."""

        # Create mixed priority alerts
        critical_alert = self._create_critical_alert()
        normal_alerts = [self._create_normal_alert() for _ in range(5)]
        user = self._create_test_user()

        # Queue during network partition
        fault_injector.inject_network_partition(30)

        with patch.object(delivery_service, '_is_network_available', return_value=False):

            # Queue normal alerts first
            for alert in normal_alerts:
                await delivery_service.deliver_alert(alert, user)

            # Queue critical alert last
            await delivery_service.deliver_alert(critical_alert, user)

        # Network recovery - process one alert
        fault_injector.clear_all_faults()

        with patch.object(delivery_service, '_is_network_available', return_value=True):

            next_alert = await delivery_service.get_next_queued_alert()

            # Critical alert should be processed first despite being queued last
            assert next_alert.alert_id == critical_alert.alert_id
            assert next_alert.alert_level == AlertLevel.CRITICAL


class TestHospitalSystemIntegration:
    """Test integration with hospital communication systems."""

    @pytest.fixture
    def delivery_service(self):
        return MultiChannelDeliveryService()

    async def test_pager_system_integration(self, delivery_service):
        """Test integration with hospital pager system."""

        urgent_alert = self._create_urgent_alert()
        doctor = self._create_doctor_user()

        with patch('hospital.pager_system.send_page') as mock_pager:
            mock_pager.return_value = {"success": True, "page_id": "PAGE_001"}

            delivery_result = await delivery_service.deliver_via_pager(urgent_alert, doctor)

            # Verify pager system called correctly
            mock_pager.assert_called_once()
            call_args = mock_pager.call_args[1]

            assert call_args["pager_number"] == doctor.contact_info["pager_number"]
            assert urgent_alert.alert_id in call_args["message"]
            assert delivery_result.delivery_success is True

    async def test_overhead_paging_integration(self, delivery_service):
        """Test integration with overhead paging system."""

        emergency_alert = self._create_emergency_alert()

        with patch('hospital.overhead_paging.announce') as mock_announce:
            mock_announce.return_value = {"success": True, "announcement_id": "ANNOUNCE_001"}

            delivery_result = await delivery_service.deliver_via_overhead_paging(emergency_alert)

            # Verify overhead paging called
            mock_announce.assert_called_once()
            announcement_text = mock_announce.call_args[0][0]

            assert "CODE BLUE" in announcement_text or "RAPID RESPONSE" in announcement_text
            assert delivery_result.delivery_success is True

    async def test_emr_integration_logging(self, delivery_service):
        """Test EMR integration for alert delivery logging."""

        alert = self._create_test_alert()
        user = self._create_test_user()

        with patch('hospital.emr.log_communication') as mock_emr_log:
            mock_emr_log.return_value = {"success": True, "log_id": "EMR_LOG_001"}

            delivery_result = await delivery_service.deliver_alert(alert, user)

            # Verify EMR logging
            mock_emr_log.assert_called_once()
            log_data = mock_emr_log.call_args[1]

            assert log_data["patient_id"] == alert.patient_id
            assert log_data["alert_type"] == alert.alert_level.value
            assert log_data["recipient_id"] == user.user_id


# Helper methods for test data creation
def _create_critical_alert() -> Alert:
    """Create critical alert for testing."""
    from uuid import uuid4
    from src.models.news2 import NEWS2Result, RiskCategory
    from src.models.patient import Patient
    from src.models.alerts import AlertDecision, AlertStatus

    news2_result = NEWS2Result(
        total_score=8,  # Critical
        individual_scores={
            "respiratory_rate": 2,
            "spo2": 2,
            "temperature": 1,
            "systolic_bp": 3,  # Critical parameter
            "heart_rate": 0,
            "consciousness": 0
        },
        risk_category=RiskCategory.HIGH,
        monitoring_frequency="continuous",
        scale_used=1,
        warnings=["Critical deterioration"],
        confidence=0.95,
        calculated_at=datetime.now(timezone.utc),
        calculation_time_ms=2.0
    )

    patient = Patient(
        patient_id="TEST_CRITICAL_001",
        ward_id="ICU",
        bed_number="ICU-01",
        age=65,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc) - timedelta(days=1),
        last_updated=datetime.now(timezone.utc)
    )

    alert_decision = AlertDecision(
        decision_id=uuid4(),
        patient_id=patient.patient_id,
        news2_result=news2_result,
        alert_level=AlertLevel.CRITICAL,
        alert_priority=AlertPriority.IMMEDIATE,
        threshold_applied=None,
        reasoning="Critical patient deterioration",
        decision_timestamp=datetime.now(timezone.utc),
        generation_latency_ms=5.0,
        single_param_trigger=True,
        suppressed=False,
        ward_id=patient.ward_id
    )

    return Alert(
        alert_id=uuid4(),
        patient_id=patient.patient_id,
        patient=patient,
        alert_decision=alert_decision,
        alert_level=AlertLevel.CRITICAL,
        alert_priority=AlertPriority.IMMEDIATE,
        title="Critical Patient Deterioration",
        message="Patient requires immediate intervention",
        clinical_context={
            "news2_total_score": news2_result.total_score,
            "critical_parameters": ["systolic_bp"]
        },
        created_at=datetime.now(timezone.utc),
        status=AlertStatus.PENDING,
        assigned_to=patient.assigned_nurse_id,
        acknowledged_at=None,
        acknowledged_by=None,
        escalation_step=0,
        max_escalation_step=4,
        next_escalation_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        resolved_at=None,
        resolved_by=None
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])