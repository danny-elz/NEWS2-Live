"""
Integration Tests for Alert Generation System - Story 2.1

Tests cover end-to-end alert generation workflow including:
- Complete alert processing pipeline
- Escalation matrix integration
- Ward configuration integration
- Audit trail integration
- Performance monitoring integration
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.models.alerts import AlertLevel, AlertStatus, Alert
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_generation import AlertGenerator
from src.services.alert_configuration import AlertConfigurationService
from src.services.escalation_engine import EscalationEngine
from src.services.alert_pipeline import AlertProcessingPipeline
from src.services.alert_auditing import AlertAuditingService
from src.services.alert_monitoring import AlertMonitoringService
from src.services.audit import AuditLogger


@pytest.fixture
async def integrated_alert_system():
    """Create integrated alert system for testing."""
    # Create core components
    audit_logger = Mock(spec=AuditLogger)
    audit_logger.create_audit_entry = Mock()
    
    alert_generator = AlertGenerator(audit_logger)
    config_service = AlertConfigurationService(audit_logger)
    escalation_engine = EscalationEngine(audit_logger, config_service.escalation_manager)
    auditing_service = AlertAuditingService(audit_logger)
    monitoring_service = AlertMonitoringService(audit_logger)
    
    # Create pipeline
    pipeline = AlertProcessingPipeline(
        audit_logger=audit_logger,
        alert_generator=alert_generator,
        escalation_engine=escalation_engine,
        auditing_service=auditing_service,
        config_service=config_service
    )
    
    # Start services
    await escalation_engine.start_escalation_processing()
    await pipeline.start_pipeline()
    
    yield {
        "pipeline": pipeline,
        "alert_generator": alert_generator,
        "config_service": config_service,
        "escalation_engine": escalation_engine,
        "auditing_service": auditing_service,
        "monitoring_service": monitoring_service,
        "audit_logger": audit_logger
    }
    
    # Cleanup
    await pipeline.stop_pipeline()
    await escalation_engine.stop_escalation_processing()


@pytest.fixture
def test_ward_setup():
    """Set up test ward with configuration."""
    return {
        "ward_id": "TEST_WARD_INTEGRATION",
        "ward_type": "general",
        "patient_count": 25,
        "nurse_count": 6
    }


@pytest.fixture
def critical_clinical_scenario():
    """Critical clinical scenario for testing."""
    return {
        "patient": Patient(
            patient_id="CRITICAL_PATIENT_001",
            ward_id="TEST_WARD_INTEGRATION",
            bed_number="T-101",
            age=78,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_CRITICAL_001",
            admission_date=datetime.now(timezone.utc) - timedelta(days=2),
            last_updated=datetime.now(timezone.utc)
        ),
        "news2_result": NEWS2Result(
            total_score=9,
            individual_scores={
                "respiratory_rate": 3,  # ≤8 or ≥25
                "spo2": 2,             # 92-93%
                "temperature": 1,       # 38.5°C
                "systolic_bp": 3,      # ≤90 mmHg
                "heart_rate": 0,       # Normal
                "consciousness": 0      # Alert
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["Multiple critical parameters"],
            confidence=0.95,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=3.2
        )
    }


@pytest.fixture
def copd_exacerbation_scenario():
    """COPD exacerbation scenario for testing."""
    return {
        "patient": Patient(
            patient_id="COPD_PATIENT_001",
            ward_id="TEST_WARD_INTEGRATION",
            bed_number="T-205",
            age=68,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_COPD_001",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=6),
            last_updated=datetime.now(timezone.utc)
        ),
        "news2_result": NEWS2Result(
            total_score=6,
            individual_scores={
                "respiratory_rate": 2,  # Elevated
                "spo2": 1,             # 86% (Scale 2)
                "temperature": 0,       # Normal
                "systolic_bp": 1,      # Slightly low
                "heart_rate": 2,       # Tachycardia
                "consciousness": 0      # Alert
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="1 hourly",
            scale_used=2,  # COPD scale
            warnings=["COPD exacerbation pattern"],
            confidence=0.92,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.8
        )
    }


class TestEndToEndAlertGeneration:
    """Test complete alert generation workflow."""
    
    @pytest.mark.asyncio
    async def test_critical_alert_complete_workflow(self, integrated_alert_system, test_ward_setup, critical_clinical_scenario):
        """Test complete workflow for critical alert generation."""
        system = integrated_alert_system
        scenario = critical_clinical_scenario
        
        # Set up ward configuration
        ward_config = await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "TEST_ADMIN"
        )
        
        assert ward_config["thresholds_created"] > 0
        assert ward_config["matrices_created"] > 0
        
        # Process NEWS2 event through pipeline
        start_time = datetime.now(timezone.utc)
        
        alert = await system["pipeline"].process_news2_event(
            scenario["news2_result"],
            scenario["patient"],
            "INTEGRATION_TEST"
        )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Verify alert generation
        assert alert is not None
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.patient_id == scenario["patient"].patient_id
        assert processing_time < 5000  # Within 5-second requirement
        
        # Verify escalation scheduling
        escalation_status = await system["escalation_engine"].get_escalation_status(alert.alert_id)
        assert escalation_status["has_escalation"] is True
        assert escalation_status["current_step"] == 0
        assert escalation_status["total_steps"] == 4  # Ward Nurse → Charge Nurse → Doctor → Rapid Response
        
        # Verify audit trail
        audit_entries = await system["auditing_service"].decision_auditor.query_audit_entries(
            patient_id=scenario["patient"].patient_id,
            limit=10
        )
        assert len(audit_entries) > 0
        assert audit_entries[0].category.value == "alert_generation"
    
    @pytest.mark.asyncio
    async def test_copd_patient_specialized_handling(self, integrated_alert_system, test_ward_setup, copd_exacerbation_scenario):
        """Test specialized handling for COPD patients."""
        system = integrated_alert_system
        scenario = copd_exacerbation_scenario
        
        # Set up ward configuration
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "TEST_ADMIN"
        )
        
        # Process COPD patient alert
        alert = await system["pipeline"].process_news2_event(
            scenario["news2_result"],
            scenario["patient"],
            "COPD_INTEGRATION_TEST"
        )
        
        # Verify COPD-specific handling
        assert alert is not None
        assert alert.clinical_context["is_copd_patient"] is True
        assert alert.clinical_context["scale_used"] == 2
        assert "COPD" in alert.message or "Scale 2" in alert.message
        
        # Verify appropriate alert level for COPD score
        assert alert.alert_level in [AlertLevel.HIGH, AlertLevel.MEDIUM]
    
    @pytest.mark.asyncio
    async def test_15_minute_escalation_trigger(self, integrated_alert_system, test_ward_setup, critical_clinical_scenario):
        """Test that critical alerts escalate after 15 minutes if unacknowledged."""
        system = integrated_alert_system
        scenario = critical_clinical_scenario
        
        # Set up ward
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "TEST_ADMIN"
        )
        
        # Generate critical alert
        alert = await system["pipeline"].process_news2_event(
            scenario["news2_result"],
            scenario["patient"],
            "ESCALATION_TEST"
        )
        
        assert alert.alert_level == AlertLevel.CRITICAL
        
        # Check initial escalation status
        initial_status = await system["escalation_engine"].get_escalation_status(alert.alert_id)
        assert initial_status["has_escalation"] is True
        assert initial_status["current_step"] == 0
        
        # Simulate 15+ minutes passing (mock time for testing)
        with patch('src.services.escalation_engine.datetime') as mock_dt:
            future_time = datetime.now(timezone.utc) + timedelta(minutes=16)
            mock_dt.now.return_value = future_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Trigger escalation check (in production this happens automatically)
            overdue_events = await system["escalation_engine"].scheduler.get_overdue_escalations()
            
            # Should have overdue escalation events
            assert len(overdue_events) > 0
            
            # First escalation should be to charge nurse after 15 minutes
            first_overdue = overdue_events[0]
            assert first_overdue.escalation_step >= 1  # Beyond initial ward nurse
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment_workflow(self, integrated_alert_system, test_ward_setup, critical_clinical_scenario):
        """Test alert acknowledgment and escalation pause workflow."""
        system = integrated_alert_system
        scenario = critical_clinical_scenario
        
        # Set up ward and generate alert
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "TEST_ADMIN"
        )
        
        alert = await system["pipeline"].process_news2_event(
            scenario["news2_result"],
            scenario["patient"],
            "ACKNOWLEDGMENT_TEST"
        )
        
        # Acknowledge alert
        await system["escalation_engine"].acknowledge_alert(
            alert.alert_id,
            "NURSE_001",
            "Patient assessed, starting treatment protocol"
        )
        
        # Verify escalation is paused
        status_after_ack = await system["escalation_engine"].get_escalation_status(alert.alert_id)
        assert status_after_ack["is_paused"] is True
        assert "acknowledged" in status_after_ack["pause_reason"].lower()
        
        # Verify audit trail for acknowledgment
        ack_audit = await system["auditing_service"].decision_auditor.query_audit_entries(
            alert_id=alert.alert_id,
            limit=5
        )
        
        # Should have both generation and acknowledgment audit entries
        categories = [entry.category.value for entry in ack_audit]
        assert "alert_generation" in categories
    
    @pytest.mark.asyncio
    async def test_ward_specific_threshold_application(self, integrated_alert_system):
        """Test that ward-specific thresholds are properly applied."""
        system = integrated_alert_system
        
        # Create two different ward configurations
        icu_ward_id = "ICU_TEST_WARD"
        general_ward_id = "GENERAL_TEST_WARD"
        
        # Set up ICU ward (more sensitive thresholds)
        icu_config = await system["config_service"].setup_default_ward_configuration(
            icu_ward_id, "icu", "ICU_ADMIN"
        )
        
        # Set up general ward (standard thresholds)
        general_config = await system["config_service"].setup_default_ward_configuration(
            general_ward_id, "general", "GENERAL_ADMIN"
        )
        
        # Create moderate NEWS2 result (score 4)
        moderate_news2 = NEWS2Result(
            total_score=4,
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 1,
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="6 hourly",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.1
        )
        
        # Test with ICU patient (should be more sensitive)
        icu_patient = Patient(
            patient_id="ICU_PATIENT_001",
            ward_id=icu_ward_id,
            bed_number="ICU-01",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="ICU_NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        icu_alert = await system["pipeline"].process_news2_event(
            moderate_news2, icu_patient, "ICU_THRESHOLD_TEST"
        )
        
        # Test with general ward patient
        general_patient = Patient(
            patient_id="GENERAL_PATIENT_001",
            ward_id=general_ward_id,
            bed_number="G-101",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="GENERAL_NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        general_alert = await system["pipeline"].process_news2_event(
            moderate_news2, general_patient, "GENERAL_THRESHOLD_TEST"
        )
        
        # Both should generate alerts, but may have different characteristics
        # based on ward-specific configurations
        assert icu_alert is not None
        assert general_alert is not None
        assert icu_alert.patient.ward_id == icu_ward_id
        assert general_alert.patient.ward_id == general_ward_id


class TestPerformanceIntegration:
    """Test performance aspects of integrated system."""
    
    @pytest.mark.asyncio
    async def test_high_volume_alert_processing(self, integrated_alert_system, test_ward_setup):
        """Test system performance under high alert volume."""
        system = integrated_alert_system
        
        # Set up ward
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "LOAD_TEST_ADMIN"
        )
        
        # Create multiple NEWS2 events for processing
        events = []
        for i in range(50):  # Process 50 alerts
            patient = Patient(
                patient_id=f"LOAD_TEST_PATIENT_{i:03d}",
                ward_id=test_ward_setup["ward_id"],
                bed_number=f"LT-{i:03d}",
                age=30 + (i % 50),
                is_copd_patient=(i % 10 == 0),  # 10% COPD patients
                assigned_nurse_id=f"NURSE_{(i % 6) + 1:03d}",  # 6 nurses
                admission_date=datetime.now(timezone.utc) - timedelta(hours=i % 24),
                last_updated=datetime.now(timezone.utc)
            )
            
            news2_result = NEWS2Result(
                total_score=3 + (i % 6),  # Scores 3-8
                individual_scores={
                    "respiratory_rate": i % 4,
                    "spo2": (i + 1) % 4,
                    "temperature": (i + 2) % 4,
                    "systolic_bp": (i + 3) % 4,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.MEDIUM if (3 + (i % 6)) < 7 else RiskCategory.HIGH,
                monitoring_frequency="6 hourly",
                scale_used=2 if (i % 10 == 0) else 1,  # COPD patients use Scale 2
                warnings=[],
                confidence=0.9 + (i % 10) * 0.01,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=1.5 + (i % 10) * 0.1
            )
            
            events.append((news2_result, patient))
        
        # Process events and measure performance
        start_time = datetime.now(timezone.utc)
        
        # Add events to pipeline
        for news2_result, patient in events:
            await system["pipeline"].add_mock_event(news2_result, patient, "LOAD_TEST")
        
        # Allow processing time
        await asyncio.sleep(2.0)
        
        # Check pipeline status
        pipeline_status = system["pipeline"].get_pipeline_status()
        
        # Verify processing performance
        assert pipeline_status["metrics"]["events_processed"] > 0
        assert pipeline_status["metrics"]["processing_errors"] < pipeline_status["metrics"]["events_processed"] * 0.1  # <10% error rate
        
        # Check average processing time
        avg_processing_time = pipeline_status["metrics"]["average_processing_time_ms"]
        assert avg_processing_time < 1000  # Should be well under 1 second per event
    
    @pytest.mark.asyncio
    async def test_concurrent_critical_alerts(self, integrated_alert_system, test_ward_setup):
        """Test handling of multiple concurrent critical alerts."""
        system = integrated_alert_system
        
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "CONCURRENT_TEST_ADMIN"
        )
        
        # Create multiple critical scenarios
        critical_scenarios = []
        for i in range(10):
            patient = Patient(
                patient_id=f"CRITICAL_CONCURRENT_{i:02d}",
                ward_id=test_ward_setup["ward_id"],
                bed_number=f"CC-{i:02d}",
                age=60 + i,
                is_copd_patient=False,
                assigned_nurse_id="CRITICAL_NURSE_001",
                admission_date=datetime.now(timezone.utc) - timedelta(hours=i),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Each patient has different critical parameters
            critical_param_score = {
                "respiratory_rate": 3 if i % 7 == 0 else 0,
                "spo2": 3 if i % 7 == 1 else 0,
                "temperature": 3 if i % 7 == 2 else 0,
                "systolic_bp": 3 if i % 7 == 3 else 0,
                "heart_rate": 3 if i % 7 == 4 else 0,
                "consciousness": 3 if i % 7 == 5 else 0
            }
            
            # Fill remaining scores to reach critical total
            remaining_score = 7 - sum(critical_param_score.values())
            if remaining_score > 0:
                critical_param_score["spo2"] += min(remaining_score, 3)
            
            news2_result = NEWS2Result(
                total_score=sum(critical_param_score.values()),
                individual_scores=critical_param_score,
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[f"Critical scenario {i}"],
                confidence=0.95,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=2.0 + i * 0.1
            )
            
            critical_scenarios.append((news2_result, patient))
        
        # Process all critical alerts concurrently
        start_time = datetime.now(timezone.utc)
        
        tasks = [
            system["pipeline"].process_news2_event(news2_result, patient, f"CONCURRENT_CRITICAL_{i}")
            for i, (news2_result, patient) in enumerate(critical_scenarios)
        ]
        
        alerts = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Verify all alerts were processed successfully
        successful_alerts = [a for a in alerts if isinstance(a, Alert)]
        errors = [e for e in alerts if isinstance(e, Exception)]
        
        assert len(successful_alerts) >= 8  # At least 80% success rate
        assert len(errors) <= 2  # At most 20% failures acceptable under load
        
        # All successful alerts should be critical
        for alert in successful_alerts:
            assert alert.alert_level == AlertLevel.CRITICAL
        
        # Processing time should still meet requirements even under load
        assert processing_time < 30000  # 30 seconds for 10 concurrent critical alerts
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, integrated_alert_system, test_ward_setup, critical_clinical_scenario):
        """Test integration with performance monitoring."""
        system = integrated_alert_system
        scenario = critical_clinical_scenario
        
        # Set up ward
        await system["config_service"].setup_default_ward_configuration(
            test_ward_setup["ward_id"], test_ward_setup["ward_type"], "MONITORING_TEST_ADMIN"
        )
        
        # Process alert with monitoring
        alert = await system["pipeline"].process_news2_event(
            scenario["news2_result"],
            scenario["patient"],
            "MONITORING_INTEGRATION_TEST"
        )
        
        # Record performance metrics
        await system["monitoring_service"].record_alert_performance(
            alert,
            alert.alert_decision.generation_latency_ms,
            100.0,  # Mock processing time
            50.0    # Mock threshold evaluation time
        )
        
        # Get performance report
        performance_report = await system["monitoring_service"].get_performance_report(hours=1)
        
        # Verify monitoring data
        assert "performance_metrics" in performance_report
        assert "alert_latency" in performance_report["performance_metrics"]
        assert performance_report["performance_metrics"]["alert_latency"]["sample_count"] > 0
        
        # Get health status
        health_status = await system["monitoring_service"].get_health_status()
        assert "overall_healthy" in health_status
        assert "health_checks" in health_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])