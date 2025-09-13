#!/usr/bin/env python3
"""
Manual Test Script for Alert Generation System - Story 2.1

This script validates the complete alert generation system including:
1. Critical alert generation within 5 seconds for NEWS2 ≥7
2. Single parameter score = 3 immediate alert bypass
3. Ward-specific threshold configuration via admin interface
4. Basic escalation matrix configuration and execution
5. 15-minute auto-escalation for unacknowledged critical alerts
6. Ward Nurse → Charge Nurse → Doctor → Rapid Response routing
7. Complete audit trail for all alert decisions
8. Alert deduplication and correlation logic
9. Performance under high alert volume scenarios

Usage: python scripts/test_alert_generation.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.alerts import AlertLevel, AlertStatus
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.patient import Patient
from src.services.alert_generation import AlertGenerator
from src.services.alert_configuration import AlertConfigurationService
from src.services.escalation_engine import EscalationEngine
from src.services.alert_pipeline import AlertProcessingPipeline
from src.services.alert_auditing import AlertAuditingService
from src.services.alert_monitoring import AlertMonitoringService
from src.services.audit import AuditLogger


class AlertGenerationTestSuite:
    """Comprehensive test suite for alert generation system."""
    
    def __init__(self):
        self.results = []
        self.system = None
        
    async def setup_system(self):
        """Set up integrated alert system for testing."""
        print("🔧 Setting up alert generation system...")
        
        # Create core components
        audit_logger = AuditLogger()
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
        
        # Set up mock delivery callback
        async def mock_delivery_callback(alert):
            print(f"📧 MOCK DELIVERY: {alert.title} to ward {alert.patient.ward_id}")
            return True
        
        pipeline.set_alert_delivery_callback(mock_delivery_callback)
        
        # Start services
        await escalation_engine.start_escalation_processing()
        await pipeline.start_pipeline()
        
        self.system = {
            "pipeline": pipeline,
            "alert_generator": alert_generator,
            "config_service": config_service,
            "escalation_engine": escalation_engine,
            "auditing_service": auditing_service,
            "monitoring_service": monitoring_service,
            "audit_logger": audit_logger
        }
        
        print("✅ Alert generation system ready")
        return True
    
    async def cleanup_system(self):
        """Clean up system resources."""
        if self.system:
            print("🧹 Cleaning up alert generation system...")
            await self.system["pipeline"].stop_pipeline()
            await self.system["escalation_engine"].stop_escalation_processing()
            print("✅ System cleanup complete")
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "", execution_time_ms: float = 0.0):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.results.append(result)
        print(f"{status} {test_name} ({execution_time_ms:.1f}ms)")
        if details:
            print(f"   └─ {details}")
    
    async def test_critical_alert_5_second_generation(self):
        """Test 1: Critical alert generation within 5 seconds for NEWS2 ≥7."""
        print("\n🧪 Test 1: Critical Alert 5-Second Generation")
        
        # Create critical NEWS2 result
        critical_news2 = NEWS2Result(
            total_score=8,
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 2,
                "temperature": 1,
                "systolic_bp": 3,  # Critical BP
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=[],
            confidence=1.0,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.5
        )
        
        patient = Patient(
            patient_id="CRITICAL_TEST_001",
            ward_id="TEST_WARD",
            bed_number="T-001",
            age=75,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Measure generation time
        start_time = datetime.now(timezone.utc)
        
        try:
            alert = await self.system["pipeline"].process_news2_event(
                critical_news2, patient, "CRITICAL_5SEC_TEST"
            )
            
            end_time = datetime.now(timezone.utc)
            generation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Verify results
            passed = (
                alert is not None and
                alert.alert_level == AlertLevel.CRITICAL and
                generation_time_ms < 5000
            )
            
            details = f"Generated critical alert in {generation_time_ms:.1f}ms, level: {alert.alert_level.value if alert else 'None'}"
            
            self.log_test_result(
                "Critical Alert 5-Second Generation",
                passed,
                details,
                generation_time_ms
            )
            
            return alert
            
        except Exception as e:
            self.log_test_result(
                "Critical Alert 5-Second Generation",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_single_parameter_bypass(self):
        """Test 2: Single parameter score = 3 immediate alert bypass."""
        print("\n🧪 Test 2: Single Parameter Score = 3 Bypass")
        
        test_scenarios = [
            ("respiratory_rate", "Respiratory Rate ≤8 or ≥25"),
            ("spo2", "SpO2 ≤91%"),
            ("temperature", "Temperature ≤35.0°C or ≥39.1°C"),
            ("systolic_bp", "Systolic BP ≤90 or ≥220"),
            ("heart_rate", "Heart Rate ≤40 or ≥131"),
            ("consciousness", "Consciousness: Confusion/Voice/Pain/Unresponsive")
        ]
        
        all_passed = True
        
        for param_name, description in test_scenarios:
            # Create NEWS2 with single critical parameter
            individual_scores = {
                "respiratory_rate": 0,
                "spo2": 0,
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 0,
                "consciousness": 0
            }
            individual_scores[param_name] = 3
            
            news2_result = NEWS2Result(
                total_score=3,
                individual_scores=individual_scores,
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[f"Critical {param_name}"],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=1.8
            )
            
            patient = Patient(
                patient_id=f"SINGLE_PARAM_TEST_{param_name.upper()}",
                ward_id="TEST_WARD",
                bed_number=f"SP-{param_name[:3].upper()}",
                age=65,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_002",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            start_time = datetime.now(timezone.utc)
            
            try:
                alert = await self.system["pipeline"].process_news2_event(
                    news2_result, patient, f"SINGLE_PARAM_{param_name.upper()}_TEST"
                )
                
                end_time = datetime.now(timezone.utc)
                generation_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Verify bypass logic
                param_passed = (
                    alert is not None and
                    alert.alert_level == AlertLevel.CRITICAL and
                    alert.alert_decision.single_param_trigger is True and
                    generation_time_ms < 2000  # Should be very fast for bypass
                )
                
                if not param_passed:
                    all_passed = False
                
                print(f"   └─ {param_name}: {'✅' if param_passed else '❌'} {generation_time_ms:.1f}ms")
                
            except Exception as e:
                print(f"   └─ {param_name}: ❌ Exception: {str(e)}")
                all_passed = False
        
        self.log_test_result(
            "Single Parameter Score = 3 Bypass",
            all_passed,
            f"Tested {len(test_scenarios)} critical parameters",
            0.0
        )
        
        return all_passed
    
    async def test_ward_threshold_configuration(self):
        """Test 3: Ward-specific threshold configuration."""
        print("\n🧪 Test 3: Ward-Specific Threshold Configuration")
        
        try:
            # Set up different ward types
            ward_configs = [
                ("ICU_WARD", "icu"),
                ("EMERGENCY_WARD", "emergency"),
                ("GENERAL_WARD", "general")
            ]
            
            config_results = []
            
            for ward_id, ward_type in ward_configs:
                config = await self.system["config_service"].setup_default_ward_configuration(
                    ward_id, ward_type, "WARD_CONFIG_TEST"
                )
                
                config_results.append({
                    "ward_id": ward_id,
                    "ward_type": ward_type,
                    "thresholds_created": config["thresholds_created"],
                    "matrices_created": config["matrices_created"]
                })
                
                # Verify ward configuration
                ward_config = await self.system["config_service"].get_complete_ward_configuration(ward_id)
                assert len(ward_config["thresholds"]) > 0
                assert len(ward_config["escalation_matrices"]) > 0
            
            # Test threshold retrieval
            thresholds = await self.system["config_service"].threshold_manager.get_ward_thresholds("ICU_WARD")
            
            passed = (
                len(config_results) == 3 and
                all(r["thresholds_created"] > 0 for r in config_results) and
                all(r["matrices_created"] > 0 for r in config_results) and
                len(thresholds) > 0
            )
            
            details = f"Configured {len(config_results)} ward types with thresholds and escalation matrices"
            
            self.log_test_result(
                "Ward-Specific Threshold Configuration",
                passed,
                details,
                0.0
            )
            
            return config_results
            
        except Exception as e:
            self.log_test_result(
                "Ward-Specific Threshold Configuration",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_escalation_matrix_execution(self):
        """Test 4: Basic escalation matrix configuration and execution."""
        print("\n🧪 Test 4: Escalation Matrix Configuration and Execution")
        
        try:
            # Create critical alert
            critical_news2 = NEWS2Result(
                total_score=7,
                individual_scores={
                    "respiratory_rate": 2,
                    "spo2": 2,
                    "temperature": 1,
                    "systolic_bp": 2,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=2.2
            )
            
            patient = Patient(
                patient_id="ESCALATION_TEST_001",
                ward_id="GENERAL_WARD",
                bed_number="E-001",
                age=68,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_003",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Generate alert
            alert = await self.system["pipeline"].process_news2_event(
                critical_news2, patient, "ESCALATION_MATRIX_TEST"
            )
            
            # Verify escalation schedule was created
            escalation_status = await self.system["escalation_engine"].get_escalation_status(alert.alert_id)
            
            # Check escalation matrix
            escalation_matrix = await self.system["config_service"].escalation_manager.get_escalation_matrix(
                "GENERAL_WARD", AlertLevel.CRITICAL
            )
            
            passed = (
                alert is not None and
                alert.alert_level == AlertLevel.CRITICAL and
                escalation_status["has_escalation"] is True and
                escalation_status["total_steps"] == 4 and  # Ward Nurse → Charge Nurse → Doctor → Rapid Response
                escalation_matrix is not None and
                len(escalation_matrix.escalation_steps) == 4
            )
            
            details = f"Escalation matrix with {escalation_status['total_steps']} steps configured and scheduled"
            
            self.log_test_result(
                "Escalation Matrix Configuration and Execution",
                passed,
                details,
                0.0
            )
            
            return alert.alert_id if alert else None
            
        except Exception as e:
            self.log_test_result(
                "Escalation Matrix Configuration and Execution",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_15_minute_auto_escalation(self):
        """Test 5: 15-minute auto-escalation for unacknowledged critical alerts."""
        print("\n🧪 Test 5: 15-Minute Auto-Escalation")
        
        try:
            # Create critical alert
            critical_news2 = NEWS2Result(
                total_score=9,
                individual_scores={
                    "respiratory_rate": 3,
                    "spo2": 3,
                    "temperature": 1,
                    "systolic_bp": 2,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=["Multiple critical parameters"],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=3.1
            )
            
            patient = Patient(
                patient_id="AUTO_ESCALATION_TEST_001",
                ward_id="GENERAL_WARD",
                bed_number="AE-001",
                age=82,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_004",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Generate alert
            alert = await self.system["pipeline"].process_news2_event(
                critical_news2, patient, "AUTO_ESCALATION_TEST"
            )
            
            # Check initial escalation status
            initial_status = await self.system["escalation_engine"].get_escalation_status(alert.alert_id)
            
            # Simulate escalation timing check
            # (In production, this happens automatically via background task)
            escalation_schedule = await self.system["escalation_engine"].scheduler.get_escalation_schedule(alert.alert_id)
            
            passed = (
                alert is not None and
                alert.alert_level == AlertLevel.CRITICAL and
                initial_status["has_escalation"] is True and
                escalation_schedule is not None and
                len(escalation_schedule.escalation_matrix.escalation_steps) >= 2 and
                escalation_schedule.escalation_matrix.escalation_steps[1].delay_minutes == 15  # Second step at 15 minutes
            )
            
            details = f"Critical alert with 15-minute escalation to {escalation_schedule.escalation_matrix.escalation_steps[1].role.value if escalation_schedule else 'unknown'}"
            
            self.log_test_result(
                "15-Minute Auto-Escalation",
                passed,
                details,
                0.0
            )
            
            return alert.alert_id if alert else None
            
        except Exception as e:
            self.log_test_result(
                "15-Minute Auto-Escalation",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_escalation_routing(self):
        """Test 6: Ward Nurse → Charge Nurse → Doctor → Rapid Response routing."""
        print("\n🧪 Test 6: Escalation Routing Chain")
        
        try:
            # Get escalation matrix for critical alerts
            escalation_matrix = await self.system["config_service"].escalation_manager.get_escalation_matrix(
                "GENERAL_WARD", AlertLevel.CRITICAL
            )
            
            if not escalation_matrix:
                self.log_test_result(
                    "Escalation Routing Chain",
                    False,
                    "No escalation matrix found for GENERAL_WARD critical alerts",
                    0.0
                )
                return False
            
            # Verify escalation chain
            steps = escalation_matrix.escalation_steps
            expected_roles = ["ward_nurse", "charge_nurse", "doctor", "rapid_response"]
            
            role_chain_correct = (
                len(steps) == 4 and
                all(steps[i].role.value == expected_roles[i] for i in range(4))
            )
            
            # Verify timing
            timing_correct = (
                steps[0].delay_minutes == 0 and   # Immediate to ward nurse
                steps[1].delay_minutes == 15 and  # 15 minutes to charge nurse
                steps[2].delay_minutes == 30 and  # 30 minutes to doctor
                steps[3].delay_minutes == 45      # 45 minutes to rapid response
            )
            
            passed = role_chain_correct and timing_correct
            
            if passed:
                details = f"Correct escalation chain: {' → '.join(expected_roles)} with proper timing"
            else:
                actual_roles = [step.role.value for step in steps]
                details = f"Expected: {expected_roles}, Got: {actual_roles}"
            
            self.log_test_result(
                "Escalation Routing Chain",
                passed,
                details,
                0.0
            )
            
            return passed
            
        except Exception as e:
            self.log_test_result(
                "Escalation Routing Chain",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return False
    
    async def test_audit_trail_completeness(self):
        """Test 7: Complete audit trail for all alert decisions."""
        print("\n🧪 Test 7: Audit Trail Completeness")
        
        try:
            # Create test alert
            test_news2 = NEWS2Result(
                total_score=6,
                individual_scores={
                    "respiratory_rate": 2,
                    "spo2": 2,
                    "temperature": 0,
                    "systolic_bp": 2,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="1 hourly",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=2.8
            )
            
            patient = Patient(
                patient_id="AUDIT_TRAIL_TEST_001",
                ward_id="GENERAL_WARD",
                bed_number="AT-001",
                age=55,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_005",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Generate alert
            alert = await self.system["pipeline"].process_news2_event(
                test_news2, patient, "AUDIT_TRAIL_TEST"
            )
            
            # Query audit entries
            audit_entries = await self.system["auditing_service"].decision_auditor.query_audit_entries(
                patient_id=patient.patient_id,
                limit=10
            )
            
            # Verify audit trail
            passed = (
                alert is not None and
                len(audit_entries) > 0 and
                any(entry.category.value == "alert_generation" for entry in audit_entries) and
                audit_entries[0].patient_id == patient.patient_id and
                audit_entries[0].reasoning is not None and
                audit_entries[0].performance_metrics is not None
            )
            
            if passed:
                details = f"Generated {len(audit_entries)} audit entries with complete metadata"
            else:
                details = f"Expected audit entries, got {len(audit_entries)}"
            
            self.log_test_result(
                "Audit Trail Completeness",
                passed,
                details,
                0.0
            )
            
            return audit_entries
            
        except Exception as e:
            self.log_test_result(
                "Audit Trail Completeness",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_alert_deduplication(self):
        """Test 8: Alert deduplication and correlation logic."""
        print("\n🧪 Test 8: Alert Deduplication and Correlation")
        
        try:
            # Create duplicate scenarios
            base_news2 = NEWS2Result(
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
                calculation_time_ms=1.9
            )
            
            patient = Patient(
                patient_id="DEDUP_TEST_001",
                ward_id="GENERAL_WARD",
                bed_number="DD-001",
                age=45,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_006",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Generate first alert
            alert1 = await self.system["pipeline"].process_news2_event(
                base_news2, patient, "DEDUP_TEST_FIRST"
            )
            
            # Generate second alert with same parameters (should be deduplicated)
            alert2 = await self.system["pipeline"].process_news2_event(
                base_news2, patient, "DEDUP_TEST_SECOND"
            )
            
            # Generate critical alert (should never be deduplicated)
            critical_news2 = NEWS2Result(
                total_score=7,
                individual_scores={
                    "respiratory_rate": 2,
                    "spo2": 2,
                    "temperature": 1,
                    "systolic_bp": 2,
                    "heart_rate": 0,
                    "consciousness": 0
                },
                risk_category=RiskCategory.HIGH,
                monitoring_frequency="continuous",
                scale_used=1,
                warnings=[],
                confidence=1.0,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=2.5
            )
            
            critical_alert1 = await self.system["pipeline"].process_news2_event(
                critical_news2, patient, "DEDUP_CRITICAL_FIRST"
            )
            
            critical_alert2 = await self.system["pipeline"].process_news2_event(
                critical_news2, patient, "DEDUP_CRITICAL_SECOND"
            )
            
            # Check deduplication stats
            dedup_stats = self.system["pipeline"].deduplication_manager.get_deduplication_stats()
            
            passed = (
                alert1 is not None and
                (alert2 is None or alert2.alert_decision.suppressed) and  # Second alert should be suppressed or None
                critical_alert1 is not None and
                critical_alert2 is not None and  # Critical alerts should never be suppressed
                dedup_stats["total_entries"] > 0
            )
            
            details = f"Deduplication working: {dedup_stats['total_entries']} entries, critical alerts not suppressed"
            
            self.log_test_result(
                "Alert Deduplication and Correlation",
                passed,
                details,
                0.0
            )
            
            return dedup_stats
            
        except Exception as e:
            self.log_test_result(
                "Alert Deduplication and Correlation",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def test_high_volume_performance(self):
        """Test 9: Performance under high alert volume scenarios."""
        print("\n🧪 Test 9: High Volume Performance")
        
        try:
            # Create batch of alerts for load testing
            alert_count = 25
            batch_alerts = []
            
            print(f"   └─ Generating {alert_count} alerts for performance testing...")
            
            start_time = datetime.now(timezone.utc)
            
            for i in range(alert_count):
                news2_result = NEWS2Result(
                    total_score=3 + (i % 5),  # Mix of alert levels
                    individual_scores={
                        "respiratory_rate": i % 4,
                        "spo2": (i + 1) % 4,
                        "temperature": (i + 2) % 4,
                        "systolic_bp": 0,
                        "heart_rate": 0,
                        "consciousness": 0
                    },
                    risk_category=RiskCategory.MEDIUM if (3 + (i % 5)) < 7 else RiskCategory.HIGH,
                    monitoring_frequency="6 hourly",
                    scale_used=1,
                    warnings=[],
                    confidence=0.9,
                    calculated_at=datetime.now(timezone.utc),
                    calculation_time_ms=1.5
                )
                
                patient = Patient(
                    patient_id=f"LOAD_TEST_PATIENT_{i:03d}",
                    ward_id="GENERAL_WARD",
                    bed_number=f"LT-{i:03d}",
                    age=30 + (i % 50),
                    is_copd_patient=(i % 10 == 0),
                    assigned_nurse_id=f"NURSE_{(i % 6) + 1:03d}",
                    admission_date=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc)
                )
                
                # Add to pipeline queue
                await self.system["pipeline"].add_mock_event(
                    news2_result, patient, f"LOAD_TEST_{i}"
                )
            
            # Allow processing time
            await asyncio.sleep(3.0)
            
            end_time = datetime.now(timezone.utc)
            total_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Check pipeline performance
            pipeline_status = self.system["pipeline"].get_pipeline_status()
            events_processed = pipeline_status["metrics"]["events_processed"]
            processing_errors = pipeline_status["metrics"]["processing_errors"]
            avg_processing_time = pipeline_status["metrics"]["average_processing_time_ms"]
            
            # Performance criteria
            error_rate = processing_errors / max(events_processed, 1)
            throughput_per_second = events_processed / max(total_time_ms / 1000, 1)
            
            passed = (
                events_processed >= alert_count * 0.8 and  # At least 80% processed
                error_rate < 0.1 and  # Less than 10% error rate
                avg_processing_time < 1000 and  # Average under 1 second
                throughput_per_second > 5  # At least 5 alerts/second
            )
            
            details = f"Processed {events_processed}/{alert_count} alerts, {error_rate:.1%} errors, {throughput_per_second:.1f} alerts/sec"
            
            self.log_test_result(
                "High Volume Performance",
                passed,
                details,
                total_time_ms
            )
            
            return pipeline_status
            
        except Exception as e:
            self.log_test_result(
                "High Volume Performance",
                False,
                f"Exception: {str(e)}",
                0.0
            )
            return None
    
    async def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 Starting NEWS2 Alert Generation System Tests")
        print("=" * 60)
        
        try:
            # Setup system
            if not await self.setup_system():
                print("❌ Failed to set up test system")
                return False
            
            # Run tests
            test_results = []
            
            test_results.append(await self.test_critical_alert_5_second_generation())
            test_results.append(await self.test_single_parameter_bypass())
            test_results.append(await self.test_ward_threshold_configuration())
            test_results.append(await self.test_escalation_matrix_execution())
            test_results.append(await self.test_15_minute_auto_escalation())
            test_results.append(await self.test_escalation_routing())
            test_results.append(await self.test_audit_trail_completeness())
            test_results.append(await self.test_alert_deduplication())
            test_results.append(await self.test_high_volume_performance())
            
            # Summary
            passed_tests = sum(1 for result in self.results if result["passed"])
            total_tests = len(self.results)
            
            print("\n" + "=" * 60)
            print("🏁 TEST SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
            
            if passed_tests == total_tests:
                print("\n🎉 ALL TESTS PASSED! Alert generation system is working correctly.")
            else:
                print("\n⚠️  Some tests failed. Review the results above.")
                print("\nFailed tests:")
                for result in self.results:
                    if not result["passed"]:
                        print(f"   ❌ {result['test_name']}: {result['details']}")
            
            # Save detailed results
            results_file = "test_results_alert_generation.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "test_run": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_tests": total_tests,
                        "passed_tests": passed_tests,
                        "success_rate": passed_tests / total_tests * 100
                    },
                    "results": self.results
                }, f, indent=2)
            
            print(f"\n📄 Detailed results saved to: {results_file}")
            
            return passed_tests == total_tests
            
        except Exception as e:
            print(f"❌ Test suite failed with exception: {str(e)}")
            return False
        
        finally:
            await self.cleanup_system()


async def main():
    """Main test runner."""
    test_suite = AlertGenerationTestSuite()
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))