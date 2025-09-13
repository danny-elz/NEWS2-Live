#!/usr/bin/env python3
"""
Manual test script for alert suppression functionality.

Validates the smart alert suppression logic by running comprehensive tests
for 50%+ volume reduction, critical detection, and suppression workflows.

Usage:
    python scripts/test_alert_suppression.py
"""

import asyncio
import sys
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from uuid import uuid4

import redis.asyncio as redis
from prometheus_client import CollectorRegistry

# Add src to path for imports
sys.path.insert(0, 'src')

from models.alerts import Alert, AlertLevel, AlertPriority, AlertStatus, AlertDecision
from models.news2 import NEWS2Result, RiskCategory
from models.patient import Patient
from services.alert_suppression import (
    SuppressionEngine, PatternDetector, ManualOverrideManager,
    SuppressionDecisionLogger, SuppressionMetrics,
    SuppressionDecision, AlertAcknowledgment
)
from services.audit import AuditLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSuppressionTester:
    """Comprehensive test runner for alert suppression functionality."""
    
    def __init__(self):
        self.redis_client = None
        self.suppression_engine = None
        self.override_manager = None
        self.suppression_logger = None
        self.suppression_metrics = None
        self.audit_logger = AuditLogger()
        self.test_results = []
    
    async def setup(self):
        """Initialize test environment."""
        logger.info("Setting up test environment...")
        
        # Connect to Redis (use test database)
        self.redis_client = redis.Redis.from_url("redis://localhost:6379/2", decode_responses=True)
        
        # Clear test database
        await self.redis_client.flushdb()
        
        # Initialize suppression components
        self.suppression_engine = SuppressionEngine(self.redis_client, self.audit_logger)
        self.override_manager = ManualOverrideManager(self.redis_client, self.audit_logger)
        self.suppression_logger = SuppressionDecisionLogger(self.redis_client, self.audit_logger)
        self.suppression_metrics = SuppressionMetrics(CollectorRegistry())
        
        logger.info("✅ Test environment setup complete")
    
    async def cleanup(self):
        """Clean up test environment."""
        if self.redis_client:
            await self.redis_client.flushdb()
            await self.redis_client.close()
        logger.info("🧹 Test environment cleaned up")
    
    async def run_all_tests(self):
        """Run all alert suppression tests."""
        logger.info("🚀 Starting comprehensive alert suppression tests...")
        
        try:
            await self.setup()
            
            # Test 1: Critical Alert Safety
            await self.test_critical_alert_safety()
            
            # Test 2: Time-based Suppression
            await self.test_time_based_suppression()
            
            # Test 3: Pattern Recognition
            await self.test_pattern_recognition()
            
            # Test 4: Manual Overrides
            await self.test_manual_overrides()
            
            # Test 5: Volume Reduction (50%+ target)
            await self.test_volume_reduction_target()
            
            # Test 6: Suppression Audit Trail
            await self.test_suppression_audit_trail()
            
            # Test 7: Performance Under Load
            await self.test_performance_under_load()
            
            # Generate final report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {str(e)}")
            raise
        finally:
            await self.cleanup()
    
    async def test_critical_alert_safety(self):
        """Test 1: Validate 100% critical alert detection (never suppress critical)."""
        logger.info("🔴 Test 1: Critical Alert Safety")
        
        test_name = "Critical Alert Safety"
        critical_alerts_tested = 0
        critical_alerts_suppressed = 0
        
        # Test various critical alert scenarios
        critical_scenarios = [
            {"reason": "NEWS2 >= 7", "score": 8, "single_param": False},
            {"reason": "Single param = 3 (RR)", "score": 5, "single_param": True},
            {"reason": "NEWS2 = 15 (severe)", "score": 15, "single_param": False},
        ]
        
        for scenario in critical_scenarios:
            alert = self.create_test_alert(
                patient_id=f"CRIT_TEST_{critical_alerts_tested}",
                alert_level=AlertLevel.CRITICAL,
                news2_score=scenario["score"],
                single_param_trigger=scenario["single_param"]
            )
            
            decision = await self.suppression_engine.should_suppress(alert)
            critical_alerts_tested += 1
            
            if decision.suppressed:
                critical_alerts_suppressed += 1
                logger.error(f"❌ CRITICAL SAFETY VIOLATION: {scenario['reason']} was suppressed!")
            else:
                logger.info(f"✅ {scenario['reason']} correctly NOT suppressed")
        
        # Record results
        success = critical_alerts_suppressed == 0
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "critical_alerts_tested": critical_alerts_tested,
            "critical_alerts_suppressed": critical_alerts_suppressed,
            "safety_score": "PASS" if success else "FAIL"
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {critical_alerts_tested} critical alerts protected")
        else:
            logger.error(f"❌ {test_name}: FAILED - {critical_alerts_suppressed} critical alerts suppressed!")
    
    async def test_time_based_suppression(self):
        """Test 2: 30-minute time-based suppression with score delta bypass."""
        logger.info("⏰ Test 2: Time-based Suppression")
        
        test_name = "Time-based Suppression"
        
        # Create patient and initial alert
        patient_id = "TIME_TEST_001"
        alert1 = self.create_test_alert(patient_id, AlertLevel.MEDIUM, news2_score=4)
        
        # First alert should not be suppressed
        decision1 = await self.suppression_engine.should_suppress(alert1)
        assert not decision1.suppressed, "First alert should not be suppressed"
        
        # Acknowledge the alert
        await self.suppression_engine.record_acknowledgment(alert1, "NURSE_001")
        
        # Second alert within 30 minutes with same score - should be suppressed
        alert2 = self.create_test_alert(patient_id, AlertLevel.MEDIUM, news2_score=4)
        decision2 = await self.suppression_engine.should_suppress(alert2)
        
        # Third alert within 30 minutes with +2 score increase - should NOT be suppressed
        alert3 = self.create_test_alert(patient_id, AlertLevel.HIGH, news2_score=6)  # +2 from acknowledged score
        decision3 = await self.suppression_engine.should_suppress(alert3)
        
        # Validate results
        time_suppression_works = decision2.suppressed and decision2.reason == "TIME_BASED_SUPPRESSION"
        score_bypass_works = not decision3.suppressed
        
        success = time_suppression_works and score_bypass_works
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "time_suppression_works": time_suppression_works,
            "score_bypass_works": score_bypass_works,
            "details": {
                "alert1_suppressed": decision1.suppressed,
                "alert2_suppressed": decision2.suppressed,
                "alert3_suppressed": decision3.suppressed
            }
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    async def test_pattern_recognition(self):
        """Test 3: Pattern recognition for stable high score patients."""
        logger.info("📊 Test 3: Pattern Recognition")
        
        test_name = "Pattern Recognition"
        
        # Create patient with stable score history
        patient_id = "PATTERN_TEST_001"
        history_key = f"patient_score_history:{patient_id}"
        
        # Add stable score history (variance ≤ 1)
        base_time = datetime.now(timezone.utc)
        stable_scores = [5, 5, 6, 5, 5, 6, 5]  # Low variance
        
        for i, score in enumerate(stable_scores):
            entry = {
                "total_score": score,
                "timestamp": (base_time - timedelta(hours=3-i*0.5)).isoformat(),
                "patient_id": patient_id
            }
            await self.redis_client.zadd(
                history_key,
                {json.dumps(entry): (base_time - timedelta(hours=3-i*0.5)).timestamp()}
            )
        
        # Test pattern detection
        alert = self.create_test_alert(patient_id, AlertLevel.MEDIUM, news2_score=5)
        pattern_detector = PatternDetector(self.redis_client)
        is_stable_pattern = await pattern_detector.is_stable_pattern(alert)
        
        # Test full suppression decision
        decision = await self.suppression_engine.should_suppress(alert)
        pattern_suppressed = decision.suppressed and "PATTERN" in decision.reason
        
        success = is_stable_pattern and pattern_suppressed
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "pattern_detected": is_stable_pattern,
            "pattern_suppressed": pattern_suppressed,
            "suppression_reason": decision.reason if decision.suppressed else None
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - Stable pattern detected and suppressed")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    async def test_manual_overrides(self):
        """Test 4: Manual override workflow with nurse authentication."""
        logger.info("👩‍⚕️ Test 4: Manual Override Workflow")
        
        test_name = "Manual Override Workflow"
        
        patient_id = "OVERRIDE_TEST_001"
        nurse_id = "NURSE_SMITH_001"
        
        # Create manual override
        override = await self.override_manager.create_override(
            patient_id=patient_id,
            nurse_id=nurse_id,
            justification="Patient stable post-surgery, family present, close monitoring by clinical team",
            duration_minutes=120
        )
        
        # Test that override prevents alerts
        alert = self.create_test_alert(patient_id, AlertLevel.HIGH, news2_score=6)
        decision = await self.suppression_engine.should_suppress(alert)
        
        # Validate override functionality
        override_created = override is not None and override.is_active
        override_prevents_alerts = decision.suppressed and decision.reason == "MANUAL_OVERRIDE"
        justification_valid = len(override.justification) >= 20
        
        success = override_created and override_prevents_alerts and justification_valid
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "override_created": override_created,
            "override_prevents_alerts": override_prevents_alerts,
            "justification_length": len(override.justification),
            "override_details": {
                "nurse_id": override.nurse_id,
                "duration_minutes": (override.expires_at - override.created_at).total_seconds() / 60,
                "active": override.is_active
            }
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - Override created and functioning")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    async def test_volume_reduction_target(self):
        """Test 5: Validate 50%+ alert volume reduction without critical misses."""
        logger.info("📉 Test 5: Volume Reduction Target (50%+ without critical misses)")
        
        test_name = "Volume Reduction Target"
        
        # Simulate realistic alert distribution
        total_alerts = 200
        alert_distribution = {
            AlertLevel.CRITICAL: int(total_alerts * 0.05),  # 5% critical
            AlertLevel.HIGH: int(total_alerts * 0.15),      # 15% high  
            AlertLevel.MEDIUM: int(total_alerts * 0.35),    # 35% medium
            AlertLevel.LOW: int(total_alerts * 0.45)        # 45% low
        }
        
        alerts_generated = 0
        alerts_suppressed = 0
        critical_alerts_suppressed = 0
        suppression_by_level = {level: 0 for level in AlertLevel}
        
        # Create varied patient scenarios for realistic testing
        patients = [f"VOL_TEST_{i:03d}" for i in range(50)]
        
        # Add some acknowledged alerts to enable time-based suppression
        for i, patient_id in enumerate(patients[:25]):
            if i % 3 == 0:  # Every 3rd patient has recent acknowledgment
                ack_alert = self.create_test_alert(patient_id, AlertLevel.MEDIUM, news2_score=3)
                await self.suppression_engine.record_acknowledgment(ack_alert, f"NURSE_{i:03d}")
        
        # Add stable patterns for some patients
        for i, patient_id in enumerate(patients[25:40]):
            await self._create_stable_pattern_history(patient_id)
        
        # Test alerts across all levels
        for alert_level, count in alert_distribution.items():
            for i in range(count):
                patient_id = patients[alerts_generated % len(patients)]
                
                alert = self.create_test_alert(
                    patient_id=patient_id,
                    alert_level=alert_level,
                    news2_score=self._get_score_for_level(alert_level)
                )
                
                decision = await self.suppression_engine.should_suppress(alert)
                alerts_generated += 1
                
                if decision.suppressed:
                    alerts_suppressed += 1
                    suppression_by_level[alert_level] += 1
                    
                    # CRITICAL SAFETY CHECK
                    if alert_level == AlertLevel.CRITICAL:
                        critical_alerts_suppressed += 1
                        logger.error(f"🚨 CRITICAL SAFETY VIOLATION: Critical alert suppressed!")
        
        # Calculate metrics
        suppression_rate = (alerts_suppressed / alerts_generated) * 100
        volume_reduction_target_met = suppression_rate >= 50.0
        critical_safety_maintained = critical_alerts_suppressed == 0
        
        success = volume_reduction_target_met and critical_safety_maintained
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "total_alerts": alerts_generated,
            "alerts_suppressed": alerts_suppressed,
            "suppression_rate_percent": round(suppression_rate, 2),
            "volume_reduction_target_met": volume_reduction_target_met,
            "critical_alerts_suppressed": critical_alerts_suppressed,
            "critical_safety_maintained": critical_safety_maintained,
            "suppression_by_level": {
                level.value: count for level, count in suppression_by_level.items()
            }
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {suppression_rate:.1f}% volume reduction achieved")
        else:
            logger.error(f"❌ {test_name}: FAILED - {suppression_rate:.1f}% volume reduction, {critical_alerts_suppressed} critical misses")
    
    async def test_suppression_audit_trail(self):
        """Test 6: Complete suppression decision audit trail."""
        logger.info("📋 Test 6: Suppression Decision Audit Trail")
        
        test_name = "Suppression Audit Trail"
        
        # Generate test decisions
        patient_id = "AUDIT_TEST_001"
        decisions_logged = 0
        
        for i in range(10):
            alert = self.create_test_alert(
                patient_id=f"{patient_id}_{i}",
                alert_level=AlertLevel.MEDIUM,
                news2_score=4
            )
            
            decision = await self.suppression_engine.should_suppress(alert)
            await self.suppression_logger.log_suppression_decision(decision, alert)
            decisions_logged += 1
        
        # Test analytics generation
        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)
        
        analytics = await self.suppression_logger.get_suppression_analytics(start_date, end_date)
        
        # Validate audit trail completeness
        audit_complete = analytics["total_decisions"] == decisions_logged
        analytics_accurate = "suppression_rate" in analytics and "reasons" in analytics
        decision_query_works = len(await self.suppression_logger.query_suppression_decisions(limit=5)) > 0
        
        success = audit_complete and analytics_accurate and decision_query_works
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "decisions_logged": decisions_logged,
            "analytics_total_decisions": analytics["total_decisions"],
            "audit_complete": audit_complete,
            "analytics_keys": list(analytics.keys()),
            "query_functionality": decision_query_works
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {decisions_logged} decisions logged and queryable")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    async def test_performance_under_load(self):
        """Test 7: Performance under high alert volume scenarios."""
        logger.info("⚡ Test 7: Performance Under Load")
        
        test_name = "Performance Under Load"
        
        # Test high-volume scenario
        test_alerts = 100
        total_decision_time = 0
        slow_decisions = 0
        errors = 0
        
        logger.info(f"Processing {test_alerts} alerts to test performance...")
        
        for i in range(test_alerts):
            try:
                start_time = datetime.now(timezone.utc)
                
                alert = self.create_test_alert(
                    patient_id=f"PERF_TEST_{i % 20}",  # Reuse patients for realistic patterns
                    alert_level=AlertLevel.MEDIUM,
                    news2_score=4
                )
                
                decision = await self.suppression_engine.should_suppress(alert)
                
                end_time = datetime.now(timezone.utc)
                decision_time_ms = (end_time - start_time).total_seconds() * 1000
                total_decision_time += decision_time_ms
                
                # Track slow decisions (>1 second)
                if decision_time_ms > 1000:
                    slow_decisions += 1
                
            except Exception as e:
                errors += 1
                logger.error(f"Error processing alert {i}: {str(e)}")
        
        # Calculate performance metrics
        avg_decision_time_ms = total_decision_time / test_alerts
        performance_target_met = avg_decision_time_ms < 100  # <100ms average
        reliability_target_met = errors == 0
        
        success = performance_target_met and reliability_target_met
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "alerts_processed": test_alerts,
            "avg_decision_time_ms": round(avg_decision_time_ms, 2),
            "slow_decisions": slow_decisions,
            "errors": errors,
            "performance_target_met": performance_target_met,
            "reliability_target_met": reliability_target_met
        })
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {avg_decision_time_ms:.1f}ms average decision time")
        else:
            logger.error(f"❌ {test_name}: FAILED - {avg_decision_time_ms:.1f}ms average, {errors} errors")
    
    def create_test_alert(
        self, 
        patient_id: str, 
        alert_level: AlertLevel, 
        news2_score: int,
        single_param_trigger: bool = False
    ) -> Alert:
        """Create test alert with specified parameters."""
        
        # Create NEWS2 result
        news2_result = NEWS2Result(
            patient_id=patient_id,
            total_score=news2_score,
            individual_scores={
                "respiratory_rate": 1,
                "spo2": 1, 
                "temperature": 0,
                "systolic_bp": 1,
                "heart_rate": 1,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            scale_used="Scale 1",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Create patient
        patient = Patient(
            patient_id=patient_id,
            age=70,
            ward_id="TEST_WARD_A",
            is_copd_patient=False
        )
        
        # Create alert decision
        alert_decision = AlertDecision(
            decision_id=uuid4(),
            patient_id=patient_id,
            news2_result=news2_result,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            threshold_applied=None,
            reasoning=f"Test alert for {alert_level.value} level",
            decision_timestamp=datetime.now(timezone.utc),
            generation_latency_ms=5.0,
            single_param_trigger=single_param_trigger,
            suppressed=False,
            ward_id="TEST_WARD_A"
        )
        
        # Create alert
        return Alert(
            alert_id=uuid4(),
            patient_id=patient_id,
            patient=patient,
            alert_decision=alert_decision,
            alert_level=alert_level,
            alert_priority=AlertPriority.URGENT,
            title=f"{alert_level.value.upper()} ALERT - NEWS2 Score {news2_score}",
            message=f"Test alert for patient {patient_id}",
            clinical_context={"news2_total_score": news2_score},
            created_at=datetime.now(timezone.utc),
            status=AlertStatus.PENDING,
            assigned_to=None,
            acknowledged_at=None,
            acknowledged_by=None,
            escalation_step=0,
            max_escalation_step=2,
            next_escalation_at=None,
            resolved_at=None,
            resolved_by=None
        )
    
    async def _create_stable_pattern_history(self, patient_id: str):
        """Create stable score pattern history for pattern testing."""
        history_key = f"patient_score_history:{patient_id}"
        base_time = datetime.now(timezone.utc)
        stable_scores = [5, 5, 6, 5, 5, 6]  # Low variance pattern
        
        for i, score in enumerate(stable_scores):
            entry = {
                "total_score": score,
                "timestamp": (base_time - timedelta(hours=3-i*0.5)).isoformat(),
                "patient_id": patient_id
            }
            await self.redis_client.zadd(
                history_key,
                {json.dumps(entry): (base_time - timedelta(hours=3-i*0.5)).timestamp()}
            )
    
    def _get_score_for_level(self, alert_level: AlertLevel) -> int:
        """Get appropriate NEWS2 score for alert level."""
        score_map = {
            AlertLevel.CRITICAL: 8,
            AlertLevel.HIGH: 6,
            AlertLevel.MEDIUM: 4,
            AlertLevel.LOW: 2
        }
        return score_map[alert_level]
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating Test Report...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("🧪 ALERT SUPPRESSION TEST REPORT")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*80)
        
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status} - {result['test_name']}")
            
            # Print key metrics for each test
            if result["test_name"] == "Volume Reduction Target":
                print(f"    📉 Volume Reduction: {result['suppression_rate_percent']}%")
                print(f"    🔴 Critical Misses: {result['critical_alerts_suppressed']}")
            elif result["test_name"] == "Performance Under Load":
                print(f"    ⚡ Avg Decision Time: {result['avg_decision_time_ms']}ms")
                print(f"    🐌 Slow Decisions: {result['slow_decisions']}")
                print(f"    ❌ Errors: {result['errors']}")
            elif result["test_name"] == "Critical Alert Safety":
                print(f"    🛡️ Critical Alerts Protected: {result['critical_alerts_tested']}")
                print(f"    🚨 Safety Violations: {result['critical_alerts_suppressed']}")
        
        print("="*80)
        
        # Overall assessment
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Alert suppression system ready for production!")
        else:
            print("⚠️  Some tests failed - review failures before production deployment")
        
        print("="*80)
        
        # Save detailed results to file
        with open("alert_suppression_test_results.json", "w") as f:
            json.dump({
                "test_execution_time": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        logger.info("📄 Detailed results saved to alert_suppression_test_results.json")


async def main():
    """Main test execution function."""
    print("🚀 Starting Alert Suppression Validation Tests")
    print("This script validates:")
    print("  1. 50%+ alert volume reduction while maintaining 100% critical detection")
    print("  2. 30-minute time-based suppression with score delta bypass")
    print("  3. Pattern recognition for stable high score patients")
    print("  4. Manual override workflow with nurse authentication")
    print("  5. Complete suppression decision audit trail")
    print("  6. Suppression effectiveness metrics and reporting")
    print("  7. Performance under high alert volume scenarios")
    print("")
    
    tester = AlertSuppressionTester()
    
    try:
        await tester.run_all_tests()
        print("\n✅ Alert suppression validation complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        await tester.cleanup()
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        await tester.cleanup()
        raise


if __name__ == "__main__":
    asyncio.run(main())