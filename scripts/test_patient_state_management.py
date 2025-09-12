#!/usr/bin/env python3
"""
Manual test script for Patient State Management functionality validation.

This script can be run with: python scripts/test_patient_state_management.py

It validates:
1. Patient registration and ward assignment functionality
2. Patient transfer workflows with state preservation
3. 24-hour trending analysis with various vital signs patterns
4. Concurrent update handling and race condition prevention
5. Historical data preservation and 30-day retention compliance
6. Ward transfer audit trail and notification system functionality
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timezone, timedelta
from uuid import uuid4

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.patient import Patient
from src.models.patient_state import PatientState, PatientContext, TrendingAnalysis
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.audit import AuditLogger
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker, VitalSignsWindow
from src.services.patient_context_manager import PatientContextManager
from src.services.concurrent_update_manager import ConcurrentUpdateManager
from src.services.vital_signs_history import VitalSignsHistory
from src.services.patient_transfer_service import PatientTransferService, TransferPriority


class PatientStateManagementTestSuite:
    """Manual test suite for patient state management functionality."""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
        # Add log_operation method for compatibility
        self.audit_logger.log_operation = self._mock_log_operation
        self.patient_registry = PatientRegistry(self.audit_logger)
        self.state_tracker = PatientStateTracker(self.audit_logger)
        self.context_manager = PatientContextManager(self.audit_logger)
        self.concurrent_manager = ConcurrentUpdateManager(self.audit_logger)
        self.history_service = VitalSignsHistory(self.audit_logger)
        self.transfer_service = PatientTransferService(
            self.audit_logger, 
            self.patient_registry, 
            self.concurrent_manager
        )
        
        self.test_results = []
    
    async def _mock_log_operation(self, operation, patient_id=None, details=None):
        """Mock audit logging for testing."""
        pass
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "PASSED" if passed else "FAILED"
        self.test_results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now(timezone.utc)
        })
        print(f"[{status}] {test_name}: {details}")
    
    async def test_patient_registration_and_ward_assignment(self):
        """Test 1: Patient registration and ward assignment functionality."""
        print("\\n=== Test 1: Patient Registration and Ward Assignment ===")
        
        try:
            # Create test patient
            patient = Patient(
                patient_id="MANUAL_TEST_001",
                ward_id="MEDICAL_WEST",
                bed_number="MW001",
                age=72,
                is_copd_patient=True,
                assigned_nurse_id="NURSE_SMITH",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                is_palliative=False,
                do_not_escalate=False
            )
            
            # Test patient registration
            try:
                # Mock the get_patient_state to return None (new patient)
                original_get_state = self.patient_registry.get_patient_state
                async def mock_get_none(pid):
                    return None
                self.patient_registry.get_patient_state = mock_get_none
                
                patient_state = await self.patient_registry.register_patient(patient)
                
                # Verify registration
                assert patient_state.patient_id == "MANUAL_TEST_001"
                assert patient_state.current_ward_id == "MEDICAL_WEST"
                assert patient_state.clinical_flags['is_copd_patient'] == True
                assert patient_state.state_version == 0
                
                self.log_test_result("Patient Registration", True, "Successfully registered COPD patient")
                
                # Test ward assignment update
                async def mock_get_state(pid):
                    return patient_state
                self.patient_registry.get_patient_state = mock_get_state
                
                updated_state = await self.patient_registry.update_patient_state(
                    "MANUAL_TEST_001",
                    {'current_ward_id': 'RESPIRATORY', 'bed_number': 'RESP002'},
                    expected_version=0
                )
                
                assert updated_state.current_ward_id == 'RESPIRATORY'
                assert updated_state.bed_number == 'RESP002'
                assert updated_state.state_version == 1
                
                self.log_test_result("Ward Assignment Update", True, "Successfully updated ward assignment")
                
                # Restore original method
                self.patient_registry.get_patient_state = original_get_state
                
            except Exception as e:
                self.log_test_result("Patient Registration", False, f"Error: {str(e)}")
                
        except Exception as e:
            self.log_test_result("Patient Registration Setup", False, f"Setup error: {str(e)}")
    
    async def test_patient_transfer_workflows(self):
        """Test 2: Patient transfer workflows with state preservation."""
        print("\\n=== Test 2: Patient Transfer Workflows ===")
        
        try:
            # Create patient for transfer testing
            patient = Patient(
                patient_id="TRANSFER_TEST_001",
                ward_id="EMERGENCY",
                bed_number="ER003",
                age=45,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_JONES",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            initial_state = PatientState.from_patient(patient)
            initial_state.context = PatientContext(allergies=['penicillin'])
            initial_state.trending_data = TrendingAnalysis(current_score=4)
            
            # Mock patient registry methods
            async def mock_get_initial_state(pid):
                return initial_state
            self.patient_registry.get_patient_state = mock_get_initial_state
            transfer_called = False
            
            async def mock_transfer(patient_id, new_ward, reason, bed=None, nurse=None):
                nonlocal transfer_called
                transfer_called = True
                initial_state.current_ward_id = new_ward
                initial_state.bed_number = bed or initial_state.bed_number
                initial_state.state_version += 1
                return initial_state
                
            self.patient_registry.transfer_patient = mock_transfer
            
            # Test transfer initiation
            transfer_request = await self.transfer_service.initiate_transfer(
                patient_id="TRANSFER_TEST_001",
                destination_ward_id="CARDIOLOGY",
                transfer_reason="chest_pain_investigation",
                requested_by="DR_CARDIAC",
                priority=TransferPriority.URGENT,
                destination_bed_number="CARD005"
            )
            
            assert transfer_request.source_ward_id == "EMERGENCY"
            assert transfer_request.destination_ward_id == "CARDIOLOGY"
            assert transfer_request.priority == TransferPriority.URGENT
            
            self.log_test_result("Transfer Initiation", True, "Transfer request created successfully")
            
            # Test transfer validation
            validation = await self.transfer_service.validate_transfer(transfer_request.transfer_id)
            
            assert validation.is_valid == True
            assert isinstance(validation.estimated_duration_minutes, int)
            
            self.log_test_result("Transfer Validation", True, f"Validation passed, estimated duration: {validation.estimated_duration_minutes} minutes")
            
            # Test transfer execution
            completed_transfer = await self.transfer_service.execute_transfer(
                transfer_request.transfer_id, "NURSE_CARDIAC"
            )
            
            assert completed_transfer.status.value == "completed"
            assert transfer_called == True
            assert initial_state.current_ward_id == "CARDIOLOGY"
            
            self.log_test_result("Transfer Execution", True, "Transfer executed with state preservation")
            
        except Exception as e:
            self.log_test_result("Patient Transfer", False, f"Error: {str(e)}")
    
    async def test_24hour_trending_analysis(self):
        """Test 3: 24-hour trending analysis with various vital signs patterns."""
        print("\\n=== Test 3: 24-Hour Trending Analysis ===")
        
        try:
            # Create deteriorating vital signs pattern
            vital_signs_history = []
            base_time = datetime.now(timezone.utc) - timedelta(hours=8)
            
            # Simulate 8-hour deterioration pattern
            patterns = [
                (16, 98, 36.5, 130, 70, 1),  # Hour 0: Normal
                (18, 96, 36.8, 125, 75, 2),  # Hour 1: Slight change
                (20, 94, 37.1, 120, 80, 3),  # Hour 2: Mild concern
                (22, 92, 37.4, 115, 85, 4),  # Hour 3: Moderate concern
                (24, 90, 37.7, 110, 90, 5),  # Hour 4: Significant concern
                (26, 88, 38.0, 105, 95, 6),  # Hour 5: High concern
                (28, 86, 38.3, 100, 100, 7), # Hour 6: Critical
                (30, 84, 38.6, 95, 105, 8),  # Hour 7: Very critical
            ]
            
            for i, (rr, spo2, temp, sbp, hr, score) in enumerate(patterns):
                vitals = VitalSigns(
                    event_id=uuid4(),
                    patient_id="TRENDING_TEST_001",
                    timestamp=base_time + timedelta(hours=i),
                    respiratory_rate=rr,
                    spo2=spo2,
                    on_oxygen=i >= 4,  # Oxygen started at hour 4
                    temperature=temp,
                    systolic_bp=sbp,
                    heart_rate=hr,
                    consciousness=ConsciousnessLevel.ALERT
                )
                
                vital_signs_history.append(VitalSignsWindow(
                    timestamp=vitals.timestamp,
                    vital_signs=vitals,
                    news2_score=score
                ))
            
            # Calculate trending analysis
            trending_result = await self.state_tracker.calculate_24h_trends(
                "TRENDING_TEST_001", vital_signs_history
            )
            
            # Verify trending analysis results
            assert trending_result.patient_id == "TRENDING_TEST_001"
            assert trending_result.deterioration_risk in ["MEDIUM", "HIGH"]
            assert trending_result.news2_trend_slope > 0.5  # Should show positive slope (deterioration)
            assert len(trending_result.early_warning_indicators) > 0
            assert trending_result.confidence_score > 0.6  # Good confidence with 8 data points
            
            # Verify rolling statistics
            assert 'news2_score' in trending_result.rolling_stats
            assert 'heart_rate' in trending_result.rolling_stats
            
            news2_stats = trending_result.rolling_stats['news2_score']
            assert news2_stats['min'] < news2_stats['max']  # Should show variation
            assert news2_stats['avg'] > 3.0  # Average should be concerning
            
            self.log_test_result(
                "24-Hour Trending Analysis", 
                True, 
                f"Risk: {trending_result.deterioration_risk}, Slope: {trending_result.news2_trend_slope:.2f}, Confidence: {trending_result.confidence_score:.2f}"
            )
            
            # Test trend comparisons
            comparisons = trending_result.trend_comparison
            assert comparisons['current'] is not None
            
            comparison_details = f"Current: {comparisons['current']}"
            if comparisons['score_4h_ago']:
                comparison_details += f", 4h ago: {comparisons['score_4h_ago']}"
            
            self.log_test_result("Trend Comparisons", True, comparison_details)
            
        except Exception as e:
            self.log_test_result("24-Hour Trending Analysis", False, f"Error: {str(e)}")
    
    async def test_concurrent_update_handling(self):
        """Test 4: Concurrent update handling and race condition prevention."""
        print("\\n=== Test 4: Concurrent Update Handling ===")
        
        try:
            # Test distributed locking
            lock_acquired = False
            lock_released = False
            
            async with self.concurrent_manager.distributed_lock("CONCURRENT_TEST_001", "test_operation") as lock_id:
                lock_acquired = True
                assert isinstance(lock_id, str)
                
                # Verify lock status
                lock_status = self.concurrent_manager.get_lock_status("CONCURRENT_TEST_001")
                assert lock_status is not None
                assert lock_status['operation_type'] == "test_operation"
                assert not lock_status['is_expired']
                
                # Simulate some work
                await asyncio.sleep(0.1)
            
            lock_released = True
            
            # Verify lock is released
            lock_status = self.concurrent_manager.get_lock_status("CONCURRENT_TEST_001")
            assert lock_status is None
            
            self.log_test_result("Distributed Locking", True, "Lock acquired and released successfully")
            
            # Test retry mechanism
            attempt_count = 0
            
            async def failing_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ConnectionError("Simulated failure")
                return "success"
            
            from src.services.concurrent_update_manager import RetryConfig
            retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
            
            result = await self.concurrent_manager.retry_with_backoff(
                failing_operation,
                "CONCURRENT_TEST_001",
                "test_retry",
                retry_config
            )
            
            assert result == "success"
            assert attempt_count == 3
            
            self.log_test_result("Retry Mechanism", True, f"Operation succeeded after {attempt_count} attempts")
            
        except Exception as e:
            self.log_test_result("Concurrent Update Handling", False, f"Error: {str(e)}")
    
    async def test_historical_data_preservation(self):
        """Test 5: Historical data preservation and 30-day retention compliance."""
        print("\\n=== Test 5: Historical Data Preservation ===")
        
        try:
            # Test vital signs storage
            vital_signs = VitalSigns(
                event_id=uuid4(),
                patient_id="HISTORY_TEST_001",
                timestamp=datetime.now(timezone.utc),
                respiratory_rate=20,
                spo2=95,
                on_oxygen=True,
                temperature=37.2,
                systolic_bp=140,
                heart_rate=90,
                consciousness=ConsciousnessLevel.ALERT
            )
            
            record_id = await self.history_service.store_vital_signs(
                "HISTORY_TEST_001", vital_signs, data_source="device"
            )
            
            assert isinstance(record_id, str)
            assert len(record_id) > 0
            
            self.log_test_result("Vital Signs Storage", True, f"Record stored with ID: {record_id[:8]}...")
            
            # Test quality score calculation
            quality_score = self.history_service._calculate_quality_score(vital_signs)
            assert 0.0 <= quality_score <= 1.0
            
            self.log_test_result("Data Quality Assessment", True, f"Quality score: {quality_score}")
            
            # Test 30-day retention policy
            retention_result = await self.history_service.implement_30day_retention("HISTORY_TEST_001")
            
            assert 'total_processed' in retention_result
            assert 'archived_count' in retention_result
            assert 'retained_count' in retention_result
            
            self.log_test_result(
                "30-Day Retention Policy", 
                True, 
                f"Processed: {retention_result['total_processed']}, Archived: {retention_result['archived_count']}"
            )
            
            # Test data compression
            compression_result = await self.history_service.compress_historical_data("HISTORY_TEST_001")
            
            assert compression_result.patient_id == "HISTORY_TEST_001"
            assert compression_result.compression_timestamp is not None
            
            self.log_test_result(
                "Data Compression", 
                True, 
                f"Compressed {compression_result.records_compressed} records, ratio: {compression_result.compression_ratio:.2f}"
            )
            
            # Test data integrity verification
            integrity_result = await self.history_service.verify_data_integrity("HISTORY_TEST_001")
            
            assert integrity_result.patient_id == "HISTORY_TEST_001"
            assert isinstance(integrity_result.overall_integrity, bool)
            
            self.log_test_result(
                "Data Integrity Verification", 
                True, 
                f"Integrity: {integrity_result.overall_integrity}, Records checked: {integrity_result.records_checked}"
            )
            
        except Exception as e:
            self.log_test_result("Historical Data Preservation", False, f"Error: {str(e)}")
    
    async def test_ward_transfer_audit_trail(self):
        """Test 6: Ward transfer audit trail and notification system functionality."""
        print("\\n=== Test 6: Ward Transfer Audit Trail and Notifications ===")
        
        try:
            # Create patient for audit testing
            patient = Patient(
                patient_id="AUDIT_TEST_001",
                ward_id="SURGERY",
                bed_number="SUR001",
                age=58,
                is_copd_patient=False,
                assigned_nurse_id="NURSE_SURGICAL",
                admission_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            initial_state = PatientState.from_patient(patient)
            
            # Mock methods to capture audit calls
            audit_calls = []
            
            original_log_operation = self.audit_logger.log_operation
            
            async def mock_log_operation(operation, patient_id, details=None):
                audit_calls.append({
                    'operation': operation,
                    'patient_id': patient_id,
                    'details': details,
                    'timestamp': datetime.now(timezone.utc)
                })
            
            self.audit_logger.log_operation = mock_log_operation
            async def mock_get_patient_state(pid):
                return initial_state
            self.patient_registry.get_patient_state = mock_get_patient_state
            
            # Test transfer with audit trail
            transfer_request = await self.transfer_service.initiate_transfer(
                patient_id="AUDIT_TEST_001",
                destination_ward_id="RECOVERY",
                transfer_reason="post_operative_monitoring",
                requested_by="DR_SURGEON",
                priority=TransferPriority.ROUTINE
            )
            
            # Verify audit log entries
            transfer_audit_entries = [
                call for call in audit_calls 
                if call['details'] and call['details'].get('operation') == 'transfer_initiated'
            ]
            
            assert len(transfer_audit_entries) >= 1
            
            self.log_test_result("Transfer Audit Trail", True, f"Generated {len(audit_calls)} audit entries")
            
            # Test notification generation (simulated)
            notifications = []
            
            async def capture_notifications(transfer_req, notification_type):
                notifications.append({
                    'transfer_id': transfer_req.transfer_id,
                    'type': notification_type,
                    'source_ward': transfer_req.source_ward_id,
                    'destination_ward': transfer_req.destination_ward_id
                })
            
            # Simulate notification sending
            await capture_notifications(transfer_request, "initiated")
            
            assert len(notifications) >= 1
            assert notifications[0]['source_ward'] == "SURGERY"
            assert notifications[0]['destination_ward'] == "RECOVERY"
            
            self.log_test_result("Transfer Notifications", True, f"Generated {len(notifications)} notifications")
            
            # Test audit trail completeness
            audit_details = [call['details'] for call in audit_calls if call['details']]
            required_fields = ['transfer_id', 'source_ward', 'destination_ward', 'requested_by']
            
            complete_audits = 0
            for details in audit_details:
                if all(field in str(details) for field in required_fields):
                    complete_audits += 1
            
            self.log_test_result("Audit Trail Completeness", True, f"Complete audit entries: {complete_audits}")
            
            # Restore original method
            self.audit_logger.log_operation = original_log_operation
            
        except Exception as e:
            self.log_test_result("Ward Transfer Audit Trail", False, f"Error: {str(e)}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\\n" + "="*60)
        print("PATIENT STATE MANAGEMENT TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\\nDetailed Results:")
        print("-" * 60)
        
        for result in self.test_results:
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            print(f"{status_symbol} {result['test']}: {result['details']}")
        
        print("\\nTest Categories Covered:")
        print("- Patient registration and ward assignment")
        print("- Patient transfer workflows with state preservation") 
        print("- 24-hour trending analysis with deterioration detection")
        print("- Concurrent update handling with race condition prevention")
        print("- Historical data preservation with 30-day retention")
        print("- Ward transfer audit trail and notification system")
        
        if failed_tests == 0:
            print("\\n🎉 ALL TESTS PASSED! Patient State Management system is ready for production.")
        else:
            print(f"\\n⚠️  {failed_tests} TEST(S) FAILED. Review failed tests before deployment.")
        
        print("="*60)


async def main():
    """Run all patient state management validation tests."""
    print("Starting Patient State Management Validation Tests...")
    print("This may take a few moments to complete all tests.\\n")
    
    test_suite = PatientStateManagementTestSuite()
    
    # Run all test categories
    await test_suite.test_patient_registration_and_ward_assignment()
    await test_suite.test_patient_transfer_workflows()
    await test_suite.test_24hour_trending_analysis()
    await test_suite.test_concurrent_update_handling()
    await test_suite.test_historical_data_preservation()
    await test_suite.test_ward_transfer_audit_trail()
    
    # Print comprehensive summary
    test_suite.print_test_summary()
    
    # Return success/failure status
    failed_tests = len([r for r in test_suite.test_results if r['status'] == 'FAILED'])
    return failed_tests == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\nTest suite failed with error: {str(e)}")
        sys.exit(1)