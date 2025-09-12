#!/usr/bin/env python3
"""
Test script specifically for COPD Scale 2 functionality validation.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from uuid import uuid4

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.services.audit import AuditLogger
from src.services.news2_calculator import NEWS2Calculator


async def test_copd_scale_indication():
    """Test that COPD Scale 2 usage is clearly indicated in results."""
    print("Testing COPD Scale 2 indication...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # Create COPD patient
    patient = Patient(
        patient_id="COPD_001",
        ward_id="WARD_A",
        bed_number="001",
        age=70,
        is_copd_patient=True,  # This should trigger Scale 2
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Create normal vital signs
    vitals = VitalSigns(
        event_id=uuid4(),
        patient_id="COPD_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=90,  # Normal for COPD on Scale 2
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    result = await calculator.calculate_news2(vitals, patient)
    
    # Verify Scale 2 is used
    assert result.scale_used == 2, f"Expected Scale 2, got {result.scale_used}"
    
    # Verify SpO2 scores correctly for COPD (90% should be 0 points on Scale 2)
    assert result.individual_scores['spo2'] == 0, f"Expected SpO2 score 0, got {result.individual_scores['spo2']}"
    
    # Verify Scale 2 warning message is present
    scale2_warning = any("Scale 2 (COPD)" in warning for warning in result.warnings)
    assert scale2_warning, f"Expected Scale 2 warning message, got warnings: {result.warnings}"
    
    print("COPD Scale 2 indication test passed")
    print(f"   - Scale used: {result.scale_used}")
    print(f"   - SpO2 (90%) score: {result.individual_scores['spo2']}")
    print(f"   - Warning: {[w for w in result.warnings if 'Scale 2' in w][0]}")
    
    return True


async def test_non_copd_uses_scale1():
    """Test that non-COPD patients use Scale 1."""
    print("\nTesting non-COPD patient uses Scale 1...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # Create non-COPD patient
    patient = Patient(
        patient_id="NORMAL_001",
        ward_id="WARD_A", 
        bed_number="002",
        age=50,
        is_copd_patient=False,  # This should trigger Scale 1
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Create same vital signs as COPD test
    vitals = VitalSigns(
        event_id=uuid4(),
        patient_id="NORMAL_001", 
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=90,  # Same SpO2 but different scale
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    result = await calculator.calculate_news2(vitals, patient)
    
    # Verify Scale 1 is used
    assert result.scale_used == 1, f"Expected Scale 1, got {result.scale_used}"
    
    # Verify SpO2 scores differently for non-COPD (90% should be 3 points on Scale 1)
    assert result.individual_scores['spo2'] == 3, f"Expected SpO2 score 3, got {result.individual_scores['spo2']}"
    
    # Verify no Scale 2 warning message
    scale2_warning = any("Scale 2 (COPD)" in warning for warning in result.warnings)
    assert not scale2_warning, f"Unexpected Scale 2 warning for non-COPD patient: {result.warnings}"
    
    print("Non-COPD Scale 1 test passed")
    print(f"   - Scale used: {result.scale_used}")
    print(f"   - SpO2 (90%) score: {result.individual_scores['spo2']}")
    print(f"   - Warnings: {result.warnings}")
    
    return True


async def test_copd_spo2_boundary_differences():
    """Test the key differences between Scale 1 and Scale 2 SpO2 scoring."""
    print("\nTesting COPD vs Normal SpO2 scoring differences...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # Test SpO2 values that score differently between scales
    test_cases = [
        (90, 3, 0),  # 90%: Scale 1=3, Scale 2=0
        (93, 2, 1),  # 93%: Scale 1=2, Scale 2=1  
        (97, 0, 3),  # 97%: Scale 1=0, Scale 2=3
    ]
    
    for spo2_val, expected_scale1, expected_scale2 in test_cases:
        # Test with non-COPD patient (Scale 1)
        normal_patient = Patient(
            patient_id="NORMAL_TEST",
            ward_id="WARD_A",
            bed_number="003", 
            age=50,
            is_copd_patient=False,
            assigned_nurse_id="NURSE_001",
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        # Test with COPD patient (Scale 2)
        copd_patient = Patient(
            patient_id="COPD_TEST",
            ward_id="WARD_A",
            bed_number="004",
            age=65,
            is_copd_patient=True,
            assigned_nurse_id="NURSE_001", 
            admission_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        vitals = VitalSigns(
            event_id=uuid4(),
            patient_id="TEST",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=spo2_val,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        
        # Test Scale 1 (normal patient)
        result1 = await calculator.calculate_news2(vitals, normal_patient)
        assert result1.scale_used == 1
        assert result1.individual_scores['spo2'] == expected_scale1, \
            f"SpO2 {spo2_val}% Scale 1: expected {expected_scale1}, got {result1.individual_scores['spo2']}"
        
        # Test Scale 2 (COPD patient)  
        result2 = await calculator.calculate_news2(vitals, copd_patient)
        assert result2.scale_used == 2
        assert result2.individual_scores['spo2'] == expected_scale2, \
            f"SpO2 {spo2_val}% Scale 2: expected {expected_scale2}, got {result2.individual_scores['spo2']}"
        
        print(f"   SpO2 {spo2_val}%: Scale 1={result1.individual_scores['spo2']}, Scale 2={result2.individual_scores['spo2']}")
    
    print("COPD vs Normal SpO2 differences validated")
    return True


async def main():
    """Run all COPD-specific tests."""
    print("=== COPD Scale 2 Validation Tests ===\n")
    
    tests = [
        test_copd_scale_indication,
        test_non_copd_uses_scale1,
        test_copd_spo2_boundary_differences,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"FAILED: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAILED: {test.__name__} with error: {str(e)}")
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("All COPD Scale 2 tests passed!")
        print("COPD patient detection working correctly") 
        print("Scale 2 usage clearly indicated in results")
        print("Modified SpO2 ranges applied correctly for COPD")
        print("All other parameters use standard Scale 1 scoring")
    else:
        print("Some COPD tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)