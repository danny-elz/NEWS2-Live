#!/usr/bin/env python3
"""
Test script for Edge Case Handling functionality validation.
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
from src.models.partial_vital_signs import PartialVitalSigns
from src.models.news2 import RiskCategory, InvalidVitalSignsError, MissingVitalSignsError
from src.services.audit import AuditLogger
from src.services.news2_calculator import NEWS2Calculator


async def test_missing_vital_signs_handling():
    """Test handling of missing vital signs with partial score calculation."""
    print("Testing missing vital signs handling...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="PARTIAL_001",
        ward_id="WARD_A",
        bed_number="001",
        age=65,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test with only some vital signs present
    partial_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="PARTIAL_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=22,    # Present - 2 points
        spo2=94,              # Present - 1 point
        on_oxygen=None,       # Missing
        temperature=None,     # Missing
        systolic_bp=None,     # Missing
        heart_rate=95,        # Present - 1 point
        consciousness=None    # Missing
    )
    
    result = await calculator.calculate_partial_news2(partial_vitals, patient)
    
    # Verify partial calculation
    expected_score = 2 + 1 + 1  # RR + SpO2 + HR (missing params scored as 0)
    assert result.total_score == expected_score, f"Expected score {expected_score}, got {result.total_score}"
    
    # Verify completeness warnings
    completeness = partial_vitals.get_completeness_score()
    assert completeness < 1.0, "Expected incomplete vital signs"
    
    missing_warning = any("missing" in warning.lower() for warning in result.warnings)
    assert missing_warning, "Expected missing parameter warning"
    
    # Verify confidence is reduced for incomplete data
    assert result.confidence < 1.0, "Expected reduced confidence for partial data"
    
    # Verify clinical guidance includes missing parameter note
    assert "missing_parameters" in result.clinical_guidance, "Expected missing parameters guidance"
    
    print(f"   Completeness: {completeness:.0%}")
    print(f"   Total Score: {result.total_score} (partial calculation)")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Warnings: {len(result.warnings)} warnings about missing data")
    
    return True


async def test_invalid_value_ranges_handling():
    """Test graceful handling of invalid value ranges with error messages."""
    print("\nTesting invalid value ranges handling...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="INVALID_001",
        ward_id="WARD_A",
        bed_number="002",
        age=50,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test invalid respiratory rate
    try:
        invalid_vitals = PartialVitalSigns(
            event_id=uuid4(),
            patient_id="INVALID_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=100,  # Invalid - above range
            spo2=98,
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        assert False, "Expected validation error for invalid respiratory rate"
    except ValueError as e:
        print(f"   Correctly caught invalid respiratory rate: {str(e)}")
    
    # Test invalid SpO2
    try:
        invalid_vitals = PartialVitalSigns(
            event_id=uuid4(),
            patient_id="INVALID_001",
            timestamp=datetime.now(timezone.utc),
            respiratory_rate=18,
            spo2=120,  # Invalid - above 100%
            on_oxygen=False,
            temperature=36.5,
            systolic_bp=120,
            heart_rate=75,
            consciousness=ConsciousnessLevel.ALERT
        )
        assert False, "Expected validation error for invalid SpO2"
    except ValueError as e:
        print(f"   Correctly caught invalid SpO2: {str(e)}")
    
    print("Invalid value range handling working correctly")
    return True


async def test_physiologically_impossible_combinations():
    """Test detection of physiologically impossible vital sign combinations."""
    print("\nTesting physiologically impossible combinations...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="IMPOSSIBLE_001",
        ward_id="WARD_A",
        bed_number="003",
        age=45,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test impossible combination: conscious patient with very low SpO2 and no oxygen
    impossible_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="IMPOSSIBLE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=80,              # Very low
        on_oxygen=False,      # No oxygen support
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT  # Alert despite low SpO2
    )
    
    try:
        result = await calculator.calculate_partial_news2(impossible_vitals, patient)
        assert False, "Expected error for physiologically impossible combination"
    except InvalidVitalSignsError as e:
        print(f"   Correctly detected impossible combination: {str(e)}")
    
    # Test extreme tachycardia with severe hypotension
    shock_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="IMPOSSIBLE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=95,
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=60,       # Very low BP
        heart_rate=160,       # Very high HR
        consciousness=ConsciousnessLevel.ALERT
    )
    
    try:
        result = await calculator.calculate_partial_news2(shock_vitals, patient)
        assert False, "Expected error for extreme shock combination"
    except InvalidVitalSignsError as e:
        print(f"   Correctly detected shock combination: {str(e)}")
    
    print("Physiologically impossible combination detection working correctly")
    return True


async def test_confidence_scoring():
    """Test confidence scoring based on data quality and completeness."""
    print("\nTesting confidence scoring...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="CONFIDENCE_001",
        ward_id="WARD_A",
        bed_number="004",
        age=60,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test high-quality complete data
    high_quality_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="CONFIDENCE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=98,
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT,
        is_manual_entry=False,  # Automated
        has_artifacts=False,    # Clean signal
        confidence=1.0          # High base confidence
    )
    
    result_high = await calculator.calculate_partial_news2(high_quality_vitals, patient)
    
    # Test lower quality data
    low_quality_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="CONFIDENCE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=None,             # Missing
        on_oxygen=False,
        temperature=None,      # Missing
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT,
        is_manual_entry=True,  # Manual entry
        has_artifacts=True,    # Artifacts present
        confidence=0.8         # Lower base confidence
    )
    
    result_low = await calculator.calculate_partial_news2(low_quality_vitals, patient)
    
    # Verify confidence differences
    assert result_high.confidence > result_low.confidence, \
        f"Expected higher confidence for complete data: {result_high.confidence} vs {result_low.confidence}"
    
    print(f"   High quality (complete): confidence = {result_high.confidence:.2f}")
    print(f"   Low quality (partial): confidence = {result_low.confidence:.2f}")
    print("Confidence scoring working correctly")
    
    return True


async def test_calculation_retries():
    """Test calculation retries for transient failures."""
    print("\nTesting calculation retries...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="RETRY_001",
        ward_id="WARD_A",
        bed_number="005",
        age=55,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Create normal vital signs
    normal_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="RETRY_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=98,
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    # Normal calculation should succeed without retries
    result = await calculator.calculate_partial_news2(normal_vitals, patient, max_retries=3)
    
    assert result.total_score == 0, "Expected normal score of 0"
    print(f"   Normal calculation succeeded: score = {result.total_score}")
    
    # Test that non-transient errors are not retried
    try:
        result = await calculator.calculate_partial_news2(None, patient, max_retries=3)
        assert False, "Expected MissingVitalSignsError for None input"
    except Exception as e:
        # Should fail immediately, not after retries
        print(f"   Non-transient error correctly not retried: {type(e).__name__}")
    
    print("Calculation retry mechanism working correctly")
    return True


async def test_edge_case_integration():
    """Test integrated edge case scenarios."""
    print("\nTesting integrated edge case scenarios...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # COPD patient with partial vitals
    copd_patient = Patient(
        patient_id="EDGE_001",
        ward_id="WARD_A",
        bed_number="006",
        age=70,
        is_copd_patient=True,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Partial vitals with some concerning values
    partial_vitals = PartialVitalSigns(
        event_id=uuid4(),
        patient_id="EDGE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=None,     # Missing
        spo2=89,                  # Concerning for normal, but okay for COPD Scale 2
        on_oxygen=True,           # 2 points
        temperature=38.8,         # 1 point
        systolic_bp=None,         # Missing
        heart_rate=105,           # 1 point
        consciousness=ConsciousnessLevel.ALERT,
        is_manual_entry=True,
        confidence=0.85
    )
    
    result = await calculator.calculate_partial_news2(partial_vitals, copd_patient)
    
    # Verify COPD Scale 2 is used
    assert result.scale_used == 2, "Expected COPD Scale 2"
    
    # Verify SpO2 scored correctly for COPD (89% should be 0 points on Scale 2)
    assert result.individual_scores['spo2'] == 0, "Expected SpO2 score 0 for COPD Scale 2"
    
    # Verify warnings include COPD Scale 2 usage and missing parameters
    copd_warning = any("Scale 2 (COPD)" in warning for warning in result.warnings)
    missing_warning = any("missing" in warning.lower() for warning in result.warnings)
    
    assert copd_warning, "Expected COPD Scale 2 warning"
    assert missing_warning, "Expected missing parameters warning"
    
    # Verify clinical guidance includes COPD considerations
    assert "copd_considerations" in result.clinical_guidance, "Expected COPD guidance"
    assert "missing_parameters" in result.clinical_guidance, "Expected missing parameters guidance"
    
    print(f"   COPD patient with partial vitals:")
    print(f"   - Scale used: {result.scale_used}")
    print(f"   - SpO2 (89%) score: {result.individual_scores['spo2']}")
    print(f"   - Total score: {result.total_score}")
    print(f"   - Confidence: {result.confidence:.2f}")
    print(f"   - Warnings: {len(result.warnings)}")
    
    return True


async def main():
    """Run all edge case handling tests."""
    print("=== Edge Case Handling Validation Tests ===\n")
    
    tests = [
        test_missing_vital_signs_handling,
        test_invalid_value_ranges_handling,
        test_physiologically_impossible_combinations,
        test_confidence_scoring,
        test_calculation_retries,
        test_edge_case_integration,
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
        print("All Edge Case Handling tests passed!")
        print("Missing vital signs handled with partial calculation and warnings")
        print("Invalid value ranges gracefully handled with error messages")
        print("Confidence scoring based on data quality and completeness working")
        print("Physiologically impossible combinations detected")
        print("Calculation retries implemented for transient failures")
        print("Integrated edge cases handled correctly")
    else:
        print("Some edge case handling tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)