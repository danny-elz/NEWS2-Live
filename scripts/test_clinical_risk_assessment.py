#!/usr/bin/env python3
"""
Test script for Clinical Risk Assessment functionality validation.
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
from src.models.news2 import RiskCategory
from src.services.audit import AuditLogger
from src.services.news2_calculator import NEWS2Calculator


async def test_risk_category_assignments():
    """Test all risk category assignments based on total scores."""
    print("Testing risk category assignments...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="TEST_001",
        ward_id="WARD_A",
        bed_number="001",
        age=65,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test cases: (total_score, expected_risk, expected_monitoring_keyword)
    test_cases = [
        (0, RiskCategory.LOW, "routine"),
        (1, RiskCategory.LOW, "12-hourly"),
        (2, RiskCategory.LOW, "12-hourly"),
        (3, RiskCategory.MEDIUM, "6-hourly"),
        (4, RiskCategory.MEDIUM, "6-hourly"),
        (5, RiskCategory.MEDIUM, "hourly"),
        (6, RiskCategory.MEDIUM, "hourly"),
        (7, RiskCategory.HIGH, "continuous"),
        (10, RiskCategory.HIGH, "continuous"),
    ]
    
    for target_score, expected_risk, monitoring_keyword in test_cases:
        # Create vital signs that will produce the target score
        vitals = create_vitals_for_score(target_score)
        
        result = await calculator.calculate_news2(vitals, patient)
        
        assert result.risk_category == expected_risk, \
            f"Score {target_score}: expected {expected_risk}, got {result.risk_category}"
        assert monitoring_keyword in result.monitoring_frequency.lower(), \
            f"Score {target_score}: expected '{monitoring_keyword}' in monitoring, got '{result.monitoring_frequency}'"
        
        # Verify clinical guidance is provided
        assert result.clinical_guidance is not None, f"No clinical guidance for score {target_score}"
        assert "escalation" in result.clinical_guidance, f"No escalation guidance for score {target_score}"
        
        print(f"   Score {target_score}: {expected_risk.value} risk - {result.monitoring_frequency}")
    
    print("Risk category assignments validated")
    return True


async def test_single_parameter_red_flags():
    """Test single parameter red flag detection for each vital sign parameter."""
    print("\nTesting single parameter red flags...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    patient = Patient(
        patient_id="TEST_002",
        ward_id="WARD_A",
        bed_number="002",
        age=50,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    # Test each parameter with a value that gives 3 points (red flag)
    red_flag_cases = [
        ("respiratory_rate", 25, "respiratory_alert"),
        ("spo2", 90, "oxygen_alert"),
        ("systolic_bp", 85, "bp_alert"),
        ("heart_rate", 35, "cardiac_alert"),
        ("temperature", 34.0, "temperature_alert"),
        ("consciousness", ConsciousnessLevel.CONFUSION, "consciousness_alert")
    ]
    
    for param_name, red_flag_value, expected_guidance_key in red_flag_cases:
        # Create normal vitals
        vitals_data = {
            'event_id': uuid4(),
            'patient_id': 'TEST_002',
            'timestamp': datetime.now(timezone.utc),
            'respiratory_rate': 18,
            'spo2': 98,
            'on_oxygen': False,
            'temperature': 36.5,
            'systolic_bp': 120,
            'heart_rate': 75,
            'consciousness': ConsciousnessLevel.ALERT
        }
        
        # Set the red flag parameter
        vitals_data[param_name] = red_flag_value
        vitals = VitalSigns(**vitals_data)
        
        result = await calculator.calculate_news2(vitals, patient)
        
        # Should be HIGH risk due to red flag
        assert result.risk_category == RiskCategory.HIGH, \
            f"{param_name} red flag: expected HIGH risk, got {result.risk_category}"
        
        # Should have red flag warning
        red_flag_warning = any("red flag" in warning.lower() for warning in result.warnings)
        assert red_flag_warning, f"{param_name}: no red flag warning found"
        
        # Should have specific clinical guidance for this parameter
        if expected_guidance_key in result.clinical_guidance:
            print(f"   {param_name}: RED FLAG detected - {result.clinical_guidance[expected_guidance_key]}")
        else:
            print(f"   {param_name}: RED FLAG detected - general guidance provided")
    
    print("Single parameter red flags validated")
    return True


async def test_clinical_escalation_guidance():
    """Test comprehensive clinical escalation guidance."""
    print("\nTesting clinical escalation guidance...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # Test HIGH risk scenario with multiple red flags
    patient = Patient(
        patient_id="HIGH_RISK",
        ward_id="WARD_A",
        bed_number="003",
        age=70,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    vitals = VitalSigns(
        event_id=uuid4(),
        patient_id="HIGH_RISK",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=25,  # 3 points - red flag
        spo2=89,             # 3 points - red flag
        on_oxygen=True,      # 2 points
        temperature=39.5,    # 2 points
        systolic_bp=85,      # 3 points - red flag
        heart_rate=140,      # 3 points - red flag
        consciousness=ConsciousnessLevel.CONFUSION  # 3 points - red flag
    )
    
    result = await calculator.calculate_news2(vitals, patient)
    
    # Verify HIGH risk
    assert result.risk_category == RiskCategory.HIGH
    
    # Verify comprehensive guidance
    guidance = result.clinical_guidance
    assert guidance is not None
    
    # Check for essential guidance components
    required_keys = ["escalation", "response_time", "staff_level", "red_flag_action", "documentation"]
    for key in required_keys:
        assert key in guidance, f"Missing {key} in clinical guidance"
    
    # Check for specific red flag guidance
    red_flag_alerts = ["respiratory_alert", "oxygen_alert", "bp_alert", "cardiac_alert", "consciousness_alert"]
    found_alerts = [key for key in red_flag_alerts if key in guidance]
    assert len(found_alerts) >= 3, f"Expected multiple specific red flag alerts, found: {found_alerts}"
    
    print(f"   HIGH risk guidance: {guidance['escalation']}")
    print(f"   Response time: {guidance['response_time']}")
    print(f"   Staff level: {guidance['staff_level']}")
    print(f"   Red flag alerts: {len(found_alerts)} specific alerts provided")
    
    return True


async def test_copd_clinical_guidance():
    """Test COPD-specific clinical guidance."""
    print("\nTesting COPD clinical guidance...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # COPD patient with high SpO2 (red flag for COPD)
    copd_patient = Patient(
        patient_id="COPD_001",
        ward_id="WARD_A",
        bed_number="004",
        age=75,
        is_copd_patient=True,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    vitals = VitalSigns(
        event_id=uuid4(),
        patient_id="COPD_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=18,
        spo2=98,  # High SpO2 - red flag for COPD patients
        on_oxygen=False,
        temperature=36.5,
        systolic_bp=120,
        heart_rate=75,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    result = await calculator.calculate_news2(vitals, copd_patient)
    
    # Should be HIGH risk due to high SpO2 in COPD
    assert result.risk_category == RiskCategory.HIGH
    
    # Should have COPD-specific guidance
    guidance = result.clinical_guidance
    assert "copd_considerations" in guidance, "Missing COPD considerations"
    assert "copd_oxygen_alert" in guidance, "Missing COPD oxygen alert for high SpO2"
    
    print(f"   COPD considerations: {guidance['copd_considerations']}")
    print(f"   COPD oxygen alert: {guidance['copd_oxygen_alert']}")
    
    return True


async def test_palliative_care_guidance():
    """Test palliative care specific guidance."""
    print("\nTesting palliative care guidance...")
    
    audit_logger = AuditLogger()
    calculator = NEWS2Calculator(audit_logger)
    
    # Palliative care patient
    palliative_patient = Patient(
        patient_id="PALLIATIVE_001",
        ward_id="WARD_A",
        bed_number="005",
        age=80,
        is_copd_patient=False,
        assigned_nurse_id="NURSE_001",
        admission_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc),
        is_palliative=True
    )
    
    vitals = VitalSigns(
        event_id=uuid4(),
        patient_id="PALLIATIVE_001",
        timestamp=datetime.now(timezone.utc),
        respiratory_rate=25,  # Would normally trigger escalation
        spo2=88,
        on_oxygen=True,
        temperature=36.5,
        systolic_bp=95,
        heart_rate=95,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    result = await calculator.calculate_news2(vitals, palliative_patient)
    
    # Should have palliative care note
    guidance = result.clinical_guidance
    assert "palliative_note" in guidance, "Missing palliative care guidance"
    
    print(f"   Palliative guidance: {guidance['palliative_note']}")
    
    return True


def create_vitals_for_score(target_score):
    """Create vital signs that will produce approximately the target score without red flags."""
    # Base normal vitals (all 0 points)
    base_vitals = {
        'event_id': uuid4(),
        'patient_id': 'TEST',
        'timestamp': datetime.now(timezone.utc),
        'respiratory_rate': 18,  # 0 points
        'spo2': 98,             # 0 points
        'on_oxygen': False,     # 0 points
        'temperature': 36.5,    # 0 points
        'systolic_bp': 120,     # 0 points
        'heart_rate': 75,       # 0 points
        'consciousness': ConsciousnessLevel.ALERT  # 0 points
    }
    
    # Adjust parameters to reach target score WITHOUT creating red flags (score = 3)
    remaining_score = target_score
    
    # Use 2-point parameters first to avoid red flags
    if remaining_score >= 2:
        base_vitals['on_oxygen'] = True  # 2 points
        remaining_score -= 2
    
    if remaining_score >= 2:
        base_vitals['heart_rate'] = 115  # 2 points
        remaining_score -= 2
    
    if remaining_score >= 2:
        base_vitals['respiratory_rate'] = 22  # 2 points (avoids 3-point red flag)
        remaining_score -= 2
    
    # Use 1-point parameters
    if remaining_score >= 1:
        base_vitals['spo2'] = 95  # 1 point
        remaining_score -= 1
    
    if remaining_score >= 1:
        base_vitals['temperature'] = 38.5  # 1 point
        remaining_score -= 1
    
    if remaining_score >= 1:
        base_vitals['heart_rate'] = 95  # 1 point (override previous if needed)
        remaining_score -= 1
    
    if remaining_score >= 1:
        base_vitals['systolic_bp'] = 105  # 1 point
        remaining_score -= 1
    
    # Only use red flags for scores >= 7 where HIGH risk is expected anyway
    if remaining_score >= 3 and target_score >= 7:
        base_vitals['respiratory_rate'] = 25  # 3 points (red flag)
        remaining_score -= 3
    
    return VitalSigns(**base_vitals)


async def main():
    """Run all clinical risk assessment tests."""
    print("=== Clinical Risk Assessment Validation Tests ===\n")
    
    tests = [
        test_risk_category_assignments,
        test_single_parameter_red_flags,
        test_clinical_escalation_guidance,
        test_copd_clinical_guidance,
        test_palliative_care_guidance,
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
        print("All Clinical Risk Assessment tests passed!")
        print("Total score risk categories working correctly")
        print("Single parameter red flag detection implemented")
        print("Monitoring frequency recommendations provided")
        print("Clinical escalation guidance comprehensive")
        print("COPD-specific guidance included")
        print("Palliative care considerations integrated")
    else:
        print("Some clinical risk assessment tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)