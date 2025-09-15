"""
Epic 3: Clinical Workflow Integration Testing
Comprehensive end-to-end testing of clinical workflows with intelligent alert management.

This framework tests complete clinical scenarios from patient deterioration
through alert generation, ML suppression decisions, multi-channel delivery,
and clinical response workflows.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.models.patient import Patient
from src.models.news2 import NEWS2Result, RiskCategory
from src.models.alerts import Alert, AlertLevel, AlertStatus
from src.models.clinical_users import ClinicalUser, UserRole
from src.services.integrated_alert_system import IntegratedAlertSystem
from src.services.clinical_workflow_simulator import ClinicalWorkflowSimulator


class TestCriticalPatientDeterioration:
    """Test complete critical patient deterioration scenarios."""

    @pytest.fixture
    def integrated_system(self):
        """Integrated alert management system."""
        return IntegratedAlertSystem()

    @pytest.fixture
    def workflow_simulator(self):
        """Clinical workflow simulator."""
        return ClinicalWorkflowSimulator()

    @pytest.fixture
    def night_shift_scenario(self):
        """Night shift clinical scenario setup."""
        return {
            "time": "02:30",
            "ward": "Medical Ward A",
            "staff": {
                "primary_nurse": ClinicalUser(
                    user_id="NURSE_NIGHT_001",
                    name="Sarah Chen",
                    role=UserRole.WARD_NURSE,
                    experience_years=8,
                    shift="night",
                    current_patient_load=6
                ),
                "charge_nurse": ClinicalUser(
                    user_id="CHARGE_NIGHT_001",
                    name="Michael Torres",
                    role=UserRole.CHARGE_NURSE,
                    shift="night"
                ),
                "on_call_doctor": ClinicalUser(
                    user_id="DOCTOR_ONCALL_001",
                    name="Dr. Amanda Williams",
                    role=UserRole.DOCTOR,
                    specialty="internal_medicine"
                )
            }
        }

    @pytest.mark.end_to_end
    @pytest.mark.critical_safety
    async def test_elderly_patient_rapid_deterioration(self, integrated_system, workflow_simulator, night_shift_scenario):
        """
        Test complete workflow: Elderly patient rapid deterioration during night shift.

        SCENARIO: 78-year-old patient with pneumonia shows rapid deterioration
        from stable (NEWS2=2) to critical (NEWS2=9) over 30 minutes.
        """

        # Create patient with clinical history
        patient = Patient(
            patient_id="DETERIORATION_001",
            ward_id="MEDICAL_A",
            bed_number="A-15",
            age=78,
            is_copd_patient=False,
            assigned_nurse_id=night_shift_scenario["staff"]["primary_nurse"].user_id,
            admission_date=datetime.now(timezone.utc) - timedelta(days=3),
            last_updated=datetime.now(timezone.utc),
            medical_history=["pneumonia", "diabetes", "hypertension"],
            current_medications=["antibiotics", "insulin", "bp_medication"]
        )

        # PHASE 1: Initial stable baseline (02:00 AM)
        await workflow_simulator.set_time("02:00")

        baseline_vitals = NEWS2Result(
            total_score=2,
            individual_scores={
                "respiratory_rate": 1,  # Slightly elevated
                "spo2": 0,              # Normal
                "temperature": 1,       # Mild fever
                "systolic_bp": 0,       # Normal
                "heart_rate": 0,        # Normal
                "consciousness": 0      # Alert
            },
            risk_category=RiskCategory.LOW,
            monitoring_frequency="4 hourly",
            scale_used=1,
            warnings=["Mild fever - monitor"],
            confidence=0.95,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.1
        )

        # Process baseline - should not generate alert
        baseline_result = await integrated_system.process_patient_vitals(
            patient, baseline_vitals, night_shift_scenario["staff"]["primary_nurse"]
        )

        assert baseline_result.alert_generated is False
        assert baseline_result.ml_suppression_applied is True
        assert "stable_baseline" in baseline_result.suppression_reasoning

        # PHASE 2: Early deterioration signs (02:15 AM)
        await workflow_simulator.set_time("02:15")

        early_deterioration = NEWS2Result(
            total_score=4,
            individual_scores={
                "respiratory_rate": 2,  # Increasing
                "spo2": 1,              # Starting to drop
                "temperature": 1,       # Persistent fever
                "systolic_bp": 0,       # Still normal
                "heart_rate": 0,        # Still normal
                "consciousness": 0      # Still alert
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="2 hourly",
            scale_used=1,
            warnings=["Deteriorating respiratory status"],
            confidence=0.92,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.3
        )

        early_result = await integrated_system.process_patient_vitals(
            patient, early_deterioration, night_shift_scenario["staff"]["primary_nurse"]
        )

        # ML should suppress due to gradual change pattern
        assert early_result.alert_generated is False
        assert early_result.ml_suppression_applied is True
        assert early_result.suppression_confidence > 0.8

        # PHASE 3: CRITICAL DETERIORATION (02:30 AM)
        await workflow_simulator.set_time("02:30")

        critical_deterioration = NEWS2Result(
            total_score=9,
            individual_scores={
                "respiratory_rate": 3,  # CRITICAL: ≤8 or ≥25
                "spo2": 2,              # Significant drop
                "temperature": 1,       # Fever continues
                "systolic_bp": 2,       # Hypotensive
                "heart_rate": 1,        # Tachycardia
                "consciousness": 0      # Still alert (concerning given other params)
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["CRITICAL: Multiple parameters severely abnormal", "Possible septic shock"],
            confidence=0.97,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.8
        )

        # CRITICAL: This MUST generate immediate alert
        critical_result = await integrated_system.process_patient_vitals(
            patient, critical_deterioration, night_shift_scenario["staff"]["primary_nurse"]
        )

        # SAFETY ASSERTION: Critical deterioration must never be suppressed
        assert critical_result.alert_generated is True
        assert critical_result.ml_suppression_applied is False
        assert critical_result.alert_level == AlertLevel.CRITICAL
        assert "NEVER_SUPPRESS_CRITICAL" in critical_result.safety_override_reason

        # PHASE 4: Multi-channel delivery validation
        delivery_result = critical_result.delivery_result

        # All channels must be attempted for critical alerts
        assert delivery_result.broadcast_mode is True
        assert delivery_result.total_channels_attempted >= 4
        assert delivery_result.successful_channels >= 1
        assert delivery_result.delivery_latency_ms < 15000  # <15 seconds

        # Primary nurse must receive immediate notification
        primary_nurse_notifications = delivery_result.recipient_notifications[
            night_shift_scenario["staff"]["primary_nurse"].user_id
        ]

        assert len(primary_nurse_notifications) >= 3  # WebSocket, Push, SMS minimum
        assert any(notif.channel == "websocket" for notif in primary_nurse_notifications)

        # PHASE 5: Escalation timeline validation
        escalation_result = critical_result.escalation_result

        assert escalation_result.escalation_scheduled is True
        assert escalation_result.escalation_steps[0].role == UserRole.WARD_NURSE
        assert escalation_result.escalation_steps[1].role == UserRole.CHARGE_NURSE
        assert escalation_result.escalation_steps[1].delay_minutes == 15

        # PHASE 6: Clinical response simulation
        await workflow_simulator.advance_time_minutes(5)

        # Simulate nurse acknowledgment
        nurse_response = await integrated_system.acknowledge_alert(
            critical_result.alert.alert_id,
            night_shift_scenario["staff"]["primary_nurse"].user_id,
            "Patient assessed - initiating rapid response protocol"
        )

        assert nurse_response.acknowledged is True
        assert nurse_response.response_time_seconds < 300  # <5 minutes
        assert "rapid response" in nurse_response.clinical_actions.lower()

        # PHASE 7: Verify clinical outcome tracking
        outcome_tracking = await integrated_system.get_clinical_outcome_tracking(
            critical_result.alert.alert_id
        )

        assert outcome_tracking.patient_outcome_monitored is True
        assert outcome_tracking.clinical_interventions_recorded is True
        assert outcome_tracking.alert_effectiveness_tracked is True

    @pytest.mark.end_to_end
    async def test_copd_patient_specialized_handling(self, integrated_system, workflow_simulator):
        """Test specialized ML handling for COPD patient exacerbation."""

        # COPD patient with oxygen dependency
        copd_patient = Patient(
            patient_id="COPD_001",
            ward_id="RESPIRATORY_WARD",
            bed_number="R-08",
            age=68,
            is_copd_patient=True,
            oxygen_dependent=True,
            baseline_spo2_range=(88, 92),  # COPD baseline
            assigned_nurse_id="RESPIRATORY_NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=12),
            last_updated=datetime.now(timezone.utc),
            medical_history=["COPD", "smoking_history", "previous_exacerbations"]
        )

        respiratory_nurse = ClinicalUser(
            user_id="RESPIRATORY_NURSE_001",
            name="Lisa Rodriguez",
            role=UserRole.RESPIRATORY_NURSE,
            specialization="pulmonary_care"
        )

        # SCENARIO 1: COPD baseline vitals (should suppress)
        copd_baseline = NEWS2Result(
            total_score=3,
            individual_scores={
                "respiratory_rate": 1,
                "spo2": 1,              # 88% - normal for COPD Scale 2
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 2,        # Slightly elevated
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="2 hourly",
            scale_used=2,  # COPD Scale 2
            warnings=["COPD patient - baseline O2 levels"],
            confidence=0.93,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.2
        )

        baseline_result = await integrated_system.process_patient_vitals(
            copd_patient, copd_baseline, respiratory_nurse
        )

        # ML should understand COPD context and suppress appropriately
        assert baseline_result.alert_generated is False
        assert baseline_result.ml_suppression_applied is True
        assert "COPD_baseline_appropriate" in baseline_result.suppression_reasoning
        assert baseline_result.suppression_confidence > 0.85

        # SCENARIO 2: COPD exacerbation (should NOT suppress)
        copd_exacerbation = NEWS2Result(
            total_score=6,
            individual_scores={
                "respiratory_rate": 3,  # Severely elevated - concerning even for COPD
                "spo2": 2,              # 84% - below COPD baseline
                "temperature": 0,
                "systolic_bp": 0,
                "heart_rate": 1,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=2,
            warnings=["COPD exacerbation detected", "Below baseline SpO2"],
            confidence=0.91,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.0
        )

        exacerbation_result = await integrated_system.process_patient_vitals(
            copd_patient, copd_exacerbation, respiratory_nurse
        )

        # Should generate alert for COPD exacerbation
        assert exacerbation_result.alert_generated is True
        assert exacerbation_result.ml_suppression_applied is False
        assert "COPD_exacerbation_detected" in exacerbation_result.clinical_reasoning
        assert exacerbation_result.alert_level == AlertLevel.HIGH

        # Should route to respiratory specialist
        assert exacerbation_result.specialized_routing is True
        assert exacerbation_result.primary_recipient.role == UserRole.RESPIRATORY_NURSE

    @pytest.mark.end_to_end
    async def test_shift_handoff_alert_continuity(self, integrated_system, workflow_simulator):
        """Test alert management during shift handoff periods."""

        patient = Patient(
            patient_id="HANDOFF_001",
            ward_id="SURGICAL_WARD",
            bed_number="S-12",
            age=55,
            is_copd_patient=False,
            assigned_nurse_id="DAY_NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=18),
            last_updated=datetime.now(timezone.utc)
        )

        day_nurse = ClinicalUser(
            user_id="DAY_NURSE_001",
            name="Jennifer Adams",
            role=UserRole.WARD_NURSE,
            shift="day"
        )

        night_nurse = ClinicalUser(
            user_id="NIGHT_NURSE_001",
            name="Carlos Martinez",
            role=UserRole.WARD_NURSE,
            shift="night"
        )

        # PHASE 1: Generate alert near shift change (6:45 AM)
        await workflow_simulator.set_time("06:45")

        pre_handoff_vitals = NEWS2Result(
            total_score=5,
            individual_scores={
                "respiratory_rate": 2,
                "spo2": 1,
                "temperature": 1,
                "systolic_bp": 1,
                "heart_rate": 0,
                "consciousness": 0
            },
            risk_category=RiskCategory.MEDIUM,
            monitoring_frequency="2 hourly",
            scale_used=1,
            warnings=["Moderate concern - monitor closely"],
            confidence=0.89,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=2.4
        )

        pre_handoff_result = await integrated_system.process_patient_vitals(
            patient, pre_handoff_vitals, day_nurse
        )

        assert pre_handoff_result.alert_generated is True
        assert pre_handoff_result.alert.assigned_to == day_nurse.user_id

        # PHASE 2: Shift handoff occurs (7:00 AM)
        await workflow_simulator.set_time("07:00")

        handoff_result = await integrated_system.process_shift_handoff(
            from_nurse=day_nurse,
            to_nurse=night_nurse,
            ward_id="SURGICAL_WARD"
        )

        # Verify alert reassignment
        assert handoff_result.alerts_transferred > 0

        updated_alert = await integrated_system.get_alert(pre_handoff_result.alert.alert_id)
        assert updated_alert.assigned_to == night_nurse.user_id

        # Night nurse should receive handoff notification
        handoff_notifications = await integrated_system.get_notifications(night_nurse.user_id)

        handoff_notification = next(
            (n for n in handoff_notifications if "handoff" in n.message.lower()),
            None
        )

        assert handoff_notification is not None
        assert patient.patient_id in handoff_notification.message
        assert "requires attention" in handoff_notification.message.lower()

        # PHASE 3: Verify continuity of care
        handoff_summary = handoff_result.handoff_summary

        assert handoff_summary.total_patients_transferred > 0
        assert handoff_summary.active_alerts_transferred > 0
        assert handoff_summary.clinical_notes_updated is True

    @pytest.mark.end_to_end
    async def test_multi_ward_rapid_response_coordination(self, integrated_system, workflow_simulator):
        """Test coordination across multiple wards during rapid response."""

        # ICU patient requiring multi-disciplinary response
        icu_patient = Patient(
            patient_id="RAPID_RESPONSE_001",
            ward_id="ICU",
            bed_number="ICU-03",
            age=45,
            is_copd_patient=False,
            post_operative=True,
            surgery_date=datetime.now(timezone.utc) - timedelta(hours=8),
            assigned_nurse_id="ICU_NURSE_001",
            admission_date=datetime.now(timezone.utc) - timedelta(hours=10),
            last_updated=datetime.now(timezone.utc)
        )

        # Generate critical alert requiring rapid response
        critical_vitals = NEWS2Result(
            total_score=12,
            individual_scores={
                "respiratory_rate": 3,  # Critical
                "spo2": 3,              # Critical
                "temperature": 1,
                "systolic_bp": 3,       # Critical
                "heart_rate": 2,
                "consciousness": 0
            },
            risk_category=RiskCategory.HIGH,
            monitoring_frequency="continuous",
            scale_used=1,
            warnings=["MULTIPLE CRITICAL PARAMETERS", "POSSIBLE CARDIAC ARREST"],
            confidence=0.98,
            calculated_at=datetime.now(timezone.utc),
            calculation_time_ms=1.5
        )

        rapid_response_result = await integrated_system.process_patient_vitals(
            icu_patient, critical_vitals, None
        )

        # Should trigger rapid response team activation
        assert rapid_response_result.rapid_response_triggered is True
        assert rapid_response_result.alert_level == AlertLevel.CRITICAL

        # Verify multi-disciplinary team notification
        team_notifications = rapid_response_result.team_notifications

        expected_roles = [
            UserRole.ICU_NURSE,
            UserRole.RAPID_RESPONSE_DOCTOR,
            UserRole.RESPIRATORY_THERAPIST,
            UserRole.PHARMACIST,
            UserRole.CHARGE_NURSE
        ]

        for role in expected_roles:
            assert role in team_notifications
            assert team_notifications[role].priority == "IMMEDIATE"

        # Verify appropriate communication channels by role
        doctor_notification = team_notifications[UserRole.RAPID_RESPONSE_DOCTOR]
        assert "pager" in doctor_notification.channels
        assert "voice_call" in doctor_notification.channels

        nurse_notification = team_notifications[UserRole.ICU_NURSE]
        assert "websocket" in nurse_notification.channels
        assert "overhead_paging" in nurse_notification.channels

        # Verify rapid response timeline
        response_timeline = rapid_response_result.response_timeline

        assert response_timeline.team_notification_target_seconds == 30
        assert response_timeline.first_responder_target_minutes == 2
        assert response_timeline.full_team_assembly_target_minutes == 5

    @pytest.mark.end_to_end
    async def test_alert_fatigue_reduction_effectiveness(self, integrated_system, workflow_simulator):
        """Test that ML suppression achieves 60% alert volume reduction without compromising safety."""

        # Simulate 24-hour ward operation
        ward_patients = [
            self._create_test_patient(f"FATIGUE_TEST_{i}")
            for i in range(20)  # 20-bed ward
        ]

        # Generate realistic alert patterns over 24 hours
        total_alerts_generated = 0
        alerts_suppressed = 0
        critical_alerts_processed = 0

        for hour in range(24):
            await workflow_simulator.set_time(f"{hour:02d}:00")

            # Generate hourly vitals for each patient
            for patient in ward_patients:
                vitals = self._generate_realistic_vitals(patient, hour)

                result = await integrated_system.process_patient_vitals(
                    patient, vitals, self._get_assigned_nurse(patient, hour)
                )

                if result.would_generate_alert_without_ml:
                    total_alerts_generated += 1

                    if result.alert_generated:
                        if result.alert_level == AlertLevel.CRITICAL:
                            critical_alerts_processed += 1
                    else:
                        alerts_suppressed += 1

        # Verify 60% volume reduction target
        suppression_rate = alerts_suppressed / total_alerts_generated if total_alerts_generated > 0 else 0

        assert suppression_rate >= 0.60, f"Suppression rate {suppression_rate:.3f} below 60% target"

        # Verify NO critical alerts were suppressed
        critical_suppression_check = await integrated_system.get_suppression_audit(
            alert_level=AlertLevel.CRITICAL,
            time_range=24
        )

        assert critical_suppression_check.total_critical_suppressions == 0, \
            "SAFETY VIOLATION: Critical alerts were suppressed"

        # Verify clinical outcomes weren't compromised
        outcome_analysis = await integrated_system.analyze_clinical_outcomes(
            time_range=24,
            include_suppressed_alerts=True
        )

        assert outcome_analysis.patient_safety_score >= 0.95
        assert outcome_analysis.missed_deterioration_rate < 0.05  # <5%


# Helper methods for clinical scenario creation
def _create_test_patient(patient_id: str) -> Patient:
    """Create test patient with realistic clinical profile."""
    # Implementation details...
    pass

def _generate_realistic_vitals(patient: Patient, hour: int) -> NEWS2Result:
    """Generate realistic vital signs based on time and patient profile."""
    # Implementation details...
    pass

def _get_assigned_nurse(patient: Patient, hour: int) -> ClinicalUser:
    """Get assigned nurse based on shift schedule."""
    # Implementation details...
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])