"""
Unit tests for Patient Detail Service (Story 3.3)
Tests patient detail views and clinical timeline functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid

from src.dashboard.patient_detail.patient_detail_service import (
    PatientDetailService,
    PatientDetailData,
    ClinicalIntervention,
    InterventionOutcome,
    DataSource
)
from src.dashboard.patient_detail.timeline_service import (
    TimelineService,
    TimeRange,
    TrendDirection
)
from src.dashboard.patient_detail.clinical_actions import (
    ClinicalActionsService,
    ActionType,
    EscalationLevel
)
from src.models.patient import Patient
from src.models.vital_signs import VitalSigns, ConsciousnessLevel
from src.models.news2 import NEWS2Result, RiskCategory


class TestPatientDetailService:
    """Test suite for PatientDetailService"""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        return {
            "patient_registry": Mock(),
            "state_tracker": Mock(),
            "news2_calculator": Mock(),
            "history_service": AsyncMock()
        }

    @pytest.fixture
    def patient_detail_service(self, mock_dependencies):
        """Create PatientDetailService instance"""
        return PatientDetailService(**mock_dependencies)

    @pytest.fixture
    def sample_patient(self):
        """Create sample patient"""
        return Patient(
            patient_id="P001",
            ward_id="ward_a",
            bed_number="A01",
            age=65,
            is_copd_patient=False,
            assigned_nurse_id="Nurse Johnson",
            admission_date=datetime.now() - timedelta(days=2),
            last_updated=datetime.now()
        )

    @pytest.fixture
    def sample_vitals(self):
        """Create sample vital signs"""
        return VitalSigns(
            event_id=uuid.uuid4(),
            patient_id="P001",
            timestamp=datetime.now(),
            respiratory_rate=18,
            sp_o2=95,
            on_oxygen=False,
            temperature=37.5,
            systolic_bp=130,
            heart_rate=80,
            consciousness=ConsciousnessLevel.ALERT,
            is_manual_entry=True
        )

    @pytest.mark.asyncio
    async def test_get_patient_detail_success(self, patient_detail_service, mock_dependencies, sample_patient, sample_vitals):
        """Test successful patient detail retrieval"""
        # Setup mocks
        mock_dependencies["patient_registry"].get_patient.return_value = sample_patient

        mock_state = Mock()
        mock_state.current_vitals = sample_vitals
        mock_state.last_update = datetime.now()
        mock_dependencies["state_tracker"].get_patient_state.return_value = mock_state

        mock_news2 = Mock()
        mock_news2.total_score = 3
        mock_news2.risk_level = RiskCategory.MEDIUM
        mock_dependencies["news2_calculator"].calculate_news2.return_value = mock_news2

        # Mock timeline data
        timeline_data = [
            {
                "timestamp": datetime.now().isoformat(),
                "news2_score": 3,
                "vital_signs": {
                    "respiratory_rate": 18,
                    "heart_rate": 80
                }
            }
        ]
        patient_detail_service._get_timeline_data = AsyncMock(return_value=timeline_data)

        # Get patient detail
        detail = await patient_detail_service.get_patient_detail("P001")

        # Assertions
        assert detail is not None
        assert detail.patient_id == "P001"
        assert detail.current_news2 == 3
        assert detail.risk_level == "medium"
        assert detail.age == 65
        assert len(detail.timeline_data) == 1

    @pytest.mark.asyncio
    async def test_get_patient_detail_not_found(self, patient_detail_service, mock_dependencies):
        """Test patient detail when patient not found"""
        mock_dependencies["patient_registry"].get_patient.return_value = None

        detail = await patient_detail_service.get_patient_detail("NONEXISTENT")

        assert detail is None

    @pytest.mark.asyncio
    async def test_timeline_data_generation(self, patient_detail_service, mock_dependencies):
        """Test timeline data generation"""
        # Setup mock history
        history_data = []
        for i in range(5):
            vitals = VitalSigns(
                event_id=uuid.uuid4(),
                patient_id="P001",
                timestamp=datetime.now() - timedelta(hours=i),
                respiratory_rate=16 + i,
                sp_o2=95 - i,
                on_oxygen=False,
                temperature=37.0 + i * 0.2,
                systolic_bp=120 + i * 5,
                heart_rate=75 + i * 3,
                consciousness=ConsciousnessLevel.ALERT
            )
            history_data.append(vitals)

        mock_dependencies["history_service"].get_patient_history.return_value = history_data

        # Mock NEWS2 calculations
        def mock_calculate_news2(vitals):
            result = Mock()
            result.total_score = 2 + vitals.respiratory_rate - 16  # Simple formula
            return result

        mock_dependencies["news2_calculator"].calculate_news2.side_effect = mock_calculate_news2

        # Get timeline data
        timeline = await patient_detail_service._get_timeline_data("P001", hours=24)

        # Assertions
        assert len(timeline) == 5
        assert all("timestamp" in point for point in timeline)
        assert all("news2_score" in point for point in timeline)
        assert all("vital_signs" in point for point in timeline)

    @pytest.mark.asyncio
    async def test_statistics_calculation(self, patient_detail_service):
        """Test statistics calculation from timeline data"""
        timeline_data = [
            {"timestamp": datetime.now().isoformat(), "news2_score": 2},
            {"timestamp": datetime.now().isoformat(), "news2_score": 4},
            {"timestamp": datetime.now().isoformat(), "news2_score": 6},
            {"timestamp": datetime.now().isoformat(), "news2_score": 3}
        ]

        stats = patient_detail_service._calculate_statistics(timeline_data)

        assert stats["avg_news2"] == 3.8  # (2+4+6+3)/4 = 3.75, rounded to 3.8
        assert stats["max_news2"] == 6
        assert stats["min_news2"] == 2
        assert stats["trend"] in ["improving", "stable", "worsening"]

    @pytest.mark.asyncio
    async def test_add_intervention(self, patient_detail_service):
        """Test adding clinical intervention"""
        intervention = ClinicalIntervention(
            intervention_id="INT001",
            patient_id="P001",
            timestamp=datetime.now(),
            intervention_type="oxygen_therapy",
            description="Started supplemental oxygen",
            performed_by="Nurse Johnson",
            outcome=InterventionOutcome.IMPROVED
        )

        success = await patient_detail_service.add_intervention(intervention)

        assert success is True
        assert "P001" in patient_detail_service._intervention_cache
        assert len(patient_detail_service._intervention_cache["P001"]) > 0

    @pytest.mark.asyncio
    async def test_patient_summary_generation(self, patient_detail_service, mock_dependencies, sample_patient, sample_vitals):
        """Test patient summary generation"""
        # Setup mocks for get_patient_detail
        mock_dependencies["patient_registry"].get_patient.return_value = sample_patient

        mock_state = Mock()
        mock_state.current_vitals = sample_vitals
        mock_state.last_update = datetime.now()
        mock_dependencies["state_tracker"].get_patient_state.return_value = mock_state

        mock_news2 = Mock()
        mock_news2.total_score = 4
        mock_news2.risk_level = RiskCategory.MEDIUM
        mock_dependencies["news2_calculator"].calculate_news2.return_value = mock_news2

        patient_detail_service._get_timeline_data = AsyncMock(return_value=[])

        # Get summary
        summary = await patient_detail_service.get_patient_summary("P001")

        # Assertions
        assert "patient_id" in summary
        assert "current_status" in summary
        assert "24h_statistics" in summary
        assert "clinical_team" in summary
        assert summary["current_status"]["news2_score"] == 4


class TestTimelineService:
    """Test suite for TimelineService"""

    @pytest.fixture
    def timeline_service(self):
        """Create TimelineService instance"""
        return TimelineService()

    @pytest.fixture
    def sample_timeline_data(self):
        """Create sample timeline data"""
        data = []
        for i in range(24):  # 24 hours of data
            timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
            data.append({
                "timestamp": timestamp,
                "news2_score": 3 + (i % 3),  # Varying scores
                "vital_signs": {
                    "respiratory_rate": 16 + (i % 4),
                    "heart_rate": 75 + (i % 10),
                    "temperature": 37.0 + (i % 2) * 0.5
                },
                "is_manual": i % 2 == 0
            })
        return data

    @pytest.mark.asyncio
    async def test_generate_timeline_basic(self, timeline_service, sample_timeline_data):
        """Test basic timeline generation"""
        timeline = await timeline_service.generate_timeline(
            patient_id="P001",
            timeline_data=sample_timeline_data,
            time_range=TimeRange.TWENTY_FOUR_HOURS
        )

        # Assertions
        assert timeline["patient_id"] == "P001"
        assert timeline["time_range"]["hours"] == 24
        assert len(timeline["data_points"]) <= 24
        assert "trend" in timeline
        assert "chart_config" in timeline

    @pytest.mark.asyncio
    async def test_time_range_filtering(self, timeline_service, sample_timeline_data):
        """Test filtering by time range"""
        # Test 4-hour range
        timeline_4h = await timeline_service.generate_timeline(
            patient_id="P001",
            timeline_data=sample_timeline_data,
            time_range=TimeRange.FOUR_HOURS
        )

        # Should have fewer data points for shorter range
        assert len(timeline_4h["data_points"]) <= 4

    @pytest.mark.asyncio
    async def test_trend_calculation(self, timeline_service):
        """Test trend calculation"""
        # Worsening trend data
        worsening_data = [
            {"timestamp": (datetime.now() - timedelta(hours=i)).isoformat(), "news2_score": i + 1}
            for i in range(10)
        ]

        timeline = await timeline_service.generate_timeline(
            patient_id="P001",
            timeline_data=worsening_data
        )

        trend = timeline["trend"]
        assert trend["direction"] in [TrendDirection.WORSENING.value, TrendDirection.STABLE.value]
        assert 0 <= trend["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_critical_events_identification(self, timeline_service):
        """Test identification of critical events"""
        critical_data = [
            {"timestamp": datetime.now().isoformat(), "news2_score": 8},  # Critical score
            {"timestamp": (datetime.now() - timedelta(hours=1)).isoformat(), "news2_score": 4},
            {"timestamp": (datetime.now() - timedelta(hours=2)).isoformat(), "news2_score": 1}  # Rapid increase
        ]

        timeline = await timeline_service.generate_timeline(
            patient_id="P001",
            timeline_data=critical_data
        )

        critical_events = timeline["critical_events"]
        assert len(critical_events) >= 1
        assert any(event["type"] == "critical_news2" for event in critical_events)

    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, timeline_service):
        """Test data quality assessment"""
        # High quality data (frequent, complete)
        high_quality_data = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=i * 30)).isoformat(),
                "news2_score": 3,
                "vital_signs": {"respiratory_rate": 16}
            }
            for i in range(48)  # Every 30 minutes for 24 hours
        ]

        timeline = await timeline_service.generate_timeline(
            patient_id="P001",
            timeline_data=high_quality_data
        )

        data_quality = timeline["data_quality"]
        assert data_quality["quality"] in ["excellent", "good"]
        assert data_quality["score"] >= 3


class TestClinicalActionsService:
    """Test suite for ClinicalActionsService"""

    @pytest.fixture
    def actions_service(self):
        """Create ClinicalActionsService instance"""
        return ClinicalActionsService()

    def test_get_quick_actions(self, actions_service):
        """Test getting available quick actions"""
        actions = actions_service.get_quick_actions()

        assert len(actions) > 0
        assert all("action_id" in action for action in actions)
        assert all("name" in action for action in actions)
        assert all("estimated_duration" in action for action in actions)

        # Check for specific actions
        action_ids = [action["action_id"] for action in actions]
        assert "reposition_patient" in action_ids
        assert "start_oxygen" in action_ids
        assert "vital_signs_recheck" in action_ids

    @pytest.mark.asyncio
    async def test_execute_reposition_action(self, actions_service):
        """Test executing reposition action"""
        result = await actions_service.execute_action(
            patient_id="P001",
            action_id="reposition_patient",
            performer="Nurse Johnson",
            parameters={"position": "right_side"}
        )

        assert result["success"] is True
        assert "execution_id" in result
        assert "estimated_completion" in result
        assert "Patient repositioned" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_oxygen_therapy_action(self, actions_service):
        """Test executing oxygen therapy action"""
        result = await actions_service.execute_action(
            patient_id="P001",
            action_id="start_oxygen",
            performer="Dr. Smith",
            parameters={"flow_rate": "4L/min", "delivery_method": "face_mask"}
        )

        assert result["success"] is True
        assert "Oxygen therapy started" in result["message"]
        assert "next_steps" in result
        assert len(result["next_steps"]) > 0

    @pytest.mark.asyncio
    async def test_execute_escalation_action(self, actions_service):
        """Test executing escalation action"""
        result = await actions_service.execute_action(
            patient_id="P001",
            action_id="rapid_response",
            performer="Nurse Johnson",
            parameters={"reason": "clinical_deterioration"}
        )

        assert result["success"] is True
        assert "Rapid Response Team activated" in result["message"]

    @pytest.mark.asyncio
    async def test_execution_history_tracking(self, actions_service):
        """Test execution history tracking"""
        # Execute several actions
        await actions_service.execute_action("P001", "reposition_patient", "Nurse A")
        await actions_service.execute_action("P001", "start_oxygen", "Nurse B")
        await actions_service.execute_action("P001", "vital_signs_recheck", "Nurse C")

        # Get history
        history = actions_service.get_execution_history("P001", hours=24)

        assert len(history) == 3
        assert all("execution_id" in record for record in history)
        assert all("performer" in record for record in history)
        assert all("timestamp" in record for record in history)

        # Check chronological order (most recent first)
        timestamps = [record["timestamp"] for record in history]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.asyncio
    async def test_invalid_action_execution(self, actions_service):
        """Test executing invalid action"""
        result = await actions_service.execute_action(
            patient_id="P001",
            action_id="invalid_action",
            performer="Nurse Johnson"
        )

        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_escalation_contacts(self, actions_service):
        """Test getting escalation contacts"""
        contacts = actions_service.get_escalation_contacts()

        assert "charge_nurse" in contacts
        assert "attending_physician" in contacts
        assert "rapid_response" in contacts

        # Check contact structure
        for contact_type, contact_info in contacts.items():
            assert "name" in contact_info
            assert "phone" in contact_info
            assert "availability" in contact_info

    @pytest.mark.asyncio
    async def test_handoff_summary_generation(self, actions_service):
        """Test handoff summary generation"""
        # Execute some actions first
        await actions_service.execute_action("P001", "reposition_patient", "Nurse A")
        await actions_service.execute_action("P001", "start_oxygen", "Nurse B")

        # Generate handoff summary
        summary = await actions_service.generate_handoff_summary("P001", "Night Nurse")

        assert summary["patient_id"] == "P001"
        assert summary["recipient"] == "Night Nurse"
        assert "recent_actions" in summary
        assert "key_points" in summary
        assert "pending_tasks" in summary
        assert len(summary["recent_actions"]) >= 2