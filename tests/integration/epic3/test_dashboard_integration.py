"""
Integration tests for Ward Dashboard (Story 3.1)
Tests end-to-end dashboard functionality with real Epic 1 & 2 integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.dashboard.services.ward_dashboard_service import (
    WardDashboardService,
    PatientFilter,
    PatientSortOption
)
from src.dashboard.services.dashboard_api import router, get_ward_service
from src.dashboard.services.dashboard_cache import DashboardCacheManager

from src.models.patient import Patient
from src.models.vital_signs import VitalSigns
from src.models.patient_state import PatientState
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker
from src.services.alert_generation import AlertGenerator
from src.services.news2_calculator import NEWS2Calculator
from src.services.validation import VitalSignsValidator


class TestDashboardIntegration:
    """Integration tests for Ward Dashboard"""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with dashboard routes"""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def setup_services(self):
        """Setup real services for integration testing"""
        # Create service instances
        patient_registry = PatientRegistry()
        state_tracker = PatientStateTracker(patient_registry)
        alert_service = AlertGenerator()
        news2_calculator = NEWS2Calculator()
        validation_service = VitalSignsValidator()

        # Create patients in ward
        patients = []
        for i in range(10):
            patient = Patient(
                patient_id=f"TEST_{i:03d}",
                name=f"Test Patient {i}",
                date_of_birth=datetime(1950 + i, 1, 1),
                admission_date=datetime.utcnow() - timedelta(days=i),
                metadata={
                    "ward_id": "test_ward",
                    "bed_number": f"T{i:02d}",
                    "attending_physician": "Dr. Test",
                    "primary_nurse": "Nurse Test"
                }
            )
            patient_registry.register_patient(patient)
            patients.append(patient)

        # Create varied vital signs for patients
        vital_signs_sets = [
            VitalSigns(  # Low risk
                respiratory_rate=16,
                oxygen_saturation=96,
                supplemental_oxygen=False,
                temperature=37.0,
                systolic_blood_pressure=120,
                heart_rate=75,
                consciousness_level="A",
                timestamp=datetime.utcnow()
            ),
            VitalSigns(  # Medium risk
                respiratory_rate=22,
                oxygen_saturation=94,
                supplemental_oxygen=False,
                temperature=38.0,
                systolic_blood_pressure=110,
                heart_rate=95,
                consciousness_level="A",
                timestamp=datetime.utcnow()
            ),
            VitalSigns(  # High risk
                respiratory_rate=25,
                oxygen_saturation=92,
                supplemental_oxygen=True,
                temperature=39.0,
                systolic_blood_pressure=100,
                heart_rate=110,
                consciousness_level="V",
                timestamp=datetime.utcnow()
            ),
            VitalSigns(  # Critical risk
                respiratory_rate=30,
                oxygen_saturation=88,
                supplemental_oxygen=True,
                temperature=40.0,
                systolic_blood_pressure=85,
                heart_rate=130,
                consciousness_level="P",
                timestamp=datetime.utcnow()
            )
        ]

        # Update patient states with vital signs
        for i, patient in enumerate(patients):
            vitals = vital_signs_sets[i % len(vital_signs_sets)]
            state_tracker.update_patient_state(
                patient_id=patient.patient_id,
                vital_signs=vitals
            )

            # Generate alerts for high-risk patients
            if i % 4 >= 2:  # High and critical risk patients
                alert_service._generate_news2_alert(
                    patient_id=patient.patient_id,
                    news2_score=news2_calculator.calculate_news2(vitals).total_score,
                    risk_level=news2_calculator.calculate_news2(vitals).risk_level.value
                )

        return {
            "patient_registry": patient_registry,
            "state_tracker": state_tracker,
            "alert_service": alert_service,
            "news2_calculator": news2_calculator,
            "validation_service": validation_service,
            "patients": patients
        }

    def test_ward_overview_endpoint(self, client, setup_services):
        """Test ward overview API endpoint"""
        # Get ward overview
        response = client.get("/api/dashboard/ward/test_ward/overview")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["ward_id"] == "test_ward"
        assert data["patient_count"] == 10
        assert len(data["patients"]) == 10
        assert "statistics" in data
        assert "filters_applied" in data

        # Verify patient tile structure
        for tile in data["patients"]:
            assert "patient_id" in tile
            assert "patient_name" in tile
            assert "bed_number" in tile
            assert "news2_score" in tile
            assert "risk_level" in tile
            assert "tile_color" in tile
            assert "alert_status" in tile

    def test_filtering_by_risk_level(self, client, setup_services):
        """Test filtering patients by risk level"""
        # Get high risk patients only
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"filter": "high"}
        )

        assert response.status_code == 200
        data = response.json()

        # All patients should be high risk
        for tile in data["patients"]:
            if tile["news2_score"] is not None:
                assert tile["risk_level"] == "high"

    def test_search_functionality_integration(self, client, setup_services):
        """Test search functionality with real data"""
        # Search by patient name
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"search": "Test Patient 5"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["patient_count"] == 1
        assert data["patients"][0]["patient_name"] == "Test Patient 5"

        # Search by bed number
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"search": "T03"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["patient_count"] == 1
        assert data["patients"][0]["bed_number"] == "T03"

    def test_sorting_integration(self, client, setup_services):
        """Test sorting with real data"""
        # Sort by NEWS2 score
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"sort": "news2_desc"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify descending order
        scores = [t["news2_score"] for t in data["patients"] if t["news2_score"] is not None]
        assert scores == sorted(scores, reverse=True)

        # Sort by bed number
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"sort": "bed_number"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify bed number order
        bed_numbers = [t["bed_number"] for t in data["patients"]]
        assert bed_numbers == sorted(bed_numbers)

    def test_refresh_endpoint(self, client, setup_services):
        """Test dashboard refresh functionality"""
        # Refresh dashboard
        response = client.post("/api/dashboard/ward/test_ward/refresh")

        assert response.status_code == 200
        data = response.json()

        # Should return fresh data
        assert data["ward_id"] == "test_ward"
        assert "patients" in data

    def test_patient_preview_endpoint(self, client, setup_services):
        """Test patient preview for navigation"""
        # Get patient preview
        response = client.get("/api/dashboard/patient/TEST_001/preview")

        assert response.status_code == 200
        data = response.json()

        assert data["patient_id"] == "TEST_001"
        assert data["patient_name"] == "Test Patient 1"
        assert "navigation_url" in data

    def test_ward_statistics_endpoint(self, client, setup_services):
        """Test ward statistics endpoint"""
        # Get ward statistics
        response = client.get("/api/dashboard/ward/test_ward/statistics")

        assert response.status_code == 200
        data = response.json()

        assert data["ward_id"] == "test_ward"
        assert "total_patients" in data
        assert "risk_distribution" in data
        assert "average_news2" in data
        assert "performance" in data

    def test_available_wards_endpoint(self, client):
        """Test available wards listing"""
        response = client.get("/api/dashboard/wards")

        assert response.status_code == 200
        data = response.json()

        assert "wards" in data
        assert len(data["wards"]) > 0
        assert "total_wards" in data

    def test_health_check_endpoint(self, client):
        """Test dashboard health check"""
        response = client.get("/api/dashboard/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "ward_dashboard"

    def test_error_handling_invalid_filter(self, client):
        """Test error handling for invalid filter"""
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"filter": "invalid_filter"}
        )

        assert response.status_code == 400
        assert "Invalid filter option" in response.json()["detail"]

    def test_error_handling_invalid_sort(self, client):
        """Test error handling for invalid sort option"""
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"sort": "invalid_sort"}
        )

        assert response.status_code == 400
        assert "Invalid sort option" in response.json()["detail"]

    def test_limit_parameter(self, client, setup_services):
        """Test limit parameter for patient count"""
        # Request only 5 patients
        response = client.get(
            "/api/dashboard/ward/test_ward/overview",
            params={"limit": 5}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["patients"]) <= 5

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, setup_services):
        """Test handling of concurrent dashboard requests"""
        # Simulate concurrent requests
        tasks = []
        for i in range(10):
            response = client.get(
                "/api/dashboard/ward/test_ward/overview",
                params={"filter": ["all", "low", "medium", "high", "critical"][i % 5]}
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_performance(self, setup_services):
        """Test cache performance with real services"""
        # Create dashboard service
        ward_service = WardDashboardService(
            patient_registry=setup_services["patient_registry"],
            state_tracker=setup_services["state_tracker"],
            alert_service=setup_services["alert_service"],
            news2_calculator=setup_services["news2_calculator"]
        )

        # First call - cache miss
        start = datetime.utcnow()
        overview1 = await ward_service.get_ward_overview("test_ward")
        first_call_time = (datetime.utcnow() - start).total_seconds()

        # Second call - cache hit
        start = datetime.utcnow()
        overview2 = await ward_service.get_ward_overview("test_ward")
        second_call_time = (datetime.utcnow() - start).total_seconds()

        # Cache should be significantly faster
        assert second_call_time < first_call_time / 2

        # Data should be identical
        assert overview1["patient_count"] == overview2["patient_count"]

    @pytest.mark.asyncio
    async def test_epic1_epic2_integration(self, setup_services):
        """Test integration with Epic 1 and Epic 2 services"""
        # Create dashboard service
        ward_service = WardDashboardService(
            patient_registry=setup_services["patient_registry"],
            state_tracker=setup_services["state_tracker"],
            alert_service=setup_services["alert_service"],
            news2_calculator=setup_services["news2_calculator"]
        )

        # Get ward overview
        overview = await ward_service.get_ward_overview("test_ward")

        # Verify Epic 1 integration (NEWS2 scores)
        for tile in overview["patients"]:
            if tile["news2_score"] is not None:
                assert tile["news2_score"] >= 0
                assert tile["risk_level"] in ["low", "medium", "high", "critical"]

        # Verify Epic 2 integration (alert status)
        high_risk_patients = [
            t for t in overview["patients"]
            if t.get("news2_score", 0) >= 5
        ]

        for tile in high_risk_patients:
            alert_status = tile["alert_status"]
            # High risk patients should have alerts
            if tile["news2_score"] >= 7:
                assert alert_status["total"] > 0

    @pytest.mark.asyncio
    async def test_real_time_updates_simulation(self, setup_services):
        """Simulate real-time updates to patient data"""
        ward_service = WardDashboardService(
            patient_registry=setup_services["patient_registry"],
            state_tracker=setup_services["state_tracker"],
            alert_service=setup_services["alert_service"],
            news2_calculator=setup_services["news2_calculator"]
        )

        # Get initial overview
        overview1 = await ward_service.get_ward_overview("test_ward")
        initial_patient = overview1["patients"][0]

        # Update patient vital signs
        new_vitals = VitalSigns(
            respiratory_rate=30,  # Critical value
            oxygen_saturation=88,
            supplemental_oxygen=True,
            temperature=40.0,
            systolic_blood_pressure=85,
            heart_rate=130,
            consciousness_level="P",
            timestamp=datetime.utcnow()
        )

        setup_services["state_tracker"].update_patient_state(
            patient_id=initial_patient["patient_id"],
            vital_signs=new_vitals
        )

        # Refresh and get updated overview
        overview2 = await ward_service.refresh_dashboard("test_ward")
        updated_patient = next(
            p for p in overview2["patients"]
            if p["patient_id"] == initial_patient["patient_id"]
        )

        # Patient should now have higher NEWS2 score
        assert updated_patient["news2_score"] > initial_patient.get("news2_score", 0)
        assert updated_patient["tile_color"] == "red"
        assert updated_patient["risk_level"] == "critical"


class TestDashboardPerformance:
    """Performance tests for Ward Dashboard"""

    @pytest.mark.asyncio
    async def test_50_patient_load_time(self, setup_services):
        """Test dashboard load time with 50 patients"""
        # Create 50 patients
        patient_registry = PatientRegistry()
        state_tracker = PatientStateTracker(patient_registry)

        for i in range(50):
            patient = Patient(
                patient_id=f"PERF_{i:03d}",
                name=f"Performance Patient {i}",
                date_of_birth=datetime(1950, 1, 1),
                admission_date=datetime.utcnow(),
                metadata={
                    "ward_id": "perf_ward",
                    "bed_number": f"P{i:02d}"
                }
            )
            patient_registry.register_patient(patient)

            # Add vital signs
            vitals = VitalSigns(
                respiratory_rate=16 + (i % 10),
                oxygen_saturation=96 - (i % 5),
                supplemental_oxygen=i % 3 == 0,
                temperature=37.0 + (i % 4) * 0.5,
                systolic_blood_pressure=120 - (i % 20),
                heart_rate=75 + (i % 30),
                consciousness_level="A",
                timestamp=datetime.utcnow()
            )
            state_tracker.update_patient_state(patient.patient_id, vitals)

        # Create dashboard service
        ward_service = WardDashboardService(
            patient_registry=patient_registry,
            state_tracker=state_tracker,
            alert_service=AlertGenerator(),
            news2_calculator=NEWS2Calculator()
        )

        # Measure load time
        start = datetime.utcnow()
        overview = await ward_service.get_ward_overview("perf_ward", limit=50)
        load_time = (datetime.utcnow() - start).total_seconds()

        # Should load within 2 seconds
        assert load_time < 2.0
        assert overview["patient_count"] == 50

    @pytest.mark.asyncio
    async def test_filter_operation_performance(self, setup_services):
        """Test filter operation performance"""
        ward_service = WardDashboardService(
            patient_registry=setup_services["patient_registry"],
            state_tracker=setup_services["state_tracker"],
            alert_service=setup_services["alert_service"],
            news2_calculator=setup_services["news2_calculator"]
        )

        # Pre-load data
        await ward_service.get_ward_overview("test_ward")

        # Measure filter operation time
        start = datetime.utcnow()
        filtered = await ward_service.get_ward_overview(
            "test_ward",
            filter_option=PatientFilter.HIGH_RISK
        )
        filter_time = (datetime.utcnow() - start).total_seconds()

        # Should complete within 500ms
        assert filter_time < 0.5

    @pytest.mark.asyncio
    async def test_cache_manager_performance(self):
        """Test cache manager performance"""
        manager = DashboardCacheManager()
        await manager.start()

        try:
            # Cache many items
            for i in range(100):
                await manager.cache_patient_tile(f"P{i:03d}", {"data": f"patient_{i}"})

            # Measure retrieval time
            start = datetime.utcnow()
            for i in range(100):
                await manager.get_patient_tile(f"P{i:03d}")
            retrieval_time = (datetime.utcnow() - start).total_seconds()

            # Should be very fast
            assert retrieval_time < 0.1  # 100ms for 100 retrievals

            # Check cache statistics
            stats = manager.get_cache_statistics()
            assert stats["patient_cache"]["hit_count"] >= 100

        finally:
            await manager.stop()