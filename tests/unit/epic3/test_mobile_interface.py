"""
Unit tests for Mobile Interface Service (Story 3.4)
Tests mobile-optimized clinical interface functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.mobile.mobile_interface_service import (
    MobileInterfaceService,
    ScreenSize,
    DeviceOrientation,
    TouchGesture,
    MobilePatientCard,
    NavigationState
)


class TestMobileInterfaceService:
    """Test suite for MobileInterfaceService"""

    @pytest.fixture
    def mobile_interface(self):
        """Create MobileInterfaceService instance"""
        return MobileInterfaceService()

    @pytest.mark.asyncio
    async def test_get_mobile_ward_overview_tablet(self, mobile_interface):
        """Test mobile ward overview for tablet"""
        overview = await mobile_interface.get_mobile_ward_overview(
            "ward_a",
            ScreenSize.TABLET,
            DeviceOrientation.PORTRAIT
        )

        # Verify structure
        assert "ward_id" in overview
        assert "layout" in overview
        assert "patients" in overview
        assert "quick_stats" in overview
        assert "navigation" in overview

        # Verify tablet layout
        assert overview["layout"]["grid_columns"] == 2
        assert overview["layout"]["card_width"] == "48%"
        assert overview["layout"]["touch_target_size"] == 48

        # Verify patient data
        assert len(overview["patients"]) > 0
        patient = overview["patients"][0]
        assert "patient_id" in patient
        assert "news2_score" in patient
        assert "compact_mode" in patient
        assert not patient["compact_mode"]  # Not compact for tablet

    @pytest.mark.asyncio
    async def test_get_mobile_ward_overview_smartphone(self, mobile_interface):
        """Test mobile ward overview for smartphone"""
        overview = await mobile_interface.get_mobile_ward_overview(
            "ward_a",
            ScreenSize.SMARTPHONE,
            DeviceOrientation.PORTRAIT
        )

        # Verify smartphone layout
        assert overview["layout"]["grid_columns"] == 1
        assert overview["layout"]["card_width"] == "100%"
        assert overview["layout"]["touch_target_size"] == 44

        # Verify compact mode
        patient = overview["patients"][0]
        assert patient["compact_mode"]
        assert len(patient["quick_actions"]) <= 3  # Limited for smartphone

    @pytest.mark.asyncio
    async def test_handle_swipe_navigation(self, mobile_interface):
        """Test swipe gesture navigation"""
        context = {
            "current_view": "patient_detail",
            "patient_list": [
                {"patient_id": "P001"},
                {"patient_id": "P002"},
                {"patient_id": "P003"}
            ],
            "current_index": 1
        }

        # Test swipe left (next patient)
        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.SWIPE_LEFT, context
        )

        assert result["success"]
        assert result["action"] == "navigate_patient"
        assert result["patient_id"] == "P003"
        assert result["index"] == 2

        # Test swipe right (previous patient)
        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.SWIPE_RIGHT, context
        )

        assert result["success"]
        assert result["patient_id"] == "P001"
        assert result["index"] == 0

    @pytest.mark.asyncio
    async def test_handle_pull_to_refresh(self, mobile_interface):
        """Test pull-to-refresh gesture"""
        context = {"current_view": "ward_overview"}

        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.SWIPE_UP, context
        )

        assert result["success"]
        assert result["action"] == "refresh_ward"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_handle_long_press_quick_actions(self, mobile_interface):
        """Test long press for quick actions menu"""
        context = {"patient_id": "P001"}

        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.LONG_PRESS, context
        )

        assert result["success"]
        assert result["action"] == "show_quick_menu"
        assert result["patient_id"] == "P001"
        assert "actions" in result
        assert len(result["actions"]) > 0

        # Verify action structure
        action = result["actions"][0]
        assert "id" in action
        assert "name" in action
        assert "icon" in action
        assert "urgent" in action

    @pytest.mark.asyncio
    async def test_handle_pinch_zoom(self, mobile_interface):
        """Test pinch-to-zoom gesture"""
        context = {
            "chart_type": "timeline",
            "zoom_level": 1.5
        }

        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.PINCH_ZOOM, context
        )

        assert result["success"]
        assert result["action"] == "update_zoom"
        assert result["chart_type"] == "timeline"
        assert 0.5 <= result["zoom_level"] <= 3.0

    @pytest.mark.asyncio
    async def test_get_mobile_patient_detail_smartphone(self, mobile_interface):
        """Test mobile patient detail for smartphone"""
        detail = await mobile_interface.get_mobile_patient_detail(
            "P001", ScreenSize.SMARTPHONE
        )

        assert detail["patient_id"] == "P001"
        assert detail["mobile_layout"]["compact"]
        assert "vital_signs" in detail["mobile_layout"]["sections"]
        assert len(detail["mobile_layout"]["action_buttons"]) == 2  # Limited for smartphone

    @pytest.mark.asyncio
    async def test_get_mobile_patient_detail_tablet(self, mobile_interface):
        """Test mobile patient detail for tablet"""
        detail = await mobile_interface.get_mobile_patient_detail(
            "P001", ScreenSize.TABLET
        )

        assert not detail["mobile_layout"]["compact"]
        assert "timeline" in detail["mobile_layout"]["sections"]
        assert len(detail["mobile_layout"]["action_buttons"]) >= 4  # More for tablet

    @pytest.mark.asyncio
    async def test_network_optimization(self, mobile_interface):
        """Test network-based optimizations"""
        # Test 2G optimization
        optimizations = await mobile_interface.optimize_for_network("2g")

        assert optimizations["success"]
        assert optimizations["optimizations"]["image_quality"] == "low"
        assert optimizations["optimizations"]["chart_detail"] == "minimal"
        assert optimizations["optimizations"]["refresh_interval"] == 60000

        # Test WiFi optimization
        optimizations = await mobile_interface.optimize_for_network("wifi")

        assert optimizations["optimizations"]["image_quality"] == "high"
        assert optimizations["optimizations"]["chart_detail"] == "full"
        assert optimizations["optimizations"]["refresh_interval"] == 15000

    @pytest.mark.asyncio
    async def test_mobile_metrics(self, mobile_interface):
        """Test mobile interface metrics collection"""
        metrics = await mobile_interface.get_mobile_metrics()

        assert "interface_metrics" in metrics
        assert "usage_stats" in metrics
        assert "performance_score" in metrics

        interface_metrics = metrics["interface_metrics"]
        assert "load_time_ms" in interface_metrics
        assert "navigation_time_ms" in interface_metrics
        assert "memory_usage_mb" in interface_metrics
        assert "battery_impact" in interface_metrics

    def test_navigation_state_management(self, mobile_interface):
        """Test navigation state management"""
        nav_state = NavigationState("initial_view")

        # Test view navigation
        nav_state.push_view("patient_detail")
        assert nav_state.current_view == "patient_detail"
        assert "initial_view" in nav_state.previous_views

        # Test going back
        previous = nav_state.pop_view()
        assert previous == "patient_detail"
        assert nav_state.current_view == "initial_view"

        # Test modal management
        nav_state.push_modal("action_menu")
        assert "action_menu" in nav_state.modal_stack

        modal = nav_state.pop_modal()
        assert modal == "action_menu"
        assert len(nav_state.modal_stack) == 0


class TestMobilePatientCard:
    """Test suite for MobilePatientCard"""

    @pytest.fixture
    def sample_card(self):
        """Create sample mobile patient card"""
        return MobilePatientCard(
            patient_id="P001",
            patient_name="John Doe",
            bed_number="A01",
            news2_score=3,
            risk_level="medium",
            status_color="#f59e0b",
            last_update=datetime.now(),
            alerts_count=1,
            quick_actions=["vitals", "notes", "call", "escalate", "medication", "oxygen"]
        )

    def test_to_mobile_dict_smartphone(self, sample_card):
        """Test mobile dictionary conversion for smartphone"""
        mobile_dict = sample_card.to_mobile_dict(ScreenSize.SMARTPHONE)

        assert mobile_dict["compact_mode"]
        assert not mobile_dict["show_details"]
        assert len(mobile_dict["quick_actions"]) == 3  # Limited for smartphone

    def test_to_mobile_dict_tablet(self, sample_card):
        """Test mobile dictionary conversion for tablet"""
        mobile_dict = sample_card.to_mobile_dict(ScreenSize.TABLET)

        assert not mobile_dict["compact_mode"]
        assert mobile_dict["show_details"]
        assert len(mobile_dict["quick_actions"]) == 6  # Full actions for tablet

    def test_card_data_structure(self, sample_card):
        """Test card data structure"""
        mobile_dict = sample_card.to_mobile_dict(ScreenSize.TABLET)

        required_fields = [
            "patient_id", "patient_name", "bed_number", "news2_score",
            "risk_level", "status_color", "last_update", "alerts_count"
        ]

        for field in required_fields:
            assert field in mobile_dict

        # Test time formatting
        assert len(mobile_dict["last_update"]) == 5  # HH:MM format


class TestMobileLayoutConfiguration:
    """Test suite for layout configuration"""

    @pytest.fixture
    def mobile_interface(self):
        """Create MobileInterfaceService instance"""
        return MobileInterfaceService()

    def test_smartphone_portrait_layout(self, mobile_interface):
        """Test smartphone portrait layout configuration"""
        config = mobile_interface._layout_configs[
            (ScreenSize.SMARTPHONE, DeviceOrientation.PORTRAIT)
        ]

        assert config.grid_columns == 1
        assert config.card_width == "100%"
        assert config.font_size_base == 16
        assert config.touch_target_size == 44
        assert config.spacing_unit == 8

    def test_smartphone_landscape_layout(self, mobile_interface):
        """Test smartphone landscape layout configuration"""
        config = mobile_interface._layout_configs[
            (ScreenSize.SMARTPHONE, DeviceOrientation.LANDSCAPE)
        ]

        assert config.grid_columns == 2
        assert config.card_width == "48%"
        assert config.font_size_base == 14
        assert config.touch_target_size == 40
        assert config.spacing_unit == 6

    def test_tablet_portrait_layout(self, mobile_interface):
        """Test tablet portrait layout configuration"""
        config = mobile_interface._layout_configs[
            (ScreenSize.TABLET, DeviceOrientation.PORTRAIT)
        ]

        assert config.grid_columns == 2
        assert config.card_width == "48%"
        assert config.font_size_base == 18
        assert config.touch_target_size == 48
        assert config.spacing_unit == 12

    def test_tablet_landscape_layout(self, mobile_interface):
        """Test tablet landscape layout configuration"""
        config = mobile_interface._layout_configs[
            (ScreenSize.TABLET, DeviceOrientation.LANDSCAPE)
        ]

        assert config.grid_columns == 3
        assert config.card_width == "32%"
        assert config.font_size_base == 16
        assert config.touch_target_size == 44
        assert config.spacing_unit == 10


class TestMobileErrorHandling:
    """Test suite for mobile interface error handling"""

    @pytest.fixture
    def mobile_interface(self):
        """Create MobileInterfaceService instance"""
        return MobileInterfaceService()

    @pytest.mark.asyncio
    async def test_invalid_gesture_handling(self, mobile_interface):
        """Test handling of invalid gestures"""
        with patch.object(mobile_interface, '_handle_swipe_navigation',
                         side_effect=Exception("Navigation error")):
            result = await mobile_interface.handle_touch_gesture(
                TouchGesture.SWIPE_LEFT, {"current_view": "test"}
            )

            assert not result["success"]
            assert "error" in result

    @pytest.mark.asyncio
    async def test_error_response_format(self, mobile_interface):
        """Test error response format"""
        error_response = mobile_interface._get_error_response("Test error")

        assert error_response["error"]
        assert error_response["message"] == "Test error"
        assert "timestamp" in error_response
        assert error_response["retry_action"] == "refresh"

    @pytest.mark.asyncio
    async def test_swipe_navigation_bounds(self, mobile_interface):
        """Test swipe navigation boundary conditions"""
        context = {
            "current_view": "patient_detail",
            "patient_list": [{"patient_id": "P001"}],
            "current_index": 0
        }

        # Test swipe left at end of list
        context["current_index"] = 0
        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.SWIPE_RIGHT, context
        )

        assert not result["success"]
        assert "No valid swipe target" in result["error"]

        # Test swipe right at beginning of list
        result = await mobile_interface.handle_touch_gesture(
            TouchGesture.SWIPE_LEFT, context
        )

        assert not result["success"]

    def test_invalid_screen_size_fallback(self, mobile_interface):
        """Test fallback behavior for invalid screen size"""
        # Test with non-existent layout configuration
        layout = mobile_interface._layout_configs.get(
            (ScreenSize.DESKTOP, DeviceOrientation.PORTRAIT)
        )

        if not layout:
            # Should fall back to tablet portrait
            fallback = mobile_interface._layout_configs[
                (ScreenSize.TABLET, DeviceOrientation.PORTRAIT)
            ]
            assert fallback is not None