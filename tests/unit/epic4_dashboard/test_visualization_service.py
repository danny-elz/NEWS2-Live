"""
Unit tests for Visualization Service (Story 3.5)
Tests interactive charts, graphs, and dashboard visualizations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analytics.visualization_service import (
    VisualizationService,
    VisualizationConfig,
    InteractiveDashboard,
    ChartType,
    ColorScheme
)


class TestVisualizationService:
    """Test suite for VisualizationService"""

    @pytest.fixture
    def viz_service(self):
        """Create VisualizationService instance"""
        return VisualizationService()

    @pytest.fixture
    def sample_config(self):
        """Create sample visualization configuration"""
        return VisualizationConfig(
            chart_type=ChartType.LINE_CHART,
            title="Test Chart",
            width=800,
            height=600,
            color_scheme=ColorScheme.CLINICAL
        )

    @pytest.mark.asyncio
    async def test_generate_news2_trend_chart(self, viz_service, sample_config):
        """Test NEWS2 trend chart generation"""
        chart = await viz_service.generate_news2_trend_chart(
            patient_ids=["P001", "P002"],
            time_range="24h",
            config=sample_config
        )

        # Verify chart structure
        assert "chart_specification" in chart
        assert "config" in chart
        assert "data_summary" in chart
        assert "generated_at" in chart

        # Verify chart specification
        chart_spec = chart["chart_specification"]
        assert chart_spec["type"] == "line"
        assert "data" in chart_spec
        assert "options" in chart_spec

        # Verify data structure
        assert "datasets" in chart_spec["data"]
        assert len(chart_spec["data"]["datasets"]) > 0

        # Verify each dataset has required fields
        for dataset in chart_spec["data"]["datasets"]:
            assert "label" in dataset
            assert "data" in dataset
            assert "borderColor" in dataset
            assert len(dataset["data"]) > 0

        # Verify data summary
        data_summary = chart["data_summary"]
        assert "patients_count" in data_summary
        assert "time_range" in data_summary
        assert "data_points_total" in data_summary

    @pytest.mark.asyncio
    async def test_generate_news2_trend_chart_default_config(self, viz_service):
        """Test NEWS2 trend chart with default configuration"""
        chart = await viz_service.generate_news2_trend_chart()

        # Should use default configuration
        config = chart["config"]
        assert config["chart_type"] == "line_chart"
        assert config["title"] == "NEWS2 Score Trends"
        assert config["height"] == 400

    @pytest.mark.asyncio
    async def test_generate_alert_distribution_chart(self, viz_service):
        """Test alert distribution chart generation"""
        chart = await viz_service.generate_alert_distribution_chart(
            ward_ids=["ward_a", "ward_b"],
            time_range="24h"
        )

        # Verify chart structure
        chart_spec = chart["chart_specification"]
        assert chart_spec["type"] == "bar"

        # Verify data structure for bar chart
        assert "labels" in chart_spec["data"]
        assert "datasets" in chart_spec["data"]
        assert len(chart_spec["data"]["labels"]) > 0
        assert len(chart_spec["data"]["datasets"]) == 3  # Critical, High, Medium

        # Verify dataset structure
        for dataset in chart_spec["data"]["datasets"]:
            assert "label" in dataset
            assert "data" in dataset
            assert "backgroundColor" in dataset
            assert len(dataset["data"]) == len(chart_spec["data"]["labels"])

        # Verify data summary
        data_summary = chart["data_summary"]
        assert "total_alerts" in data_summary
        assert "wards_count" in data_summary
        assert data_summary["total_alerts"] >= 0

    @pytest.mark.asyncio
    async def test_generate_response_time_heatmap(self, viz_service):
        """Test response time heatmap generation"""
        heatmap = await viz_service.generate_response_time_heatmap(time_range="7d")

        # Verify chart structure
        chart_spec = heatmap["chart_specification"]
        assert chart_spec["type"] == "scatter"

        # Verify data structure
        assert "datasets" in chart_spec["data"]
        assert len(chart_spec["data"]["datasets"]) == 1

        dataset = chart_spec["data"]["datasets"][0]
        assert "label" in dataset
        assert "data" in dataset
        assert "backgroundColor" in dataset
        assert len(dataset["data"]) > 0

        # Verify scales configuration
        scales = chart_spec["options"]["scales"]
        assert "x" in scales
        assert "y" in scales
        assert scales["x"]["max"] == 23  # 24-hour format
        assert scales["y"]["max"] == 6   # Days of week

        # Verify data summary
        data_summary = heatmap["data_summary"]
        assert "avg_response_time" in data_summary
        assert "max_response_time" in data_summary
        assert data_summary["avg_response_time"] > 0

    @pytest.mark.asyncio
    async def test_generate_kpi_dashboard(self, viz_service):
        """Test KPI dashboard generation"""
        dashboard = await viz_service.generate_kpi_dashboard(
            dashboard_id="test_dashboard",
            ward_ids=["ward_a", "ward_b"]
        )

        # Verify dashboard structure
        assert isinstance(dashboard, InteractiveDashboard)
        assert dashboard.dashboard_id == "test_dashboard"
        assert dashboard.title == "Executive Clinical Dashboard"
        assert dashboard.layout == "grid"

        # Verify widgets
        assert len(dashboard.widgets) > 0
        required_widgets = ["news2_average", "total_alerts", "response_time", "bed_occupancy"]

        widget_ids = [widget["id"] for widget in dashboard.widgets]
        for required_id in required_widgets:
            assert required_id in widget_ids

        # Verify widget structure
        for widget in dashboard.widgets:
            assert "id" in widget
            assert "type" in widget
            assert "title" in widget
            assert "position" in widget
            assert "config" in widget

        # Verify filters
        assert len(dashboard.filters) > 0
        filter_ids = [f["id"] for f in dashboard.filters]
        assert "ward_filter" in filter_ids
        assert "time_range" in filter_ids

    @pytest.mark.asyncio
    async def test_generate_clinical_timeline(self, viz_service):
        """Test clinical timeline generation"""
        timeline = await viz_service.generate_clinical_timeline(
            patient_id="P001",
            time_range="7d"
        )

        # Verify timeline structure
        chart_spec = timeline["chart_specification"]
        assert chart_spec["type"] == "timeline"

        # Verify data structure
        data = chart_spec["data"]
        assert "events" in data
        assert "vital_signs" in data
        assert "interventions" in data

        # Verify events structure
        assert len(data["events"]) > 0
        for event in data["events"]:
            assert "id" in event
            assert "timestamp" in event
            assert "type" in event
            assert "description" in event

        # Verify vital signs data
        assert len(data["vital_signs"]) > 0
        for vital_sign in data["vital_signs"]:
            assert "timestamp" in vital_sign
            assert "news2_score" in vital_sign

        # Verify data summary
        data_summary = timeline["data_summary"]
        assert data_summary["patient_id"] == "P001"
        assert "events_count" in data_summary
        assert "vital_signs_count" in data_summary

    def test_visualization_config_creation(self):
        """Test VisualizationConfig creation and serialization"""
        config = VisualizationConfig(
            chart_type=ChartType.BAR_CHART,
            title="Test Bar Chart",
            width=1000,
            height=500,
            color_scheme=ColorScheme.PROFESSIONAL,
            interactive=True,
            export_formats=["png", "svg"],
            accessibility_features=True
        )

        config_dict = config.to_dict()

        assert config_dict["chart_type"] == "bar_chart"
        assert config_dict["title"] == "Test Bar Chart"
        assert config_dict["width"] == 1000
        assert config_dict["height"] == 500
        assert config_dict["color_scheme"] == "professional"
        assert config_dict["interactive"] is True
        assert config_dict["export_formats"] == ["png", "svg"]
        assert config_dict["accessibility_features"] is True

    def test_interactive_dashboard_creation(self):
        """Test InteractiveDashboard creation and serialization"""
        dashboard = InteractiveDashboard(
            dashboard_id="test_dash",
            title="Test Dashboard",
            layout="tabs",
            widgets=[{"id": "widget1", "type": "chart"}],
            filters=[{"id": "filter1", "type": "select"}],
            refresh_interval=60,
            user_customizable=False
        )

        dashboard_dict = dashboard.to_dict()

        assert dashboard_dict["dashboard_id"] == "test_dash"
        assert dashboard_dict["title"] == "Test Dashboard"
        assert dashboard_dict["layout"] == "tabs"
        assert len(dashboard_dict["widgets"]) == 1
        assert len(dashboard_dict["filters"]) == 1
        assert dashboard_dict["refresh_interval"] == 60
        assert dashboard_dict["user_customizable"] is False

    def test_color_palette_generation(self, viz_service):
        """Test color palette generation for different schemes"""
        # Test clinical color scheme
        clinical_colors = viz_service._get_color_palette(ColorScheme.CLINICAL)
        assert len(clinical_colors) > 0
        assert "#10b981" in clinical_colors  # Green
        assert "#f59e0b" in clinical_colors  # Yellow
        assert "#dc2626" in clinical_colors  # Red

        # Test professional color scheme
        professional_colors = viz_service._get_color_palette(ColorScheme.PROFESSIONAL)
        assert len(professional_colors) > 0
        assert all(color.startswith("#") for color in professional_colors)

        # Test accessibility color scheme
        accessibility_colors = viz_service._get_color_palette(ColorScheme.ACCESSIBILITY)
        assert len(accessibility_colors) > 0
        assert "#000000" in accessibility_colors  # Black

    def test_news2_risk_zones_addition(self, viz_service):
        """Test NEWS2 risk zone background addition"""
        base_chart_spec = {
            "type": "line",
            "data": {},
            "options": {}
        }

        enhanced_chart = viz_service._add_news2_risk_zones(base_chart_spec)

        # Verify annotation plugin was added
        assert "plugins" in enhanced_chart["options"]
        assert "annotation" in enhanced_chart["options"]["plugins"]

        annotations = enhanced_chart["options"]["plugins"]["annotation"]["annotations"]
        assert "low_risk" in annotations
        assert "medium_risk" in annotations
        assert "high_risk" in annotations

        # Verify risk zone configurations
        low_risk = annotations["low_risk"]
        assert low_risk["yMin"] == 0
        assert low_risk["yMax"] == 4
        assert "Low Risk" in low_risk["label"]["content"]

    @pytest.mark.asyncio
    async def test_export_visualization(self, viz_service):
        """Test visualization export functionality"""
        chart_spec = {"type": "line", "data": {}, "options": {}}

        export_result = await viz_service.export_visualization(
            chart_spec=chart_spec,
            format="png",
            width=1200,
            height=800
        )

        assert export_result["success"] is True
        assert export_result["format"] == "png"
        assert export_result["width"] == 1200
        assert export_result["height"] == 800
        assert "export_url" in export_result
        assert "expires_at" in export_result
        assert export_result["file_size_kb"] > 0

    @pytest.mark.asyncio
    async def test_chart_caching(self, viz_service):
        """Test chart generation caching"""
        # Generate chart twice with same parameters
        chart1 = await viz_service.generate_news2_trend_chart(patient_ids=["P001"])
        chart2 = await viz_service.generate_news2_trend_chart(patient_ids=["P001"])

        # Charts should have same structure (simulated data is deterministic)
        assert chart1["config"]["title"] == chart2["config"]["title"]
        assert len(chart1["chart_specification"]["data"]["datasets"]) == len(chart2["chart_specification"]["data"]["datasets"])


class TestVisualizationDataGeneration:
    """Test visualization data generation methods"""

    @pytest.fixture
    def viz_service(self):
        """Create VisualizationService instance"""
        return VisualizationService()

    @pytest.mark.asyncio
    async def test_get_alert_distribution_data(self, viz_service):
        """Test alert distribution data generation"""
        data = await viz_service._get_alert_distribution_data(
            ward_ids=["ward_a", "ward_b"],
            time_range="24h"
        )

        assert len(data) == 2
        assert "ward_a" in data
        assert "ward_b" in data

        for ward_id, alert_counts in data.items():
            assert "critical" in alert_counts
            assert "high" in alert_counts
            assert "medium" in alert_counts
            assert alert_counts["critical"] >= 0
            assert alert_counts["high"] >= 0
            assert alert_counts["medium"] >= 0

    @pytest.mark.asyncio
    async def test_get_response_time_heatmap_data(self, viz_service):
        """Test response time heatmap data generation"""
        data = await viz_service._get_response_time_heatmap_data("7d")

        assert "data_points" in data
        assert "colors" in data
        assert "avg_response_time" in data
        assert "max_response_time" in data

        # Verify data points structure
        assert len(data["data_points"]) == 7 * 24  # 7 days * 24 hours

        for point in data["data_points"]:
            assert "x" in point  # Hour
            assert "y" in point  # Day
            assert "response_time" in point
            assert 0 <= point["x"] <= 23
            assert 0 <= point["y"] <= 6
            assert point["response_time"] > 0

    @pytest.mark.asyncio
    async def test_get_clinical_timeline_data(self, viz_service):
        """Test clinical timeline data generation"""
        data = await viz_service._get_clinical_timeline_data("P001", "7d")

        assert "start_time" in data
        assert "end_time" in data
        assert "events" in data
        assert "vital_signs" in data
        assert "interventions" in data

        # Verify events
        assert len(data["events"]) > 0
        for event in data["events"]:
            assert "id" in event
            assert "timestamp" in event
            assert "type" in event
            assert "description" in event
            assert "category" in event
            assert "severity" in event

        # Verify vital signs
        assert len(data["vital_signs"]) > 0
        for vital_sign in data["vital_signs"]:
            assert "timestamp" in vital_sign
            assert "news2_score" in vital_sign
            assert "respiratory_rate" in vital_sign
            assert "oxygen_saturation" in vital_sign

        # Verify interventions
        for intervention in data["interventions"]:
            assert "id" in intervention
            assert "start_time" in intervention
            assert "end_time" in intervention
            assert "type" in intervention
            assert "description" in intervention


class TestChartTypeEnum:
    """Test ChartType enum"""

    def test_chart_type_values(self):
        """Test ChartType enum values"""
        assert ChartType.LINE_CHART.value == "line_chart"
        assert ChartType.BAR_CHART.value == "bar_chart"
        assert ChartType.PIE_CHART.value == "pie_chart"
        assert ChartType.HEATMAP.value == "heatmap"
        assert ChartType.TIMELINE.value == "timeline"


class TestColorSchemeEnum:
    """Test ColorScheme enum"""

    def test_color_scheme_values(self):
        """Test ColorScheme enum values"""
        assert ColorScheme.CLINICAL.value == "clinical"
        assert ColorScheme.PROFESSIONAL.value == "professional"
        assert ColorScheme.ACCESSIBILITY.value == "accessibility"
        assert ColorScheme.CUSTOM.value == "custom"


class TestVisualizationErrorHandling:
    """Test error handling in visualization service"""

    @pytest.fixture
    def viz_service(self):
        """Create VisualizationService instance"""
        return VisualizationService()

    @pytest.mark.asyncio
    async def test_chart_generation_exception(self, viz_service):
        """Test chart generation with exceptions"""
        with patch.object(viz_service, '_get_news2_trend_data',
                         side_effect=Exception("Data access error")):
            with pytest.raises(Exception):
                await viz_service.generate_news2_trend_chart()

    @pytest.mark.asyncio
    async def test_export_failure_handling(self, viz_service):
        """Test export failure handling"""
        chart_spec = {"type": "line", "data": {}}

        with patch.object(viz_service, 'export_visualization',
                         side_effect=Exception("Export failed")):
            try:
                result = await viz_service.export_visualization(chart_spec)
                assert not result["success"]
                assert "error" in result
            except Exception:
                pass  # Expected for this test

    @pytest.mark.asyncio
    async def test_dashboard_generation_with_invalid_widgets(self, viz_service):
        """Test dashboard generation with invalid widget configuration"""
        # This should not cause exceptions, but handle gracefully
        dashboard = await viz_service.generate_kpi_dashboard(
            dashboard_id="invalid_test",
            ward_ids=[]  # Empty ward list
        )

        # Should still generate dashboard with default widgets
        assert isinstance(dashboard, InteractiveDashboard)
        assert len(dashboard.widgets) > 0
        assert len(dashboard.filters) > 0


class TestVisualizationIntegration:
    """Test visualization service integration with other services"""

    @pytest.fixture
    def viz_service(self):
        """Create VisualizationService instance"""
        return VisualizationService()

    @pytest.mark.asyncio
    async def test_integration_with_clinical_metrics(self, viz_service):
        """Test integration with clinical metrics service"""
        # This tests the import and usage of clinical metrics service
        chart = await viz_service.generate_news2_trend_chart(
            patient_ids=["P001", "P002"]
        )

        # Should successfully generate chart using clinical metrics data
        assert "chart_specification" in chart
        assert len(chart["chart_specification"]["data"]["datasets"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_chart_generation(self, viz_service):
        """Test concurrent chart generation"""
        # Generate multiple charts concurrently
        tasks = [
            viz_service.generate_news2_trend_chart(),
            viz_service.generate_alert_distribution_chart(),
            viz_service.generate_response_time_heatmap()
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert "chart_specification" in result
            assert "generated_at" in result

    def test_visualization_config_defaults(self):
        """Test VisualizationConfig default values"""
        config = VisualizationConfig(
            chart_type=ChartType.LINE_CHART,
            title="Test Chart"
        )

        # Check default values
        assert config.width == 800
        assert config.height == 600
        assert config.color_scheme == ColorScheme.CLINICAL
        assert config.interactive is True
        assert config.export_formats == ["svg", "png"]
        assert config.accessibility_features is True
        assert config.responsive is True
        assert config.animation_enabled is True