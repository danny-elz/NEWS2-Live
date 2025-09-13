"""
Unit tests for Report Builder Service (Story 3.5)
Tests flexible report creation and scheduling capabilities
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analytics.report_builder import (
    ReportBuilder,
    ReportTemplate,
    ScheduledReport,
    ReportFormat,
    ReportFrequency
)


class TestReportBuilder:
    """Test suite for ReportBuilder"""

    @pytest.fixture
    def report_builder(self):
        """Create ReportBuilder instance"""
        return ReportBuilder()

    @pytest.fixture
    def sample_parameters(self):
        """Sample parameters for report generation"""
        return {
            "time_range": "7d",
            "ward_ids": ["ward_a", "ward_b"],
            "include_charts": True
        }

    @pytest.mark.asyncio
    async def test_create_executive_summary_report(self, report_builder, sample_parameters):
        """Test creating executive summary report"""
        result = await report_builder.create_report(
            template_id="executive_summary",
            parameters=sample_parameters,
            format=ReportFormat.PDF
        )

        assert result["success"] is True
        assert "report" in result
        assert "download_url" in result
        assert "preview_url" in result

        # Verify report metadata
        report = result["report"]
        assert report["template_id"] == "executive_summary"
        assert report["format"] == "pdf"
        assert report["status"] == "completed"
        assert "generated_at" in report
        assert "page_count" in report
        assert "file_size_kb" in report

    @pytest.mark.asyncio
    async def test_create_ward_performance_report(self, report_builder):
        """Test creating ward performance report"""
        parameters = {
            "ward_ids": ["ward_a"],
            "time_range": "30d",
            "include_comparisons": True
        }

        result = await report_builder.create_report(
            template_id="ward_performance",
            parameters=parameters,
            format=ReportFormat.EXCEL
        )

        assert result["success"] is True
        report = result["report"]
        assert report["template_name"] == "Ward Performance Report"
        assert report["format"] == "excel"

    @pytest.mark.asyncio
    async def test_create_clinical_outcomes_report(self, report_builder):
        """Test creating clinical outcomes report"""
        result = await report_builder.create_report(
            template_id="clinical_outcomes",
            format=ReportFormat.HTML
        )

        assert result["success"] is True
        report = result["report"]
        assert report["template_name"] == "Clinical Outcomes Analysis"
        assert report["format"] == "html"

    @pytest.mark.asyncio
    async def test_create_regulatory_compliance_report(self, report_builder):
        """Test creating regulatory compliance report"""
        result = await report_builder.create_report(
            template_id="regulatory_compliance",
            format=ReportFormat.PDF
        )

        assert result["success"] is True
        report = result["report"]
        assert report["template_name"] == "Regulatory Compliance Report"

    @pytest.mark.asyncio
    async def test_create_report_with_invalid_template(self, report_builder):
        """Test creating report with invalid template ID"""
        with pytest.raises(ValueError, match="Template 'invalid_template' not found"):
            await report_builder.create_report(
                template_id="invalid_template"
            )

    @pytest.mark.asyncio
    async def test_schedule_daily_report(self, report_builder):
        """Test scheduling a daily report"""
        scheduled_report = await report_builder.schedule_report(
            template_id="executive_summary",
            frequency=ReportFrequency.DAILY,
            recipients=["manager@hospital.com", "director@hospital.com"],
            format=ReportFormat.PDF,
            parameters={"time_range": "24h"}
        )

        assert isinstance(scheduled_report, ScheduledReport)
        assert scheduled_report.template_id == "executive_summary"
        assert scheduled_report.frequency == ReportFrequency.DAILY
        assert len(scheduled_report.recipients) == 2
        assert scheduled_report.format == ReportFormat.PDF
        assert scheduled_report.is_active is True
        assert scheduled_report.next_run is not None

    @pytest.mark.asyncio
    async def test_schedule_weekly_report(self, report_builder):
        """Test scheduling a weekly report"""
        scheduled_report = await report_builder.schedule_report(
            template_id="ward_performance",
            frequency=ReportFrequency.WEEKLY,
            recipients=["ward.manager@hospital.com"],
            format=ReportFormat.EXCEL
        )

        assert scheduled_report.frequency == ReportFrequency.WEEKLY
        assert scheduled_report.next_run.weekday() == 0  # Should be Monday

    @pytest.mark.asyncio
    async def test_schedule_monthly_report(self, report_builder):
        """Test scheduling a monthly report"""
        scheduled_report = await report_builder.schedule_report(
            template_id="clinical_outcomes",
            frequency=ReportFrequency.MONTHLY,
            recipients=["clinical.director@hospital.com"]
        )

        assert scheduled_report.frequency == ReportFrequency.MONTHLY
        assert scheduled_report.next_run.day == 1  # Should be first of month

    @pytest.mark.asyncio
    async def test_get_scheduled_reports(self, report_builder):
        """Test getting all scheduled reports"""
        # Schedule a few reports
        await report_builder.schedule_report(
            template_id="executive_summary",
            frequency=ReportFrequency.DAILY,
            recipients=["test@hospital.com"]
        )

        await report_builder.schedule_report(
            template_id="ward_performance",
            frequency=ReportFrequency.WEEKLY,
            recipients=["test@hospital.com"]
        )

        scheduled_reports = await report_builder.get_scheduled_reports()

        assert len(scheduled_reports) >= 2
        for report in scheduled_reports:
            assert isinstance(report, ScheduledReport)
            assert report.template_id in ["executive_summary", "ward_performance"]

    @pytest.mark.asyncio
    async def test_update_scheduled_report(self, report_builder):
        """Test updating scheduled report configuration"""
        # Create scheduled report
        scheduled_report = await report_builder.schedule_report(
            template_id="executive_summary",
            frequency=ReportFrequency.DAILY,
            recipients=["original@hospital.com"]
        )

        original_schedule_id = scheduled_report.schedule_id

        # Update the report
        updates = {
            "frequency": ReportFrequency.WEEKLY,
            "recipients": ["new@hospital.com", "additional@hospital.com"],
            "is_active": False
        }

        updated_report = await report_builder.update_scheduled_report(
            original_schedule_id,
            updates
        )

        assert updated_report.frequency == ReportFrequency.WEEKLY
        assert len(updated_report.recipients) == 2
        assert updated_report.is_active is False
        assert updated_report.next_run != scheduled_report.next_run  # Should be recalculated

    @pytest.mark.asyncio
    async def test_delete_scheduled_report(self, report_builder):
        """Test deleting scheduled report"""
        # Create scheduled report
        scheduled_report = await report_builder.schedule_report(
            template_id="executive_summary",
            frequency=ReportFrequency.DAILY,
            recipients=["test@hospital.com"]
        )

        schedule_id = scheduled_report.schedule_id

        # Delete the report
        success = await report_builder.delete_scheduled_report(schedule_id)
        assert success is True

        # Try to delete non-existent report
        success = await report_builder.delete_scheduled_report("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_report_templates(self, report_builder):
        """Test getting all available report templates"""
        templates = await report_builder.get_report_templates()

        assert len(templates) >= 4  # Should have default templates

        template_ids = [t.template_id for t in templates]
        assert "executive_summary" in template_ids
        assert "ward_performance" in template_ids
        assert "clinical_outcomes" in template_ids
        assert "regulatory_compliance" in template_ids

        # Verify template structure
        for template in templates:
            assert isinstance(template, ReportTemplate)
            assert template.name
            assert template.description
            assert template.category
            assert len(template.sections) > 0

    @pytest.mark.asyncio
    async def test_create_custom_template(self, report_builder):
        """Test creating custom report template"""
        template_data = {
            "template_id": "custom_test",
            "name": "Custom Test Report",
            "description": "Test custom template",
            "category": "custom",
            "sections": [
                {
                    "id": "test_section",
                    "title": "Test Section",
                    "type": "metrics_grid",
                    "content": ["test_metric"]
                }
            ],
            "parameters": {"test_param": "test_value"},
            "default_format": "pdf"
        }

        template = await report_builder.create_custom_template(template_data)

        assert isinstance(template, ReportTemplate)
        assert template.template_id == "custom_test"
        assert template.name == "Custom Test Report"
        assert template.category == "custom"
        assert len(template.sections) == 1

        # Verify template was added to available templates
        templates = await report_builder.get_report_templates()
        custom_template_ids = [t.template_id for t in templates]
        assert "custom_test" in custom_template_ids

    @pytest.mark.asyncio
    async def test_get_report_history(self, report_builder):
        """Test getting report generation history"""
        # Generate a few reports to create history
        await report_builder.create_report("executive_summary")
        await report_builder.create_report("ward_performance")

        history = await report_builder.get_report_history(limit=10)

        assert len(history) >= 2
        for report_metadata in history:
            assert "report_id" in report_metadata
            assert "template_id" in report_metadata
            assert "generated_at" in report_metadata
            assert "status" in report_metadata

        # Verify history is sorted by date (newest first)
        if len(history) > 1:
            first_date = datetime.fromisoformat(history[0]["generated_at"])
            second_date = datetime.fromisoformat(history[1]["generated_at"])
            assert first_date >= second_date

    def test_report_template_creation(self):
        """Test ReportTemplate creation and serialization"""
        template = ReportTemplate(
            template_id="test_template",
            name="Test Template",
            description="Test description",
            category="test",
            sections=[{
                "id": "section1",
                "title": "Test Section",
                "type": "chart"
            }],
            parameters={"param1": "value1"},
            default_format=ReportFormat.HTML,
            include_charts=True,
            include_tables=False
        )

        template_dict = template.to_dict()

        assert template_dict["template_id"] == "test_template"
        assert template_dict["name"] == "Test Template"
        assert template_dict["category"] == "test"
        assert template_dict["default_format"] == "html"
        assert template_dict["include_charts"] is True
        assert template_dict["include_tables"] is False

    def test_scheduled_report_creation(self):
        """Test ScheduledReport creation and serialization"""
        now = datetime.now()
        next_run = now + timedelta(days=1)

        scheduled_report = ScheduledReport(
            schedule_id="sched_001",
            template_id="executive_summary",
            name="Daily Executive Summary",
            frequency=ReportFrequency.DAILY,
            recipients=["manager@hospital.com"],
            format=ReportFormat.PDF,
            parameters={"time_range": "24h"},
            next_run=next_run,
            last_run=now,
            is_active=True
        )

        scheduled_dict = scheduled_report.to_dict()

        assert scheduled_dict["schedule_id"] == "sched_001"
        assert scheduled_dict["frequency"] == "daily"
        assert scheduled_dict["format"] == "pdf"
        assert len(scheduled_dict["recipients"]) == 1
        assert scheduled_dict["is_active"] is True
        assert "next_run" in scheduled_dict
        assert "last_run" in scheduled_dict

    def test_next_run_calculation(self, report_builder):
        """Test next run time calculation for different frequencies"""
        now = datetime.now()

        # Test hourly
        next_hourly = report_builder._calculate_next_run(ReportFrequency.HOURLY)
        assert (next_hourly - now).total_seconds() <= 3600  # Within an hour

        # Test daily
        next_daily = report_builder._calculate_next_run(ReportFrequency.DAILY)
        assert next_daily.hour == 6  # Should be at 6 AM

        # Test weekly
        next_weekly = report_builder._calculate_next_run(ReportFrequency.WEEKLY)
        assert next_weekly.weekday() == 0  # Should be Monday

        # Test monthly
        next_monthly = report_builder._calculate_next_run(ReportFrequency.MONTHLY)
        assert next_monthly.day == 1  # Should be first of month

    @pytest.mark.asyncio
    async def test_report_section_building(self, report_builder):
        """Test report section content building"""
        # Test metrics section
        metrics_section = {
            "id": "test_metrics",
            "title": "Test Metrics",
            "type": "metrics_grid",
            "content": ["avg_news2_score", "total_alerts"]
        }

        section_content = await report_builder._build_section_content(
            metrics_section, {"time_range": "24h"}
        )

        assert section_content["id"] == "test_metrics"
        assert section_content["type"] == "metrics_grid"
        assert section_content["data"] is not None
        assert len(section_content["data"]) > 0

        # Test chart section
        chart_section = {
            "id": "test_chart",
            "title": "Test Chart",
            "type": "chart",
            "content": {"chart_type": "line", "data_source": "news2_trends"}
        }

        chart_content = await report_builder._build_section_content(
            chart_section, {"time_range": "24h"}
        )

        assert chart_content["type"] == "chart"
        assert chart_content["visualization"] is not None

    @pytest.mark.asyncio
    async def test_pdf_report_formatting(self, report_builder):
        """Test PDF report formatting"""
        content = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "sections": []
        }

        template = ReportTemplate(
            template_id="test",
            name="Test",
            description="Test",
            category="test",
            include_charts=True,
            include_tables=True
        )

        formatted = await report_builder._format_as_pdf(content, template)

        assert formatted["format"] == "pdf"
        assert formatted["page_count"] > 0
        assert formatted["file_size_kb"] > 0
        assert formatted["contains_charts"] is True
        assert formatted["contains_tables"] is True

    @pytest.mark.asyncio
    async def test_excel_report_formatting(self, report_builder):
        """Test Excel report formatting"""
        content = {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "sections": []
        }

        template = ReportTemplate(
            template_id="test",
            name="Test",
            description="Test",
            category="test",
            include_charts=True,
            include_tables=True
        )

        formatted = await report_builder._format_as_excel(content, template)

        assert formatted["format"] == "excel"
        assert "worksheets" in formatted
        assert len(formatted["worksheets"]) > 0
        assert "Summary" in formatted["worksheets"]

    @pytest.mark.asyncio
    async def test_json_report_formatting(self, report_builder):
        """Test JSON report formatting"""
        content = {
            "title": "Test Report",
            "data": {"test": "value"}
        }

        template = ReportTemplate(
            template_id="test",
            name="Test",
            description="Test",
            category="test"
        )

        formatted = await report_builder._format_report(content, ReportFormat.JSON, template)

        assert "json_content" in formatted
        assert "file_size_kb" in formatted


class TestReportFormatEnum:
    """Test ReportFormat enum"""

    def test_report_format_values(self):
        """Test ReportFormat enum values"""
        assert ReportFormat.PDF.value == "pdf"
        assert ReportFormat.EXCEL.value == "excel"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.POWERPOINT.value == "powerpoint"


class TestReportFrequencyEnum:
    """Test ReportFrequency enum"""

    def test_report_frequency_values(self):
        """Test ReportFrequency enum values"""
        assert ReportFrequency.HOURLY.value == "hourly"
        assert ReportFrequency.DAILY.value == "daily"
        assert ReportFrequency.WEEKLY.value == "weekly"
        assert ReportFrequency.MONTHLY.value == "monthly"
        assert ReportFrequency.QUARTERLY.value == "quarterly"
        assert ReportFrequency.ON_DEMAND.value == "on_demand"


class TestReportBuilderErrorHandling:
    """Test error handling in report builder"""

    @pytest.fixture
    def report_builder(self):
        """Create ReportBuilder instance"""
        return ReportBuilder()

    @pytest.mark.asyncio
    async def test_report_generation_with_missing_data(self, report_builder):
        """Test report generation when data is missing"""
        # Mock empty data responses
        with patch.object(report_builder, '_get_metrics_data', return_value=[]):
            result = await report_builder.create_report("executive_summary")

            # Should still generate report successfully
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scheduled_report_invalid_template(self, report_builder):
        """Test scheduling report with invalid template"""
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            await report_builder.schedule_report(
                template_id="invalid",
                frequency=ReportFrequency.DAILY,
                recipients=["test@hospital.com"]
            )

    @pytest.mark.asyncio
    async def test_update_nonexistent_scheduled_report(self, report_builder):
        """Test updating non-existent scheduled report"""
        with pytest.raises(ValueError, match="Scheduled report 'nonexistent' not found"):
            await report_builder.update_scheduled_report(
                "nonexistent",
                {"frequency": ReportFrequency.WEEKLY}
            )

    @pytest.mark.asyncio
    async def test_chart_integration_failure(self, report_builder):
        """Test report generation when chart integration fails"""
        with patch('src.analytics.visualization_service.VisualizationService') as mock_viz:
            mock_viz.return_value.generate_news2_trend_chart.side_effect = Exception("Chart error")

            # Should handle chart generation errors gracefully
            result = await report_builder.create_report("executive_summary")
            assert result["success"] is True


class TestReportBuilderIntegration:
    """Test report builder integration with other services"""

    @pytest.fixture
    def report_builder(self):
        """Create ReportBuilder instance"""
        return ReportBuilder()

    @pytest.mark.asyncio
    async def test_integration_with_visualization_service(self, report_builder):
        """Test integration with visualization service"""
        # Create report that includes charts
        result = await report_builder.create_report(
            template_id="executive_summary",
            parameters={"include_charts": True}
        )

        assert result["success"] is True
        # Report should be generated successfully with charts included

    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, report_builder):
        """Test concurrent report generation"""
        tasks = [
            report_builder.create_report("executive_summary"),
            report_builder.create_report("ward_performance"),
            report_builder.create_report("clinical_outcomes")
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert "report" in result