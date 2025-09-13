"""
Report Builder Service for Story 3.5
Provides flexible report creation and scheduling capabilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    POWERPOINT = "powerpoint"


class ReportFrequency(Enum):
    """Report scheduling frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"


@dataclass
class ReportTemplate:
    """Report template configuration"""
    template_id: str
    name: str
    description: str
    category: str  # "clinical", "operational", "quality", "regulatory"
    sections: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_format: ReportFormat = ReportFormat.PDF
    auto_refresh_data: bool = True
    include_charts: bool = True
    include_tables: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "sections": self.sections,
            "parameters": self.parameters,
            "default_format": self.default_format.value,
            "auto_refresh_data": self.auto_refresh_data,
            "include_charts": self.include_charts,
            "include_tables": self.include_tables
        }


@dataclass
class ScheduledReport:
    """Scheduled report configuration"""
    schedule_id: str
    template_id: str
    name: str
    frequency: ReportFrequency
    recipients: List[str]
    format: ReportFormat
    parameters: Dict[str, Any] = field(default_factory=dict)
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "template_id": self.template_id,
            "name": self.name,
            "frequency": self.frequency.value,
            "recipients": self.recipients,
            "format": self.format.value,
            "parameters": self.parameters,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "is_active": self.is_active
        }


class ReportBuilder:
    """Service for flexible report creation and management"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._templates = {}
        self._scheduled_reports = {}
        self._report_cache = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize default report templates"""
        # Executive Summary Template
        self._templates["executive_summary"] = ReportTemplate(
            template_id="executive_summary",
            name="Executive Clinical Summary",
            description="High-level overview of clinical performance metrics",
            category="clinical",
            sections=[
                {
                    "id": "kpis",
                    "title": "Key Performance Indicators",
                    "type": "metrics_grid",
                    "content": ["avg_news2_score", "total_alerts", "response_time", "bed_occupancy"]
                },
                {
                    "id": "trends",
                    "title": "24-Hour Trends",
                    "type": "chart",
                    "content": {"chart_type": "line", "data_source": "news2_trends"}
                },
                {
                    "id": "alerts_summary",
                    "title": "Alert Summary",
                    "type": "chart",
                    "content": {"chart_type": "bar", "data_source": "alert_distribution"}
                },
                {
                    "id": "recommendations",
                    "title": "Clinical Recommendations",
                    "type": "text",
                    "content": "auto_generated"
                }
            ],
            parameters={
                "time_range": "24h",
                "include_all_wards": True,
                "risk_threshold": 5
            }
        )

        # Ward Performance Template
        self._templates["ward_performance"] = ReportTemplate(
            template_id="ward_performance",
            name="Ward Performance Report",
            description="Detailed analysis of ward-level performance metrics",
            category="operational",
            sections=[
                {
                    "id": "ward_overview",
                    "title": "Ward Overview",
                    "type": "table",
                    "content": {"data_source": "ward_metrics"}
                },
                {
                    "id": "patient_outcomes",
                    "title": "Patient Outcomes",
                    "type": "chart",
                    "content": {"chart_type": "pie", "data_source": "outcome_distribution"}
                },
                {
                    "id": "staffing_metrics",
                    "title": "Staffing Metrics",
                    "type": "chart",
                    "content": {"chart_type": "bar", "data_source": "staffing_data"}
                },
                {
                    "id": "quality_indicators",
                    "title": "Quality Indicators",
                    "type": "scorecard",
                    "content": {"data_source": "quality_metrics"}
                }
            ],
            parameters={
                "ward_ids": [],
                "time_range": "7d",
                "include_comparisons": True
            }
        )

        # Clinical Outcomes Template
        self._templates["clinical_outcomes"] = ReportTemplate(
            template_id="clinical_outcomes",
            name="Clinical Outcomes Analysis",
            description="Comprehensive analysis of patient outcomes and interventions",
            category="clinical",
            sections=[
                {
                    "id": "outcome_summary",
                    "title": "Outcome Summary",
                    "type": "metrics_grid",
                    "content": ["mortality_rate", "los_average", "readmission_rate", "complication_rate"]
                },
                {
                    "id": "news2_analysis",
                    "title": "NEWS2 Score Analysis",
                    "type": "chart",
                    "content": {"chart_type": "histogram", "data_source": "news2_distribution"}
                },
                {
                    "id": "intervention_effectiveness",
                    "title": "Intervention Effectiveness",
                    "type": "table",
                    "content": {"data_source": "intervention_outcomes"}
                },
                {
                    "id": "risk_stratification",
                    "title": "Risk Stratification",
                    "type": "chart",
                    "content": {"chart_type": "scatter", "data_source": "risk_outcomes"}
                }
            ]
        )

        # Regulatory Compliance Template
        self._templates["regulatory_compliance"] = ReportTemplate(
            template_id="regulatory_compliance",
            name="Regulatory Compliance Report",
            description="Report for regulatory compliance and quality assurance",
            category="regulatory",
            sections=[
                {
                    "id": "compliance_metrics",
                    "title": "Compliance Metrics",
                    "type": "scorecard",
                    "content": {"data_source": "compliance_data"}
                },
                {
                    "id": "incident_summary",
                    "title": "Incident Summary",
                    "type": "table",
                    "content": {"data_source": "incidents"}
                },
                {
                    "id": "quality_measures",
                    "title": "Quality Measures",
                    "type": "chart",
                    "content": {"chart_type": "gauge", "data_source": "quality_scores"}
                },
                {
                    "id": "corrective_actions",
                    "title": "Corrective Actions",
                    "type": "text",
                    "content": "manual_input"
                }
            ]
        )

    async def create_report(self, template_id: str,
                          parameters: Dict[str, Any] = None,
                          format: ReportFormat = ReportFormat.PDF) -> Dict[str, Any]:
        """Create a report from template"""
        try:
            if template_id not in self._templates:
                raise ValueError(f"Template '{template_id}' not found")

            template = self._templates[template_id]
            report_params = {**template.parameters, **(parameters or {})}

            # Generate report ID
            report_id = str(uuid.uuid4())

            # Build report content
            report_content = await self._build_report_content(template, report_params)

            # Format report based on output format
            formatted_report = await self._format_report(report_content, format, template)

            report_metadata = {
                "report_id": report_id,
                "template_id": template_id,
                "template_name": template.name,
                "format": format.value,
                "parameters": report_params,
                "generated_at": datetime.now().isoformat(),
                "page_count": formatted_report.get("page_count", 1),
                "file_size_kb": formatted_report.get("file_size_kb", 0),
                "status": "completed"
            }

            # Cache the report
            self._report_cache[report_id] = {
                "metadata": report_metadata,
                "content": formatted_report,
                "expires_at": datetime.now() + timedelta(hours=24)
            }

            return {
                "success": True,
                "report": report_metadata,
                "download_url": f"/reports/{report_id}/download",
                "preview_url": f"/reports/{report_id}/preview"
            }

        except Exception as e:
            self.logger.error(f"Error creating report: {e}")
            raise

    async def _build_report_content(self, template: ReportTemplate,
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build report content from template and parameters"""
        report_content = {
            "title": template.name,
            "description": template.description,
            "generated_at": datetime.now().isoformat(),
            "parameters": parameters,
            "sections": []
        }

        for section_config in template.sections:
            section_content = await self._build_section_content(section_config, parameters)
            report_content["sections"].append(section_content)

        return report_content

    async def _build_section_content(self, section_config: Dict[str, Any],
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build content for individual report section"""
        section_id = section_config["id"]
        section_type = section_config["type"]

        section_content = {
            "id": section_id,
            "title": section_config["title"],
            "type": section_type,
            "data": None,
            "visualization": None
        }

        if section_type == "metrics_grid":
            section_content["data"] = await self._get_metrics_data(section_config["content"], parameters)
        elif section_type == "chart":
            section_content["visualization"] = await self._get_chart_data(section_config["content"], parameters)
        elif section_type == "table":
            section_content["data"] = await self._get_table_data(section_config["content"], parameters)
        elif section_type == "scorecard":
            section_content["data"] = await self._get_scorecard_data(section_config["content"], parameters)
        elif section_type == "text":
            section_content["data"] = await self._get_text_content(section_config["content"], parameters)

        return section_content

    async def _get_metrics_data(self, metric_names: List[str],
                              parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get metrics data for KPI section"""
        metrics = []

        for metric_name in metric_names:
            if metric_name == "avg_news2_score":
                metrics.append({
                    "name": "Average NEWS2 Score",
                    "value": 3.2,
                    "unit": "points",
                    "trend": "stable",
                    "trend_percentage": 0.5,
                    "status": "normal"
                })
            elif metric_name == "total_alerts":
                metrics.append({
                    "name": "Total Alerts",
                    "value": 47,
                    "unit": "alerts",
                    "trend": "down",
                    "trend_percentage": -12.5,
                    "status": "improving"
                })
            elif metric_name == "response_time":
                metrics.append({
                    "name": "Avg Response Time",
                    "value": 4.8,
                    "unit": "minutes",
                    "trend": "up",
                    "trend_percentage": 2.1,
                    "status": "warning"
                })
            elif metric_name == "bed_occupancy":
                metrics.append({
                    "name": "Bed Occupancy",
                    "value": 87,
                    "unit": "%",
                    "trend": "stable",
                    "trend_percentage": 0.8,
                    "status": "normal"
                })

        return metrics

    async def _get_chart_data(self, chart_config: Dict[str, Any],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data for visualization sections"""
        try:
            # Import and use visualization service
            from src.analytics.visualization_service import VisualizationService, VisualizationConfig, ChartType

            viz_service = VisualizationService()

            data_source = chart_config.get("data_source", "")
            chart_type = chart_config.get("chart_type", "line")

            if data_source == "news2_trends":
                config = VisualizationConfig(
                    chart_type=ChartType.LINE_CHART,
                    title="NEWS2 Score Trends"
                )
                return await viz_service.generate_news2_trend_chart(config=config)

            elif data_source == "alert_distribution":
                config = VisualizationConfig(
                    chart_type=ChartType.BAR_CHART,
                    title="Alert Distribution"
                )
                return await viz_service.generate_alert_distribution_chart(config=config)

            else:
                # Default empty chart
                return {
                    "chart_specification": {"type": chart_type, "data": {}, "options": {}},
                    "data_summary": {"message": "No data available"}
                }
        except Exception as e:
            # Return fallback chart data if visualization fails
            return {
                "chart_specification": {"type": "line", "data": {}, "options": {}},
                "data_summary": {"message": f"Chart generation failed: {str(e)}"}
            }

    async def _get_table_data(self, table_config: Dict[str, Any],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get table data for tabular sections"""
        data_source = table_config.get("data_source", "")

        if data_source == "ward_metrics":
            return {
                "headers": ["Ward", "Patients", "Avg NEWS2", "Alerts", "Response Time"],
                "rows": [
                    ["Ward A", 24, 3.1, 15, "4.2 min"],
                    ["Ward B", 18, 3.8, 12, "3.9 min"],
                    ["Ward C", 22, 2.9, 20, "5.1 min"]
                ]
            }
        elif data_source == "intervention_outcomes":
            return {
                "headers": ["Intervention", "Count", "Avg Improvement", "Success Rate"],
                "rows": [
                    ["Oxygen Therapy", 23, "2.1 points", "85%"],
                    ["Medication Adjustment", 18, "1.8 points", "78%"],
                    ["Position Change", 31, "0.9 points", "92%"]
                ]
            }
        else:
            return {"headers": [], "rows": []}

    async def _get_scorecard_data(self, scorecard_config: Dict[str, Any],
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get scorecard data for quality metrics"""
        return {
            "overall_score": 87,
            "max_score": 100,
            "categories": [
                {"name": "Documentation Completeness", "score": 92, "target": 90, "status": "good"},
                {"name": "Response Time Compliance", "score": 78, "target": 85, "status": "needs_improvement"},
                {"name": "Clinical Outcomes", "score": 91, "target": 85, "status": "excellent"},
                {"name": "Patient Safety", "score": 88, "target": 95, "status": "needs_improvement"}
            ]
        }

    async def _get_text_content(self, content_config: str,
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get text content for narrative sections"""
        if content_config == "auto_generated":
            return {
                "content": "Based on the current clinical data analysis, the following recommendations are suggested: "
                         "1) Increase monitoring frequency for patients with NEWS2 scores above 5, "
                         "2) Review response time protocols in Ward C, "
                         "3) Consider staff reinforcement during peak alert periods."
            }
        else:
            return {"content": "Manual input required for this section."}

    async def _format_report(self, content: Dict[str, Any],
                           format: ReportFormat,
                           template: ReportTemplate) -> Dict[str, Any]:
        """Format report content for specified output format"""
        if format == ReportFormat.PDF:
            return await self._format_as_pdf(content, template)
        elif format == ReportFormat.EXCEL:
            return await self._format_as_excel(content, template)
        elif format == ReportFormat.HTML:
            return await self._format_as_html(content, template)
        elif format == ReportFormat.JSON:
            return {"json_content": json.dumps(content, indent=2), "file_size_kb": 15}
        else:
            return content

    async def _format_as_pdf(self, content: Dict[str, Any],
                           template: ReportTemplate) -> Dict[str, Any]:
        """Format report as PDF"""
        # Simulate PDF generation
        return {
            "format": "pdf",
            "page_count": 5,
            "file_size_kb": 450,
            "contains_charts": template.include_charts,
            "contains_tables": template.include_tables,
            "pdf_metadata": {
                "title": content["title"],
                "author": "Clinical Analytics System",
                "created": content["generated_at"]
            }
        }

    async def _format_as_excel(self, content: Dict[str, Any],
                             template: ReportTemplate) -> Dict[str, Any]:
        """Format report as Excel workbook"""
        # Simulate Excel generation
        worksheets = ["Summary"]
        if template.include_tables:
            worksheets.extend(["Data Tables", "Metrics"])
        if template.include_charts:
            worksheets.append("Charts")

        return {
            "format": "excel",
            "worksheets": worksheets,
            "file_size_kb": 280,
            "excel_metadata": {
                "title": content["title"],
                "created": content["generated_at"]
            }
        }

    async def _format_as_html(self, content: Dict[str, Any],
                            template: ReportTemplate) -> Dict[str, Any]:
        """Format report as HTML"""
        # Simulate HTML generation
        return {
            "format": "html",
            "file_size_kb": 125,
            "interactive": True,
            "css_included": True,
            "javascript_included": template.include_charts
        }

    async def schedule_report(self, template_id: str,
                            frequency: ReportFrequency,
                            recipients: List[str],
                            format: ReportFormat = ReportFormat.PDF,
                            parameters: Dict[str, Any] = None) -> ScheduledReport:
        """Schedule a report for automatic generation"""
        try:
            if template_id not in self._templates:
                raise ValueError(f"Template '{template_id}' not found")

            schedule_id = str(uuid.uuid4())
            next_run = self._calculate_next_run(frequency)

            scheduled_report = ScheduledReport(
                schedule_id=schedule_id,
                template_id=template_id,
                name=f"Scheduled {self._templates[template_id].name}",
                frequency=frequency,
                recipients=recipients,
                format=format,
                parameters=parameters or {},
                next_run=next_run,
                is_active=True
            )

            self._scheduled_reports[schedule_id] = scheduled_report

            return scheduled_report

        except Exception as e:
            self.logger.error(f"Error scheduling report: {e}")
            raise

    def _calculate_next_run(self, frequency: ReportFrequency) -> datetime:
        """Calculate next run time for scheduled report"""
        now = datetime.now()

        if frequency == ReportFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ReportFrequency.DAILY:
            return now.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            days_ahead = 0 - now.weekday()  # Monday = 0, Sunday = 6
            if days_ahead <= 0:
                days_ahead += 7
            return now.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif frequency == ReportFrequency.MONTHLY:
            next_month = now.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=1, hour=6, minute=0, second=0, microsecond=0)
        else:
            return now + timedelta(days=1)

    async def get_scheduled_reports(self) -> List[ScheduledReport]:
        """Get all scheduled reports"""
        return list(self._scheduled_reports.values())

    async def update_scheduled_report(self, schedule_id: str,
                                    updates: Dict[str, Any]) -> ScheduledReport:
        """Update scheduled report configuration"""
        if schedule_id not in self._scheduled_reports:
            raise ValueError(f"Scheduled report '{schedule_id}' not found")

        scheduled_report = self._scheduled_reports[schedule_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(scheduled_report, key):
                setattr(scheduled_report, key, value)

        # Recalculate next run if frequency changed
        if "frequency" in updates:
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.frequency)

        return scheduled_report

    async def delete_scheduled_report(self, schedule_id: str) -> bool:
        """Delete scheduled report"""
        if schedule_id in self._scheduled_reports:
            del self._scheduled_reports[schedule_id]
            return True
        return False

    async def get_report_templates(self) -> List[ReportTemplate]:
        """Get all available report templates"""
        return list(self._templates.values())

    async def create_custom_template(self, template_data: Dict[str, Any]) -> ReportTemplate:
        """Create custom report template"""
        template_id = template_data.get("template_id", str(uuid.uuid4()))

        template = ReportTemplate(
            template_id=template_id,
            name=template_data["name"],
            description=template_data.get("description", ""),
            category=template_data.get("category", "custom"),
            sections=template_data.get("sections", []),
            parameters=template_data.get("parameters", {}),
            default_format=ReportFormat(template_data.get("default_format", "pdf"))
        )

        self._templates[template_id] = template
        return template

    async def get_report_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get report generation history"""
        history = []
        for report_data in list(self._report_cache.values())[-limit:]:
            history.append(report_data["metadata"])

        return sorted(history, key=lambda x: x["generated_at"], reverse=True)