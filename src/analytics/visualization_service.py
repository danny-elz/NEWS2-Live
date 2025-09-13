"""
Visualization Service for Story 3.5
Provides interactive charts, graphs, and dashboard visualizations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts and visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    AREA_CHART = "area_chart"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    GAUGE_CHART = "gauge_chart"
    TIMELINE = "timeline"


class ColorScheme(Enum):
    """Color schemes for visualizations"""
    CLINICAL = "clinical"  # Red, yellow, green based on clinical risk
    PROFESSIONAL = "professional"  # Blues and grays
    ACCESSIBILITY = "accessibility"  # High contrast colors
    CUSTOM = "custom"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    chart_type: ChartType
    title: str
    width: int = 800
    height: int = 600
    color_scheme: ColorScheme = ColorScheme.CLINICAL
    interactive: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["svg", "png"])
    accessibility_features: bool = True
    responsive: bool = True
    animation_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "color_scheme": self.color_scheme.value,
            "interactive": self.interactive,
            "export_formats": self.export_formats,
            "accessibility_features": self.accessibility_features,
            "responsive": self.responsive,
            "animation_enabled": self.animation_enabled
        }


@dataclass
class InteractiveDashboard:
    """Interactive dashboard configuration"""
    dashboard_id: str
    title: str
    layout: str  # "grid", "tabs", "sidebar"
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    refresh_interval: int = 30  # seconds
    user_customizable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "title": self.title,
            "layout": self.layout,
            "widgets": self.widgets,
            "filters": self.filters,
            "refresh_interval": self.refresh_interval,
            "user_customizable": self.user_customizable
        }


class VisualizationService:
    """Service for generating interactive visualizations and dashboards"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._chart_cache = {}
        self._dashboard_cache = {}

    async def generate_news2_trend_chart(self, patient_ids: List[str] = None,
                                       time_range: str = "24h",
                                       config: VisualizationConfig = None) -> Dict[str, Any]:
        """Generate NEWS2 trend line chart"""
        try:
            if not config:
                config = VisualizationConfig(
                    chart_type=ChartType.LINE_CHART,
                    title="NEWS2 Score Trends",
                    height=400
                )

            # Get data (simulated)
            data = await self._get_news2_trend_data(patient_ids, time_range)

            # Generate chart specification (Chart.js format)
            chart_spec = {
                "type": "line",
                "data": {
                    "datasets": []
                },
                "options": {
                    "responsive": config.responsive,
                    "maintainAspectRatio": False,
                    "animation": {
                        "duration": 1000 if config.animation_enabled else 0
                    },
                    "interaction": {
                        "mode": "nearest",
                        "intersect": False
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.title
                        },
                        "legend": {
                            "display": True,
                            "position": "top"
                        },
                        "tooltip": {
                            "enabled": config.interactive,
                            "mode": "index",
                            "intersect": False
                        }
                    },
                    "scales": {
                        "x": {
                            "type": "time",
                            "display": True,
                            "title": {
                                "display": True,
                                "text": "Time"
                            }
                        },
                        "y": {
                            "display": True,
                            "title": {
                                "display": True,
                                "text": "NEWS2 Score"
                            },
                            "min": 0,
                            "max": 15,
                            "ticks": {
                                "stepSize": 1
                            }
                        }
                    }
                }
            }

            # Add patient data
            colors = self._get_color_palette(config.color_scheme)
            for i, (patient_id, scores) in enumerate(data.items()):
                dataset = {
                    "label": f"Patient {patient_id}",
                    "data": [
                        {"x": timestamp.isoformat(), "y": score}
                        for timestamp, score in scores
                    ],
                    "borderColor": colors[i % len(colors)],
                    "backgroundColor": colors[i % len(colors)] + "20",  # 20% opacity
                    "fill": False,
                    "tension": 0.1
                }
                chart_spec["data"]["datasets"].append(dataset)

            # Add risk level background areas
            chart_spec = self._add_news2_risk_zones(chart_spec)

            return {
                "chart_specification": chart_spec,
                "config": config.to_dict(),
                "data_summary": {
                    "patients_count": len(data),
                    "time_range": time_range,
                    "data_points_total": sum(len(scores) for scores in data.values())
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating NEWS2 trend chart: {e}")
            raise

    async def generate_alert_distribution_chart(self, ward_ids: List[str] = None,
                                              time_range: str = "24h",
                                              config: VisualizationConfig = None) -> Dict[str, Any]:
        """Generate alert distribution chart"""
        try:
            if not config:
                config = VisualizationConfig(
                    chart_type=ChartType.BAR_CHART,
                    title="Alert Distribution by Ward"
                )

            # Get alert data (simulated)
            alert_data = await self._get_alert_distribution_data(ward_ids, time_range)

            chart_spec = {
                "type": "bar",
                "data": {
                    "labels": list(alert_data.keys()),
                    "datasets": [
                        {
                            "label": "Critical Alerts",
                            "data": [ward_data["critical"] for ward_data in alert_data.values()],
                            "backgroundColor": "#dc2626",
                            "borderColor": "#b91c1c",
                            "borderWidth": 1
                        },
                        {
                            "label": "High Priority Alerts",
                            "data": [ward_data["high"] for ward_data in alert_data.values()],
                            "backgroundColor": "#f59e0b",
                            "borderColor": "#d97706",
                            "borderWidth": 1
                        },
                        {
                            "label": "Medium Priority Alerts",
                            "data": [ward_data["medium"] for ward_data in alert_data.values()],
                            "backgroundColor": "#10b981",
                            "borderColor": "#059669",
                            "borderWidth": 1
                        }
                    ]
                },
                "options": {
                    "responsive": config.responsive,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "stacked": True,
                            "title": {
                                "display": True,
                                "text": "Ward"
                            }
                        },
                        "y": {
                            "stacked": True,
                            "title": {
                                "display": True,
                                "text": "Number of Alerts"
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.title
                        },
                        "legend": {
                            "display": True
                        }
                    }
                }
            }

            return {
                "chart_specification": chart_spec,
                "config": config.to_dict(),
                "data_summary": {
                    "total_alerts": sum(
                        sum(ward_data.values()) for ward_data in alert_data.values()
                    ),
                    "wards_count": len(alert_data),
                    "time_range": time_range
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating alert distribution chart: {e}")
            raise

    async def generate_response_time_heatmap(self, time_range: str = "7d",
                                           config: VisualizationConfig = None) -> Dict[str, Any]:
        """Generate response time heatmap"""
        try:
            if not config:
                config = VisualizationConfig(
                    chart_type=ChartType.HEATMAP,
                    title="Response Time Heatmap",
                    height=300
                )

            # Get response time data (simulated)
            heatmap_data = await self._get_response_time_heatmap_data(time_range)

            # Generate heatmap specification (using Chart.js matrix plugin format)
            chart_spec = {
                "type": "scatter",
                "data": {
                    "datasets": [{
                        "label": "Response Times",
                        "data": heatmap_data["data_points"],
                        "backgroundColor": heatmap_data["colors"],
                        "pointRadius": 8,
                        "pointHoverRadius": 10
                    }]
                },
                "options": {
                    "responsive": config.responsive,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "type": "linear",
                            "title": {
                                "display": True,
                                "text": "Hour of Day"
                            },
                            "min": 0,
                            "max": 23
                        },
                        "y": {
                            "type": "linear",
                            "title": {
                                "display": True,
                                "text": "Day of Week"
                            },
                            "min": 0,
                            "max": 6,
                            "ticks": {
                                "callback": "function(value) { return ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][value]; }"
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.title
                        },
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return 'Response Time: ' + context.raw.response_time + ' min'; }"
                            }
                        }
                    }
                }
            }

            return {
                "chart_specification": chart_spec,
                "config": config.to_dict(),
                "data_summary": {
                    "data_points": len(heatmap_data["data_points"]),
                    "avg_response_time": heatmap_data["avg_response_time"],
                    "max_response_time": heatmap_data["max_response_time"],
                    "time_range": time_range
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating response time heatmap: {e}")
            raise

    async def generate_kpi_dashboard(self, dashboard_id: str = "executive_kpis",
                                   ward_ids: List[str] = None) -> InteractiveDashboard:
        """Generate executive KPI dashboard"""
        try:
            # Define dashboard widgets
            widgets = [
                {
                    "id": "news2_average",
                    "type": "gauge",
                    "title": "Average NEWS2 Score",
                    "position": {"row": 1, "col": 1, "width": 1, "height": 1},
                    "config": {
                        "min": 0,
                        "max": 15,
                        "value": 3.2,
                        "thresholds": [3, 5, 7],
                        "colors": ["#10b981", "#f59e0b", "#dc2626"]
                    }
                },
                {
                    "id": "total_alerts",
                    "type": "counter",
                    "title": "Total Alerts (24h)",
                    "position": {"row": 1, "col": 2, "width": 1, "height": 1},
                    "config": {
                        "value": 47,
                        "trend": "down",
                        "trend_percentage": -12.5,
                        "color": "#059669"
                    }
                },
                {
                    "id": "response_time",
                    "type": "gauge",
                    "title": "Avg Response Time",
                    "position": {"row": 1, "col": 3, "width": 1, "height": 1},
                    "config": {
                        "min": 0,
                        "max": 30,
                        "value": 4.8,
                        "unit": "minutes",
                        "target": 5.0,
                        "colors": ["#10b981", "#f59e0b", "#dc2626"]
                    }
                },
                {
                    "id": "bed_occupancy",
                    "type": "progress",
                    "title": "Bed Occupancy",
                    "position": {"row": 1, "col": 4, "width": 1, "height": 1},
                    "config": {
                        "value": 87,
                        "unit": "%",
                        "color": "#3b82f6"
                    }
                },
                {
                    "id": "news2_trends",
                    "type": "chart",
                    "title": "NEWS2 Trends (24h)",
                    "position": {"row": 2, "col": 1, "width": 2, "height": 2},
                    "config": {
                        "chart_type": "line",
                        "data_source": "news2_trends",
                        "refresh_interval": 300
                    }
                },
                {
                    "id": "alert_distribution",
                    "type": "chart",
                    "title": "Alert Distribution",
                    "position": {"row": 2, "col": 3, "width": 2, "height": 2},
                    "config": {
                        "chart_type": "doughnut",
                        "data_source": "alert_distribution",
                        "refresh_interval": 300
                    }
                },
                {
                    "id": "patient_outcomes",
                    "type": "table",
                    "title": "Recent Patient Outcomes",
                    "position": {"row": 3, "col": 1, "width": 4, "height": 1},
                    "config": {
                        "columns": ["Patient ID", "Ward", "Outcome", "Length of Stay", "NEWS2 Score"],
                        "max_rows": 10,
                        "sortable": True
                    }
                }
            ]

            # Define dashboard filters
            filters = [
                {
                    "id": "ward_filter",
                    "type": "multi-select",
                    "label": "Wards",
                    "options": ward_ids or ["ward_a", "ward_b", "ward_c"],
                    "default": ward_ids or ["ward_a", "ward_b", "ward_c"]
                },
                {
                    "id": "time_range",
                    "type": "select",
                    "label": "Time Range",
                    "options": ["1h", "4h", "24h", "7d", "30d"],
                    "default": "24h"
                },
                {
                    "id": "risk_level",
                    "type": "multi-select",
                    "label": "Risk Levels",
                    "options": ["low", "medium", "high"],
                    "default": ["low", "medium", "high"]
                }
            ]

            dashboard = InteractiveDashboard(
                dashboard_id=dashboard_id,
                title="Executive Clinical Dashboard",
                layout="grid",
                widgets=widgets,
                filters=filters,
                refresh_interval=30,
                user_customizable=True
            )

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating KPI dashboard: {e}")
            raise

    async def generate_clinical_timeline(self, patient_id: str,
                                       time_range: str = "7d",
                                       config: VisualizationConfig = None) -> Dict[str, Any]:
        """Generate clinical timeline visualization"""
        try:
            if not config:
                config = VisualizationConfig(
                    chart_type=ChartType.TIMELINE,
                    title=f"Clinical Timeline - Patient {patient_id}"
                )

            # Get timeline data (simulated)
            timeline_data = await self._get_clinical_timeline_data(patient_id, time_range)

            # Generate timeline specification
            chart_spec = {
                "type": "timeline",
                "data": {
                    "events": timeline_data["events"],
                    "vital_signs": timeline_data["vital_signs"],
                    "interventions": timeline_data["interventions"]
                },
                "options": {
                    "responsive": config.responsive,
                    "timeline": {
                        "startTime": timeline_data["start_time"],
                        "endTime": timeline_data["end_time"],
                        "tickInterval": "1h",
                        "showGrid": True
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.title
                        },
                        "legend": {
                            "display": True,
                            "position": "bottom"
                        }
                    }
                }
            }

            return {
                "chart_specification": chart_spec,
                "config": config.to_dict(),
                "data_summary": {
                    "patient_id": patient_id,
                    "events_count": len(timeline_data["events"]),
                    "vital_signs_count": len(timeline_data["vital_signs"]),
                    "interventions_count": len(timeline_data["interventions"]),
                    "time_range": time_range
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating clinical timeline: {e}")
            raise

    def _get_color_palette(self, scheme: ColorScheme) -> List[str]:
        """Get color palette for visualization scheme"""
        palettes = {
            ColorScheme.CLINICAL: [
                "#10b981",  # Green - low risk
                "#f59e0b",  # Yellow - medium risk
                "#dc2626",  # Red - high risk
                "#3b82f6",  # Blue
                "#8b5cf6",  # Purple
            ],
            ColorScheme.PROFESSIONAL: [
                "#1f2937",  # Dark gray
                "#374151",  # Gray
                "#6b7280",  # Light gray
                "#3b82f6",  # Blue
                "#1d4ed8",  # Dark blue
            ],
            ColorScheme.ACCESSIBILITY: [
                "#000000",  # Black
                "#0066cc",  # Blue
                "#ff6600",  # Orange
                "#009900",  # Green
                "#cc0066",  # Pink
            ]
        }
        return palettes.get(scheme, palettes[ColorScheme.CLINICAL])

    def _add_news2_risk_zones(self, chart_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Add NEWS2 risk zone background to chart"""
        # Add background annotation plugin config
        if "plugins" not in chart_spec["options"]:
            chart_spec["options"]["plugins"] = {}

        chart_spec["options"]["plugins"]["annotation"] = {
            "annotations": {
                "low_risk": {
                    "type": "box",
                    "yMin": 0,
                    "yMax": 4,
                    "backgroundColor": "rgba(16, 185, 129, 0.1)",
                    "borderColor": "rgba(16, 185, 129, 0.3)",
                    "borderWidth": 1,
                    "label": {
                        "content": "Low Risk",
                        "enabled": True,
                        "position": "start"
                    }
                },
                "medium_risk": {
                    "type": "box",
                    "yMin": 5,
                    "yMax": 6,
                    "backgroundColor": "rgba(245, 158, 11, 0.1)",
                    "borderColor": "rgba(245, 158, 11, 0.3)",
                    "borderWidth": 1,
                    "label": {
                        "content": "Medium Risk",
                        "enabled": True,
                        "position": "start"
                    }
                },
                "high_risk": {
                    "type": "box",
                    "yMin": 7,
                    "yMax": 15,
                    "backgroundColor": "rgba(220, 38, 38, 0.1)",
                    "borderColor": "rgba(220, 38, 38, 0.3)",
                    "borderWidth": 1,
                    "label": {
                        "content": "High Risk",
                        "enabled": True,
                        "position": "start"
                    }
                }
            }
        }

        return chart_spec

    async def _get_news2_trend_data(self, patient_ids: List[str] = None,
                                  time_range: str = "24h") -> Dict[str, List[Tuple[datetime, float]]]:
        """Get simulated NEWS2 trend data"""
        # Reuse logic from clinical_metrics service
        from src.analytics.clinical_metrics import ClinicalMetricsService

        clinical_service = ClinicalMetricsService()
        return await clinical_service._get_patient_news2_data(patient_ids, time_range)

    async def _get_alert_distribution_data(self, ward_ids: List[str] = None,
                                         time_range: str = "24h") -> Dict[str, Dict[str, int]]:
        """Get simulated alert distribution data"""
        ward_ids = ward_ids or ["Ward A", "Ward B", "Ward C"]

        alert_data = {}
        for ward_id in ward_ids:
            # Simulate alert counts
            base_alerts = hash(ward_id) % 10 + 5  # 5-14 base alerts

            alert_data[ward_id] = {
                "critical": max(0, base_alerts // 4),
                "high": max(0, base_alerts // 3),
                "medium": max(0, base_alerts // 2)
            }

        return alert_data

    async def _get_response_time_heatmap_data(self, time_range: str = "7d") -> Dict[str, Any]:
        """Get simulated response time heatmap data"""
        data_points = []
        response_times = []

        # Generate data for each hour of each day
        for day in range(7):  # Days of week
            for hour in range(24):  # Hours of day
                # Simulate response time (slower at night, faster during day shifts)
                base_time = 5.0  # 5 minutes base
                if 22 <= hour or hour <= 6:  # Night shift
                    response_time = base_time + 3 + (hash(f"{day}{hour}") % 5)
                elif 7 <= hour <= 19:  # Day shift
                    response_time = base_time + (hash(f"{day}{hour}") % 3)
                else:  # Evening shift
                    response_time = base_time + 1 + (hash(f"{day}{hour}") % 4)

                # Color based on response time
                if response_time <= 5:
                    color = "#10b981"  # Green
                elif response_time <= 8:
                    color = "#f59e0b"  # Yellow
                else:
                    color = "#dc2626"  # Red

                data_points.append({
                    "x": hour,
                    "y": day,
                    "response_time": round(response_time, 1)
                })
                response_times.append(response_time)

        return {
            "data_points": data_points,
            "colors": [point["response_time"] for point in data_points],
            "avg_response_time": round(sum(response_times) / len(response_times), 1),
            "max_response_time": max(response_times)
        }

    async def _get_clinical_timeline_data(self, patient_id: str,
                                        time_range: str = "7d") -> Dict[str, Any]:
        """Get simulated clinical timeline data"""
        now = datetime.now()
        hours_back = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}.get(time_range, 168)
        start_time = now - timedelta(hours=hours_back)

        # Generate events
        events = [
            {
                "id": "admission",
                "timestamp": start_time.isoformat(),
                "type": "admission",
                "description": "Patient admitted to ward",
                "category": "administrative",
                "severity": "info"
            },
            {
                "id": "alert_1",
                "timestamp": (start_time + timedelta(hours=2)).isoformat(),
                "type": "alert",
                "description": "NEWS2 score increased to 6",
                "category": "clinical",
                "severity": "warning"
            },
            {
                "id": "intervention_1",
                "timestamp": (start_time + timedelta(hours=2.5)).isoformat(),
                "type": "intervention",
                "description": "Oxygen therapy initiated",
                "category": "treatment",
                "severity": "info"
            }
        ]

        # Generate vital signs data points
        vital_signs = []
        current_time = start_time
        while current_time <= now:
            vital_signs.append({
                "timestamp": current_time.isoformat(),
                "news2_score": 3 + (hash(current_time.isoformat()) % 4),
                "respiratory_rate": 16 + (hash(current_time.isoformat()) % 6),
                "oxygen_saturation": 94 + (hash(current_time.isoformat()) % 6),
                "temperature": 36.5 + ((hash(current_time.isoformat()) % 10) / 10)
            })
            current_time += timedelta(hours=1)

        # Generate interventions
        interventions = [
            {
                "id": "oxygen_therapy",
                "start_time": (start_time + timedelta(hours=2.5)).isoformat(),
                "end_time": (start_time + timedelta(hours=8)).isoformat(),
                "type": "oxygen_therapy",
                "description": "2L/min via nasal cannula"
            }
        ]

        return {
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "events": events,
            "vital_signs": vital_signs,
            "interventions": interventions
        }

    async def export_visualization(self, chart_spec: Dict[str, Any],
                                 format: str = "png",
                                 width: int = 800,
                                 height: int = 600) -> Dict[str, Any]:
        """Export visualization to specified format"""
        try:
            # Simulate export process
            export_result = {
                "success": True,
                "format": format,
                "width": width,
                "height": height,
                "file_size_kb": 245,  # Simulated
                "export_url": f"/exports/chart_{datetime.now().timestamp()}.{format}",
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }

            return export_result

        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            return {
                "success": False,
                "error": str(e)
            }