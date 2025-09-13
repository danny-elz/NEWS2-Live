"""
Dashboard and reporting system for NEWS2 Alert Suppression monitoring.
Provides real-time dashboards and comprehensive reporting capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import csv
from io import StringIO

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Container for dashboard metrics."""
    total_suppressions: int = 0
    active_suppressions: int = 0
    avg_processing_time: float = 0.0
    suppression_effectiveness: float = 0.0
    error_rate: float = 0.0
    system_health_score: float = 100.0
    last_updated: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_suppressions": self.total_suppressions,
            "active_suppressions": self.active_suppressions,
            "avg_processing_time": self.avg_processing_time,
            "suppression_effectiveness": self.suppression_effectiveness,
            "error_rate": self.error_rate,
            "system_health_score": self.system_health_score,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


class SuppressionDashboard:
    """Real-time dashboard for alert suppression monitoring."""
    
    def __init__(self, monitoring_system, redis_client=None):
        self.monitoring_system = monitoring_system
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.metrics_cache = {}
        self.cache_ttl = 60  # Cache for 1 minute
    
    async def get_real_time_metrics(self) -> DashboardMetrics:
        """Get real-time metrics for dashboard."""
        cache_key = "dashboard_metrics"
        now = datetime.utcnow()
        
        # Check cache first
        if cache_key in self.metrics_cache:
            cached_data, cache_time = self.metrics_cache[cache_key]
            if (now - cache_time).total_seconds() < self.cache_ttl:
                return cached_data
        
        metrics = DashboardMetrics(last_updated=now)
        
        try:
            if self.redis_client:
                # Get active suppressions count
                metrics.active_suppressions = await self.redis_client.scard("news2:active_suppressions")
                
                # Get total suppressions from today
                today_key = f"news2:suppression_decisions:{now.strftime('%Y-%m-%d')}"
                metrics.total_suppressions = await self.redis_client.get(today_key) or 0
                metrics.total_suppressions = int(metrics.total_suppressions)
                
                # Calculate average processing time from recent metrics
                processing_times = []
                for i in range(10):  # Check last 10 minutes
                    time_key = f"news2:processing_times:{(now - timedelta(minutes=i)).strftime('%Y-%m-%d:%H:%M')}"
                    times = await self.redis_client.lrange(time_key, 0, -1)
                    processing_times.extend([float(t) for t in times])
                
                if processing_times:
                    metrics.avg_processing_time = sum(processing_times) / len(processing_times)
                
                # Calculate error rate
                error_count = await self.redis_client.get("news2:errors:today") or 0
                total_requests = await self.redis_client.get("news2:requests:today") or 1
                metrics.error_rate = (int(error_count) / int(total_requests)) * 100
                
                # Get system health score
                health_status = await self.monitoring_system.get_health_status()
                healthy_components = sum(1 for status in health_status["components"].values() if status == "healthy")
                total_components = len(health_status["components"])
                metrics.system_health_score = (healthy_components / max(total_components, 1)) * 100
                
                # Calculate suppression effectiveness
                prevented_alerts = await self.redis_client.get("news2:prevented_alerts:today") or 0
                total_potential_alerts = await self.redis_client.get("news2:total_alerts:today") or 1
                metrics.suppression_effectiveness = (int(prevented_alerts) / int(total_potential_alerts)) * 100
        
        except Exception as e:
            self.logger.error(f"Error calculating dashboard metrics: {e}")
        
        # Cache the metrics
        self.metrics_cache[cache_key] = (metrics, now)
        
        return metrics
    
    async def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""
        metrics = await self.get_real_time_metrics()
        
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NEWS2 Alert Suppression Dashboard</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 20px; background-color: #f5f5f5;
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
                    text-align: center;
                }
                .metrics-grid { 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; margin-bottom: 20px;
                }
                .metric-card { 
                    background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #667eea;
                }
                .metric-value { 
                    font-size: 2.5em; font-weight: bold; margin: 10px 0;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                }
                .metric-label { 
                    color: #666; font-size: 0.9em; text-transform: uppercase; 
                    letter-spacing: 1px; margin-bottom: 5px;
                }
                .health-indicator {
                    display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                    margin-right: 8px;
                }
                .healthy { background-color: #4CAF50; }
                .warning { background-color: #FF9800; }
                .critical { background-color: #F44336; }
                .status-section {
                    background: white; padding: 20px; border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px;
                }
                .refresh-info {
                    text-align: center; color: #666; margin-top: 20px; font-size: 0.9em;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .live-indicator {
                    animation: pulse 2s infinite;
                    color: #4CAF50; font-weight: bold;
                }
            </style>
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function() {
                    location.reload();
                }, 30000);
            </script>
        </head>
        <body>
            <div class="header">
                <h1>NEWS2 Alert Suppression Dashboard</h1>
                <p><span class="live-indicator">● LIVE</span> - Real-time monitoring and analytics</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Active Suppressions</div>
                    <div class="metric-value">{active_suppressions}</div>
                    <div>Currently active alert suppressions</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Total Today</div>
                    <div class="metric-value">{total_suppressions}</div>
                    <div>Suppression decisions made today</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Avg Processing Time</div>
                    <div class="metric-value">{avg_processing_time:.2f}s</div>
                    <div>Average decision processing time</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Effectiveness Rate</div>
                    <div class="metric-value">{suppression_effectiveness:.1f}%</div>
                    <div>Alerts successfully suppressed</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Error Rate</div>
                    <div class="metric-value">{error_rate:.2f}%</div>
                    <div>System error percentage</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">System Health</div>
                    <div class="metric-value">{system_health_score:.0f}%</div>
                    <div>
                        <span class="health-indicator {health_class}"></span>
                        Overall system status
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>System Status Overview</h3>
                <div id="system-status">
                    {system_status_html}
                </div>
            </div>
            
            <div class="refresh-info">
                <p>Last updated: {last_updated} | Auto-refresh: 30 seconds</p>
                <p>Dashboard automatically refreshes to show real-time data</p>
            </div>
        </body>
        </html>
        """
        
        # Determine health indicator class
        if metrics.system_health_score >= 90:
            health_class = "healthy"
        elif metrics.system_health_score >= 70:
            health_class = "warning"
        else:
            health_class = "critical"
        
        # Generate system status HTML
        system_status_html = await self._generate_system_status_html()
        
        return html_template.format(
            active_suppressions=metrics.active_suppressions,
            total_suppressions=metrics.total_suppressions,
            avg_processing_time=metrics.avg_processing_time,
            suppression_effectiveness=metrics.suppression_effectiveness,
            error_rate=metrics.error_rate,
            system_health_score=metrics.system_health_score,
            health_class=health_class,
            system_status_html=system_status_html,
            last_updated=metrics.last_updated.strftime("%Y-%m-%d %H:%M:%S UTC") if metrics.last_updated else "N/A"
        )
    
    async def _generate_system_status_html(self) -> str:
        """Generate system status HTML section."""
        try:
            health_status = await self.monitoring_system.get_health_status()
            
            status_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>"
            
            for component, status in health_status["components"].items():
                if status == "healthy":
                    indicator_class = "healthy"
                    status_text = "Healthy"
                elif status == "disabled":
                    indicator_class = "warning"
                    status_text = "Disabled"
                else:
                    indicator_class = "critical"
                    status_text = "Unhealthy"
                
                component_name = component.replace("_", " ").title()
                status_html += f"""
                    <div style="padding: 10px; background: #f9f9f9; border-radius: 5px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{component_name}</div>
                        <div><span class="health-indicator {indicator_class}"></span>{status_text}</div>
                    </div>
                """
            
            status_html += "</div>"
            
            return status_html
            
        except Exception as e:
            self.logger.error(f"Error generating system status HTML: {e}")
            return "<p>Error loading system status</p>"
    
    async def generate_performance_chart(self, hours: int = 24) -> Optional[str]:
        """Generate performance chart if plotly is available."""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Get historical data from Redis
            now = datetime.utcnow()
            timestamps = []
            processing_times = []
            active_counts = []
            
            for i in range(hours * 60):  # Every minute for specified hours
                time_point = now - timedelta(minutes=i)
                timestamp_str = time_point.strftime('%Y-%m-%d:%H:%M')
                
                if self.redis_client:
                    # Get processing times
                    time_key = f"news2:processing_times:{timestamp_str}"
                    times = await self.redis_client.lrange(time_key, 0, -1)
                    avg_time = sum([float(t) for t in times]) / len(times) if times else 0
                    
                    # Get active suppressions count
                    count_key = f"news2:active_count:{timestamp_str}"
                    count = await self.redis_client.get(count_key) or 0
                    
                    timestamps.append(time_point)
                    processing_times.append(avg_time)
                    active_counts.append(int(count))
            
            # Reverse to show chronological order
            timestamps.reverse()
            processing_times.reverse()
            active_counts.reverse()
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Processing Time (seconds)', 'Active Suppressions'),
                vertical_spacing=0.1
            )
            
            # Add processing time trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=processing_times,
                    mode='lines+markers',
                    name='Processing Time',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Add active suppressions trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=active_counts,
                    mode='lines+markers',
                    name='Active Suppressions',
                    line=dict(color='#764ba2', width=2),
                    marker=dict(size=4),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='NEWS2 Alert Suppression Performance - Last 24 Hours',
                showlegend=True,
                height=600,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Seconds", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error generating performance chart: {e}")
            return None


class SuppressionReportGenerator:
    """Generate comprehensive reports for alert suppression system."""
    
    def __init__(self, monitoring_system, redis_client=None):
        self.monitoring_system = monitoring_system
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def generate_daily_report(self, date: datetime = None) -> Dict[str, Any]:
        """Generate daily suppression report."""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        date_str = date.strftime('%Y-%m-%d')
        
        report = {
            "report_date": date_str,
            "report_generated": datetime.utcnow().isoformat(),
            "summary": {},
            "suppressions_by_hour": {},
            "patient_statistics": {},
            "performance_metrics": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        try:
            if self.redis_client:
                # Get daily suppression counts
                total_decisions = await self.redis_client.get(f"news2:suppression_decisions:{date_str}") or 0
                report["summary"]["total_decisions"] = int(total_decisions)
                
                # Get hourly breakdown
                for hour in range(24):
                    hour_key = f"news2:suppression_decisions:{date_str}:{hour:02d}"
                    hour_count = await self.redis_client.get(hour_key) or 0
                    report["suppressions_by_hour"][f"{hour:02d}:00"] = int(hour_count)
                
                # Get patient statistics
                patient_stats_key = f"news2:patient_stats:{date_str}"
                patient_data = await self.redis_client.hgetall(patient_stats_key)
                report["patient_statistics"] = {
                    "unique_patients": len(patient_data),
                    "avg_suppressions_per_patient": sum(int(v) for v in patient_data.values()) / max(len(patient_data), 1),
                    "top_patients": sorted(
                        [(k, int(v)) for k, v in patient_data.items()], 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                }
                
                # Performance metrics
                perf_key = f"news2:performance:{date_str}"
                perf_data = await self.redis_client.hgetall(perf_key)
                if perf_data:
                    report["performance_metrics"] = {
                        "avg_processing_time": float(perf_data.get("avg_processing_time", 0)),
                        "max_processing_time": float(perf_data.get("max_processing_time", 0)),
                        "min_processing_time": float(perf_data.get("min_processing_time", 0)),
                        "total_requests": int(perf_data.get("total_requests", 0))
                    }
                
                # Error analysis
                error_key = f"news2:errors:{date_str}"
                error_data = await self.redis_client.hgetall(error_key)
                if error_data:
                    report["error_analysis"] = {
                        "total_errors": sum(int(v) for v in error_data.values()),
                        "error_types": dict(error_data),
                        "error_rate": (sum(int(v) for v in error_data.values()) / 
                                     max(report["performance_metrics"].get("total_requests", 1), 1)) * 100
                    }
                
                # Generate recommendations
                report["recommendations"] = self._generate_recommendations(report)
        
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            report["error"] = str(e)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on report data."""
        recommendations = []
        
        try:
            # Performance recommendations
            if report.get("performance_metrics", {}).get("avg_processing_time", 0) > 2.0:
                recommendations.append(
                    "Consider optimizing suppression logic - average processing time exceeds 2 seconds"
                )
            
            # Error rate recommendations
            error_rate = report.get("error_analysis", {}).get("error_rate", 0)
            if error_rate > 5:
                recommendations.append(
                    f"Error rate is {error_rate:.1f}% - investigate and resolve recurring errors"
                )
            elif error_rate > 1:
                recommendations.append(
                    f"Error rate is {error_rate:.1f}% - monitor error patterns for optimization opportunities"
                )
            
            # Usage pattern recommendations
            patient_stats = report.get("patient_statistics", {})
            if patient_stats.get("unique_patients", 0) > 0:
                avg_suppressions = patient_stats.get("avg_suppressions_per_patient", 0)
                if avg_suppressions > 10:
                    recommendations.append(
                        "High suppression rate per patient - review suppression criteria for optimization"
                    )
            
            # Hourly pattern recommendations
            hourly_data = report.get("suppressions_by_hour", {})
            if hourly_data:
                peak_hour = max(hourly_data.items(), key=lambda x: x[1])
                if peak_hour[1] > sum(hourly_data.values()) * 0.3:  # More than 30% in one hour
                    recommendations.append(
                        f"Peak suppression activity at {peak_hour[0]} - consider load balancing"
                    )
            
            # General health recommendations
            if not recommendations:
                recommendations.append("System performance is optimal - continue monitoring")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def export_report_csv(self, report: Dict[str, Any]) -> str:
        """Export report data to CSV format."""
        if not PANDAS_AVAILABLE:
            # Fallback CSV generation without pandas
            output = StringIO()
            writer = csv.writer(output)
            
            # Write summary
            writer.writerow(["Metric", "Value"])
            for key, value in report.get("summary", {}).items():
                writer.writerow([key.replace("_", " ").title(), value])
            
            # Write hourly data
            writer.writerow([])  # Empty row
            writer.writerow(["Hour", "Suppressions"])
            for hour, count in report.get("suppressions_by_hour", {}).items():
                writer.writerow([hour, count])
            
            return output.getvalue()
        
        # Use pandas for better CSV generation
        try:
            dfs = []
            
            # Summary DataFrame
            summary_df = pd.DataFrame(list(report.get("summary", {}).items()), 
                                    columns=["Metric", "Value"])
            summary_df["Metric"] = summary_df["Metric"].str.replace("_", " ").str.title()
            
            # Hourly DataFrame
            hourly_df = pd.DataFrame(list(report.get("suppressions_by_hour", {}).items()), 
                                   columns=["Hour", "Suppressions"])
            
            # Patient statistics DataFrame
            patient_stats = report.get("patient_statistics", {})
            if patient_stats.get("top_patients"):
                patient_df = pd.DataFrame(patient_stats["top_patients"], 
                                        columns=["Patient_ID", "Suppression_Count"])
            else:
                patient_df = pd.DataFrame(columns=["Patient_ID", "Suppression_Count"])
            
            # Combine all DataFrames
            output = StringIO()
            
            # Write each section with headers
            output.write("DAILY SUMMARY\n")
            summary_df.to_csv(output, index=False)
            output.write("\n\nHOURLY BREAKDOWN\n")
            hourly_df.to_csv(output, index=False)
            output.write("\n\nTOP PATIENTS\n")
            patient_df.to_csv(output, index=False)
            
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            return f"Error generating CSV: {e}"
    
    async def generate_weekly_summary(self, week_start: datetime = None) -> Dict[str, Any]:
        """Generate weekly summary report."""
        if week_start is None:
            # Start from Monday of current week
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)
        
        weekly_summary = {
            "week_start": week_start.strftime('%Y-%m-%d'),
            "week_end": (week_start + timedelta(days=6)).strftime('%Y-%m-%d'),
            "generated_at": datetime.utcnow().isoformat(),
            "daily_totals": {},
            "weekly_trends": {},
            "performance_summary": {},
            "key_insights": []
        }
        
        try:
            daily_reports = []
            
            # Generate daily reports for the week
            for day_offset in range(7):
                day_date = week_start + timedelta(days=day_offset)
                daily_report = await self.generate_daily_report(day_date)
                daily_reports.append(daily_report)
                
                day_str = day_date.strftime('%Y-%m-%d')
                weekly_summary["daily_totals"][day_str] = daily_report.get("summary", {}).get("total_decisions", 0)
            
            # Calculate trends
            if len(daily_reports) >= 2:
                first_half = daily_reports[:3]  # First 3 days
                second_half = daily_reports[4:]  # Last 3 days
                
                first_half_avg = sum(r.get("summary", {}).get("total_decisions", 0) for r in first_half) / 3
                second_half_avg = sum(r.get("summary", {}).get("total_decisions", 0) for r in second_half) / 3
                
                if first_half_avg > 0:
                    trend_percentage = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                    weekly_summary["weekly_trends"]["suppression_trend"] = {
                        "percentage_change": trend_percentage,
                        "direction": "increasing" if trend_percentage > 0 else "decreasing",
                        "first_half_avg": first_half_avg,
                        "second_half_avg": second_half_avg
                    }
            
            # Performance summary
            all_processing_times = []
            total_errors = 0
            total_requests = 0
            
            for report in daily_reports:
                perf_metrics = report.get("performance_metrics", {})
                if perf_metrics.get("avg_processing_time"):
                    all_processing_times.append(perf_metrics["avg_processing_time"])
                
                error_analysis = report.get("error_analysis", {})
                total_errors += error_analysis.get("total_errors", 0)
                total_requests += perf_metrics.get("total_requests", 0)
            
            if all_processing_times:
                weekly_summary["performance_summary"] = {
                    "avg_processing_time": sum(all_processing_times) / len(all_processing_times),
                    "total_errors": total_errors,
                    "total_requests": total_requests,
                    "weekly_error_rate": (total_errors / max(total_requests, 1)) * 100
                }
            
            # Generate key insights
            weekly_summary["key_insights"] = self._generate_weekly_insights(weekly_summary, daily_reports)
        
        except Exception as e:
            self.logger.error(f"Error generating weekly summary: {e}")
            weekly_summary["error"] = str(e)
        
        return weekly_summary
    
    def _generate_weekly_insights(self, weekly_summary: Dict[str, Any], daily_reports: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from weekly data."""
        insights = []
        
        try:
            # Trend insights
            trend_data = weekly_summary.get("weekly_trends", {}).get("suppression_trend", {})
            if trend_data:
                trend_pct = trend_data.get("percentage_change", 0)
                if abs(trend_pct) > 20:
                    direction = "increased" if trend_pct > 0 else "decreased"
                    insights.append(f"Suppression activity {direction} by {abs(trend_pct):.1f}% this week")
            
            # Performance insights
            perf_summary = weekly_summary.get("performance_summary", {})
            if perf_summary.get("avg_processing_time", 0) > 1.5:
                insights.append("Processing times are above optimal range - consider performance optimization")
            
            # Error rate insights
            error_rate = perf_summary.get("weekly_error_rate", 0)
            if error_rate > 3:
                insights.append(f"Weekly error rate of {error_rate:.1f}% requires attention")
            elif error_rate < 1:
                insights.append("Excellent system reliability with low error rates")
            
            # Usage pattern insights
            daily_totals = list(weekly_summary.get("daily_totals", {}).values())
            if daily_totals:
                max_day = max(daily_totals)
                min_day = min(daily_totals)
                if max_day > min_day * 2:
                    insights.append("Significant variation in daily suppression volumes - review capacity planning")
            
            # Overall health insight
            if not insights:
                insights.append("System operating within normal parameters across all metrics")
        
        except Exception as e:
            self.logger.error(f"Error generating weekly insights: {e}")
        
        return insights


# Integration helper function
def setup_dashboard_and_reporting(monitoring_system, redis_client=None, config: Dict[str, Any] = None) -> tuple:
    """Setup dashboard and reporting components."""
    config = config or {}
    
    dashboard = SuppressionDashboard(monitoring_system, redis_client)
    report_generator = SuppressionReportGenerator(monitoring_system, redis_client)
    
    return dashboard, report_generator