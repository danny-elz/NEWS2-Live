"""
Timeline Service for Story 3.3
Generates and manages 24-hour clinical timeline data
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Predefined time ranges for timeline views"""
    FOUR_HOURS = 4
    EIGHT_HOURS = 8
    TWELVE_HOURS = 12
    TWENTY_FOUR_HOURS = 24
    FORTY_EIGHT_HOURS = 48
    SEVEN_DAYS = 168


class TrendDirection(Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class TimelineService:
    """Service for generating clinical timeline visualizations"""

    def __init__(self):
        self.threshold_markers = {
            "low_medium": 3,    # Between low and medium risk
            "medium_high": 5,    # Between medium and high risk
            "high_critical": 7   # High risk threshold
        }
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes

    async def generate_timeline(self,
                               patient_id: str,
                               timeline_data: List[Dict[str, Any]],
                               time_range: TimeRange = TimeRange.TWENTY_FOUR_HOURS,
                               parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate timeline visualization data

        Args:
            patient_id: Patient identifier
            timeline_data: Raw timeline data points
            time_range: Time range to display
            parameters: Vital sign parameters to include

        Returns:
            Timeline visualization data
        """
        try:
            # Filter data by time range
            filtered_data = self._filter_by_time_range(timeline_data, time_range)

            # Process data points
            processed_data = self._process_timeline_data(filtered_data, parameters)

            # Calculate trend indicators
            trend_info = self._calculate_trend(processed_data)

            # Identify critical events
            critical_events = self._identify_critical_events(processed_data)

            # Generate chart configuration
            chart_config = self._generate_chart_config(time_range, parameters)

            # Build timeline response
            timeline = {
                "patient_id": patient_id,
                "time_range": {
                    "hours": time_range.value,
                    "label": self._get_time_range_label(time_range),
                    "start": (datetime.now() - timedelta(hours=time_range.value)).isoformat(),
                    "end": datetime.now().isoformat()
                },
                "data_points": processed_data,
                "trend": trend_info,
                "critical_events": critical_events,
                "chart_config": chart_config,
                "threshold_markers": self.threshold_markers,
                "statistics": self._calculate_statistics(processed_data),
                "data_quality": self._assess_data_quality(processed_data)
            }

            return timeline

        except Exception as e:
            logger.error(f"Error generating timeline for patient {patient_id}: {e}")
            return {}

    def _filter_by_time_range(self,
                             timeline_data: List[Dict[str, Any]],
                             time_range: TimeRange) -> List[Dict[str, Any]]:
        """Filter timeline data by time range"""
        if not timeline_data:
            return []

        cutoff_time = datetime.now() - timedelta(hours=time_range.value)
        filtered = []

        for point in timeline_data:
            if point.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(point["timestamp"])
                    if timestamp >= cutoff_time:
                        filtered.append(point)
                except (ValueError, TypeError):
                    continue

        return filtered

    def _process_timeline_data(self,
                              data: List[Dict[str, Any]],
                              parameters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process timeline data for visualization"""
        processed = []

        # Default parameters if not specified
        if not parameters:
            parameters = ["news2_score", "heart_rate", "respiratory_rate", "spo2", "temperature"]

        for point in data:
            processed_point = {
                "timestamp": point.get("timestamp"),
                "news2_score": point.get("news2_score"),
                "values": {},
                "is_manual": point.get("is_manual", True),
                "confidence": point.get("confidence", 1.0)
            }

            # Extract requested parameters
            vital_signs = point.get("vital_signs", {})
            for param in parameters:
                if param == "news2_score":
                    processed_point["values"][param] = point.get("news2_score")
                elif param in vital_signs:
                    processed_point["values"][param] = vital_signs[param]

            # Add risk level indicator
            news2 = processed_point.get("news2_score")
            if news2 is not None:
                processed_point["risk_level"] = self._get_risk_level(news2)

            processed.append(processed_point)

        return processed

    def _calculate_trend(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trend information from timeline data"""
        if len(data) < 2:
            return {
                "direction": TrendDirection.INSUFFICIENT_DATA.value,
                "confidence": 0,
                "description": "Insufficient data for trend analysis"
            }

        # Get NEWS2 scores
        scores = [p.get("news2_score") for p in data if p.get("news2_score") is not None]

        if len(scores) < 2:
            return {
                "direction": TrendDirection.INSUFFICIENT_DATA.value,
                "confidence": 0,
                "description": "Insufficient NEWS2 data"
            }

        # Calculate trend using simple linear regression
        recent_scores = scores[-min(6, len(scores)):]  # Last 6 data points
        older_scores = scores[:min(6, len(scores))]    # First 6 data points

        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)

        # Determine trend direction
        difference = recent_avg - older_avg

        if difference > 2:
            direction = TrendDirection.WORSENING
            description = f"NEWS2 trending up (avg increase: {difference:.1f})"
        elif difference < -2:
            direction = TrendDirection.IMPROVING
            description = f"NEWS2 trending down (avg decrease: {abs(difference):.1f})"
        elif recent_avg >= 7:
            direction = TrendDirection.CRITICAL
            description = f"Sustained critical NEWS2 levels (avg: {recent_avg:.1f})"
        else:
            direction = TrendDirection.STABLE
            description = f"NEWS2 stable (current avg: {recent_avg:.1f})"

        # Calculate confidence based on data consistency
        if len(scores) >= 10:
            confidence = min(1.0, len(scores) / 20.0)  # Max confidence at 20+ points
        else:
            confidence = len(scores) / 10.0

        return {
            "direction": direction.value,
            "confidence": round(confidence, 2),
            "description": description,
            "recent_average": round(recent_avg, 1),
            "older_average": round(older_avg, 1),
            "change": round(difference, 1)
        }

    def _identify_critical_events(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify critical events in timeline"""
        critical_events = []

        for i, point in enumerate(data):
            news2 = point.get("news2_score")
            if news2 is None:
                continue

            # Check for critical NEWS2 scores
            if news2 >= 7:
                critical_events.append({
                    "timestamp": point.get("timestamp"),
                    "type": "critical_news2",
                    "description": f"Critical NEWS2 score: {news2}",
                    "severity": "critical"
                })

            # Check for rapid deterioration
            if i > 0:
                prev_news2 = data[i-1].get("news2_score")
                if prev_news2 is not None and news2 - prev_news2 >= 3:
                    critical_events.append({
                        "timestamp": point.get("timestamp"),
                        "type": "rapid_deterioration",
                        "description": f"NEWS2 increased by {news2 - prev_news2} points",
                        "severity": "high"
                    })

        return critical_events

    def _generate_chart_config(self,
                              time_range: TimeRange,
                              parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate chart configuration for visualization"""
        # Time axis configuration
        time_config = {
            "format": self._get_time_format(time_range),
            "ticks": self._get_tick_interval(time_range),
            "grid": True
        }

        # Y-axis configurations for different parameters
        y_axes = []

        if not parameters:
            parameters = ["news2_score"]

        for param in parameters:
            y_axes.append(self._get_y_axis_config(param))

        return {
            "type": "line",
            "time_axis": time_config,
            "y_axes": y_axes,
            "threshold_lines": [
                {"value": 3, "color": "#FFA500", "label": "Medium Risk"},
                {"value": 5, "color": "#FF6347", "label": "High Risk"},
                {"value": 7, "color": "#DC143C", "label": "Critical"}
            ],
            "legend": {
                "position": "top",
                "show": True
            },
            "tooltip": {
                "enabled": True,
                "format": "detailed"
            },
            "zoom": {
                "enabled": True,
                "mode": "x"
            }
        }

    def _calculate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from timeline data"""
        if not data:
            return {}

        news2_scores = [p.get("news2_score") for p in data if p.get("news2_score") is not None]

        if not news2_scores:
            return {}

        return {
            "total_readings": len(data),
            "news2_readings": len(news2_scores),
            "average_news2": round(statistics.mean(news2_scores), 1),
            "median_news2": round(statistics.median(news2_scores), 1),
            "max_news2": max(news2_scores),
            "min_news2": min(news2_scores),
            "std_dev": round(statistics.stdev(news2_scores), 2) if len(news2_scores) > 1 else 0,
            "manual_readings": sum(1 for p in data if p.get("is_manual", True)),
            "automatic_readings": sum(1 for p in data if not p.get("is_manual", True))
        }

    def _assess_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality and completeness of timeline data"""
        if not data:
            return {"quality": "no_data", "score": 0}

        total_points = len(data)
        complete_points = sum(1 for p in data if p.get("news2_score") is not None)

        # Calculate time gaps
        gaps = []
        for i in range(1, len(data)):
            try:
                t1 = datetime.fromisoformat(data[i-1]["timestamp"])
                t2 = datetime.fromisoformat(data[i]["timestamp"])
                gap_hours = (t2 - t1).total_seconds() / 3600
                gaps.append(gap_hours)
            except:
                continue

        avg_gap = statistics.mean(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0

        # Calculate quality score
        completeness = complete_points / total_points if total_points > 0 else 0

        if avg_gap <= 1 and completeness >= 0.9:
            quality = "excellent"
            score = 5
        elif avg_gap <= 2 and completeness >= 0.7:
            quality = "good"
            score = 4
        elif avg_gap <= 4 and completeness >= 0.5:
            quality = "fair"
            score = 3
        else:
            quality = "poor"
            score = 2

        return {
            "quality": quality,
            "score": score,
            "completeness": round(completeness * 100, 1),
            "average_gap_hours": round(avg_gap, 1),
            "max_gap_hours": round(max_gap, 1),
            "total_points": total_points,
            "complete_points": complete_points
        }

    def _get_risk_level(self, news2_score: int) -> str:
        """Get risk level from NEWS2 score"""
        if news2_score <= 2:
            return "low"
        elif news2_score <= 4:
            return "medium"
        elif news2_score <= 6:
            return "high"
        else:
            return "critical"

    def _get_time_range_label(self, time_range: TimeRange) -> str:
        """Get human-readable label for time range"""
        labels = {
            TimeRange.FOUR_HOURS: "4 hours",
            TimeRange.EIGHT_HOURS: "8 hours",
            TimeRange.TWELVE_HOURS: "12 hours",
            TimeRange.TWENTY_FOUR_HOURS: "24 hours",
            TimeRange.FORTY_EIGHT_HOURS: "48 hours",
            TimeRange.SEVEN_DAYS: "7 days"
        }
        return labels.get(time_range, f"{time_range.value} hours")

    def _get_time_format(self, time_range: TimeRange) -> str:
        """Get appropriate time format for range"""
        if time_range.value <= 12:
            return "%H:%M"
        elif time_range.value <= 48:
            return "%d %H:%M"
        else:
            return "%m/%d %H:%M"

    def _get_tick_interval(self, time_range: TimeRange) -> int:
        """Get tick interval in minutes for time axis"""
        intervals = {
            TimeRange.FOUR_HOURS: 30,
            TimeRange.EIGHT_HOURS: 60,
            TimeRange.TWELVE_HOURS: 120,
            TimeRange.TWENTY_FOUR_HOURS: 240,
            TimeRange.FORTY_EIGHT_HOURS: 360,
            TimeRange.SEVEN_DAYS: 1440
        }
        return intervals.get(time_range, 60)

    def _get_y_axis_config(self, parameter: str) -> Dict[str, Any]:
        """Get Y-axis configuration for parameter"""
        configs = {
            "news2_score": {
                "id": "news2",
                "label": "NEWS2 Score",
                "min": 0,
                "max": 20,
                "ticks": 5,
                "position": "left"
            },
            "heart_rate": {
                "id": "hr",
                "label": "Heart Rate (bpm)",
                "min": 40,
                "max": 180,
                "ticks": 10,
                "position": "right"
            },
            "respiratory_rate": {
                "id": "rr",
                "label": "Respiratory Rate (bpm)",
                "min": 8,
                "max": 40,
                "ticks": 5,
                "position": "right"
            },
            "temperature": {
                "id": "temp",
                "label": "Temperature (°C)",
                "min": 35,
                "max": 41,
                "ticks": 1,
                "position": "right"
            },
            "spo2": {
                "id": "spo2",
                "label": "SpO2 (%)",
                "min": 80,
                "max": 100,
                "ticks": 5,
                "position": "right"
            },
            "systolic_bp": {
                "id": "bp",
                "label": "Systolic BP (mmHg)",
                "min": 70,
                "max": 200,
                "ticks": 10,
                "position": "right"
            }
        }
        return configs.get(parameter, {
            "id": parameter,
            "label": parameter,
            "position": "right"
        })