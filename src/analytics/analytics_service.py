"""
Core Analytics Service for Story 3.5
Provides data aggregation, statistical analysis, and performance metrics
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Time range options for analytics queries"""
    LAST_HOUR = "last_hour"
    LAST_4_HOURS = "last_4_hours"
    LAST_8_HOURS = "last_8_hours"
    LAST_24_HOURS = "last_24_hours"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


class AggregationType(Enum):
    """Data aggregation methods"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTILE_95 = "percentile_95"
    STANDARD_DEVIATION = "std_dev"


class MetricType(Enum):
    """Types of clinical and operational metrics"""
    NEWS2_SCORES = "news2_scores"
    ALERT_COUNTS = "alert_counts"
    RESPONSE_TIMES = "response_times"
    PATIENT_OUTCOMES = "patient_outcomes"
    STAFF_WORKLOAD = "staff_workload"
    RESOURCE_UTILIZATION = "resource_utilization"
    COMPLIANCE_RATES = "compliance_rates"


@dataclass
class AnalyticsQuery:
    """Analytics query configuration"""
    metric_type: MetricType
    time_range: TimeRange
    aggregation: AggregationType
    filters: Dict[str, Any] = field(default_factory=dict)
    grouping: List[str] = field(default_factory=list)
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    ward_ids: List[str] = field(default_factory=list)
    patient_cohort: Optional[str] = None

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get actual start and end dates for the query"""
        now = datetime.now()

        if self.time_range == TimeRange.CUSTOM:
            return self.custom_start_date or now, self.custom_end_date or now
        elif self.time_range == TimeRange.LAST_HOUR:
            return now - timedelta(hours=1), now
        elif self.time_range == TimeRange.LAST_4_HOURS:
            return now - timedelta(hours=4), now
        elif self.time_range == TimeRange.LAST_8_HOURS:
            return now - timedelta(hours=8), now
        elif self.time_range == TimeRange.LAST_24_HOURS:
            return now - timedelta(days=1), now
        elif self.time_range == TimeRange.LAST_WEEK:
            return now - timedelta(weeks=1), now
        elif self.time_range == TimeRange.LAST_MONTH:
            return now - timedelta(days=30), now
        elif self.time_range == TimeRange.LAST_QUARTER:
            return now - timedelta(days=90), now
        elif self.time_range == TimeRange.LAST_YEAR:
            return now - timedelta(days=365), now
        else:
            return now - timedelta(days=1), now


@dataclass
class AnalyticsResult:
    """Result of analytics query execution"""
    query: AnalyticsQuery
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    summary_stats: Dict[str, float]
    execution_time_ms: int
    data_quality_score: float
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": {
                "metric_type": self.query.metric_type.value,
                "time_range": self.query.time_range.value,
                "aggregation": self.query.aggregation.value,
                "filters": self.query.filters,
                "grouping": self.query.grouping
            },
            "data": self.data,
            "metadata": self.metadata,
            "summary_stats": self.summary_stats,
            "execution_time_ms": self.execution_time_ms,
            "data_quality_score": self.data_quality_score,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class KPIMetric:
    """Key Performance Indicator metric"""
    name: str
    value: Union[int, float, str]
    unit: str
    trend: str  # "up", "down", "stable"
    trend_percentage: float
    target_value: Optional[Union[int, float]] = None
    status: str = "normal"  # "normal", "warning", "critical"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "trend": self.trend,
            "trend_percentage": self.trend_percentage,
            "target_value": self.target_value,
            "status": self.status,
            "description": self.description
        }


class AnalyticsService:
    """Core service for clinical analytics and reporting"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._query_cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def execute_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query and return results"""
        start_time = datetime.now()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if (datetime.now() - cached_result["timestamp"]).seconds < self._cache_ttl:
                    return cached_result["result"]

            # Execute query based on metric type
            if query.metric_type == MetricType.NEWS2_SCORES:
                data = await self._query_news2_data(query)
            elif query.metric_type == MetricType.ALERT_COUNTS:
                data = await self._query_alert_data(query)
            elif query.metric_type == MetricType.RESPONSE_TIMES:
                data = await self._query_response_time_data(query)
            elif query.metric_type == MetricType.PATIENT_OUTCOMES:
                data = await self._query_outcome_data(query)
            elif query.metric_type == MetricType.STAFF_WORKLOAD:
                data = await self._query_workload_data(query)
            elif query.metric_type == MetricType.RESOURCE_UTILIZATION:
                data = await self._query_resource_data(query)
            else:
                data = []

            # Apply aggregation
            aggregated_data = self._apply_aggregation(data, query.aggregation, query.grouping)

            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(aggregated_data, query.aggregation)

            # Assess data quality
            data_quality_score = self._assess_data_quality(data, query)

            # Create result
            execution_time = max(1, int((datetime.now() - start_time).total_seconds() * 1000))

            result = AnalyticsResult(
                query=query,
                data=aggregated_data,
                metadata={
                    "total_records": len(data),
                    "aggregated_records": len(aggregated_data),
                    "date_range": {
                        "start": query.get_date_range()[0].isoformat(),
                        "end": query.get_date_range()[1].isoformat()
                    },
                    "filters_applied": query.filters,
                    "grouping": query.grouping
                },
                summary_stats=summary_stats,
                execution_time_ms=execution_time,
                data_quality_score=data_quality_score
            )

            # Cache result
            self._query_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error executing analytics query: {e}")
            raise

    async def _query_news2_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query NEWS2 score data (simulated)"""
        start_date, end_date = query.get_date_range()

        # Simulate NEWS2 data
        data = []
        current_date = start_date

        while current_date <= end_date:
            for ward_id in query.ward_ids or ["ward_a", "ward_b", "ward_c"]:
                # Generate sample NEWS2 scores
                for i in range(5):  # 5 patients per ward per time period
                    patient_id = f"P{ward_id[-1]}{i:03d}"
                    news2_score = max(0, min(15, 3 + (i % 4) + (hash(current_date.isoformat()) % 3)))

                    data.append({
                        "timestamp": current_date.isoformat(),
                        "patient_id": patient_id,
                        "ward_id": ward_id,
                        "news2_score": news2_score,
                        "risk_level": "low" if news2_score < 3 else "medium" if news2_score < 5 else "high"
                    })

            current_date += timedelta(hours=1)

        return self._apply_filters(data, query.filters)

    async def _query_alert_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query alert count data (simulated)"""
        start_date, end_date = query.get_date_range()

        data = []
        current_date = start_date

        while current_date <= end_date:
            for ward_id in query.ward_ids or ["ward_a", "ward_b", "ward_c"]:
                alert_count = max(0, 2 + (hash(current_date.isoformat() + ward_id) % 8))
                critical_alerts = max(0, alert_count // 4)

                data.append({
                    "timestamp": current_date.isoformat(),
                    "ward_id": ward_id,
                    "total_alerts": alert_count,
                    "critical_alerts": critical_alerts,
                    "acknowledged_alerts": max(0, alert_count - 1),
                    "avg_response_time_minutes": 5 + (hash(ward_id) % 15)
                })

            current_date += timedelta(hours=1)

        return self._apply_filters(data, query.filters)

    async def _query_response_time_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query response time data (simulated)"""
        start_date, end_date = query.get_date_range()

        data = []
        current_date = start_date

        while current_date <= end_date:
            for ward_id in query.ward_ids or ["ward_a", "ward_b", "ward_c"]:
                # Simulate response times for different alert levels
                data.extend([
                    {
                        "timestamp": current_date.isoformat(),
                        "ward_id": ward_id,
                        "alert_level": "critical",
                        "response_time_minutes": 2 + (hash(current_date.isoformat()) % 3),
                        "staff_id": f"staff_{ward_id}_{i}"
                    }
                    for i in range(2)  # 2 critical alerts per hour
                ])

                data.extend([
                    {
                        "timestamp": current_date.isoformat(),
                        "ward_id": ward_id,
                        "alert_level": "medium",
                        "response_time_minutes": 8 + (hash(current_date.isoformat()) % 12),
                        "staff_id": f"staff_{ward_id}_{i}"
                    }
                    for i in range(3)  # 3 medium alerts per hour
                ])

            current_date += timedelta(hours=1)

        return self._apply_filters(data, query.filters)

    async def _query_outcome_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query patient outcome data (simulated)"""
        start_date, end_date = query.get_date_range()

        data = []
        for i in range(50):  # 50 patients over the period
            admission_date = start_date + timedelta(
                days=(end_date - start_date).days * (i / 50)
            )

            length_of_stay = 2 + (i % 10)  # 2-12 days
            discharge_date = admission_date + timedelta(days=length_of_stay)

            if discharge_date <= end_date:
                data.append({
                    "patient_id": f"P{i:03d}",
                    "admission_date": admission_date.isoformat(),
                    "discharge_date": discharge_date.isoformat(),
                    "length_of_stay_days": length_of_stay,
                    "outcome": "discharged_home" if i % 10 < 8 else "transferred" if i % 10 < 9 else "deceased",
                    "readmitted_30_days": i % 20 == 0,  # 5% readmission rate
                    "complications": i % 15 == 0,  # 6.7% complication rate
                    "ward_id": ["ward_a", "ward_b", "ward_c"][i % 3]
                })

        return self._apply_filters(data, query.filters)

    async def _query_workload_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query staff workload data (simulated)"""
        start_date, end_date = query.get_date_range()

        data = []
        current_date = start_date

        while current_date <= end_date:
            for ward_id in query.ward_ids or ["ward_a", "ward_b", "ward_c"]:
                for shift in ["day", "evening", "night"]:
                    staff_count = 4 + (hash(ward_id + shift) % 3)
                    patient_count = 15 + (hash(current_date.isoformat() + ward_id) % 10)

                    data.append({
                        "timestamp": current_date.isoformat(),
                        "ward_id": ward_id,
                        "shift": shift,
                        "staff_count": staff_count,
                        "patient_count": patient_count,
                        "patient_to_staff_ratio": round(patient_count / staff_count, 2),
                        "overtime_hours": max(0, (patient_count - 18) * 0.5),
                        "tasks_completed": patient_count * 8 + (hash(shift) % 10),
                        "avg_task_time_minutes": 12 + (hash(ward_id) % 8)
                    })

            current_date += timedelta(hours=8)  # 8-hour shifts

        return self._apply_filters(data, query.filters)

    async def _query_resource_data(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Query resource utilization data (simulated)"""
        start_date, end_date = query.get_date_range()

        data = []
        current_date = start_date

        while current_date <= end_date:
            for ward_id in query.ward_ids or ["ward_a", "ward_b", "ward_c"]:
                bed_capacity = 25
                occupied_beds = 18 + (hash(current_date.isoformat() + ward_id) % 8)

                data.append({
                    "timestamp": current_date.isoformat(),
                    "ward_id": ward_id,
                    "bed_capacity": bed_capacity,
                    "occupied_beds": min(occupied_beds, bed_capacity),
                    "bed_utilization_pct": round((min(occupied_beds, bed_capacity) / bed_capacity) * 100, 2),
                    "equipment_monitors": {"total": 10, "in_use": 7 + (hash(ward_id) % 3)},
                    "medication_compliance_pct": 85 + (hash(current_date.isoformat()) % 15),
                    "documentation_completion_pct": 90 + (hash(ward_id) % 10)
                })

            current_date += timedelta(hours=4)  # Every 4 hours

        return self._apply_filters(data, query.filters)

    def _apply_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to query results"""
        if not filters:
            return data

        filtered_data = []
        for record in data:
            include_record = True

            for filter_key, filter_value in filters.items():
                if filter_key in record:
                    if isinstance(filter_value, list):
                        if record[filter_key] not in filter_value:
                            include_record = False
                            break
                    else:
                        if record[filter_key] != filter_value:
                            include_record = False
                            break

            if include_record:
                filtered_data.append(record)

        return filtered_data

    def _apply_aggregation(self, data: List[Dict[str, Any]],
                          aggregation: AggregationType,
                          grouping: List[str]) -> List[Dict[str, Any]]:
        """Apply aggregation to query results"""
        if not data:
            return []

        if not grouping:
            # No grouping, aggregate entire dataset
            return [self._aggregate_records(data, aggregation)]

        # Group by specified fields
        groups = {}
        for record in data:
            group_key = tuple(record.get(field, "") for field in grouping)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)

        # Aggregate each group
        aggregated_data = []
        for group_key, group_records in groups.items():
            aggregated_record = self._aggregate_records(group_records, aggregation)

            # Add grouping fields to the result
            for i, field in enumerate(grouping):
                aggregated_record[field] = group_key[i]

            aggregated_data.append(aggregated_record)

        return aggregated_data

    def _aggregate_records(self, records: List[Dict[str, Any]],
                          aggregation: AggregationType) -> Dict[str, Any]:
        """Aggregate a group of records"""
        if not records:
            return {}

        result = {}
        numeric_fields = []

        # Identify numeric fields
        for field, value in records[0].items():
            if isinstance(value, (int, float)):
                numeric_fields.append(field)

        # Apply aggregation to numeric fields
        for field in numeric_fields:
            values = [record.get(field, 0) for record in records if isinstance(record.get(field), (int, float))]

            if values:
                if aggregation == AggregationType.COUNT:
                    result[f"{field}_count"] = len(values)
                elif aggregation == AggregationType.SUM:
                    result[f"{field}_sum"] = sum(values)
                elif aggregation == AggregationType.AVERAGE:
                    result[f"{field}_avg"] = statistics.mean(values)
                elif aggregation == AggregationType.MEDIAN:
                    result[f"{field}_median"] = statistics.median(values)
                elif aggregation == AggregationType.MIN:
                    result[f"{field}_min"] = min(values)
                elif aggregation == AggregationType.MAX:
                    result[f"{field}_max"] = max(values)
                elif aggregation == AggregationType.STANDARD_DEVIATION:
                    result[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                elif aggregation == AggregationType.PERCENTILE_95:
                    sorted_values = sorted(values)
                    idx = int(0.95 * len(sorted_values))
                    result[f"{field}_p95"] = sorted_values[min(idx, len(sorted_values) - 1)]

        # Add record count
        result["record_count"] = len(records)

        return result

    def _calculate_summary_stats(self, data: List[Dict[str, Any]],
                                aggregation: AggregationType) -> Dict[str, float]:
        """Calculate summary statistics for the result set"""
        if not data:
            return {}

        summary = {
            "total_records": len(data),
            "aggregation_method": aggregation.value
        }

        # Find numeric fields and calculate summary stats
        numeric_fields = []
        for record in data:
            for field, value in record.items():
                if isinstance(value, (int, float)) and field not in numeric_fields:
                    numeric_fields.append(field)

        for field in numeric_fields:
            values = [record[field] for record in data if field in record and isinstance(record[field], (int, float))]

            if values:
                summary[f"{field}_mean"] = statistics.mean(values)
                summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                summary[f"{field}_min"] = min(values)
                summary[f"{field}_max"] = max(values)

        return summary

    def _assess_data_quality(self, data: List[Dict[str, Any]], query: AnalyticsQuery) -> float:
        """Assess data quality score (0-1)"""
        if not data:
            return 0.0

        quality_factors = []

        # Completeness: percentage of non-null values
        total_fields = 0
        complete_fields = 0

        for record in data:
            for field, value in record.items():
                total_fields += 1
                if value is not None and value != "":
                    complete_fields += 1

        completeness = complete_fields / total_fields if total_fields > 0 else 0
        quality_factors.append(completeness)

        # Freshness: how recent the data is
        if data and "timestamp" in data[0]:
            latest_timestamp = max(
                datetime.fromisoformat(record["timestamp"])
                for record in data if "timestamp" in record
            )
            age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
            freshness = max(0, 1 - (age_hours / 24))  # Decreases over 24 hours
            quality_factors.append(freshness)

        # Volume: sufficient data for reliable analysis
        expected_records = self._calculate_expected_records(query)
        volume_ratio = min(1.0, len(data) / max(expected_records, 1))
        quality_factors.append(volume_ratio)

        return statistics.mean(quality_factors)

    def _calculate_expected_records(self, query: AnalyticsQuery) -> int:
        """Calculate expected number of records for data quality assessment"""
        start_date, end_date = query.get_date_range()
        time_span_hours = (end_date - start_date).total_seconds() / 3600

        # Base expectations on metric type
        if query.metric_type == MetricType.NEWS2_SCORES:
            return int(time_span_hours * 5 * len(query.ward_ids or ["ward_a"]))  # 5 patients per ward per hour
        elif query.metric_type == MetricType.ALERT_COUNTS:
            return int(time_span_hours * len(query.ward_ids or ["ward_a"]))  # 1 record per ward per hour
        else:
            return int(time_span_hours * 2)  # Default expectation

    def _generate_cache_key(self, query: AnalyticsQuery) -> str:
        """Generate cache key for query"""
        import hashlib

        # Handle case where metric_type might be a string (for testing)
        metric_type_value = query.metric_type.value if hasattr(query.metric_type, 'value') else str(query.metric_type)
        time_range_value = query.time_range.value if hasattr(query.time_range, 'value') else str(query.time_range)
        aggregation_value = query.aggregation.value if hasattr(query.aggregation, 'value') else str(query.aggregation)

        query_str = json.dumps({
            "metric_type": metric_type_value,
            "time_range": time_range_value,
            "aggregation": aggregation_value,
            "filters": query.filters,
            "grouping": query.grouping,
            "ward_ids": query.ward_ids
        }, sort_keys=True)

        return hashlib.md5(query_str.encode()).hexdigest()

    async def get_kpi_dashboard(self, ward_ids: List[str] = None) -> Dict[str, Any]:
        """Get key performance indicators for executive dashboard"""
        try:
            # Define KPI queries
            kpi_queries = [
                AnalyticsQuery(
                    metric_type=MetricType.NEWS2_SCORES,
                    time_range=TimeRange.LAST_24_HOURS,
                    aggregation=AggregationType.AVERAGE,
                    ward_ids=ward_ids or []
                ),
                AnalyticsQuery(
                    metric_type=MetricType.ALERT_COUNTS,
                    time_range=TimeRange.LAST_24_HOURS,
                    aggregation=AggregationType.SUM,
                    ward_ids=ward_ids or []
                ),
                AnalyticsQuery(
                    metric_type=MetricType.RESPONSE_TIMES,
                    time_range=TimeRange.LAST_24_HOURS,
                    aggregation=AggregationType.AVERAGE,
                    ward_ids=ward_ids or []
                )
            ]

            # Execute queries
            results = await asyncio.gather(*[self.execute_query(query) for query in kpi_queries])

            # Build KPI metrics
            kpis = []

            # Average NEWS2 Score
            if results[0].summary_stats:
                avg_news2 = next(
                    (v for k, v in results[0].summary_stats.items() if "news2_score_mean" in k),
                    0
                )
                kpis.append(KPIMetric(
                    name="Average NEWS2 Score",
                    value=round(avg_news2, 2),
                    unit="points",
                    trend="stable",
                    trend_percentage=0.5,
                    target_value=3.0,
                    status="normal" if avg_news2 <= 3.0 else "warning",
                    description="24-hour average NEWS2 score across all monitored patients"
                ))

            # Total Alerts
            if results[1].summary_stats:
                total_alerts = results[1].summary_stats.get("total_alerts_sum", 0)
                kpis.append(KPIMetric(
                    name="Total Alerts",
                    value=int(total_alerts),
                    unit="alerts",
                    trend="down",
                    trend_percentage=-2.3,
                    target_value=50,
                    status="normal" if total_alerts <= 50 else "warning",
                    description="Total alerts generated in the last 24 hours"
                ))

            # Average Response Time
            if results[2].summary_stats:
                avg_response = next(
                    (v for k, v in results[2].summary_stats.items() if "response_time_minutes_mean" in k),
                    0
                )
                kpis.append(KPIMetric(
                    name="Average Response Time",
                    value=round(avg_response, 1),
                    unit="minutes",
                    trend="up",
                    trend_percentage=1.2,
                    target_value=5.0,
                    status="normal" if avg_response <= 5.0 else "warning",
                    description="Average time to respond to alerts"
                ))

            return {
                "kpis": [kpi.to_dict() for kpi in kpis],
                "last_updated": datetime.now().isoformat(),
                "data_quality": statistics.mean([r.data_quality_score for r in results])
            }

        except Exception as e:
            self.logger.error(f"Error generating KPI dashboard: {e}")
            raise

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get analytics service performance metrics"""
        return {
            "cache_size": len(self._query_cache),
            "cache_hit_rate": 0.75,  # Simulated
            "avg_query_time_ms": 450,
            "queries_executed_24h": 1250,
            "data_quality_avg": 0.89,
            "system_load": 0.65
        }