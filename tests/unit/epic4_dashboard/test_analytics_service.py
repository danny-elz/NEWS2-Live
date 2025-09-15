"""
Unit tests for Analytics Service (Story 3.5)
Tests core analytics functionality and KPI dashboard generation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analytics.analytics_service import (
    AnalyticsService,
    AnalyticsQuery,
    AnalyticsResult,
    TimeRange,
    AggregationType,
    MetricType,
    KPIMetric
)


class TestAnalyticsService:
    """Test suite for AnalyticsService"""

    @pytest.fixture
    def analytics_service(self):
        """Create AnalyticsService instance"""
        return AnalyticsService()

    @pytest.fixture
    def sample_query(self):
        """Create sample analytics query"""
        return AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE,
            ward_ids=["ward_a", "ward_b"],
            filters={"risk_level": "high"}
        )

    @pytest.mark.asyncio
    async def test_execute_news2_query(self, analytics_service, sample_query):
        """Test executing NEWS2 scores query"""
        result = await analytics_service.execute_query(sample_query)

        assert isinstance(result, AnalyticsResult)
        assert result.query == sample_query
        assert len(result.data) > 0
        assert result.execution_time_ms > 0
        assert 0 <= result.data_quality_score <= 1

        # Verify metadata structure
        assert "total_records" in result.metadata
        assert "date_range" in result.metadata
        assert "filters_applied" in result.metadata

        # Verify summary statistics
        assert isinstance(result.summary_stats, dict)
        assert result.summary_stats["aggregation_method"] == AggregationType.AVERAGE.value

    @pytest.mark.asyncio
    async def test_execute_alert_counts_query(self, analytics_service):
        """Test executing alert counts query"""
        query = AnalyticsQuery(
            metric_type=MetricType.ALERT_COUNTS,
            time_range=TimeRange.LAST_4_HOURS,
            aggregation=AggregationType.SUM
        )

        result = await analytics_service.execute_query(query)

        assert result.query.metric_type == MetricType.ALERT_COUNTS
        assert len(result.data) > 0

        # Check that alert data contains expected fields
        for record in result.data:
            if "total_alerts_sum" in record:
                assert isinstance(record["total_alerts_sum"], (int, float))

    @pytest.mark.asyncio
    async def test_execute_response_time_query(self, analytics_service):
        """Test executing response time query"""
        query = AnalyticsQuery(
            metric_type=MetricType.RESPONSE_TIMES,
            time_range=TimeRange.LAST_HOUR,
            aggregation=AggregationType.AVERAGE,
            grouping=["ward_id"]
        )

        result = await analytics_service.execute_query(query)

        assert len(result.data) > 0
        # With grouping, each record should have ward_id
        for record in result.data:
            assert "ward_id" in record

    def test_time_range_calculation(self, sample_query):
        """Test time range date calculation"""
        start_date, end_date = sample_query.get_date_range()

        assert isinstance(start_date, datetime)
        assert isinstance(end_date, datetime)
        assert start_date < end_date

        # Test 24-hour range
        time_diff = end_date - start_date
        assert 23 <= time_diff.total_seconds() / 3600 <= 25  # ~24 hours with some tolerance

    def test_custom_time_range(self):
        """Test custom time range query"""
        custom_start = datetime(2024, 1, 1, 10, 0, 0)
        custom_end = datetime(2024, 1, 1, 18, 0, 0)

        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.CUSTOM,
            aggregation=AggregationType.COUNT,
            custom_start_date=custom_start,
            custom_end_date=custom_end
        )

        start_date, end_date = query.get_date_range()
        assert start_date == custom_start
        assert end_date == custom_end

    @pytest.mark.asyncio
    async def test_query_with_filters(self, analytics_service):
        """Test query with filters applied"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE,
            filters={"ward_id": ["ward_a"], "risk_level": "high"}
        )

        result = await analytics_service.execute_query(query)

        # Verify filters were applied in metadata
        assert result.metadata["filters_applied"] == query.filters

    @pytest.mark.asyncio
    async def test_aggregation_functions(self, analytics_service):
        """Test different aggregation functions"""
        aggregations = [
            AggregationType.COUNT,
            AggregationType.SUM,
            AggregationType.AVERAGE,
            AggregationType.MIN,
            AggregationType.MAX
        ]

        for agg_type in aggregations:
            query = AnalyticsQuery(
                metric_type=MetricType.NEWS2_SCORES,
                time_range=TimeRange.LAST_HOUR,
                aggregation=agg_type
            )

            result = await analytics_service.execute_query(query)
            assert result.summary_stats["aggregation_method"] == agg_type.value
            assert len(result.data) > 0

    @pytest.mark.asyncio
    async def test_grouping_functionality(self, analytics_service):
        """Test query grouping functionality"""
        query = AnalyticsQuery(
            metric_type=MetricType.ALERT_COUNTS,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.SUM,
            grouping=["ward_id", "alert_level"]
        )

        result = await analytics_service.execute_query(query)

        # With grouping, results should be grouped
        assert len(result.data) > 1  # Multiple groups expected

        for record in result.data:
            assert "ward_id" in record
            assert "record_count" in record  # Aggregation adds record count

    @pytest.mark.asyncio
    async def test_get_kpi_dashboard(self, analytics_service):
        """Test KPI dashboard generation"""
        dashboard = await analytics_service.get_kpi_dashboard(ward_ids=["ward_a", "ward_b"])

        assert "kpis" in dashboard
        assert "last_updated" in dashboard
        assert "data_quality" in dashboard

        # Verify KPI structure
        kpis = dashboard["kpis"]
        assert len(kpis) > 0

        for kpi in kpis:
            assert "name" in kpi
            assert "value" in kpi
            assert "unit" in kpi
            assert "trend" in kpi
            assert "status" in kpi

        # Check for expected KPIs
        kpi_names = [kpi["name"] for kpi in kpis]
        assert "Average NEWS2 Score" in kpi_names
        assert "Total Alerts" in kpi_names
        assert "Average Response Time" in kpi_names

    @pytest.mark.asyncio
    async def test_query_caching(self, analytics_service):
        """Test query result caching"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_HOUR,
            aggregation=AggregationType.AVERAGE
        )

        # First execution
        result1 = await analytics_service.execute_query(query)
        first_execution_time = result1.execution_time_ms

        # Second execution (should be cached)
        result2 = await analytics_service.execute_query(query)

        # Results should be identical
        assert result1.data == result2.data
        assert result1.metadata == result2.metadata

    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, analytics_service):
        """Test data quality score calculation"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE
        )

        result = await analytics_service.execute_query(query)

        # Data quality should be between 0 and 1
        assert 0 <= result.data_quality_score <= 1

        # Should have reasonable quality for simulated data
        assert result.data_quality_score > 0.5

    @pytest.mark.asyncio
    async def test_performance_metrics(self, analytics_service):
        """Test analytics service performance metrics"""
        metrics = await analytics_service.get_performance_metrics()

        expected_metrics = [
            "cache_size", "cache_hit_rate", "avg_query_time_ms",
            "queries_executed_24h", "data_quality_avg", "system_load"
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_kpi_metric_creation(self):
        """Test KPIMetric creation and serialization"""
        kpi = KPIMetric(
            name="Test Metric",
            value=75.5,
            unit="%",
            trend="up",
            trend_percentage=5.2,
            target_value=80.0,
            status="warning",
            description="Test metric description"
        )

        kpi_dict = kpi.to_dict()

        assert kpi_dict["name"] == "Test Metric"
        assert kpi_dict["value"] == 75.5
        assert kpi_dict["unit"] == "%"
        assert kpi_dict["trend"] == "up"
        assert kpi_dict["trend_percentage"] == 5.2
        assert kpi_dict["status"] == "warning"

    def test_analytics_result_serialization(self, sample_query):
        """Test AnalyticsResult serialization"""
        result = AnalyticsResult(
            query=sample_query,
            data=[{"test": "data"}],
            metadata={"test": "metadata"},
            summary_stats={"mean": 5.0},
            execution_time_ms=100,
            data_quality_score=0.85,
            confidence_interval=(0.7, 0.9)
        )

        result_dict = result.to_dict()

        assert "query" in result_dict
        assert "data" in result_dict
        assert "metadata" in result_dict
        assert "summary_stats" in result_dict
        assert "execution_time_ms" in result_dict
        assert "data_quality_score" in result_dict
        assert "confidence_interval" in result_dict

        # Verify query serialization
        assert result_dict["query"]["metric_type"] == MetricType.NEWS2_SCORES.value

    @pytest.mark.asyncio
    async def test_empty_query_results(self, analytics_service):
        """Test handling of queries with no results"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_HOUR,
            aggregation=AggregationType.COUNT,
            filters={"ward_id": "nonexistent_ward"}
        )

        # Mock empty data
        with patch.object(analytics_service, '_query_news2_data', return_value=[]):
            result = await analytics_service.execute_query(query)

            assert len(result.data) == 0
            assert result.data_quality_score == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, analytics_service):
        """Test concurrent query execution"""
        queries = [
            AnalyticsQuery(
                metric_type=MetricType.NEWS2_SCORES,
                time_range=TimeRange.LAST_HOUR,
                aggregation=AggregationType.AVERAGE
            ),
            AnalyticsQuery(
                metric_type=MetricType.ALERT_COUNTS,
                time_range=TimeRange.LAST_4_HOURS,
                aggregation=AggregationType.SUM
            ),
            AnalyticsQuery(
                metric_type=MetricType.RESPONSE_TIMES,
                time_range=TimeRange.LAST_24_HOURS,
                aggregation=AggregationType.AVERAGE
            )
        ]

        # Execute queries concurrently
        results = await asyncio.gather(*[analytics_service.execute_query(q) for q in queries])

        assert len(results) == 3
        for result in results:
            assert isinstance(result, AnalyticsResult)
            assert result.execution_time_ms > 0

    def test_cache_key_generation(self, analytics_service, sample_query):
        """Test query cache key generation"""
        cache_key1 = analytics_service._generate_cache_key(sample_query)
        cache_key2 = analytics_service._generate_cache_key(sample_query)

        # Same query should generate same key
        assert cache_key1 == cache_key2
        assert len(cache_key1) == 32  # MD5 hash length

        # Different query should generate different key
        different_query = AnalyticsQuery(
            metric_type=MetricType.ALERT_COUNTS,
            time_range=TimeRange.LAST_HOUR,
            aggregation=AggregationType.COUNT
        )
        cache_key3 = analytics_service._generate_cache_key(different_query)
        assert cache_key1 != cache_key3


class TestAnalyticsQueryValidation:
    """Test suite for analytics query validation"""

    def test_valid_query_creation(self):
        """Test creation of valid analytics query"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE
        )

        assert query.metric_type == MetricType.NEWS2_SCORES
        assert query.time_range == TimeRange.LAST_24_HOURS
        assert query.aggregation == AggregationType.AVERAGE
        assert query.filters == {}
        assert query.grouping == []

    def test_query_with_filters(self):
        """Test query creation with filters"""
        filters = {"ward_id": "ward_a", "risk_level": ["medium", "high"]}

        query = AnalyticsQuery(
            metric_type=MetricType.ALERT_COUNTS,
            time_range=TimeRange.LAST_WEEK,
            aggregation=AggregationType.SUM,
            filters=filters
        )

        assert query.filters == filters

    def test_query_with_grouping(self):
        """Test query creation with grouping"""
        grouping = ["ward_id", "shift"]

        query = AnalyticsQuery(
            metric_type=MetricType.STAFF_WORKLOAD,
            time_range=TimeRange.LAST_MONTH,
            aggregation=AggregationType.AVERAGE,
            grouping=grouping
        )

        assert query.grouping == grouping


class TestAnalyticsErrorHandling:
    """Test suite for analytics error handling"""

    @pytest.fixture
    def analytics_service(self):
        """Create AnalyticsService instance"""
        return AnalyticsService()

    @pytest.mark.asyncio
    async def test_invalid_metric_type_handling(self, analytics_service):
        """Test handling of invalid metric types"""
        # This would normally be caught at the enum level, but test graceful handling
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,  # Valid type
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE
        )

        # Mock an unknown metric type scenario
        with patch.object(query, 'metric_type', 'unknown_metric'):
            result = await analytics_service.execute_query(query)
            # Should return empty data rather than crash
            assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_query_execution_exception(self, analytics_service):
        """Test handling of exceptions during query execution"""
        query = AnalyticsQuery(
            metric_type=MetricType.NEWS2_SCORES,
            time_range=TimeRange.LAST_24_HOURS,
            aggregation=AggregationType.AVERAGE
        )

        with patch.object(analytics_service, '_query_news2_data',
                         side_effect=Exception("Database connection error")):
            with pytest.raises(Exception):
                await analytics_service.execute_query(query)

    @pytest.mark.asyncio
    async def test_kpi_dashboard_exception(self, analytics_service):
        """Test KPI dashboard generation with exceptions"""
        with patch.object(analytics_service, 'execute_query',
                         side_effect=Exception("Query failed")):
            with pytest.raises(Exception):
                await analytics_service.get_kpi_dashboard()


class TestMetricTypeEnums:
    """Test metric type and related enums"""

    def test_time_range_enum(self):
        """Test TimeRange enum values"""
        assert TimeRange.LAST_HOUR.value == "last_hour"
        assert TimeRange.LAST_24_HOURS.value == "last_24_hours"
        assert TimeRange.CUSTOM.value == "custom"

    def test_aggregation_type_enum(self):
        """Test AggregationType enum values"""
        assert AggregationType.COUNT.value == "count"
        assert AggregationType.AVERAGE.value == "average"
        assert AggregationType.PERCENTILE_95.value == "percentile_95"

    def test_metric_type_enum(self):
        """Test MetricType enum values"""
        assert MetricType.NEWS2_SCORES.value == "news2_scores"
        assert MetricType.ALERT_COUNTS.value == "alert_counts"
        assert MetricType.PATIENT_OUTCOMES.value == "patient_outcomes"