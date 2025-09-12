import asyncio
import statistics
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats

from ..models.patient_state import TrendingAnalysis, TrendingCalculationError
from ..models.vital_signs import VitalSigns
from ..models.news2 import NEWS2Result
from ..services.audit import AuditLogger, AuditOperation


@dataclass
class VitalSignsWindow:
    """Vital signs data for time window analysis."""
    timestamp: datetime
    vital_signs: VitalSigns
    news2_score: int
    

@dataclass 
class TrendingResult:
    """Result of trending analysis calculations."""
    patient_id: str
    window_start: datetime
    window_end: datetime
    rolling_stats: Dict[str, Dict[str, float]]  # parameter -> {min, max, avg, std}
    news2_trend_slope: float
    deterioration_risk: str
    early_warning_indicators: List[str]
    trend_comparison: Dict[str, Optional[int]]  # timepoint -> score
    confidence_score: float
    

class PatientStateTracker:
    """Service for vital signs trend analysis and deterioration detection."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        
    async def calculate_24h_trends(self, patient_id: str, 
                                  vital_signs_history: List[VitalSignsWindow]) -> TrendingResult:
        """Calculate 24-hour rolling window trends for patient deterioration detection."""
        if not vital_signs_history:
            raise TrendingCalculationError(f"No vital signs history available for patient")
        
        # Sort by timestamp to ensure proper chronological order
        sorted_history = sorted(vital_signs_history, key=lambda x: x.timestamp)
        
        # Filter to 24-hour window
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=24)
        windowed_data = [
            vs for vs in sorted_history 
            if vs.timestamp >= window_start
        ]
        
        if not windowed_data:
            raise TrendingCalculationError(f"No vital signs data in 24-hour window for patient")
        
        # Calculate rolling window statistics
        rolling_stats = self._calculate_rolling_statistics(windowed_data)
        
        # Calculate NEWS2 trend slope
        news2_trend_slope = self._calculate_trend_slope(windowed_data)
        
        # Assess deterioration risk
        deterioration_risk = self._assess_deterioration_risk(windowed_data, news2_trend_slope)
        
        # Generate early warning indicators
        early_warning_indicators = self._generate_early_warnings(windowed_data, rolling_stats)
        
        # Calculate trend comparisons
        trend_comparison = self._calculate_trend_comparisons(windowed_data, now)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(windowed_data)
        
        result = TrendingResult(
            patient_id=patient_id,
            window_start=window_start,
            window_end=now,
            rolling_stats=rolling_stats,
            news2_trend_slope=news2_trend_slope,
            deterioration_risk=deterioration_risk,
            early_warning_indicators=early_warning_indicators,
            trend_comparison=trend_comparison,
            confidence_score=confidence_score
        )
        
        # Audit trending calculation
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,  # Using existing operation for now
            patient_id=patient_id,
            details={
                'operation': 'trending_analysis',
                'window_duration_hours': 24,
                'data_points_analyzed': len(windowed_data),
                'deterioration_risk': deterioration_risk,
                'trend_slope': news2_trend_slope,
                'confidence_score': confidence_score
            }
        )
        
        return result
    
    def _calculate_rolling_statistics(self, windowed_data: List[VitalSignsWindow]) -> Dict[str, Dict[str, float]]:
        """Calculate rolling window statistics for vital sign parameters."""
        stats_dict = {}
        
        # Extract vital sign parameters
        parameters = {
            'respiratory_rate': [vs.vital_signs.respiratory_rate for vs in windowed_data],
            'spo2': [vs.vital_signs.spo2 for vs in windowed_data],
            'temperature': [vs.vital_signs.temperature for vs in windowed_data],
            'systolic_bp': [vs.vital_signs.systolic_bp for vs in windowed_data],
            'heart_rate': [vs.vital_signs.heart_rate for vs in windowed_data],
            'news2_score': [vs.news2_score for vs in windowed_data]
        }
        
        for param, values in parameters.items():
            if values and all(v is not None for v in values):
                stats_dict[param] = {
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'avg': float(statistics.mean(values)),
                    'std': float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                    'median': float(statistics.median(values))
                }
            else:
                stats_dict[param] = {
                    'min': 0.0, 'max': 0.0, 'avg': 0.0, 'std': 0.0, 'median': 0.0
                }
        
        return stats_dict
    
    def _calculate_trend_slope(self, windowed_data: List[VitalSignsWindow]) -> float:
        """Calculate NEWS2 trend slope using linear regression."""
        if len(windowed_data) < 2:
            return 0.0
            
        # Convert timestamps to hours since first reading
        first_timestamp = windowed_data[0].timestamp
        x_values = [
            (vs.timestamp - first_timestamp).total_seconds() / 3600 
            for vs in windowed_data
        ]
        y_values = [vs.news2_score for vs in windowed_data]
        
        try:
            # Calculate linear regression slope
            slope, _, _, _, _ = stats.linregress(x_values, y_values)
            return float(slope)
        except Exception:
            return 0.0
    
    def _assess_deterioration_risk(self, windowed_data: List[VitalSignsWindow], 
                                  trend_slope: float) -> str:
        """Assess patient deterioration risk based on trends."""
        if not windowed_data:
            return "UNKNOWN"
        
        latest_score = windowed_data[-1].news2_score
        
        # High risk conditions
        if latest_score >= 7:
            return "HIGH"
        
        if trend_slope > 0.5:  # Rapidly increasing NEWS2 score
            return "HIGH"
        
        # Check for sustained elevation
        recent_scores = [vs.news2_score for vs in windowed_data[-6:]]  # Last 6 readings
        if all(score >= 5 for score in recent_scores):
            return "HIGH"
        
        # Medium risk conditions
        if latest_score >= 5 or trend_slope > 0.2:
            return "MEDIUM"
        
        # Check for concerning parameter patterns
        if self._has_concerning_patterns(windowed_data):
            return "MEDIUM"
        
        return "LOW"
    
    def _has_concerning_patterns(self, windowed_data: List[VitalSignsWindow]) -> bool:
        """Detect concerning patterns in vital signs."""
        if len(windowed_data) < 3:
            return False
        
        recent_data = windowed_data[-3:]
        
        # Check for steadily increasing heart rate
        heart_rates = [vs.vital_signs.heart_rate for vs in recent_data]
        if all(hr is not None for hr in heart_rates):
            if heart_rates[2] > heart_rates[1] > heart_rates[0] and heart_rates[2] - heart_rates[0] > 20:
                return True
        
        # Check for steadily decreasing SpO2
        spo2_values = [vs.vital_signs.spo2 for vs in recent_data]
        if all(spo2 is not None for spo2 in spo2_values):
            if spo2_values[0] > spo2_values[1] > spo2_values[2] and spo2_values[0] - spo2_values[2] > 5:
                return True
        
        # Check for steadily increasing respiratory rate
        rr_values = [vs.vital_signs.respiratory_rate for vs in recent_data]
        if all(rr is not None for rr in rr_values):
            if rr_values[2] > rr_values[1] > rr_values[0] and rr_values[2] - rr_values[0] > 5:
                return True
        
        return False
    
    def _generate_early_warnings(self, windowed_data: List[VitalSignsWindow],
                               rolling_stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate early warning indicators based on trend analysis."""
        warnings = []
        
        if not windowed_data:
            return warnings
        
        latest = windowed_data[-1]
        
        # Check for parameters outside normal variance
        if latest.vital_signs.heart_rate is not None and rolling_stats['heart_rate']['std'] > 15:
            warnings.append("Heart rate variability increased")
        
        if latest.vital_signs.respiratory_rate is not None and rolling_stats['respiratory_rate']['std'] > 3:
            warnings.append("Respiratory rate variability increased")
        
        # Check for sustained parameter elevation
        if rolling_stats['news2_score']['avg'] > 3 and rolling_stats['news2_score']['min'] >= 2:
            warnings.append("Sustained NEWS2 score elevation")
        
        # Check for rapid parameter changes
        if len(windowed_data) >= 2:
            previous = windowed_data[-2]
            current = windowed_data[-1]
            
            if (current.vital_signs.heart_rate and previous.vital_signs.heart_rate and 
                current.vital_signs.heart_rate - previous.vital_signs.heart_rate > 15):
                warnings.append("Rapid heart rate increase detected")
            
            if (current.vital_signs.spo2 and previous.vital_signs.spo2 and
                previous.vital_signs.spo2 - current.vital_signs.spo2 > 4):
                warnings.append("Significant SpO2 decrease detected")
        
        return warnings
    
    def _calculate_trend_comparisons(self, windowed_data: List[VitalSignsWindow], 
                                   now: datetime) -> Dict[str, Optional[int]]:
        """Calculate NEWS2 score comparisons at different time points."""
        comparisons = {
            'current': None,
            'score_2h_ago': None,
            'score_4h_ago': None,
            'score_8h_ago': None,
            'score_12h_ago': None,
            'score_24h_ago': None
        }
        
        if not windowed_data:
            return comparisons
        
        # Current score (most recent)
        comparisons['current'] = windowed_data[-1].news2_score
        
        # Find scores at specific time points
        time_points = {
            'score_2h_ago': now - timedelta(hours=2),
            'score_4h_ago': now - timedelta(hours=4),
            'score_8h_ago': now - timedelta(hours=8),
            'score_12h_ago': now - timedelta(hours=12),
            'score_24h_ago': now - timedelta(hours=24)
        }
        
        for key, target_time in time_points.items():
            # Find closest reading to target time
            closest_reading = min(
                windowed_data,
                key=lambda vs: abs((vs.timestamp - target_time).total_seconds()),
                default=None
            )
            
            if closest_reading and abs((closest_reading.timestamp - target_time).total_seconds()) < 3600:  # Within 1 hour
                comparisons[key] = closest_reading.news2_score
        
        return comparisons
    
    def _calculate_confidence_score(self, windowed_data: List[VitalSignsWindow]) -> float:
        """Calculate confidence score based on data quality and completeness."""
        if not windowed_data:
            return 0.0
        
        total_data_points = len(windowed_data)
        
        # Base confidence on data quantity
        if total_data_points >= 24:  # Hourly readings for 24 hours
            quantity_score = 1.0
        elif total_data_points >= 12:  # Every 2 hours
            quantity_score = 0.8
        elif total_data_points >= 6:  # Every 4 hours
            quantity_score = 0.6
        else:
            quantity_score = 0.4
        
        # Check data completeness (all vital signs present)
        complete_readings = 0
        for vs in windowed_data:
            if all([
                vs.vital_signs.respiratory_rate is not None,
                vs.vital_signs.spo2 is not None,
                vs.vital_signs.temperature is not None,
                vs.vital_signs.systolic_bp is not None,
                vs.vital_signs.heart_rate is not None
            ]):
                complete_readings += 1
        
        completeness_score = complete_readings / total_data_points
        
        # Calculate time distribution score
        if total_data_points > 1:
            time_spans = []
            for i in range(1, len(windowed_data)):
                time_diff = (windowed_data[i].timestamp - windowed_data[i-1].timestamp).total_seconds()
                time_spans.append(time_diff)
            
            # Good distribution if readings are relatively evenly spaced
            mean_interval = statistics.mean(time_spans)
            std_interval = statistics.stdev(time_spans) if len(time_spans) > 1 else 0
            distribution_score = max(0.0, 1.0 - (std_interval / mean_interval) if mean_interval > 0 else 0.5)
        else:
            distribution_score = 0.5
        
        # Weighted final confidence score
        final_confidence = (
            quantity_score * 0.4 +
            completeness_score * 0.4 +
            distribution_score * 0.2
        )
        
        return round(final_confidence, 3)
    
    async def update_trending_analysis(self, patient_id: str, 
                                     trending_analysis: TrendingAnalysis) -> TrendingAnalysis:
        """Update patient's trending analysis data."""
        # This would typically update the database
        # For now, just log the update
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'trending_update',
                'deterioration_risk': trending_analysis.deterioration_risk,
                'trend_slope': trending_analysis.trend_slope,
                'early_warnings': len(trending_analysis.early_warning_indicators)
            }
        )
        
        return trending_analysis