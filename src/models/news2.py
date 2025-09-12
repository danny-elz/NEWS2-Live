from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RiskCategory(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NEWS2Result:
    total_score: int
    individual_scores: Dict[str, int]
    risk_category: RiskCategory
    monitoring_frequency: str
    scale_used: int
    warnings: List[str]
    confidence: float
    calculated_at: datetime
    calculation_time_ms: float
    clinical_guidance: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "total_score": self.total_score,
            "individual_scores": self.individual_scores,
            "risk_category": self.risk_category.value,
            "monitoring_frequency": self.monitoring_frequency,
            "scale_used": self.scale_used,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "calculated_at": self.calculated_at.isoformat(),
            "calculation_time_ms": self.calculation_time_ms
        }
        if self.clinical_guidance:
            result["clinical_guidance"] = self.clinical_guidance
        return result


class CalculationError(Exception):
    """Base exception for NEWS2 calculation errors"""
    pass


class MissingVitalSignsError(CalculationError):
    """Raised when required vital signs are missing"""
    pass


class InvalidVitalSignsError(CalculationError):
    """Raised when vital signs are outside valid ranges"""
    pass