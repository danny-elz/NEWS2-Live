from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from enum import Enum

from .vital_signs import ConsciousnessLevel


@dataclass
class PartialVitalSigns:
    """
    Partial vital signs model that allows missing parameters for edge case handling.
    Used when not all vital signs are available but calculation should still proceed.
    """
    event_id: UUID
    patient_id: str
    timestamp: datetime
    respiratory_rate: Optional[int] = None
    spo2: Optional[int] = None
    on_oxygen: Optional[bool] = None
    temperature: Optional[float] = None
    systolic_bp: Optional[int] = None
    heart_rate: Optional[int] = None
    consciousness: Optional[ConsciousnessLevel] = None
    is_manual_entry: bool = False
    has_artifacts: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        self._validate_present_fields()
    
    def _validate_present_fields(self):
        """Validate only the fields that are present (not None)."""
        if not self.event_id:
            raise ValueError("event_id is required")
        
        if not self.patient_id:
            raise ValueError("patient_id is required")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be datetime")
        
        # Validate present vital signs
        if self.respiratory_rate is not None:
            if not isinstance(self.respiratory_rate, int) or not (4 <= self.respiratory_rate <= 50):
                raise ValueError("respiratory_rate must be between 4 and 50")
        
        if self.spo2 is not None:
            if not isinstance(self.spo2, int) or not (50 <= self.spo2 <= 100):
                raise ValueError("spo2 must be between 50 and 100")
        
        if self.on_oxygen is not None:
            if not isinstance(self.on_oxygen, bool):
                raise ValueError("on_oxygen must be boolean")
        
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)) or not (30 <= self.temperature <= 45):
                raise ValueError("temperature must be between 30 and 45°C")
        
        if self.systolic_bp is not None:
            if not isinstance(self.systolic_bp, int) or not (40 <= self.systolic_bp <= 300):
                raise ValueError("systolic_bp must be between 40 and 300")
        
        if self.heart_rate is not None:
            if not isinstance(self.heart_rate, int) or not (20 <= self.heart_rate <= 220):
                raise ValueError("heart_rate must be between 20 and 220")
        
        if self.consciousness is not None:
            if not isinstance(self.consciousness, ConsciousnessLevel):
                raise ValueError("consciousness must be ConsciousnessLevel enum")
        
        if not isinstance(self.is_manual_entry, bool):
            raise ValueError("is_manual_entry must be boolean")
        
        if not isinstance(self.has_artifacts, bool):
            raise ValueError("has_artifacts must be boolean")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
    
    def get_completeness_score(self) -> float:
        """Calculate completeness score based on how many vital signs are present."""
        total_params = 7  # Total vital sign parameters
        present_params = sum([
            self.respiratory_rate is not None,
            self.spo2 is not None,
            self.on_oxygen is not None,
            self.temperature is not None,
            self.systolic_bp is not None,
            self.heart_rate is not None,
            self.consciousness is not None
        ])
        return present_params / total_params
    
    def get_missing_parameters(self) -> list[str]:
        """Get list of missing parameter names."""
        missing = []
        if self.respiratory_rate is None:
            missing.append("respiratory_rate")
        if self.spo2 is None:
            missing.append("spo2")
        if self.on_oxygen is None:
            missing.append("on_oxygen")
        if self.temperature is None:
            missing.append("temperature")
        if self.systolic_bp is None:
            missing.append("systolic_bp")
        if self.heart_rate is None:
            missing.append("heart_rate")
        if self.consciousness is None:
            missing.append("consciousness")
        return missing
    
    def has_physiologically_impossible_combination(self) -> Optional[str]:
        """
        Check for physiologically impossible combinations of vital signs.
        
        Returns:
            Error message if impossible combination found, None otherwise
        """
        # Check for impossible combinations only when both values are present
        
        # Very low SpO2 with no oxygen support in conscious patient
        if (self.spo2 is not None and self.on_oxygen is not None and self.consciousness is not None):
            if self.spo2 < 85 and not self.on_oxygen and self.consciousness == ConsciousnessLevel.ALERT:
                return "Conscious patient with SpO2 < 85% without oxygen support is physiologically unlikely"
        
        # Very high heart rate with very low blood pressure (severe shock)
        if (self.heart_rate is not None and self.systolic_bp is not None):
            if self.heart_rate > 150 and self.systolic_bp < 70:
                return "Extreme tachycardia with severe hypotension indicates critical condition requiring immediate intervention"
        
        # Very low heart rate with consciousness issues
        if (self.heart_rate is not None and self.consciousness is not None):
            if self.heart_rate < 40 and self.consciousness != ConsciousnessLevel.ALERT:
                return "Severe bradycardia with altered consciousness indicates serious cardiac compromise"
        
        # Hypothermia with high heart rate (unless septic)
        if (self.temperature is not None and self.heart_rate is not None):
            if self.temperature < 35.0 and self.heart_rate > 130:
                return "Hypothermia with extreme tachycardia suggests severe sepsis or measurement error"
        
        # Very high SpO2 (>99%) with high oxygen flow (COPD concern handled separately)
        if (self.spo2 is not None and self.on_oxygen is not None):
            if self.spo2 > 99 and self.on_oxygen:
                return "SpO2 > 99% on supplemental oxygen may indicate over-oxygenation"
        
        return None