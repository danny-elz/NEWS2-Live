from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4, UUID
from enum import Enum


class ConsciousnessLevel(Enum):
    ALERT = "A"
    CONFUSION = "C"
    VOICE = "V"
    PAIN = "P"
    UNRESPONSIVE = "U"


@dataclass
class VitalSigns:
    event_id: UUID
    patient_id: str
    timestamp: datetime
    respiratory_rate: int
    spo2: int
    on_oxygen: bool
    temperature: float
    systolic_bp: int
    heart_rate: int
    consciousness: ConsciousnessLevel
    is_manual_entry: bool = False
    has_artifacts: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        self._validate_fields()
    
    def _validate_fields(self):
        if not self.event_id:
            raise ValueError("event_id is required")
        
        if not self.patient_id:
            raise ValueError("patient_id is required")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be datetime")
        
        if not isinstance(self.respiratory_rate, int) or not (4 <= self.respiratory_rate <= 50):
            raise ValueError("respiratory_rate must be between 4 and 50")
        
        if not isinstance(self.spo2, int) or not (50 <= self.spo2 <= 100):
            raise ValueError("spo2 must be between 50 and 100")
        
        if not isinstance(self.on_oxygen, bool):
            raise ValueError("on_oxygen must be boolean")
        
        if not isinstance(self.temperature, (int, float)) or not (30 <= self.temperature <= 45):
            raise ValueError("temperature must be between 30 and 45°C")
        
        if not isinstance(self.systolic_bp, int) or not (40 <= self.systolic_bp <= 300):
            raise ValueError("systolic_bp must be between 40 and 300")
        
        if not isinstance(self.heart_rate, int) or not (20 <= self.heart_rate <= 220):
            raise ValueError("heart_rate must be between 20 and 220")
        
        if not isinstance(self.consciousness, ConsciousnessLevel):
            raise ValueError("consciousness must be ConsciousnessLevel enum")
        
        if not isinstance(self.is_manual_entry, bool):
            raise ValueError("is_manual_entry must be boolean")
        
        if not isinstance(self.has_artifacts, bool):
            raise ValueError("has_artifacts must be boolean")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
    
    def to_dict(self) -> dict:
        return {
            "event_id": str(self.event_id),
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "vitals": {
                "respiratory_rate": self.respiratory_rate,
                "spo2": self.spo2,
                "on_oxygen": self.on_oxygen,
                "temperature": self.temperature,
                "systolic_bp": self.systolic_bp,
                "heart_rate": self.heart_rate,
                "consciousness": self.consciousness.value
            },
            "quality_flags": {
                "is_manual_entry": self.is_manual_entry,
                "has_artifacts": self.has_artifacts,
                "confidence": self.confidence
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VitalSigns':
        vitals = data.get('vitals', {})
        quality_flags = data.get('quality_flags', {})
        
        return cls(
            event_id=UUID(data['event_id']) if isinstance(data['event_id'], str) else data['event_id'],
            patient_id=data['patient_id'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            respiratory_rate=vitals.get('respiratory_rate', data.get('respiratory_rate')),
            spo2=vitals.get('spo2', data.get('spo2')),
            on_oxygen=vitals.get('on_oxygen', data.get('on_oxygen')),
            temperature=vitals.get('temperature', data.get('temperature')),
            systolic_bp=vitals.get('systolic_bp', data.get('systolic_bp')),
            heart_rate=vitals.get('heart_rate', data.get('heart_rate')),
            consciousness=ConsciousnessLevel(vitals.get('consciousness', data.get('consciousness'))),
            is_manual_entry=quality_flags.get('is_manual_entry', False),
            has_artifacts=quality_flags.get('has_artifacts', False),
            confidence=quality_flags.get('confidence', 1.0)
        )
    
    @classmethod
    def generate_event_id(cls) -> UUID:
        return uuid4()