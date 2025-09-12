from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from .patient import Patient


@dataclass
class PatientContext:
    """Patient clinical context and medical history."""
    allergies: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)  
    comorbidities: list[str] = field(default_factory=list)
    medical_history: str = ""
    special_instructions: str = ""
    age_risk_factors: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.allergies is None:
            self.allergies = []
        if self.medications is None:
            self.medications = []
        if self.comorbidities is None:
            self.comorbidities = []


@dataclass
class TrendingAnalysis:
    """Patient trending data for 24-hour analysis."""
    current_score: int = 0
    score_2h_ago: Optional[int] = None
    score_4h_ago: Optional[int] = None
    score_8h_ago: Optional[int] = None
    score_12h_ago: Optional[int] = None
    score_24h_ago: Optional[int] = None
    trend_slope: float = 0.0
    deterioration_risk: str = "LOW"  # LOW, MEDIUM, HIGH
    early_warning_indicators: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.early_warning_indicators is None:
            self.early_warning_indicators = []


@dataclass 
class PatientState:
    """Comprehensive patient state including registry and trending data."""
    patient_id: str
    current_ward_id: str
    bed_number: Optional[str]
    clinical_flags: Dict[str, bool]
    assigned_nurse_id: Optional[str]
    admission_date: datetime
    last_transfer_date: Optional[datetime]
    context: PatientContext
    trending_data: TrendingAnalysis
    state_version: int  # for optimistic locking
    last_updated: datetime
    
    def __post_init__(self):
        self._validate_fields()
        
    def _validate_fields(self):
        if not self.patient_id:
            raise ValueError("patient_id is required")
            
        if not self.current_ward_id:
            raise ValueError("current_ward_id is required")
            
        if not isinstance(self.clinical_flags, dict):
            raise ValueError("clinical_flags must be a dictionary")
            
        if not isinstance(self.admission_date, datetime):
            raise ValueError("admission_date must be datetime")
            
        if not isinstance(self.last_updated, datetime):
            raise ValueError("last_updated must be datetime")
            
        if not isinstance(self.state_version, int) or self.state_version < 0:
            raise ValueError("state_version must be non-negative integer")
            
        if self.last_transfer_date and not isinstance(self.last_transfer_date, datetime):
            raise ValueError("last_transfer_date must be datetime or None")
    
    @classmethod
    def from_patient(cls, patient: Patient) -> 'PatientState':
        """Create PatientState from existing Patient model."""
        clinical_flags = {
            'is_copd_patient': patient.is_copd_patient,
            'is_palliative': patient.is_palliative, 
            'do_not_escalate': patient.do_not_escalate,
            'oxygen_dependent': patient.oxygen_dependent
        }
        
        return cls(
            patient_id=patient.patient_id,
            current_ward_id=patient.ward_id,
            bed_number=patient.bed_number,
            clinical_flags=clinical_flags,
            assigned_nurse_id=patient.assigned_nurse_id,
            admission_date=patient.admission_date,
            last_transfer_date=None,
            context=PatientContext(),
            trending_data=TrendingAnalysis(),
            state_version=0,
            last_updated=datetime.now(timezone.utc)
        )
    
    def to_dict(self) -> dict:
        """Convert PatientState to dictionary representation."""
        return {
            'patient_id': self.patient_id,
            'current_ward_id': self.current_ward_id,
            'bed_number': self.bed_number,
            'clinical_flags': self.clinical_flags,
            'assigned_nurse_id': self.assigned_nurse_id,
            'admission_date': self.admission_date.isoformat(),
            'last_transfer_date': self.last_transfer_date.isoformat() if self.last_transfer_date else None,
            'context': {
                'allergies': self.context.allergies,
                'medications': self.context.medications,
                'comorbidities': self.context.comorbidities,
                'medical_history': self.context.medical_history,
                'special_instructions': self.context.special_instructions,
                'age_risk_factors': self.context.age_risk_factors
            },
            'trending_data': {
                'current_score': self.trending_data.current_score,
                'score_2h_ago': self.trending_data.score_2h_ago,
                'score_4h_ago': self.trending_data.score_4h_ago,
                'score_8h_ago': self.trending_data.score_8h_ago,
                'score_12h_ago': self.trending_data.score_12h_ago,
                'score_24h_ago': self.trending_data.score_24h_ago,
                'trend_slope': self.trending_data.trend_slope,
                'deterioration_risk': self.trending_data.deterioration_risk,
                'early_warning_indicators': self.trending_data.early_warning_indicators
            },
            'state_version': self.state_version,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PatientState':
        """Create PatientState from dictionary representation."""
        context_data = data.get('context', {})
        trending_data_dict = data.get('trending_data', {})
        
        context = PatientContext(
            allergies=context_data.get('allergies', []),
            medications=context_data.get('medications', []),
            comorbidities=context_data.get('comorbidities', []),
            medical_history=context_data.get('medical_history', ''),
            special_instructions=context_data.get('special_instructions', ''),
            age_risk_factors=context_data.get('age_risk_factors', {})
        )
        
        trending_data = TrendingAnalysis(
            current_score=trending_data_dict.get('current_score', 0),
            score_2h_ago=trending_data_dict.get('score_2h_ago'),
            score_4h_ago=trending_data_dict.get('score_4h_ago'),
            score_8h_ago=trending_data_dict.get('score_8h_ago'),
            score_12h_ago=trending_data_dict.get('score_12h_ago'),
            score_24h_ago=trending_data_dict.get('score_24h_ago'),
            trend_slope=trending_data_dict.get('trend_slope', 0.0),
            deterioration_risk=trending_data_dict.get('deterioration_risk', 'LOW'),
            early_warning_indicators=trending_data_dict.get('early_warning_indicators', [])
        )
        
        return cls(
            patient_id=data['patient_id'],
            current_ward_id=data['current_ward_id'],
            bed_number=data.get('bed_number'),
            clinical_flags=data['clinical_flags'],
            assigned_nurse_id=data.get('assigned_nurse_id'),
            admission_date=datetime.fromisoformat(data['admission_date']) if isinstance(data['admission_date'], str) else data['admission_date'],
            last_transfer_date=datetime.fromisoformat(data['last_transfer_date']) if data.get('last_transfer_date') else None,
            context=context,
            trending_data=trending_data,
            state_version=data.get('state_version', 0),
            last_updated=datetime.fromisoformat(data['last_updated']) if isinstance(data['last_updated'], str) else data['last_updated']
        )


class PatientStateError(Exception):
    """Base exception for patient state management errors"""
    pass


class PatientTransferError(PatientStateError):
    """Raised when patient transfer fails validation"""
    pass


class ConcurrentUpdateError(PatientStateError):
    """Raised when optimistic locking detects conflicts"""
    pass


class TrendingCalculationError(PatientStateError):
    """Raised when trending analysis encounters data issues"""
    pass