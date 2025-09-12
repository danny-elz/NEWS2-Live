from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class ClinicalFlag(Enum):
    IS_PALLIATIVE = "is_palliative"
    DO_NOT_ESCALATE = "do_not_escalate"
    OXYGEN_DEPENDENT = "oxygen_dependent"


@dataclass
class Patient:
    patient_id: str
    ward_id: str
    bed_number: str
    age: int
    is_copd_patient: bool
    assigned_nurse_id: str
    admission_date: datetime
    last_updated: datetime
    is_palliative: bool = False
    do_not_escalate: bool = False
    oxygen_dependent: bool = False
    
    def __post_init__(self):
        self._validate_fields()
    
    def _validate_fields(self):
        if not self.patient_id:
            raise ValueError("patient_id is required")
        
        if not self.ward_id:
            raise ValueError("ward_id is required")
        
        if not self.bed_number:
            raise ValueError("bed_number is required")
        
        if not isinstance(self.age, int) or self.age < 0 or self.age > 150:
            raise ValueError("age must be between 0 and 150")
        
        if not isinstance(self.is_copd_patient, bool):
            raise ValueError("is_copd_patient must be boolean")
        
        if not self.assigned_nurse_id:
            raise ValueError("assigned_nurse_id is required")
        
        if not isinstance(self.admission_date, datetime):
            raise ValueError("admission_date must be datetime")
        
        if not isinstance(self.last_updated, datetime):
            raise ValueError("last_updated must be datetime")
        
        if not isinstance(self.is_palliative, bool):
            raise ValueError("is_palliative must be boolean")
        
        if not isinstance(self.do_not_escalate, bool):
            raise ValueError("do_not_escalate must be boolean")
        
        if not isinstance(self.oxygen_dependent, bool):
            raise ValueError("oxygen_dependent must be boolean")
    
    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "ward_id": self.ward_id,
            "bed_number": self.bed_number,
            "age": self.age,
            "is_copd_patient": self.is_copd_patient,
            "assigned_nurse_id": self.assigned_nurse_id,
            "admission_date": self.admission_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "clinical_flags": {
                "is_palliative": self.is_palliative,
                "do_not_escalate": self.do_not_escalate,
                "oxygen_dependent": self.oxygen_dependent
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Patient':
        clinical_flags = data.get('clinical_flags', {})
        return cls(
            patient_id=data['patient_id'],
            ward_id=data['ward_id'],
            bed_number=data['bed_number'],
            age=data['age'],
            is_copd_patient=data['is_copd_patient'],
            assigned_nurse_id=data['assigned_nurse_id'],
            admission_date=datetime.fromisoformat(data['admission_date']) if isinstance(data['admission_date'], str) else data['admission_date'],
            last_updated=datetime.fromisoformat(data['last_updated']) if isinstance(data['last_updated'], str) else data['last_updated'],
            is_palliative=clinical_flags.get('is_palliative', False),
            do_not_escalate=clinical_flags.get('do_not_escalate', False),
            oxygen_dependent=clinical_flags.get('oxygen_dependent', False)
        )