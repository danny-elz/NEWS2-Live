import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..models.patient import Patient
from ..models.patient_state import PatientContext, PatientState, PatientStateError
from ..models.news2 import NEWS2Result
from ..services.audit import AuditLogger, AuditOperation


@dataclass
class AgeRiskProfile:
    """Age-based risk factor calculations."""
    age: int
    risk_category: str  # PEDIATRIC, ADULT, GERIATRIC
    baseline_adjustment: int  # NEWS2 score adjustment
    monitoring_frequency_modifier: float  # Multiplier for standard monitoring
    special_considerations: List[str]


@dataclass
class AdmissionRecord:
    """Patient admission and discharge tracking."""
    admission_id: str
    patient_id: str
    admission_date: datetime
    admission_reason: str
    presenting_complaint: str
    initial_assessment: Dict[str, Any]
    discharge_date: Optional[datetime] = None
    discharge_reason: Optional[str] = None
    length_of_stay_hours: Optional[int] = None


class PatientContextManager:
    """Service for managing patient clinical context and medical history."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._lock = asyncio.Lock()
    
    async def create_patient_context(self, patient_id: str, 
                                   context_data: Dict[str, Any]) -> PatientContext:
        """Create initial patient context from admission data."""
        context = PatientContext(
            allergies=context_data.get('allergies', []),
            medications=context_data.get('medications', []),
            comorbidities=context_data.get('comorbidities', []),
            medical_history=context_data.get('medical_history', ''),
            special_instructions=context_data.get('special_instructions', ''),
            age_risk_factors=context_data.get('age_risk_factors', {})
        )
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.INSERT,
            patient_id=patient_id,
            details={
                'operation': 'context_creation',
                'allergies_count': len(context.allergies),
                'medications_count': len(context.medications),
                'comorbidities_count': len(context.comorbidities),
                'has_medical_history': bool(context.medical_history),
                'has_special_instructions': bool(context.special_instructions)
            }
        )
        
        return context
    
    async def update_allergies(self, patient_id: str, allergies: List[str], 
                             update_reason: str = "routine_update") -> PatientContext:
        """Update patient allergy information with audit trail."""
        # This would typically get current context from database
        # For now, create a simple context update
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'allergy_update',
                'new_allergies': allergies,
                'update_reason': update_reason,
                'allergy_count': len(allergies)
            }
        )
        
        # Return updated context (would be from database in real implementation)
        return PatientContext(allergies=allergies)
    
    async def update_medications(self, patient_id: str, medications: List[str],
                               update_reason: str = "medication_review") -> PatientContext:
        """Update patient medication list with audit trail."""
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'medication_update',
                'new_medications': medications,
                'update_reason': update_reason,
                'medication_count': len(medications)
            }
        )
        
        return PatientContext(medications=medications)
    
    async def update_comorbidities(self, patient_id: str, comorbidities: List[str],
                                 update_reason: str = "diagnosis_update") -> PatientContext:
        """Update patient comorbidity list with audit trail."""
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'comorbidity_update',
                'new_comorbidities': comorbidities,
                'update_reason': update_reason,
                'comorbidity_count': len(comorbidities)
            }
        )
        
        return PatientContext(comorbidities=comorbidities)
    
    def calculate_age_risk_profile(self, patient: Patient) -> AgeRiskProfile:
        """Calculate age-based risk factors and adjustments."""
        age = patient.age
        
        # Determine age category and risk adjustments
        if age < 18:
            risk_category = "PEDIATRIC"
            baseline_adjustment = 0  # Pediatric NEWS2 uses different scales
            monitoring_frequency_modifier = 1.5  # More frequent monitoring
            special_considerations = [
                "Pediatric vital sign ranges differ from adults",
                "Consider developmental stage in assessment",
                "Family/caregiver involvement required"
            ]
        elif age >= 65:
            risk_category = "GERIATRIC"
            baseline_adjustment = 0  # Consider lowering alert thresholds
            monitoring_frequency_modifier = 1.2  # Slightly more frequent
            special_considerations = [
                "Higher risk of rapid deterioration",
                "Consider polypharmacy interactions",
                "Assess cognitive status with vital signs",
                "Higher risk of falls and delirium"
            ]
            
            # Additional considerations for very elderly
            if age >= 80:
                monitoring_frequency_modifier = 1.3
                special_considerations.append("Consider frailty assessment")
                
        else:
            risk_category = "ADULT"
            baseline_adjustment = 0
            monitoring_frequency_modifier = 1.0
            special_considerations = ["Standard adult monitoring protocols"]
        
        return AgeRiskProfile(
            age=age,
            risk_category=risk_category,
            baseline_adjustment=baseline_adjustment,
            monitoring_frequency_modifier=monitoring_frequency_modifier,
            special_considerations=special_considerations
        )
    
    def calculate_news2_adjustments(self, patient: Patient, 
                                  context: PatientContext) -> Dict[str, Any]:
        """Calculate context-aware NEWS2 adjustments for special populations."""
        adjustments = {
            'baseline_adjustment': 0,
            'threshold_modifications': {},
            'monitoring_adjustments': {},
            'special_considerations': []
        }
        
        # Age-based adjustments
        age_profile = self.calculate_age_risk_profile(patient)
        adjustments['baseline_adjustment'] += age_profile.baseline_adjustment
        adjustments['monitoring_adjustments']['frequency_modifier'] = age_profile.monitoring_frequency_modifier
        adjustments['special_considerations'].extend(age_profile.special_considerations)
        
        # Comorbidity-based adjustments
        if 'chronic_kidney_disease' in context.comorbidities:
            adjustments['special_considerations'].append("Monitor for fluid overload")
            
        if 'heart_failure' in context.comorbidities:
            adjustments['threshold_modifications']['heart_rate_lower'] = True
            adjustments['special_considerations'].append("Lower heart rate thresholds due to heart failure")
            
        if 'diabetes' in context.comorbidities:
            adjustments['special_considerations'].append("Monitor for diabetic complications")
            
        # Medication-based adjustments
        if any('beta-blocker' in med.lower() for med in context.medications):
            adjustments['threshold_modifications']['heart_rate_masked'] = True
            adjustments['special_considerations'].append("Beta-blockers may mask tachycardia")
            
        if any('steroid' in med.lower() for med in context.medications):
            adjustments['special_considerations'].append("Steroids may mask fever and inflammatory response")
        
        # Allergy considerations
        if context.allergies:
            adjustments['special_considerations'].append(f"Patient has {len(context.allergies)} documented allergies")
        
        return adjustments
    
    async def create_admission_record(self, patient_id: str, 
                                    admission_data: Dict[str, Any]) -> AdmissionRecord:
        """Create admission record with initial assessment."""
        admission_record = AdmissionRecord(
            admission_id=admission_data['admission_id'],
            patient_id=patient_id,
            admission_date=admission_data['admission_date'],
            admission_reason=admission_data['admission_reason'],
            presenting_complaint=admission_data['presenting_complaint'],
            initial_assessment=admission_data.get('initial_assessment', {})
        )
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.INSERT,
            patient_id=patient_id,
            details={
                'operation': 'admission_created',
                'admission_id': admission_record.admission_id,
                'admission_reason': admission_record.admission_reason,
                'presenting_complaint': admission_record.presenting_complaint,
                'admission_timestamp': admission_record.admission_date.isoformat()
            }
        )
        
        return admission_record
    
    async def complete_discharge(self, patient_id: str, admission_id: str,
                               discharge_data: Dict[str, Any]) -> AdmissionRecord:
        """Complete patient discharge and calculate length of stay."""
        # Would typically retrieve admission record from database
        # For now, create a sample completed record
        
        admission_date = discharge_data.get('admission_date', datetime.now(timezone.utc))
        discharge_date = discharge_data.get('discharge_date', datetime.now(timezone.utc))
        
        length_of_stay = (discharge_date - admission_date).total_seconds() / 3600  # Hours
        
        completed_record = AdmissionRecord(
            admission_id=admission_id,
            patient_id=patient_id,
            admission_date=admission_date,
            admission_reason=discharge_data.get('admission_reason', ''),
            presenting_complaint=discharge_data.get('presenting_complaint', ''),
            initial_assessment=discharge_data.get('initial_assessment', {}),
            discharge_date=discharge_date,
            discharge_reason=discharge_data.get('discharge_reason', ''),
            length_of_stay_hours=int(length_of_stay)
        )
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'discharge_completed',
                'admission_id': admission_id,
                'discharge_reason': completed_record.discharge_reason,
                'length_of_stay_hours': completed_record.length_of_stay_hours,
                'discharge_timestamp': discharge_date.isoformat()
            }
        )
        
        return completed_record
    
    async def update_medical_history(self, patient_id: str, medical_history: str,
                                   update_reason: str = "history_review") -> PatientContext:
        """Update patient medical history with audit trail."""
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'medical_history_update',
                'update_reason': update_reason,
                'history_length': len(medical_history),
                'has_content': bool(medical_history.strip())
            }
        )
        
        return PatientContext(medical_history=medical_history)
    
    async def update_special_instructions(self, patient_id: str, instructions: str,
                                        instruction_type: str = "general") -> PatientContext:
        """Update patient special instructions with audit trail."""
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'special_instructions_update',
                'instruction_type': instruction_type,
                'instructions_length': len(instructions),
                'has_content': bool(instructions.strip())
            }
        )
        
        return PatientContext(special_instructions=instructions)
    
    def validate_context_completeness(self, context: PatientContext) -> Dict[str, Any]:
        """Validate completeness of patient context for clinical decision support."""
        completeness_score = 0.0
        missing_elements = []
        recommendations = []
        
        # Check critical elements
        if context.allergies:
            completeness_score += 0.2
        else:
            missing_elements.append("allergies")
            recommendations.append("Document patient allergies")
        
        if context.medications:
            completeness_score += 0.2
        else:
            missing_elements.append("current_medications")
            recommendations.append("Review current medications")
        
        if context.comorbidities:
            completeness_score += 0.2
        else:
            missing_elements.append("comorbidities")
            recommendations.append("Document relevant comorbidities")
        
        if context.medical_history.strip():
            completeness_score += 0.2
        else:
            missing_elements.append("medical_history")
            recommendations.append("Obtain relevant medical history")
        
        if context.special_instructions.strip():
            completeness_score += 0.2
        else:
            missing_elements.append("special_instructions")
            recommendations.append("Add special care instructions if applicable")
        
        return {
            'completeness_score': completeness_score,
            'missing_elements': missing_elements,
            'recommendations': recommendations,
            'is_sufficient_for_clinical_decision': completeness_score >= 0.6
        }