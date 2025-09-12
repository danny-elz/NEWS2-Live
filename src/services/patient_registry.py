import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..models.patient import Patient
from ..models.patient_state import PatientState, PatientContext, TrendingAnalysis
from ..models.patient_state import PatientStateError, PatientTransferError, ConcurrentUpdateError
from ..services.audit import AuditLogger, AuditOperation
from ..services.patient_cache import PatientDataCache


class PatientRegistry:
    """Patient registry service with CRUD operations and state management."""
    
    def __init__(self, audit_logger: AuditLogger, cache: Optional[PatientDataCache] = None):
        self.audit_logger = audit_logger
        self.cache = cache
        self._lock = asyncio.Lock()
    
    async def register_patient(self, patient: Patient, context: Optional[PatientContext] = None) -> PatientState:
        """Register new patient with initial state."""
        async with self._lock:
            patient_hash = self._hash_patient_id(patient.patient_id)
            
            # Check if patient already exists
            existing_state = await self.get_patient_state(patient.patient_id)
            if existing_state:
                raise PatientStateError(f"Patient {patient_hash} already registered")
            
            # Create initial patient state
            patient_state = PatientState.from_patient(patient)
            
            # Set context if provided
            if context:
                patient_state.context = context
            
            # Generate audit trail entry
            await self.audit_logger.log_operation(
                operation=AuditOperation.PATIENT_REGISTRATION,
                patient_id=patient.patient_id,
                details={
                    'ward_id': patient.ward_id,
                    'bed_number': patient.bed_number,
                    'clinical_flags': patient_state.clinical_flags,
                    'assigned_nurse_id': patient.assigned_nurse_id
                }
            )
            
            # Cache patient state for fast lookup
            if self.cache:
                await self.cache.put(patient_state)
            
            return patient_state
    
    async def get_patient_state(self, patient_id: str) -> Optional[PatientState]:
        """Retrieve patient state by patient ID."""
        # Try cache first
        if self.cache:
            cached_state = await self.cache.get(patient_id)
            if cached_state:
                return cached_state
        
        # If not in cache, would query database here
        # For now, return None as we don't have database implementation
        return None
    
    async def update_patient_state(self, patient_id: str, updates: Dict[str, Any], 
                                 expected_version: Optional[int] = None) -> PatientState:
        """Update patient state with optimistic locking."""
        async with self._lock:
            current_state = await self.get_patient_state(patient_id)
            if not current_state:
                raise PatientStateError(f"Patient {self._hash_patient_id(patient_id)} not found")
            
            # Optimistic locking check
            if expected_version is not None and current_state.state_version != expected_version:
                raise ConcurrentUpdateError(
                    f"State version mismatch for patient {self._hash_patient_id(patient_id)}. "
                    f"Expected: {expected_version}, Current: {current_state.state_version}"
                )
            
            # Apply updates
            old_state_dict = current_state.to_dict()
            
            # Update allowed fields
            if 'current_ward_id' in updates:
                current_state.current_ward_id = updates['current_ward_id']
            if 'bed_number' in updates:
                current_state.bed_number = updates['bed_number']
            if 'clinical_flags' in updates:
                current_state.clinical_flags.update(updates['clinical_flags'])
            if 'assigned_nurse_id' in updates:
                current_state.assigned_nurse_id = updates['assigned_nurse_id']
            if 'context' in updates:
                # Update context fields
                context_updates = updates['context']
                if 'allergies' in context_updates:
                    current_state.context.allergies = context_updates['allergies']
                if 'medications' in context_updates:
                    current_state.context.medications = context_updates['medications']
                if 'comorbidities' in context_updates:
                    current_state.context.comorbidities = context_updates['comorbidities']
                if 'medical_history' in context_updates:
                    current_state.context.medical_history = context_updates['medical_history']
                if 'special_instructions' in context_updates:
                    current_state.context.special_instructions = context_updates['special_instructions']
            
            # Increment version and update timestamp
            current_state.state_version += 1
            current_state.last_updated = datetime.now(timezone.utc)
            
            # Generate audit trail
            await self.audit_logger.log_operation(
                operation=AuditOperation.PATIENT_UPDATE,
                patient_id=patient_id,
                details={
                    'old_state': old_state_dict,
                    'updates': updates,
                    'new_version': current_state.state_version
                }
            )
            
            # Update cache
            if self.cache:
                await self.cache.put(current_state)
            
            return current_state
    
    async def transfer_patient(self, patient_id: str, new_ward_id: str, 
                             transfer_reason: str, new_bed_number: Optional[str] = None,
                             new_nurse_id: Optional[str] = None) -> PatientState:
        """Transfer patient to new ward with audit trail."""
        current_state = await self.get_patient_state(patient_id)
        if not current_state:
            raise PatientTransferError(f"Patient {self._hash_patient_id(patient_id)} not found")
        
        old_ward_id = current_state.current_ward_id
        
        # Validate transfer eligibility
        if current_state.clinical_flags.get('do_not_escalate', False):
            raise PatientTransferError(
                f"Patient {self._hash_patient_id(patient_id)} has do_not_escalate flag set"
            )
        
        # Prepare transfer updates
        transfer_updates = {
            'current_ward_id': new_ward_id,
            'bed_number': new_bed_number or current_state.bed_number,
            'assigned_nurse_id': new_nurse_id or current_state.assigned_nurse_id
        }
        
        # Update state with transfer
        updated_state = await self.update_patient_state(
            patient_id, 
            transfer_updates,
            expected_version=current_state.state_version
        )
        
        # Update transfer timestamp
        updated_state.last_transfer_date = datetime.now(timezone.utc)
        
        # Generate transfer audit entry
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_TRANSFER,
            patient_id=patient_id,
            details={
                'old_ward_id': old_ward_id,
                'new_ward_id': new_ward_id,
                'transfer_reason': transfer_reason,
                'old_bed_number': current_state.bed_number,
                'new_bed_number': new_bed_number,
                'transfer_timestamp': updated_state.last_transfer_date.isoformat()
            }
        )
        
        # TODO: Notify receiving ward staff
        await self._notify_ward_transfer(patient_id, old_ward_id, new_ward_id, transfer_reason)
        
        return updated_state
    
    async def assign_nurse(self, patient_id: str, nurse_id: str, 
                          assignment_reason: str = "routine_assignment") -> PatientState:
        """Assign or reassign nurse to patient with audit trail."""
        current_state = await self.get_patient_state(patient_id)
        if not current_state:
            raise PatientStateError(f"Patient {self._hash_patient_id(patient_id)} not found")
        
        old_nurse_id = current_state.assigned_nurse_id
        
        # Update nurse assignment
        updated_state = await self.update_patient_state(
            patient_id,
            {'assigned_nurse_id': nurse_id},
            expected_version=current_state.state_version
        )
        
        # Generate nurse assignment audit entry
        await self.audit_logger.log_operation(
            operation=AuditOperation.NURSE_ASSIGNMENT,
            patient_id=patient_id,
            details={
                'old_nurse_id': old_nurse_id,
                'new_nurse_id': nurse_id,
                'assignment_reason': assignment_reason,
                'assignment_timestamp': updated_state.last_updated.isoformat()
            }
        )
        
        return updated_state
    
    async def update_clinical_flags(self, patient_id: str, 
                                   flag_updates: Dict[str, bool]) -> PatientState:
        """Update clinical flags with audit trail."""
        current_state = await self.get_patient_state(patient_id)
        if not current_state:
            raise PatientStateError(f"Patient {self._hash_patient_id(patient_id)} not found")
        
        # Update clinical flags
        updated_state = await self.update_patient_state(
            patient_id,
            {'clinical_flags': flag_updates},
            expected_version=current_state.state_version
        )
        
        # Generate clinical flags update audit entry
        await self.audit_logger.log_operation(
            operation=AuditOperation.CLINICAL_FLAGS_UPDATE,
            patient_id=patient_id,
            details={
                'old_flags': current_state.clinical_flags,
                'flag_updates': flag_updates,
                'new_flags': updated_state.clinical_flags
            }
        )
        
        return updated_state
    
    async def get_patients_by_ward(self, ward_id: str) -> List[PatientState]:
        """Get all patients in a specific ward."""
        # TODO: Implement database query for ward patients
        # For now, return empty list
        return []
    
    async def get_patients_by_nurse(self, nurse_id: str) -> List[PatientState]:
        """Get all patients assigned to a specific nurse."""
        # TODO: Implement database query for nurse patients
        # For now, return empty list
        return []
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """Generate privacy-safe hash of patient ID for logging."""
        return hashlib.sha256(f"patient_{patient_id}".encode()).hexdigest()[:8]
    
    async def _notify_ward_transfer(self, patient_id: str, old_ward_id: str, 
                                   new_ward_id: str, reason: str) -> None:
        """Send transfer notifications to ward staff."""
        # TODO: Implement notification system
        patient_hash = self._hash_patient_id(patient_id)
        print(f"TRANSFER NOTIFICATION: Patient {patient_hash} transferred from {old_ward_id} to {new_ward_id} - {reason}")
    
    async def discharge_patient(self, patient_id: str, discharge_reason: str) -> PatientState:
        """Mark patient as discharged while preserving state for audit."""
        current_state = await self.get_patient_state(patient_id)
        if not current_state:
            raise PatientStateError(f"Patient {self._hash_patient_id(patient_id)} not found")
        
        # Update clinical flags to indicate discharge
        discharge_updates = {
            'clinical_flags': {
                **current_state.clinical_flags,
                'is_discharged': True
            }
        }
        
        updated_state = await self.update_patient_state(
            patient_id,
            discharge_updates,
            expected_version=current_state.state_version
        )
        
        # Generate discharge audit entry
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_DISCHARGE,
            patient_id=patient_id,
            details={
                'discharge_reason': discharge_reason,
                'discharge_timestamp': updated_state.last_updated.isoformat(),
                'final_ward_id': current_state.current_ward_id,
                'length_of_stay_days': (updated_state.last_updated - current_state.admission_date).days
            }
        )
        
        return updated_state