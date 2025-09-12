import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from ..models.patient import Patient
from ..models.patient_state import PatientState, PatientTransferError, PatientStateError
from ..models.vital_signs import VitalSigns
from ..services.audit import AuditLogger, AuditOperation
from ..services.patient_registry import PatientRegistry
from ..services.concurrent_update_manager import ConcurrentUpdateManager


class TransferStatus(Enum):
    INITIATED = "initiated"
    VALIDATED = "validated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class TransferPriority(Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


@dataclass
class TransferValidation:
    """Result of transfer validation checks."""
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]
    required_approvals: List[str]
    estimated_duration_minutes: int


@dataclass
class TransferRequest:
    """Ward transfer request with complete workflow information."""
    transfer_id: str
    patient_id: str
    source_ward_id: str
    destination_ward_id: str
    source_bed_number: Optional[str]
    destination_bed_number: Optional[str]
    transfer_reason: str
    priority: TransferPriority
    requested_by: str
    requested_at: datetime
    scheduled_time: Optional[datetime]
    status: TransferStatus
    validation_result: Optional[TransferValidation] = None
    completion_time: Optional[datetime] = None
    rollback_reason: Optional[str] = None


@dataclass
class TransferNotification:
    """Notification for ward staff about transfers."""
    notification_id: str
    transfer_id: str
    ward_id: str
    notification_type: str  # "incoming", "outgoing", "status_update"
    recipient_role: str  # "nurse", "doctor", "admin"
    message: str
    priority: TransferPriority
    sent_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class PatientTransferService:
    """Service for managing patient ward transfers with comprehensive workflow orchestration."""
    
    def __init__(self, audit_logger: AuditLogger, 
                 patient_registry: PatientRegistry,
                 concurrency_manager: ConcurrentUpdateManager):
        self.audit_logger = audit_logger
        self.patient_registry = patient_registry
        self.concurrency_manager = concurrency_manager
        self._active_transfers: Dict[str, TransferRequest] = {}
        self._transfer_lock = asyncio.Lock()
    
    async def initiate_transfer(self, patient_id: str, destination_ward_id: str,
                              transfer_reason: str, requested_by: str,
                              priority: TransferPriority = TransferPriority.ROUTINE,
                              scheduled_time: Optional[datetime] = None,
                              destination_bed_number: Optional[str] = None) -> TransferRequest:
        """Initiate patient transfer with comprehensive validation."""
        
        # Get current patient state
        current_state = await self.patient_registry.get_patient_state(patient_id)
        if not current_state:
            raise PatientTransferError(f"Patient {patient_id} not found")
        
        # Create transfer request
        transfer_request = TransferRequest(
            transfer_id=str(uuid4()),
            patient_id=patient_id,
            source_ward_id=current_state.current_ward_id,
            destination_ward_id=destination_ward_id,
            source_bed_number=current_state.bed_number,
            destination_bed_number=destination_bed_number,
            transfer_reason=transfer_reason,
            priority=priority,
            requested_by=requested_by,
            requested_at=datetime.now(timezone.utc),
            scheduled_time=scheduled_time,
            status=TransferStatus.INITIATED
        )
        
        # Store transfer request
        async with self._transfer_lock:
            self._active_transfers[transfer_request.transfer_id] = transfer_request
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_TRANSFER,
            patient_id=patient_id,
            details={
                'operation': 'transfer_initiated',
                'transfer_id': transfer_request.transfer_id,
                'source_ward': current_state.current_ward_id,
                'destination_ward': destination_ward_id,
                'transfer_reason': transfer_reason,
                'priority': priority.value,
                'requested_by': requested_by
            }
        )
        
        # Send notifications to both wards
        await self._send_transfer_notifications(transfer_request, "initiated")
        
        return transfer_request
    
    async def validate_transfer(self, transfer_id: str) -> TransferValidation:
        """Validate transfer request with comprehensive checks."""
        transfer_request = await self._get_transfer_request(transfer_id)
        
        validation_errors = []
        warnings = []
        required_approvals = []
        
        # Get current patient state
        current_state = await self.patient_registry.get_patient_state(transfer_request.patient_id)
        if not current_state:
            validation_errors.append("Patient state not found")
            return TransferValidation(
                is_valid=False,
                validation_errors=validation_errors,
                warnings=warnings,
                required_approvals=required_approvals,
                estimated_duration_minutes=0
            )
        
        # Validate patient eligibility
        if current_state.clinical_flags.get('do_not_escalate', False):
            validation_errors.append("Patient has do_not_escalate flag - requires senior approval")
            required_approvals.append("senior_doctor")
        
        # Check destination ward capacity
        destination_capacity_ok = await self._check_ward_capacity(transfer_request.destination_ward_id)
        if not destination_capacity_ok:
            validation_errors.append("Destination ward at capacity")
        
        # Check if destination bed is available
        if transfer_request.destination_bed_number:
            bed_available = await self._check_bed_availability(
                transfer_request.destination_ward_id, 
                transfer_request.destination_bed_number
            )
            if not bed_available:
                validation_errors.append(f"Destination bed {transfer_request.destination_bed_number} not available")
        
        # Clinical warnings
        if current_state.trending_data.deterioration_risk == "HIGH":
            warnings.append("Patient has HIGH deterioration risk - consider delaying transfer")
        
        if current_state.trending_data.current_score >= 7:
            warnings.append("Patient has critical NEWS2 score - ensure appropriate transfer team")
            required_approvals.append("critical_care_team")
        
        # Estimate transfer duration based on priority and clinical status
        estimated_duration = self._estimate_transfer_duration(transfer_request, current_state)
        
        validation_result = TransferValidation(
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            warnings=warnings,
            required_approvals=required_approvals,
            estimated_duration_minutes=estimated_duration
        )
        
        # Update transfer request with validation
        transfer_request.validation_result = validation_result
        transfer_request.status = TransferStatus.VALIDATED if validation_result.is_valid else TransferStatus.FAILED
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_TRANSFER,
            patient_id=transfer_request.patient_id,
            details={
                'operation': 'transfer_validated',
                'transfer_id': transfer_id,
                'is_valid': validation_result.is_valid,
                'validation_errors': validation_errors,
                'warnings': warnings,
                'required_approvals': required_approvals
            }
        )
        
        return validation_result
    
    async def execute_transfer(self, transfer_id: str, 
                             executing_user: str) -> TransferRequest:
        """Execute validated transfer with cross-ward state synchronization."""
        transfer_request = await self._get_transfer_request(transfer_id)
        
        if transfer_request.status != TransferStatus.VALIDATED:
            raise PatientTransferError(f"Transfer {transfer_id} not validated - current status: {transfer_request.status.value}")
        
        # Use distributed locking for transfer execution
        async with self.concurrency_manager.distributed_lock(
            transfer_request.patient_id, 
            "ward_transfer"
        ):
            try:
                # Update status to in progress
                transfer_request.status = TransferStatus.IN_PROGRESS
                await self._send_transfer_notifications(transfer_request, "in_progress")
                
                # Execute transfer in patient registry
                updated_state = await self.patient_registry.transfer_patient(
                    transfer_request.patient_id,
                    transfer_request.destination_ward_id,
                    transfer_request.transfer_reason,
                    transfer_request.destination_bed_number,
                    None  # Keep existing nurse assignment initially
                )
                
                # Synchronize state across both wards
                await self._synchronize_cross_ward_state(transfer_request, updated_state)
                
                # Complete transfer
                transfer_request.status = TransferStatus.COMPLETED
                transfer_request.completion_time = datetime.now(timezone.utc)
                
                await self.audit_logger.log_operation(
                    operation=AuditOperation.PATIENT_TRANSFER,
                    patient_id=transfer_request.patient_id,
                    details={
                        'operation': 'transfer_completed',
                        'transfer_id': transfer_id,
                        'executing_user': executing_user,
                        'completion_time': transfer_request.completion_time.isoformat(),
                        'final_ward': transfer_request.destination_ward_id,
                        'final_bed': transfer_request.destination_bed_number
                    }
                )
                
                # Send completion notifications
                await self._send_transfer_notifications(transfer_request, "completed")
                
                return transfer_request
                
            except Exception as e:
                # Transfer failed - attempt rollback
                await self._rollback_transfer(transfer_request, str(e))
                raise PatientTransferError(f"Transfer execution failed: {str(e)}")
    
    async def _rollback_transfer(self, transfer_request: TransferRequest, 
                               failure_reason: str):
        """Implement rollback mechanism for failed transfer operations."""
        try:
            # Attempt to restore original state
            original_state = await self.patient_registry.get_patient_state(transfer_request.patient_id)
            if original_state and original_state.current_ward_id != transfer_request.source_ward_id:
                # Restore to original ward
                await self.patient_registry.transfer_patient(
                    transfer_request.patient_id,
                    transfer_request.source_ward_id,
                    f"Rollback due to failed transfer: {failure_reason}",
                    transfer_request.source_bed_number
                )
            
            transfer_request.status = TransferStatus.ROLLED_BACK
            transfer_request.rollback_reason = failure_reason
            
            await self.audit_logger.log_operation(
                operation=AuditOperation.PATIENT_TRANSFER,
                patient_id=transfer_request.patient_id,
                details={
                    'operation': 'transfer_rolled_back',
                    'transfer_id': transfer_request.transfer_id,
                    'failure_reason': failure_reason,
                    'rollback_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Notify about rollback
            await self._send_transfer_notifications(transfer_request, "rolled_back")
            
        except Exception as rollback_error:
            # Rollback failed - critical situation
            await self.audit_logger.log_operation(
                operation=AuditOperation.PATIENT_TRANSFER,
                patient_id=transfer_request.patient_id,
                details={
                    'operation': 'rollback_failed',
                    'transfer_id': transfer_request.transfer_id,
                    'original_failure': failure_reason,
                    'rollback_error': str(rollback_error),
                    'requires_manual_intervention': True
                }
            )
    
    async def _synchronize_cross_ward_state(self, transfer_request: TransferRequest,
                                          updated_state: PatientState):
        """Implement cross-ward state synchronization during transfers."""
        
        # Notify source ward of departure
        source_ward_sync = {
            'operation': 'patient_departed',
            'patient_id': transfer_request.patient_id,
            'destination_ward': transfer_request.destination_ward_id,
            'transfer_time': datetime.now(timezone.utc).isoformat(),
            'final_state_version': updated_state.state_version
        }
        
        # Notify destination ward of arrival
        destination_ward_sync = {
            'operation': 'patient_arrived',
            'patient_id': transfer_request.patient_id,
            'source_ward': transfer_request.source_ward_id,
            'transfer_time': datetime.now(timezone.utc).isoformat(),
            'patient_state': updated_state.to_dict(),
            'clinical_flags': updated_state.clinical_flags,
            'trending_data': {
                'current_score': updated_state.trending_data.current_score,
                'deterioration_risk': updated_state.trending_data.deterioration_risk,
                'early_warnings': updated_state.trending_data.early_warning_indicators
            }
        }
        
        # In real implementation, would send to ward management systems
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_TRANSFER,
            patient_id=transfer_request.patient_id,
            details={
                'operation': 'cross_ward_sync',
                'transfer_id': transfer_request.transfer_id,
                'source_ward_sync': source_ward_sync,
                'destination_ward_sync': destination_ward_sync
            }
        )
    
    async def _send_transfer_notifications(self, transfer_request: TransferRequest, 
                                         notification_type: str):
        """Send transfer notifications to receiving ward staff."""
        notifications = []
        
        # Notification to source ward
        source_notification = TransferNotification(
            notification_id=str(uuid4()),
            transfer_id=transfer_request.transfer_id,
            ward_id=transfer_request.source_ward_id,
            notification_type=f"outgoing_{notification_type}",
            recipient_role="nurse",
            message=self._generate_notification_message(transfer_request, notification_type, "outgoing"),
            priority=transfer_request.priority,
            sent_at=datetime.now(timezone.utc)
        )
        
        # Notification to destination ward  
        destination_notification = TransferNotification(
            notification_id=str(uuid4()),
            transfer_id=transfer_request.transfer_id,
            ward_id=transfer_request.destination_ward_id,
            notification_type=f"incoming_{notification_type}",
            recipient_role="nurse", 
            message=self._generate_notification_message(transfer_request, notification_type, "incoming"),
            priority=transfer_request.priority,
            sent_at=datetime.now(timezone.utc)
        )
        
        notifications.extend([source_notification, destination_notification])
        
        # In real implementation, would send via notification system
        for notification in notifications:
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=transfer_request.patient_id,
                details={
                    'operation': 'transfer_notification_sent',
                    'notification_id': notification.notification_id,
                    'ward_id': notification.ward_id,
                    'notification_type': notification.notification_type,
                    'priority': notification.priority.value
                }
            )
    
    def _generate_notification_message(self, transfer_request: TransferRequest,
                                     notification_type: str, direction: str) -> str:
        """Generate appropriate notification message."""
        patient_hash = self._hash_patient_id(transfer_request.patient_id)
        
        if direction == "outgoing":
            if notification_type == "initiated":
                return f"Transfer initiated: Patient {patient_hash} scheduled to transfer to {transfer_request.destination_ward_id}"
            elif notification_type == "completed":
                return f"Transfer completed: Patient {patient_hash} has been transferred to {transfer_request.destination_ward_id}"
            elif notification_type == "rolled_back":
                return f"Transfer cancelled: Patient {patient_hash} transfer to {transfer_request.destination_ward_id} was rolled back"
        else:  # incoming
            if notification_type == "initiated":
                return f"Incoming transfer: Patient {patient_hash} scheduled to arrive from {transfer_request.source_ward_id}"
            elif notification_type == "completed":
                return f"Patient arrived: Patient {patient_hash} has been transferred from {transfer_request.source_ward_id}"
            elif notification_type == "rolled_back":
                return f"Transfer cancelled: Expected patient {patient_hash} transfer was cancelled"
        
        return f"Transfer {notification_type}: Patient {patient_hash}"
    
    async def _get_transfer_request(self, transfer_id: str) -> TransferRequest:
        """Get transfer request by ID."""
        async with self._transfer_lock:
            transfer_request = self._active_transfers.get(transfer_id)
            if not transfer_request:
                raise PatientTransferError(f"Transfer {transfer_id} not found")
            return transfer_request
    
    async def _check_ward_capacity(self, ward_id: str) -> bool:
        """Check if ward has capacity for new patient."""
        # In real implementation, would query ward management system
        # For now, assume capacity is available
        return True
    
    async def _check_bed_availability(self, ward_id: str, bed_number: str) -> bool:
        """Check if specific bed is available."""
        # In real implementation, would query bed management system
        # For now, assume bed is available
        return True
    
    def _estimate_transfer_duration(self, transfer_request: TransferRequest,
                                  patient_state: PatientState) -> int:
        """Estimate transfer duration in minutes."""
        base_duration = 30  # Base 30 minutes
        
        # Add time based on priority
        if transfer_request.priority == TransferPriority.ROUTINE:
            base_duration += 15
        elif transfer_request.priority == TransferPriority.URGENT:
            base_duration += 5
        # Emergency transfers don't add time
        
        # Add time based on clinical complexity
        if patient_state.trending_data.current_score >= 7:
            base_duration += 20  # Critical patients need more prep time
        elif patient_state.trending_data.current_score >= 5:
            base_duration += 10
        
        # Add time for special conditions
        if patient_state.clinical_flags.get('is_copd_patient', False):
            base_duration += 5
        if patient_state.clinical_flags.get('oxygen_dependent', False):
            base_duration += 10
        
        return base_duration
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """Generate privacy-safe hash of patient ID."""
        import hashlib
        return hashlib.sha256(f"patient_{patient_id}".encode()).hexdigest()[:8]
    
    async def get_active_transfers(self, ward_id: Optional[str] = None) -> List[TransferRequest]:
        """Get active transfers, optionally filtered by ward."""
        async with self._transfer_lock:
            transfers = list(self._active_transfers.values())
            
            if ward_id:
                transfers = [
                    t for t in transfers 
                    if t.source_ward_id == ward_id or t.destination_ward_id == ward_id
                ]
            
            return transfers
    
    async def cancel_transfer(self, transfer_id: str, cancellation_reason: str,
                            cancelled_by: str) -> TransferRequest:
        """Cancel pending transfer."""
        transfer_request = await self._get_transfer_request(transfer_id)
        
        if transfer_request.status in [TransferStatus.COMPLETED, TransferStatus.ROLLED_BACK]:
            raise PatientTransferError(f"Cannot cancel transfer in status {transfer_request.status.value}")
        
        transfer_request.status = TransferStatus.FAILED
        transfer_request.rollback_reason = f"Cancelled by {cancelled_by}: {cancellation_reason}"
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.PATIENT_TRANSFER,
            patient_id=transfer_request.patient_id,
            details={
                'operation': 'transfer_cancelled',
                'transfer_id': transfer_id,
                'cancellation_reason': cancellation_reason,
                'cancelled_by': cancelled_by,
                'original_status': transfer_request.status.value
            }
        )
        
        await self._send_transfer_notifications(transfer_request, "cancelled")
        
        return transfer_request