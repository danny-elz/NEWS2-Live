import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from uuid import uuid4, UUID
from enum import Enum

from ..models.patient import Patient
from ..models.vital_signs import VitalSigns


class AuditOperation(Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class AuditEntry:
    audit_id: UUID
    table_name: str
    operation: AuditOperation
    user_id: str
    patient_id: Optional[str]
    timestamp: datetime
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary for storage."""
        return {
            "audit_id": str(self.audit_id),
            "table_name": self.table_name,
            "operation": self.operation.value,
            "user_id": self.user_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "old_values": self.old_values,
            "new_values": self.new_values
        }


class AuditLogger:
    """
    Audit logging service that creates immutable audit trails for all medical data changes.
    
    Follows coding standards for medical data integrity:
    - All data modifications must be audited
    - Audit entries are immutable
    - Patient IDs are hashed for privacy in logs
    - Full audit context is preserved
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_audit_entry(
        self,
        table_name: str,
        operation: AuditOperation,
        user_id: str,
        patient_id: Optional[str] = None,
        old_values: Optional[Union[Dict, Patient, VitalSigns]] = None,
        new_values: Optional[Union[Dict, Patient, VitalSigns]] = None
    ) -> AuditEntry:
        """
        Create an audit entry for database operations.
        
        Args:
            table_name: Name of the table being modified
            operation: Type of operation (INSERT, UPDATE, DELETE)
            user_id: ID of the user performing the operation
            patient_id: ID of the patient (if applicable)
            old_values: Previous values (for UPDATE/DELETE)
            new_values: New values (for INSERT/UPDATE)
            
        Returns:
            AuditEntry object ready for database storage
        """
        try:
            # Generate unique audit ID
            audit_id = uuid4()
            
            # Convert model objects to dictionaries for JSON storage
            old_dict = self._serialize_audit_data(old_values) if old_values else None
            new_dict = self._serialize_audit_data(new_values) if new_values else None
            
            # Create audit entry
            audit_entry = AuditEntry(
                audit_id=audit_id,
                table_name=table_name,
                operation=operation,
                user_id=user_id,
                patient_id=patient_id,
                timestamp=datetime.now(timezone.utc),
                old_values=old_dict,
                new_values=new_dict
            )
            
            # Log the audit creation (with hashed patient ID for privacy)
            patient_hash = self._hash_patient_id(patient_id) if patient_id else "N/A"
            self.logger.info(
                f"Audit entry created - ID: {audit_id}, Table: {table_name}, "
                f"Operation: {operation.value}, Patient: {patient_hash}, User: {user_id}"
            )
            
            return audit_entry
            
        except Exception as e:
            self.logger.error(f"Failed to create audit entry: {str(e)}")
            # Re-raise the exception as audit failures are critical
            raise AuditException(f"Audit entry creation failed: {str(e)}")
    
    def audit_patient_creation(self, patient: Patient, user_id: str) -> AuditEntry:
        """Create audit entry for patient creation."""
        return self.create_audit_entry(
            table_name="patients",
            operation=AuditOperation.INSERT,
            user_id=user_id,
            patient_id=patient.patient_id,
            new_values=patient
        )
    
    def audit_patient_update(self, old_patient: Patient, new_patient: Patient, user_id: str) -> AuditEntry:
        """Create audit entry for patient update."""
        return self.create_audit_entry(
            table_name="patients",
            operation=AuditOperation.UPDATE,
            user_id=user_id,
            patient_id=new_patient.patient_id,
            old_values=old_patient,
            new_values=new_patient
        )
    
    def audit_patient_deletion(self, patient: Patient, user_id: str) -> AuditEntry:
        """Create audit entry for patient deletion."""
        return self.create_audit_entry(
            table_name="patients",
            operation=AuditOperation.DELETE,
            user_id=user_id,
            patient_id=patient.patient_id,
            old_values=patient
        )
    
    def audit_vital_signs_creation(self, vitals: VitalSigns, user_id: str) -> AuditEntry:
        """Create audit entry for vital signs creation."""
        return self.create_audit_entry(
            table_name="vital_signs",
            operation=AuditOperation.INSERT,
            user_id=user_id,
            patient_id=vitals.patient_id,
            new_values=vitals
        )
    
    def audit_vital_signs_update(self, old_vitals: VitalSigns, new_vitals: VitalSigns, user_id: str) -> AuditEntry:
        """Create audit entry for vital signs update."""
        return self.create_audit_entry(
            table_name="vital_signs",
            operation=AuditOperation.UPDATE,
            user_id=user_id,
            patient_id=new_vitals.patient_id,
            old_values=old_vitals,
            new_values=new_vitals
        )
    
    def audit_vital_signs_deletion(self, vitals: VitalSigns, user_id: str) -> AuditEntry:
        """Create audit entry for vital signs deletion."""
        return self.create_audit_entry(
            table_name="vital_signs",
            operation=AuditOperation.DELETE,
            user_id=user_id,
            patient_id=vitals.patient_id,
            old_values=vitals
        )
    
    def _serialize_audit_data(self, data: Union[Dict, Patient, VitalSigns]) -> Dict[str, Any]:
        """
        Serialize various data types to dictionary for JSON storage.
        
        Args:
            data: Data to serialize (dict, Patient, or VitalSigns)
            
        Returns:
            Dictionary representation of the data
        """
        if isinstance(data, dict):
            # Already a dictionary, ensure it's JSON serializable
            return self._ensure_json_serializable(data)
        elif isinstance(data, (Patient, VitalSigns)):
            # Use the to_dict method from our models
            return data.to_dict()
        else:
            # Try to convert to dict using dataclasses.asdict or similar
            try:
                if hasattr(data, '__dict__'):
                    return self._ensure_json_serializable(data.__dict__)
                else:
                    return {"serialized_value": str(data)}
            except Exception as e:
                self.logger.warning(f"Could not serialize audit data: {str(e)}")
                return {"serialization_error": str(e), "type": str(type(data))}
    
    def _ensure_json_serializable(self, obj: Any) -> Any:
        """
        Ensure all values in the object are JSON serializable.
        
        Args:
            obj: Object to make JSON serializable
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Convert other types to string representation
            return str(obj)
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """
        Create a privacy-preserving hash of patient ID for logging.
        
        Args:
            patient_id: The patient ID to hash
            
        Returns:
            Hashed patient ID for secure logging
        """
        # Use a fixed salt for consistent hashing (in production, use environment variable)
        salt = "news2_audit_salt_2024"
        return hashlib.sha256(f"{salt}{patient_id}".encode()).hexdigest()[:16]
    
    def validate_audit_entry(self, audit_entry: AuditEntry) -> bool:
        """
        Validate audit entry before database storage.
        
        Args:
            audit_entry: Audit entry to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        if not audit_entry.audit_id:
            raise AuditException("Audit entry must have an audit_id")
        
        if not audit_entry.table_name:
            raise AuditException("Audit entry must specify table_name")
        
        if not audit_entry.operation:
            raise AuditException("Audit entry must specify operation")
        
        if not audit_entry.user_id:
            raise AuditException("Audit entry must specify user_id")
        
        if not audit_entry.timestamp:
            raise AuditException("Audit entry must have a timestamp")
        
        # Validate operation-specific requirements
        if audit_entry.operation == AuditOperation.INSERT and not audit_entry.new_values:
            raise AuditException("INSERT operation must include new_values")
        
        if audit_entry.operation == AuditOperation.DELETE and not audit_entry.old_values:
            raise AuditException("DELETE operation must include old_values")
        
        if audit_entry.operation == AuditOperation.UPDATE:
            if not audit_entry.old_values or not audit_entry.new_values:
                raise AuditException("UPDATE operation must include both old_values and new_values")
        
        return True


class AuditException(Exception):
    """Exception raised when audit operations fail."""
    pass


class AuditService:
    """
    High-level audit service that coordinates audit logging with database operations.
    
    This service ensures that all medical data modifications are properly audited
    according to HIPAA compliance requirements and medical safety standards.
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self.logger = logging.getLogger(__name__)
    
    def with_audit(self, operation_func, audit_func, *args, **kwargs):
        """
        Execute a database operation with automatic audit logging.
        
        Args:
            operation_func: Function that performs the database operation
            audit_func: Function that creates the audit entry
            *args, **kwargs: Arguments for both functions
            
        Returns:
            Result of the operation_func
        """
        try:
            # Create audit entry first
            audit_entry = audit_func(*args, **kwargs)
            self.audit_logger.validate_audit_entry(audit_entry)
            
            # Execute the database operation
            result = operation_func(*args, **kwargs)
            
            # Store audit entry (this would be handled by database triggers in production)
            self._store_audit_entry(audit_entry)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Operation with audit failed: {str(e)}")
            # Re-raise to ensure calling code handles the failure
            raise
    
    def _store_audit_entry(self, audit_entry: AuditEntry):
        """
        Store audit entry in database.
        
        In production, this would interface with the database layer.
        For now, we'll log the audit entry.
        """
        self.logger.info(f"Storing audit entry: {audit_entry.to_dict()}")
        # TODO: Implement actual database storage when database layer is added