import logging
import hashlib
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..models.patient import Patient
from ..models.vital_signs import VitalSigns, ConsciousnessLevel


class ValidationErrorCode(Enum):
    RESPIRATORY_RATE_OUT_OF_RANGE = "RR_OUT_OF_RANGE"
    HEART_RATE_OUT_OF_RANGE = "HR_OUT_OF_RANGE"
    SPO2_OUT_OF_RANGE = "SPO2_OUT_OF_RANGE"
    TEMPERATURE_OUT_OF_RANGE = "TEMP_OUT_OF_RANGE"
    SYSTOLIC_BP_OUT_OF_RANGE = "SBP_OUT_OF_RANGE"
    CONSCIOUSNESS_INVALID = "CONSCIOUSNESS_INVALID"
    MISSING_REQUIRED_FIELD = "MISSING_FIELD"
    INVALID_DATA_TYPE = "INVALID_TYPE"
    PARTIAL_VITALS = "PARTIAL_VITALS"
    NULL_VALUE = "NULL_VALUE"
    CONFIDENCE_OUT_OF_RANGE = "CONFIDENCE_OUT_OF_RANGE"


@dataclass
class ValidationError:
    field: str
    code: ValidationErrorCode
    message: str
    received_value: Optional[Union[str, int, float]] = None
    valid_range: Optional[str] = None


class VitalSignsValidator:
    """
    Validates vital signs data according to clinical ranges and safety requirements.
    
    Clinical ranges per NEWS2 guidelines:
    - Respiratory Rate: 4-50 breaths/min
    - Heart Rate: 20-220 beats/min
    - SpO2: 50-100%
    - Temperature: 30-45°C
    - Systolic BP: 40-300 mmHg
    - Consciousness: A, C, V, P, U (AVPU scale)
    """
    
    # Clinical validation ranges
    RESPIRATORY_RATE_RANGE = (4, 50)
    HEART_RATE_RANGE = (20, 220)
    SPO2_RANGE = (50, 100)
    TEMPERATURE_RANGE = (30.0, 45.0)
    SYSTOLIC_BP_RANGE = (40, 300)
    CONFIDENCE_RANGE = (0.0, 1.0)
    VALID_CONSCIOUSNESS_LEVELS = {'A', 'C', 'V', 'P', 'U'}
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_vital_signs(self, vitals: VitalSigns) -> List[ValidationError]:
        """
        Validate a VitalSigns object against clinical ranges.
        
        Args:
            vitals: VitalSigns object to validate
            
        Returns:
            List of validation errors. Empty list if validation passes.
        """
        errors = []
        
        try:
            # Validate respiratory rate
            if vitals.respiratory_rate is None:
                errors.append(ValidationError(
                    field="respiratory_rate",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Respiratory rate cannot be null",
                    received_value=vitals.respiratory_rate
                ))
            elif not isinstance(vitals.respiratory_rate, int):
                errors.append(ValidationError(
                    field="respiratory_rate",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="Respiratory rate must be an integer",
                    received_value=vitals.respiratory_rate
                ))
            elif not (self.RESPIRATORY_RATE_RANGE[0] <= vitals.respiratory_rate <= self.RESPIRATORY_RATE_RANGE[1]):
                errors.append(ValidationError(
                    field="respiratory_rate",
                    code=ValidationErrorCode.RESPIRATORY_RATE_OUT_OF_RANGE,
                    message=f"Respiratory rate must be between {self.RESPIRATORY_RATE_RANGE[0]} and {self.RESPIRATORY_RATE_RANGE[1]} breaths/min",
                    received_value=vitals.respiratory_rate,
                    valid_range=f"{self.RESPIRATORY_RATE_RANGE[0]}-{self.RESPIRATORY_RATE_RANGE[1]} breaths/min"
                ))
                
            # Validate heart rate
            if vitals.heart_rate is None:
                errors.append(ValidationError(
                    field="heart_rate",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Heart rate cannot be null",
                    received_value=vitals.heart_rate
                ))
            elif not isinstance(vitals.heart_rate, int):
                errors.append(ValidationError(
                    field="heart_rate",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="Heart rate must be an integer",
                    received_value=vitals.heart_rate
                ))
            elif not (self.HEART_RATE_RANGE[0] <= vitals.heart_rate <= self.HEART_RATE_RANGE[1]):
                errors.append(ValidationError(
                    field="heart_rate",
                    code=ValidationErrorCode.HEART_RATE_OUT_OF_RANGE,
                    message=f"Heart rate must be between {self.HEART_RATE_RANGE[0]} and {self.HEART_RATE_RANGE[1]} beats/min",
                    received_value=vitals.heart_rate,
                    valid_range=f"{self.HEART_RATE_RANGE[0]}-{self.HEART_RATE_RANGE[1]} beats/min"
                ))
                
            # Validate SpO2
            if vitals.spo2 is None:
                errors.append(ValidationError(
                    field="spo2",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="SpO2 cannot be null",
                    received_value=vitals.spo2
                ))
            elif not isinstance(vitals.spo2, int):
                errors.append(ValidationError(
                    field="spo2",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="SpO2 must be an integer",
                    received_value=vitals.spo2
                ))
            elif not (self.SPO2_RANGE[0] <= vitals.spo2 <= self.SPO2_RANGE[1]):
                errors.append(ValidationError(
                    field="spo2",
                    code=ValidationErrorCode.SPO2_OUT_OF_RANGE,
                    message=f"SpO2 must be between {self.SPO2_RANGE[0]} and {self.SPO2_RANGE[1]}%",
                    received_value=vitals.spo2,
                    valid_range=f"{self.SPO2_RANGE[0]}-{self.SPO2_RANGE[1]}%"
                ))
                
            # Validate temperature
            if vitals.temperature is None:
                errors.append(ValidationError(
                    field="temperature",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Temperature cannot be null",
                    received_value=vitals.temperature
                ))
            elif not isinstance(vitals.temperature, (int, float)):
                errors.append(ValidationError(
                    field="temperature",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="Temperature must be a number",
                    received_value=vitals.temperature
                ))
            elif not (self.TEMPERATURE_RANGE[0] <= vitals.temperature <= self.TEMPERATURE_RANGE[1]):
                errors.append(ValidationError(
                    field="temperature",
                    code=ValidationErrorCode.TEMPERATURE_OUT_OF_RANGE,
                    message=f"Temperature must be between {self.TEMPERATURE_RANGE[0]} and {self.TEMPERATURE_RANGE[1]}°C",
                    received_value=vitals.temperature,
                    valid_range=f"{self.TEMPERATURE_RANGE[0]}-{self.TEMPERATURE_RANGE[1]}°C"
                ))
                
            # Validate systolic blood pressure
            if vitals.systolic_bp is None:
                errors.append(ValidationError(
                    field="systolic_bp",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Systolic blood pressure cannot be null",
                    received_value=vitals.systolic_bp
                ))
            elif not isinstance(vitals.systolic_bp, int):
                errors.append(ValidationError(
                    field="systolic_bp",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="Systolic blood pressure must be an integer",
                    received_value=vitals.systolic_bp
                ))
            elif not (self.SYSTOLIC_BP_RANGE[0] <= vitals.systolic_bp <= self.SYSTOLIC_BP_RANGE[1]):
                errors.append(ValidationError(
                    field="systolic_bp",
                    code=ValidationErrorCode.SYSTOLIC_BP_OUT_OF_RANGE,
                    message=f"Systolic blood pressure must be between {self.SYSTOLIC_BP_RANGE[0]} and {self.SYSTOLIC_BP_RANGE[1]} mmHg",
                    received_value=vitals.systolic_bp,
                    valid_range=f"{self.SYSTOLIC_BP_RANGE[0]}-{self.SYSTOLIC_BP_RANGE[1]} mmHg"
                ))
                
            # Validate consciousness level
            if vitals.consciousness is None:
                errors.append(ValidationError(
                    field="consciousness",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Consciousness level cannot be null",
                    received_value=vitals.consciousness
                ))
            elif not isinstance(vitals.consciousness, ConsciousnessLevel):
                errors.append(ValidationError(
                    field="consciousness",
                    code=ValidationErrorCode.CONSCIOUSNESS_INVALID,
                    message="Consciousness level must be a valid ConsciousnessLevel enum",
                    received_value=str(vitals.consciousness),
                    valid_range="A, C, V, P, U"
                ))
                
            # Validate confidence score
            if vitals.confidence is None:
                errors.append(ValidationError(
                    field="confidence",
                    code=ValidationErrorCode.NULL_VALUE,
                    message="Confidence score cannot be null",
                    received_value=vitals.confidence
                ))
            elif not isinstance(vitals.confidence, (int, float)):
                errors.append(ValidationError(
                    field="confidence",
                    code=ValidationErrorCode.INVALID_DATA_TYPE,
                    message="Confidence score must be a number",
                    received_value=vitals.confidence
                ))
            elif not (self.CONFIDENCE_RANGE[0] <= vitals.confidence <= self.CONFIDENCE_RANGE[1]):
                errors.append(ValidationError(
                    field="confidence",
                    code=ValidationErrorCode.CONFIDENCE_OUT_OF_RANGE,
                    message=f"Confidence score must be between {self.CONFIDENCE_RANGE[0]} and {self.CONFIDENCE_RANGE[1]}",
                    received_value=vitals.confidence,
                    valid_range=f"{self.CONFIDENCE_RANGE[0]}-{self.CONFIDENCE_RANGE[1]}"
                ))
            
            # Log validation failures with patient hash for privacy
            if errors:
                patient_hash = self._hash_patient_id(vitals.patient_id)
                self.logger.warning(
                    f"Vital signs validation failed for patient {patient_hash}. "
                    f"Event ID: {vitals.event_id}. Errors: {len(errors)}"
                )
                
            return errors
            
        except Exception as e:
            self.logger.error(f"Unexpected error during vital signs validation: {str(e)}")
            # Return a generic validation error for unexpected exceptions
            return [ValidationError(
                field="unknown",
                code=ValidationErrorCode.INVALID_DATA_TYPE,
                message=f"Validation failed due to unexpected error: {str(e)}"
            )]
    
    def validate_partial_vitals(self, vitals_dict: Dict) -> List[ValidationError]:
        """
        Validate partial vital signs data (e.g., from API requests).
        Allows for missing fields but validates present ones.
        
        Args:
            vitals_dict: Dictionary containing partial vital signs data
            
        Returns:
            List of validation errors for present fields
        """
        errors = []
        
        # Check respiratory rate if present
        if 'respiratory_rate' in vitals_dict and vitals_dict['respiratory_rate'] is not None:
            rr = vitals_dict['respiratory_rate']
            if not isinstance(rr, int) or not (self.RESPIRATORY_RATE_RANGE[0] <= rr <= self.RESPIRATORY_RATE_RANGE[1]):
                errors.append(ValidationError(
                    field="respiratory_rate",
                    code=ValidationErrorCode.RESPIRATORY_RATE_OUT_OF_RANGE,
                    message=f"Respiratory rate must be between {self.RESPIRATORY_RATE_RANGE[0]} and {self.RESPIRATORY_RATE_RANGE[1]} breaths/min",
                    received_value=rr,
                    valid_range=f"{self.RESPIRATORY_RATE_RANGE[0]}-{self.RESPIRATORY_RATE_RANGE[1]} breaths/min"
                ))
        
        # Check heart rate if present
        if 'heart_rate' in vitals_dict and vitals_dict['heart_rate'] is not None:
            hr = vitals_dict['heart_rate']
            if not isinstance(hr, int) or not (self.HEART_RATE_RANGE[0] <= hr <= self.HEART_RATE_RANGE[1]):
                errors.append(ValidationError(
                    field="heart_rate",
                    code=ValidationErrorCode.HEART_RATE_OUT_OF_RANGE,
                    message=f"Heart rate must be between {self.HEART_RATE_RANGE[0]} and {self.HEART_RATE_RANGE[1]} beats/min",
                    received_value=hr,
                    valid_range=f"{self.HEART_RATE_RANGE[0]}-{self.HEART_RATE_RANGE[1]} beats/min"
                ))
        
        # Check SpO2 if present
        if 'spo2' in vitals_dict and vitals_dict['spo2'] is not None:
            spo2 = vitals_dict['spo2']
            if not isinstance(spo2, int) or not (self.SPO2_RANGE[0] <= spo2 <= self.SPO2_RANGE[1]):
                errors.append(ValidationError(
                    field="spo2",
                    code=ValidationErrorCode.SPO2_OUT_OF_RANGE,
                    message=f"SpO2 must be between {self.SPO2_RANGE[0]} and {self.SPO2_RANGE[1]}%",
                    received_value=spo2,
                    valid_range=f"{self.SPO2_RANGE[0]}-{self.SPO2_RANGE[1]}%"
                ))
        
        # Check temperature if present
        if 'temperature' in vitals_dict and vitals_dict['temperature'] is not None:
            temp = vitals_dict['temperature']
            if not isinstance(temp, (int, float)) or not (self.TEMPERATURE_RANGE[0] <= temp <= self.TEMPERATURE_RANGE[1]):
                errors.append(ValidationError(
                    field="temperature",
                    code=ValidationErrorCode.TEMPERATURE_OUT_OF_RANGE,
                    message=f"Temperature must be between {self.TEMPERATURE_RANGE[0]} and {self.TEMPERATURE_RANGE[1]}°C",
                    received_value=temp,
                    valid_range=f"{self.TEMPERATURE_RANGE[0]}-{self.TEMPERATURE_RANGE[1]}°C"
                ))
        
        # Check systolic blood pressure if present
        if 'systolic_bp' in vitals_dict and vitals_dict['systolic_bp'] is not None:
            sbp = vitals_dict['systolic_bp']
            if not isinstance(sbp, int) or not (self.SYSTOLIC_BP_RANGE[0] <= sbp <= self.SYSTOLIC_BP_RANGE[1]):
                errors.append(ValidationError(
                    field="systolic_bp",
                    code=ValidationErrorCode.SYSTOLIC_BP_OUT_OF_RANGE,
                    message=f"Systolic blood pressure must be between {self.SYSTOLIC_BP_RANGE[0]} and {self.SYSTOLIC_BP_RANGE[1]} mmHg",
                    received_value=sbp,
                    valid_range=f"{self.SYSTOLIC_BP_RANGE[0]}-{self.SYSTOLIC_BP_RANGE[1]} mmHg"
                ))
        
        # Check confidence if present
        if 'confidence' in vitals_dict and vitals_dict['confidence'] is not None:
            conf = vitals_dict['confidence']
            if not isinstance(conf, (int, float)) or not (self.CONFIDENCE_RANGE[0] <= conf <= self.CONFIDENCE_RANGE[1]):
                errors.append(ValidationError(
                    field="confidence",
                    code=ValidationErrorCode.CONFIDENCE_OUT_OF_RANGE,
                    message=f"Confidence score must be between {self.CONFIDENCE_RANGE[0]} and {self.CONFIDENCE_RANGE[1]}",
                    received_value=conf,
                    valid_range=f"{self.CONFIDENCE_RANGE[0]}-{self.CONFIDENCE_RANGE[1]}"
                ))
        
        return errors
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """
        Create a privacy-preserving hash of patient ID for logging.
        Uses salted hash to prevent reverse lookup while maintaining uniqueness.
        
        Args:
            patient_id: The patient ID to hash
            
        Returns:
            Hashed patient ID for logging
        """
        # Use a fixed salt for consistent hashing (in production, use environment variable)
        salt = "news2_validation_salt_2024"
        return hashlib.sha256(f"{salt}{patient_id}".encode()).hexdigest()[:16]
    
    def format_validation_errors(self, errors: List[ValidationError]) -> Dict:
        """
        Format validation errors for API responses.
        
        Args:
            errors: List of validation errors
            
        Returns:
            Formatted error response
        """
        if not errors:
            return {"valid": True, "errors": []}
            
        formatted_errors = []
        for error in errors:
            error_dict = {
                "field": error.field,
                "code": error.code.value,
                "message": error.message
            }
            if error.received_value is not None:
                error_dict["received_value"] = error.received_value
            if error.valid_range is not None:
                error_dict["valid_range"] = error.valid_range
                
            formatted_errors.append(error_dict)
            
        return {
            "valid": False,
            "errors": formatted_errors,
            "error_count": len(errors)
        }