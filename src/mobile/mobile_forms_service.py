"""
Mobile Forms Service for Story 3.4
Touch-optimized forms for vital signs entry and clinical documentation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


class FormFieldType(Enum):
    """Form field input types"""
    NUMBER = "number"
    TEXT = "text"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SLIDER = "slider"
    VOICE = "voice"
    BARCODE = "barcode"


class ValidationSeverity(Enum):
    """Validation message severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FormField:
    """Mobile-optimized form field configuration"""
    field_id: str
    field_type: FormFieldType
    label: str
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    options: List[Dict[str, Any]] = field(default_factory=list)
    placeholder: str = ""
    help_text: str = ""
    validation_pattern: Optional[str] = None
    touch_target_size: int = 48
    keyboard_type: str = "default"  # numeric, email, tel, etc.
    auto_focus: bool = False
    voice_enabled: bool = False


@dataclass
class ValidationResult:
    """Form field validation result"""
    field_id: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class FormValidationResult:
    """Complete form validation result"""
    is_valid: bool
    field_results: Dict[str, ValidationResult]
    overall_score: float
    completion_percentage: float
    critical_errors: int
    warnings: int

    def get_first_error(self) -> Optional[ValidationResult]:
        """Get first critical error or error"""
        for result in self.field_results.values():
            if result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                return result
        return None


@dataclass
class VitalSignsFormData:
    """Mobile vital signs form data structure"""
    patient_id: str
    timestamp: datetime
    respiratory_rate: Optional[int] = None
    spo2: Optional[int] = None
    on_oxygen: bool = False
    oxygen_flow_rate: Optional[float] = None
    temperature: Optional[float] = None
    temperature_method: str = "oral"
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    heart_rate: Optional[int] = None
    consciousness: str = "A"  # A, C, V, P, U
    pain_score: Optional[int] = None
    mobility: str = "independent"
    notes: str = ""
    is_manual_entry: bool = True
    entered_by: str = ""
    auto_saved: bool = False

    def get_completion_percentage(self) -> float:
        """Calculate form completion percentage"""
        total_fields = 11  # Core vital signs fields
        completed_fields = 0

        if self.respiratory_rate is not None:
            completed_fields += 1
        if self.spo2 is not None:
            completed_fields += 1
        if self.temperature is not None:
            completed_fields += 1
        if self.systolic_bp is not None:
            completed_fields += 1
        if self.heart_rate is not None:
            completed_fields += 1
        if self.consciousness != "A":  # Non-default
            completed_fields += 1
        if self.pain_score is not None:
            completed_fields += 1
        if self.mobility != "independent":  # Non-default
            completed_fields += 1
        if self.notes.strip():
            completed_fields += 1
        if self.entered_by.strip():
            completed_fields += 1
        # Always count timestamp as completed
        completed_fields += 1

        return (completed_fields / total_fields) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "respiratory_rate": self.respiratory_rate,
            "spo2": self.spo2,
            "on_oxygen": self.on_oxygen,
            "oxygen_flow_rate": self.oxygen_flow_rate,
            "temperature": self.temperature,
            "temperature_method": self.temperature_method,
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "heart_rate": self.heart_rate,
            "consciousness": self.consciousness,
            "pain_score": self.pain_score,
            "mobility": self.mobility,
            "notes": self.notes,
            "is_manual_entry": self.is_manual_entry,
            "entered_by": self.entered_by,
            "completion_percentage": self.get_completion_percentage(),
            "auto_saved": self.auto_saved
        }


class MobileFormsService:
    """Service for mobile-optimized form management and validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._auto_save_interval = 30  # seconds
        self._draft_storage = {}
        self._validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for vital signs"""
        return {
            "respiratory_rate": {
                "min": 4,
                "max": 50,
                "normal_min": 12,
                "normal_max": 20,
                "critical_min": 8,
                "critical_max": 35
            },
            "spo2": {
                "min": 50,
                "max": 100,
                "normal_min": 95,
                "normal_max": 100,
                "critical_min": 88,
                "critical_max": 100
            },
            "temperature": {
                "min": 30.0,
                "max": 45.0,
                "normal_min": 36.1,
                "normal_max": 37.8,
                "critical_min": 35.0,
                "critical_max": 39.0
            },
            "systolic_bp": {
                "min": 50,
                "max": 250,
                "normal_min": 90,
                "normal_max": 140,
                "critical_min": 90,
                "critical_max": 180
            },
            "heart_rate": {
                "min": 20,
                "max": 200,
                "normal_min": 60,
                "normal_max": 100,
                "critical_min": 50,
                "critical_max": 120
            },
            "pain_score": {
                "min": 0,
                "max": 10,
                "normal_min": 0,
                "normal_max": 3,
                "critical_min": 7,
                "critical_max": 10
            }
        }

    def get_vital_signs_form_config(self, screen_size: str = "tablet") -> Dict[str, Any]:
        """Get mobile-optimized vital signs form configuration"""
        is_compact = screen_size == "smartphone"

        fields = [
            FormField(
                field_id="respiratory_rate",
                field_type=FormFieldType.NUMBER,
                label="Respiratory Rate",
                required=True,
                min_value=4,
                max_value=50,
                placeholder="16-20 normal",
                help_text="Breaths per minute",
                keyboard_type="numeric",
                voice_enabled=True,
                touch_target_size=56 if not is_compact else 48
            ),
            FormField(
                field_id="spo2",
                field_type=FormFieldType.SLIDER if is_compact else FormFieldType.NUMBER,
                label="SpO₂",
                required=True,
                min_value=50,
                max_value=100,
                placeholder="95-100% normal",
                help_text="Oxygen saturation percentage",
                keyboard_type="numeric",
                voice_enabled=True
            ),
            FormField(
                field_id="on_oxygen",
                field_type=FormFieldType.CHECKBOX,
                label="On Supplemental Oxygen",
                required=False,
                help_text="Check if patient receiving oxygen therapy"
            ),
            FormField(
                field_id="temperature",
                field_type=FormFieldType.NUMBER,
                label="Temperature (°C)",
                required=True,
                min_value=30.0,
                max_value=45.0,
                placeholder="36.1-37.8 normal",
                help_text="Body temperature in Celsius",
                keyboard_type="decimal",
                voice_enabled=True
            ),
            FormField(
                field_id="temperature_method",
                field_type=FormFieldType.SELECT,
                label="Temperature Method",
                required=False,
                options=[
                    {"value": "oral", "label": "Oral"},
                    {"value": "axillary", "label": "Axillary"},
                    {"value": "tympanic", "label": "Tympanic"},
                    {"value": "rectal", "label": "Rectal"}
                ]
            ),
            FormField(
                field_id="systolic_bp",
                field_type=FormFieldType.NUMBER,
                label="Systolic BP",
                required=True,
                min_value=50,
                max_value=250,
                placeholder="90-140 normal",
                help_text="Systolic blood pressure (mmHg)",
                keyboard_type="numeric",
                voice_enabled=True
            ),
            FormField(
                field_id="heart_rate",
                field_type=FormFieldType.NUMBER,
                label="Heart Rate",
                required=True,
                min_value=20,
                max_value=200,
                placeholder="60-100 normal",
                help_text="Beats per minute",
                keyboard_type="numeric",
                voice_enabled=True
            ),
            FormField(
                field_id="consciousness",
                field_type=FormFieldType.RADIO,
                label="Consciousness Level",
                required=True,
                options=[
                    {"value": "A", "label": "Alert", "description": "Awake and responsive"},
                    {"value": "C", "label": "Confused", "description": "Confusion or disorientation"},
                    {"value": "V", "label": "Voice", "description": "Responds to voice"},
                    {"value": "P", "label": "Pain", "description": "Responds to pain only"},
                    {"value": "U", "label": "Unresponsive", "description": "No response"}
                ]
            )
        ]

        # Add additional fields for larger screens
        if not is_compact:
            fields.extend([
                FormField(
                    field_id="pain_score",
                    field_type=FormFieldType.SLIDER,
                    label="Pain Score (0-10)",
                    required=False,
                    min_value=0,
                    max_value=10,
                    help_text="Patient-reported pain level"
                ),
                FormField(
                    field_id="mobility",
                    field_type=FormFieldType.SELECT,
                    label="Mobility",
                    required=False,
                    options=[
                        {"value": "independent", "label": "Independent"},
                        {"value": "assistance", "label": "Needs Assistance"},
                        {"value": "bedbound", "label": "Bedbound"},
                        {"value": "immobile", "label": "Immobile"}
                    ]
                ),
                FormField(
                    field_id="notes",
                    field_type=FormFieldType.TEXT,
                    label="Clinical Notes",
                    required=False,
                    placeholder="Additional observations...",
                    voice_enabled=True
                )
            ])

        return {
            "form_id": "vital_signs_mobile",
            "title": "Vital Signs Entry",
            "fields": [field.__dict__ for field in fields],
            "layout": {
                "compact_mode": is_compact,
                "sections": self._get_form_sections(is_compact),
                "auto_save_interval": self._auto_save_interval,
                "validation_mode": "real_time"
            },
            "actions": [
                {"id": "save", "label": "Save", "type": "primary", "size": "large"},
                {"id": "save_draft", "label": "Save Draft", "type": "secondary"},
                {"id": "cancel", "label": "Cancel", "type": "text"}
            ]
        }

    def _get_form_sections(self, is_compact: bool) -> List[Dict[str, Any]]:
        """Get form sections based on screen size"""
        if is_compact:
            return [
                {
                    "id": "core_vitals",
                    "title": "Core Vitals",
                    "fields": ["respiratory_rate", "spo2", "temperature", "heart_rate"],
                    "collapsible": False
                },
                {
                    "id": "blood_pressure",
                    "title": "Blood Pressure",
                    "fields": ["systolic_bp"],
                    "collapsible": True
                },
                {
                    "id": "assessment",
                    "title": "Assessment",
                    "fields": ["consciousness", "on_oxygen"],
                    "collapsible": True
                }
            ]
        else:
            return [
                {
                    "id": "vital_signs",
                    "title": "Vital Signs",
                    "fields": ["respiratory_rate", "spo2", "temperature", "systolic_bp", "heart_rate"],
                    "collapsible": False
                },
                {
                    "id": "clinical_assessment",
                    "title": "Clinical Assessment",
                    "fields": ["consciousness", "pain_score", "mobility"],
                    "collapsible": False
                },
                {
                    "id": "additional",
                    "title": "Additional Information",
                    "fields": ["on_oxygen", "temperature_method", "notes"],
                    "collapsible": True
                }
            ]

    async def validate_form_data(self, form_data: VitalSignsFormData) -> FormValidationResult:
        """Validate vital signs form data with clinical rules"""
        field_results = {}
        critical_errors = 0
        warnings = 0

        # Validate respiratory rate
        if form_data.respiratory_rate is not None:
            field_results["respiratory_rate"] = self._validate_vital_sign(
                "respiratory_rate", form_data.respiratory_rate
            )
        else:
            field_results["respiratory_rate"] = ValidationResult(
                field_id="respiratory_rate",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Respiratory rate is required"
            )

        # Validate SpO2
        if form_data.spo2 is not None:
            result = self._validate_vital_sign("spo2", form_data.spo2)

            # Special validation for oxygen therapy
            if form_data.on_oxygen and form_data.spo2 < 92:
                result.severity = ValidationSeverity.CRITICAL
                result.message += " - Patient on oxygen with low SpO2!"
                result.suggestions.append("Consider increasing oxygen flow or escalating care")

            field_results["spo2"] = result
        else:
            field_results["spo2"] = ValidationResult(
                field_id="spo2",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="SpO2 is required"
            )

        # Validate temperature
        if form_data.temperature is not None:
            field_results["temperature"] = self._validate_vital_sign(
                "temperature", form_data.temperature
            )

        # Validate blood pressure
        if form_data.systolic_bp is not None:
            result = self._validate_vital_sign("systolic_bp", form_data.systolic_bp)

            # Additional BP validation logic
            if form_data.systolic_bp < 90:
                result.severity = ValidationSeverity.CRITICAL
                result.message = "Critical hypotension - immediate intervention required"
            elif form_data.systolic_bp > 180:
                result.severity = ValidationSeverity.CRITICAL
                result.message = "Hypertensive crisis - urgent medical attention required"

            field_results["systolic_bp"] = result

        # Validate heart rate
        if form_data.heart_rate is not None:
            result = self._validate_vital_sign("heart_rate", form_data.heart_rate)

            # Tachycardia/bradycardia alerts
            if form_data.heart_rate < 50:
                result.severity = ValidationSeverity.WARNING
                result.message += " - Bradycardia detected"
            elif form_data.heart_rate > 120:
                result.severity = ValidationSeverity.WARNING
                result.message += " - Tachycardia detected"

            field_results["heart_rate"] = result

        # Count errors and warnings
        for result in field_results.values():
            if result.severity == ValidationSeverity.CRITICAL:
                critical_errors += 1
            elif result.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
                warnings += 1

        # Calculate overall validation
        is_valid = critical_errors == 0 and all(r.is_valid for r in field_results.values())
        completion_pct = form_data.get_completion_percentage()

        # Overall score based on validity and completion
        overall_score = 0.0
        if is_valid and completion_pct >= 80:
            overall_score = 1.0
        elif completion_pct >= 60:
            overall_score = 0.7
        elif completion_pct >= 40:
            overall_score = 0.5
        else:
            overall_score = 0.3

        return FormValidationResult(
            is_valid=is_valid,
            field_results=field_results,
            overall_score=overall_score,
            completion_percentage=completion_pct,
            critical_errors=critical_errors,
            warnings=warnings
        )

    def _validate_vital_sign(self, field_name: str, value: Union[int, float]) -> ValidationResult:
        """Validate individual vital sign value"""
        rules = self._validation_rules.get(field_name, {})

        if not rules:
            return ValidationResult(
                field_id=field_name,
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="No validation rules defined"
            )

        # Range validation
        if value < rules.get("min", 0) or value > rules.get("max", 999):
            return ValidationResult(
                field_id=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Value {value} is outside valid range ({rules['min']}-{rules['max']})",
                suggestions=[f"Enter value between {rules['min']} and {rules['max']}"]
            )

        # Critical value check
        if (value < rules.get("critical_min", 0) or
            value > rules.get("critical_max", 999)):
            return ValidationResult(
                field_id=field_name,
                is_valid=True,  # Valid but critical
                severity=ValidationSeverity.CRITICAL,
                message=f"CRITICAL: {field_name} = {value} requires immediate attention",
                suggestions=["Consider escalating care", "Recheck measurement", "Notify physician"]
            )

        # Normal range check
        normal_min = rules.get("normal_min", 0)
        normal_max = rules.get("normal_max", 999)

        if value < normal_min or value > normal_max:
            return ValidationResult(
                field_id=field_name,
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"{field_name} = {value} is outside normal range ({normal_min}-{normal_max})",
                suggestions=["Monitor closely", "Consider recheck in 15 minutes"]
            )

        # Normal value
        return ValidationResult(
            field_id=field_name,
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} within normal range"
        )

    async def save_draft(self, patient_id: str, form_data: VitalSignsFormData) -> Dict[str, Any]:
        """Auto-save form data as draft"""
        try:
            draft_key = f"{patient_id}_{datetime.now().strftime('%Y%m%d')}"
            form_data.auto_saved = True
            self._draft_storage[draft_key] = form_data

            return {
                "success": True,
                "draft_id": draft_key,
                "timestamp": datetime.now().isoformat(),
                "completion_percentage": form_data.get_completion_percentage()
            }

        except Exception as e:
            self.logger.error(f"Error saving draft for {patient_id}: {e}")
            return {"success": False, "error": str(e)}

    async def load_draft(self, patient_id: str) -> Optional[VitalSignsFormData]:
        """Load most recent draft for patient"""
        try:
            draft_key = f"{patient_id}_{datetime.now().strftime('%Y%m%d')}"
            return self._draft_storage.get(draft_key)

        except Exception as e:
            self.logger.error(f"Error loading draft for {patient_id}: {e}")
            return None

    async def process_voice_input(self, field_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process voice input for form fields"""
        # Simulated voice processing
        try:
            # In real implementation, this would use speech-to-text service
            voice_results = {
                "respiratory_rate": ["eighteen", "twenty", "sixteen"],
                "spo2": ["ninety-five percent", "ninety-eight percent"],
                "temperature": ["thirty-seven point two", "thirty-eight degrees"],
                "heart_rate": ["eighty", "seventy-five", "ninety"],
                "notes": ["patient appears comfortable", "no acute distress noted"]
            }

            suggestions = voice_results.get(field_id, ["Unable to process"])

            return {
                "success": True,
                "field_id": field_id,
                "suggestions": suggestions,
                "confidence": 0.85,
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing voice input for {field_id}: {e}")
            return {
                "success": False,
                "error": "Voice processing failed",
                "field_id": field_id
            }

    async def get_smart_suggestions(self, patient_id: str, field_id: str,
                                  partial_value: str = "") -> List[Dict[str, Any]]:
        """Get smart suggestions based on patient history and partial input"""
        try:
            # Simulated smart suggestions based on patient history
            suggestions = []

            if field_id == "respiratory_rate":
                suggestions = [
                    {"value": 18, "reason": "Patient's typical range", "confidence": 0.9},
                    {"value": 16, "reason": "Normal adult range", "confidence": 0.8},
                    {"value": 20, "reason": "Recent average", "confidence": 0.7}
                ]
            elif field_id == "temperature":
                suggestions = [
                    {"value": 37.2, "reason": "Patient baseline", "confidence": 0.85},
                    {"value": 36.8, "reason": "Normal range", "confidence": 0.8}
                ]
            elif field_id == "consciousness":
                suggestions = [
                    {"value": "A", "reason": "Patient usually alert", "confidence": 0.95}
                ]

            return suggestions

        except Exception as e:
            self.logger.error(f"Error getting suggestions for {field_id}: {e}")
            return []

    def get_form_accessibility_config(self) -> Dict[str, Any]:
        """Get accessibility configuration for mobile forms"""
        return {
            "screen_reader": {
                "enabled": True,
                "field_descriptions": True,
                "validation_announcements": True
            },
            "high_contrast": {
                "available": True,
                "error_colors": {"background": "#fff", "text": "#d32f2f"},
                "success_colors": {"background": "#fff", "text": "#388e3c"}
            },
            "font_scaling": {
                "min_scale": 0.8,
                "max_scale": 2.0,
                "respect_system_settings": True
            },
            "touch_targets": {
                "min_size": 44,
                "recommended_size": 48,
                "spacing": 8
            },
            "haptic_feedback": {
                "enabled": True,
                "on_error": "error_pattern",
                "on_success": "success_pattern",
                "on_critical": "warning_pattern"
            }
        }