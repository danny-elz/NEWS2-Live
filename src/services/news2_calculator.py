import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Union

from ..models.patient import Patient
from ..models.vital_signs import VitalSigns, ConsciousnessLevel
from ..models.partial_vital_signs import PartialVitalSigns
from ..models.news2 import NEWS2Result, RiskCategory, CalculationError, MissingVitalSignsError, InvalidVitalSignsError
from ..services.audit import AuditLogger


class NEWS2Calculator:
    """
    NEWS2 (National Early Warning Score 2) calculator implementing RCP guidelines.
    
    Supports both Scale 1 (standard) and Scale 2 (COPD patients) scoring systems
    with thread-safe concurrent processing capability.
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    async def calculate_news2(self, vital_signs: VitalSigns, patient: Patient) -> NEWS2Result:
        """
        Calculate NEWS2 score for given vital signs and patient.
        
        Args:
            vital_signs: Patient's vital signs measurements
            patient: Patient information including COPD status
            
        Returns:
            NEWS2Result with complete scoring and risk assessment
            
        Raises:
            MissingVitalSignsError: If required vital signs are missing
            InvalidVitalSignsError: If vital signs are outside valid ranges
        """
        start_time = time.perf_counter()
        
        try:
            # Validate required vital signs parameters before calculation
            self._validate_vital_signs(vital_signs)
            self._validate_patient(patient)
            # Determine which scale to use based on patient COPD status
            scale_used = 2 if patient.is_copd_patient else 1
            
            # Calculate individual parameter scores
            individual_scores = {}
            warnings = []
            
            # Respiratory rate scoring (same for both scales)
            individual_scores['respiratory_rate'] = self._score_respiratory_rate(vital_signs.respiratory_rate)
            
            # SpO2 scoring (different for COPD patients)
            if scale_used == 1:
                individual_scores['spo2'] = self._score_spo2_scale1(vital_signs.spo2)
            else:
                individual_scores['spo2'] = self._score_spo2_scale2(vital_signs.spo2)
            
            # Supplemental oxygen scoring (same for both scales)
            individual_scores['oxygen'] = self._score_oxygen(vital_signs.on_oxygen)
            
            # Temperature scoring (same for both scales)
            individual_scores['temperature'] = self._score_temperature(vital_signs.temperature)
            
            # Systolic blood pressure scoring (same for both scales)
            individual_scores['systolic_bp'] = self._score_systolic_bp(vital_signs.systolic_bp)
            
            # Heart rate scoring (same for both scales)
            individual_scores['heart_rate'] = self._score_heart_rate(vital_signs.heart_rate)
            
            # Consciousness scoring (same for both scales)
            individual_scores['consciousness'] = self._score_consciousness(vital_signs.consciousness)
            
            # Calculate total score
            total_score = sum(individual_scores.values())
            
            # Determine risk category and monitoring frequency
            risk_category, monitoring_frequency = self._assess_risk_category(total_score, individual_scores)
            
            # Check for red flags (any parameter = 3)
            red_flags = [param for param, score in individual_scores.items() if score == 3]
            if red_flags:
                warnings.append(f"Single parameter red flag detected: {', '.join(red_flags)} - requires immediate review")
            
            # Add clear indication when Scale 2 is used
            if scale_used == 2:
                warnings.append(f"NEWS2 Scale 2 (COPD) used - modified SpO2 ranges applied for COPD patient")
            
            # Generate clinical escalation guidance
            clinical_guidance = self._generate_clinical_guidance(total_score, risk_category, red_flags, patient)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(vital_signs, individual_scores)
            
            # Record calculation time
            end_time = time.perf_counter()
            calculation_time_ms = (end_time - start_time) * 1000
            
            # Create result
            result = NEWS2Result(
                total_score=total_score,
                individual_scores=individual_scores,
                risk_category=risk_category,
                monitoring_frequency=monitoring_frequency,
                scale_used=scale_used,
                warnings=warnings,
                confidence=confidence,
                calculated_at=datetime.now(timezone.utc),
                calculation_time_ms=calculation_time_ms,
                clinical_guidance=clinical_guidance
            )
            
            # Log calculation for audit trail
            patient_hash = self.audit_logger._hash_patient_id(patient.patient_id)
            self.audit_logger.logger.info(
                f"NEWS2 calculation - Patient: {patient_hash}, Score: {total_score}, "
                f"Risk: {risk_category.value}, Scale: {scale_used}, Time: {calculation_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            self.audit_logger.logger.error(f"NEWS2 calculation failed: {str(e)}")
            raise CalculationError(f"NEWS2 calculation failed: {str(e)}")
    
    async def calculate_partial_news2(self, partial_vitals: PartialVitalSigns, patient: Patient, max_retries: int = 3) -> NEWS2Result:
        """
        Calculate NEWS2 score for partial vital signs with edge case handling.
        
        Args:
            partial_vitals: PartialVitalSigns object with potentially missing parameters
            patient: Patient information including COPD status
            max_retries: Maximum number of calculation retries for transient failures
            
        Returns:
            NEWS2Result with partial scoring and comprehensive warnings
            
        Raises:
            CalculationError: If calculation fails after retries
            InvalidVitalSignsError: If vital signs contain impossible combinations
        """
        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                
                # Validate patient information
                self._validate_patient(patient)
                
                # Check for physiologically impossible combinations
                impossible_combo = partial_vitals.has_physiologically_impossible_combination()
                if impossible_combo:
                    raise InvalidVitalSignsError(f"Physiologically impossible combination detected: {impossible_combo}")
                
                # Determine which scale to use based on patient COPD status
                scale_used = 2 if patient.is_copd_patient else 1
                
                # Calculate individual parameter scores (handling missing values)
                individual_scores = {}
                warnings = []
                missing_params = partial_vitals.get_missing_parameters()
                
                # Respiratory rate scoring
                if partial_vitals.respiratory_rate is not None:
                    individual_scores['respiratory_rate'] = self._score_respiratory_rate(partial_vitals.respiratory_rate)
                else:
                    individual_scores['respiratory_rate'] = 0
                    warnings.append("Respiratory rate missing - scored as 0 (may underestimate risk)")
                
                # SpO2 scoring
                if partial_vitals.spo2 is not None:
                    if scale_used == 1:
                        individual_scores['spo2'] = self._score_spo2_scale1(partial_vitals.spo2)
                    else:
                        individual_scores['spo2'] = self._score_spo2_scale2(partial_vitals.spo2)
                else:
                    individual_scores['spo2'] = 0
                    warnings.append("SpO2 missing - scored as 0 (may significantly underestimate risk)")
                
                # Supplemental oxygen scoring
                if partial_vitals.on_oxygen is not None:
                    individual_scores['oxygen'] = self._score_oxygen(partial_vitals.on_oxygen)
                else:
                    individual_scores['oxygen'] = 0
                    warnings.append("Oxygen status missing - scored as 0 (may underestimate risk)")
                
                # Temperature scoring
                if partial_vitals.temperature is not None:
                    individual_scores['temperature'] = self._score_temperature(partial_vitals.temperature)
                else:
                    individual_scores['temperature'] = 0
                    warnings.append("Temperature missing - scored as 0 (may underestimate risk)")
                
                # Systolic blood pressure scoring
                if partial_vitals.systolic_bp is not None:
                    individual_scores['systolic_bp'] = self._score_systolic_bp(partial_vitals.systolic_bp)
                else:
                    individual_scores['systolic_bp'] = 0
                    warnings.append("Systolic BP missing - scored as 0 (may underestimate risk)")
                
                # Heart rate scoring
                if partial_vitals.heart_rate is not None:
                    individual_scores['heart_rate'] = self._score_heart_rate(partial_vitals.heart_rate)
                else:
                    individual_scores['heart_rate'] = 0
                    warnings.append("Heart rate missing - scored as 0 (may underestimate risk)")
                
                # Consciousness scoring
                if partial_vitals.consciousness is not None:
                    individual_scores['consciousness'] = self._score_consciousness(partial_vitals.consciousness)
                else:
                    individual_scores['consciousness'] = 0
                    warnings.append("Consciousness level missing - scored as 0 (may significantly underestimate risk)")
                
                # Calculate total score
                total_score = sum(individual_scores.values())
                
                # Add completeness warning
                completeness = partial_vitals.get_completeness_score()
                if completeness < 1.0:
                    warnings.append(f"Partial vital signs: {completeness:.0%} complete. Score may underestimate actual risk.")
                
                # Determine risk category and monitoring frequency
                risk_category, monitoring_frequency = self._assess_risk_category(total_score, individual_scores)
                
                # Check for red flags (any parameter = 3)
                red_flags = [param for param, score in individual_scores.items() if score == 3]
                if red_flags:
                    warnings.append(f"Single parameter red flag detected: {', '.join(red_flags)} - requires immediate review")
                
                # Add clear indication when Scale 2 is used
                if scale_used == 2:
                    warnings.append(f"NEWS2 Scale 2 (COPD) used - modified SpO2 ranges applied for COPD patient")
                
                # Generate clinical escalation guidance
                clinical_guidance = self._generate_clinical_guidance(total_score, risk_category, red_flags, patient)
                
                # Add guidance for missing parameters
                if missing_params:
                    clinical_guidance["missing_parameters"] = f"Missing vital signs ({', '.join(missing_params)}) may result in risk underestimation. Obtain complete vitals when possible."
                
                # Calculate enhanced confidence based on completeness and data quality
                confidence = self._calculate_enhanced_confidence(partial_vitals, individual_scores, completeness)
                
                # Record calculation time
                end_time = time.perf_counter()
                calculation_time_ms = (end_time - start_time) * 1000
                
                # Create result
                result = NEWS2Result(
                    total_score=total_score,
                    individual_scores=individual_scores,
                    risk_category=risk_category,
                    monitoring_frequency=monitoring_frequency,
                    scale_used=scale_used,
                    warnings=warnings,
                    confidence=confidence,
                    calculated_at=datetime.now(timezone.utc),
                    calculation_time_ms=calculation_time_ms,
                    clinical_guidance=clinical_guidance
                )
                
                # Log calculation for audit trail
                patient_hash = self.audit_logger._hash_patient_id(patient.patient_id)
                self.audit_logger.logger.info(
                    f"Partial NEWS2 calculation - Patient: {patient_hash}, Score: {total_score}, "
                    f"Risk: {risk_category.value}, Scale: {scale_used}, Completeness: {completeness:.0%}, "
                    f"Time: {calculation_time_ms:.2f}ms"
                )
                
                return result
                
            except (MissingVitalSignsError, InvalidVitalSignsError) as e:
                # These are not transient errors - don't retry
                raise e
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed
                    self.audit_logger.logger.error(f"Partial NEWS2 calculation failed after {max_retries} attempts: {str(e)}")
                    raise CalculationError(f"Partial NEWS2 calculation failed after {max_retries} attempts: {str(e)}")
                else:
                    # Log retry attempt
                    self.audit_logger.logger.warning(f"NEWS2 calculation attempt {attempt + 1} failed, retrying: {str(e)}")
                    await asyncio.sleep(0.1 * (attempt + 1))  # Progressive delay
    
    def _score_respiratory_rate(self, respiratory_rate: int) -> int:
        """Score respiratory rate according to NEWS2 guidelines."""
        if respiratory_rate <= 8:
            return 3
        elif 9 <= respiratory_rate <= 11:
            return 1
        elif 12 <= respiratory_rate <= 20:
            return 0
        elif 21 <= respiratory_rate <= 24:
            return 2
        elif respiratory_rate >= 25:
            return 3
        else:
            raise InvalidVitalSignsError(f"Invalid respiratory rate: {respiratory_rate}")
    
    def _score_spo2_scale1(self, spo2: int) -> int:
        """Score SpO2 according to NEWS2 Scale 1 (standard patients)."""
        if spo2 <= 91:
            return 3
        elif 92 <= spo2 <= 93:
            return 2
        elif 94 <= spo2 <= 95:
            return 1
        elif spo2 >= 96:
            return 0
        else:
            raise InvalidVitalSignsError(f"Invalid SpO2: {spo2}")
    
    def _score_spo2_scale2(self, spo2: int) -> int:
        """Score SpO2 according to NEWS2 Scale 2 (COPD patients)."""
        if spo2 <= 83:
            return 3
        elif 84 <= spo2 <= 85:
            return 2
        elif 86 <= spo2 <= 87:
            return 1
        elif 88 <= spo2 <= 92:
            return 0
        elif 93 <= spo2 <= 94:
            return 1
        elif 95 <= spo2 <= 96:
            return 2
        elif spo2 >= 97:
            return 3
        else:
            raise InvalidVitalSignsError(f"Invalid SpO2: {spo2}")
    
    def _score_oxygen(self, on_oxygen: bool) -> int:
        """Score supplemental oxygen usage."""
        return 2 if on_oxygen else 0
    
    def _score_temperature(self, temperature: float) -> int:
        """Score temperature according to NEWS2 guidelines."""
        if temperature <= 35.0:
            return 3
        elif 35.1 <= temperature <= 36.0:
            return 1
        elif 36.1 <= temperature <= 38.0:
            return 0
        elif 38.1 <= temperature <= 39.0:
            return 1
        elif temperature >= 39.1:
            return 2
        else:
            raise InvalidVitalSignsError(f"Invalid temperature: {temperature}")
    
    def _score_systolic_bp(self, systolic_bp: int) -> int:
        """Score systolic blood pressure according to NEWS2 guidelines."""
        if systolic_bp <= 90:
            return 3
        elif 91 <= systolic_bp <= 100:
            return 2
        elif 101 <= systolic_bp <= 110:
            return 1
        elif 111 <= systolic_bp <= 219:
            return 0
        elif systolic_bp >= 220:
            return 3
        else:
            raise InvalidVitalSignsError(f"Invalid systolic BP: {systolic_bp}")
    
    def _score_heart_rate(self, heart_rate: int) -> int:
        """Score heart rate according to NEWS2 guidelines."""
        if heart_rate <= 40:
            return 3
        elif 41 <= heart_rate <= 50:
            return 1
        elif 51 <= heart_rate <= 90:
            return 0
        elif 91 <= heart_rate <= 110:
            return 1
        elif 111 <= heart_rate <= 130:
            return 2
        elif heart_rate >= 131:
            return 3
        else:
            raise InvalidVitalSignsError(f"Invalid heart rate: {heart_rate}")
    
    def _score_consciousness(self, consciousness: ConsciousnessLevel) -> int:
        """Score consciousness level according to NEWS2 guidelines."""
        if consciousness == ConsciousnessLevel.ALERT:
            return 0
        else:  # CONFUSION, VOICE, PAIN, UNRESPONSIVE
            return 3
    
    def _assess_risk_category(self, total_score: int, individual_scores: Dict[str, int]) -> tuple[RiskCategory, str]:
        """
        Assess risk category and monitoring frequency based on total score.
        
        Returns:
            Tuple of (risk_category, monitoring_frequency)
        """
        # Check for single parameter red flag (any parameter = 3)
        has_red_flag = any(score >= 3 for score in individual_scores.values())
        
        # Single parameter red flag overrides total score for HIGH risk
        if has_red_flag:
            return RiskCategory.HIGH, "continuous monitoring + urgent medical response"
        elif total_score == 0:
            return RiskCategory.LOW, "routine monitoring"
        elif 1 <= total_score <= 2:
            return RiskCategory.LOW, "12-hourly observations"
        elif 3 <= total_score <= 4:
            return RiskCategory.MEDIUM, "6-hourly observations"
        elif 5 <= total_score <= 6:
            return RiskCategory.MEDIUM, "hourly observations + medical review"
        elif total_score >= 7:
            return RiskCategory.HIGH, "continuous monitoring + urgent medical response"
        else:
            return RiskCategory.LOW, "routine monitoring"
    
    def _calculate_confidence(self, vital_signs: VitalSigns, individual_scores: Dict[str, int]) -> float:
        """
        Calculate confidence score based on data quality and completeness.
        
        Args:
            vital_signs: VitalSigns object with quality indicators
            individual_scores: Calculated individual parameter scores
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = vital_signs.confidence
        
        # Reduce confidence for manual entries
        if vital_signs.is_manual_entry:
            base_confidence *= 0.95
        
        # Reduce confidence if artifacts detected
        if vital_signs.has_artifacts:
            base_confidence *= 0.90
        
        # Ensure confidence stays within valid range
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_enhanced_confidence(self, vitals: Union[VitalSigns, PartialVitalSigns], individual_scores: Dict[str, int], completeness: float = 1.0) -> float:
        """
        Calculate enhanced confidence score based on data quality, completeness, and clinical context.
        
        Args:
            vitals: VitalSigns or PartialVitalSigns object with quality indicators
            individual_scores: Calculated individual parameter scores
            completeness: Completeness ratio (0.0 to 1.0)
            
        Returns:
            Enhanced confidence score between 0.0 and 1.0
        """
        base_confidence = vitals.confidence
        
        # Reduce confidence for incomplete data
        completeness_factor = max(0.5, completeness)  # Never go below 50% for partial data
        base_confidence *= completeness_factor
        
        # Reduce confidence for manual entries
        if vitals.is_manual_entry:
            base_confidence *= 0.95
        
        # Reduce confidence if artifacts detected
        if vitals.has_artifacts:
            base_confidence *= 0.90
        
        # Reduce confidence for extreme values that may indicate measurement error
        extreme_value_penalty = 0.0
        
        # Check for potentially erroneous values
        if hasattr(vitals, 'respiratory_rate') and vitals.respiratory_rate is not None:
            if vitals.respiratory_rate < 6 or vitals.respiratory_rate > 40:
                extreme_value_penalty += 0.05
        
        if hasattr(vitals, 'heart_rate') and vitals.heart_rate is not None:
            if vitals.heart_rate < 30 or vitals.heart_rate > 200:
                extreme_value_penalty += 0.05
        
        if hasattr(vitals, 'systolic_bp') and vitals.systolic_bp is not None:
            if vitals.systolic_bp < 60 or vitals.systolic_bp > 250:
                extreme_value_penalty += 0.05
        
        if hasattr(vitals, 'temperature') and vitals.temperature is not None:
            if vitals.temperature < 32.0 or vitals.temperature > 42.0:
                extreme_value_penalty += 0.05
        
        # Apply extreme value penalty
        base_confidence *= (1.0 - min(0.2, extreme_value_penalty))
        
        # Boost confidence for consistent vital signs pattern
        if completeness == 1.0:  # Complete vitals
            # Check if vital signs are consistent with each other
            consistency_bonus = self._assess_vital_signs_consistency(vitals, individual_scores)
            base_confidence += consistency_bonus
        
        # Ensure confidence stays within valid range
        return max(0.0, min(1.0, base_confidence))
    
    def _assess_vital_signs_consistency(self, vitals: Union[VitalSigns, PartialVitalSigns], individual_scores: Dict[str, int]) -> float:
        """
        Assess consistency of vital signs pattern and provide confidence bonus.
        
        Args:
            vitals: Complete vital signs
            individual_scores: Individual parameter scores
            
        Returns:
            Consistency bonus (0.0 to 0.1)
        """
        if not hasattr(vitals, 'heart_rate') or not hasattr(vitals, 'systolic_bp'):
            return 0.0
        
        # Check for consistent patterns that increase confidence
        consistency_bonus = 0.0
        
        # Consistent shock pattern (high HR + low BP + altered consciousness)
        if (vitals.heart_rate and vitals.systolic_bp and 
            vitals.heart_rate > 110 and vitals.systolic_bp < 100 and
            hasattr(vitals, 'consciousness') and vitals.consciousness != ConsciousnessLevel.ALERT):
            consistency_bonus += 0.05
        
        # Consistent normal pattern (all parameters normal)
        normal_params = sum(1 for score in individual_scores.values() if score == 0)
        if normal_params >= 5:  # Most parameters normal
            consistency_bonus += 0.03
        
        # Consistent severe illness pattern (multiple elevated scores)
        elevated_params = sum(1 for score in individual_scores.values() if score >= 2)
        if elevated_params >= 3:
            consistency_bonus += 0.02
        
        return min(0.1, consistency_bonus)  # Cap at 10% bonus
    
    def _generate_clinical_guidance(self, total_score: int, risk_category: RiskCategory, red_flags: List[str], patient: Patient) -> Dict[str, str]:
        """
        Generate comprehensive clinical escalation guidance based on NEWS2 score and patient context.
        
        Args:
            total_score: Total NEWS2 score
            risk_category: Risk category (LOW, MEDIUM, HIGH)
            red_flags: List of parameters with score >= 3
            patient: Patient information for context
            
        Returns:
            Dictionary with clinical guidance for different aspects
        """
        guidance = {}
        
        # Base escalation guidance by risk category
        if risk_category == RiskCategory.LOW:
            if total_score == 0:
                guidance["escalation"] = "Continue routine care. No immediate escalation required."
                guidance["response_time"] = "Next routine observation round"
                guidance["staff_level"] = "Registered nurse or healthcare assistant"
            else:  # Score 1-2
                guidance["escalation"] = "Continue care with increased observations. Inform primary nurse of score."
                guidance["response_time"] = "Within 12 hours - next scheduled round"
                guidance["staff_level"] = "Registered nurse"
                
        elif risk_category == RiskCategory.MEDIUM:
            if 3 <= total_score <= 4:
                guidance["escalation"] = "Notify medical team. Consider bedside assessment within 1 hour."
                guidance["response_time"] = "Within 1 hour"
                guidance["staff_level"] = "Registered nurse + medical team notification"
            else:  # Score 5-6
                guidance["escalation"] = "Urgent medical review required. Senior nurse assessment mandatory."
                guidance["response_time"] = "Within 30 minutes"
                guidance["staff_level"] = "Senior registered nurse + doctor"
                
        elif risk_category == RiskCategory.HIGH:
            guidance["escalation"] = "URGENT: Emergency medical response required. Senior clinical review immediately."
            guidance["response_time"] = "IMMEDIATE - within 15 minutes"
            guidance["staff_level"] = "Senior doctor/consultant + senior nurse"
        
        # Enhanced guidance for red flag situations
        if red_flags:
            guidance["red_flag_action"] = f"RED FLAG: Critical parameter(s) detected ({', '.join(red_flags)}). Immediate clinical assessment required regardless of total score."
            if "consciousness" in red_flags:
                guidance["consciousness_alert"] = "Altered consciousness detected. Consider neurological assessment, glucose check, and urgent medical review."
            if "respiratory_rate" in red_flags:
                guidance["respiratory_alert"] = "Critical respiratory rate. Assess airway, breathing, oxygen requirement. Consider ABG analysis."
            if "spo2" in red_flags:
                guidance["oxygen_alert"] = "Critical oxygen saturation. Immediate oxygen therapy assessment and respiratory support evaluation."
            if "systolic_bp" in red_flags:
                guidance["bp_alert"] = "Critical blood pressure. Assess circulation, consider fluid resuscitation or cardiovascular support."
            if "heart_rate" in red_flags:
                guidance["cardiac_alert"] = "Critical heart rate. ECG assessment and cardiac monitoring recommended."
            if "temperature" in red_flags:
                guidance["temperature_alert"] = "Critical temperature. Assess for sepsis, infection, or environmental factors."
        
        # COPD-specific guidance
        if patient.is_copd_patient:
            guidance["copd_considerations"] = "COPD patient: Use Scale 2 SpO2 targets (88-92%). Avoid high-flow oxygen without monitoring CO2."
            if "spo2" in red_flags and patient.is_copd_patient:
                guidance["copd_oxygen_alert"] = "COPD + high SpO2: Risk of CO2 retention. Consider controlled oxygen therapy and blood gas monitoring."
        
        # Additional clinical context
        if patient.is_palliative:
            guidance["palliative_note"] = "Palliative care patient: Escalation decisions should align with care goals and advance directives."
        
        if patient.do_not_escalate:
            guidance["escalation_limit"] = "Do Not Escalate order in place. Focus on comfort measures and symptom management per care plan."
        
        # Documentation requirements
        guidance["documentation"] = f"Document NEWS2 score ({total_score}), risk category ({risk_category.value}), actions taken, and clinical response."
        
        return guidance
    
    def _validate_vital_signs(self, vital_signs: VitalSigns) -> None:
        """
        Validate that all required vital signs are present and within valid ranges.
        
        Args:
            vital_signs: VitalSigns object to validate
            
        Raises:
            MissingVitalSignsError: If required vital signs are missing
            InvalidVitalSignsError: If vital signs are outside valid ranges
        """
        if not vital_signs:
            raise MissingVitalSignsError("VitalSigns object is required")
        
        # Check for None values in required fields
        required_fields = [
            ('respiratory_rate', vital_signs.respiratory_rate),
            ('spo2', vital_signs.spo2), 
            ('temperature', vital_signs.temperature),
            ('systolic_bp', vital_signs.systolic_bp),
            ('heart_rate', vital_signs.heart_rate),
            ('consciousness', vital_signs.consciousness),
            ('on_oxygen', vital_signs.on_oxygen)
        ]
        
        missing_fields = [field for field, value in required_fields if value is None]
        if missing_fields:
            raise MissingVitalSignsError(f"Missing required vital signs: {', '.join(missing_fields)}")
        
        # The VitalSigns model already validates ranges in __post_init__,
        # but we can add additional validation here if needed
        
    def _validate_patient(self, patient: Patient) -> None:
        """
        Validate that patient information is present and complete.
        
        Args:
            patient: Patient object to validate
            
        Raises:
            MissingVitalSignsError: If patient information is missing
        """
        if not patient:
            raise MissingVitalSignsError("Patient object is required")
        
        if not patient.patient_id:
            raise MissingVitalSignsError("Patient ID is required")
        
        if patient.is_copd_patient is None:
            raise MissingVitalSignsError("Patient COPD status must be specified")