"""
Alert Configuration Management for NEWS2 Live System

This module implements the alert rules configuration system, including:
- Ward-specific threshold management
- Escalation matrix configuration
- Admin interface for threshold configuration
- Validation for threshold ranges and conflicts
- Time-based threshold variations (day/night shifts)
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4, UUID
from dataclasses import replace

from ..models.alerts import (
    AlertLevel, AlertThreshold, EscalationMatrix, EscalationStep, EscalationRole,
    ThresholdConfigurationException, EscalationMatrixException
)
from ..services.audit import AuditLogger, AuditOperation


class ThresholdConfigurationManager:
    """
    Manages ward-specific alert threshold configurations.
    
    Responsibilities:
    - CRUD operations for alert thresholds
    - Validation of threshold ranges and conflicts
    - Time-based threshold variations
    - Admin interface support
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        # In-memory storage for thresholds (replace with database in production)
        self._thresholds: Dict[str, List[AlertThreshold]] = {}
        self._threshold_by_id: Dict[UUID, AlertThreshold] = {}
    
    async def create_threshold(
        self,
        ward_id: str,
        alert_level: AlertLevel,
        news2_min: int,
        news2_max: Optional[int],
        single_param_trigger: bool,
        active_hours: Tuple[int, int],
        user_id: str,
        enabled: bool = True
    ) -> AlertThreshold:
        """
        Create new alert threshold configuration.
        
        Args:
            ward_id: Ward identifier
            alert_level: Alert level for this threshold
            news2_min: Minimum NEWS2 score (inclusive)
            news2_max: Maximum NEWS2 score (inclusive, None for open-ended)
            single_param_trigger: Whether single parameter score=3 triggers this level
            active_hours: (start_hour, end_hour) in 24h format
            user_id: ID of user creating threshold
            enabled: Whether threshold is active
            
        Returns:
            Created AlertThreshold object
            
        Raises:
            ThresholdConfigurationException: If validation fails
        """
        try:
            # Validate threshold configuration
            await self._validate_threshold_config(
                ward_id, alert_level, news2_min, news2_max, active_hours
            )
            
            # Check for conflicts with existing thresholds
            await self._check_threshold_conflicts(
                ward_id, alert_level, news2_min, news2_max, active_hours
            )
            
            # Create threshold
            threshold = AlertThreshold(
                threshold_id=uuid4(),
                ward_id=ward_id,
                alert_level=alert_level,
                news2_min=news2_min,
                news2_max=news2_max,
                single_param_trigger=single_param_trigger,
                active_hours=active_hours,
                enabled=enabled,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Store threshold
            if ward_id not in self._thresholds:
                self._thresholds[ward_id] = []
            self._thresholds[ward_id].append(threshold)
            self._threshold_by_id[threshold.threshold_id] = threshold
            
            # Audit threshold creation
            await self._audit_threshold_operation(
                "CREATE", threshold, user_id
            )
            
            self.logger.info(
                f"Created {alert_level.value} threshold for ward {ward_id}: "
                f"NEWS2 {news2_min}-{news2_max or '∞'}"
            )
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Failed to create threshold for ward {ward_id}: {str(e)}")
            raise ThresholdConfigurationException(f"Threshold creation failed: {str(e)}")
    
    async def update_threshold(
        self,
        threshold_id: UUID,
        updates: Dict[str, Any],
        user_id: str
    ) -> AlertThreshold:
        """
        Update existing alert threshold configuration.
        
        Args:
            threshold_id: ID of threshold to update
            updates: Dictionary of fields to update
            user_id: ID of user making update
            
        Returns:
            Updated AlertThreshold object
        """
        try:
            # Get existing threshold
            old_threshold = self._threshold_by_id.get(threshold_id)
            if not old_threshold:
                raise ThresholdConfigurationException(f"Threshold {threshold_id} not found")
            
            # Create updated threshold
            new_threshold = replace(
                old_threshold,
                **updates,
                updated_at=datetime.now(timezone.utc)
            )
            
            # Validate updated configuration
            await self._validate_threshold_config(
                new_threshold.ward_id,
                new_threshold.alert_level,
                new_threshold.news2_min,
                new_threshold.news2_max,
                new_threshold.active_hours,
                exclude_threshold_id=threshold_id
            )
            
            # Check for conflicts (excluding current threshold)
            await self._check_threshold_conflicts(
                new_threshold.ward_id,
                new_threshold.alert_level,
                new_threshold.news2_min,
                new_threshold.news2_max,
                new_threshold.active_hours,
                exclude_threshold_id=threshold_id
            )
            
            # Update storage
            self._threshold_by_id[threshold_id] = new_threshold
            ward_thresholds = self._thresholds[new_threshold.ward_id]
            for i, threshold in enumerate(ward_thresholds):
                if threshold.threshold_id == threshold_id:
                    ward_thresholds[i] = new_threshold
                    break
            
            # Audit threshold update
            await self._audit_threshold_operation(
                "UPDATE", new_threshold, user_id, old_threshold
            )
            
            self.logger.info(f"Updated threshold {threshold_id} for ward {new_threshold.ward_id}")
            
            return new_threshold
            
        except Exception as e:
            self.logger.error(f"Failed to update threshold {threshold_id}: {str(e)}")
            raise ThresholdConfigurationException(f"Threshold update failed: {str(e)}")
    
    async def delete_threshold(self, threshold_id: UUID, user_id: str) -> bool:
        """
        Delete alert threshold configuration.
        
        Args:
            threshold_id: ID of threshold to delete
            user_id: ID of user deleting threshold
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get threshold to delete
            threshold = self._threshold_by_id.get(threshold_id)
            if not threshold:
                raise ThresholdConfigurationException(f"Threshold {threshold_id} not found")
            
            # Remove from storage
            del self._threshold_by_id[threshold_id]
            ward_thresholds = self._thresholds[threshold.ward_id]
            self._thresholds[threshold.ward_id] = [
                t for t in ward_thresholds if t.threshold_id != threshold_id
            ]
            
            # Audit threshold deletion
            await self._audit_threshold_operation(
                "DELETE", threshold, user_id
            )
            
            self.logger.info(f"Deleted threshold {threshold_id} for ward {threshold.ward_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete threshold {threshold_id}: {str(e)}")
            raise ThresholdConfigurationException(f"Threshold deletion failed: {str(e)}")
    
    async def get_ward_thresholds(
        self, 
        ward_id: str, 
        active_only: bool = True
    ) -> List[AlertThreshold]:
        """
        Get all thresholds for a specific ward.
        
        Args:
            ward_id: Ward identifier
            active_only: If True, only return enabled thresholds
            
        Returns:
            List of AlertThreshold objects for the ward
        """
        ward_thresholds = self._thresholds.get(ward_id, [])
        
        if active_only:
            return [t for t in ward_thresholds if t.enabled]
        else:
            return ward_thresholds.copy()
    
    async def get_active_thresholds_for_time(
        self, 
        ward_id: str, 
        check_time: Optional[datetime] = None
    ) -> List[AlertThreshold]:
        """
        Get thresholds active at specific time.
        
        Args:
            ward_id: Ward identifier
            check_time: Time to check (defaults to now)
            
        Returns:
            List of AlertThreshold objects active at the specified time
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        ward_thresholds = await self.get_ward_thresholds(ward_id, active_only=True)
        active_thresholds = []
        
        for threshold in ward_thresholds:
            if self._is_threshold_active_at_time(threshold, check_time):
                active_thresholds.append(threshold)
        
        return active_thresholds
    
    async def get_threshold_by_id(self, threshold_id: UUID) -> Optional[AlertThreshold]:
        """Get threshold by ID."""
        return self._threshold_by_id.get(threshold_id)
    
    async def list_all_wards_with_thresholds(self) -> List[str]:
        """Get list of all wards that have threshold configurations."""
        return list(self._thresholds.keys())
    
    async def _validate_threshold_config(
        self,
        ward_id: str,
        alert_level: AlertLevel,
        news2_min: int,
        news2_max: Optional[int],
        active_hours: Tuple[int, int],
        exclude_threshold_id: Optional[UUID] = None
    ):
        """Validate threshold configuration parameters."""
        
        # Validate NEWS2 score ranges
        if news2_min < 0 or news2_min > 20:
            raise ThresholdConfigurationException("NEWS2 minimum must be between 0 and 20")
        
        if news2_max is not None:
            if news2_max < 0 or news2_max > 20:
                raise ThresholdConfigurationException("NEWS2 maximum must be between 0 and 20")
            if news2_max < news2_min:
                raise ThresholdConfigurationException("NEWS2 maximum must be >= minimum")
        
        # Validate active hours
        start_hour, end_hour = active_hours
        if start_hour < 0 or start_hour > 23:
            raise ThresholdConfigurationException("Start hour must be between 0 and 23")
        if end_hour < 0 or end_hour > 23:
            raise ThresholdConfigurationException("End hour must be between 0 and 23")
        
        # Ward ID validation
        if not ward_id or not ward_id.strip():
            raise ThresholdConfigurationException("Ward ID cannot be empty")
    
    async def _check_threshold_conflicts(
        self,
        ward_id: str,
        alert_level: AlertLevel,
        news2_min: int,
        news2_max: Optional[int],
        active_hours: Tuple[int, int],
        exclude_threshold_id: Optional[UUID] = None
    ):
        """Check for conflicts with existing thresholds."""
        
        existing_thresholds = await self.get_ward_thresholds(ward_id, active_only=True)
        
        for existing in existing_thresholds:
            # Skip if this is the threshold being updated
            if exclude_threshold_id and existing.threshold_id == exclude_threshold_id:
                continue
            
            # Check for overlapping score ranges and time periods
            if self._ranges_overlap(news2_min, news2_max, existing.news2_min, existing.news2_max):
                if self._time_periods_overlap(active_hours, existing.active_hours):
                    raise ThresholdConfigurationException(
                        f"Threshold conflicts with existing {existing.alert_level.value} threshold "
                        f"(NEWS2 {existing.news2_min}-{existing.news2_max or '∞'}) "
                        f"for overlapping time period"
                    )
    
    def _ranges_overlap(
        self, 
        min1: int, max1: Optional[int], 
        min2: int, max2: Optional[int]
    ) -> bool:
        """Check if two NEWS2 score ranges overlap."""
        # Convert None (open-ended) to large number for comparison
        max1_val = max1 if max1 is not None else 999
        max2_val = max2 if max2 is not None else 999
        
        # Ranges overlap if: start1 <= end2 AND start2 <= end1
        return min1 <= max2_val and min2 <= max1_val
    
    def _time_periods_overlap(
        self, 
        period1: Tuple[int, int], 
        period2: Tuple[int, int]
    ) -> bool:
        """Check if two time periods overlap."""
        start1, end1 = period1
        start2, end2 = period2
        
        # Handle overnight periods
        def normalize_period(start: int, end: int) -> List[Tuple[int, int]]:
            if start <= end:
                return [(start, end)]
            else:
                # Overnight period: split into two ranges
                return [(start, 23), (0, end)]
        
        periods1 = normalize_period(start1, end1)
        periods2 = normalize_period(start2, end2)
        
        # Check if any period from periods1 overlaps with any period from periods2
        for p1_start, p1_end in periods1:
            for p2_start, p2_end in periods2:
                if p1_start <= p2_end and p2_start <= p1_end:
                    return True
        
        return False
    
    def _is_threshold_active_at_time(self, threshold: AlertThreshold, check_time: datetime) -> bool:
        """Check if threshold is active at specific time."""
        if not threshold.enabled:
            return False
        
        hour = check_time.hour
        start_hour, end_hour = threshold.active_hours
        
        if start_hour <= end_hour:
            # Normal range
            return start_hour <= hour < end_hour
        else:
            # Overnight range
            return hour >= start_hour or hour < end_hour
    
    async def _audit_threshold_operation(
        self,
        operation: str,
        threshold: AlertThreshold,
        user_id: str,
        old_threshold: Optional[AlertThreshold] = None
    ):
        """Audit threshold configuration operations."""
        try:
            if operation == "CREATE":
                audit_op = AuditOperation.INSERT
                old_values = None
                new_values = threshold.to_dict()
            elif operation == "UPDATE":
                audit_op = AuditOperation.UPDATE
                old_values = old_threshold.to_dict() if old_threshold else None
                new_values = threshold.to_dict()
            elif operation == "DELETE":
                audit_op = AuditOperation.DELETE
                old_values = threshold.to_dict()
                new_values = None
            else:
                return
            
            audit_entry = self.audit_logger.create_audit_entry(
                table_name="alert_thresholds",
                operation=audit_op,
                user_id=user_id,
                patient_id=None,  # Threshold operations are not patient-specific
                old_values=old_values,
                new_values=new_values
            )
            
            self.logger.debug(f"Threshold {operation} operation audited")
            
        except Exception as e:
            self.logger.error(f"Failed to audit threshold {operation}: {str(e)}")
            # Don't fail the operation due to audit issues


class EscalationMatrixManager:
    """
    Manages escalation matrix configurations for different wards and alert levels.
    
    Responsibilities:
    - CRUD operations for escalation matrices
    - Validation of escalation steps and timing
    - Role hierarchy management
    - Ward-specific escalation customization
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        # In-memory storage for escalation matrices
        self._matrices: Dict[Tuple[str, AlertLevel], EscalationMatrix] = {}
        self._matrix_by_id: Dict[UUID, EscalationMatrix] = {}
    
    async def create_escalation_matrix(
        self,
        ward_id: str,
        alert_level: AlertLevel,
        escalation_steps: List[EscalationStep],
        user_id: str,
        enabled: bool = True
    ) -> EscalationMatrix:
        """
        Create new escalation matrix configuration.
        
        Args:
            ward_id: Ward identifier
            alert_level: Alert level for this matrix
            escalation_steps: List of escalation steps in order
            user_id: ID of user creating matrix
            enabled: Whether matrix is active
            
        Returns:
            Created EscalationMatrix object
        """
        try:
            # Validate escalation steps
            await self._validate_escalation_steps(escalation_steps)
            
            # Check if matrix already exists for this ward/level
            matrix_key = (ward_id, alert_level)
            if matrix_key in self._matrices:
                raise EscalationMatrixException(
                    f"Escalation matrix already exists for ward {ward_id} and alert level {alert_level.value}"
                )
            
            # Create matrix
            matrix = EscalationMatrix(
                matrix_id=uuid4(),
                ward_id=ward_id,
                alert_level=alert_level,
                escalation_steps=escalation_steps.copy(),
                enabled=enabled,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Store matrix
            self._matrices[matrix_key] = matrix
            self._matrix_by_id[matrix.matrix_id] = matrix
            
            # Audit matrix creation
            await self._audit_matrix_operation("CREATE", matrix, user_id)
            
            self.logger.info(
                f"Created escalation matrix for ward {ward_id}, alert level {alert_level.value} "
                f"with {len(escalation_steps)} steps"
            )
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Failed to create escalation matrix: {str(e)}")
            raise EscalationMatrixException(f"Escalation matrix creation failed: {str(e)}")
    
    async def update_escalation_matrix(
        self,
        matrix_id: UUID,
        escalation_steps: Optional[List[EscalationStep]] = None,
        enabled: Optional[bool] = None,
        user_id: str = "SYSTEM"
    ) -> EscalationMatrix:
        """
        Update existing escalation matrix.
        
        Args:
            matrix_id: ID of matrix to update
            escalation_steps: New escalation steps (optional)
            enabled: New enabled status (optional)
            user_id: ID of user making update
            
        Returns:
            Updated EscalationMatrix object
        """
        try:
            # Get existing matrix
            old_matrix = self._matrix_by_id.get(matrix_id)
            if not old_matrix:
                raise EscalationMatrixException(f"Escalation matrix {matrix_id} not found")
            
            # Prepare updates
            updates = {"updated_at": datetime.now(timezone.utc)}
            if escalation_steps is not None:
                await self._validate_escalation_steps(escalation_steps)
                updates["escalation_steps"] = escalation_steps.copy()
            if enabled is not None:
                updates["enabled"] = enabled
            
            # Create updated matrix
            new_matrix = replace(old_matrix, **updates)
            
            # Update storage
            matrix_key = (new_matrix.ward_id, new_matrix.alert_level)
            self._matrices[matrix_key] = new_matrix
            self._matrix_by_id[matrix_id] = new_matrix
            
            # Audit matrix update
            await self._audit_matrix_operation("UPDATE", new_matrix, user_id, old_matrix)
            
            self.logger.info(f"Updated escalation matrix {matrix_id}")
            
            return new_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to update escalation matrix {matrix_id}: {str(e)}")
            raise EscalationMatrixException(f"Escalation matrix update failed: {str(e)}")
    
    async def get_escalation_matrix(
        self, 
        ward_id: str, 
        alert_level: AlertLevel
    ) -> Optional[EscalationMatrix]:
        """
        Get escalation matrix for specific ward and alert level.
        
        Args:
            ward_id: Ward identifier
            alert_level: Alert level
            
        Returns:
            EscalationMatrix if found, None otherwise
        """
        matrix_key = (ward_id, alert_level)
        matrix = self._matrices.get(matrix_key)
        
        if matrix and matrix.enabled:
            return matrix
        return None
    
    async def list_ward_matrices(self, ward_id: str) -> List[EscalationMatrix]:
        """Get all escalation matrices for a ward."""
        ward_matrices = []
        for (matrix_ward_id, alert_level), matrix in self._matrices.items():
            if matrix_ward_id == ward_id:
                ward_matrices.append(matrix)
        return ward_matrices
    
    async def delete_escalation_matrix(self, matrix_id: UUID, user_id: str) -> bool:
        """
        Delete escalation matrix.
        
        Args:
            matrix_id: ID of matrix to delete
            user_id: ID of user deleting matrix
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get matrix to delete
            matrix = self._matrix_by_id.get(matrix_id)
            if not matrix:
                raise EscalationMatrixException(f"Escalation matrix {matrix_id} not found")
            
            # Remove from storage
            matrix_key = (matrix.ward_id, matrix.alert_level)
            del self._matrices[matrix_key]
            del self._matrix_by_id[matrix_id]
            
            # Audit matrix deletion
            await self._audit_matrix_operation("DELETE", matrix, user_id)
            
            self.logger.info(f"Deleted escalation matrix {matrix_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete escalation matrix {matrix_id}: {str(e)}")
            raise EscalationMatrixException(f"Escalation matrix deletion failed: {str(e)}")
    
    async def _validate_escalation_steps(self, escalation_steps: List[EscalationStep]):
        """Validate escalation steps configuration."""
        if not escalation_steps:
            raise EscalationMatrixException("Escalation matrix must have at least one step")
        
        if len(escalation_steps) > 10:
            raise EscalationMatrixException("Escalation matrix cannot have more than 10 steps")
        
        # Validate each step
        seen_delays = set()
        for i, step in enumerate(escalation_steps):
            # Validate delay
            if step.delay_minutes < 0:
                raise EscalationMatrixException(f"Step {i+1}: Delay cannot be negative")
            if step.delay_minutes > 1440:  # 24 hours
                raise EscalationMatrixException(f"Step {i+1}: Delay cannot exceed 24 hours")
            
            # Check for duplicate delays (except 0 which can appear once at start)
            if step.delay_minutes in seen_delays and step.delay_minutes != 0:
                raise EscalationMatrixException(f"Step {i+1}: Duplicate delay time {step.delay_minutes} minutes")
            seen_delays.add(step.delay_minutes)
            
            # Validate attempts and retry interval
            if step.max_attempts < 1 or step.max_attempts > 10:
                raise EscalationMatrixException(f"Step {i+1}: Max attempts must be between 1 and 10")
            if step.retry_interval_minutes < 1 or step.retry_interval_minutes > 60:
                raise EscalationMatrixException(f"Step {i+1}: Retry interval must be between 1 and 60 minutes")
        
        # Validate escalation timing order
        delays = [step.delay_minutes for step in escalation_steps]
        if delays != sorted(delays):
            raise EscalationMatrixException("Escalation steps must be ordered by increasing delay time")
    
    async def _audit_matrix_operation(
        self,
        operation: str,
        matrix: EscalationMatrix,
        user_id: str,
        old_matrix: Optional[EscalationMatrix] = None
    ):
        """Audit escalation matrix operations."""
        try:
            if operation == "CREATE":
                audit_op = AuditOperation.INSERT
                old_values = None
                new_values = matrix.to_dict()
            elif operation == "UPDATE":
                audit_op = AuditOperation.UPDATE
                old_values = old_matrix.to_dict() if old_matrix else None
                new_values = matrix.to_dict()
            elif operation == "DELETE":
                audit_op = AuditOperation.DELETE
                old_values = matrix.to_dict()
                new_values = None
            else:
                return
            
            audit_entry = self.audit_logger.create_audit_entry(
                table_name="escalation_matrices",
                operation=audit_op,
                user_id=user_id,
                patient_id=None,
                old_values=old_values,
                new_values=new_values
            )
            
            self.logger.debug(f"Escalation matrix {operation} operation audited")
            
        except Exception as e:
            self.logger.error(f"Failed to audit escalation matrix {operation}: {str(e)}")


class AlertConfigurationService:
    """
    High-level service for alert configuration management.
    
    Provides unified interface for threshold and escalation matrix management.
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.threshold_manager = ThresholdConfigurationManager(audit_logger)
        self.escalation_manager = EscalationMatrixManager(audit_logger)
        self.logger = logging.getLogger(__name__)
    
    async def setup_default_ward_configuration(
        self, 
        ward_id: str, 
        ward_type: str = "general",
        user_id: str = "SYSTEM"
    ) -> Dict[str, Any]:
        """
        Set up default alert configuration for a new ward.
        
        Args:
            ward_id: Ward identifier
            ward_type: Type of ward (general, icu, emergency, etc.)
            user_id: ID of user setting up configuration
            
        Returns:
            Dictionary with created thresholds and escalation matrices
        """
        try:
            self.logger.info(f"Setting up default configuration for ward {ward_id} (type: {ward_type})")
            
            # Create default thresholds based on ward type
            thresholds = await self._create_default_thresholds(ward_id, ward_type, user_id)
            
            # Create default escalation matrices
            matrices = await self._create_default_escalation_matrices(ward_id, ward_type, user_id)
            
            result = {
                "ward_id": ward_id,
                "ward_type": ward_type,
                "thresholds_created": len(thresholds),
                "matrices_created": len(matrices),
                "thresholds": [t.to_dict() for t in thresholds],
                "escalation_matrices": [m.to_dict() for m in matrices]
            }
            
            self.logger.info(
                f"Successfully set up default configuration for ward {ward_id}: "
                f"{len(thresholds)} thresholds, {len(matrices)} escalation matrices"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to setup default configuration for ward {ward_id}: {str(e)}")
            raise ThresholdConfigurationException(f"Default configuration setup failed: {str(e)}")
    
    async def _create_default_thresholds(
        self, 
        ward_id: str, 
        ward_type: str, 
        user_id: str
    ) -> List[AlertThreshold]:
        """Create default thresholds based on ward type."""
        
        # Define ward-type specific configurations
        if ward_type == "icu":
            # ICU has more sensitive thresholds
            threshold_configs = [
                (AlertLevel.CRITICAL, 5, None, True, (0, 24)),  # Lower critical threshold
                (AlertLevel.HIGH, 3, 4, False, (0, 24)),
                (AlertLevel.MEDIUM, 2, 2, False, (0, 24)),  # More sensitive monitoring
                (AlertLevel.LOW, 1, 1, False, (0, 24))
            ]
        elif ward_type == "emergency":
            # Emergency department - standard thresholds with 24/7 active hours
            threshold_configs = [
                (AlertLevel.CRITICAL, 7, None, True, (0, 24)),
                (AlertLevel.HIGH, 5, 6, False, (0, 24)),
                (AlertLevel.MEDIUM, 3, 4, False, (0, 24)),
                (AlertLevel.LOW, 1, 2, False, (0, 24))
            ]
        else:  # general ward
            # Standard NEWS2 thresholds with day/night variations
            threshold_configs = [
                (AlertLevel.CRITICAL, 7, None, True, (0, 24)),
                (AlertLevel.HIGH, 5, 6, False, (0, 24)),
                (AlertLevel.MEDIUM, 3, 4, False, (6, 22)),  # Only during day hours
                (AlertLevel.LOW, 1, 2, False, (6, 22))      # Only during day hours
            ]
        
        thresholds = []
        for alert_level, news2_min, news2_max, single_param, active_hours in threshold_configs:
            threshold = await self.threshold_manager.create_threshold(
                ward_id=ward_id,
                alert_level=alert_level,
                news2_min=news2_min,
                news2_max=news2_max,
                single_param_trigger=single_param,
                active_hours=active_hours,
                user_id=user_id
            )
            thresholds.append(threshold)
        
        return thresholds
    
    async def _create_default_escalation_matrices(
        self, 
        ward_id: str, 
        ward_type: str, 
        user_id: str
    ) -> List[EscalationMatrix]:
        """Create default escalation matrices based on ward type."""
        
        matrices = []
        
        # Define escalation patterns by ward type
        if ward_type == "icu":
            # ICU - faster escalation with more roles
            escalation_configs = {
                AlertLevel.CRITICAL: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 5),   # Faster escalation
                    EscalationStep(EscalationRole.DOCTOR, 10),
                    EscalationStep(EscalationRole.CONSULTANT, 20)     # Add consultant level
                ],
                AlertLevel.HIGH: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 15),
                    EscalationStep(EscalationRole.DOCTOR, 30)
                ],
                AlertLevel.MEDIUM: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 30)
                ]
            }
        elif ward_type == "emergency":
            # Emergency - immediate escalation patterns
            escalation_configs = {
                AlertLevel.CRITICAL: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.DOCTOR, 5),         # Skip charge nurse
                    EscalationStep(EscalationRole.RAPID_RESPONSE, 15)
                ],
                AlertLevel.HIGH: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.DOCTOR, 10)
                ],
                AlertLevel.MEDIUM: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0)
                ]
            }
        else:  # general ward
            # Standard escalation as defined in story requirements
            escalation_configs = {
                AlertLevel.CRITICAL: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 15),
                    EscalationStep(EscalationRole.DOCTOR, 30),
                    EscalationStep(EscalationRole.RAPID_RESPONSE, 45)
                ],
                AlertLevel.HIGH: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 30),
                    EscalationStep(EscalationRole.DOCTOR, 60)
                ],
                AlertLevel.MEDIUM: [
                    EscalationStep(EscalationRole.WARD_NURSE, 0),
                    EscalationStep(EscalationRole.CHARGE_NURSE, 60)
                ]
            }
        
        # Create escalation matrices
        for alert_level, steps in escalation_configs.items():
            matrix = await self.escalation_manager.create_escalation_matrix(
                ward_id=ward_id,
                alert_level=alert_level,
                escalation_steps=steps,
                user_id=user_id
            )
            matrices.append(matrix)
        
        return matrices
    
    async def get_complete_ward_configuration(
        self, 
        ward_id: str
    ) -> Dict[str, Any]:
        """
        Get complete alert configuration for a ward.
        
        Args:
            ward_id: Ward identifier
            
        Returns:
            Complete configuration including thresholds and escalation matrices
        """
        try:
            thresholds = await self.threshold_manager.get_ward_thresholds(ward_id)
            matrices = await self.escalation_manager.list_ward_matrices(ward_id)
            
            return {
                "ward_id": ward_id,
                "thresholds": [t.to_dict() for t in thresholds],
                "escalation_matrices": [m.to_dict() for m in matrices],
                "active_thresholds_count": len([t for t in thresholds if t.enabled]),
                "active_matrices_count": len([m for m in matrices if m.enabled])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ward configuration for {ward_id}: {str(e)}")
            raise ThresholdConfigurationException(f"Ward configuration retrieval failed: {str(e)}")