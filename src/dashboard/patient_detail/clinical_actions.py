"""
Clinical Actions Service for Story 3.3
Handles quick clinical actions and interventions
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from dataclasses import dataclass

from src.dashboard.patient_detail.patient_detail_service import ClinicalIntervention, InterventionOutcome

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of clinical actions"""
    REPOSITION = "reposition"
    OXYGEN_THERAPY = "oxygen_therapy"
    VITAL_SIGNS_CHECK = "vital_signs_check"
    MEDICATION = "medication"
    ESCALATION = "escalation"
    NOTIFICATION = "notification"
    DOCUMENTATION = "documentation"


class EscalationLevel(Enum):
    """Escalation levels"""
    CHARGE_NURSE = "charge_nurse"
    PHYSICIAN = "physician"
    RAPID_RESPONSE = "rapid_response"
    CODE_BLUE = "code_blue"


@dataclass
class QuickAction:
    """Quick action template"""
    action_id: str
    name: str
    action_type: ActionType
    description: str
    icon: str
    estimated_duration: int  # minutes
    requires_documentation: bool = True
    default_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.default_parameters is None:
            self.default_parameters = {}


@dataclass
class ActionExecution:
    """Action execution record"""
    execution_id: str
    action_id: str
    patient_id: str
    performer: str
    timestamp: datetime
    parameters: Dict[str, Any]
    status: str  # pending, completed, failed
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None
    outcome: Optional[InterventionOutcome] = None


class ClinicalActionsService:
    """Service for managing clinical actions and quick interventions"""

    def __init__(self):
        self.quick_actions = self._initialize_quick_actions()
        self._execution_history: Dict[str, List[ActionExecution]] = {}
        self._escalation_contacts = self._initialize_contacts()

    def _initialize_quick_actions(self) -> Dict[str, QuickAction]:
        """Initialize predefined quick actions"""
        actions = [
            QuickAction(
                action_id="reposition_patient",
                name="Reposition Patient",
                action_type=ActionType.REPOSITION,
                description="Reposition patient to improve comfort and circulation",
                icon="rotate-3d",
                estimated_duration=5,
                default_parameters={
                    "position": "left_side",
                    "frequency": "every_2_hours"
                }
            ),
            QuickAction(
                action_id="start_oxygen",
                name="Start Oxygen Therapy",
                action_type=ActionType.OXYGEN_THERAPY,
                description="Initiate supplemental oxygen therapy",
                icon="lungs",
                estimated_duration=10,
                default_parameters={
                    "flow_rate": "2L/min",
                    "delivery_method": "nasal_cannula",
                    "target_spo2": "95%"
                }
            ),
            QuickAction(
                action_id="vital_signs_recheck",
                name="Vital Signs Recheck",
                action_type=ActionType.VITAL_SIGNS_CHECK,
                description="Perform immediate vital signs assessment",
                icon="heartbeat",
                estimated_duration=8,
                default_parameters={
                    "include_news2": True,
                    "frequency": "every_30_min_x3"
                }
            ),
            QuickAction(
                action_id="pain_medication",
                name="Administer Pain Relief",
                action_type=ActionType.MEDICATION,
                description="Administer prescribed pain medication",
                icon="pills",
                estimated_duration=15,
                default_parameters={
                    "medication": "paracetamol",
                    "dose": "1g",
                    "route": "oral"
                }
            ),
            QuickAction(
                action_id="escalate_physician",
                name="Contact Physician",
                action_type=ActionType.ESCALATION,
                description="Escalate patient concerns to attending physician",
                icon="phone",
                estimated_duration=5,
                requires_documentation=False,
                default_parameters={
                    "urgency": "routine",
                    "include_vitals": True
                }
            ),
            QuickAction(
                action_id="rapid_response",
                name="Rapid Response Team",
                action_type=ActionType.ESCALATION,
                description="Activate rapid response team",
                icon="emergency",
                estimated_duration=2,
                requires_documentation=False,
                default_parameters={
                    "reason": "clinical_deterioration",
                    "location": "auto_detect"
                }
            )
        ]

        return {action.action_id: action for action in actions}

    def _initialize_contacts(self) -> Dict[str, Dict[str, str]]:
        """Initialize escalation contacts"""
        return {
            "charge_nurse": {
                "name": "Charge Nurse",
                "phone": "ext-1234",
                "pager": "1234",
                "availability": "24/7"
            },
            "attending_physician": {
                "name": "Dr. Smith",
                "phone": "ext-5678",
                "pager": "5678",
                "availability": "8am-6pm"
            },
            "on_call_physician": {
                "name": "On-Call Physician",
                "phone": "ext-9999",
                "pager": "9999",
                "availability": "after_hours"
            },
            "rapid_response": {
                "name": "Rapid Response Team",
                "phone": "ext-7777",
                "pager": "7777",
                "availability": "24/7"
            }
        }

    def get_quick_actions(self, patient_id: str = None) -> List[Dict[str, Any]]:
        """
        Get available quick actions for patient

        Args:
            patient_id: Patient identifier (for future customization)

        Returns:
            List of available quick actions
        """
        actions = []

        for action in self.quick_actions.values():
            action_data = {
                "action_id": action.action_id,
                "name": action.name,
                "description": action.description,
                "icon": action.icon,
                "estimated_duration": action.estimated_duration,
                "requires_documentation": action.requires_documentation,
                "parameters": action.default_parameters.copy()
            }
            actions.append(action_data)

        return actions

    async def execute_action(self,
                            patient_id: str,
                            action_id: str,
                            performer: str,
                            parameters: Optional[Dict[str, Any]] = None,
                            notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a clinical action

        Args:
            patient_id: Patient identifier
            action_id: Action to execute
            performer: Person performing action
            parameters: Action parameters
            notes: Additional notes

        Returns:
            Execution result
        """
        if action_id not in self.quick_actions:
            return {
                "success": False,
                "error": f"Unknown action: {action_id}"
            }

        action = self.quick_actions[action_id]

        try:
            # Create execution record
            execution = ActionExecution(
                execution_id=f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{patient_id}",
                action_id=action_id,
                patient_id=patient_id,
                performer=performer,
                timestamp=datetime.now(),
                parameters=parameters or action.default_parameters.copy(),
                status="pending",
                notes=notes
            )

            # Execute action based on type
            result = await self._execute_action_by_type(action, execution)

            # Record execution
            if patient_id not in self._execution_history:
                self._execution_history[patient_id] = []
            self._execution_history[patient_id].append(execution)

            return {
                "success": result["success"],
                "execution_id": execution.execution_id,
                "timestamp": execution.timestamp.isoformat(),
                "estimated_completion": (execution.timestamp + timedelta(minutes=action.estimated_duration)).isoformat(),
                "message": result.get("message", "Action initiated successfully"),
                "next_steps": result.get("next_steps", [])
            }

        except Exception as e:
            logger.error(f"Error executing action {action_id} for patient {patient_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to execute action: {str(e)}"
            }

    async def _execute_action_by_type(self, action: QuickAction, execution: ActionExecution) -> Dict[str, Any]:
        """Execute action based on its type"""
        if action.action_type == ActionType.REPOSITION:
            return await self._execute_reposition(execution)
        elif action.action_type == ActionType.OXYGEN_THERAPY:
            return await self._execute_oxygen_therapy(execution)
        elif action.action_type == ActionType.VITAL_SIGNS_CHECK:
            return await self._execute_vital_signs_check(execution)
        elif action.action_type == ActionType.MEDICATION:
            return await self._execute_medication(execution)
        elif action.action_type == ActionType.ESCALATION:
            return await self._execute_escalation(execution)
        else:
            return {
                "success": True,
                "message": "Action logged successfully"
            }

    async def _execute_reposition(self, execution: ActionExecution) -> Dict[str, Any]:
        """Execute patient repositioning"""
        position = execution.parameters.get("position", "left_side")
        frequency = execution.parameters.get("frequency", "every_2_hours")

        execution.status = "completed"
        execution.duration_minutes = 5

        return {
            "success": True,
            "message": f"Patient repositioned to {position}",
            "next_steps": [
                f"Monitor patient comfort",
                f"Next repositioning due: {frequency}",
                "Document skin integrity"
            ]
        }

    async def _execute_oxygen_therapy(self, execution: ActionExecution) -> Dict[str, Any]:
        """Execute oxygen therapy initiation"""
        flow_rate = execution.parameters.get("flow_rate", "2L/min")
        method = execution.parameters.get("delivery_method", "nasal_cannula")
        target = execution.parameters.get("target_spo2", "95%")

        execution.status = "completed"
        execution.duration_minutes = 10

        return {
            "success": True,
            "message": f"Oxygen therapy started at {flow_rate} via {method}",
            "next_steps": [
                f"Monitor SpO2 - target: {target}",
                "Recheck vital signs in 15 minutes",
                "Assess patient comfort with device"
            ]
        }

    async def _execute_vital_signs_check(self, execution: ActionExecution) -> Dict[str, Any]:
        """Execute vital signs check"""
        include_news2 = execution.parameters.get("include_news2", True)
        frequency = execution.parameters.get("frequency", "every_30_min_x3")

        execution.status = "completed"
        execution.duration_minutes = 8

        message = "Vital signs check completed"
        if include_news2:
            message += " with NEWS2 calculation"

        return {
            "success": True,
            "message": message,
            "next_steps": [
                f"Continue monitoring: {frequency}",
                "Compare with previous readings",
                "Escalate if deterioration noted"
            ]
        }

    async def _execute_medication(self, execution: ActionExecution) -> Dict[str, Any]:
        """Execute medication administration"""
        medication = execution.parameters.get("medication", "medication")
        dose = execution.parameters.get("dose", "as prescribed")
        route = execution.parameters.get("route", "oral")

        execution.status = "completed"
        execution.duration_minutes = 15

        return {
            "success": True,
            "message": f"Administered {medication} {dose} ({route})",
            "next_steps": [
                "Monitor for therapeutic effect",
                "Assess for adverse reactions",
                "Document in medication record"
            ]
        }

    async def _execute_escalation(self, execution: ActionExecution) -> Dict[str, Any]:
        """Execute escalation to clinical staff"""
        if execution.action_id == "rapid_response":
            return await self._activate_rapid_response(execution)
        else:
            return await self._contact_clinician(execution)

    async def _activate_rapid_response(self, execution: ActionExecution) -> Dict[str, Any]:
        """Activate rapid response team"""
        reason = execution.parameters.get("reason", "clinical_deterioration")

        execution.status = "completed"
        execution.duration_minutes = 2

        return {
            "success": True,
            "message": "Rapid Response Team activated",
            "next_steps": [
                "Team dispatched - ETA 2-5 minutes",
                "Continue monitoring patient",
                "Prepare handoff summary",
                f"Activation reason: {reason}"
            ]
        }

    async def _contact_clinician(self, execution: ActionExecution) -> Dict[str, Any]:
        """Contact physician or charge nurse"""
        urgency = execution.parameters.get("urgency", "routine")
        include_vitals = execution.parameters.get("include_vitals", True)

        execution.status = "completed"
        execution.duration_minutes = 5

        contact_info = "Contact information sent to clinician"
        if include_vitals:
            contact_info += " with current vital signs"

        return {
            "success": True,
            "message": f"Clinician contacted - {urgency} priority",
            "next_steps": [
                contact_info,
                "Await clinical response",
                "Document conversation when complete"
            ]
        }

    def get_execution_history(self, patient_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get execution history for patient

        Args:
            patient_id: Patient identifier
            hours: Hours of history to retrieve

        Returns:
            List of execution records
        """
        if patient_id not in self._execution_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_executions = [
            exec for exec in self._execution_history[patient_id]
            if exec.timestamp >= cutoff_time
        ]

        # Convert to dict format
        history = []
        for execution in recent_executions:
            action = self.quick_actions.get(execution.action_id)
            history.append({
                "execution_id": execution.execution_id,
                "action_name": action.name if action else execution.action_id,
                "performer": execution.performer,
                "timestamp": execution.timestamp.isoformat(),
                "status": execution.status,
                "duration_minutes": execution.duration_minutes,
                "notes": execution.notes,
                "parameters": execution.parameters
            })

        return sorted(history, key=lambda x: x["timestamp"], reverse=True)

    def get_escalation_contacts(self) -> Dict[str, Dict[str, str]]:
        """Get escalation contact information"""
        return self._escalation_contacts.copy()

    async def generate_handoff_summary(self, patient_id: str, recipient: str) -> Dict[str, Any]:
        """
        Generate handoff summary for patient

        Args:
            patient_id: Patient identifier
            recipient: Recipient of handoff

        Returns:
            Handoff summary data
        """
        # Get recent actions
        recent_actions = self.get_execution_history(patient_id, hours=8)

        # Create handoff summary
        summary = {
            "patient_id": patient_id,
            "handoff_time": datetime.now().isoformat(),
            "recipient": recipient,
            "recent_actions": recent_actions[:10],  # Last 10 actions
            "key_points": self._generate_key_points(recent_actions),
            "pending_tasks": self._get_pending_tasks(patient_id),
            "escalations": [
                action for action in recent_actions
                if action.get("action_name", "").lower() in ["contact physician", "rapid response team"]
            ]
        }

        return summary

    def _generate_key_points(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Generate key points from recent actions"""
        key_points = []

        # Look for significant interventions
        for action in actions[:5]:  # Recent 5 actions
            name = action.get("action_name", "")
            timestamp = action.get("timestamp", "")

            if "oxygen" in name.lower():
                key_points.append(f"Oxygen therapy initiated ({timestamp})")
            elif "rapid response" in name.lower():
                key_points.append(f"Rapid response activated ({timestamp})")
            elif "medication" in name.lower():
                key_points.append(f"Pain medication administered ({timestamp})")

        return key_points

    def _get_pending_tasks(self, patient_id: str) -> List[str]:
        """Get pending tasks for patient"""
        # In production, would check scheduled tasks and follow-ups
        return [
            "Monitor oxygen saturation",
            "Vital signs due in 30 minutes",
            "Medication review pending"
        ]