"""
Mobile Interface Service for Story 3.4
Responsive, touch-optimized clinical interface management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ScreenSize(Enum):
    """Screen size categories for responsive design"""
    SMARTPHONE = "smartphone"  # 320px - 767px
    TABLET = "tablet"         # 768px - 1024px
    DESKTOP = "desktop"       # 1025px+


class DeviceOrientation(Enum):
    """Device orientation states"""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class TouchGesture(Enum):
    """Supported touch gestures"""
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_ZOOM = "pinch_zoom"
    LONG_PRESS = "long_press"


@dataclass
class LayoutConfiguration:
    """Layout configuration for different screen sizes"""
    screen_size: ScreenSize
    orientation: DeviceOrientation
    grid_columns: int
    card_width: str
    font_size_base: int
    touch_target_size: int
    spacing_unit: int


@dataclass
class MobilePatientCard:
    """Mobile-optimized patient card data"""
    patient_id: str
    patient_name: str
    bed_number: str
    news2_score: int
    risk_level: str
    status_color: str
    last_update: datetime
    alerts_count: int
    quick_actions: List[str] = field(default_factory=list)

    def to_mobile_dict(self, screen_size: ScreenSize) -> Dict[str, Any]:
        """Convert to mobile-optimized dictionary"""
        base_data = {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "bed_number": self.bed_number,
            "news2_score": self.news2_score,
            "risk_level": self.risk_level,
            "status_color": self.status_color,
            "last_update": self.last_update.strftime("%H:%M"),
            "alerts_count": self.alerts_count,
            "quick_actions": self.quick_actions[:3 if screen_size == ScreenSize.SMARTPHONE else 6]
        }

        # Add display optimizations based on screen size
        if screen_size == ScreenSize.SMARTPHONE:
            base_data["compact_mode"] = True
            base_data["show_details"] = False
        else:
            base_data["compact_mode"] = False
            base_data["show_details"] = True

        return base_data


@dataclass
class NavigationState:
    """Mobile navigation state management"""
    current_view: str
    previous_views: List[str] = field(default_factory=list)
    modal_stack: List[str] = field(default_factory=list)
    swipe_enabled: bool = True

    def push_view(self, view: str):
        """Navigate to new view"""
        if self.current_view:
            self.previous_views.append(self.current_view)
        self.current_view = view

    def pop_view(self) -> Optional[str]:
        """Navigate back to previous view"""
        if self.previous_views:
            previous = self.current_view
            self.current_view = self.previous_views.pop()
            return previous
        return None

    def push_modal(self, modal: str):
        """Show modal dialog"""
        self.modal_stack.append(modal)

    def pop_modal(self) -> Optional[str]:
        """Close current modal"""
        return self.modal_stack.pop() if self.modal_stack else None


class MobileInterfaceService:
    """Service for mobile-optimized clinical interface management"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._layout_configs = self._initialize_layouts()
        self._navigation_state = NavigationState(current_view="ward_overview")
        self._touch_handlers = {}
        self._offline_queue = []

    def _initialize_layouts(self) -> Dict[Tuple[ScreenSize, DeviceOrientation], LayoutConfiguration]:
        """Initialize responsive layout configurations"""
        configs = {}

        # Smartphone Portrait
        configs[(ScreenSize.SMARTPHONE, DeviceOrientation.PORTRAIT)] = LayoutConfiguration(
            screen_size=ScreenSize.SMARTPHONE,
            orientation=DeviceOrientation.PORTRAIT,
            grid_columns=1,
            card_width="100%",
            font_size_base=16,
            touch_target_size=44,
            spacing_unit=8
        )

        # Smartphone Landscape
        configs[(ScreenSize.SMARTPHONE, DeviceOrientation.LANDSCAPE)] = LayoutConfiguration(
            screen_size=ScreenSize.SMARTPHONE,
            orientation=DeviceOrientation.LANDSCAPE,
            grid_columns=2,
            card_width="48%",
            font_size_base=14,
            touch_target_size=40,
            spacing_unit=6
        )

        # Tablet Portrait
        configs[(ScreenSize.TABLET, DeviceOrientation.PORTRAIT)] = LayoutConfiguration(
            screen_size=ScreenSize.TABLET,
            orientation=DeviceOrientation.PORTRAIT,
            grid_columns=2,
            card_width="48%",
            font_size_base=18,
            touch_target_size=48,
            spacing_unit=12
        )

        # Tablet Landscape
        configs[(ScreenSize.TABLET, DeviceOrientation.LANDSCAPE)] = LayoutConfiguration(
            screen_size=ScreenSize.TABLET,
            orientation=DeviceOrientation.LANDSCAPE,
            grid_columns=3,
            card_width="32%",
            font_size_base=16,
            touch_target_size=44,
            spacing_unit=10
        )

        return configs

    async def get_mobile_ward_overview(self, ward_id: str, screen_size: ScreenSize,
                                     orientation: DeviceOrientation) -> Dict[str, Any]:
        """Get mobile-optimized ward overview"""
        try:
            layout = self._layout_configs.get((screen_size, orientation))
            if not layout:
                layout = self._layout_configs[(ScreenSize.TABLET, DeviceOrientation.PORTRAIT)]

            # This would integrate with the ward dashboard service
            # For demo purposes, create sample mobile patient cards
            sample_patients = self._create_sample_mobile_patients()

            mobile_overview = {
                "ward_id": ward_id,
                "layout": {
                    "grid_columns": layout.grid_columns,
                    "card_width": layout.card_width,
                    "font_size": layout.font_size_base,
                    "touch_target_size": layout.touch_target_size,
                    "spacing": layout.spacing_unit
                },
                "patients": [
                    patient.to_mobile_dict(screen_size)
                    for patient in sample_patients
                ],
                "quick_stats": {
                    "total_patients": len(sample_patients),
                    "high_risk": sum(1 for p in sample_patients if p.risk_level == "high"),
                    "pending_alerts": sum(p.alerts_count for p in sample_patients)
                },
                "navigation": {
                    "current_view": self._navigation_state.current_view,
                    "can_swipe": self._navigation_state.swipe_enabled,
                    "has_modals": bool(self._navigation_state.modal_stack)
                },
                "last_updated": datetime.now().isoformat()
            }

            return mobile_overview

        except Exception as e:
            self.logger.error(f"Error getting mobile ward overview: {e}")
            return self._get_error_response("Failed to load ward overview")

    def _create_sample_mobile_patients(self) -> List[MobilePatientCard]:
        """Create sample mobile patient cards"""
        return [
            MobilePatientCard(
                patient_id="P001",
                patient_name="Alice Johnson",
                bed_number="A01",
                news2_score=1,
                risk_level="low",
                status_color="#22c55e",
                last_update=datetime.now() - timedelta(minutes=15),
                alerts_count=0,
                quick_actions=["vitals", "notes", "call"]
            ),
            MobilePatientCard(
                patient_id="P002",
                patient_name="Bob Smith",
                bed_number="A02",
                news2_score=7,
                risk_level="high",
                status_color="#ef4444",
                last_update=datetime.now() - timedelta(minutes=5),
                alerts_count=2,
                quick_actions=["escalate", "vitals", "oxygen", "notes"]
            ),
            MobilePatientCard(
                patient_id="P003",
                patient_name="Carol Davis",
                bed_number="A03",
                news2_score=3,
                risk_level="medium",
                status_color="#f59e0b",
                last_update=datetime.now() - timedelta(minutes=30),
                alerts_count=1,
                quick_actions=["vitals", "recheck", "notes"]
            )
        ]

    async def handle_touch_gesture(self, gesture: TouchGesture, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle touch gestures for mobile interface"""
        try:
            if gesture == TouchGesture.SWIPE_LEFT:
                return await self._handle_swipe_navigation("next", context)
            elif gesture == TouchGesture.SWIPE_RIGHT:
                return await self._handle_swipe_navigation("previous", context)
            elif gesture == TouchGesture.SWIPE_UP:
                return await self._handle_pull_to_refresh(context)
            elif gesture == TouchGesture.LONG_PRESS:
                return await self._handle_quick_actions(context)
            elif gesture == TouchGesture.PINCH_ZOOM:
                return await self._handle_zoom_gesture(context)
            else:
                return {"success": True, "action": "gesture_ignored"}

        except Exception as e:
            self.logger.error(f"Error handling touch gesture {gesture}: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_swipe_navigation(self, direction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle swipe navigation between views"""
        if not self._navigation_state.swipe_enabled:
            return {"success": False, "error": "Swipe navigation disabled"}

        current_view = context.get("current_view", "")
        patient_list = context.get("patient_list", [])
        current_index = context.get("current_index", 0)

        if current_view == "patient_detail" and patient_list:
            if direction == "next" and current_index < len(patient_list) - 1:
                new_index = current_index + 1
                next_patient = patient_list[new_index]
                return {
                    "success": True,
                    "action": "navigate_patient",
                    "patient_id": next_patient["patient_id"],
                    "index": new_index
                }
            elif direction == "previous" and current_index > 0:
                new_index = current_index - 1
                prev_patient = patient_list[new_index]
                return {
                    "success": True,
                    "action": "navigate_patient",
                    "patient_id": prev_patient["patient_id"],
                    "index": new_index
                }

        return {"success": False, "error": "No valid swipe target"}

    async def _handle_pull_to_refresh(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull-to-refresh gesture"""
        view = context.get("current_view", "")

        if view == "ward_overview":
            return {
                "success": True,
                "action": "refresh_ward",
                "message": "Refreshing ward data..."
            }
        elif view == "patient_detail":
            patient_id = context.get("patient_id")
            return {
                "success": True,
                "action": "refresh_patient",
                "patient_id": patient_id,
                "message": "Refreshing patient data..."
            }

        return {"success": False, "error": "Refresh not available"}

    async def _handle_quick_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle long press for quick actions menu"""
        patient_id = context.get("patient_id")
        if not patient_id:
            return {"success": False, "error": "No patient context"}

        actions = [
            {"id": "vital_signs", "name": "Record Vitals", "icon": "heart", "urgent": False},
            {"id": "escalate", "name": "Escalate Care", "icon": "alert", "urgent": True},
            {"id": "medication", "name": "Medications", "icon": "pill", "urgent": False},
            {"id": "notes", "name": "Add Notes", "icon": "edit", "urgent": False}
        ]

        return {
            "success": True,
            "action": "show_quick_menu",
            "patient_id": patient_id,
            "actions": actions
        }

    async def _handle_zoom_gesture(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pinch-to-zoom on charts and timelines"""
        chart_type = context.get("chart_type", "")
        zoom_level = context.get("zoom_level", 1.0)

        if chart_type in ["timeline", "trend_chart"]:
            return {
                "success": True,
                "action": "update_zoom",
                "chart_type": chart_type,
                "zoom_level": max(0.5, min(3.0, zoom_level))  # Constrain zoom
            }

        return {"success": False, "error": "Zoom not supported"}

    async def get_mobile_patient_detail(self, patient_id: str, screen_size: ScreenSize) -> Dict[str, Any]:
        """Get mobile-optimized patient detail view"""
        try:
            # This would integrate with the patient detail service
            # For demo purposes, create sample mobile patient detail

            detail = {
                "patient_id": patient_id,
                "patient_name": "Sample Patient",
                "bed_number": "A02",
                "age": 65,
                "current_news2": 7,
                "risk_level": "high",
                "status_color": "#ef4444",
                "last_update": datetime.now().strftime("%H:%M:%S"),
                "mobile_layout": {
                    "compact": screen_size == ScreenSize.SMARTPHONE,
                    "sections": self._get_mobile_sections(screen_size),
                    "action_buttons": self._get_mobile_action_buttons(screen_size)
                },
                "vital_signs": {
                    "respiratory_rate": {"value": 22, "status": "high", "trend": "up"},
                    "spo2": {"value": 92, "status": "low", "trend": "down"},
                    "heart_rate": {"value": 95, "status": "normal", "trend": "stable"},
                    "temperature": {"value": 38.5, "status": "high", "trend": "up"}
                },
                "recent_timeline": self._get_mobile_timeline(patient_id, screen_size),
                "quick_actions": [
                    {"id": "vitals", "name": "Record Vitals", "primary": True},
                    {"id": "escalate", "name": "Escalate", "primary": True, "urgent": True},
                    {"id": "notes", "name": "Notes", "primary": False}
                ]
            }

            return detail

        except Exception as e:
            self.logger.error(f"Error getting mobile patient detail for {patient_id}: {e}")
            return self._get_error_response("Failed to load patient detail")

    def _get_mobile_sections(self, screen_size: ScreenSize) -> List[str]:
        """Get sections to display based on screen size"""
        if screen_size == ScreenSize.SMARTPHONE:
            return ["vital_signs", "alerts", "quick_actions"]
        else:
            return ["vital_signs", "timeline", "alerts", "interventions", "quick_actions"]

    def _get_mobile_action_buttons(self, screen_size: ScreenSize) -> List[Dict[str, Any]]:
        """Get action buttons optimized for screen size"""
        buttons = [
            {"id": "vitals", "label": "Vitals", "icon": "heart", "size": "large"},
            {"id": "escalate", "label": "Escalate", "icon": "alert", "size": "large", "variant": "danger"}
        ]

        if screen_size != ScreenSize.SMARTPHONE:
            buttons.extend([
                {"id": "medication", "label": "Meds", "icon": "pill", "size": "medium"},
                {"id": "notes", "label": "Notes", "icon": "edit", "size": "medium"}
            ])

        return buttons

    def _get_mobile_timeline(self, patient_id: str, screen_size: ScreenSize) -> Dict[str, Any]:
        """Get mobile-optimized timeline data"""
        data_points = 6 if screen_size == ScreenSize.SMARTPHONE else 12

        return {
            "data_points": data_points,
            "time_range": "4 hours" if screen_size == ScreenSize.SMARTPHONE else "8 hours",
            "chart_type": "simplified" if screen_size == ScreenSize.SMARTPHONE else "detailed",
            "trend": "worsening",
            "confidence": 0.8
        }

    def _get_error_response(self, message: str) -> Dict[str, Any]:
        """Get standardized error response"""
        return {
            "error": True,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "retry_action": "refresh"
        }

    def set_navigation_state(self, view: str):
        """Set current navigation state"""
        self._navigation_state.push_view(view)

    def enable_swipe_navigation(self, enabled: bool):
        """Enable or disable swipe navigation"""
        self._navigation_state.swipe_enabled = enabled

    async def optimize_for_network(self, network_type: str) -> Dict[str, Any]:
        """Optimize interface for different network conditions"""
        optimizations = {}

        if network_type in ["2g", "slow-2g"]:
            optimizations = {
                "image_quality": "low",
                "chart_detail": "minimal",
                "refresh_interval": 60000,  # 1 minute
                "preload_data": False,
                "compress_responses": True
            }
        elif network_type == "3g":
            optimizations = {
                "image_quality": "medium",
                "chart_detail": "standard",
                "refresh_interval": 30000,  # 30 seconds
                "preload_data": True,
                "compress_responses": True
            }
        else:  # 4g, wifi
            optimizations = {
                "image_quality": "high",
                "chart_detail": "full",
                "refresh_interval": 15000,  # 15 seconds
                "preload_data": True,
                "compress_responses": False
            }

        return {
            "success": True,
            "network_type": network_type,
            "optimizations": optimizations
        }

    async def get_mobile_metrics(self) -> Dict[str, Any]:
        """Get mobile interface performance metrics"""
        return {
            "interface_metrics": {
                "load_time_ms": 1200,  # Simulated
                "navigation_time_ms": 250,
                "memory_usage_mb": 85,
                "battery_impact": "low"
            },
            "usage_stats": {
                "active_sessions": 1,
                "gestures_processed": 45,
                "offline_actions_queued": len(self._offline_queue)
            },
            "performance_score": 95
        }