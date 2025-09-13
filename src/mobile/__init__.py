"""
Mobile-Optimized Clinical Interface for Epic 3 Story 3.4
Responsive, touch-optimized healthcare provider interface
"""

from src.mobile.mobile_interface_service import (
    MobileInterfaceService,
    ScreenSize,
    DeviceOrientation,
    TouchGesture
)
from src.mobile.mobile_forms_service import (
    MobileFormsService,
    VitalSignsFormData,
    FormValidationResult
)
from src.mobile.offline_manager import (
    OfflineManager,
    SyncStatus,
    CacheStrategy
)
from src.mobile.device_integration import (
    DeviceIntegrationService,
    BarcodeScanner,
    VoiceInput,
    CameraService
)

__all__ = [
    "MobileInterfaceService",
    "ScreenSize",
    "DeviceOrientation",
    "TouchGesture",
    "MobileFormsService",
    "VitalSignsFormData",
    "FormValidationResult",
    "OfflineManager",
    "SyncStatus",
    "CacheStrategy",
    "DeviceIntegrationService",
    "BarcodeScanner",
    "VoiceInput",
    "CameraService"
]