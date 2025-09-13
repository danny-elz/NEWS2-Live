"""
Ward Dashboard Service for Story 3.1
Provides real-time ward overview with patient status indicators
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

from src.models.patient import Patient
from src.models.patient_state import PatientState
from src.models.news2 import NEWS2Result, RiskCategory
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker
from src.services.alert_generation import AlertGenerator
from src.services.news2_calculator import NEWS2Calculator

logger = logging.getLogger(__name__)


class PatientSortOption(Enum):
    """Sorting options for patient display"""
    NEWS2_HIGH_TO_LOW = "news2_desc"
    BED_NUMBER = "bed_number"
    LAST_UPDATE = "last_update"


class PatientFilter(Enum):
    """Filter options for patient display"""
    ALL = "all"
    LOW_RISK = "low"  # NEWS2 0-2
    MEDIUM_RISK = "medium"  # NEWS2 3-4
    HIGH_RISK = "high"  # NEWS2 5-6
    CRITICAL_RISK = "critical"  # NEWS2 7+


class WardDashboardService:
    """Service for managing ward dashboard data and operations"""

    def __init__(self,
                 patient_registry: PatientRegistry,
                 state_tracker: PatientStateTracker,
                 alert_service: AlertGenerator,
                 news2_calculator: NEWS2Calculator):
        self.patient_registry = patient_registry
        self.state_tracker = state_tracker
        self.alert_service = alert_service
        self.news2_calculator = news2_calculator
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=5)  # Cache for 5 seconds

    async def get_ward_overview(self,
                                ward_id: str,
                                filter_option: PatientFilter = PatientFilter.ALL,
                                sort_option: PatientSortOption = PatientSortOption.NEWS2_HIGH_TO_LOW,
                                search_query: Optional[str] = None,
                                limit: int = 50) -> Dict[str, Any]:
        """
        Get comprehensive ward overview with patient tiles

        Args:
            ward_id: Ward identifier
            filter_option: Filter patients by risk level
            sort_option: Sort order for patients
            search_query: Search by name or bed number
            limit: Maximum number of patients to return

        Returns:
            Ward overview data with patient tiles and statistics
        """
        try:
            # Check cache
            cache_key = f"{ward_id}_{filter_option.value}_{sort_option.value}_{search_query}_{limit}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"Returning cached ward overview for {ward_id}")
                return self._cache[cache_key]

            # Get all patients in ward
            patients = await self._get_ward_patients(ward_id)

            # Apply search filter
            if search_query:
                patients = self._filter_by_search(patients, search_query)

            # Get patient tiles with NEWS2 and alert data
            patient_tiles = await self._create_patient_tiles(patients)

            # Apply risk level filter
            if filter_option != PatientFilter.ALL:
                patient_tiles = self._filter_by_risk(patient_tiles, filter_option)

            # Sort patients
            patient_tiles = self._sort_patients(patient_tiles, sort_option)

            # Apply limit
            patient_tiles = patient_tiles[:limit]

            # Calculate ward statistics
            statistics = self._calculate_ward_statistics(patient_tiles)

            # Prepare response
            response = {
                "ward_id": ward_id,
                "timestamp": datetime.utcnow().isoformat(),
                "patient_count": len(patient_tiles),
                "patients": patient_tiles,
                "statistics": statistics,
                "filters_applied": {
                    "risk_level": filter_option.value,
                    "sort_by": sort_option.value,
                    "search": search_query
                }
            }

            # Update cache
            self._update_cache(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error getting ward overview: {e}")
            raise

    async def _get_ward_patients(self, ward_id: str) -> List[Patient]:
        """Get all patients in a specific ward"""
        all_patients = self.patient_registry.get_all_patients()
        # Filter by ward
        ward_patients = [
            p for p in all_patients
            if p.ward_id == ward_id
        ]
        return ward_patients

    async def _create_patient_tiles(self, patients: List[Patient]) -> List[Dict[str, Any]]:
        """Create patient tile data with NEWS2 and alert information"""
        tiles = []

        for patient in patients:
            try:
                # Get patient state
                state = self.state_tracker.get_patient_state(patient.patient_id)
                if not state:
                    continue

                # Get latest NEWS2 score
                news2_score = None
                risk_level = RiskCategory.LOW
                if state.current_vitals:
                    news2_result = self.news2_calculator.calculate_news2(state.current_vitals)
                    if news2_result:
                        news2_score = news2_result.total_score
                        risk_level = news2_result.risk_level

                # Get alert status from Epic 2
                alert_status = await self._get_alert_status(patient.patient_id)

                # Determine tile color based on NEWS2 score
                tile_color = self._get_tile_color(news2_score)

                # Create tile data
                tile = {
                    "patient_id": patient.patient_id,
                    "patient_name": f"Patient {patient.patient_id}",  # Generate name from ID
                    "bed_number": patient.bed_number,
                    "news2_score": news2_score,
                    "risk_level": risk_level.value,
                    "tile_color": tile_color,
                    "last_update": state.last_update.isoformat() if state.last_update else None,
                    "alert_status": alert_status,
                    "admission_date": patient.admission_date.isoformat() if patient.admission_date else None,
                    "attending_physician": "Dr. Unknown",  # Not in model
                    "primary_nurse": patient.assigned_nurse_id
                }
                tiles.append(tile)

            except Exception as e:
                logger.error(f"Error creating tile for patient {patient.patient_id}: {e}")
                continue

        return tiles

    async def _get_alert_status(self, patient_id: str) -> Dict[str, Any]:
        """Get alert status from Epic 2 alert service"""
        try:
            # TODO: Integrate with AlertGenerator when patient alert retrieval is available
            # For now, return simulated alert status
            active_alerts = []

            # Count alerts by status
            active_count = sum(1 for a in active_alerts if a.status == "active")
            suppressed_count = sum(1 for a in active_alerts if a.status == "suppressed")
            acknowledged_count = sum(1 for a in active_alerts if a.status == "acknowledged")

            return {
                "active": active_count,
                "suppressed": suppressed_count,
                "acknowledged": acknowledged_count,
                "total": len(active_alerts),
                "has_critical": any(a.priority == "critical" for a in active_alerts)
            }
        except Exception as e:
            logger.error(f"Error getting alert status for patient {patient_id}: {e}")
            return {
                "active": 0,
                "suppressed": 0,
                "acknowledged": 0,
                "total": 0,
                "has_critical": False
            }

    def _get_tile_color(self, news2_score: Optional[int]) -> str:
        """Determine tile color based on NEWS2 score"""
        if news2_score is None:
            return "gray"
        elif news2_score <= 2:
            return "green"
        elif news2_score <= 4:
            return "yellow"
        elif news2_score <= 6:
            return "orange"
        else:
            return "red"

    def _filter_by_search(self, patients: List[Patient], query: str) -> List[Patient]:
        """Filter patients by name or bed number search"""
        query_lower = query.lower()
        filtered = []

        for patient in patients:
            # Check patient ID (as name proxy)
            if query_lower in patient.patient_id.lower():
                filtered.append(patient)
                continue

            # Check bed number
            bed_number = str(patient.bed_number).lower()
            if query_lower in bed_number:
                filtered.append(patient)

        return filtered

    def _filter_by_risk(self, tiles: List[Dict[str, Any]], filter_option: PatientFilter) -> List[Dict[str, Any]]:
        """Filter patient tiles by risk level"""
        filtered = []

        for tile in tiles:
            score = tile.get("news2_score")
            if score is None:
                continue

            if filter_option == PatientFilter.LOW_RISK and score <= 2:
                filtered.append(tile)
            elif filter_option == PatientFilter.MEDIUM_RISK and 3 <= score <= 4:
                filtered.append(tile)
            elif filter_option == PatientFilter.HIGH_RISK and 5 <= score <= 6:
                filtered.append(tile)
            elif filter_option == PatientFilter.CRITICAL_RISK and score >= 7:
                filtered.append(tile)

        return filtered

    def _sort_patients(self, tiles: List[Dict[str, Any]], sort_option: PatientSortOption) -> List[Dict[str, Any]]:
        """Sort patient tiles based on selected option"""
        if sort_option == PatientSortOption.NEWS2_HIGH_TO_LOW:
            # Sort by NEWS2 score descending, None values at end
            return sorted(tiles, key=lambda x: (x.get("news2_score") is None, -(x.get("news2_score") or 0)))

        elif sort_option == PatientSortOption.BED_NUMBER:
            # Sort by bed number
            return sorted(tiles, key=lambda x: str(x.get("bed_number", "")))

        elif sort_option == PatientSortOption.LAST_UPDATE:
            # Sort by last update time, most recent first
            return sorted(tiles, key=lambda x: x.get("last_update", ""), reverse=True)

        return tiles

    def _calculate_ward_statistics(self, tiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ward-level statistics from patient tiles"""
        total_patients = len(tiles)

        # Count by risk level
        risk_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }

        # Calculate average NEWS2
        news2_scores = []
        alert_counts = {"active": 0, "total": 0}

        for tile in tiles:
            # Risk level counts
            risk_level = tile.get("risk_level", "low")
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1

            # NEWS2 scores for average
            score = tile.get("news2_score")
            if score is not None:
                news2_scores.append(score)

            # Alert counts
            alert_status = tile.get("alert_status", {})
            alert_counts["active"] += alert_status.get("active", 0)
            alert_counts["total"] += alert_status.get("total", 0)

        # Calculate average NEWS2
        avg_news2 = sum(news2_scores) / len(news2_scores) if news2_scores else 0

        return {
            "total_patients": total_patients,
            "risk_distribution": risk_counts,
            "average_news2": round(avg_news2, 1),
            "patients_with_alerts": sum(1 for t in tiles if t.get("alert_status", {}).get("total", 0) > 0),
            "active_alerts": alert_counts["active"],
            "total_alerts": alert_counts["total"]
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache:
            return False

        if self._cache_timestamp is None:
            return False

        age = datetime.utcnow() - self._cache_timestamp
        return age < self._cache_ttl

    def _update_cache(self, cache_key: str, data: Dict[str, Any]):
        """Update cache with new data"""
        self._cache[cache_key] = data
        self._cache_timestamp = datetime.utcnow()

    async def refresh_dashboard(self, ward_id: str) -> Dict[str, Any]:
        """Force refresh of dashboard data, clearing cache"""
        # Clear cache for this ward
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{ward_id}_")]
        for key in keys_to_remove:
            del self._cache[key]

        # Get fresh data
        return await self.get_ward_overview(ward_id)

    def get_patient_detail_preview(self, patient_id: str) -> Dict[str, Any]:
        """Get preview data for patient detail navigation (Story 3.3 integration)"""
        try:
            patient = self.patient_registry.get_patient(patient_id)
            if not patient:
                return {}

            state = self.state_tracker.get_patient_state(patient_id)
            if not state:
                return {}

            return {
                "patient_id": patient_id,
                "patient_name": f"Patient {patient.patient_id}",
                "current_state": "stable",  # Default state
                "navigation_url": f"/patient/{patient_id}/detail"  # For Story 3.3
            }
        except Exception as e:
            logger.error(f"Error getting patient detail preview: {e}")
            return {}