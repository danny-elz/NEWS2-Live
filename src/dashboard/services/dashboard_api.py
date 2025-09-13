"""
Dashboard API endpoints for Story 3.1
FastAPI routes for ward dashboard functionality
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from src.dashboard.services.ward_dashboard_service import (
    WardDashboardService,
    PatientFilter,
    PatientSortOption
)
from src.services.patient_registry import PatientRegistry
from src.services.patient_state_tracker import PatientStateTracker
from src.services.alert_generation import AlertGenerator
from src.services.news2_calculator import NEWS2Calculator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Service instances (would be dependency injected in production)
_ward_service: Optional[WardDashboardService] = None


def get_ward_service() -> WardDashboardService:
    """Get or create ward dashboard service instance"""
    global _ward_service
    if _ward_service is None:
        # Initialize dependencies
        patient_registry = PatientRegistry()
        state_tracker = PatientStateTracker(patient_registry)
        alert_service = AlertGenerator()
        news2_calculator = NEWS2Calculator()

        _ward_service = WardDashboardService(
            patient_registry=patient_registry,
            state_tracker=state_tracker,
            alert_service=alert_service,
            news2_calculator=news2_calculator
        )
    return _ward_service


@router.get("/ward/{ward_id}/overview")
async def get_ward_overview(
    ward_id: str,
    filter: Optional[str] = Query(None, description="Filter by risk level: all, low, medium, high, critical"),
    sort: Optional[str] = Query("news2_desc", description="Sort option: news2_desc, bed_number, last_update"),
    search: Optional[str] = Query(None, description="Search by patient name or bed number"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of patients to return"),
    service: WardDashboardService = Depends(get_ward_service)
) -> Dict[str, Any]:
    """
    Get comprehensive ward overview with patient tiles

    Returns:
        - Patient tiles with NEWS2 scores and alert status
        - Ward statistics and risk distribution
        - Applied filters and search parameters
    """
    try:
        # Parse filter option
        filter_option = PatientFilter.ALL
        if filter:
            try:
                filter_option = PatientFilter(filter)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid filter option: {filter}")

        # Parse sort option
        sort_option = PatientSortOption.NEWS2_HIGH_TO_LOW
        if sort:
            try:
                sort_option = PatientSortOption(sort)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid sort option: {sort}")

        # Get ward overview
        overview = await service.get_ward_overview(
            ward_id=ward_id,
            filter_option=filter_option,
            sort_option=sort_option,
            search_query=search,
            limit=limit
        )

        return overview

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ward overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ward/{ward_id}/refresh")
async def refresh_ward_dashboard(
    ward_id: str,
    service: WardDashboardService = Depends(get_ward_service)
) -> Dict[str, Any]:
    """
    Force refresh of ward dashboard data

    Clears cache and returns fresh data
    """
    try:
        overview = await service.refresh_dashboard(ward_id)
        return overview

    except Exception as e:
        logger.error(f"Error refreshing ward dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/patient/{patient_id}/preview")
async def get_patient_preview(
    patient_id: str,
    service: WardDashboardService = Depends(get_ward_service)
) -> Dict[str, Any]:
    """
    Get patient preview for navigation to detail view

    Used when clicking on patient tile to navigate to Story 3.3 detail view
    """
    try:
        preview = service.get_patient_detail_preview(patient_id)
        if not preview:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

        return preview

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient preview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ward/{ward_id}/statistics")
async def get_ward_statistics(
    ward_id: str,
    service: WardDashboardService = Depends(get_ward_service)
) -> Dict[str, Any]:
    """
    Get detailed ward statistics

    Returns:
        - Patient count by risk level
        - Average NEWS2 score
        - Alert statistics
        - Resource utilization metrics
    """
    try:
        # Get full overview to extract statistics
        overview = await service.get_ward_overview(ward_id)

        # Extract and enhance statistics
        stats = overview.get("statistics", {})
        stats["timestamp"] = datetime.utcnow().isoformat()
        stats["ward_id"] = ward_id

        # Add performance metrics
        stats["performance"] = {
            "dashboard_load_time_ms": 1500,  # Simulated
            "filter_response_time_ms": 300,  # Simulated
            "cache_hit_rate": 0.85  # Simulated
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting ward statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/wards")
async def get_available_wards() -> Dict[str, Any]:
    """
    Get list of available wards

    Returns list of wards with basic information
    """
    try:
        # Simulated ward data
        wards = [
            {
                "ward_id": "ward_a",
                "name": "Ward A - General Medicine",
                "capacity": 50,
                "current_occupancy": 42
            },
            {
                "ward_id": "ward_b",
                "name": "Ward B - Surgical",
                "capacity": 40,
                "current_occupancy": 35
            },
            {
                "ward_id": "icu",
                "name": "Intensive Care Unit",
                "capacity": 20,
                "current_occupancy": 18
            }
        ]

        return {
            "wards": wards,
            "total_wards": len(wards),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting available wards: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def dashboard_health_check() -> Dict[str, str]:
    """
    Health check endpoint for dashboard service

    Returns service status
    """
    return {
        "status": "healthy",
        "service": "ward_dashboard",
        "timestamp": datetime.utcnow().isoformat()
    }