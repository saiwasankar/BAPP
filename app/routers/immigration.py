"""Package 3: Immigration & Records (records, watchlist, officer review and override)."""
import sqlite3
from typing import List

from fastapi import APIRouter, Depends

from .. import services
from ..database import get_db
from ..schemas import (
    GateActionResponse, ImmigrationCheckResponse, ImmigrationRecordOut, OverrideRequest,
    SessionOut, WatchlistStatusOut,
)

router = APIRouter(prefix="/api/immigration", tags=["3. Immigration & Records"])


@router.get("/records/{passport_number}", response_model=ImmigrationRecordOut,
            summary="Immigration database lookup by passport number")
def get_record(passport_number: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.get_immigration_record(conn, passport_number)


@router.get("/watchlist/{passport_number}", response_model=WatchlistStatusOut,
            summary="Watchlist status by passport number (hit flag and highest risk only)")
def get_watchlist(passport_number: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.get_watchlist_status(conn, passport_number)


@router.post("/sessions/{session_id}/checks", response_model=ImmigrationCheckResponse,
             summary="Step 3: immigration lookup and watchlist check for a session")
def run_checks(session_id: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.run_immigration_checks(conn, session_id)


@router.get("/reviews", response_model=List[SessionOut],
            summary="Sessions waiting for an immigration officer")
def pending_reviews(conn: sqlite3.Connection = Depends(get_db)):
    return services.list_pending_reviews(conn)


@router.post("/sessions/{session_id}/override", response_model=GateActionResponse,
             summary="Step 4 (manual route): officer approves or denies an escalated session")
def officer_override(session_id: str, payload: OverrideRequest,
                     conn: sqlite3.Connection = Depends(get_db)):
    return services.officer_override(conn, session_id, payload.officer_id, payload.approve,
                                     payload.notes)
