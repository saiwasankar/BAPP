"""Package 4: Administration & Audit (thresholds, session overview, audit trail)."""
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from .. import audit, config, services
from ..database import get_db, reset_database
from ..errors import ForbiddenError
from ..schemas import (
    AuditPage, AuditVerifyOut, MessageOut, SessionOut, ThresholdConfigOut, ThresholdUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["4. Administration & Audit"])


@router.get("/config", response_model=ThresholdConfigOut, summary="Current thresholds")
def get_config(conn: sqlite3.Connection = Depends(get_db)):
    return services.get_config(conn)


@router.put("/config", response_model=ThresholdConfigOut,
            summary="Update match/liveness thresholds or the retry cap (audited)")
def update_config(payload: ThresholdUpdate, conn: sqlite3.Connection = Depends(get_db)):
    changes = payload.model_dump(exclude={"admin_id"}, exclude_none=True)
    return services.update_thresholds(conn, payload.admin_id, changes)


@router.get("/sessions", response_model=List[SessionOut], summary="All verification sessions")
def list_sessions(
    state: Optional[str] = Query(None, description="Filter by session state"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
):
    return services.list_sessions(conn, state.upper() if state else None, limit, offset)


@router.get("/audit-log", response_model=AuditPage, summary="Query the audit trail")
def audit_log(
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    actor_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
):
    return audit.list_events(conn, session_id, event_type, actor_id, limit, offset)


@router.get("/audit-log/verify", response_model=AuditVerifyOut,
            summary="Verify the audit hash chain has not been tampered with")
def verify_audit_log(conn: sqlite3.Connection = Depends(get_db)):
    return audit.verify_chain(conn)


@router.post("/demo/reset", response_model=MessageOut,
             summary="Demo only: wipe everything and re-seed the database")
def demo_reset():
    if not config.ALLOW_DEMO_RESET:
        raise ForbiddenError("Demo reset is disabled (BAPPS_ALLOW_RESET=0).")
    reset_database()
    return {"message": "Database reset and re-seeded."}
