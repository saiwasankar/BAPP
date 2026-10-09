"""Package 2: E-Gate Processing (kiosk, face verification, access decision, gate)."""
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, Depends

from .. import real_face, services
from ..database import get_db
from ..schemas import (
    EnrollFaceRequest, FaceVerifyRequest, FaceVerifyResponse, GateActionResponse, GateOut,
    KioskOut, LiveFaceDetectRequest, LiveFaceDetectResponse, MessageOut, SessionOut,
)

router = APIRouter(prefix="/api/egate", tags=["2. E-Gate Processing"])


@router.get("/kiosks", response_model=List[KioskOut], summary="List kiosks and their gates")
def list_kiosks(conn: sqlite3.Connection = Depends(get_db)):
    return services.list_kiosks(conn)


@router.get("/gates", response_model=List[GateOut], summary="List gate states")
def list_gates(conn: sqlite3.Connection = Depends(get_db)):
    return services.list_gates(conn)


@router.get("/sessions/{session_id}", response_model=SessionOut,
            summary="Current state of a verification session")
def get_session(session_id: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.get_session(conn, session_id)


@router.post("/sessions/{session_id}/face-verification", response_model=FaceVerifyResponse,
             summary="Step 2: one liveness + face-match attempt (3 attempts max, then escalation)")
def verify_face(session_id: str, payload: Optional[FaceVerifyRequest] = None,
                conn: sqlite3.Connection = Depends(get_db)):
    frame_id = payload.frame_id if payload else None
    image_base64 = payload.image_base64 if payload else None
    return services.verify_face(conn, session_id, frame_id=frame_id, image_base64=image_base64)


@router.post("/live-face-detect", response_model=LiveFaceDetectResponse,
             summary="Real-time camera frame AI face & liveness detection for HUD preview")
def detect_live_face(payload: LiveFaceDetectRequest):
    return real_face.detect_live_face(payload.image_base64)


@router.post("/passengers/{passenger_id}/enroll-face", response_model=MessageOut,
             summary="Enroll live camera face photo as biometric template for passenger")
def enroll_passenger_face(passenger_id: str, payload: EnrollFaceRequest,
                          conn: sqlite3.Connection = Depends(get_db)):
    result = real_face.process_face_image(conn, passenger_id, payload.image_base64)
    if not result.get("face_detected"):
        return MessageOut(message=f"Enrollment failed: {result.get('message', 'No face detected')}")
    return MessageOut(message=result.get("message", "Face enrolled successfully"))


@router.post("/sessions/{session_id}/decision", response_model=GateActionResponse,
             summary="Step 4: gate authorisation or escalation to an officer")
def decide_access(session_id: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.decide_access(conn, session_id)


@router.post("/sessions/{session_id}/passage", response_model=GateActionResponse,
             summary="Confirm the passenger passed through; closes the gate and the transaction")
def confirm_passage(session_id: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.confirm_passage(conn, session_id)

