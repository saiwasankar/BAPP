"""Pydantic request and response models for the BAPPS REST API."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------ passengers & documents

class PassengerSummary(BaseModel):
    passenger_id: str
    full_name: str
    nationality: str
    date_of_birth: str
    passport_number: str
    demo_scenario: str


class PassportOut(BaseModel):
    passport_number: str
    issuing_country: str
    expiry_date: str


class BoardingPassOut(BaseModel):
    booking_id: str
    passport_number: str
    flight_number: str
    gate_number: str
    departure_time: str


class TemplateInfo(BaseModel):
    template_id: str
    enrollment_date: str


class PassengerDetail(PassengerSummary):
    passport: PassportOut
    boarding_passes: List[BoardingPassOut]
    biometric_template: Optional[TemplateInfo] = None


class ScanRequest(BaseModel):
    passport_number: Optional[str] = Field(None, min_length=5, max_length=20)
    booking_id: Optional[str] = Field(None, min_length=3, max_length=20)
    kiosk_id: str = Field("KIOSK-01", min_length=3, max_length=20)
    passport_image_b64: Optional[str] = Field(None, description="Base64 data URI of uploaded passport document.")

    @model_validator(mode="after")
    def require_a_document(self) -> "ScanRequest":
        if not (self.passport_number or self.booking_id or self.passport_image_b64):
            raise ValueError("Provide passport_number, booking_id, or upload a passport document.")
        return self


# ------------------------------------------------------------------ e-gate processing

SessionState = Literal[
    "AWAITING_FACE", "FACE_VERIFIED", "CHECKS_COMPLETE", "AWAITING_OFFICER",
    "GATE_OPEN", "DENIED", "COMPLETED",
]


class SessionOut(BaseModel):
    session_id: str
    passenger_id: str
    passenger_name: str
    passport_number: str
    booking_id: Optional[str]
    kiosk_id: str
    gate_id: Optional[str]
    state: SessionState
    next_action: str
    escalation_reason: Optional[str]
    document_issues: List[str]
    attempts: int
    max_attempts: int
    liveness_score: Optional[float]
    match_score: Optional[float]
    match_passed: Optional[bool]
    passport_photo_b64: Optional[str] = None
    immigration_status: Optional[str]
    entry_permitted: Optional[bool]
    watchlist_hit: Optional[bool]
    watchlist_risk: Optional[str]
    decision: Optional[str]
    decision_reasons: List[str]
    officer_id: Optional[str]
    officer_notes: Optional[str]
    created_at: str
    updated_at: str


class GateOut(BaseModel):
    gate_id: str
    status: Literal["OPEN", "CLOSED"]


class KioskOut(BaseModel):
    kiosk_id: str
    location: str
    status: str
    gate_id: str
    gate_status: str


class FaceVerifyRequest(BaseModel):
    frame_id: Optional[str] = Field(None, max_length=64, description="Optional camera frame id for the audit trail.")
    image_base64: Optional[str] = Field(None, description="Base64 data URI or raw image from live camera.")


class FaceAttemptOut(BaseModel):
    attempt_number: int
    liveness_score: Optional[float]
    liveness_passed: bool
    match_score: Optional[float]
    match_passed: Optional[bool]
    attempts_remaining: int
    outcome: Literal["RETRY", "FACE_VERIFIED", "ESCALATED"]
    engine_type: Optional[str] = "AI_REAL_MEDIAPIPE"
    comparison_source: Optional[str] = "UPLOADED_PASSPORT_PORTRAIT"
    passport_photo_b64: Optional[str] = None
    face_details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class FaceVerifyResponse(BaseModel):
    session: SessionOut
    attempt: FaceAttemptOut


class UploadPassportRequest(BaseModel):
    passport_image_b64: str = Field(..., description="Base64 data URI or raw image of passport.")
    passport_number: Optional[str] = None
    booking_id: Optional[str] = None
    kiosk_id: str = "KIOSK-01"


class UploadPassportResponse(BaseModel):
    session: SessionOut
    face_detected: bool
    face_thumbnail_b64: Optional[str] = None
    message: str


class LiveFaceDetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded live webcam snapshot.")


class LiveFaceDetectResponse(BaseModel):
    face_detected: bool
    num_faces: int = 0
    landmarks_count: Optional[int] = 0
    liveness_score: Optional[float] = None
    liveness_passed: Optional[bool] = None
    depth_score: Optional[float] = None
    pose_score: Optional[float] = None
    eye_open_score: Optional[float] = None
    lighting_score: Optional[float] = None
    spoof_probability: Optional[float] = None
    bbox: Optional[Dict[str, Any]] = None
    head_angles: Optional[Dict[str, Any]] = None
    guidance: Optional[str] = None
    message: Optional[str] = None


class EnrollFaceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded face photo.")




class GateActionResponse(BaseModel):
    session: SessionOut
    authorized: bool
    gate: Optional[GateOut]
    reasons: List[str]


# ------------------------------------------------------------------ immigration & records

class ImmigrationRecordOut(BaseModel):
    passport_number: str
    visa_status: str
    entry_permitted: bool
    notes: str


class WatchlistStatusOut(BaseModel):
    passport_number: str
    hit: bool
    highest_risk: Optional[str]
    entry_count: int


class ImmigrationCheckOut(BaseModel):
    passport_number: str
    visa_status: str
    entry_permitted: bool
    record_found: bool


class ImmigrationCheckResponse(BaseModel):
    session: SessionOut
    immigration: ImmigrationCheckOut
    watchlist: WatchlistStatusOut


class OverrideRequest(BaseModel):
    officer_id: str = Field(..., min_length=3, max_length=20)
    approve: bool
    notes: str = Field(..., min_length=10, max_length=500,
                       description="Mandatory justification recorded in the audit log.")


# ------------------------------------------------------------------ administration & audit

class ThresholdConfigOut(BaseModel):
    match_threshold: float
    liveness_threshold: float
    max_attempts: int


class ThresholdUpdate(BaseModel):
    admin_id: str = Field(..., min_length=3, max_length=20)
    match_threshold: Optional[float] = Field(None, gt=0, le=1)
    liveness_threshold: Optional[float] = Field(None, gt=0, le=1)
    max_attempts: Optional[int] = Field(None, ge=1, le=10)

    @model_validator(mode="after")
    def require_a_change(self) -> "ThresholdUpdate":
        if (self.match_threshold is None and self.liveness_threshold is None
                and self.max_attempts is None):
            raise ValueError("Provide at least one of match_threshold, liveness_threshold, max_attempts.")
        return self


class AuditEntry(BaseModel):
    log_id: int
    timestamp: str
    session_id: Optional[str]
    event_type: str
    actor_type: str
    actor_id: str
    decision: Optional[str]
    details: Dict[str, Any]
    prev_hash: str
    entry_hash: str


class AuditPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[AuditEntry]


class AuditVerifyOut(BaseModel):
    valid: bool
    entries_checked: int
    first_invalid_log_id: Optional[int]
    message: str


class MessageOut(BaseModel):
    message: str
