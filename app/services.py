"""Business logic for the BAPPS verification pipeline.

Pipeline (matches the activity and sequence diagrams):

  1. start_session        scan passport / boarding pass, verify documents
  2. verify_face          liveness + face match, up to max_attempts attempts
  3. run_immigration_checks   immigration record lookup + watchlist check
  4. decide_access        auto-clear (gate opens) or escalate to an officer
     officer_override     officer approves (gate opens) or denies
     confirm_passage      passenger walked through, gate closes
  5. every step appends to the hash-chained audit log; terminal states also
     write a TRANSACTION_CLOSED summary event.

Session states:
  AWAITING_FACE -> FACE_VERIFIED -> CHECKS_COMPLETE -> GATE_OPEN -> COMPLETED
  any escalation -> AWAITING_OFFICER -> GATE_OPEN (approved) or DENIED
"""
import json
import sqlite3
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from . import config, face_engine, real_face
from .audit import log_event
from .database import transaction
from .errors import ConflictError, ForbiddenError, NotFoundError

NEXT_ACTION = {
    "AWAITING_FACE": "CAPTURE_FACE",
    "FACE_VERIFIED": "RUN_IMMIGRATION_CHECKS",
    "CHECKS_COMPLETE": "REQUEST_ACCESS_DECISION",
    "AWAITING_OFFICER": "OFFICER_REVIEW",
    "GATE_OPEN": "PROCEED_THROUGH_GATE",
    "DENIED": "NONE",
    "COMPLETED": "NONE",
}


# --------------------------------------------------------------------------- helpers

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().upper()
    return value or None


def _opt_bool(value: Optional[int]) -> Optional[bool]:
    return None if value is None else bool(value)


def get_thresholds(conn: sqlite3.Connection) -> Dict[str, Any]:
    values = dict(config.DEFAULT_THRESHOLDS)
    for row in conn.execute("SELECT key, value FROM threshold_config"):
        values[row["key"]] = row["value"]
    return {
        "match_threshold": float(values["match_threshold"]),
        "liveness_threshold": float(values["liveness_threshold"]),
        "max_attempts": int(values["max_attempts"]),
    }


def _get_session_row(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM verification_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"Session {session_id} does not exist.")
    return row


def _require_state(session: sqlite3.Row, expected: str, action: str) -> None:
    if session["state"] != expected:
        raise ConflictError(
            f"Session {session['session_id']} is in state {session['state']}; "
            f"{action} requires state {expected}."
        )


def _update_session(conn: sqlite3.Connection, session_id: str, **fields: Any) -> None:
    fields["updated_at"] = _now()
    assignments = ", ".join(f"{column} = ?" for column in fields)
    conn.execute(
        f"UPDATE verification_sessions SET {assignments} WHERE session_id = ?",
        (*fields.values(), session_id),
    )


def serialize_session(conn: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    passenger = conn.execute(
        "SELECT full_name FROM passengers WHERE passenger_id = ?", (row["passenger_id"],)
    ).fetchone()
    return {
        "session_id": row["session_id"],
        "passenger_id": row["passenger_id"],
        "passenger_name": passenger["full_name"] if passenger else "",
        "passport_number": row["passport_number"],
        "booking_id": row["booking_id"],
        "kiosk_id": row["kiosk_id"],
        "gate_id": row["gate_id"],
        "state": row["state"],
        "next_action": NEXT_ACTION[row["state"]],
        "escalation_reason": row["escalation_reason"],
        "document_issues": json.loads(row["document_issues"]),
        "attempts": row["attempts"],
        "max_attempts": get_thresholds(conn)["max_attempts"],
        "liveness_score": row["liveness_score"],
        "match_score": row["match_score"],
        "match_passed": _opt_bool(row["match_passed"]),
        "passport_photo_b64": row["passport_photo_b64"] if "passport_photo_b64" in row.keys() else None,
        "immigration_status": row["immigration_status"],
        "entry_permitted": _opt_bool(row["entry_permitted"]),
        "watchlist_hit": _opt_bool(row["watchlist_hit"]),
        "watchlist_risk": row["watchlist_risk"],
        "decision": row["decision"],
        "decision_reasons": json.loads(row["decision_reasons"]),
        "officer_id": row["officer_id"],
        "officer_notes": row["officer_notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _session_dict(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    return serialize_session(conn, _get_session_row(conn, session_id))


def _gate_dict(conn: sqlite3.Connection, gate_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not gate_id:
        return None
    row = conn.execute("SELECT gate_id, status FROM gates WHERE gate_id = ?", (gate_id,)).fetchone()
    return {"gate_id": row["gate_id"], "status": row["status"]} if row else None


def _gate_action_response(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    session = _session_dict(conn, session_id)
    gate = _gate_dict(conn, session["gate_id"])
    return {
        "session": session,
        "authorized": bool(gate and gate["status"] == "OPEN"),
        "gate": gate,
        "reasons": session["decision_reasons"],
    }


def _open_gate(conn: sqlite3.Connection, session: sqlite3.Row) -> str:
    kiosk = conn.execute(
        "SELECT gate_id FROM kiosks WHERE kiosk_id = ?", (session["kiosk_id"],)
    ).fetchone()
    gate_id = kiosk["gate_id"]
    conn.execute("UPDATE gates SET status = 'OPEN' WHERE gate_id = ?", (gate_id,))
    _update_session(conn, session["session_id"], gate_id=gate_id)
    log_event(
        conn, event_type="GATE_OPENED", actor_type="SYSTEM", actor_id="GateController",
        session_id=session["session_id"], decision="OPEN", details={"gate_id": gate_id},
    )
    return gate_id


def _log_transaction_closed(conn: sqlite3.Connection, session_id: str, outcome: str) -> None:
    row = _get_session_row(conn, session_id)
    log_event(
        conn, event_type="TRANSACTION_CLOSED", actor_type="SYSTEM", actor_id="AuditService",
        session_id=session_id, decision=outcome,
        details={
            "passenger_id": row["passenger_id"],
            "kiosk_id": row["kiosk_id"],
            "gate_id": row["gate_id"],
            "attempts": row["attempts"],
            "liveness_score": row["liveness_score"],
            "match_score": row["match_score"],
            "watchlist_hit": _opt_bool(row["watchlist_hit"]),
            "watchlist_risk": row["watchlist_risk"],
            "decision": row["decision"],
            "decision_reasons": json.loads(row["decision_reasons"]),
            "officer_id": row["officer_id"],
        },
    )


# --------------------------------------------------------------------------- package 1: passengers

def list_passengers(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """SELECT p.passenger_id, p.full_name, p.nationality, p.date_of_birth, p.demo_scenario,
                  pp.passport_number
           FROM passengers p JOIN passports pp ON pp.passenger_id = p.passenger_id
           ORDER BY p.passenger_id"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_passenger_detail(conn: sqlite3.Connection, passenger_id: str) -> Dict[str, Any]:
    passenger = conn.execute(
        """SELECT p.passenger_id, p.full_name, p.nationality, p.date_of_birth, p.demo_scenario,
                  pp.passport_number, pp.issuing_country, pp.expiry_date
           FROM passengers p JOIN passports pp ON pp.passenger_id = p.passenger_id
           WHERE p.passenger_id = ?""",
        (passenger_id,),
    ).fetchone()
    if passenger is None:
        raise NotFoundError(f"Passenger {passenger_id} does not exist.")
    passes = conn.execute(
        """SELECT booking_id, flight_number, gate_number, departure_time, passport_number
           FROM boarding_passes WHERE passenger_id = ? ORDER BY departure_time""",
        (passenger_id,),
    ).fetchall()
    template = conn.execute(
        "SELECT template_id, enrollment_date FROM biometric_templates WHERE passenger_id = ?",
        (passenger_id,),
    ).fetchone()
    return {
        "passenger_id": passenger["passenger_id"],
        "full_name": passenger["full_name"],
        "nationality": passenger["nationality"],
        "date_of_birth": passenger["date_of_birth"],
        "demo_scenario": passenger["demo_scenario"],
        "passport_number": passenger["passport_number"],
        "passport": {
            "passport_number": passenger["passport_number"],
            "issuing_country": passenger["issuing_country"],
            "expiry_date": passenger["expiry_date"],
        },
        "boarding_passes": [dict(r) for r in passes],
        "biometric_template": dict(template) if template else None,
    }


def start_session(
    conn: sqlite3.Connection,
    passport_number: Optional[str] = None,
    booking_id: Optional[str] = None,
    kiosk_id: str = "KIOSK-01",
    passport_image_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """Step 1: scan and ingest a travel document, then verify it.
    Supports uploaded passport document image (extracts face portrait & template).
    """
    passport_number = _norm(passport_number)
    booking_id = _norm(booking_id)
    kiosk_id = kiosk_id.strip().upper()

    kiosk = conn.execute("SELECT * FROM kiosks WHERE kiosk_id = ?", (kiosk_id,)).fetchone()
    if kiosk is None:
        raise NotFoundError(f"Kiosk {kiosk_id} is not registered.")

    unknown_reason: Optional[str] = None
    session_id = "SES-" + uuid.uuid4().hex[:10].upper()

    with transaction(conn):
        booking = None
        passport = None
        if booking_id:
            booking = conn.execute(
                "SELECT * FROM boarding_passes WHERE booking_id = ?", (booking_id,)
            ).fetchone()
        if passport_number:
            passport = conn.execute(
                "SELECT * FROM passports WHERE passport_number = ?", (passport_number,)
            ).fetchone()
            if passport is None:
                if passport_image_b64:
                    # Auto-enrol new passenger from uploaded passport document
                    new_pax_id = f"P-{uuid.uuid4().hex[:6].upper()}"
                    new_name = f"Traveler {passport_number}"
                    conn.execute(
                        "INSERT INTO passengers (passenger_id, full_name, nationality, date_of_birth, demo_scenario) VALUES (?, ?, ?, ?, ?)",
                        (new_pax_id, new_name, "GLOBAL", "1995-06-15", "Live Uploaded Passport")
                    )
                    conn.execute(
                        "INSERT INTO passports (passport_number, passenger_id, issuing_country, expiry_date) VALUES (?, ?, ?, ?)",
                        (passport_number, new_pax_id, "GLO", "2035-12-31")
                    )
                    existing_b = conn.execute("SELECT 1 FROM boarding_passes WHERE booking_id = ?", (booking_id,)).fetchone() if booking_id else None
                    new_booking_id = booking_id if (booking_id and not existing_b) else f"BK-{uuid.uuid4().hex[:6].upper()}"
                    conn.execute(
                        "INSERT INTO boarding_passes (booking_id, passenger_id, passport_number, flight_number, gate_number, departure_time) VALUES (?, ?, ?, ?, ?, ?)",
                        (new_booking_id, new_pax_id, passport_number, "BA-100", "EG-01", "2026-10-09T22:00:00Z")
                    )
                    dummy_vec = [0.0] * 128
                    conn.execute(
                        "INSERT INTO biometric_templates (template_id, passenger_id, feature_vector, enrollment_date, sim_liveness_failures, sim_impostor) VALUES (?, ?, ?, ?, 0, 0)",
                        (f"TMP-{new_pax_id}", new_pax_id, json.dumps(dummy_vec), _now())
                    )
                    conn.execute(
                        "INSERT INTO immigration_records (passport_number, visa_status, entry_permitted, notes) VALUES (?, ?, 1, ?)",
                        (passport_number, "VALID_EXIT_PERMIT", "Uploaded passport traveler")
                    )
                    passport = conn.execute("SELECT * FROM passports WHERE passport_number = ?", (passport_number,)).fetchone()
                    booking = conn.execute("SELECT * FROM boarding_passes WHERE booking_id = ?", (new_booking_id,)).fetchone()
                else:
                    unknown_reason = f"Passport {passport_number} is not on record."
        elif booking is not None:
            passport = conn.execute(
                "SELECT * FROM passports WHERE passport_number = ?", (booking["passport_number"],)
            ).fetchone()
        else:
            unknown_reason = f"Booking {booking_id} is not on record."

        if unknown_reason:
            log_event(
                conn, event_type="SCAN_REJECTED_UNKNOWN_DOCUMENT", actor_type="KIOSK",
                actor_id=kiosk_id, decision="REJECTED",
                details={"passport_number": passport_number, "booking_id": booking_id,
                         "reason": unknown_reason},
            )
        else:
            issues: List[str] = []
            if date.fromisoformat(passport["expiry_date"]) < date.today():
                issues.append("PASSPORT_EXPIRED")
            if booking_id and booking is None:
                if not passport_image_b64:
                    issues.append("BOOKING_NOT_FOUND")
            elif booking is None:
                booking = conn.execute(
                    """SELECT * FROM boarding_passes WHERE passport_number = ?
                       ORDER BY departure_time DESC LIMIT 1""",
                    (passport["passport_number"],),
                ).fetchone()
                if booking is None:
                    if passport_image_b64:
                        new_b_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
                        conn.execute(
                            "INSERT OR REPLACE INTO boarding_passes (booking_id, passenger_id, passport_number, flight_number, gate_number, departure_time) VALUES (?, ?, ?, ?, ?, ?)",
                            (new_b_id, passport["passenger_id"], passport["passport_number"], "BA-100", "EG-01", "2026-10-09T22:00:00Z")
                        )
                        booking = conn.execute("SELECT * FROM boarding_passes WHERE booking_id = ?", (new_b_id,)).fetchone()
                    else:
                        issues.append("NO_BOARDING_PASS")
            if booking is not None and booking["passport_number"] != passport["passport_number"]:
                if passport_image_b64:
                    new_b_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
                    conn.execute(
                        "INSERT OR REPLACE INTO boarding_passes (booking_id, passenger_id, passport_number, flight_number, gate_number, departure_time) VALUES (?, ?, ?, ?, ?, ?)",
                        (new_b_id, passport["passenger_id"], passport["passport_number"], "BA-100", "EG-01", "2026-10-09T22:00:00Z")
                    )
                    booking = conn.execute("SELECT * FROM boarding_passes WHERE booking_id = ?", (new_b_id,)).fetchone()
                else:
                    issues.append("BOOKING_PASSPORT_MISMATCH")

            passport_photo_thumb = None
            passport_face_json = None

            if passport_image_b64:
                try:
                    doc_res = real_face.extract_face_from_document(passport_image_b64)
                    if doc_res.get("face_detected"):
                        passport_photo_thumb = doc_res.get("face_thumbnail_b64")
                        passport_face_json = json.dumps(doc_res.get("embedding"))
                        # Update passenger template
                        conn.execute(
                            "UPDATE biometric_templates SET real_face_encoding = ? WHERE passenger_id = ?",
                            (passport_face_json, passport["passenger_id"])
                        )
                    else:
                        issues.append("PASSPORT_PHOTO_UNREADABLE")
                except Exception as exc:
                    issues.append(f"PASSPORT_SCAN_ERROR: {exc}")

            now = _now()
            valid = not issues
            conn.execute(
                """INSERT INTO verification_sessions
                   (session_id, passenger_id, passport_number, booking_id, kiosk_id, state,
                    escalation_reason, document_issues, passport_photo_b64, passport_face_encoding,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, passport["passenger_id"], passport["passport_number"],
                 booking["booking_id"] if booking is not None else booking_id, kiosk_id,
                 "AWAITING_FACE" if valid else "AWAITING_OFFICER",
                 None if valid else "DOCUMENT_INVALID",
                 json.dumps(issues), passport_photo_thumb, passport_face_json, now, now),
            )
            log_event(
                conn, event_type="SESSION_STARTED", actor_type="KIOSK", actor_id=kiosk_id,
                session_id=session_id,
                details={"passenger_id": passport["passenger_id"],
                         "passport_number": passport["passport_number"],
                         "booking_id": booking["booking_id"] if booking is not None else booking_id,
                         "passport_uploaded": bool(passport_image_b64)},
            )
            log_event(
                conn, event_type="DOCUMENT_VERIFIED" if valid else "DOCUMENT_REJECTED",
                actor_type="SYSTEM", actor_id="DocumentVerificationModule",
                session_id=session_id, decision="VALID" if valid else "INVALID",
                details={"issues": issues, "passport_photo_extracted": bool(passport_photo_thumb)},
            )
            if not valid:
                log_event(
                    conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                    actor_id="DocumentVerificationModule", session_id=session_id,
                    decision="MANUAL_DOCUMENT_REVIEW",
                    details={"reason": "DOCUMENT_INVALID", "issues": issues},
                )

    if unknown_reason:
        raise NotFoundError(unknown_reason)
    return _session_dict(conn, session_id)


# --------------------------------------------------------------------------- package 2: e-gate

def list_kiosks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """SELECT k.kiosk_id, k.location, k.status, k.gate_id, g.status AS gate_status
           FROM kiosks k JOIN gates g ON g.gate_id = k.gate_id ORDER BY k.kiosk_id"""
    ).fetchall()
    return [dict(r) for r in rows]


def list_gates(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT gate_id, status FROM gates ORDER BY gate_id").fetchall()
    return [dict(r) for r in rows]


def get_session(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    return _session_dict(conn, session_id)


def verify_face(
    conn: sqlite3.Connection, session_id: str, frame_id: Optional[str] = None,
    image_base64: Optional[str] = None
) -> Dict[str, Any]:
    """Step 2: one capture attempt (liveness, then face match). Retries are capped.
    Supports real live webcam frame (image_base64) or fallback simulated capture.
    """
    with transaction(conn):
        session = _get_session_row(conn, session_id)
        _require_state(session, "AWAITING_FACE", "face verification")
        cfg = get_thresholds(conn)
        template = conn.execute(
            "SELECT * FROM biometric_templates WHERE passenger_id = ?", (session["passenger_id"],)
        ).fetchone()

        if template is None and not image_base64:
            _update_session(conn, session_id, state="AWAITING_OFFICER",
                            escalation_reason="NO_ENROLLED_TEMPLATE")
            log_event(
                conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                actor_id="FaceRecognitionModule", session_id=session_id,
                decision="MANUAL_IDENTITY_CHECK", details={"reason": "NO_ENROLLED_TEMPLATE"},
            )
            attempt = {
                "attempt_number": session["attempts"], "liveness_score": None,
                "liveness_passed": False, "match_score": None, "match_passed": None,
                "attempts_remaining": 0, "outcome": "ESCALATED",
                "engine_type": "SIMULATED",
                "face_details": None,
            }
        elif image_base64 and real_face.is_available():
            # Real AI Face Recognition using MediaPipe
            attempt_number = session["attempts"] + 1
            attempts_remaining = max(cfg["max_attempts"] - attempt_number, 0)
            real_result = real_face.process_face_image(
                conn, session["passenger_id"], image_base64,
                match_threshold=cfg["match_threshold"],
                liveness_threshold=cfg["liveness_threshold"],
                session_id=session_id,
            )

            liveness_score = real_result.get("liveness_score", 0.0)
            liveness_passed = real_result.get("liveness_passed", False)
            match_score = real_result.get("match_score")
            match_passed = real_result.get("match_passed")

            event_details = {
                "attempt": attempt_number, "frame_id": frame_id,
                "liveness_score": liveness_score,
                "liveness_threshold": cfg["liveness_threshold"],
                "engine": "MEDIAPIPE_AI",
                "face_detected": real_result.get("face_detected", False),
                "message": real_result.get("message"),
            }

            if not liveness_passed:
                _update_session(conn, session_id, attempts=attempt_number,
                                liveness_score=liveness_score)
                log_event(
                    conn, event_type="LIVENESS_FAILED", actor_type="SYSTEM",
                    actor_id="FaceRecognitionModule", session_id=session_id,
                    decision="NOT_LIVE",
                    details={**event_details, "attempts_remaining": attempts_remaining},
                )
                if attempts_remaining == 0:
                    _update_session(conn, session_id, state="AWAITING_OFFICER",
                                    escalation_reason="LIVENESS_RETRIES_EXHAUSTED")
                    log_event(
                        conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                        actor_id="FaceRecognitionModule", session_id=session_id,
                        decision="MANUAL_IDENTITY_CHECK",
                        details={"reason": "LIVENESS_RETRIES_EXHAUSTED",
                                 "attempts": attempt_number},
                    )
                    outcome = "ESCALATED"
                else:
                    outcome = "RETRY"
                score = None
                passed = None
            else:
                score = match_score if match_score is not None else 0.45
                passed = bool(match_passed) if match_passed is not None else False
                if passed:
                    new_state = "FACE_VERIFIED"
                    outcome = "FACE_VERIFIED"
                    esc_reason = None
                else:
                    if attempts_remaining == 0:
                        new_state = "AWAITING_OFFICER"
                        outcome = "ESCALATED"
                        esc_reason = "FACE_MATCH_RETRIES_EXHAUSTED"
                    else:
                        new_state = "AWAITING_FACE"
                        outcome = "RETRY"
                        esc_reason = None

                _update_session(
                    conn, session_id, attempts=attempt_number,
                    liveness_score=liveness_score, match_score=score,
                    match_passed=int(passed), state=new_state,
                    escalation_reason=esc_reason,
                )
                log_event(
                    conn, event_type="FACE_MATCH_COMPLETED", actor_type="SYSTEM",
                    actor_id="FaceRecognitionModule", session_id=session_id,
                    decision="MATCH" if passed else "NO_MATCH",
                    details={**event_details, "match_score": score,
                             "match_threshold": cfg["match_threshold"],
                             "attempts_remaining": attempts_remaining},
                )
                if not passed and attempts_remaining == 0:
                    log_event(
                        conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                        actor_id="FaceRecognitionModule", session_id=session_id,
                        decision="MANUAL_IDENTITY_CHECK",
                        details={"reason": "FACE_MATCH_RETRIES_EXHAUSTED",
                                 "attempts": attempt_number},
                    )

            attempt = {
                "attempt_number": attempt_number,
                "liveness_score": liveness_score,
                "liveness_passed": liveness_passed,
                "match_score": score,
                "match_passed": passed,
                "attempts_remaining": attempts_remaining,
                "outcome": outcome,
                "engine_type": "AI_REAL_MEDIAPIPE",
                "comparison_source": "UPLOADED_PASSPORT_PORTRAIT" if session["passport_photo_b64"] else "BIOMETRIC_TEMPLATE",
                "passport_photo_b64": session["passport_photo_b64"] if "passport_photo_b64" in session.keys() else None,
                "face_details": real_result.get("details"),
            }
        else:
            # Fallback to simulated engine
            attempt_number = session["attempts"] + 1
            attempts_remaining = max(cfg["max_attempts"] - attempt_number, 0)
            capture = face_engine.simulate_capture(
                json.loads(template["feature_vector"]), session["passenger_id"], attempt_number,
                template["sim_liveness_failures"], bool(template["sim_impostor"]),
            )
            liveness_passed = capture.liveness_score >= cfg["liveness_threshold"]
            event_details = {
                "attempt": attempt_number, "frame_id": frame_id,
                "liveness_score": capture.liveness_score,
                "liveness_threshold": cfg["liveness_threshold"],
            }

            if not liveness_passed:
                _update_session(conn, session_id, attempts=attempt_number,
                                liveness_score=capture.liveness_score)
                log_event(
                    conn, event_type="LIVENESS_FAILED", actor_type="SYSTEM",
                    actor_id="FaceRecognitionModule", session_id=session_id,
                    decision="NOT_LIVE",
                    details={**event_details, "attempts_remaining": attempts_remaining},
                )
                if attempts_remaining == 0:
                    _update_session(conn, session_id, state="AWAITING_OFFICER",
                                    escalation_reason="LIVENESS_RETRIES_EXHAUSTED")
                    log_event(
                        conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                        actor_id="FaceRecognitionModule", session_id=session_id,
                        decision="MANUAL_IDENTITY_CHECK",
                        details={"reason": "LIVENESS_RETRIES_EXHAUSTED",
                                 "attempts": attempt_number},
                    )
                    outcome = "ESCALATED"
                else:
                    outcome = "RETRY"
                score = None
                passed = None
            else:
                score = face_engine.match_score(
                    json.loads(template["feature_vector"]), capture.embedding
                )
                passed = score >= cfg["match_threshold"]
                if passed:
                    new_state = "FACE_VERIFIED"
                    outcome = "FACE_VERIFIED"
                    esc_reason = None
                else:
                    if attempts_remaining == 0:
                        new_state = "AWAITING_OFFICER"
                        outcome = "ESCALATED"
                        esc_reason = "FACE_MATCH_RETRIES_EXHAUSTED"
                    else:
                        new_state = "AWAITING_FACE"
                        outcome = "RETRY"
                        esc_reason = None

                _update_session(
                    conn, session_id, attempts=attempt_number,
                    liveness_score=capture.liveness_score, match_score=score,
                    match_passed=int(passed), state=new_state,
                    escalation_reason=esc_reason,
                )
                log_event(
                    conn, event_type="FACE_MATCH_COMPLETED", actor_type="SYSTEM",
                    actor_id="FaceRecognitionModule", session_id=session_id,
                    decision="MATCH" if passed else "NO_MATCH",
                    details={**event_details, "match_score": score,
                             "match_threshold": cfg["match_threshold"],
                             "attempts_remaining": attempts_remaining},
                )
                if not passed and attempts_remaining == 0:
                    log_event(
                        conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                        actor_id="FaceRecognitionModule", session_id=session_id,
                        decision="MANUAL_IDENTITY_CHECK",
                        details={"reason": "FACE_MATCH_RETRIES_EXHAUSTED",
                                 "attempts": attempt_number},
                    )

            attempt = {
                "attempt_number": attempt_number,
                "liveness_score": capture.liveness_score,
                "liveness_passed": liveness_passed,
                "match_score": score,
                "match_passed": passed,
                "attempts_remaining": attempts_remaining,
                "outcome": outcome,
                "engine_type": "SIMULATED",
                "face_details": None,
            }

    return {"session": _session_dict(conn, session_id), "attempt": attempt}


def decide_access(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """Step 4: auto-clear the passenger or escalate to an immigration officer."""
    with transaction(conn):
        session = _get_session_row(conn, session_id)
        _require_state(session, "CHECKS_COMPLETE", "an access decision")

        reasons: List[str] = []
        if not session["match_passed"]:
            reasons.append("FACE_MATCH_BELOW_THRESHOLD")
        if session["watchlist_hit"]:
            reasons.append("WATCHLIST_HIT")
        if not session["entry_permitted"]:
            reasons.append("ENTRY_NOT_PERMITTED")

        if not reasons:
            _update_session(conn, session_id, state="GATE_OPEN", decision="AUTO_CLEARED",
                            decision_reasons=json.dumps([]))
            log_event(
                conn, event_type="ACCESS_GRANTED_AUTO", actor_type="SYSTEM",
                actor_id="GateController", session_id=session_id, decision="AUTO_CLEARED",
                details={"match_score": session["match_score"]},
            )
            _open_gate(conn, session)
        else:
            _update_session(conn, session_id, state="AWAITING_OFFICER",
                            escalation_reason="VERIFICATION_FAILED",
                            decision_reasons=json.dumps(reasons))
            log_event(
                conn, event_type="ESCALATED_TO_OFFICER", actor_type="SYSTEM",
                actor_id="GateController", session_id=session_id,
                decision="MANUAL_REVIEW",
                details={"reason": "VERIFICATION_FAILED", "reasons": reasons},
            )

    return _gate_action_response(conn, session_id)


def confirm_passage(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """Passenger walked through: close the gate and finish the transaction."""
    with transaction(conn):
        session = _get_session_row(conn, session_id)
        _require_state(session, "GATE_OPEN", "passage confirmation")
        conn.execute("UPDATE gates SET status = 'CLOSED' WHERE gate_id = ?", (session["gate_id"],))
        _update_session(conn, session_id, state="COMPLETED")
        log_event(
            conn, event_type="GATE_CLOSED", actor_type="SYSTEM", actor_id="GateController",
            session_id=session_id, decision="CLOSED", details={"gate_id": session["gate_id"]},
        )
        _log_transaction_closed(conn, session_id, "COMPLETED")
    return _gate_action_response(conn, session_id)


# --------------------------------------------------------------------------- package 3: immigration

def get_immigration_record(conn: sqlite3.Connection, passport_number: str) -> Dict[str, Any]:
    passport_number = _norm(passport_number) or ""
    row = conn.execute(
        "SELECT passport_number, visa_status, entry_permitted, notes "
        "FROM immigration_records WHERE passport_number = ?",
        (passport_number,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"No immigration record for passport {passport_number}.")
    return {
        "passport_number": row["passport_number"],
        "visa_status": row["visa_status"],
        "entry_permitted": bool(row["entry_permitted"]),
        "notes": row["notes"],
    }


def _watchlist_summary(conn: sqlite3.Connection, passport_number: str) -> Dict[str, Any]:
    rows = conn.execute(
        "SELECT risk_level FROM watchlist_entries WHERE passport_number = ?", (passport_number,)
    ).fetchall()
    highest = None
    if rows:
        highest = max((r["risk_level"] for r in rows), key=lambda level: config.RISK_ORDER[level])
    return {
        "passport_number": passport_number,
        "hit": bool(rows),
        "highest_risk": highest,
        "entry_count": len(rows),
    }


def get_watchlist_status(conn: sqlite3.Connection, passport_number: str) -> Dict[str, Any]:
    passport_number = _norm(passport_number) or ""
    exists = conn.execute(
        "SELECT 1 FROM passports WHERE passport_number = ?", (passport_number,)
    ).fetchone()
    if exists is None:
        raise NotFoundError(f"Passport {passport_number} is not on record.")
    return _watchlist_summary(conn, passport_number)


def run_immigration_checks(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """Step 3: immigration database lookup and watchlist check."""
    with transaction(conn):
        session = _get_session_row(conn, session_id)
        _require_state(session, "FACE_VERIFIED", "immigration checks")

        record = conn.execute(
            "SELECT * FROM immigration_records WHERE passport_number = ?",
            (session["passport_number"],),
        ).fetchone()
        watchlist = _watchlist_summary(conn, session["passport_number"])
        visa_status = record["visa_status"] if record else "NO_RECORD"
        entry_permitted = bool(record["entry_permitted"]) if record else False

        _update_session(
            conn, session_id, state="CHECKS_COMPLETE", immigration_status=visa_status,
            entry_permitted=int(entry_permitted), watchlist_hit=int(watchlist["hit"]),
            watchlist_risk=watchlist["highest_risk"],
        )
        log_event(
            conn, event_type="IMMIGRATION_CHECKS_COMPLETED", actor_type="SYSTEM",
            actor_id="ImmigrationDatabase", session_id=session_id,
            decision="WATCHLIST_HIT" if watchlist["hit"] else "CLEAR",
            details={"visa_status": visa_status, "entry_permitted": entry_permitted,
                     "record_found": record is not None,
                     "watchlist_hit": watchlist["hit"],
                     "watchlist_highest_risk": watchlist["highest_risk"],
                     "watchlist_entry_count": watchlist["entry_count"]},
        )

    return {
        "session": _session_dict(conn, session_id),
        "immigration": {
            "passport_number": session["passport_number"],
            "visa_status": visa_status,
            "entry_permitted": entry_permitted,
            "record_found": record is not None,
        },
        "watchlist": watchlist,
    }


def list_pending_reviews(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM verification_sessions WHERE state = 'AWAITING_OFFICER' ORDER BY updated_at"
    ).fetchall()
    return [serialize_session(conn, r) for r in rows]


def officer_override(
    conn: sqlite3.Connection, session_id: str, officer_id: str, approve: bool, notes: str
) -> Dict[str, Any]:
    """Step 4 (manual route): the officer approves or denies an escalated session."""
    officer_id = officer_id.strip().upper()
    blocked: Optional[Exception] = None

    with transaction(conn):
        session = _get_session_row(conn, session_id)
        officer = conn.execute(
            "SELECT * FROM officers WHERE officer_id = ?", (officer_id,)
        ).fetchone()

        if officer is None or not officer["active"]:
            log_event(
                conn, event_type="OFFICER_AUTH_REJECTED", actor_type="OFFICER",
                actor_id=officer_id, session_id=session_id, decision="REJECTED",
                details={"reason": "Officer is not registered or not active."},
            )
            blocked = ForbiddenError(f"Officer {officer_id} is not authorised to review cases.")
        else:
            _require_state(session, "AWAITING_OFFICER", "an officer override")
            if approve and session["watchlist_risk"] == "CRITICAL":
                log_event(
                    conn, event_type="OFFICER_OVERRIDE_BLOCKED", actor_type="OFFICER",
                    actor_id=officer_id, session_id=session_id, decision="BLOCKED",
                    details={"reason": "CRITICAL watchlist hit cannot be approved at the gate.",
                             "notes": notes},
                )
                blocked = ConflictError(
                    "A CRITICAL watchlist hit cannot be approved by gate override. "
                    "Deny the passenger and escalate through the security chain."
                )
            elif approve:
                _update_session(conn, session_id, officer_id=officer_id, officer_notes=notes,
                                decision="OFFICER_APPROVED", state="GATE_OPEN")
                log_event(
                    conn, event_type="OFFICER_OVERRIDE_APPROVED", actor_type="OFFICER",
                    actor_id=officer_id, session_id=session_id, decision="APPROVED",
                    details={"notes": notes, "escalation_reason": session["escalation_reason"],
                             "decision_reasons": json.loads(session["decision_reasons"]),
                             "document_issues": json.loads(session["document_issues"])},
                )
                _open_gate(conn, _get_session_row(conn, session_id))
            else:
                _update_session(conn, session_id, officer_id=officer_id, officer_notes=notes,
                                decision="OFFICER_DENIED", state="DENIED")
                log_event(
                    conn, event_type="OFFICER_OVERRIDE_DENIED", actor_type="OFFICER",
                    actor_id=officer_id, session_id=session_id, decision="DENIED",
                    details={"notes": notes, "escalation_reason": session["escalation_reason"],
                             "decision_reasons": json.loads(session["decision_reasons"]),
                             "document_issues": json.loads(session["document_issues"])},
                )
                _log_transaction_closed(conn, session_id, "DENIED")

    if blocked is not None:
        raise blocked
    return _gate_action_response(conn, session_id)


# --------------------------------------------------------------------------- package 4: admin & audit

def get_config(conn: sqlite3.Connection) -> Dict[str, Any]:
    return get_thresholds(conn)


def update_thresholds(
    conn: sqlite3.Connection, admin_id: str, changes: Dict[str, Any]
) -> Dict[str, Any]:
    admin_id = admin_id.strip().upper()
    blocked: Optional[Exception] = None

    with transaction(conn):
        admin = conn.execute(
            "SELECT * FROM administrators WHERE admin_id = ?", (admin_id,)
        ).fetchone()
        if admin is None:
            log_event(
                conn, event_type="ADMIN_AUTH_REJECTED", actor_type="ADMIN", actor_id=admin_id,
                decision="REJECTED", details={"attempted_changes": changes},
            )
            blocked = ForbiddenError(f"{admin_id} is not an authorised administrator.")
        else:
            previous = get_thresholds(conn)
            for key, value in changes.items():
                conn.execute("UPDATE threshold_config SET value = ? WHERE key = ?",
                             (float(value), key))
            current = get_thresholds(conn)
            log_event(
                conn, event_type="CONFIG_UPDATED", actor_type="ADMIN", actor_id=admin_id,
                decision="APPLIED", details={"previous": previous, "current": current},
            )

    if blocked is not None:
        raise blocked
    return get_thresholds(conn)


def list_sessions(
    conn: sqlite3.Connection, state: Optional[str], limit: int, offset: int
) -> List[Dict[str, Any]]:
    if state:
        rows = conn.execute(
            "SELECT * FROM verification_sessions WHERE state = ? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (state, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM verification_sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [serialize_session(conn, r) for r in rows]
