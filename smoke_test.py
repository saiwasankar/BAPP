"""End-to-end smoke test for every seeded BAPPS scenario.

Usage (from the project root):  python smoke_test.py
Uses a temporary database, so it never touches bapps.db.
"""
import os
import sqlite3
import sys
import tempfile

TMP_DIR = tempfile.mkdtemp(prefix="bapps_test_")
os.environ["BAPPS_DB_PATH"] = os.path.join(TMP_DIR, "test.db")
os.environ["BAPPS_ALLOW_RESET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

passed = 0
failed = 0


def check(condition: bool, label: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")


def scan(client, **body):
    return client.post("/api/passengers/scan", json=body)


def face(client, sid):
    return client.post(f"/api/egate/sessions/{sid}/face-verification")


def checks(client, sid):
    return client.post(f"/api/immigration/sessions/{sid}/checks")


def decide(client, sid):
    return client.post(f"/api/egate/sessions/{sid}/decision")


def override(client, sid, officer, approve, notes="Verified identity in person at counter."):
    return client.post(
        f"/api/immigration/sessions/{sid}/override",
        json={"officer_id": officer, "approve": approve, "notes": notes},
    )


def run() -> None:
    with TestClient(app) as client:
        print("\n[0] Health and seed data")
        check(client.get("/api/health").json()["status"] == "ok", "health endpoint")
        passengers = client.get("/api/passengers").json()
        check(len(passengers) == 7, "7 passengers seeded")
        detail = client.get("/api/passengers/P-1001").json()
        check(detail["passport"]["passport_number"] == "N4821907"
              and detail["biometric_template"]["template_id"] == "TPL-1001",
              "passenger detail includes passport and template metadata")
        check(client.get("/api/passengers/P-9999").status_code == 404, "unknown passenger -> 404")

        print("\n[1] P-1001 Aarav Sharma: clean passenger, auto clearance")
        r = scan(client, passport_number="N4821907")
        check(r.status_code == 201 and r.json()["state"] == "AWAITING_FACE", "scan by passport -> AWAITING_FACE")
        sid = r.json()["session_id"]
        check(r.json()["booking_id"] == "BK-5001", "boarding pass resolved from passport")
        f = face(client, sid).json()
        check(f["attempt"]["outcome"] == "FACE_VERIFIED" and f["attempt"]["match_passed"] is True
              and f["attempt"]["match_score"] > 0.85, "face verified on first attempt")
        c = checks(client, sid).json()
        check(c["watchlist"]["hit"] is False and c["immigration"]["entry_permitted"] is True,
              "immigration and watchlist clear")
        d = decide(client, sid).json()
        check(d["authorized"] is True and d["gate"]["status"] == "OPEN"
              and d["session"]["decision"] == "AUTO_CLEARED", "gate opened automatically")
        p = client.post(f"/api/egate/sessions/{sid}/passage").json()
        check(p["session"]["state"] == "COMPLETED" and p["gate"]["status"] == "CLOSED",
              "passage confirmed, gate closed, session COMPLETED")
        trail = client.get("/api/admin/audit-log", params={"session_id": sid}).json()
        events = [e["event_type"] for e in trail["items"]]
        check(events == ["SESSION_STARTED", "DOCUMENT_VERIFIED", "FACE_MATCH_COMPLETED",
                         "IMMIGRATION_CHECKS_COMPLETED", "ACCESS_GRANTED_AUTO", "GATE_OPENED",
                         "GATE_CLOSED", "TRANSACTION_CLOSED"], "audit trail complete and ordered")
        check(face(client, sid).status_code == 409, "face verification after completion -> 409")

        print("\n[2] P-1002 Emily Carter: flaky camera, passes on attempt 3 (booking-only scan)")
        r = scan(client, booking_id="BK-5002", kiosk_id="KIOSK-02")
        sid = r.json()["session_id"]
        check(r.status_code == 201 and r.json()["passport_number"] == "G7305518",
              "scan by boarding pass resolves passport")
        a1 = face(client, sid).json()["attempt"]
        a2 = face(client, sid).json()["attempt"]
        a3 = face(client, sid).json()["attempt"]
        check(a1["outcome"] == "RETRY" and a1["attempts_remaining"] == 2, "attempt 1 liveness fail -> RETRY")
        check(a2["outcome"] == "RETRY" and a2["attempts_remaining"] == 1, "attempt 2 liveness fail -> RETRY")
        check(a3["outcome"] == "FACE_VERIFIED" and a3["attempts_remaining"] == 0,
              "attempt 3 passes liveness and matches")
        checks(client, sid)
        d = decide(client, sid).json()
        check(d["authorized"] is True and d["gate"]["gate_id"] == "EG-02", "auto-cleared through KIOSK-02 / EG-02")
        client.post(f"/api/egate/sessions/{sid}/passage")

        print("\n[3] P-1003 Rohan Mehta: spoof, 3 failed attempts, officer approves")
        sid = scan(client, passport_number="N9917342").json()["session_id"]
        outcomes = [face(client, sid).json()["attempt"]["outcome"] for _ in range(3)]
        check(outcomes == ["RETRY", "RETRY", "ESCALATED"], "RETRY, RETRY, ESCALATED")
        s = client.get(f"/api/egate/sessions/{sid}").json()
        check(s["state"] == "AWAITING_OFFICER" and s["escalation_reason"] == "LIVENESS_RETRIES_EXHAUSTED"
              and s["attempts"] == 3, "session escalated after 3 attempts")
        check(face(client, sid).status_code == 409, "4th attempt blocked -> 409")
        pending = [x["session_id"] for x in client.get("/api/immigration/reviews").json()]
        check(sid in pending, "session listed in pending officer reviews")
        check(override(client, sid, "OFF-001", True, "too short").status_code == 422,
              "override with short notes -> 422")
        check(override(client, sid, "OFF-003", True).status_code == 403, "inactive officer -> 403")
        check(override(client, sid, "OFF-999", True).status_code == 403, "unknown officer -> 403")
        o = override(client, sid, "OFF-001", True)
        check(o.status_code == 200 and o.json()["authorized"] is True
              and o.json()["session"]["decision"] == "OFFICER_APPROVED", "OFF-001 approves, gate opens")
        check(override(client, sid, "OFF-001", True).status_code == 409, "second override -> 409")
        client.post(f"/api/egate/sessions/{sid}/passage")

        print("\n[4] P-1004 Marcus Hale: HIGH watchlist hit, officer denies")
        sid = scan(client, passport_number="C5520871").json()["session_id"]
        check(face(client, sid).json()["attempt"]["outcome"] == "FACE_VERIFIED", "face verified")
        c = checks(client, sid).json()
        check(c["watchlist"]["hit"] is True and c["watchlist"]["highest_risk"] == "HIGH", "HIGH watchlist hit found")
        d = decide(client, sid).json()
        check(d["authorized"] is False and d["session"]["state"] == "AWAITING_OFFICER"
              and d["reasons"] == ["WATCHLIST_HIT"], "escalated with reason WATCHLIST_HIT")
        o = override(client, sid, "OFF-002", False, "Passenger referred to security for questioning.")
        check(o.json()["session"]["state"] == "DENIED" and o.json()["session"]["decision"] == "OFFICER_DENIED"
              and o.json()["authorized"] is False, "OFF-002 denies, session DENIED")
        closed = client.get("/api/admin/audit-log", params={"session_id": sid, "event_type": "TRANSACTION_CLOSED"}).json()
        check(closed["total"] == 1 and closed["items"][0]["decision"] == "DENIED", "TRANSACTION_CLOSED logged for denial")

        print("\n[5] P-1005 Priya Nair: expired passport -> manual document review")
        r = scan(client, passport_number="N3304415")
        s = r.json()
        check(r.status_code == 201 and s["state"] == "AWAITING_OFFICER"
              and s["escalation_reason"] == "DOCUMENT_INVALID"
              and s["document_issues"] == ["PASSPORT_EXPIRED"], "expired passport escalated")
        check(face(client, s["session_id"]).status_code == 409, "face step blocked for invalid documents")
        o = override(client, s["session_id"], "OFF-001", False, "Passport expired; passenger redirected to renew.")
        check(o.json()["session"]["state"] == "DENIED", "officer denies after manual document review")

        print("\n[6] P-1006 Sofia Alvarez: face mismatch -> officer approves")
        sid = scan(client, passport_number="E6618240").json()["session_id"]
        f = face(client, sid).json()["attempt"]
        check(f["outcome"] == "FACE_VERIFIED" and f["match_passed"] is False and f["match_score"] < 0.85,
              f"live face fails match (score {f['match_score']})")
        checks(client, sid)
        d = decide(client, sid).json()
        check(d["reasons"] == ["FACE_MATCH_BELOW_THRESHOLD"], "escalated with FACE_MATCH_BELOW_THRESHOLD")
        o = override(client, sid, "OFF-002", True, "Identity confirmed against passport photo in person.")
        check(o.json()["authorized"] is True and o.json()["session"]["decision"] == "OFFICER_APPROVED",
              "officer approval opens gate")
        client.post(f"/api/egate/sessions/{sid}/passage")

        print("\n[7] P-1007 Liam O'Connor: CRITICAL hit + restricted entry")
        sid = scan(client, passport_number="I2290763").json()["session_id"]
        face(client, sid)
        c = checks(client, sid).json()
        check(c["watchlist"]["highest_risk"] == "CRITICAL" and c["immigration"]["entry_permitted"] is False,
              "CRITICAL hit and entry not permitted")
        d = decide(client, sid).json()
        check(set(d["reasons"]) == {"WATCHLIST_HIT", "ENTRY_NOT_PERMITTED"}, "both failure reasons reported")
        check(override(client, sid, "OFF-001", True).status_code == 409, "approving a CRITICAL hit is blocked -> 409")
        check(client.get(f"/api/egate/sessions/{sid}").json()["state"] == "AWAITING_OFFICER",
              "session still awaiting officer after blocked approval")
        blocked = client.get("/api/admin/audit-log", params={"event_type": "OFFICER_OVERRIDE_BLOCKED"}).json()
        check(blocked["total"] == 1, "blocked approval is audited")
        o = override(client, sid, "OFF-001", False, "Critical alert on file; passenger detained for security.")
        check(o.json()["session"]["state"] == "DENIED", "officer denies")

        print("\n[8] Document edge cases")
        r = scan(client, passport_number="ZZ999999")
        check(r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND", "unknown passport -> 404")
        check(client.get("/api/admin/audit-log", params={"event_type": "SCAN_REJECTED_UNKNOWN_DOCUMENT"}).json()["total"] == 1,
              "rejected scan audited without a session")
        check(scan(client, booking_id="BK-0000").status_code == 404, "unknown booking only -> 404")
        check(scan(client).status_code == 422, "scan with no document -> 422")
        check(scan(client, passport_number="N4821907", kiosk_id="KIOSK-99").status_code == 404, "unknown kiosk -> 404")
        r = scan(client, passport_number="N4821907", booking_id="BK-5002")
        check(r.json()["document_issues"] == ["BOOKING_PASSPORT_MISMATCH"] and r.json()["state"] == "AWAITING_OFFICER",
              "boarding pass of another passenger -> BOOKING_PASSPORT_MISMATCH")
        r = scan(client, passport_number="N4821907", booking_id="BK-0000")
        check(r.json()["document_issues"] == ["BOOKING_NOT_FOUND"], "unknown booking with valid passport -> BOOKING_NOT_FOUND")
        check(client.get("/api/immigration/records/N4821907").json()["visa_status"] == "VALID_EXIT_PERMIT", "immigration record lookup")
        check(client.get("/api/immigration/watchlist/C5520871").json()["highest_risk"] == "HIGH", "watchlist status lookup")
        check(client.get("/api/immigration/watchlist/NOPE12345").status_code == 404, "watchlist lookup for unknown passport -> 404")
        check(decide(client, scan(client, passport_number="N4821907").json()["session_id"]).status_code == 409,
              "decision before checks -> 409")

        print("\n[9] Admin: thresholds and audit integrity")
        check(client.get("/api/admin/config").json() == {"match_threshold": 0.85, "liveness_threshold": 0.8, "max_attempts": 3},
              "default thresholds")
        check(client.put("/api/admin/config", json={"admin_id": "OFF-001", "max_attempts": 2}).status_code == 403,
              "non-admin cannot change config -> 403")
        check(client.put("/api/admin/config", json={"admin_id": "ADM-001"}).status_code == 422, "empty update -> 422")
        check(client.put("/api/admin/config", json={"admin_id": "ADM-001", "match_threshold": 1.5}).status_code == 422,
              "out-of-range threshold -> 422")
        client.put("/api/admin/config", json={"admin_id": "ADM-001", "max_attempts": 2})
        sid = scan(client, passport_number="N9917342").json()["session_id"]
        outcomes = [face(client, sid).json()["attempt"]["outcome"] for _ in range(2)]
        check(outcomes == ["RETRY", "ESCALATED"], "lowering max_attempts to 2 escalates on attempt 2")
        client.put("/api/admin/config", json={"admin_id": "ADM-001", "max_attempts": 3})
        check(client.get("/api/admin/audit-log", params={"event_type": "CONFIG_UPDATED"}).json()["total"] == 2,
              "config changes audited")
        awaiting = client.get("/api/admin/sessions", params={"state": "awaiting_officer"}).json()
        check(len(awaiting) >= 1 and all(x["state"] == "AWAITING_OFFICER" for x in awaiting), "session list filter by state")

        v = client.get("/api/admin/audit-log/verify").json()
        check(v["valid"] is True and v["entries_checked"] > 50, f"audit chain valid ({v['entries_checked']} entries)")

        raw = sqlite3.connect(os.environ["BAPPS_DB_PATH"])
        try:
            raw.execute("UPDATE audit_log SET decision = 'X' WHERE log_id = 1")
            blocked_update = False
        except sqlite3.DatabaseError:
            blocked_update = True
        try:
            raw.execute("DELETE FROM audit_log WHERE log_id = 1")
            blocked_delete = False
        except sqlite3.DatabaseError:
            blocked_delete = True
        check(blocked_update and blocked_delete, "database triggers block audit UPDATE and DELETE")

        raw.execute("DROP TRIGGER audit_log_no_update")
        raw.execute("UPDATE audit_log SET decision = 'TAMPERED' WHERE log_id = 5")
        raw.commit()
        raw.close()
        v = client.get("/api/admin/audit-log/verify").json()
        check(v["valid"] is False and v["first_invalid_log_id"] == 5, "tampering with an entry is detected at log_id 5")

        print("\n[10] Demo reset")
        check(client.post("/api/admin/demo/reset").status_code == 200, "reset endpoint")
        check(client.get("/api/admin/audit-log").json()["total"] == 0, "audit log empty after reset")
        check(client.get("/api/admin/audit-log/verify").json()["valid"] is True, "chain valid after reset")
        check(len(client.get("/api/passengers").json()) == 7, "seed data restored")


if __name__ == "__main__":
    run()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
