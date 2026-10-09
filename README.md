# BAPPS Backend (FastAPI + SQLite)

Backend and database layer for the Biometric Aviation Passenger Processing System. It implements the full pipeline from the DFDs, sequence flow and activity diagram. Face capture is simulated, so no camera or ML libraries are needed.

## Install and run

Requires Python 3.10 or newer.

```bash
cd bapps_backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Interactive API docs (try every endpoint): http://127.0.0.1:8000/docs
- The SQLite file `bapps.db` is created and seeded on first start.
- Run the automated end-to-end test (68 checks, uses a temporary database): `python smoke_test.py`

## Pipeline

| Step | Endpoint | Result |
|---|---|---|
| 1. Scan and ingest | `POST /api/passengers/scan` | Session created; passport and boarding pass verified. Invalid documents go straight to an officer |
| 2. Liveness and face match | `POST /api/egate/sessions/{id}/face-verification` | One attempt per call. After 3 failed liveness attempts the session escalates |
| 3. Checks | `POST /api/immigration/sessions/{id}/checks` | Immigration record lookup and watchlist check |
| 4a. Decision | `POST /api/egate/sessions/{id}/decision` | Gate opens automatically, or the session escalates to an officer |
| 4b. Officer route | `POST /api/immigration/sessions/{id}/override` | Officer approves (gate opens) or denies |
| 5. Passage and audit | `POST /api/egate/sessions/{id}/passage` | Gate closes; `TRANSACTION_CLOSED` is logged. Every step is audited |

Every session response includes `state` and `next_action`, so a UI can drive the flow without hard-coding it.

Session states: `AWAITING_FACE`, `FACE_VERIFIED`, `CHECKS_COMPLETE`, `AWAITING_OFFICER`, `GATE_OPEN`, `DENIED`, `COMPLETED`.

## Endpoints by package

| Package | Endpoints |
|---|---|
| 1. Passenger & Travel Documents (`/api/passengers`) | `GET /`, `GET /{passenger_id}`, `POST /scan` |
| 2. E-Gate Processing (`/api/egate`) | `GET /kiosks`, `GET /gates`, `GET /sessions/{id}`, `POST /sessions/{id}/face-verification`, `/decision`, `/passage` |
| 3. Immigration & Records (`/api/immigration`) | `GET /records/{passport}`, `GET /watchlist/{passport}`, `POST /sessions/{id}/checks`, `GET /reviews`, `POST /sessions/{id}/override` |
| 4. Administration & Audit (`/api/admin`) | `GET/PUT /config`, `GET /sessions`, `GET /audit-log`, `GET /audit-log/verify`, `POST /demo/reset` |

## Seeded passengers (all fictitious)

| Passenger | Passport | Booking | Scenario | Expected outcome |
|---|---|---|---|---|
| P-1001 Aarav Sharma | N4821907 | BK-5001 | Clean | Auto-cleared on attempt 1 |
| P-1002 Emily Carter | G7305518 | BK-5002 | Flaky camera | Liveness fails twice, passes on attempt 3, auto-cleared |
| P-1003 Rohan Mehta | N9917342 | BK-5003 | Spoof | 3 liveness failures, escalated to officer |
| P-1004 Marcus Hale | C5520871 | BK-5004 | Watchlist HIGH | Escalated, officer may approve or deny |
| P-1005 Priya Nair | N3304415 | BK-5005 | Expired passport | Document rejected, manual review |
| P-1006 Sofia Alvarez | E6618240 | BK-5006 | Face mismatch | Escalated (match score about 0.58) |
| P-1007 Liam O'Connor | I2290763 | BK-5007 | Watchlist CRITICAL, entry restricted | Escalated; officer cannot approve, must deny |

Officers: `OFF-001`, `OFF-002` (active), `OFF-003` (inactive, rejected with 403). Administrator: `ADM-001`. Kiosks: `KIOSK-01` (gate EG-01), `KIOSK-02` (gate EG-02).

## Walkthrough with curl

```bash
# 1. Scan (the Flaky-camera passenger, using only the boarding pass)
curl -s -X POST localhost:8000/api/passengers/scan \
  -H 'Content-Type: application/json' -d '{"booking_id":"BK-5002"}'
# copy the session_id from the response, e.g. SES-XXXXXXXXXX

# 2. Face verification: call 3 times (RETRY, RETRY, FACE_VERIFIED)
curl -s -X POST localhost:8000/api/egate/sessions/SES-XXXXXXXXXX/face-verification

# 3. Immigration and watchlist checks
curl -s -X POST localhost:8000/api/immigration/sessions/SES-XXXXXXXXXX/checks

# 4. Access decision
curl -s -X POST localhost:8000/api/egate/sessions/SES-XXXXXXXXXX/decision

# 5. Passenger walks through
curl -s -X POST localhost:8000/api/egate/sessions/SES-XXXXXXXXXX/passage

# Officer override on an escalated session
curl -s -X POST localhost:8000/api/immigration/sessions/SES-XXXXXXXXXX/override \
  -H 'Content-Type: application/json' \
  -d '{"officer_id":"OFF-001","approve":true,"notes":"Identity confirmed in person at counter."}'

# Audit trail for a session, and chain integrity check
curl -s "localhost:8000/api/admin/audit-log?session_id=SES-XXXXXXXXXX"
curl -s localhost:8000/api/admin/audit-log/verify
```

## Design notes

- **Simulated face engine** (`app/face_engine.py`): enrolled templates are 128-dimension vectors, and the match score is a real cosine similarity. Per-passenger simulation hooks (`sim_liveness_failures`, `sim_impostor` in `biometric_templates`) decide how each capture behaves. Thresholds (`match_threshold` 0.85, `liveness_threshold` 0.80, `max_attempts` 3) are live-editable via `PUT /api/admin/config`.
- **Officer rules**: overrides need an active officer and notes of at least 10 characters. Approving a passenger with a CRITICAL watchlist hit is blocked (409) and the attempt is audited.
- **Audit log**: append-only (SQLite triggers reject UPDATE and DELETE) and hash-chained. `GET /api/admin/audit-log/verify` recomputes the chain and reports the first altered entry.
- **Atomic writes**: each pipeline step updates the session and writes its audit events in one `BEGIN IMMEDIATE` transaction.
- **Prototype limits**: there is no authentication. `officer_id` and `admin_id` are trusted from the request body, and CORS is open for local development. Do not expose this service publicly as-is.
- `POST /api/admin/demo/reset` drops and re-seeds everything, including the audit log. Disable it with `BAPPS_ALLOW_RESET=0`. Use `BAPPS_DB_PATH` to point at a different database file.

## Project layout

```
app/
  main.py            FastAPI app, error handler, router wiring
  config.py          paths, default thresholds
  database.py        schema, connections, transactions, init/reset
  seed.py            seeded passengers, officers, kiosks, gates
  face_engine.py     simulated liveness and face matching
  audit.py           hash-chained audit log
  services.py        pipeline business logic
  schemas.py         request/response models
  errors.py          domain errors
  routers/           passengers.py, egate.py, immigration.py, admin.py
smoke_test.py        end-to-end test of every scenario
```
