"""SQLite database layer: schema, connections, transactions, init and reset.

Connections run in autocommit mode (isolation_level=None). Every write path wraps
its work in transaction(), which issues BEGIN IMMEDIATE so writers are serialised
and the audit hash chain cannot fork.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS passengers (
    passenger_id   TEXT PRIMARY KEY,
    full_name      TEXT NOT NULL,
    nationality    TEXT NOT NULL,
    date_of_birth  TEXT NOT NULL,
    demo_scenario  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS passports (
    passport_number  TEXT PRIMARY KEY,
    passenger_id     TEXT NOT NULL UNIQUE REFERENCES passengers(passenger_id),
    issuing_country  TEXT NOT NULL,
    expiry_date      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS boarding_passes (
    booking_id       TEXT PRIMARY KEY,
    passenger_id     TEXT NOT NULL REFERENCES passengers(passenger_id),
    passport_number  TEXT NOT NULL REFERENCES passports(passport_number),
    flight_number    TEXT NOT NULL,
    gate_number      TEXT NOT NULL,
    departure_time   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS biometric_templates (
    template_id            TEXT PRIMARY KEY,
    passenger_id           TEXT NOT NULL UNIQUE REFERENCES passengers(passenger_id),
    feature_vector         TEXT NOT NULL,
    enrollment_date        TEXT NOT NULL,
    sim_liveness_failures  INTEGER NOT NULL DEFAULT 0,
    sim_impostor           INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS immigration_records (
    passport_number  TEXT PRIMARY KEY REFERENCES passports(passport_number),
    visa_status      TEXT NOT NULL,
    entry_permitted  INTEGER NOT NULL,
    notes            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS watchlist_entries (
    entry_id         TEXT PRIMARY KEY,
    passport_number  TEXT NOT NULL REFERENCES passports(passport_number),
    risk_level       TEXT NOT NULL CHECK (risk_level IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    reason           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS officers (
    officer_id    TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    badge_number  TEXT NOT NULL UNIQUE,
    active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS administrators (
    admin_id   TEXT PRIMARY KEY,
    full_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gates (
    gate_id  TEXT PRIMARY KEY,
    status   TEXT NOT NULL CHECK (status IN ('CLOSED','OPEN'))
);

CREATE TABLE IF NOT EXISTS kiosks (
    kiosk_id  TEXT PRIMARY KEY,
    location  TEXT NOT NULL,
    status    TEXT NOT NULL,
    gate_id   TEXT NOT NULL REFERENCES gates(gate_id)
);

CREATE TABLE IF NOT EXISTS threshold_config (
    key    TEXT PRIMARY KEY,
    value  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_sessions (
    session_id         TEXT PRIMARY KEY,
    passenger_id       TEXT NOT NULL REFERENCES passengers(passenger_id),
    passport_number    TEXT NOT NULL REFERENCES passports(passport_number),
    booking_id         TEXT,
    kiosk_id           TEXT NOT NULL REFERENCES kiosks(kiosk_id),
    state              TEXT NOT NULL CHECK (state IN
        ('AWAITING_FACE','FACE_VERIFIED','CHECKS_COMPLETE','AWAITING_OFFICER',
         'GATE_OPEN','DENIED','COMPLETED')),
    escalation_reason  TEXT,
    document_issues    TEXT NOT NULL DEFAULT '[]',
    attempts           INTEGER NOT NULL DEFAULT 0,
    liveness_score     REAL,
    match_score        REAL,
    match_passed       INTEGER,
    watchlist_hit      INTEGER,
    watchlist_risk     TEXT,
    immigration_status TEXT,
    entry_permitted    INTEGER,
    decision           TEXT CHECK (decision IS NULL OR decision IN
        ('AUTO_CLEARED','OFFICER_APPROVED','OFFICER_DENIED')),
    decision_reasons   TEXT NOT NULL DEFAULT '[]',
    gate_id            TEXT REFERENCES gates(gate_id),
    officer_id         TEXT REFERENCES officers(officer_id),
    officer_notes      TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON verification_sessions(state);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    session_id  TEXT,
    event_type  TEXT NOT NULL,
    actor_type  TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    decision    TEXT,
    details     TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    entry_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one connection per request."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Atomic write transaction. Do not nest."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Ensure optional extended columns exist for passport photo and real face templates."""
    columns_to_add = [
        ("verification_sessions", "passport_photo_b64", "TEXT"),
        ("verification_sessions", "passport_face_encoding", "TEXT"),
        ("biometric_templates", "real_face_encoding", "TEXT"),
    ]
    for table, col, col_type in columns_to_add:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except Exception:
            pass


def init_db() -> None:
    """Create the schema and seed demo data if the database is empty."""
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        _migrate_columns(conn)
        count = conn.execute("SELECT COUNT(*) FROM passengers").fetchone()[0]
        if count == 0:
            from .seed import seed_database

            with transaction(conn):
                seed_database(conn)
    finally:
        conn.close()


def reset_database() -> None:
    """Drop every table and rebuild the seeded demo database."""
    conn = connect()
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute("PRAGMA foreign_keys = ON")
    finally:
        conn.close()
    init_db()
