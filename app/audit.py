"""Structured, tamper-evident audit logging (Administration & Audit package).

Each audit row stores the hash of the previous row and its own hash, forming a
chain. SQLite triggers (see database.py) block UPDATE and DELETE on the table,
and verify_chain() recomputes every hash to detect tampering.

log_event() must be called inside database.transaction() so the read of the
previous hash and the insert happen atomically.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import config


def _canonical(details: Optional[Dict[str, Any]]) -> str:
    return json.dumps(details or {}, sort_keys=True, separators=(",", ":"), default=str)


def _compute_hash(
    prev_hash: str,
    timestamp: str,
    session_id: Optional[str],
    event_type: str,
    actor_type: str,
    actor_id: str,
    decision: Optional[str],
    details_json: str,
) -> str:
    payload = json.dumps(
        [prev_hash, timestamp, session_id, event_type, actor_type, actor_id, decision, details_json],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    actor_type: str,
    actor_id: str,
    session_id: Optional[str] = None,
    decision: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    last = conn.execute("SELECT entry_hash FROM audit_log ORDER BY log_id DESC LIMIT 1").fetchone()
    prev_hash = last["entry_hash"] if last else config.GENESIS_HASH
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    details_json = _canonical(details)
    entry_hash = _compute_hash(
        prev_hash, timestamp, session_id, event_type, actor_type, actor_id, decision, details_json
    )
    conn.execute(
        """INSERT INTO audit_log
           (timestamp, session_id, event_type, actor_type, actor_id, decision,
            details, prev_hash, entry_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (timestamp, session_id, event_type, actor_type, actor_id, decision,
         details_json, prev_hash, entry_hash),
    )


def _row_to_entry(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "log_id": row["log_id"],
        "timestamp": row["timestamp"],
        "session_id": row["session_id"],
        "event_type": row["event_type"],
        "actor_type": row["actor_type"],
        "actor_id": row["actor_id"],
        "decision": row["decision"],
        "details": json.loads(row["details"]),
        "prev_hash": row["prev_hash"],
        "entry_hash": row["entry_hash"],
    }


def list_events(
    conn: sqlite3.Connection,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    actor_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    clauses = []
    params = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if actor_id:
        clauses.append("actor_id = ?")
        params.append(actor_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    total = conn.execute(f"SELECT COUNT(*) FROM audit_log {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM audit_log {where} ORDER BY log_id ASC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_entry(r) for r in rows],
    }


def verify_chain(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Recompute every hash in order and report the first inconsistency, if any."""
    expected_prev = config.GENESIS_HASH
    checked = 0
    for row in conn.execute("SELECT * FROM audit_log ORDER BY log_id ASC"):
        recomputed = _compute_hash(
            row["prev_hash"], row["timestamp"], row["session_id"], row["event_type"],
            row["actor_type"], row["actor_id"], row["decision"], row["details"],
        )
        if row["prev_hash"] != expected_prev:
            return {
                "valid": False,
                "entries_checked": checked,
                "first_invalid_log_id": row["log_id"],
                "message": "Chain link broken: an earlier entry was removed or altered.",
            }
        if row["entry_hash"] != recomputed:
            return {
                "valid": False,
                "entries_checked": checked,
                "first_invalid_log_id": row["log_id"],
                "message": "Entry content does not match its stored hash.",
            }
        expected_prev = row["entry_hash"]
        checked += 1
    return {
        "valid": True,
        "entries_checked": checked,
        "first_invalid_log_id": None,
        "message": "Audit chain is intact.",
    }
