"""Pre-seeded mock data covering every branch of the verification pipeline.

All people, passports and watchlist entries are fictitious demo data.
Dates are relative to the day the database is seeded.
"""
import json
import sqlite3
from datetime import date, datetime, time, timedelta

from . import config
from .face_engine import generate_template

# sim_liveness_failures: number of leading capture attempts that fail liveness (99 = always).
# sim_impostor: 1 = the live face does not belong to the enrolled passenger.
PASSENGERS = [
    {
        "passenger_id": "P-1001", "full_name": "Aarav Sharma", "nationality": "IN",
        "date_of_birth": "1992-03-14", "passport_number": "N4821907", "issuing_country": "IN",
        "expiry_days": 1500, "booking_id": "BK-5001", "flight_number": "BX204",
        "gate_number": "B12", "departure": (14, 35),
        "sim_liveness_failures": 0, "sim_impostor": 0,
        "visa_status": "VALID_EXIT_PERMIT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": None,
        "demo_scenario": "Clean passenger: automatic gate clearance on the first attempt.",
    },
    {
        "passenger_id": "P-1002", "full_name": "Emily Carter", "nationality": "GB",
        "date_of_birth": "1988-11-02", "passport_number": "G7305518", "issuing_country": "GB",
        "expiry_days": 900, "booking_id": "BK-5002", "flight_number": "BX310",
        "gate_number": "A07", "departure": (16, 10),
        "sim_liveness_failures": 2, "sim_impostor": 0,
        "visa_status": "VISA_EXEMPT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": None,
        "demo_scenario": "Flaky camera: liveness fails twice, passes on attempt 3, then auto-clears.",
    },
    {
        "passenger_id": "P-1003", "full_name": "Rohan Mehta", "nationality": "IN",
        "date_of_birth": "1995-07-21", "passport_number": "N9917342", "issuing_country": "IN",
        "expiry_days": 2200, "booking_id": "BK-5003", "flight_number": "BX118",
        "gate_number": "C03", "departure": (18, 5),
        "sim_liveness_failures": 99, "sim_impostor": 0,
        "visa_status": "VALID_EXIT_PERMIT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": None,
        "demo_scenario": "Spoof attempt: liveness fails all 3 attempts, escalated to an officer.",
    },
    {
        "passenger_id": "P-1004", "full_name": "Marcus Hale", "nationality": "CA",
        "date_of_birth": "1979-05-30", "passport_number": "C5520871", "issuing_country": "CA",
        "expiry_days": 700, "booking_id": "BK-5004", "flight_number": "BX442",
        "gate_number": "B04", "departure": (19, 45),
        "sim_liveness_failures": 0, "sim_impostor": 0,
        "visa_status": "VISA_EXEMPT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": ("W-2001", "HIGH", "Open financial inquiry (demo data)"),
        "demo_scenario": "Watchlist hit (HIGH): face matches, officer review required; officer may override.",
    },
    {
        "passenger_id": "P-1005", "full_name": "Priya Nair", "nationality": "IN",
        "date_of_birth": "1990-01-09", "passport_number": "N3304415", "issuing_country": "IN",
        "expiry_days": -30, "booking_id": "BK-5005", "flight_number": "BX275",
        "gate_number": "A15", "departure": (20, 20),
        "sim_liveness_failures": 0, "sim_impostor": 0,
        "visa_status": "VALID_EXIT_PERMIT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": None,
        "demo_scenario": "Expired passport: rejected at document stage, sent to manual document review.",
    },
    {
        "passenger_id": "P-1006", "full_name": "Sofia Alvarez", "nationality": "ES",
        "date_of_birth": "1985-09-17", "passport_number": "E6618240", "issuing_country": "ES",
        "expiry_days": 1100, "booking_id": "BK-5006", "flight_number": "BX530",
        "gate_number": "C11", "departure": (21, 0),
        "sim_liveness_failures": 0, "sim_impostor": 1,
        "visa_status": "VISA_EXEMPT", "entry_permitted": 1, "immigration_notes": "",
        "watchlist": None,
        "demo_scenario": "Face mismatch: live face does not match the enrolled template, officer review.",
    },
    {
        "passenger_id": "P-1007", "full_name": "Liam O'Connor", "nationality": "IE",
        "date_of_birth": "1983-12-04", "passport_number": "I2290763", "issuing_country": "IE",
        "expiry_days": 1300, "booking_id": "BK-5007", "flight_number": "BX601",
        "gate_number": "B09", "departure": (22, 15),
        "sim_liveness_failures": 0, "sim_impostor": 0,
        "visa_status": "ENTRY_RESTRICTED", "entry_permitted": 0,
        "immigration_notes": "Travel restriction on file (demo data)",
        "watchlist": ("W-2002", "CRITICAL", "Active international alert (demo data)"),
        "demo_scenario": "Watchlist hit (CRITICAL) and restricted entry: officer cannot override, must deny.",
    },
]

OFFICERS = [
    ("OFF-001", "Meera Iyer", "B-7731", 1),
    ("OFF-002", "Daniel Brooks", "B-4410", 1),
    ("OFF-003", "Hana Kimura", "B-9002", 0),  # inactive: used to test authorisation checks
]

ADMINISTRATORS = [("ADM-001", "Anita Rao")]

GATES = [("EG-01", "CLOSED"), ("EG-02", "CLOSED")]

KIOSKS = [
    ("KIOSK-01", "Terminal 1 - Zone A", "ONLINE", "EG-01"),
    ("KIOSK-02", "Terminal 1 - Zone B", "ONLINE", "EG-02"),
]


def seed_database(conn: sqlite3.Connection) -> None:
    today = date.today()

    for key, value in config.DEFAULT_THRESHOLDS.items():
        conn.execute("INSERT INTO threshold_config (key, value) VALUES (?, ?)", (key, float(value)))

    conn.executemany("INSERT INTO gates (gate_id, status) VALUES (?, ?)", GATES)
    conn.executemany(
        "INSERT INTO kiosks (kiosk_id, location, status, gate_id) VALUES (?, ?, ?, ?)", KIOSKS
    )
    conn.executemany(
        "INSERT INTO officers (officer_id, full_name, badge_number, active) VALUES (?, ?, ?, ?)",
        OFFICERS,
    )
    conn.executemany(
        "INSERT INTO administrators (admin_id, full_name) VALUES (?, ?)", ADMINISTRATORS
    )

    for p in PASSENGERS:
        conn.execute(
            """INSERT INTO passengers
               (passenger_id, full_name, nationality, date_of_birth, demo_scenario)
               VALUES (?, ?, ?, ?, ?)""",
            (p["passenger_id"], p["full_name"], p["nationality"], p["date_of_birth"],
             p["demo_scenario"]),
        )
        conn.execute(
            """INSERT INTO passports (passport_number, passenger_id, issuing_country, expiry_date)
               VALUES (?, ?, ?, ?)""",
            (p["passport_number"], p["passenger_id"], p["issuing_country"],
             (today + timedelta(days=p["expiry_days"])).isoformat()),
        )
        hour, minute = p["departure"]
        conn.execute(
            """INSERT INTO boarding_passes
               (booking_id, passenger_id, passport_number, flight_number, gate_number, departure_time)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (p["booking_id"], p["passenger_id"], p["passport_number"], p["flight_number"],
             p["gate_number"], datetime.combine(today, time(hour, minute)).isoformat()),
        )
        conn.execute(
            """INSERT INTO biometric_templates
               (template_id, passenger_id, feature_vector, enrollment_date,
                sim_liveness_failures, sim_impostor)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("TPL-" + p["passenger_id"].split("-")[1], p["passenger_id"],
             json.dumps(generate_template("template:" + p["passenger_id"])),
             (today - timedelta(days=200)).isoformat(),
             p["sim_liveness_failures"], p["sim_impostor"]),
        )
        conn.execute(
            """INSERT INTO immigration_records (passport_number, visa_status, entry_permitted, notes)
               VALUES (?, ?, ?, ?)""",
            (p["passport_number"], p["visa_status"], p["entry_permitted"], p["immigration_notes"]),
        )
        if p["watchlist"]:
            entry_id, risk, reason = p["watchlist"]
            conn.execute(
                """INSERT INTO watchlist_entries (entry_id, passport_number, risk_level, reason)
                   VALUES (?, ?, ?, ?)""",
                (entry_id, p["passport_number"], risk, reason),
            )
