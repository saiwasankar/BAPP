"""Central configuration and constants for the BAPPS backend."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SQLite file location. Override with the BAPPS_DB_PATH environment variable.
DB_PATH = os.environ.get("BAPPS_DB_PATH", str(BASE_DIR / "bapps.db"))

# The demo reset endpoint wipes every table (including the audit log).
# Set BAPPS_ALLOW_RESET=0 to disable it.
ALLOW_DEMO_RESET = os.environ.get("BAPPS_ALLOW_RESET", "1") == "1"

# First link of the audit hash chain.
GENESIS_HASH = "0" * 64

# Used only if a key is missing from the threshold_config table.
DEFAULT_THRESHOLDS = {
    "match_threshold": 0.85,     # minimum face similarity (0-1) to count as a match
    "liveness_threshold": 0.80,  # minimum liveness score (0-1) to count as a live face
    "max_attempts": 3,           # face capture attempts before escalation to an officer
}

RISK_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
