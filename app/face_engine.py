"""Simulated face-recognition engine (FaceRecognitionModule).

There is no camera in this prototype, so a capture is simulated deterministically:

* Every enrolled passenger has a 128-dimension template vector.
* A "live" embedding is derived from that template. A genuine passenger gets
  the template plus small noise (cosine similarity about 0.99). An impostor gets
  a look-alike embedding that only partly correlates (about 0.55).
* A liveness score is simulated per attempt. The first N attempts of a passenger
  can be configured to fail (N = sim_liveness_failures in the database), which
  models a spoof attempt (N large) or a flaky camera (N = 2).

The match score is a real cosine similarity, and both thresholds come from the
admin-configurable threshold_config table, so changing a threshold changes outcomes.
"""
import hashlib
import math
import random
from dataclasses import dataclass
from typing import List, Sequence

EMBEDDING_DIM = 128
GENUINE_NOISE = 0.15
IMPOSTOR_BLEND = 0.55


def _rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_template(seed_key: str) -> List[float]:
    """Deterministic enrolled template for a seed key."""
    rng = _rng("template", seed_key)
    return [round(rng.gauss(0.0, 1.0), 6) for _ in range(EMBEDDING_DIM)]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class CaptureResult:
    liveness_score: float
    embedding: List[float]


def simulate_capture(
    template: Sequence[float],
    passenger_id: str,
    attempt_number: int,
    liveness_failures: int,
    impostor: bool,
) -> CaptureResult:
    """Simulate one camera capture for the given attempt (1-based)."""
    rng = _rng("capture", passenger_id, str(attempt_number))

    if attempt_number <= liveness_failures:
        liveness = rng.uniform(0.20, 0.55)   # printed photo / replay / bad frame
    else:
        liveness = rng.uniform(0.88, 0.99)   # real, live face

    if impostor:
        residual = math.sqrt(1.0 - IMPOSTOR_BLEND ** 2)
        embedding = [IMPOSTOR_BLEND * t + residual * rng.gauss(0.0, 1.0) for t in template]
    else:
        embedding = [t + rng.gauss(0.0, GENUINE_NOISE) for t in template]

    return CaptureResult(liveness_score=round(liveness, 4), embedding=embedding)


def match_score(template: Sequence[float], embedding: Sequence[float]) -> float:
    """Similarity in [0, 1] between the enrolled template and a live embedding."""
    return round(max(0.0, cosine_similarity(template, embedding)), 4)
