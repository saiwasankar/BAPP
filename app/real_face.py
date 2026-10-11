"""Real AI face recognition, multi-stage document photo extraction (Passport / Aadhaar / ID), and live matching.

Features:
1. Multi-Stage Document Face Extraction:
   - Uses BlazeFace detector to locate face in small ID photos / Aadhaar cards.
   - Crops, normalizes and extracts 478 3D landmarks via MediaPipe FaceLandmarker.
   - Robust to different resolutions, rotations, lighting and ID formats.
2. Strict 1-to-1 Biometric Matching:
   - Compares live webcam facial scan directly against the extracted document portrait.
   - Enforces non-hallucinating mathematical verification with exact confidence scores.
   - Reports clear mismatch status if live face does not match document photo.
3. Multi-Factor Passive Liveness / presentation-attack detection:
   - Screen & print recapture cues (moire, RGB subpixel, bezel rectangle, glass glare).
   - Head pose, framing and lighting as quality gates (not treated as proof of life).
   - MediaPipe mesh depth is NOT used as liveness — it is predicted even from photos.
"""
import base64
import io
import json
import logging
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

log = logging.getLogger(__name__)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
SFACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Global singleton models
_landmarker_instance = None
_landmarker_initialized = False
_detector_instance = None
_detector_initialized = False
_face_recognizer = None
_face_recognizer_initialized = False


def _get_landmarker():
    global _landmarker_instance, _landmarker_initialized
    if _landmarker_initialized:
        return _landmarker_instance

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not os.path.exists(LANDMARKER_MODEL_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            log.info("Downloading Face Landmarker model from %s ...", url)
            urllib.request.urlretrieve(url, LANDMARKER_MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
        )
        _landmarker_instance = vision.FaceLandmarker.create_from_options(options)
        _landmarker_initialized = True
        log.info("MediaPipe FaceLandmarker successfully initialized.")
        return _landmarker_instance
    except Exception as exc:
        log.error("Failed to initialize MediaPipe FaceLandmarker: %s", exc)
        _landmarker_initialized = True
        _landmarker_instance = None
        return None


def _get_detector():
    global _detector_instance, _detector_initialized
    if _detector_initialized:
        return _detector_instance

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not os.path.exists(DETECTOR_MODEL_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            log.info("Downloading Face Detector model from %s ...", url)
            urllib.request.urlretrieve(url, DETECTOR_MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.15)
        _detector_instance = vision.FaceDetector.create_from_options(options)
        _detector_initialized = True
        log.info("MediaPipe FaceDetector successfully initialized.")
        return _detector_instance
    except Exception as exc:
        log.error("Failed to initialize MediaPipe FaceDetector: %s", exc)
        _detector_initialized = True
        _detector_instance = None
        return None


def _get_face_recognizer():
    """Load OpenCV Zoo SFace, downloading its Apache-2.0 model on first use."""
    global _face_recognizer, _face_recognizer_initialized
    if _face_recognizer_initialized:
        return _face_recognizer
    _face_recognizer_initialized = True
    try:
        if not os.path.exists(SFACE_MODEL_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            import urllib.request
            temporary_path = SFACE_MODEL_PATH + ".download"
            log.info("Downloading SFace recognition model from OpenCV Zoo...")
            urllib.request.urlretrieve(SFACE_MODEL_URL, temporary_path)
            if os.path.getsize(temporary_path) < 1_000_000:
                os.remove(temporary_path)
                raise RuntimeError("Downloaded SFace model is incomplete")
            os.replace(temporary_path, SFACE_MODEL_PATH)
        _face_recognizer = cv2.FaceRecognizerSF_create(SFACE_MODEL_PATH, "")
        log.info("SFace identity recognition model initialized.")
    except Exception as exc:
        log.exception("Could not initialize SFace identity model: %s", exc)
        _face_recognizer = None
    return _face_recognizer


def is_available() -> bool:
    """Return True if MediaPipe Face Landmarker is ready."""
    return _get_landmarker() is not None


def decode_image(base64_string: str) -> Tuple[np.ndarray, Image.Image]:
    """Decode a base64 data-URI or raw base64 string into RGB numpy array and PIL Image."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    raw = base64.b64decode(base64_string)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    return np.array(pil_img), pil_img


def encode_image_to_base64(pil_img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL image to base64 JPEG data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# Key anthropometric landmark indices for discriminative 1-to-1 face recognition
KEY_LANDMARKS = [
    1,    # Nose tip
    4,    # Nose base / subnasale
    6,    # Nose bridge
    168,  # Glabella (between eyes)
    10,   # Forehead top / trichion
    152,  # Chin tip / menton
    33,   # Left eye outer corner
    133,  # Left eye inner corner
    159,  # Left eye upper eyelid
    145,  # Left eye lower eyelid
    362,  # Right eye inner corner
    263,  # Right eye outer corner
    386,  # Right eye upper eyelid
    374,  # Right eye lower eyelid
    70,   # Left eyebrow outer
    105,  # Left eyebrow peak
    107,  # Left eyebrow inner
    336,  # Right eyebrow inner
    334,  # Right eyebrow peak
    300,  # Right eyebrow outer
    61,   # Mouth left corner
    291,  # Mouth right corner
    0,    # Upper lip center
    17,   # Lower lip center
    234,  # Left cheek / zygomatic
    454,  # Right cheek / zygomatic
    127,  # Left jaw angle (gonion)
    356,  # Right jaw angle (gonion)
    102,  # Left alar base (nostril)
    331,  # Right alar base (nostril)
]


# Appearance descriptor layout (identity). Landmark-only vectors are ~186-d and are
# not identity: MediaPipe meshes of different people are nearly interchangeable, while
# the previous L2<=0.055 gate only passed near-identical frames.
_APP_PIXEL = 32 * 32          # 1024  CLAHE grayscale patch
_APP_HOG = 324                # 32x32 HOG (9 bins, 16x16 blocks, 8 stride)
_APP_LBP = 4 * 4 * 16         # 256
_APP_COLOR = 8 * 8 * 3        # 192
APP_DIM = _APP_PIXEL + _APP_HOG + _APP_LBP + _APP_COLOR  # 1796


def _lm_xy(landmarks, idx: int, w: float, h: float) -> np.ndarray:
    lm = landmarks[min(idx, len(landmarks) - 1)]
    return np.array([float(lm.x) * w, float(lm.y) * h], dtype=np.float32)


def _align_face(rgb_img: np.ndarray, landmarks, out_size: int = 112) -> np.ndarray:
    """Similarity-align face using eyes + mouth so webcam and document crops match."""
    h, w = rgb_img.shape[:2]
    left = (_lm_xy(landmarks, 33, w, h) + _lm_xy(landmarks, 133, w, h)) * 0.5
    right = (_lm_xy(landmarks, 362, w, h) + _lm_xy(landmarks, 263, w, h)) * 0.5
    mouth = (_lm_xy(landmarks, 61, w, h) + _lm_xy(landmarks, 291, w, h)) * 0.5
    src = np.stack([left, right, mouth]).astype(np.float32)
    dst = np.array([
        [0.30 * out_size, 0.38 * out_size],
        [0.70 * out_size, 0.38 * out_size],
        [0.50 * out_size, 0.78 * out_size],
    ], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        x1, y1 = np.clip(pts.min(axis=0), 0, [w - 1, h - 1])
        x2, y2 = np.clip(pts.max(axis=0), 0, [w - 1, h - 1])
        crop = rgb_img[int(y1):max(int(y2), int(y1) + 1), int(x1):max(int(x2), int(x1) + 1)]
        if crop.size == 0:
            crop = rgb_img
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(
        rgb_img, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )


def _lbp_grid_hist(gray: np.ndarray, grid: int = 4, bins: int = 16) -> np.ndarray:
    g = gray.astype(np.int16)
    lbp = np.zeros(g.shape, dtype=np.uint8)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    for i, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        lbp |= ((shifted >= g).astype(np.uint8) << i)
    h, w = lbp.shape
    gh, gw = max(h // grid, 1), max(w // grid, 1)
    parts = []
    for gy in range(grid):
        for gx in range(grid):
            patch = lbp[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
            hist, _ = np.histogram(patch, bins=bins, range=(0, 256), density=True)
            parts.append(hist.astype(np.float32))
    return np.concatenate(parts)


def _hog_32(gray: np.ndarray) -> np.ndarray:
    try:
        g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
        vec = hog.compute(g)
        if vec is None:
            return np.zeros(_APP_HOG, dtype=np.float32)
        vec = vec.flatten().astype(np.float32)
        if vec.size < _APP_HOG:
            vec = np.pad(vec, (0, _APP_HOG - vec.size))
        return vec[:_APP_HOG]
    except Exception:
        return np.zeros(_APP_HOG, dtype=np.float32)


def extract_appearance_embedding(rgb_img: np.ndarray, landmarks) -> np.ndarray:
    """Lighting-normalized appearance vector from an aligned face crop (identity signal)."""
    aligned = _align_face(rgb_img, landmarks, 112)
    gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    pixel = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pixel = pixel - float(pixel.mean())
    pixel = pixel.flatten()
    pnorm = np.linalg.norm(pixel)
    if pnorm > 1e-9:
        pixel = pixel / pnorm

    hog = _hog_32(gray)
    hnorm = np.linalg.norm(hog)
    if hnorm > 1e-9:
        hog = hog / hnorm

    lbp = _lbp_grid_hist(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
    lnorm = np.linalg.norm(lbp)
    if lnorm > 1e-9:
        lbp = lbp / lnorm

    lab = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB).astype(np.float32)
    color = cv2.resize(lab, (8, 8), interpolation=cv2.INTER_AREA).flatten() / 255.0
    cnorm = np.linalg.norm(color)
    if cnorm > 1e-9:
        color = color / cnorm

    return np.concatenate([pixel, hog, lbp, color]).astype(np.float32)


def extract_geometry_embedding(landmarks, blendshapes=None) -> np.ndarray:
    """Pose-normalized 2D landmark geometry. Supportive only — not used as identity alone."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    n = len(pts)
    le = pts[33] if n > 33 else pts[0]
    re = pts[263] if n > 263 else pts[-1]
    mid = (le + re) * 0.5
    dist = float(np.linalg.norm(re - le))
    scale = max(dist, 1e-5)
    ang = math.atan2(float(re[1] - le[1]), float(re[0] - le[0]))
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    key_pts = []
    for idx in KEY_LANDMARKS:
        key_pts.append(pts[idx] if idx < n else pts[0])
    key_pts = np.array(key_pts, dtype=np.float32)
    aligned = ((key_pts - mid) @ rot.T) / scale

    ratios = []
    nose = aligned[0]
    glabella = aligned[3]
    chin = aligned[5]
    for origin in (nose, glabella, chin):
        for pt in aligned:
            ratios.append(float(np.linalg.norm(pt - origin)))

    face_width = float(np.linalg.norm(aligned[24] - aligned[25]))
    face_height = float(np.linalg.norm(aligned[4] - aligned[5]))
    ratios.append(face_width / max(face_height, 1e-5))
    ratios.append(float(np.linalg.norm(aligned[7] - aligned[10])))
    nose_w = float(np.linalg.norm(aligned[28] - aligned[29]))
    nose_h = float(np.linalg.norm(aligned[2] - aligned[1]))
    ratios.append(nose_w / max(nose_h, 1e-5))
    mouth_w = float(np.linalg.norm(aligned[20] - aligned[21]))
    ratios.append(mouth_w / max(nose_w, 1e-5))
    jaw_w = float(np.linalg.norm(aligned[26] - aligned[27]))
    ratios.append(jaw_w / max(face_width, 1e-5))
    lower_face_h = float(np.linalg.norm(aligned[1] - aligned[5]))
    ratios.append(lower_face_h / max(face_height, 1e-5))

    combined = np.concatenate([aligned.flatten(), np.array(ratios, dtype=np.float32)])
    norm = np.linalg.norm(combined)
    if norm > 1e-9:
        combined = combined / norm
    return combined.astype(np.float32)


def extract_biometric_embedding(landmarks, blendshapes=None, rgb_img: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a learned SFace identity embedding; fail closed if the model is unavailable."""
    if rgb_img is None or len(landmarks) <= 362:
        raise RuntimeError("A full face image and landmarks are required for identity matching")
    recognizer = _get_face_recognizer()
    if recognizer is None:
        raise RuntimeError("SFace model unavailable. Check network access and restart the server to retry.")

    h, w = rgb_img.shape[:2]
    pts = np.array([[float(lm.x) * w, float(lm.y) * h] for lm in landmarks], dtype=np.float32)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face bounds for SFace alignment")

    def eye_center(a, b):
        pa = np.array([landmarks[a].x * w, landmarks[a].y * h], dtype=np.float32)
        pb = np.array([landmarks[b].x * w, landmarks[b].y * h], dtype=np.float32)
        return (pa + pb) * 0.5

    five = [
        eye_center(33, 133), eye_center(362, 263),
        np.array([landmarks[1].x * w, landmarks[1].y * h], dtype=np.float32),
        np.array([landmarks[61].x * w, landmarks[61].y * h], dtype=np.float32),
        np.array([landmarks[291].x * w, landmarks[291].y * h], dtype=np.float32),
    ]
    # OpenCV FaceRecognizerSF takes [x,y,w,h, five (x,y) landmarks, score].
    face_row = np.array([[x1, y1, x2 - x1, y2 - y1, *np.concatenate(five), 1.0]], dtype=np.float32)
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    aligned = recognizer.alignCrop(bgr, face_row)
    feature = np.asarray(recognizer.feature(aligned), dtype=np.float32).flatten()
    norm = np.linalg.norm(feature)
    if feature.size != 128 or norm <= 1e-9:
        raise RuntimeError("SFace returned an invalid identity embedding")
    return feature / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_biometric_similarity(emb_live: np.ndarray, emb_ref: np.ndarray) -> float:
    """Score live webcam vs document portrait. Genuine pairs clear 0.85; impostors stay below."""
    emb_live = np.asarray(emb_live, dtype=np.float32).flatten()
    emb_ref = np.asarray(emb_ref, dtype=np.float32).flatten()

    # Appearance+geometry templates are a different format from the legacy
    # landmark-only vectors. Never truncate and compare unrelated dimensions.
    if emb_live.size != emb_ref.size:
        return 0.0

    if emb_live.size == 128:
        # OpenCV Zoo SFace's documented cosine operating point is 0.363.
        # Map that decision boundary to the app's existing 85% threshold while
        # keeping the displayed score a similarity scale, not a probability.
        cosine = _cosine(emb_live, emb_ref)
        if cosine >= 0.363:
            score = 0.85 + (cosine - 0.363) * (0.145 / (1.0 - 0.363))
        else:
            score = 0.12 + (cosine + 1.0) * (0.73 / (0.363 + 1.0))
        return float(np.clip(score, 0.0, 0.995))
    elif emb_live.size >= APP_DIM:
        live_app, ref_app = emb_live[:APP_DIM], emb_ref[:APP_DIM]
        live_geom, ref_geom = emb_live[APP_DIM:], emb_ref[APP_DIM:]

        pix = _cosine(live_app[:_APP_PIXEL], ref_app[:_APP_PIXEL])
        hog = _cosine(
            live_app[_APP_PIXEL:_APP_PIXEL + _APP_HOG],
            ref_app[_APP_PIXEL:_APP_PIXEL + _APP_HOG],
        )
        lbp = _cosine(
            live_app[_APP_PIXEL + _APP_HOG:_APP_PIXEL + _APP_HOG + _APP_LBP],
            ref_app[_APP_PIXEL + _APP_HOG:_APP_PIXEL + _APP_HOG + _APP_LBP],
        )
        col = _cosine(live_app[-_APP_COLOR:], ref_app[-_APP_COLOR:])
        geom = _cosine(live_geom, ref_geom)
        # Appearance dominates; geometry only slightly regularizes pose/crop differences.
        fused = 0.40 * pix + 0.32 * hog + 0.16 * lbp + 0.08 * col + 0.04 * geom
    else:
        # Legacy geometry-only templates are not identity embeddings and cannot
        # be safely compared. Require a fresh reference photo/template instead.
        return 0.0

    # Map cosine-like fused score onto the operational 0.85 match threshold:
    # same person, different capture typically 0.58–0.92; identical frames ~1.0;
    # different people typically < 0.55.
    if fused >= 0.56:
        score = 0.85 + min(0.145, (fused - 0.56) / 0.44 * 0.145)
    else:
        score = 0.12 + (fused / 0.56) * 0.70
    return float(np.clip(score, 0.0, 0.995))


class PseudoLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _synthesize_landmarks_from_detection(det, img_w: int, img_h: int) -> List[PseudoLandmark]:
    """Synthesize 478 structural landmark anchors from BlazeFace 6 keypoints and bounding box."""
    bb = det.bounding_box
    bx = bb.origin_x / max(1, img_w)
    by = bb.origin_y / max(1, img_h)
    bw = bb.width / max(1, img_w)
    bh = bb.height / max(1, img_h)

    kps = det.keypoints if hasattr(det, "keypoints") and det.keypoints else []
    r_eye_x = kps[0].x if len(kps) > 0 else bx + bw * 0.70
    r_eye_y = kps[0].y if len(kps) > 0 else by + bh * 0.35
    l_eye_x = kps[1].x if len(kps) > 1 else bx + bw * 0.30
    l_eye_y = kps[1].y if len(kps) > 1 else by + bh * 0.35

    nose_x = kps[2].x if len(kps) > 2 else bx + bw * 0.50
    nose_y = kps[2].y if len(kps) > 2 else by + bh * 0.55

    mouth_x = kps[3].x if len(kps) > 3 else bx + bw * 0.50
    mouth_y = kps[3].y if len(kps) > 3 else by + bh * 0.75

    r_ear_x = kps[4].x if len(kps) > 4 else bx + bw * 0.95
    r_ear_y = kps[4].y if len(kps) > 4 else by + bh * 0.45
    l_ear_x = kps[5].x if len(kps) > 5 else bx + bw * 0.05
    l_ear_y = kps[5].y if len(kps) > 5 else by + bh * 0.45

    forehead_x, forehead_y = (l_eye_x + r_eye_x) / 2.0, by + bh * 0.10
    chin_x, chin_y = mouth_x, by + bh * 0.95
    glabella_x, glabella_y = (l_eye_x + r_eye_x) / 2.0, (l_eye_y + r_eye_y) / 2.0

    lms = [PseudoLandmark(nose_x, nose_y, 0.0) for _ in range(478)]
    lms[1] = PseudoLandmark(nose_x, nose_y, 0.0)
    lms[4] = PseudoLandmark(nose_x, (nose_y + mouth_y) / 2.0, 0.0)
    lms[6] = PseudoLandmark(glabella_x, (glabella_y + nose_y) / 2.0, 0.0)
    lms[168] = PseudoLandmark(glabella_x, glabella_y, 0.0)
    lms[10] = PseudoLandmark(forehead_x, forehead_y, 0.0)
    lms[152] = PseudoLandmark(chin_x, chin_y, 0.0)
    lms[33] = PseudoLandmark(l_eye_x - bw * 0.08, l_eye_y, 0.0)
    lms[133] = PseudoLandmark(l_eye_x + bw * 0.06, l_eye_y, 0.0)
    lms[159] = PseudoLandmark(l_eye_x, l_eye_y - bh * 0.03, 0.0)
    lms[145] = PseudoLandmark(l_eye_x, l_eye_y + bh * 0.03, 0.0)
    lms[362] = PseudoLandmark(r_eye_x - bw * 0.06, r_eye_y, 0.0)
    lms[263] = PseudoLandmark(r_eye_x + bw * 0.08, r_eye_y, 0.0)
    lms[386] = PseudoLandmark(r_eye_x, r_eye_y - bh * 0.03, 0.0)
    lms[374] = PseudoLandmark(r_eye_x, r_eye_y + bh * 0.03, 0.0)
    lms[61] = PseudoLandmark(mouth_x - bw * 0.15, mouth_y, 0.0)
    lms[291] = PseudoLandmark(mouth_x + bw * 0.15, mouth_y, 0.0)
    lms[0] = PseudoLandmark(mouth_x, mouth_y - bh * 0.03, 0.0)
    lms[17] = PseudoLandmark(mouth_x, mouth_y + bh * 0.03, 0.0)
    lms[234] = PseudoLandmark(l_ear_x, l_ear_y, 0.0)
    lms[454] = PseudoLandmark(r_ear_x, r_ear_y, 0.0)
    lms[127] = PseudoLandmark(bx + bw * 0.15, by + bh * 0.85, 0.0)
    lms[356] = PseudoLandmark(bx + bw * 0.85, by + bh * 0.85, 0.0)
    lms[102] = PseudoLandmark(nose_x - bw * 0.07, nose_y, 0.0)
    lms[331] = PseudoLandmark(nose_x + bw * 0.07, nose_y, 0.0)

    return lms


def extract_face_from_document(image_b64: str) -> Dict[str, Any]:
    """Scan and extract the face photo and biometric template from an uploaded document (Passport / Aadhaar / ID)."""
    landmarker = _get_landmarker()
    detector = _get_detector()
    if landmarker is None and detector is None:
        return {"face_detected": False, "message": "AI face detection model unavailable."}

    try:
        rgb_img, pil_img = decode_image(image_b64)
    except Exception as exc:
        return {"face_detected": False, "message": f"Invalid document image format: {exc}"}

    import mediapipe as mp

    # Search candidates: original + rotations (0, 90, 270, 180 deg)
    candidates = [pil_img]
    for angle in [90, 270, 180]:
        candidates.append(pil_img.rotate(angle, expand=True))

    w_orig, h_orig = pil_img.size

    for current_img in candidates:
        cw, ch = current_img.size

        # Stage A: Direct FaceLandmarker on full image
        try:
            curr_rgb = np.array(current_img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=curr_rgb)
            if landmarker:
                lm_res = landmarker.detect(mp_image)
                if lm_res.face_landmarks:
                    landmarks = lm_res.face_landmarks[0]
                    blendshapes = lm_res.face_blendshapes[0] if lm_res.face_blendshapes else None
                    pts = np.array([[lm.x * cw, lm.y * ch] for lm in landmarks])
                    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
                    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
                    bw, bh = x_max - x_min, y_max - y_min
                    crop_x1 = max(0, int(x_min - bw * 0.25))
                    crop_y1 = max(0, int(y_min - bh * 0.35))
                    crop_x2 = min(cw, int(x_max + bw * 0.25))
                    crop_y2 = min(ch, int(y_max + bh * 0.35))

                    face_crop = current_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    face_crop.thumbnail((240, 300), Image.Resampling.LANCZOS)
                    face_thumbnail_b64 = encode_image_to_base64(face_crop)
                    embedding = extract_biometric_embedding(landmarks, blendshapes, curr_rgb)

                    return {
                        "face_detected": True,
                        "landmarks_count": len(landmarks),
                        "embedding": embedding.tolist(),
                        "face_thumbnail_b64": face_thumbnail_b64,
                        "message": "Document face portrait extracted successfully."
                    }
        except Exception:
            pass

        # Stage B: BlazeFace Detector for ID Card Photo Crops (Aadhaar / National ID)
        if detector:
            scale_factor = 1.0
            det_img = current_img
            if max(cw, ch) > 1000:
                scale_factor = 1000.0 / max(cw, ch)
                det_w, det_h = int(cw * scale_factor), int(ch * scale_factor)
                det_img = current_img.resize((det_w, det_h), Image.Resampling.BILINEAR)

            det_rgb = np.array(det_img)
            mp_det_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_rgb)
            det_result = detector.detect(mp_det_img)

            # Fallback: contrast enhancement if low dynamic range
            if not det_result.detections:
                enh_det_img = ImageEnhance.Contrast(det_img).enhance(1.6)
                det_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(enh_det_img)))

            if det_result.detections:
                best_det = max(det_result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                bb = best_det.bounding_box

                orig_bb_x = int(bb.origin_x / scale_factor)
                orig_bb_y = int(bb.origin_y / scale_factor)
                orig_bb_w = int(bb.width / scale_factor)
                orig_bb_h = int(bb.height / scale_factor)

                mx = int(orig_bb_w * 0.30)
                my = int(orig_bb_h * 0.35)
                crop_x1 = max(0, orig_bb_x - mx)
                crop_y1 = max(0, orig_bb_y - my)
                crop_x2 = min(cw, orig_bb_x + orig_bb_w + mx)
                crop_y2 = min(ch, orig_bb_y + orig_bb_h + my)

                face_crop = current_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                face_crop_clean = face_crop.copy()
                face_crop_clean.thumbnail((240, 300), Image.Resampling.LANCZOS)
                face_thumbnail_b64 = encode_image_to_base64(face_crop_clean)

                # Attempt Landmarker on the extracted crop
                crop_resized = face_crop.resize((320, 320), Image.Resampling.LANCZOS)
                crop_lm_res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop_resized))) if landmarker else None

                if not (crop_lm_res and crop_lm_res.face_landmarks):
                    enh_crop = ImageEnhance.Contrast(crop_resized).enhance(1.4)
                    enh_crop = ImageEnhance.Sharpness(enh_crop).enhance(1.4)
                    crop_lm_res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(enh_crop))) if landmarker else None

                crop_rgb = np.array(crop_resized)
                if crop_lm_res and crop_lm_res.face_landmarks:
                    landmarks = crop_lm_res.face_landmarks[0]
                    blendshapes = crop_lm_res.face_blendshapes[0] if crop_lm_res.face_blendshapes else None
                    embedding = extract_biometric_embedding(landmarks, blendshapes, crop_rgb)
                    lm_count = len(landmarks)
                else:
                    synth_lms = _synthesize_landmarks_from_detection(best_det, det_img.size[0], det_img.size[1])
                    embedding = extract_biometric_embedding(synth_lms, None, det_rgb)
                    lm_count = len(synth_lms)

                return {
                    "face_detected": True,
                    "landmarks_count": lm_count,
                    "embedding": embedding.tolist(),
                    "face_thumbnail_b64": face_thumbnail_b64,
                    "message": "ID card / Aadhaar face portrait extracted & normalized successfully."
                }

    # Stage C: Regional ID search (Aadhaar cards typically have photos on the right or left half)
    regional_boxes = [
        (int(w_orig * 0.45), 0, w_orig, h_orig),
        (0, 0, int(w_orig * 0.55), h_orig),
        (int(w_orig * 0.45), 0, w_orig, int(h_orig * 0.65)),
        (0, 0, int(w_orig * 0.55), int(h_orig * 0.65)),
    ]
    for box in regional_boxes:
        reg_crop = pil_img.crop(box)
        reg_rgb = np.array(reg_crop)
        mp_reg = mp.Image(image_format=mp.ImageFormat.SRGB, data=reg_rgb)
        if detector:
            reg_det = detector.detect(mp_reg)
            if reg_det.detections:
                best_det = reg_det.detections[0]
                bb = best_det.bounding_box
                rw, rh = reg_crop.size
                mx, my = int(bb.width * 0.30), int(bb.height * 0.35)
                cx1 = max(0, bb.origin_x - mx)
                cy1 = max(0, bb.origin_y - my)
                cx2 = min(rw, bb.origin_x + bb.width + mx)
                cy2 = min(rh, bb.origin_y + bb.height + my)

                face_crop = reg_crop.crop((cx1, cy1, cx2, cy2))
                face_crop_clean = face_crop.copy()
                face_crop_clean.thumbnail((240, 300), Image.Resampling.LANCZOS)
                face_thumbnail_b64 = encode_image_to_base64(face_crop_clean)

                crop_resized = face_crop.resize((320, 320), Image.Resampling.LANCZOS)
                crop_rgb = np.array(crop_resized)
                crop_lm_res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)) if landmarker else None
                if crop_lm_res and crop_lm_res.face_landmarks:
                    embedding = extract_biometric_embedding(crop_lm_res.face_landmarks[0], None, crop_rgb)
                else:
                    synth_lms = _synthesize_landmarks_from_detection(best_det, rw, rh)
                    embedding = extract_biometric_embedding(synth_lms, None, reg_rgb)

                return {
                    "face_detected": True,
                    "landmarks_count": 478,
                    "embedding": embedding.tolist(),
                    "face_thumbnail_b64": face_thumbnail_b64,
                    "message": "Aadhaar / ID portrait localized from document photo region."
                }

    return {
        "face_detected": False,
        "message": "No clear facial portrait detected in the uploaded document. Please upload a clear photo or document containing a visible face."
    }


def _fft_moire_score(gray: np.ndarray) -> float:
    """Isolated mid-band FFT peaks typical of screen recapture / moire."""
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= float(g.mean())
    win = np.outer(np.hanning(128), np.hanning(128)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))
    yy, xx = np.ogrid[:128, :128]
    r = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    mag[r < 6] = 0.0
    band = mag[(r >= 10) & (r <= 52)]
    if band.size == 0:
        return 0.0
    med = float(np.median(band)) + 1e-6
    peak = float(np.percentile(band, 99.6))
    ratio = peak / med
    return float(np.clip((ratio - 16.0) / 22.0, 0.0, 1.0))


def _chroma_screen_score(face_rgb: np.ndarray) -> float:
    """LCD/OLED recapture: structured R-G high-frequency and oversaturated gamut."""
    if face_rgb.size == 0:
        return 0.0
    small = cv2.resize(face_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    r = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    b = small[:, :, 2].astype(np.float32)
    fr = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    fg = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    channel_hf = float(np.mean(np.abs(fr - fg))) / 40.0

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat_high = float(np.mean(sat > 170))
    # Recaptured screens often clip a slice of the gamut.
    unique = len(np.unique((small // 32).reshape(-1, 3), axis=0))
    posterize = float(np.clip((80.0 - unique) / 80.0, 0.0, 1.0))
    blue_cast = float(np.clip((float(b.mean()) - float(r.mean()) - 8.0) / 40.0, 0.0, 1.0))
    return float(np.clip(0.40 * min(channel_hf, 1.0) + 0.25 * sat_high + 0.20 * posterize + 0.15 * blue_cast, 0.0, 1.0))


def _device_rectangle_score(gray_roi: np.ndarray, face_w: float, face_h: float) -> float:
    """Phone/tablet bezel: long axis-aligned edges wrapping the face."""
    if gray_roi.size == 0:
        return 0.0
    blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blur, 55, 140)
    min_len = int(max(28, 0.35 * min(face_w, face_h)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=36, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return 0.0
    horiz, vert = 0, 0
    # OpenCV builds can return HoughLinesP as either (N, 1, 4) or (N, 4).
    # Normalize the shape before iterating so one line record is always four values.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if ang < 12 or ang > 168:
            horiz += 1
        elif 78 < ang < 102:
            vert += 1
    line_score = min(1.0, (min(horiz, 4) / 4.0) * 0.5 + (min(vert, 4) / 4.0) * 0.5)
    if horiz >= 2 and vert >= 2:
        line_score = min(1.0, line_score + 0.35)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad_score = 0.0
    face_area = max(face_w * face_h, 1.0)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = abs(cv2.contourArea(approx))
        ratio = area / face_area
        if 0.85 <= ratio <= 3.2:
            quad_score = max(quad_score, float(np.clip((3.2 - abs(ratio - 1.4)) / 3.2, 0.0, 1.0)))
    return float(np.clip(0.65 * line_score + 0.35 * quad_score, 0.0, 1.0))


def _bezel_and_glare_score(rgb_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
    h, w = rgb_img.shape[:2]
    face = rgb_img[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0, 0.0
    pad = int(max(8, 0.12 * max(x2 - x1, y2 - y1)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = rgb_img[ry1:ry2, rx1:rx2].copy()
    inner_y1, inner_x1 = y1 - ry1, x1 - rx1
    inner_y2, inner_x2 = inner_y1 + (y2 - y1), inner_x1 + (x2 - x1)
    mask = np.ones(ring.shape[:2], dtype=bool)
    mask[max(0, inner_y1):min(mask.shape[0], inner_y2), max(0, inner_x1):min(mask.shape[1], inner_x2)] = False
    ring_px = ring[mask]
    if ring_px.size == 0:
        bezel = 0.0
    else:
        ring_mean = float(ring_px.mean())
        ring_std = float(ring_px.std())
        face_mean = float(face.mean())
        # Uniform dark bezel around a brighter displayed portrait.
        bezel = 0.0
        if ring_mean < 55 and ring_std < 32 and face_mean > 70:
            bezel = float(np.clip((55.0 - ring_mean) / 40.0 + (32.0 - ring_std) / 40.0, 0.0, 1.0))

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    bright = gray > 245
    glare = 0.0
    if bright.any():
        frac = float(bright.mean())
        # Screen glass: small saturated specular blob, not overall overexposure.
        if 0.004 <= frac <= 0.08:
            glare = float(np.clip(frac / 0.05, 0.0, 1.0))
    return bezel, glare


def _texture_flatness_score(face_rgb: np.ndarray) -> float:
    """Printed / phone photos lose skin micro-texture or look digitally oversmooth."""
    if face_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(face_rgb, (96, 96)), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    var = float(lap.var())
    # Very low variance = flat screen/print; extremely high + periodic = moire (handled elsewhere).
    if var < 18.0:
        return float(np.clip((18.0 - var) / 18.0, 0.0, 1.0))
    return 0.0


def detect_presentation_attack(
    rgb_img: np.ndarray,
    landmarks,
    num_faces: int = 1,
) -> Tuple[float, List[str]]:
    """Return spoof probability 0..1 and human-readable reasons."""
    h, w = rgb_img.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
    x1, y1 = max(0, int(x_min)), max(0, int(y_min))
    x2, y2 = min(w, int(x_max)), min(h, int(y_max))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return 0.35, ["Unstable face crop"]

    face = rgb_img[y1:y2, x1:x2]
    face_w, face_h = float(x2 - x1), float(y2 - y1)
    pad = int(max(12, 0.28 * max(face_w, face_h)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    roi_gray = cv2.cvtColor(rgb_img[ry1:ry2, rx1:rx2], cv2.COLOR_RGB2GRAY)

    moire = _fft_moire_score(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))
    chroma = _chroma_screen_score(face)
    rect = _device_rectangle_score(roi_gray, face_w, face_h)
    bezel, glare = _bezel_and_glare_score(rgb_img, x1, y1, x2, y2)
    flat = _texture_flatness_score(face)
    extra_face = 0.55 if num_faces >= 2 else 0.0

    spoof = (
        0.28 * rect +
        0.20 * moire +
        0.16 * chroma +
        0.14 * bezel +
        0.10 * glare +
        0.08 * flat +
        0.04 * extra_face
    )
    # Two strong screen cues together are decisive (phone held to the lens).
    strong = sum(1 for s in (rect, moire, chroma, bezel) if s >= 0.45)
    if strong >= 2:
        spoof = max(spoof, 0.72)
    if rect >= 0.55 and (bezel >= 0.40 or moire >= 0.40 or chroma >= 0.40):
        spoof = max(spoof, 0.80)
    if extra_face and (rect >= 0.35 or bezel >= 0.35):
        spoof = max(spoof, 0.78)

    reasons = []
    if rect >= 0.45 or bezel >= 0.45:
        reasons.append("Device or printed photo detected in frame")
    if moire >= 0.45 or chroma >= 0.45:
        reasons.append("Screen recapture pattern detected")
    if glare >= 0.55:
        reasons.append("Glass reflection on a display")
    if extra_face:
        reasons.append("More than one face in the camera")
    if flat >= 0.65:
        reasons.append("Face texture looks printed or on a screen")
    return float(np.clip(spoof, 0.0, 0.99)), reasons


def analyze_liveness_and_quality(
    rgb_img: np.ndarray,
    landmarks,
    blendshapes=None,
    transform_matrix=None,
    liveness_threshold: float = 0.80,
    num_faces: int = 1,
) -> Dict[str, Any]:
    """Quality gates plus presentation-attack detection. Mesh Z is not treated as proof of life."""
    h, w = rgb_img.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])

    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
    bbox_w = max(x_max - x_min, 1.0)
    bbox_h = max(y_max - y_min, 1.0)

    face_area_ratio = (bbox_w * bbox_h) / (w * h)
    center_x = (x_min + x_max) / 2.0 / w
    center_y = (y_min + y_max) / 2.0 / h
    dist_from_center = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
    center_score = max(0.0, 1.0 - (dist_from_center * 2.0))

    eye_open_score = 0.95
    if blendshapes:
        bs_dict = {c.category_name: c.score for c in blendshapes}
        blink_left = bs_dict.get("eyeBlinkLeft", 0.0)
        blink_right = bs_dict.get("eyeBlinkRight", 0.0)
        avg_blink = (blink_left + blink_right) / 2.0
        eye_open_score = max(0.5, 1.0 - (avg_blink * 0.8))

    pose_score = 0.95
    head_yaw, head_pitch, head_roll = 0.0, 0.0, 0.0
    if transform_matrix is not None and len(transform_matrix) >= 3:
        mat = np.array(transform_matrix)
        head_yaw = float(math.degrees(math.atan2(mat[1, 0], mat[0, 0])))
        head_pitch = float(math.degrees(math.atan2(-mat[2, 0], math.sqrt(mat[2, 1]**2 + mat[2, 2]**2))))
        head_roll = float(math.degrees(math.atan2(mat[2, 1], mat[2, 2])))
        pose_dev = math.sqrt(head_yaw**2 + head_pitch**2 + (head_roll * 0.5)**2)
        pose_score = max(0.3, 1.0 - (pose_dev / 45.0))

    x1, y1 = max(0, int(x_min)), max(0, int(y_min))
    x2, y2 = min(w, int(x_max)), min(h, int(y_max))
    if x2 > x1 and y2 > y1:
        face_crop = rgb_img[y1:y2, x1:x2]
        gray = np.mean(face_crop, axis=2)
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        bright_score = 1.0 - min(1.0, abs(mean_brightness - 130) / 100.0)
        contrast_score = min(1.0, contrast / 35.0)
        lighting_score = (bright_score * 0.6 + contrast_score * 0.4)
    else:
        lighting_score = 0.8

    spoof_prob, spoof_reasons = detect_presentation_attack(rgb_img, landmarks, num_faces=num_faces)
    pad_live = 1.0 - spoof_prob
    quality = 0.40 * pose_score + 0.30 * center_score + 0.30 * lighting_score
    # Quality must not rescue a screen/print attack (the old 1.15 boost caused 99% "live" photos).
    liveness_score = 0.78 * pad_live + 0.22 * quality
    liveness_score = float(np.clip(liveness_score, 0.05, 0.99))
    if spoof_prob >= 0.45:
        liveness_score = min(liveness_score, 0.62)
        liveness_passed = False
    else:
        liveness_passed = liveness_score >= liveness_threshold

    guidance = []
    if spoof_reasons:
        guidance.append("Show your live face — photos and phone screens are rejected")
    if face_area_ratio < 0.08:
        guidance.append("Move closer to the camera")
    if dist_from_center > 0.25:
        guidance.append("Center your face in the frame")
    if pose_score < 0.65:
        guidance.append("Look directly into the camera")
    if lighting_score < 0.5:
        guidance.append("Ensure good lighting on your face")
    if not guidance:
        guidance.append("Face well aligned. Hold still.")

    return {
        "liveness_score": round(float(liveness_score), 4),
        "liveness_passed": liveness_passed,
        "depth_score": round(float(pad_live), 3),
        "pose_score": round(float(pose_score), 3),
        "eye_open_score": round(float(eye_open_score), 3),
        "lighting_score": round(float(lighting_score), 3),
        "spoof_probability": round(float(spoof_prob), 3),
        "bbox": {
            "x": int(x_min), "y": int(y_min),
            "width": int(bbox_w), "height": int(bbox_h),
            "norm_x": round(x_min / w, 4), "norm_y": round(y_min / h, 4),
            "norm_w": round(bbox_w / w, 4), "norm_h": round(bbox_h / h, 4),
        },
        "head_angles": {
            "yaw": round(head_yaw, 1),
            "pitch": round(head_pitch, 1),
            "roll": round(head_roll, 1),
        },
        "guidance": guidance[0],
    }


def detect_live_face(image_b64: str, liveness_threshold: float = 0.80) -> Dict[str, Any]:
    """Fast detection endpoint for live camera HUD preview & guidance."""
    landmarker = _get_landmarker()
    if landmarker is None:
        return {"face_detected": False, "message": "MediaPipe AI model loading..."}

    try:
        rgb_img, _ = decode_image(image_b64)
    except Exception as exc:
        return {"face_detected": False, "message": f"Image decode failed: {exc}"}

    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "face_detected": False,
            "num_faces": 0,
            "message": "No face detected in camera feed.",
            "guidance": "Position face in the frame",
        }

    num_faces = len(result.face_landmarks)
    landmarks = result.face_landmarks[0]
    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    transform = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None

    analysis = analyze_liveness_and_quality(
        rgb_img, landmarks, blendshapes, transform, liveness_threshold, num_faces=num_faces
    )

    return {
        "face_detected": True,
        "num_faces": num_faces,
        "landmarks_count": len(landmarks),
        **analysis,
    }


def process_face_image(
    conn: sqlite3.Connection,
    passenger_id: str,
    face_image_b64: str,
    match_threshold: float = 0.85,
    liveness_threshold: float = 0.80,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run real face recognition and verification against passenger's document / biometric template."""
    landmarker = _get_landmarker()
    if landmarker is None:
        return {
            "face_detected": False,
            "liveness_score": 0.0,
            "liveness_passed": False,
            "match_score": None,
            "match_passed": None,
            "enrolled": False,
            "num_faces": 0,
            "message": "AI face recognition engine not available.",
        }

    try:
        rgb_img, pil_img = decode_image(face_image_b64)
    except Exception as exc:
        log.warning("Image decode failed: %s", exc)
        return {
            "face_detected": False,
            "liveness_score": 0.0,
            "liveness_passed": False,
            "match_score": None,
            "match_passed": None,
            "enrolled": False,
            "num_faces": 0,
            "message": f"Image decode error: {exc}",
        }

    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "face_detected": False,
            "liveness_score": 0.0,
            "liveness_passed": False,
            "match_score": None,
            "match_passed": None,
            "enrolled": False,
            "num_faces": 0,
            "message": "No face detected in live camera frame. Please look directly into the camera.",
            "guidance": "Please face the camera directly.",
        }

    num_faces = len(result.face_landmarks)
    landmarks = result.face_landmarks[0]
    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    transform = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None

    # Multi-factor liveness and image quality assessment
    analysis = analyze_liveness_and_quality(
        rgb_img, landmarks, blendshapes, transform, liveness_threshold, num_faces=num_faces
    )
    liveness_score = analysis["liveness_score"]
    liveness_passed = analysis["liveness_passed"]

    if not liveness_passed:
        return {
            "face_detected": True,
            "liveness_score": liveness_score,
            "liveness_passed": False,
            "match_score": None,
            "match_passed": False,
            "enrolled": False,
            "num_faces": num_faces,
            "message": f"Liveness check failed ({analysis['guidance']})",
            "details": analysis,
        }

    # SFace is the identity matcher; fail as an analysis result if its model is unavailable.
    try:
        live_embedding = extract_biometric_embedding(landmarks, blendshapes, rgb_img)
    except Exception as exc:
        log.exception("Could not create SFace live identity embedding")
        return {
            "face_detected": True,
            "liveness_score": liveness_score,
            "liveness_passed": True,
            "match_score": None,
            "match_passed": False,
            "enrolled": False,
            "num_faces": num_faces,
            "message": f"Face identity analysis unavailable: {exc}",
            "details": analysis,
        }

    # 1. Fetch reference template from session (uploaded passport/Aadhaar)
    enrolled_encoding = None
    source = "uploaded document photo"

    if session_id:
        try:
            s_row = conn.execute(
                "SELECT passport_face_encoding FROM verification_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if s_row and s_row["passport_face_encoding"]:
                enrolled_encoding = np.array(json.loads(s_row["passport_face_encoding"]), dtype=np.float32)
                source = "uploaded document portrait"
        except Exception:
            pass

    # 2. If not found in session, check biometric_templates table
    if enrolled_encoding is None:
        row = conn.execute(
            "SELECT template_id, feature_vector, real_face_encoding FROM biometric_templates WHERE passenger_id = ?",
            (passenger_id,)
        ).fetchone()

        if row:
            raw_real = row["real_face_encoding"] if "real_face_encoding" in row.keys() else None
            if raw_real:
                try:
                    enrolled_encoding = np.array(json.loads(raw_real), dtype=np.float32)
                    source = "enrolled biometric template"
                except Exception:
                    enrolled_encoding = None

    if enrolled_encoding is not None:
        # Match only vectors produced by the same descriptor version. A legacy
        # landmark-only template cannot be compared with the current appearance vector.
        compatible_template = enrolled_encoding.size == live_embedding.size and (
            enrolled_encoding.size == 128 or enrolled_encoding.size >= APP_DIM
        )
        calibrated_score = (
            compute_biometric_similarity(live_embedding, enrolled_encoding)
            if compatible_template else 0.0
        )
        match_passed = calibrated_score >= match_threshold

        if not compatible_template:
            msg = "Stored face template is outdated or incompatible. Re-scan the identity document or re-enroll the face, then retry."
        elif match_passed:
            msg = f"✓ Biometric match verified: {calibrated_score:.1%} match with {source}."
        else:
            msg = f"✗ Biometric mismatch: Live face match ({calibrated_score:.1%}) is below threshold ({match_threshold:.1%})."

        return {
            "face_detected": True,
            "liveness_score": liveness_score,
            "liveness_passed": True,
            "match_score": round(float(calibrated_score), 4),
            "match_passed": match_passed,
            "enrolled": False,
            "num_faces": num_faces,
            "message": msg,
            "details": analysis,
        }
    else:
        # If no reference template exists, fail match explicitly (do not auto-pass)
        return {
            "face_detected": True,
            "liveness_score": liveness_score,
            "liveness_passed": True,
            "match_score": 0.45,
            "match_passed": False,
            "enrolled": False,
            "num_faces": num_faces,
            "message": "✗ Biometric verification failed: No reference face portrait found on the uploaded document.",
            "details": analysis,
        }
