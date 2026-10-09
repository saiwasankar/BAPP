import json
import base64
import io
import numpy as np
from PIL import Image, ImageDraw

from app import database, services, real_face

def generate_test_face(skin_color, eye_color=(20, 20, 20), eye_spread=0, jaw_width=0, nose_y=0, mouth_offset=0):
    """Generate synthetic realistic test face image for biometric validation."""
    img = Image.new("RGB", (320, 320), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Head outline (jaw width)
    draw.ellipse([80 - jaw_width, 50, 240 + jaw_width, 250], fill=skin_color, outline=(60, 40, 30), width=2)
    # Eyes (eye spread)
    draw.ellipse([115 - eye_spread, 120, 145 - eye_spread, 140], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([175 + eye_spread, 120, 205 + eye_spread, 140], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.ellipse([125 - eye_spread, 125, 137 - eye_spread, 137], fill=eye_color)
    draw.ellipse([183 + eye_spread, 125, 195 + eye_spread, 137], fill=eye_color)
    # Nose (nose y)
    draw.line([160, 135 + nose_y, 155, 175 + nose_y], fill=(120, 80, 60), width=2)
    draw.line([155, 175 + nose_y, 168, 175 + nose_y], fill=(120, 80, 60), width=2)
    # Mouth
    draw.arc([130, 190 + mouth_offset, 190, 220 + mouth_offset], start=10, end=170, fill=(180, 40, 40), width=3)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    database.reset_database()
    conn = database.connect()
    print("Database reset.")

    # Face 1 (e.g. Passenger / Aadhaar holder)
    face_user_a = generate_test_face(skin_color=(220, 180, 145), eye_color=(30, 20, 10))
    # Face 2 (Different person / impostor with structurally different face geometry)
    face_impostor_b = generate_test_face(
        skin_color=(160, 120, 90), 
        eye_color=(10, 40, 80), 
        eye_spread=15, 
        jaw_width=20, 
        nose_y=10, 
        mouth_offset=15
    )

    print("\n1. Testing document face extraction on Face A...")
    doc_res = real_face.extract_face_from_document(face_user_a)
    print(f"Extraction result: face_detected={doc_res.get('face_detected')}, landmarks={doc_res.get('landmarks_count')}")
    assert doc_res.get("face_detected"), "Face A extraction failed"

    print("\n2. Starting session with Face A document photo...")
    session = services.start_session(
        conn,
        kiosk_id="KIOSK-01",
        passport_number="N4821907",
        passport_image_b64=face_user_a
    )
    session_id = session["session_id"]
    print(f"Session started: {session_id}, state: {session['state']}, passport_photo_extracted: {bool(session.get('passport_photo_b64'))}")
    assert session["state"] == "AWAITING_FACE", "Session should be AWAITING_FACE"

    print("\n3. Testing face verification with IMPOSTOR face (Face B)...")
    verif_impostor = services.verify_face(
        conn,
        session_id=session_id,
        image_base64=face_impostor_b
    )
    attempt_1 = verif_impostor["attempt"]
    sess_after_impostor = verif_impostor["session"]
    print(f"Attempt 1 (Impostor): match_passed={attempt_1['match_passed']}, match_score={attempt_1['match_score']}, outcome={attempt_1['outcome']}")
    print(f"Session state after impostor: {sess_after_impostor['state']}")
    assert attempt_1["match_passed"] is False, "Impostor should NOT match Face A"
    assert sess_after_impostor["state"] == "AWAITING_FACE", "Session should remain in AWAITING_FACE (retry allowed)"

    print("\n4. Testing face verification with MATCHING face (Face A)...")
    verif_match = services.verify_face(
        conn,
        session_id=session_id,
        image_base64=face_user_a
    )
    attempt_2 = verif_match["attempt"]
    sess_after_match = verif_match["session"]
    print(f"Attempt 2 (Real Face A): match_passed={attempt_2['match_passed']}, match_score={attempt_2['match_score']}, outcome={attempt_2['outcome']}")
    print(f"Session state after match: {sess_after_match['state']}")
    assert attempt_2["match_passed"] is True, "Face A should match uploaded Face A"
    assert sess_after_match["state"] == "FACE_VERIFIED", "Session state should now be FACE_VERIFIED"

    print("\n5. Proceeding to immigration checks...")
    chk_res = services.run_immigration_checks(conn, session_id=session_id)
    print(f"Checks outcome: {chk_res.get('final_decision')}, reason: {chk_res.get('rejection_reason')}")
    print(f"Checks completed: state={chk_res['session']['state']}")
    assert chk_res['session']['state'] == "CHECKS_COMPLETE"

    print("\n6. Requesting gate decision...")
    dec_res = services.decide_access(conn, session_id=session_id)
    print(f"Decision: {dec_res['session']['decision']}, state={dec_res['session']['state']}")
    assert dec_res['session']['state'] == "GATE_OPEN"

    print("\nALL FACE VERIFICATION AND MISMATCH TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
