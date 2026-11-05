import math

# Landmark indices (MediaPipe BlazePose, 33 points)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

HEAD_OFFSET_CM = 12.0
ANKLE_OFFSET_CM = 8.0
MIN_VISIBILITY = 0.5


def _dist_cm(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) * 100.0


def _mid(a, b):
    return [(a[i] + b[i]) / 2.0 for i in range(3)]


def _ellipse_circumference(semi_a, semi_b):
    return math.pi * (3 * (semi_a + semi_b) - math.sqrt((3 * semi_a + semi_b) * (semi_a + 3 * semi_b)))


def estimate_measurements(world_landmarks, landmarks_2d=None):
    if len(world_landmarks) < 33:
        raise ValueError("Expected 33 BlazePose landmarks")

    notes = []
    confidence = "high"

    if landmarks_2d is not None:
        key_ids = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE]
        vis = [landmarks_2d[i][2] if len(landmarks_2d[i]) > 2 else 1.0 for i in key_ids]
        if any(v < MIN_VISIBILITY for v in vis):
            confidence = "medium"
            notes.append("Some body parts were partially visible — results are approximate.")

    lm = world_landmarks
    shoulder_w = _dist_cm(lm[L_SHOULDER], lm[R_SHOULDER])
    hip_w = _dist_cm(lm[L_HIP], lm[R_HIP])
    torso_len = _dist_cm(_mid(lm[L_SHOULDER], lm[R_SHOULDER]), _mid(lm[L_HIP], lm[R_HIP]))

    ankle_y = (lm[L_ANKLE][1] + lm[R_ANKLE][1]) / 2.0
    nose_y = lm[NOSE][1]
    height_raw = (ankle_y - nose_y) * 100.0
    height_cm = height_raw + HEAD_OFFSET_CM + ANKLE_OFFSET_CM

    if shoulder_w < 20 or hip_w < 15:
        confidence = "low"
        notes.append("Stand fully in frame, a few steps back, facing the camera.")
    if height_cm < 120 or height_cm > 220:
        confidence = "low"
        notes.append("Could not estimate height reliably — make sure your full body is visible.")
        height_cm = max(130.0, min(210.0, height_cm))

    chest = _ellipse_circumference(shoulder_w / 2.0, shoulder_w * 0.25)
    hip = _ellipse_circumference(hip_w / 2.0, hip_w * 0.35)
    waist = hip * 0.85

    notes.append("Measurements are AI estimates (±5 cm), not tailoring-grade.")

    return {
        "height_cm": round(height_cm, 1),
        "shoulder_width_cm": round(shoulder_w, 1),
        "torso_length_cm": round(torso_len, 1),
        "chest_cm": round(chest, 1),
        "waist_cm": round(waist, 1),
        "hip_cm": round(hip, 1),
        "confidence": confidence,
        "notes": notes,
    }
