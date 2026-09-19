"""
HandsToVoice — Landmark Normalization
Centers hand landmarks relative to the wrist and normalizes scale, then
appends the hand's position relative to the face as two extra features.

Wrist-centering/scaling is CRITICAL for generalization: without it the
model learns absolute hand position in the frame rather than hand
shape/motion. But taken alone it also throws away WHERE on the body the
hand is — so signs that differ only by touching the chest vs. the chin,
for example, become indistinguishable. The face-relative offset restores
just enough of that information to disambiguate such signs, without
reintroducing camera-framing sensitivity (it's normalized by face width,
so it stays invariant to how close the person is to the camera).
"""

import numpy as np

NUM_LANDMARKS = 21
HAND_FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63
FEATURE_LENGTH = HAND_FEATURE_LENGTH + 2  # 65: +2 for face-relative (dx, dy)


def normalize_landmarks(flat: np.ndarray, face_ref=None) -> np.ndarray:
    """Normalize a (63,) landmark vector to be position- and scale-invariant,
    then append the wrist's position relative to the face (2 features).

    1. Reshape to (21, 3)
    2. Subtract the wrist (landmark 0) → centers the hand at the origin
    3. Divide by the hand span (wrist → middle-finger MCP, landmark 9)
       → normalizes for distance from camera
    4. Append (wrist_x - face_cx, wrist_y - face_cy) / face_width — the
       hand's position relative to the face, invariant to camera distance.
       Zero if no face was detected (face_ref is None), which reads as
       "unknown" rather than a false signal.

    Args:
        flat: numpy array of shape (63,) with raw x, y, z landmarks.
        face_ref: optional (cx, cy, width) of the detected face, in the
            same normalized [0, 1] image coordinates as the landmarks.

    Returns:
        numpy array of shape (65,): 63 normalized hand + 2 face-relative.
    """
    pts = np.array(flat, dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    # If all zeros (no detection), return as-is (padded to the new length)
    if not pts.any():
        return np.concatenate([flat.astype(np.float32), [0.0, 0.0]])

    # Center on wrist (landmark 0) — keep the raw wrist position for the
    # face-relative offset below, before it's zeroed out by centering.
    wrist = pts[0].copy()
    pts = pts - wrist

    # Scale by hand span: distance from wrist to middle-finger MCP (landmark 9)
    span = np.linalg.norm(pts[9])
    if span > 1e-6:
        pts = pts / span

    hand_features = pts.reshape(HAND_FEATURE_LENGTH).astype(np.float32)

    if face_ref is not None:
        cx, cy, face_width = face_ref
        if face_width > 1e-6:
            rel_x = (wrist[0] - cx) / face_width
            rel_y = (wrist[1] - cy) / face_width
        else:
            rel_x = rel_y = 0.0
    else:
        rel_x = rel_y = 0.0

    return np.concatenate([hand_features, [rel_x, rel_y]]).astype(np.float32)
