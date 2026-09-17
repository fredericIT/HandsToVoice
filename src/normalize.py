"""
HandsToVoice — Landmark Normalization
Centers hand landmarks relative to the wrist and normalizes scale.

This is CRITICAL for generalization: without normalization the model
learns absolute hand position in the frame rather than hand shape/motion.
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63


def normalize_landmarks(flat: np.ndarray) -> np.ndarray:
    """Normalize a (63,) landmark vector to be position- and scale-invariant.

    1. Reshape to (21, 3)
    2. Subtract the wrist (landmark 0) → centers the hand at the origin
    3. Divide by the hand span (wrist → middle-finger MCP, landmark 9)
       → normalizes for distance from camera

    Args:
        flat: numpy array of shape (63,) with raw x, y, z landmarks.

    Returns:
        numpy array of shape (63,) with normalized landmarks.
    """
    pts = np.array(flat, dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    # If all zeros (no detection), return as-is
    if not pts.any():
        return flat.astype(np.float32)

    # Center on wrist (landmark 0)
    wrist = pts[0].copy()
    pts = pts - wrist

    # Scale by hand span: distance from wrist to middle-finger MCP (landmark 9)
    span = np.linalg.norm(pts[9])
    if span > 1e-6:
        pts = pts / span

    return pts.reshape(FEATURE_LENGTH).astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize each frame in a (T, 63) sequence independently.

    Args:
        seq: numpy array of shape (T, 63).

    Returns:
        numpy array of shape (T, 63) with each frame normalized.
    """
    out = np.zeros_like(seq)
    for i in range(len(seq)):
        out[i] = normalize_landmarks(seq[i])
    return out
