"""
HandsToVoice — Lighting check
Dim light degrades hand landmark detection (lost frames, jitter, motion
blur), which is a real source of sign confusion. Recorded clips ranged from
24 to 149 brightness (0-255); the darkest were the least reliable.
"""

import cv2

LOW_LIGHT = 70   # mean grey level (0-255) below which detection gets unreliable
CHECK_EVERY = 10  # frames between brightness checks — cheap, but not needed per frame


def frame_brightness(frame):
    """Mean grey level (0-255) of a BGR frame, measured on a small copy."""
    small = cv2.resize(frame, (64, 48))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def draw_low_light_warning(frame):
    """Draw a warning banner onto the frame in place."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Too dark - face a light for accurate signs",
                (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 217, 255), 2)
