"""
HandsToVoice — Hand Landmark Detector
Uses MediaPipe Hand Landmarker (Tasks API) for real-time detection.
Falls back to motion-based detection if unavailable.
"""

import os
import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger("detector")


class HandDetector:
    NUM_LANDMARKS = 21
    FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63 features

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self._mediapipe_available = False
        self.frame_count = 0
        self._timestamp_ms = 0

        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        model_path = os.path.join(models_dir, "hand_landmarker.task")

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            self._mp = mp
            self._mp_vision = mp_vision

            base_options = mp_python.BaseOptions(
                model_asset_path=model_path
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._mediapipe_available = True
            logger.info("[Detector] MediaPipe Hand Landmarker initialized successfully")
        except Exception as e:
            logger.info(f"[Detector] MediaPipe unavailable ({e}). Using fallback.")
            self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Face detector — gives a stable body-relative reference point (see
        # src/normalize.py) so signs that differ only by WHERE on the body
        # the hand touches aren't indistinguishable to the model. Best
        # effort: if unavailable, face_ref just comes back None everywhere
        # and normalization falls back to hand-shape-only as before.
        self._face_available = False
        self._face_timestamp_ms = 0
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            face_model_path = os.path.join(models_dir, "face_detector.tflite")
            face_base_options = mp_python.BaseOptions(model_asset_path=face_model_path)
            face_options = mp_vision.FaceDetectorOptions(
                base_options=face_base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(face_options)
            self._face_available = True
            logger.info("[Detector] MediaPipe Face Detector initialized successfully")
        except Exception as e:
            logger.info(f"[Detector] Face detector unavailable ({e}) — "
                        f"signs will be distinguished by hand shape only.")

    def process_frame(self, bgr_frame):
        """Return (annotated_frame, landmarks_list).
        Each entry is a float32 array of shape (63,).
        """
        self.frame_count += 1
        if self._mediapipe_available:
            return self._mediapipe(bgr_frame)
        return self._fallback(bgr_frame)

    # ── MediaPipe Tasks path ─────────────────────────────────────────────────
    def _mediapipe(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )

        self._timestamp_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        annotated = bgr_frame.copy()
        landmarks_list = []
        h, w = bgr_frame.shape[:2]

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # Draw landmarks and connections on the frame
                points = []
                flat = []
                for lm in hand_lms:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))
                    flat.extend([lm.x, lm.y, lm.z])
                    cv2.circle(annotated, (px, py), 4, (0, 212, 170), -1)

                # Draw connections (simplified skeleton)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),        # thumb
                    (0,5),(5,6),(6,7),(7,8),        # index
                    (5,9),(9,10),(10,11),(11,12),   # middle
                    (9,13),(13,14),(14,15),(15,16),  # ring
                    (13,17),(17,18),(18,19),(19,20), # pinky
                    (0,17),
                ]
                for s, e in connections:
                    if s < len(points) and e < len(points):
                        cv2.line(annotated, points[s], points[e],
                                 (0, 170, 136), 2)

                landmarks_list.append(
                    np.array(flat, dtype=np.float32)
                )

        return annotated, landmarks_list

    # ── Fallback path ─────────────────────────────────────────────────────────
    def _fallback(self, bgr_frame):
        fg = self._bg.apply(bgr_frame)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        annotated = bgr_frame.copy()
        landmarks_list = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5000 < area < 50000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                if self.frame_count % 10 == 0:
                    lm = self._mock_landmarks(x, y, w, h, bgr_frame.shape)
                    landmarks_list.append(lm)
                if len(landmarks_list) >= self.max_num_hands:
                    break

        return annotated, landmarks_list

    def _mock_landmarks(self, bx, by, bw, bh, shape):
        fh, fw = shape[:2]
        wx = (bx + bw / 2) / fw
        wy = (by + bh * 0.8) / fh
        lms = [wx, wy, 0.0]
        for i in range(1, 21):
            angle = (i - 1) * (2 * np.pi / 20)
            r = 0.05 + (i % 5) * 0.02
            x = float(np.clip(wx + r * np.cos(angle), 0, 1))
            y = float(np.clip(wy - r * np.sin(angle) * 2, 0, 1))
            lms.extend([x, y, float(np.random.uniform(-0.1, 0.1))])
        return np.array(lms, dtype=np.float32)

    def detect_face_ref(self, bgr_frame):
        """Return (cx, cy, width) of the detected face in normalized [0, 1]
        image coordinates, or None if unavailable/no face found.

        Used to compute the hand's position relative to the body (see
        src/normalize.py) — a separate, lightweight call from
        process_frame() since most callers (raw data collection, the
        static classifier) don't need it.
        """
        if not self._face_available:
            return None
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._face_timestamp_ms += 33
            result = self._face_detector.detect_for_video(mp_image, self._face_timestamp_ms)
            if not result.detections:
                return None

            h, w = bgr_frame.shape[:2]
            box = result.detections[0].bounding_box
            cx = (box.origin_x + box.width / 2) / w
            cy = (box.origin_y + box.height / 2) / h
            face_width = box.width / w
            return (cx, cy, face_width)
        except Exception as e:
            logger.error(f"[Detector] Face detection error: {e}")
            return None

    def close(self):
        if self._mediapipe_available and hasattr(self, '_landmarker'):
            self._landmarker.close()
        if self._face_available and hasattr(self, '_face_detector'):
            self._face_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
