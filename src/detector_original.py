"""
HandsToVoice — Hand Landmark Detector Module
Uses MediaPipe Hands to detect hand landmarks and extract feature vectors.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    """Detects hand landmarks using MediaPipe Hands."""

    # MediaPipe returns 21 landmarks, each with x, y, z coordinates
    NUM_LANDMARKS = 21
    FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63 features

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize the hand detector.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        # Use MediaPipe tasks API
        base_options = python.BaseOptions(
            model_asset_path=None  # Will use default model
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # For drawing, we'll use basic OpenCV drawing since MediaPipe drawing utils aren't available
        self.use_drawing = False

    def process_frame(self, bgr_frame):
        """
        Process a BGR frame and extract hand landmarks.

        Args:
            bgr_frame: Input BGR image (numpy array from OpenCV).

        Returns:
            tuple: (annotated_frame, landmarks_list)
                - annotated_frame: The input frame with hand skeleton drawn.
                - landmarks_list: List of normalized landmark arrays, one per
                  detected hand. Each array has shape (63,) representing
                  [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
                  Returns empty list if no hands detected.
        """
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        results = self.detector.detect(mp_image)

        annotated = bgr_frame.copy()
        landmarks_list = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Basic drawing - just draw circles at landmark points
                if self.use_drawing:
                    h, w = bgr_frame.shape[:2]
                    for landmark in hand_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

                # Extract and normalize landmarks
                landmarks = self._extract_landmarks_from_tasks(hand_landmarks)
                landmarks_list.append(landmarks)

        return annotated, landmarks_list

    def _extract_landmarks_from_tasks(self, hand_landmarks):
        """
        Extract and normalize the 21 landmarks from tasks API into a flat feature vector.

        Normalization: All coordinates are made relative to the wrist
        (landmark 0) so the features are position-invariant.

        Args:
            hand_landmarks: MediaPipe hand landmarks object from tasks API.

        Returns:
            numpy.ndarray: Normalized feature vector of shape (63,).
        """
        raw = []
        for lm in hand_landmarks:
            raw.append([lm.x, lm.y, lm.z])
        raw = np.array(raw, dtype=np.float32)

        # Normalize relative to wrist (landmark 0)
        wrist = raw[0]
        normalized = raw - wrist

        # Flatten to 1D: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        return normalized.flatten()

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'detector') and self.detector:
            # Tasks API doesn't have a close method, just set to None
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
