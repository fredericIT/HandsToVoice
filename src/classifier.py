"""
HandsToVoice — Sign Language Classifier Module
Wraps a trained TensorFlow/Keras model for KSL sign prediction.
Supports both static (Dense) and sequence (LSTM) models.
"""

import os
import numpy as np
from collections import deque
from src.normalize import normalize_landmarks

from src.logger import get_logger

logger = get_logger("classifier")


class SignClassifier:
    """Loads and runs inference with a trained Keras sign classification model."""

    def __init__(self, model_path="models/ksl_model.h5", confidence_threshold=0.6):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the saved Keras model file.
            confidence_threshold: Minimum confidence to consider a prediction valid.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels = []
        self.scaler = None
        self._load_model()

    def _load_model(self):
        """Load the Keras model from disk if it exists."""
        if not os.path.exists(self.model_path):
            logger.info(f"[Classifier] No model found at '{self.model_path}'. "
                  "Please train a model first using train_model.py.")
            self.model = None
            return

        try:
            # Import TensorFlow lazily to avoid slow startup when no model exists
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(self.model_path)

            # Load the label mapping
            label_path = self.model_path.replace('.h5', '_labels.npy')
            if os.path.exists(label_path):
                self.labels = list(np.load(label_path, allow_pickle=True))
            else:
                logger.warning(f"[Classifier] Warning: No label file found at '{label_path}'.")

            # Load the feature scaler the model was trained with (train_model.py /
            # train_simple.py fit a StandardScaler on the training landmarks
            # before training — without applying that same transform here,
            # the model receives raw out-of-distribution inputs at inference
            # and its predictions are meaningless).
            scaler_path = os.path.join(os.path.dirname(self.model_path) or ".", "scaler.pkl")
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
                logger.info(f"[Classifier] Loaded feature scaler from '{scaler_path}'.")
            else:
                logger.info(f"[Classifier] No scaler found at '{scaler_path}' — "
                             "assuming this model was trained on raw landmarks.")

            logger.info(f"[Classifier] Model loaded successfully. "
                  f"Classes: {len(self.labels)}")
        except Exception as e:
            logger.error(f"[Classifier] Error loading model: {e}")
            self.model = None

    def predict(self, landmarks):
        """
        Predict the sign from hand landmarks.

        Args:
            landmarks: numpy array of shape (63,) — normalized hand landmarks.

        Returns:
            tuple: (sign_label, confidence) if prediction is valid,
                   (None, 0.0) if no model is loaded or confidence is too low.
        """
        if self.model is None:
            return None, 0.0

        try:
            # Reshape for model input: (1, 63)
            input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            if self.scaler is not None:
                input_data = self.scaler.transform(input_data)

            # Run inference
            predictions = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            if confidence < self.confidence_threshold:
                return None, confidence

            # Map class index to label
            if predicted_class < len(self.labels):
                sign_label = self.labels[predicted_class]
            else:
                sign_label = f"sign_{predicted_class}"

            return sign_label, confidence

        except Exception as e:
            logger.error(f"[Classifier] Prediction error: {e}")
            return None, 0.0

    def is_ready(self):
        """Check if the model is loaded and ready for predictions."""
        return self.model is not None

    def get_num_classes(self):
        """Return the number of sign classes the model was trained on."""
        return len(self.labels)


class LSTMClassifier:
    """Loads and runs inference with a trained LSTM sequence model.

    Uses a sliding-window approach: after a successful prediction the buffer
    is shifted forward (not fully cleared), allowing overlapping predictions
    and faster re-detection.  A short prediction history provides consensus-
    based smoothing to avoid spurious one-off detections.
    """

    SEQUENCE_LENGTH = 30
    FEATURE_LENGTH  = 65   # 63 hand + 2 face-relative (see src/normalize.py)
    SLIDE_STEP      = 10   # frames to drop after a detection (sliding window)

    def __init__(self, model_path="models/ksl_lstm_model.h5",
                 confidence_threshold=0.6):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels = []
        self._buffer = deque(maxlen=self.SEQUENCE_LENGTH)

        self._load_model()

    def _load_model(self):
        """Load the LSTM Keras model from disk if it exists."""
        if not os.path.exists(self.model_path):
            logger.info(f"[LSTMClassifier] No LSTM model at '{self.model_path}'.")
            self.model = None
            return

        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(self.model_path)

            label_path = self.model_path.replace('.h5', '_labels.npy')
            # Also check alternate naming (ksl_lstm_labels.npy vs ksl_lstm_model_labels.npy)
            label_path_alt = os.path.join(os.path.dirname(self.model_path), 'ksl_lstm_labels.npy')
            if os.path.exists(label_path):
                self.labels = list(np.load(label_path, allow_pickle=True))
            elif os.path.exists(label_path_alt):
                self.labels = list(np.load(label_path_alt, allow_pickle=True))

            # Read sequence length from metadata if available
            meta_path = self.model_path.replace('.h5', '_metadata.json')
            if os.path.exists(meta_path):
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                self.SEQUENCE_LENGTH = meta.get("sequence_length", 30)
                self.FEATURE_LENGTH = meta.get("feature_length", self.FEATURE_LENGTH)
                self._buffer = deque(maxlen=self.SEQUENCE_LENGTH)

            logger.info(f"[LSTMClassifier] Model loaded. Classes: {len(self.labels)}, "
                  f"Seq: {self.SEQUENCE_LENGTH} frames")
        except Exception as e:
            logger.error(f"[LSTMClassifier] Error: {e}")
            self.model = None

    def push_frame(self, landmarks, face_ref=None):
        """Add a single frame's landmarks (63,) to the rolling buffer.

        face_ref: optional (cx, cy, width) of the detected face, used to
        encode the hand's position relative to the body (see
        src/normalize.py). None if no face was detected this frame.
        """
        if landmarks is not None:
            self._buffer.append(normalize_landmarks(landmarks, face_ref))

    def predict(self):
        """Predict using the full sequence buffer.

        Returns:
            tuple: (sign_label, confidence) or (None, 0.0)
        """
        if self.model is None:
            return None, 0.0

        if len(self._buffer) < self.SEQUENCE_LENGTH:
            return None, 0.0  # not enough frames yet

        try:
            seq = np.array(list(self._buffer), dtype=np.float32)
            seq = seq.reshape(1, self.SEQUENCE_LENGTH, self.FEATURE_LENGTH)

            preds = self.model.predict(seq, verbose=0)
            cls   = int(np.argmax(preds[0]))
            conf  = float(preds[0][cls])

            if conf < self.confidence_threshold:
                return None, conf

            label = self.labels[cls] if cls < len(self.labels) else f"sign_{cls}"
            return label, conf

        except Exception as e:
            logger.error(f"[LSTMClassifier] Prediction error: {e}")
            return None, 0.0

    def slide_buffer(self):
        """Slide the buffer forward by SLIDE_STEP frames (keep recent frames)."""
        for _ in range(min(self.SLIDE_STEP, len(self._buffer))):
            self._buffer.popleft()

    def clear_buffer(self):
        """Reset the frame buffer."""
        self._buffer.clear()

    def buffer_fill(self):
        """Return how full the buffer is as a fraction 0..1."""
        return len(self._buffer) / self.SEQUENCE_LENGTH

    def is_ready(self):
        return self.model is not None

    def get_num_classes(self):
        return len(self.labels)
