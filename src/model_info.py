"""
HandsToVoice — Model Info Helper
Reads the LSTM training metadata (written by train_lstm_model.py) so the GUI
can show the user what the model actually knows, instead of a generic
"LSTM (Video)" label that implies more than the data supports.
"""

import json
import os

from src.logger import get_logger

logger = get_logger("model_info")

LSTM_METADATA_PATH = "models/ksl_lstm_metadata.json"


def load_lstm_metadata(path=LSTM_METADATA_PATH):
    """Return the LSTM training metadata dict, or None if it doesn't exist
    or can't be parsed. Tolerant of older metadata files that predate the
    honest LOOCV evaluation (no 'loocv_accuracy'/'untestable_classes' keys).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[ModelInfo] Could not read {path}: {e}")
        return None


def summarize(metadata, vocabulary=None):
    """Build a small, display-ready summary dict from raw metadata.

    Returns None if no metadata is available. Fields are all optional-safe:
    metadata written before the LOOCV pipeline existed only has
    'test_accuracy' (from duplicating a single sample — not a real
    generalization estimate), which is reported as such rather than shown
    as if it were trustworthy.
    """
    if not metadata:
        return None

    classes = metadata.get("classes", [])
    untestable = metadata.get("untestable_classes", [])
    tested = [c for c in classes if c not in untestable]

    total_vocab = vocabulary.get_total_signs() if vocabulary else None

    has_loocv = "loocv_accuracy" in metadata
    summary = {
        "model_type": metadata.get("model_type", "unknown"),
        "timestamp": metadata.get("timestamp"),
        "num_classes": len(classes),
        "classes": classes,
        "tested_classes": tested,
        "untested_classes": untestable,
        "total_vocab_signs": total_vocab,
        "has_honest_eval": has_loocv,
    }

    if has_loocv:
        summary["accuracy"] = metadata.get("loocv_accuracy")
        summary["accuracy_n"] = metadata.get("loocv_n_samples")
        summary["accuracy_label"] = "cross-validated accuracy"
    elif "test_accuracy" in metadata:
        # Legacy metadata from before the honest-evaluation pipeline —
        # this number is not a real generalization estimate (see
        # train_lstm_model.py's old duplicate-sample fallback) so it's
        # labeled clearly rather than presented at face value.
        summary["accuracy"] = metadata.get("test_accuracy")
        summary["accuracy_n"] = None
        summary["accuracy_label"] = "self-test accuracy (unverified — trained before honest evaluation existed)"

    summary["per_class"] = metadata.get("per_class")  # may be absent on older runs

    return summary
