#!/usr/bin/env python3
"""
HandsToVoice — LSTM Model Training Script
Trains a sequence-based LSTM model on landmark sequences extracted from videos.
Uses robust data augmentation so even 1 video per sign is enough to start.

Professional evaluation pipeline:
  1. Hyperparameter search (small grid) via stratified k-fold CV, restricted to
     classes with enough raw samples to split (proxy for architecture choice).
  2. Leave-one-out cross-validation with the winning config, over every class
     that has >=2 raw samples, for an honest per-class accuracy estimate.
     Classes with exactly 1 raw sample CANNOT be evaluated this way (removing
     their only example makes the class unlearnable) — they are reported
     separately as "insufficient data".
  3. Final production model: the winning config retrained on 100% of the raw
     data (every class, including 1-sample ones) for deployment.

Every run's config and metrics are appended to models/training_log.jsonl so
different runs/architectures can be compared over time.
"""

import os, sys, json, csv, argparse, time
from collections import Counter
from datetime import datetime
import numpy as np

from src.logger import get_logger

logger = get_logger("train_lstm")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

SEQUENCE_DIR    = "data/sequences"
LABELS_CSV      = os.path.join(SEQUENCE_DIR, "labels.csv")
MODELS_DIR      = "models"
LOG_PATH        = os.path.join(MODELS_DIR, "training_log.jsonl")
SEQUENCE_LENGTH = 30
FEATURE_LENGTH  = 63


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sequences():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Run extract_video_landmarks.py first.\n{LABELS_CSV} not found.")
    mapping = {}
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["file"]] = row["label"]
    X, y = [], []
    for fname, label in mapping.items():
        fpath = os.path.join(SEQUENCE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        seq = np.load(fpath)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_LENGTH):
            continue
        X.append(seq)
        y.append(label)
    if not X:
        raise ValueError("No valid sequences loaded.")
    logger.info(f"[Training] Loaded {len(X)} sequences, labels: {sorted(set(y))}")
    return np.array(X, dtype=np.float32), np.array(y)


# ── Augmentation ──────────────────────────────────────────────────────────────

def _time_warp(seq, rng):
    """Warp the temporal axis via interpolation (speed-up / slow-down)."""
    T, F = seq.shape
    warp_strength = rng.uniform(0.1, 0.3)
    orig_t = np.linspace(0, 1, T)
    warped_t = orig_t + warp_strength * np.sin(2 * np.pi * orig_t * rng.uniform(0.5, 2.0))
    warped_t = (warped_t - warped_t.min()) / (warped_t.max() - warped_t.min() + 1e-8)
    out = np.zeros_like(seq)
    for f in range(F):
        out[:, f] = np.interp(orig_t, warped_t, seq[:, f])
    return out.astype(np.float32)


def _per_joint_jitter(seq, rng, sigma=0.008):
    """Add small independent, sequence-consistent offsets per landmark joint."""
    T, F = seq.shape
    n_joints = F // 3
    joint_offsets = rng.normal(0, sigma, (1, n_joints, 3)).astype(np.float32)
    joint_offsets = np.tile(joint_offsets, (T, 1, 1))
    out = seq.reshape(T, n_joints, 3) + joint_offsets
    return out.reshape(T, F).astype(np.float32)


def _temporal_dropout(seq, rng, drop_rate=0.1):
    """Randomly zero out a few frames to build robustness to detection gaps."""
    T, F = seq.shape
    out = seq.copy()
    n_drop = max(1, int(T * drop_rate))
    drop_idx = rng.choice(T, size=n_drop, replace=False)
    out[drop_idx] = 0.0
    return out


def _speed_change(seq, rng):
    """Simulate slightly faster or slower signing by resampling frames."""
    T, F = seq.shape
    speed = rng.uniform(0.8, 1.2)
    new_len = int(T * speed)
    if new_len < 5:
        return seq
    indices = np.linspace(0, T - 1, new_len).astype(int)
    resampled = seq[indices]
    if len(resampled) < T:
        pad = np.tile(resampled[-1:], (T - len(resampled), 1))
        resampled = np.vstack([resampled, pad])
    elif len(resampled) > T:
        idx = np.linspace(0, len(resampled) - 1, T, dtype=int)
        resampled = resampled[idx]
    return resampled.astype(np.float32)


def _rotation_jitter(seq, rng, max_deg=15):
    """Rotate x,y coords by a small constant angle — simulates the camera or
    the signer's hand being held at a slightly different angle. Landmarks are
    already wrist-centered by normalize_landmarks, so this rotates in place
    and does not need a separate pivot. Z is left untouched (depth axis is
    less reliably estimated by MediaPipe and rotating it adds noise, not
    signal).
    """
    T, F = seq.shape
    n_joints = F // 3
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(angle), np.sin(angle)
    out = seq.reshape(T, n_joints, 3).copy()
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out.reshape(T, F).astype(np.float32)


def augment_sequence(seq, n=50, rng=None):
    """Generate n augmented variants of a single (30, 63) sequence."""
    rng = rng or np.random.default_rng()
    out = [seq]  # always keep original
    for _ in range(n - 1):
        s = seq.copy()
        s += rng.normal(0, 0.008, s.shape).astype(np.float32)
        s *= rng.uniform(0.92, 1.08)
        if rng.random() > 0.3:
            s = _time_warp(s, rng)
        if rng.random() > 0.4:
            s = _per_joint_jitter(s, rng)
        if rng.random() > 0.5:
            s = _rotation_jitter(s, rng)
        if rng.random() > 0.7:
            s = _temporal_dropout(s, rng)
        if rng.random() > 0.5:
            s = _speed_change(s, rng)
        # NOTE: Horizontal flip REMOVED — flipping x-coords produces
        # mirror-image hands which are anatomically different signs
        # in sign language (left hand != right hand).
        out.append(s.astype(np.float32))
    return out


def augment_dataset(X, y, n_per_sample=50, rng=None):
    """Augment every sequence n_per_sample times and shuffle."""
    rng = rng or np.random.default_rng()
    X_aug, y_aug = [], []
    for seq, label in zip(X, y):
        for aug in augment_sequence(seq, n=n_per_sample, rng=rng):
            X_aug.append(aug)
            y_aug.append(label)
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_lstm(num_classes, units=64, dropout=0.15, dense_units=64, use_gru=False):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout,
                                          BatchNormalization, Bidirectional)
    Rec = GRU if use_gru else LSTM
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURE_LENGTH)),
        BatchNormalization(),
        Bidirectional(Rec(units, return_sequences=False)),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        BatchNormalization(),
        Dropout(dropout),
        Dense(num_classes, activation="softmax"),
    ], name="ksl_lstm")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ── Fold training helper (shared by CV, LOOCV, final fit) ──────────────────────

def _fit_fold(X_train_raw, y_train_enc, X_test_raw, y_test_enc, nc, config,
              epochs, n_aug_train, n_aug_test, rng, verbose=0):
    """Train on raw train samples (augmented), evaluate on raw test samples
    (lightly augmented). Returns (accuracy, y_true, y_pred, best_epoch)."""
    from tensorflow import keras

    X_train, y_train = augment_dataset(X_train_raw, y_train_enc, n_per_sample=n_aug_train, rng=rng)
    X_test, y_test = augment_dataset(X_test_raw, y_test_enc, n_per_sample=n_aug_test, rng=rng)

    # Carve a small validation slice out of the augmented train set purely for
    # early stopping — NOT a generalization estimate (it shares raw samples
    # with the train set), unlike the held-out fold/LOOCV test set above.
    n_val = max(nc, int(0.1 * len(X_train)))
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    X_fit, y_fit = X_train[n_val:], y_train[n_val:]

    from sklearn.utils.class_weight import compute_class_weight
    present = np.unique(y_fit)
    cw_vals = compute_class_weight("balanced", classes=present, y=y_fit)
    class_weights = {int(c): float(w) for c, w in zip(present, cw_vals)}

    model = build_lstm(nc, **config)
    cb = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                         monitor="val_accuracy")]
    hist = model.fit(X_fit, y_fit, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=16, class_weight=class_weights,
                      callbacks=cb, verbose=verbose)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = float(np.mean(y_pred == y_test))
    best_epoch = len(hist.history["val_accuracy"]) - 10  # approx (patience offset)
    return acc, y_test, y_pred, max(1, best_epoch), model


# ── Hyperparameter search (fast proxy on well-represented classes) ────────────

CANDIDATE_CONFIGS = [
    {"units": 64, "dropout": 0.15, "dense_units": 64, "use_gru": False},  # baseline
    {"units": 32, "dropout": 0.30, "dense_units": 32, "use_gru": False},  # smaller, more regularized
    {"units": 64, "dropout": 0.20, "dense_units": 64, "use_gru": True},   # GRU variant
]


def hyperparam_search(X_raw, y_raw, k=3, epochs=60, n_aug=25, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    counts = Counter(y_raw)
    mask = np.array([counts[lbl] >= k for lbl in y_raw])
    X_e, y_e = X_raw[mask], y_raw[mask]
    if len(set(y_e)) < 2:
        logger.info("[HP-Search] Not enough classes with >= "
              f"{k} raw samples to run a search — skipping, using baseline config.")
        return CANDIDATE_CONFIGS[0], []

    le = LabelEncoder()
    y_enc = le.fit_transform(y_e)
    logger.info(f"[HP-Search] Using {len(X_e)} samples from {len(le.classes_)} "
          f"well-represented classes: {list(le.classes_)}")

    results = []
    for cfg in CANDIDATE_CONFIGS:
        rng = np.random.default_rng(seed)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        accs = []
        t0 = time.time()
        for tr_idx, te_idx in skf.split(X_e, y_enc):
            acc, *_ = _fit_fold(X_e[tr_idx], y_enc[tr_idx], X_e[te_idx], y_enc[te_idx],
                                 nc=len(le.classes_), config=cfg, epochs=epochs,
                                 n_aug_train=n_aug, n_aug_test=5, rng=rng, verbose=0)
            accs.append(acc)
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        dt = time.time() - t0
        logger.info(f"[HP-Search] {cfg} → {mean_acc:.3f} ± {std_acc:.3f}  ({dt:.0f}s)")
        results.append({"config": cfg, "mean_acc": mean_acc, "std_acc": std_acc})

    best = max(results, key=lambda r: r["mean_acc"])
    logger.info(f"[HP-Search] Winner: {best['config']} ({best['mean_acc']:.3f})")
    return best["config"], results


# ── Leave-one-out cross-validation (honest generalization estimate) ───────────

def loocv_evaluate(X_raw, y_raw, config, epochs=60, n_aug=25, seed=42):
    """For every class with >=2 raw samples, hold out one raw sample at a
    time and test on it, training on everything else (including 1-sample
    classes, so they still shape the decision boundary even though they
    can't be scored). Classes with exactly 1 sample are reported separately
    as untestable.
    """
    from sklearn.preprocessing import LabelEncoder

    counts = Counter(y_raw)
    eligible_idx = [i for i, lbl in enumerate(y_raw) if counts[lbl] >= 2]
    untestable = sorted([lbl for lbl, c in counts.items() if c < 2])

    le = LabelEncoder()
    y_enc_all = le.fit_transform(y_raw)
    nc = len(le.classes_)

    logger.info(f"[LOOCV] {len(eligible_idx)} of {len(X_raw)} samples are eligible "
          f"(class has >=2 raw examples).")
    if untestable:
        logger.info(f"[LOOCV] Untestable (only 1 raw sample, excluded from scoring): {untestable}")

    y_true_all, y_pred_all = [], []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for n, i in enumerate(eligible_idx, 1):
        train_mask = np.ones(len(X_raw), dtype=bool)
        train_mask[i] = False
        acc, y_test, y_pred, _, _ = _fit_fold(
            X_raw[train_mask], y_enc_all[train_mask],
            X_raw[i:i+1], y_enc_all[i:i+1],
            nc=nc, config=config, epochs=epochs,
            n_aug_train=n_aug, n_aug_test=5, rng=rng, verbose=0)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        if n % 5 == 0 or n == len(eligible_idx):
            logger.info(f"[LOOCV] {n}/{len(eligible_idx)} folds done "
                  f"({time.time()-t0:.0f}s elapsed)")

    return np.array(y_true_all), np.array(y_pred_all), le, untestable


# ── Final production model (trained on 100% of raw data) ──────────────────────

def fit_final_model(X_raw, y_raw, config, epochs, n_aug, seed=42):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    nc = len(le.classes_)
    rng = np.random.default_rng(seed)
    # No held-out test here by design — every raw sample is precious and
    # LOOCV above already gives the honest generalization estimate. A small
    # slice of augmented data is still carved out inside _fit_fold purely
    # for early-stopping.
    acc, y_test, y_pred, best_epoch, model = _fit_fold(
        X_raw, y_enc, X_raw, y_enc,  # "test" here is train-derived, informational only
        nc=nc, config=config, epochs=epochs, n_aug_train=n_aug, n_aug_test=3,
        rng=rng, verbose=1)
    return model, le, best_epoch


# ── Reporting ───────────────────────────────────────────────────────────────────

def report_and_save(y_true, y_pred, le, untestable, model, best_config, hp_results,
                     final_epochs):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(MODELS_DIR, exist_ok=True)
    present_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [le.classes_[i] for i in present_classes]

    logger.info("\n" + "=" * 60)
    logger.info("[LOOCV] Honest per-class generalization report")
    print("=" * 60)
    report_str = classification_report(y_true, y_pred, labels=present_classes,
                                        target_names=target_names, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=present_classes,
                                         target_names=target_names, zero_division=0,
                                         output_dict=True)
    logger.info(report_str)
    overall_acc = float(np.mean(y_true == y_pred))
    logger.info(f"[LOOCV] Overall LOOCV accuracy: {overall_acc:.2%} "
          f"(n={len(y_true)} held-out samples)")
    if untestable:
        logger.info(f"[LOOCV] NOT evaluated (only 1 raw sample each): {untestable}")

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    plt.figure(figsize=(max(6, len(present_classes)), max(5, len(present_classes) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("LOOCV Confusion Matrix (held-out samples only)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "lstm_confusion_matrix.png"), dpi=150)
    plt.close()

    # Save final production model
    model.save(os.path.join(MODELS_DIR, "ksl_lstm_model.h5"))
    np.save(os.path.join(MODELS_DIR, "ksl_lstm_labels.npy"), le.classes_)
    metadata = {
        "model_type": "lstm",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_length": FEATURE_LENGTH,
        "num_classes": len(le.classes_),
        "classes": list(le.classes_),
        "loocv_accuracy": overall_acc,
        "loocv_n_samples": int(len(y_true)),
        "untestable_classes": untestable,
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in target_names
        },
        "best_config": best_config,
        "final_model_epochs": final_epochs,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "ksl_lstm_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            **metadata,
            "classification_report": report_str,
            "hp_search_results": hp_results,
        }) + "\n")

    logger.info("\n[Training] Model saved -> models/ksl_lstm_model.h5")
    logger.info("[Training] Metadata    -> models/ksl_lstm_metadata.json")
    logger.info(f"[Training] Run log     -> {LOG_PATH}")
    logger.info("[Training] Run the app: python main.py")
    return overall_acc


# ── Entry point ───────────────────────────────────────────────────────────────

def run_training(hp_epochs=60, loocv_epochs=60, final_epochs=150, n_aug=25,
                  skip_hp_search=False, skip_loocv=False):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    X_raw, y_raw = load_sequences()
    counts = Counter(y_raw.tolist())
    logger.info(f"[Training] Per-class raw sample counts: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")

    if skip_hp_search:
        best_config, hp_results = CANDIDATE_CONFIGS[0], []
    else:
        best_config, hp_results = hyperparam_search(X_raw, y_raw, epochs=hp_epochs, n_aug=n_aug)

    if skip_loocv:
        y_true, y_pred, le, untestable = np.array([]), np.array([]), None, []
    else:
        y_true, y_pred, le, untestable = loocv_evaluate(
            X_raw, y_raw, best_config, epochs=loocv_epochs, n_aug=n_aug)

    logger.info("\n[Training] Fitting FINAL production model on 100% of raw data...")
    model, le_final, final_ep = fit_final_model(X_raw, y_raw, best_config,
                                                 epochs=final_epochs, n_aug=max(n_aug, 40))

    le_report = le if le is not None else le_final
    if len(y_true) == 0:
        logger.info("[Training] LOOCV was skipped — no honest generalization metric available.")
        model.save(os.path.join(MODELS_DIR, "ksl_lstm_model.h5"))
        np.save(os.path.join(MODELS_DIR, "ksl_lstm_labels.npy"), le_final.classes_)
        return None

    return report_and_save(y_true, y_pred, le_report, untestable, model,
                            best_config, hp_results, final_ep)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model on video sequences")
    parser.add_argument("--hp-epochs", type=int, default=60)
    parser.add_argument("--loocv-epochs", type=int, default=60)
    parser.add_argument("--final-epochs", type=int, default=150)
    parser.add_argument("--n-aug", type=int, default=25,
                         help="Augmented copies per raw train sample during CV/LOOCV")
    parser.add_argument("--skip-hp-search", action="store_true")
    parser.add_argument("--skip-loocv", action="store_true")
    args = parser.parse_args()
    print("=" * 60)
    logger.info("HandsToVoice — LSTM Model Training (CV + LOOCV + final fit)")
    print("=" * 60)
    run_training(hp_epochs=args.hp_epochs, loocv_epochs=args.loocv_epochs,
                 final_epochs=args.final_epochs, n_aug=args.n_aug,
                 skip_hp_search=args.skip_hp_search, skip_loocv=args.skip_loocv)


if __name__ == "__main__":
    main()
