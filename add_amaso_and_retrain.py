#!/usr/bin/env python3
"""
Add "amaso" (Eyes) training data and retrain the HandsToVoice model.

The 'amaso' (Eyes) sign in Kinyarwanda Sign Language:
  - Index and middle fingers extended together, pointing upward
  - Ring finger and pinky curled into the palm
  - Thumb tucked across ring/pinky
  - Hand raised to face level, fingertips near the eye
  - Palm faces inward (toward face)

MediaPipe hand landmark indices:
  0  = WRIST
  1-4  = THUMB  (CMC → TIP)
  5-8  = INDEX  (MCP → TIP)
  9-12 = MIDDLE (MCP → TIP)
  13-16= RING   (MCP → TIP)
  17-20= PINKY  (MCP → TIP)

Each landmark stored as (x, y, z) → 21×3 = 63 features, normalized 0..1.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Realistic "amaso" (Eyes) landmark generator
# ──────────────────────────────────────────────────────────────────────────────

def _amaso_base():
    """
    Return a (21, 3) array of *normalized* MediaPipe-style landmarks
    for the 'amaso' (Eyes) sign.

    The hand is held high, near the face.  Index and middle fingers are
    extended upward together.  Ring, pinky, and thumb are curled in.

    Convention: x ∈ [0,1] left→right, y ∈ [0,1] top→bottom, z ≈ 0 (depth).
    The hand centre sits around (0.52, 0.40) — upper part of frame (face area).
    """
    lm = np.zeros((21, 3), dtype=np.float32)

    cx, cy = 0.52, 0.42   # hand centre — raised to eye level

    # ── WRIST (0) ──────────────────────────────────────────────────────────
    lm[0] = [cx, cy + 0.22, 0.00]

    # ── THUMB (1-4): tucked across palm, curled inward ─────────────────────
    lm[1]  = [cx - 0.03, cy + 0.14, 0.02]   # CMC
    lm[2]  = [cx - 0.06, cy + 0.10, 0.03]   # MCP
    lm[3]  = [cx - 0.04, cy + 0.07, 0.04]   # IP  — curling back
    lm[4]  = [cx - 0.01, cy + 0.06, 0.05]   # TIP — tucked under index

    # ── INDEX (5-8): fully extended, pointing UP toward eye ────────────────
    lm[5]  = [cx + 0.02, cy + 0.08, 0.00]   # MCP
    lm[6]  = [cx + 0.02, cy + 0.01, -0.01]  # PIP
    lm[7]  = [cx + 0.02, cy - 0.06, -0.01]  # DIP
    lm[8]  = [cx + 0.02, cy - 0.12, -0.02]  # TIP — pointing at eye

    # ── MIDDLE (9-12): fully extended, pointing UP right next to index ─────
    lm[9]  = [cx + 0.06, cy + 0.08, 0.00]   # MCP
    lm[10] = [cx + 0.06, cy + 0.01, -0.01]  # PIP
    lm[11] = [cx + 0.06, cy - 0.06, -0.01]  # DIP
    lm[12] = [cx + 0.06, cy - 0.12, -0.02]  # TIP — pointing at eye

    # ── RING (13-16): curled into palm ─────────────────────────────────────
    lm[13] = [cx + 0.08, cy + 0.10, 0.00]   # MCP
    lm[14] = [cx + 0.09, cy + 0.06, 0.03]   # PIP — bending down
    lm[15] = [cx + 0.07, cy + 0.09, 0.05]   # DIP — curled back
    lm[16] = [cx + 0.05, cy + 0.11, 0.06]   # TIP — touching palm

    # ── PINKY (17-20): curled into palm ────────────────────────────────────
    lm[17] = [cx + 0.10, cy + 0.12, 0.00]   # MCP
    lm[18] = [cx + 0.11, cy + 0.09, 0.03]   # PIP — bending down
    lm[19] = [cx + 0.09, cy + 0.11, 0.05]   # DIP — curled back
    lm[20] = [cx + 0.07, cy + 0.13, 0.06]   # TIP — touching palm

    return lm


def generate_amaso_samples(n: int = 150, seed: int = 42) -> np.ndarray:
    """
    Generate `n` augmented samples around the amaso base pose.

    Augmentations applied per sample:
      • Gaussian noise on all landmarks         (σ ≈ 0.007)
      • Small random global translation         (±0.04 each axis)
      • Small random scale                      (0.88 – 1.12)
      • Small random rotation in the xy plane   (±10°)
      • Slight finger-splay variation            (±0.005 on index/middle x)
    """
    rng = np.random.default_rng(seed)
    base = _amaso_base()          # (21, 3)
    samples = []

    for _ in range(n):
        lm = base.copy()

        # 1. Noise
        lm += rng.normal(0, 0.007, lm.shape)

        # 2. Slight splay variation on the two extended fingers
        #    (index tips 5-8 and middle tips 9-12)
        splay = rng.uniform(-0.005, 0.005)
        lm[5:9, 0]  -= splay   # index shifts slightly left
        lm[9:13, 0] += splay   # middle shifts slightly right

        # 3. Global scale (around the wrist)
        scale = rng.uniform(0.88, 1.12)
        lm = (lm - lm[0]) * scale + lm[0]

        # 4. Global translation
        lm[:, 0] += rng.uniform(-0.04, 0.04)
        lm[:, 1] += rng.uniform(-0.04, 0.04)
        lm[:, 2] += rng.uniform(-0.005, 0.005)

        # 5. Small 2-D rotation around wrist
        angle = rng.uniform(-10, 10) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = lm[:, 0] - lm[0, 0]
        dy = lm[:, 1] - lm[0, 1]
        lm[:, 0] = lm[0, 0] + cos_a * dx - sin_a * dy
        lm[:, 1] = lm[0, 1] + sin_a * dx + cos_a * dy

        # Clip to [0, 1]
        lm = np.clip(lm, 0.0, 1.0)

        samples.append(lm.flatten())   # 63 values

    return np.array(samples, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Append to existing CSV and retrain
# ──────────────────────────────────────────────────────────────────────────────

def append_amaso_to_csv(csv_path: str, n_samples: int = 150):
    """Append amaso rows to the processed CSV."""
    print(f"[Data] Generating {n_samples} amaso (Eyes) samples …")
    X = generate_amaso_samples(n=n_samples)

    cols = [f'landmark_{i}' for i in range(63)] + ['label']
    rows = []
    for vec in X:
        row = dict(zip([f'landmark_{i}' for i in range(63)], vec.tolist()))
        row['label'] = 'amaso'
        rows.append(row)

    df_new = pd.DataFrame(rows, columns=cols)

    # Load existing data
    df_old = pd.read_csv(csv_path)

    # Drop any old amaso rows so we don't duplicate
    df_old = df_old[df_old['label'] != 'amaso']

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)

    label_counts = df_combined['label'].value_counts()
    print(f"[Data] Updated CSV — {len(df_combined)} total rows")
    print(f"[Data] Label distribution:\n{label_counts.to_string()}")
    return csv_path


def retrain(csv_path: str, models_dir: str = 'models', epochs: int = 60):
    """Retrain the model with the updated CSV (includes amaso)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib

    tf.get_logger().setLevel('ERROR')

    print("\n[Train] Loading data …")
    df = pd.read_csv(csv_path)
    X = df[[f'landmark_{i}' for i in range(63)]].values
    y = df['label'].values

    # Encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"[Train] {len(X)} samples — {num_classes} classes: {list(le.classes_)}")

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
    )

    # Scale
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # Model
    model = keras.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    print("\n[Train] Training …")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5, verbose=0),
        ],
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n[Train] Test accuracy: {acc:.4f}")

    # Save
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ksl_model.h5')
    model.save(model_path)
    print(f"[Train] Model  → {model_path}")

    labels_path = os.path.join(models_dir, 'ksl_model_labels.npy')
    np.save(labels_path, le.classes_)
    print(f"[Train] Labels → {labels_path}")

    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"[Train] Scaler → {scaler_path}")

    meta = {
        'model_type': 'neural_network',
        'num_classes': num_classes,
        'feature_length': 63,
        'classes': list(le.classes_),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'test_accuracy': float(acc),
    }
    meta_path = os.path.join(models_dir, 'ksl_model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Train] Meta   → {meta_path}")

    print("\n✅  Done! amaso (Eyes) is now part of the model.")
    print(f"    Recognised signs: {list(le.classes_)}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CSV = 'data/processed/sample_data.csv'

    print("=" * 60)
    print("HandsToVoice — Add amaso (Eyes) & Retrain")
    print("=" * 60)

    csv_path = append_amaso_to_csv(CSV, n_samples=150)
    retrain(csv_path, models_dir='models', epochs=60)
