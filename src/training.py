"""
HandsToVoice — LSTM training pipeline shared by the in-app trainer
(src/gui/add_sign_dialog.py TrainWorker) and evaluate_model.py, so the
evaluation report always measures exactly what the app trains.
"""

import numpy as np

from src.normalize import FEATURE_LENGTH, HAND_FEATURE_LENGTH  # noqa: F401 (re-exported)

SEQUENCE_LENGTH = 30


def split_per_class(y_enc, rng, val_fraction=0.15):
    """Split ORIGINAL (unaugmented) samples into train/val indices, per class.
    Augmenting before splitting puts near-duplicate copies of the same clip
    on both sides, so validation would just measure memorization."""
    train_idx, val_idx = [], []
    for cls in np.unique(y_enc):
        cls_idx = np.where(y_enc == cls)[0]
        rng.shuffle(cls_idx)
        n_val_cls = max(1, int(len(cls_idx) * val_fraction)) if len(cls_idx) > 1 else 0
        val_idx.extend(cls_idx[:n_val_cls])
        train_idx.extend(cls_idx[n_val_cls:])
    return np.array(train_idx), np.array(val_idx)


def augment(X_raw, y_raw, rng):
    """Each raw training sequence plus n_aug-1 randomized copies."""
    n_aug = max(20, 100 // max(1, len(X_raw)))
    X_out, y_out = [], []
    for seq, lbl in zip(X_raw, y_raw):
        X_out.append(seq)
        y_out.append(lbl)
        for _ in range(n_aug - 1):
            s = seq.copy()
            s += rng.normal(0, 0.01, s.shape).astype(np.float32)
            s *= rng.uniform(0.9, 1.1)
            s = np.roll(s, rng.integers(-3, 4), axis=0)
            if rng.random() > 0.5:
                # Left-right mirror. Sequences are already wrist-centered by
                # normalize_landmarks, so a mirror is negation of x (not
                # 1.0 - x), plus the face-relative horizontal offset.
                hand_x_idx = np.arange(0, HAND_FEATURE_LENGTH, 3)
                s[:, hand_x_idx] = -s[:, hand_x_idx]
                s[:, HAND_FEATURE_LENGTH] = -s[:, HAND_FEATURE_LENGTH]
            # Simulate the face not being detected. Without this the model
            # leaned on face features so hard that accuracy fell from 96% to
            # 46% whenever the face was missing live.
            if rng.random() < 0.25:
                s[:, HAND_FEATURE_LENGTH:] = 0.0
            X_out.append(s.astype(np.float32))
            y_out.append(lbl)
    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out)
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def build_model(num_classes):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURE_LENGTH)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ], name="ksl_lstm")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def fit(model, X_train, y_train, X_val, y_val, epochs=60):
    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                      monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5, monitor="val_loss"),
    ]
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=16, callbacks=callbacks, verbose=0)
