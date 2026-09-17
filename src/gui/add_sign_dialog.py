"""
HandsToVoice — Add Sign Dialog
3-step wizard: Define sign → Record videos → Extract + Train LSTM
"""

import os, sys, time, csv, json
from datetime import datetime
from collections import deque
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QProgressBar, QTextEdit, QStackedWidget,
    QWidget, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QAudioRecorder, QAudioEncoderSettings, QMultimedia

from src.logger import get_logger

logger = get_logger("gui.add_sign_dialog")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

VIDEO_DIR = "data/videos"
SEQUENCE_DIR = "data/sequences"
SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 63
TEST_CONFIDENCE_THRESHOLD = 0.5


# ── Background training thread ───────────────────────────────────────────────

class TrainWorker(QThread):
    """Runs landmark extraction + LSTM training in a background thread."""
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, sign_label):
        super().__init__()
        self.sign_label = sign_label

    def run(self):
        try:
            self._extract()
            self._train()
            self.finished.emit(True, "Training complete!")
        except Exception as e:
            self.finished.emit(False, str(e))

    def _extract(self):
        from src.detector import HandDetector
        self.log.emit("[1/2] Extracting landmarks from videos...")
        detector = HandDetector()
        sign_folder = os.path.join(VIDEO_DIR, self.sign_label)
        os.makedirs(SEQUENCE_DIR, exist_ok=True)

        supported = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = [f for f in os.listdir(sign_folder)
                  if os.path.splitext(f)[1].lower() in supported]

        if not videos:
            raise ValueError(f"No videos found in {sign_folder}")

        # Load existing labels.csv rows (from other signs)
        labels_csv = os.path.join(SEQUENCE_DIR, "labels.csv")
        existing_rows = []
        if os.path.exists(labels_csv):
            with open(labels_csv, newline="") as f:
                existing_rows = list(csv.DictReader(f))

        # Remove old entries for this sign label
        existing_rows = [r for r in existing_rows if r["label"] != self.sign_label]

        new_rows = []
        for idx, vfile in enumerate(sorted(videos)):
            vpath = os.path.join(sign_folder, vfile)
            self.log.emit(f"  → {vfile}")
            cap = cv2.VideoCapture(vpath)
            frames_lm = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, lm_list = detector.process_frame(frame)
                if lm_list:
                    frames_lm.append(lm_list[0])
                else:
                    frames_lm.append(np.zeros(FEATURE_LENGTH, dtype=np.float32))
            cap.release()

            if len(frames_lm) < 3:
                self.log.emit(f"  ⚠ Skipped {vfile} (too few frames)")
                continue

            seq = self._pad_or_trim(frames_lm, SEQUENCE_LENGTH)
            out_name = f"{self.sign_label}_{idx:04d}.npy"
            np.save(os.path.join(SEQUENCE_DIR, out_name), seq)
            new_rows.append({"file": out_name, "label": self.sign_label})
            self.log.emit(f"  ✅ {out_name}")

        if not new_rows:
            raise ValueError("No valid sequences extracted")

        all_rows = existing_rows + new_rows
        with open(labels_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "label"])
            w.writeheader()
            w.writerows(all_rows)

        detector.close()
        self.log.emit(f"[1/2] Done — {len(new_rows)} sequences extracted\n")

    @staticmethod
    def _pad_or_trim(frames, target_len):
        arr = np.array(frames, dtype=np.float32)
        n = len(arr)
        if n < target_len:
            pad = np.tile(arr[-1], (target_len - n, 1))
            arr = np.vstack([arr, pad])
        elif n > target_len:
            indices = np.linspace(0, n - 1, target_len, dtype=int)
            arr = arr[indices]
        return arr

    def _train(self):
        self.log.emit("[2/2] Training LSTM model...")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from sklearn.preprocessing import LabelEncoder
        from tensorflow import keras

        labels_csv = os.path.join(SEQUENCE_DIR, "labels.csv")
        mapping = {}
        with open(labels_csv, newline="") as f:
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
            raise ValueError("No valid sequences to train on")

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        self.log.emit(f"  Loaded {len(X)} sequences, labels: {sorted(set(y))}")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        nc = len(le.classes_)
        self.log.emit(f"  {nc} classes: {list(le.classes_)}")

        # Augment
        rng = np.random.default_rng(42)
        n_aug = max(20, 100 // len(X))
        X_aug, y_aug = [], []
        for seq, lbl in zip(X, y_enc):
            X_aug.append(seq)
            y_aug.append(lbl)
            for _ in range(n_aug - 1):
                s = seq.copy()
                s += rng.normal(0, 0.01, s.shape).astype(np.float32)
                s *= rng.uniform(0.9, 1.1)
                shift = rng.integers(-3, 4)
                s = np.roll(s, shift, axis=0)
                if rng.random() > 0.5:
                    x_idx = np.arange(0, FEATURE_LENGTH, 3)
                    s[:, x_idx] = 1.0 - s[:, x_idx]
                X_aug.append(s.astype(np.float32))
                y_aug.append(lbl)

        X_aug = np.array(X_aug, dtype=np.float32)
        y_aug = np.array(y_aug)
        idx = np.random.permutation(len(X_aug))
        X_aug, y_aug = X_aug[idx], y_aug[idx]
        self.log.emit(f"  Augmented: {X_aug.shape[0]} samples")

        # Build model
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input, LSTM as LSTMLayer, Dense, Dropout
        model = Sequential([
            Input(shape=(SEQUENCE_LENGTH, FEATURE_LENGTH)),
            LSTMLayer(64, return_sequences=True),
            Dropout(0.3),
            LSTMLayer(32),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(nc, activation="softmax"),
        ], name="ksl_lstm")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # Train
        split = max(1, int(len(X_aug) * 0.15))
        X_train, X_val = X_aug[split:], X_aug[:split]
        y_train, y_val = y_aug[split:], y_aug[:split]
        self.log.emit(f"  Train: {len(X_train)}  Val: {len(X_val)}")

        cb = [
            keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                          monitor="val_accuracy"),
            keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5,
                                              monitor="val_loss"),
        ]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=60, batch_size=16, callbacks=cb, verbose=0)

        best_acc = max(history.history.get("val_accuracy", [0]))
        self.log.emit(f"  Best val accuracy: {best_acc:.2%}")

        # Save
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model.save(os.path.join(models_dir, "ksl_lstm_model.h5"))
        np.save(os.path.join(models_dir, "ksl_lstm_labels.npy"), le.classes_)
        with open(os.path.join(models_dir, "ksl_lstm_metadata.json"), "w") as f:
            json.dump({
                "model_type": "lstm", "sequence_length": SEQUENCE_LENGTH,
                "feature_length": FEATURE_LENGTH, "num_classes": nc,
                "classes": list(le.classes_), "test_accuracy": float(best_acc),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        self.log.emit(f"\n[2/2] ✅ Model saved! Accuracy: {best_acc:.2%}")


# ── Add Sign Dialog ──────────────────────────────────────────────────────────

class AddSignDialog(QDialog):
    """4-step wizard: Define → Record → Train → Test."""
    training_complete = pyqtSignal()

    def _current_label(self):
        """The canonical slug for the sign label field — lowercase, stripped,
        spaces collapsed to underscores. Every step (voice recording, video
        recording, vocabulary registration, training) must read the label
        through this single helper, not `self.inp_label.text()` directly:
        previously only the voice-recording step normalized spaces, so a
        label typed with a space could end up saved under a different slug
        for its video folder / vocabulary entry than for its audio file,
        silently orphaning the custom recording.
        """
        return self.inp_label.text().strip().lower().replace(" ", "_")

    def __init__(self, vocabulary, camera_index=0, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.camera_index = camera_index
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.record_start = 0
        self.clips_recorded = 0
        self._worker = None

        # Test step state
        self._test_detector = None
        self._test_lstm = None
        self._test_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self._test_match_count = 0
        self._test_total_preds = 0

        # Voice recording components
        self.voice_recorder = QAudioRecorder(self)
        self.voice_is_recording = False
        
        self.voice_record_timer = QTimer(self)
        self.voice_record_timer.setSingleShot(True)
        self.voice_record_timer.timeout.connect(self._stop_voice_recording)
        
        self.voice_elapsed_timer = QTimer(self)
        self.voice_elapsed_timer.timeout.connect(self._on_voice_elapsed_tick)
        self.voice_elapsed_start = 0

        self.setWindowTitle("HandsToVoice — Add New Sign")
        self.setMinimumSize(800, 600)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # Title
        title = QLabel("➕ Add New Sign")
        title.setObjectName("dialogTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:700; color:#7C4DFF; padding:8px;")
        root.addWidget(title)

        # Step indicators
        self.step_bar = QHBoxLayout()
        self.step_labels = []
        for i, txt in enumerate(["① Define Sign", "② Record Video", "③ Train Model", "④ Test Sign"]):
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "font-size:13px; font-weight:600; padding:8px 16px;"
                "border-radius:8px; background:#1A2332; color:#8899AA;")
            self.step_labels.append(lbl)
            self.step_bar.addWidget(lbl)
        root.addLayout(self.step_bar)

        # Stacked pages
        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_define())
        self.stack.addWidget(self._page_record())
        self.stack.addWidget(self._page_train())
        self.stack.addWidget(self._page_test())
        root.addWidget(self.stack, stretch=1)

        # Navigation buttons
        nav = QHBoxLayout()
        self.btn_back = QPushButton("← Back")
        self.btn_back.setObjectName("addSignButton")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setVisible(False)
        nav.addWidget(self.btn_back)
        nav.addStretch()
        self.btn_next = QPushButton("Next →")
        self.btn_next.setObjectName("addSignButton")
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        self._highlight_step(0)

    # ── Page 1: Define ──
    def _page_define(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)

        grp = QGroupBox("Sign Details")
        gl = QVBoxLayout(grp)

        gl.addWidget(QLabel("Sign Label (slug, lowercase, e.g. 'ndumva'):"))
        self.inp_label = QLineEdit()
        self.inp_label.setPlaceholderText("ndumva")
        self.inp_label.setStyleSheet(
            "background:#1A2332; color:#E8ECF1; padding:10px;"
            "border:1px solid #2A3A4A; border-radius:8px; font-size:14px;")
        gl.addWidget(self.inp_label)

        gl.addWidget(QLabel("Kinyarwanda Word (spoken text):"))
        self.inp_kiny = QLineEdit()
        self.inp_kiny.setPlaceholderText("Ndumva")
        self.inp_kiny.setStyleSheet(self.inp_label.styleSheet())
        gl.addWidget(self.inp_kiny)

        gl.addWidget(QLabel("English Meaning:"))
        self.inp_eng = QLineEdit()
        self.inp_eng.setPlaceholderText("I understand")
        self.inp_eng.setStyleSheet(self.inp_label.styleSheet())
        gl.addWidget(self.inp_eng)

        gl.addWidget(QLabel("Category:"))
        self.inp_cat = QComboBox()
        self.inp_cat.addItems(["greetings", "common", "emergency", "body", "alphabet", "other"])
        self.inp_cat.setStyleSheet(
            "QComboBox{background:#1A2332; color:#E8ECF1; padding:10px;"
            "border:1px solid #2A3A4A; border-radius:8px; font-size:14px;}")
        gl.addWidget(self.inp_cat)
        lay.addWidget(grp)

        # Custom Voice recording section
        voice_grp = QGroupBox("Custom Voice Recording (Optional)")
        vl = QVBoxLayout(voice_grp)
        
        self.lbl_voice_status = QLabel("No custom voice recorded (will use AI TTS)")
        self.lbl_voice_status.setStyleSheet("color: #8899AA;")
        vl.addWidget(self.lbl_voice_status)
        
        self.voice_progress = QProgressBar()
        self.voice_progress.setRange(0, 100)
        self.voice_progress.setValue(0)
        self.voice_progress.setTextVisible(False)
        self.voice_progress.setFixedHeight(6)
        self.voice_progress.setStyleSheet(
            "QProgressBar { background-color: #0F1419; border: 1px solid #2A3A4A; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #00D4AA; border-radius: 3px; }"
        )
        vl.addWidget(self.voice_progress)
        
        voice_btn_layout = QHBoxLayout()
        self.btn_voice_record = QPushButton("🎙️ Record Voice")
        self.btn_voice_record.clicked.connect(self._toggle_voice_recording)
        voice_btn_layout.addWidget(self.btn_voice_record)
        
        self.btn_voice_play = QPushButton("▶️ Play")
        self.btn_voice_play.clicked.connect(self._play_voice_recording)
        self.btn_voice_play.setEnabled(False)
        voice_btn_layout.addWidget(self.btn_voice_play)
        
        vl.addLayout(voice_btn_layout)
        lay.addWidget(voice_grp)

        lay.addStretch()
        return page

    # ── Page 2: Record ──
    def _page_record(self):
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setSpacing(12)

        # Camera preview
        cam_box = QVBoxLayout()
        self.cam_label = QLabel("Camera will start when you reach this step")
        self.cam_label.setMinimumSize(480, 360)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet(
            "background:#1A2332; border:2px solid #2A3A4A; border-radius:12px; color:#8899AA;")
        cam_box.addWidget(self.cam_label)
        self.rec_status = QLabel("Ready")
        self.rec_status.setStyleSheet("color:#00D4AA; font-weight:bold; font-size:13px;")
        cam_box.addWidget(self.rec_status)
        lay.addLayout(cam_box, stretch=2)

        # Controls
        ctrl = QVBoxLayout()
        ctrl.addWidget(QLabel("Duration (seconds):"))
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(2, 10)
        self.dur_spin.setValue(3)
        self.dur_spin.setStyleSheet(
            "QSpinBox{background:#1A2332; color:#E8ECF1; padding:8px;"
            "border:1px solid #2A3A4A; border-radius:8px;}")
        ctrl.addWidget(self.dur_spin)

        self.rec_progress = QProgressBar()
        self.rec_progress.setRange(0, 100)
        self.rec_progress.setValue(0)
        ctrl.addWidget(self.rec_progress)

        self.rec_btn = QPushButton("🔴 Record Clip")
        self.rec_btn.setStyleSheet(
            "QPushButton{background:#FF4444; color:white; font-weight:bold;"
            "padding:12px; border:none; border-radius:8px; font-size:14px;}"
            "QPushButton:hover{background:#FF6666;}")
        self.rec_btn.clicked.connect(self._toggle_record)
        ctrl.addWidget(self.rec_btn)

        self.clips_label = QLabel("📁 0 clip(s) recorded")
        self.clips_label.setAlignment(Qt.AlignCenter)
        self.clips_label.setStyleSheet("font-size:13px; padding:8px;")
        ctrl.addWidget(self.clips_label)

        ctrl.addStretch()
        hint = QLabel("Record at least 1 clip.\nMore clips = better accuracy.\n3–5 clips recommended.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8899AA; padding:8px; background:#1A2332; border-radius:8px;")
        ctrl.addWidget(hint)
        lay.addLayout(ctrl, stretch=1)

        # Camera timer
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._cam_tick)

        return page

    # ── Page 3: Train ──
    def _page_train(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)

        self.train_btn = QPushButton("🚀 Extract Landmarks & Train LSTM")
        self.train_btn.setStyleSheet(
            "QPushButton{background:#7C4DFF; color:white; font-weight:bold;"
            "padding:14px; border:none; border-radius:10px; font-size:15px;}"
            "QPushButton:hover{background:#9C7CFF;}"
            "QPushButton:disabled{background:#3A3A5A; color:#888;}")
        self.train_btn.clicked.connect(self._start_training)
        lay.addWidget(self.train_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 0)  # indeterminate
        self.train_progress.setVisible(False)
        lay.addWidget(self.train_progress)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet(
            "background:#0F1419; color:#00D4AA; border:1px solid #2A3A4A;"
            "border-radius:8px; padding:8px; font-family:monospace; font-size:12px;")
        lay.addWidget(self.train_log, stretch=1)

        return page

    # ── Page 4: Test ──
    def _page_test(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(12)

        # Instruction
        instr = QLabel(
            "Perform the sign in front of the camera to verify it is recognized correctly.")
        instr.setWordWrap(True)
        instr.setAlignment(Qt.AlignCenter)
        instr.setStyleSheet("color:#8899AA; font-size:13px; padding:4px;")
        lay.addWidget(instr)

        # Main body: camera + result side-by-side
        body = QHBoxLayout()
        body.setSpacing(12)

        # Camera preview
        cam_box = QVBoxLayout()
        self.test_cam_label = QLabel("Camera will start when you reach this step")
        self.test_cam_label.setMinimumSize(420, 320)
        self.test_cam_label.setAlignment(Qt.AlignCenter)
        self.test_cam_label.setStyleSheet(
            "background:#1A2332; border:2px solid #2A3A4A; border-radius:12px; color:#8899AA;")
        cam_box.addWidget(self.test_cam_label)

        # LSTM buffer fill bar
        self.test_buffer_bar = QProgressBar()
        self.test_buffer_bar.setRange(0, 100)
        self.test_buffer_bar.setValue(0)
        self.test_buffer_bar.setFormat("Buffer: %p%")
        self.test_buffer_bar.setFixedHeight(14)
        self.test_buffer_bar.setStyleSheet(
            "QProgressBar { border-radius:4px; background:#0F1419; text-align:center;"
            " color:#FFFFFF; font-size:10px; }"
            "QProgressBar::chunk { background:#7C4DFF; border-radius:4px; }")
        cam_box.addWidget(self.test_buffer_bar)
        body.addLayout(cam_box, stretch=2)

        # Result panel
        result_box = QVBoxLayout()

        # Big status icon
        self.test_icon_label = QLabel("🔍")
        self.test_icon_label.setAlignment(Qt.AlignCenter)
        self.test_icon_label.setStyleSheet("font-size:64px; padding:8px;")
        result_box.addWidget(self.test_icon_label)

        # Status text
        self.test_status_label = QLabel("Waiting for sign...")
        self.test_status_label.setAlignment(Qt.AlignCenter)
        self.test_status_label.setWordWrap(True)
        self.test_status_label.setStyleSheet(
            "font-size:16px; font-weight:700; color:#8899AA; padding:4px;")
        result_box.addWidget(self.test_status_label)

        # Detected sign
        self.test_detected_label = QLabel("—")
        self.test_detected_label.setAlignment(Qt.AlignCenter)
        self.test_detected_label.setWordWrap(True)
        self.test_detected_label.setStyleSheet(
            "font-size:20px; font-weight:700; color:#E8ECF1; padding:8px;"
            "background:#1A2332; border-radius:10px;")
        result_box.addWidget(self.test_detected_label)

        # Confidence bar
        self.test_confidence_bar = QProgressBar()
        self.test_confidence_bar.setRange(0, 100)
        self.test_confidence_bar.setValue(0)
        self.test_confidence_bar.setFormat("Confidence: 0%")
        self.test_confidence_bar.setStyleSheet(
            "QProgressBar { border-radius:6px; background:#0F1419; text-align:center;"
            " color:#FFFFFF; font-weight:600; font-size:11px; }"
            "QProgressBar::chunk { background:#00D4AA; border-radius:6px; }")
        result_box.addWidget(self.test_confidence_bar)

        # Match counter
        self.test_match_label = QLabel("Matches: 0 / 0")
        self.test_match_label.setAlignment(Qt.AlignCenter)
        self.test_match_label.setStyleSheet(
            "font-size:13px; font-weight:600; color:#8899AA; padding:8px;")
        result_box.addWidget(self.test_match_label)

        result_box.addStretch()

        # Close button
        self.test_close_btn = QPushButton("✅ Done — Close")
        self.test_close_btn.setStyleSheet(
            "QPushButton{background:#00D4AA; color:#0F1419; font-weight:bold;"
            "padding:14px; border:none; border-radius:10px; font-size:15px;}"
            "QPushButton:hover{background:#00F5C8;}")
        self.test_close_btn.clicked.connect(self.accept)
        result_box.addWidget(self.test_close_btn)

        body.addLayout(result_box, stretch=1)
        lay.addLayout(body, stretch=1)

        # Test camera timer
        self.test_cam_timer = QTimer(self)
        self.test_cam_timer.timeout.connect(self._test_cam_tick)

        return page

    # ── Navigation ──
    def _highlight_step(self, idx):
        for i, lbl in enumerate(self.step_labels):
            if i == idx:
                lbl.setStyleSheet(
                    "font-size:13px; font-weight:700; padding:8px 16px;"
                    "border-radius:8px; background:#7C4DFF; color:white;")
            elif i < idx:
                lbl.setStyleSheet(
                    "font-size:13px; font-weight:600; padding:8px 16px;"
                    "border-radius:8px; background:#1E2D3D; color:#00D4AA;")
            else:
                lbl.setStyleSheet(
                    "font-size:13px; font-weight:600; padding:8px 16px;"
                    "border-radius:8px; background:#1A2332; color:#8899AA;")

    def _go_next(self):
        cur = self.stack.currentIndex()
        if cur == 0:
            if not self._validate_step1():
                return
            self._register_sign()
            self._start_camera()
            self.stack.setCurrentIndex(1)
            self.btn_back.setVisible(True)
            self.btn_next.setText("Next →")
        elif cur == 1:
            if self.clips_recorded < 1:
                QMessageBox.warning(self, "No Clips", "Record at least 1 video clip first.")
                return
            self._stop_camera()
            self.stack.setCurrentIndex(2)
            self.btn_next.setVisible(False)
        elif cur == 2:
            # Train → Test
            self._start_test()
            self.stack.setCurrentIndex(3)
            self.btn_back.setVisible(True)
            self.btn_next.setVisible(False)
        elif cur == 3:
            # Test page → close dialog
            self.accept()
            return
        self._highlight_step(self.stack.currentIndex())

    def _go_back(self):
        cur = self.stack.currentIndex()
        if cur == 1:
            self._stop_camera()
            self.stack.setCurrentIndex(0)
            self.btn_back.setVisible(False)
        elif cur == 2:
            self._start_camera()
            self.stack.setCurrentIndex(1)
            self.btn_next.setVisible(True)
        elif cur == 3:
            self._stop_test_camera()
            self.stack.setCurrentIndex(2)
            self.btn_next.setVisible(False)
        self._highlight_step(self.stack.currentIndex())

    def _validate_step1(self):
        label = self._current_label()
        kiny = self.inp_kiny.text().strip()
        if not label:
            QMessageBox.warning(self, "Missing Label", "Please enter a sign label.")
            return False
        if not kiny:
            QMessageBox.warning(self, "Missing Word", "Please enter the Kinyarwanda word.")
            return False
        if label in self.vocabulary.signs:
            QMessageBox.warning(self, "Duplicate", f"Sign '{label}' already exists.")
            return False
        # Reflect the sanitized slug back into the field so the user sees
        # exactly what will be used for the video folder / audio filename.
        self.inp_label.setText(label)
        return True

    def _register_sign(self):
        label = self._current_label()
        kiny = self.inp_kiny.text().strip()
        eng = self.inp_eng.text().strip()
        cat = self.inp_cat.currentText()
        self.vocabulary.add_sign(label, kiny, eng, cat)

    # ── Camera ──
    def _start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            self.cam_timer.start(33)
        else:
            self.cam_label.setText("❌ Camera failed")

    def _stop_camera(self):
        self.cam_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _cam_tick(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.is_recording:
            self.writer.write(frame)
            elapsed = time.time() - self.record_start
            pct = min(100, int((elapsed / self.dur_spin.value()) * 100))
            self.rec_progress.setValue(pct)
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {elapsed:.1f}s", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed >= self.dur_spin.value():
                self._stop_record()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi).scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.cam_label.setPixmap(pm)

    def _toggle_record(self):
        if self.is_recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        label = self._current_label()
        out_dir = os.path.join(VIDEO_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{label}_{ts}.mp4")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        self.record_start = time.time()
        self.is_recording = True
        self.rec_btn.setText("⏹ Stop")
        self.rec_status.setText(f"🔴 Recording '{label}'...")
        self.rec_status.setStyleSheet("color:#FF4444; font-weight:bold; font-size:13px;")

    def _stop_record(self):
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        self.clips_recorded += 1
        self.rec_btn.setText("🔴 Record Clip")
        self.rec_status.setText("✅ Clip saved!")
        self.rec_status.setStyleSheet("color:#00D4AA; font-weight:bold; font-size:13px;")
        self.rec_progress.setValue(100)
        self.clips_label.setText(f"📁 {self.clips_recorded} clip(s) recorded")

    # ── Training ──
    def _start_training(self):
        label = self._current_label()
        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_log.clear()
        self.btn_back.setEnabled(False)

        self._worker = TrainWorker(label)
        self._worker.log.connect(self._on_train_log)
        self._worker.finished.connect(self._on_train_done)
        self._worker.start()

    def _on_train_log(self, msg):
        self.train_log.append(msg)

    def _on_train_done(self, success, message):
        self.train_progress.setVisible(False)
        self.btn_back.setEnabled(True)
        if success:
            self.train_log.append(f"\n🎉 {message}")
            self.train_log.append("Click 'Test Sign →' to verify recognition, or close the dialog.")
            self.training_complete.emit()
            # Show Next button to go to test step
            self.btn_next.setVisible(True)
            self.btn_next.setText("Test Sign →")
            # Also change train button to allow skipping the test
            self.train_btn.setText("⏩ Skip Test — Close")
            self.train_btn.setEnabled(True)
            self.train_btn.setStyleSheet(
                "QPushButton{background:#2A3A4A; color:#8899AA; font-weight:bold;"
                "padding:14px; border:1px solid #3A4A5A; border-radius:10px; font-size:15px;}"
                "QPushButton:hover{background:#3A4A5A; color:#E8ECF1;}")
            self.train_btn.clicked.disconnect()
            self.train_btn.clicked.connect(self.accept)
        else:
            self.train_log.append(f"\n❌ Error: {message}")
            self.train_btn.setEnabled(True)

    # ── Test step: camera + live LSTM inference ──
    def _start_test(self):
        """Initialize the test step: load fresh LSTM model and start camera."""
        self._test_match_count = 0
        self._test_total_preds = 0
        self._test_buffer.clear()

        # Reset UI
        self.test_icon_label.setText("🔍")
        self.test_status_label.setText("Waiting for sign...")
        self.test_status_label.setStyleSheet(
            "font-size:16px; font-weight:700; color:#8899AA; padding:4px;")
        self.test_detected_label.setText("—")
        self.test_confidence_bar.setValue(0)
        self.test_confidence_bar.setFormat("Confidence: 0%")
        self.test_match_label.setText("Matches: 0 / 0")
        self.test_buffer_bar.setValue(0)

        # Load a fresh detector for the test camera
        try:
            from src.detector import HandDetector
            self._test_detector = HandDetector()
        except Exception as e:
            logger.error(f"[AddSign/Test] Detector init error: {e}")
            self._test_detector = None

        # Load the freshly trained LSTM model
        try:
            from src.classifier import LSTMClassifier
            self._test_lstm = LSTMClassifier(
                model_path="models/ksl_lstm_model.h5",
                confidence_threshold=TEST_CONFIDENCE_THRESHOLD)
            if self._test_lstm.is_ready():
                logger.info(f"[AddSign/Test] LSTM loaded for test. Classes: {self._test_lstm.labels}")
            else:
                logger.warning("[AddSign/Test] Warning: LSTM not ready after loading.")
        except Exception as e:
            logger.error(f"[AddSign/Test] LSTM load error: {e}")
            self._test_lstm = None

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            self.test_cam_timer.start(33)
        else:
            self.test_cam_label.setText("❌ Camera failed")

    def _stop_test_camera(self):
        """Stop the test camera and clean up test resources."""
        self.test_cam_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._test_detector:
            self._test_detector.close()
            self._test_detector = None
        self._test_lstm = None
        self._test_buffer.clear()

    def _test_cam_tick(self):
        """Process one camera frame during the test step."""
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        label = self._current_label()
        annotated = frame

        # Run hand detection
        if self._test_detector:
            annotated, lm_list = self._test_detector.process_frame(frame)

            if lm_list and self._test_lstm and self._test_lstm.is_ready():
                landmarks = lm_list[0]
                self._test_lstm.push_frame(landmarks)
                fill = self._test_lstm.buffer_fill()
                self.test_buffer_bar.setValue(int(fill * 100))

                # Try prediction when buffer is full
                if fill >= 1.0:
                    pred_label, confidence = self._test_lstm.predict()
                    self._test_lstm.slide_buffer()

                    if pred_label and confidence >= TEST_CONFIDENCE_THRESHOLD:
                        self._test_total_preds += 1
                        pred_label = str(pred_label)
                        conf_pct = int(confidence * 100)
                        self.test_confidence_bar.setValue(conf_pct)
                        self.test_confidence_bar.setFormat(f"Confidence: {conf_pct}%")

                        kiny = self.vocabulary.get_kinyarwanda(pred_label)
                        eng = self.vocabulary.get_english(pred_label)
                        display = f"{kiny} ({eng})" if eng else kiny
                        self.test_detected_label.setText(display)

                        if pred_label == label:
                            # ✅ Correct match
                            self._test_match_count += 1
                            self.test_icon_label.setText("✅")
                            self.test_status_label.setText("MATCH — Sign recognized correctly!")
                            self.test_status_label.setStyleSheet(
                                "font-size:16px; font-weight:700; color:#00D4AA; padding:4px;")
                            self.test_detected_label.setStyleSheet(
                                "font-size:20px; font-weight:700; color:#00D4AA; padding:8px;"
                                "background:#0D2818; border:2px solid #00D4AA; border-radius:10px;")
                        else:
                            # ❌ Wrong sign detected
                            self.test_icon_label.setText("❌")
                            self.test_status_label.setText(
                                f"Detected '{pred_label}' instead of '{label}'")
                            self.test_status_label.setStyleSheet(
                                "font-size:16px; font-weight:700; color:#FF6B6B; padding:4px;")
                            self.test_detected_label.setStyleSheet(
                                "font-size:20px; font-weight:700; color:#FF6B6B; padding:8px;"
                                "background:#2D1515; border:2px solid #FF6B6B; border-radius:10px;")

                        self.test_match_label.setText(
                            f"Matches: {self._test_match_count} / {self._test_total_preds}")
            elif not lm_list:
                # No hand visible — show waiting state
                self.test_icon_label.setText("🔍")
                self.test_status_label.setText("No hand detected — show your sign")
                self.test_status_label.setStyleSheet(
                    "font-size:16px; font-weight:700; color:#8899AA; padding:4px;")

        # Display frame
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi).scaled(
            self.test_cam_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.test_cam_label.setPixmap(pm)

    def _toggle_voice_recording(self):
        if self.voice_is_recording:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()

    def _start_voice_recording(self):
        label = self._current_label()
        if not label:
            QMessageBox.warning(self, "Missing Label", "Please enter a sign label before recording your voice.")
            return
        self.inp_label.setText(label)
        
        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass
            
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/x-raw")
        settings.setSampleRate(22050)
        settings.setChannelCount(1)
        settings.setQuality(QMultimedia.NormalQuality)
        settings.setEncodingMode(QMultimedia.ConstantQualityEncoding)
        
        self.voice_recorder.setEncodingSettings(settings)
        self.voice_recorder.setContainerFormat("audio/x-wav")
        
        audio_dir = "data/audio"
        os.makedirs(audio_dir, exist_ok=True)
        output_file = os.path.join(audio_dir, f"{label}.wav")
        
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except Exception:
                pass
                
        inputs = self.voice_recorder.audioInputs()
        if inputs:
            default_inp = "default:" if "default:" in inputs else inputs[0]
            self.voice_recorder.setAudioInput(default_inp)
            
        self.voice_recorder.setOutputLocation(QUrl.fromLocalFile(output_file))
        
        self.voice_recorder.record()
        self.voice_is_recording = True
        self.btn_voice_record.setText("⏹️ Stop")
        self.btn_voice_record.setStyleSheet("background-color: #FFD93D; color: #0F1419; font-weight: bold;")
        self.btn_voice_play.setEnabled(False)
        
        self.voice_elapsed_start = time.time()
        self.voice_progress.setValue(0)
        self.lbl_voice_status.setText("Recording... Speak now!")
        self.lbl_voice_status.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        
        self.voice_record_timer.start(3000)
        self.voice_elapsed_timer.start(50)

    def _stop_voice_recording(self):
        if not self.voice_is_recording:
            return
            
        self.voice_is_recording = False
        self.voice_record_timer.stop()
        self.voice_elapsed_timer.stop()

        if self.voice_recorder.state() == QAudioRecorder.RecordingState:
            self.voice_recorder.stop()

        time.sleep(0.2)

        self.btn_voice_record.setText("🎙️ Record Voice")
        self.btn_voice_record.setStyleSheet("")
        self.voice_progress.setValue(100)

        label = self._current_label()
        output_file = os.path.join("data/audio", f"{label}.wav")
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            self.lbl_voice_status.setText("Custom voice recorded successfully!")
            self.lbl_voice_status.setStyleSheet("color: #00D4AA; font-weight: bold;")
            self.btn_voice_play.setEnabled(True)
        else:
            self.lbl_voice_status.setText("Recording failed or empty.")
            self.lbl_voice_status.setStyleSheet("color: #FF6B6B;")
            self.btn_voice_play.setEnabled(False)

    def _on_voice_elapsed_tick(self):
        elapsed = time.time() - self.voice_elapsed_start
        if elapsed >= 3.0:
            self._stop_voice_recording()
            return
        pct = int((elapsed / 3.0) * 100)
        self.voice_progress.setValue(pct)
        self.lbl_voice_status.setText(f"Recording: {elapsed:.1f}s / 3.0s — Speak now!")

    def _play_voice_recording(self):
        label = self._current_label()
        output_file = os.path.join("data/audio", f"{label}.wav")
        if os.path.exists(output_file):
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=1)
                pygame.mixer.stop()
                sound = pygame.mixer.Sound(output_file)
                sound.play()
            except Exception as e:
                logger.error(f"[AddSignDialog] Voice preview error: {e}")

    def closeEvent(self, ev):
        self._stop_camera()
        self._stop_test_camera()
        self._stop_voice_recording()
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
        ev.accept()
