"""
HandsToVoice — Learn a Sign Dialog
Lets someone who doesn't know a sign look it up by word (the sign's slug,
its Kinyarwanda word, or its English meaning) and watch one of the
reference video clips that was recorded for it during the "Add Sign"
wizard — so they can see exactly how to perform it.
"""

import os
import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from src.logger import get_logger

logger = get_logger("gui.learn_sign_dialog")

VIDEO_DIR = "data/videos"
SUPPORTED_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class LearnSignDialog(QDialog):
    """Search the vocabulary by word and watch a reference video of the sign."""

    def __init__(self, vocabulary, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.cap = None
        self.current_video_path = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)

        self.setWindowTitle("HandsToVoice — Learn a Sign")
        self.setMinimumSize(700, 560)
        self.setModal(True)
        self._build_ui()
        self._populate_results("")

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("📖 Learn a Sign")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:700; color:#00D4AA; padding:8px;")
        root.addWidget(title)

        hint = QLabel(
            "Type a word — the sign's name, its Kinyarwanda word, or its "
            "English meaning — to watch how it's performed.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8899AA; font-size:13px;")
        root.addWidget(hint)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("e.g. 'hello', 'muraho', 'good'...")
        self.search_box.textChanged.connect(self._populate_results)
        root.addWidget(self.search_box)

        body = QHBoxLayout()
        body.setSpacing(16)

        results_group = QGroupBox("Matching Signs")
        rl = QVBoxLayout(results_group)
        self.results_list = QListWidget()
        self.results_list.currentItemChanged.connect(self._on_selection_changed)
        rl.addWidget(self.results_list)
        body.addWidget(results_group, stretch=1)

        video_group = QGroupBox("Reference Video")
        vl = QVBoxLayout(video_group)
        self.video_label = QLabel("Select a sign to watch how it's performed.")
        self.video_label.setMinimumSize(360, 270)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet(
            "background:#1A2332; border:2px dashed #2A3A4A; border-radius:12px; "
            "color:#8899AA; padding:12px;")
        vl.addWidget(self.video_label, stretch=1)

        self.sign_info_label = QLabel("")
        self.sign_info_label.setWordWrap(True)
        self.sign_info_label.setStyleSheet("color:#E8ECF1; font-size:14px; padding:4px;")
        vl.addWidget(self.sign_info_label)

        self.replay_btn = QPushButton("🔁 Replay")
        self.replay_btn.clicked.connect(self._play_current_video)
        self.replay_btn.setEnabled(False)
        vl.addWidget(self.replay_btn)

        body.addWidget(video_group, stretch=1)
        root.addLayout(body, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    # ── Search ───────────────────────────────────────────────────────────────
    def _populate_results(self, query):
        self.results_list.clear()
        query = query.strip().lower()

        for label in self.vocabulary.get_all_labels():
            info = self.vocabulary.signs.get(label, {})
            kiny = str(info.get("kinyarwanda", "")).lower()
            eng = str(info.get("english", "")).lower()
            if not query or query in label or query in kiny or query in eng:
                item = QListWidgetItem(self.vocabulary.get_display_text(label))
                item.setData(Qt.UserRole, label)
                self.results_list.addItem(item)

        if self.results_list.count() == 0:
            item = QListWidgetItem("No matching sign — it may not be trained yet.")
            item.setFlags(Qt.NoItemFlags)
            self.results_list.addItem(item)

    # ── Selection / playback ─────────────────────────────────────────────────
    def _on_selection_changed(self, current, _previous):
        self._stop_video()
        self.current_video_path = None
        self.replay_btn.setEnabled(False)

        if current is None:
            return
        label = current.data(Qt.UserRole)
        if not label:
            self.sign_info_label.setText("")
            self.video_label.setText("Select a sign to watch how it's performed.")
            return

        info = self.vocabulary.signs.get(label, {})
        kiny = info.get("kinyarwanda", label)
        eng = info.get("english", "")
        self.sign_info_label.setText(f"<b>{kiny}</b>" + (f"  —  {eng}" if eng else ""))

        video_path = self._find_video(label)
        if video_path:
            self.current_video_path = video_path
            self.replay_btn.setEnabled(True)
            self._play_current_video()
        else:
            self.video_label.setText(
                "No reference video was recorded for this sign yet.")

    def _find_video(self, label):
        """Return the path to a recorded clip for this sign, or None."""
        sign_dir = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(sign_dir):
            return None
        videos = sorted(
            f for f in os.listdir(sign_dir)
            if f.lower().endswith(SUPPORTED_EXT)
        )
        if not videos:
            return None
        return os.path.join(sign_dir, videos[0])

    def _play_current_video(self):
        if not self.current_video_path:
            return
        self._stop_video()
        self.cap = cv2.VideoCapture(self.current_video_path)
        if not self.cap.isOpened():
            logger.warning(f"[LearnSign] Could not open video: {self.current_video_path}")
            self.video_label.setText("Could not open the reference video.")
            self.cap = None
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval_ms = max(15, int(1000 / fps))
        self.play_timer.start(interval_ms)

    def _next_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            # Loop the clip so the user can watch it repeatedly.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self._stop_video()
                return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pm)

    def _stop_video(self):
        self.play_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop_video()
        event.accept()

    def reject(self):
        self._stop_video()
        super().reject()
