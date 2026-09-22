"""
HandsToVoice — Sign Popup
Shows the reference video of the sign for each word a hearing person just
said, one after another in the order spoken, then closes itself.
"""

import os
from collections import deque

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from src.logger import get_logger

logger = get_logger("gui.sign_popup")

VIDEO_DIR = "data/videos"
SUPPORTED_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
CLOSE_DELAY_MS = 1500     # how long the last sign stays up after it finishes
NO_VIDEO_MS = 2500        # how long a word with no video is shown as text


class SignPopup(QDialog):
    """Non-modal window that plays the sign video for each word heard."""

    def __init__(self, vocabulary, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.queue = deque()
        self.cap = None
        self.playing = False

        self.setWindowTitle("HandsToVoice — Heard")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)   # don't steal focus mid-signing
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.heard_lbl = QLabel("🎧 Heard")
        self.heard_lbl.setAlignment(Qt.AlignCenter)
        self.heard_lbl.setStyleSheet("color:#8899AA; font-size:12px;")
        layout.addWidget(self.heard_lbl)

        self.word_lbl = QLabel("")
        self.word_lbl.setAlignment(Qt.AlignCenter)
        self.word_lbl.setStyleSheet("font-size:30px; font-weight:700; color:#00D4AA;")
        layout.addWidget(self.word_lbl)

        self.meaning_lbl = QLabel("")
        self.meaning_lbl.setAlignment(Qt.AlignCenter)
        self.meaning_lbl.setStyleSheet("color:#8899AA; font-size:14px;")
        layout.addWidget(self.meaning_lbl)

        self.video_lbl = QLabel()
        self.video_lbl.setFixedSize(480, 360)
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background:#1A2332; border-radius:12px; color:#8899AA;")
        layout.addWidget(self.video_lbl, alignment=Qt.AlignCenter)

        self.next_lbl = QLabel("")
        self.next_lbl.setAlignment(Qt.AlignCenter)
        self.next_lbl.setStyleSheet("color:#7C4DFF; font-size:12px;")
        layout.addWidget(self.next_lbl)

        close = QPushButton("Close")
        close.setObjectName("neutralButton")
        close.clicked.connect(self.close_popup)
        layout.addWidget(close)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._next_frame)
        self.hold_timer = QTimer(self)
        self.hold_timer.setSingleShot(True)
        self.hold_timer.timeout.connect(self._advance)

    # ── public ──────────────────────────────────────────────────────────────
    def show_words(self, labels):
        """Queue signs to play, in spoken order. Words heard while one is
        playing are added to the end of the queue."""
        self.queue.extend(labels)
        if not self.playing:
            self._advance()

    def close_popup(self):
        self.queue.clear()
        self._stop()
        self.hide()

    # ── playback ────────────────────────────────────────────────────────────
    def _advance(self):
        self.hold_timer.stop()
        if not self.queue:
            self.playing = False
            self.hold_timer.singleShot(CLOSE_DELAY_MS, self._close_if_idle)
            return
        self.playing = True
        label = self.queue.popleft()
        info = self.vocabulary.signs.get(label, {})
        self.word_lbl.setText(info.get("kinyarwanda", label))
        self.meaning_lbl.setText(info.get("english", ""))
        remaining = [self.vocabulary.signs.get(l, {}).get("kinyarwanda", l) for l in self.queue]
        self.next_lbl.setText("Next: " + "  →  ".join(remaining) if remaining else "")
        if not self.isVisible():
            self.show()
        self._play(label)

    def _close_if_idle(self):
        if not self.playing and not self.queue:
            self.hide()

    def _find_video(self, label):
        folder = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(folder):
            return None
        videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXT))
        return os.path.join(folder, videos[0]) if videos else None

    def _play(self, label):
        self._stop()
        path = self._find_video(label)
        if path:
            self.cap = cv2.VideoCapture(path)
        if not path or not self.cap.isOpened():
            self.video_lbl.setPixmap(QPixmap())
            self.video_lbl.setText("No video recorded for this sign yet")
            self.hold_timer.start(NO_VIDEO_MS)
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_timer.start(int(1000 / min(max(fps, 10), 60)))

    def _next_frame(self):
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok:
            self._stop()
            self.hold_timer.start(600)      # brief pause, then the next sign
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_lbl.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _stop(self):
        self.frame_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def closeEvent(self, event):
        self.queue.clear()
        self._stop()
        event.accept()
