#!/usr/bin/env python3
"""
HandsToVoice — Video Data Collection GUI
Records short video clips of each sign via webcam and saves them
in the correct folder structure for training.
"""

import os, sys, time, argparse
from datetime import datetime
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QSpinBox, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.vocabulary import Vocabulary
from src.gui.styles import get_stylesheet

from src.logger import get_logger

logger = get_logger("collect_video")

VIDEO_DIR = "data/videos"


class VideoRecorderGUI(QMainWindow):
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.vocabulary = Vocabulary()
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.record_start = 0
        self.record_duration = 3
        self.frames_written = 0
        self._setup_ui()
        self._open_camera()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _setup_ui(self):
        self.setWindowTitle("HandsToVoice — Video Recorder")
        self.setGeometry(100, 100, 1000, 700)
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QHBoxLayout(cw); ml.setContentsMargins(16,16,16,16); ml.setSpacing(16)

        # Camera
        cam_box = QGroupBox("📷 Camera Preview")
        cam_box.setStyleSheet("QGroupBox{font-weight:bold;font-size:14px;padding-top:10px;}")
        cl = QVBoxLayout(cam_box)
        self.cam_label = QLabel("Starting camera...")
        self.cam_label.setMinimumSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background:#1A2332;border:2px solid #2A3A4A;border-radius:12px;")
        cl.addWidget(self.cam_label)
        self.status_lbl = QLabel("Ready"); self.status_lbl.setStyleSheet("color:#00D4AA;font-weight:bold;")
        cl.addWidget(self.status_lbl)
        ml.addWidget(cam_box, stretch=2)

        # Controls
        ctrl_box = QGroupBox("🎬 Recording Controls")
        ctrl_box.setStyleSheet("QGroupBox{font-weight:bold;font-size:14px;padding-top:10px;}")
        ctl = QVBoxLayout(ctrl_box)

        ctl.addWidget(QLabel("Select Sign:"))
        self.sign_combo = QComboBox()
        self.sign_combo.addItems(self.vocabulary.get_all_labels())
        self.sign_combo.setStyleSheet("QComboBox{background:#1A2332;color:#E8ECF1;padding:8px;border:1px solid #2A3A4A;border-radius:8px;}")
        ctl.addWidget(self.sign_combo)

        ctl.addWidget(QLabel("Duration (seconds):"))
        self.dur_spin = QSpinBox(); self.dur_spin.setRange(1, 10); self.dur_spin.setValue(3)
        self.dur_spin.setStyleSheet("QSpinBox{background:#1A2332;color:#E8ECF1;padding:8px;border:1px solid #2A3A4A;border-radius:8px;}")
        ctl.addWidget(self.dur_spin)

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        ctl.addWidget(self.progress)

        self.rec_btn = QPushButton("🔴 Record")
        self.rec_btn.setStyleSheet(
            "QPushButton{background:#FF4444;color:white;font-weight:bold;padding:12px;"
            "border:none;border-radius:8px;font-size:14px;}"
            "QPushButton:hover{background:#FF6666;}"
            "QPushButton:disabled{background:#4A5568;color:#888;}")
        self.rec_btn.clicked.connect(self._toggle_record)
        ctl.addWidget(self.rec_btn)

        self.count_lbl = QLabel("")
        self.count_lbl.setAlignment(Qt.AlignCenter)
        ctl.addWidget(self.count_lbl)
        self._update_counts()

        ctl.addStretch()
        instr = QLabel("1. Select a sign\n2. Set duration\n3. Click Record\n4. Perform the sign\n5. Repeat for more clips")
        instr.setStyleSheet("color:#8899AA;padding:8px;background:#1A2332;border-radius:8px;")
        instr.setWordWrap(True)
        ctl.addWidget(instr)

        ml.addWidget(ctrl_box, stretch=1)

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_lbl.setText("❌ Camera failed"); self.rec_btn.setEnabled(False)

    def _update_counts(self):
        sign = self.sign_combo.currentText()
        d = os.path.join(VIDEO_DIR, sign)
        n = len([f for f in os.listdir(d) if f.endswith(('.mp4','.avi'))]) if os.path.exists(d) else 0
        self.count_lbl.setText(f"📁 {n} video(s) saved for '{sign}'")

    def _toggle_record(self):
        if self.is_recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        sign = self.sign_combo.currentText()
        out_dir = os.path.join(VIDEO_DIR, sign)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{sign}_{ts}.mp4")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        self.record_duration = self.dur_spin.value()
        self.record_start = time.time()
        self.frames_written = 0
        self.is_recording = True
        self.rec_btn.setText("⏹ Stop")
        self.sign_combo.setEnabled(False)
        self.status_lbl.setText(f"🔴 Recording '{sign}'...")
        self.status_lbl.setStyleSheet("color:#FF4444;font-weight:bold;")

    def _stop_record(self):
        self.is_recording = False
        if self.writer:
            self.writer.release(); self.writer = None
        self.rec_btn.setText("🔴 Record")
        self.sign_combo.setEnabled(True)
        self.status_lbl.setText(f"✅ Saved! ({self.frames_written} frames)")
        self.status_lbl.setStyleSheet("color:#00D4AA;font-weight:bold;")
        self.progress.setValue(100)
        self._update_counts()

    def _tick(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.is_recording:
            self.writer.write(frame)
            self.frames_written += 1
            elapsed = time.time() - self.record_start
            pct = min(100, int((elapsed / self.record_duration) * 100))
            self.progress.setValue(pct)
            # Draw recording indicator
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {elapsed:.1f}s", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed >= self.record_duration:
                self._stop_record()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi).scaled(self.cam_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.cam_label.setPixmap(pm)

    def closeEvent(self, ev):
        if self.cap: self.cap.release()
        if self.writer: self.writer.release()
        ev.accept()


def main():
    parser = argparse.ArgumentParser(description="Record sign language video clips")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    print("=" * 60)
    logger.info("HandsToVoice — Video Recorder")
    print("=" * 60)
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())
    win = VideoRecorderGUI(camera_index=args.camera)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
