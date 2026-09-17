"""
HandsToVoice — Camera Widget
QWidget that displays live OpenCV frames in the PyQt5 GUI.
"""

import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class CameraWidget(QWidget):
    """Displays live webcam feed with hand landmark overlay."""

    # Emitted when a new frame with landmarks is ready
    landmarks_detected = pyqtSignal(object)   # list of landmark arrays

    def __init__(self, camera, detector, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.detector = detector
        self._running = False
        self._frame_count = 0
        self._fps = 0.0
        self._setup_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._tick_fps)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setObjectName("cameraPlaceholder")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("📷  Camera Off")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.image_label.setStyleSheet(
            "QLabel { background-color:#1A2332; border-radius:16px;"
            " border:2px solid #2A3A4A; }"
        )
        layout.addWidget(self.image_label)

    def start(self):
        if self._running:
            return
        try:
            if not self.camera.is_opened():
                self.camera.open()
            self._running = True
            self.timer.start(33)          # ~30 FPS
            self._fps_timer.start(1000)   # update FPS every second
        except Exception as e:
            self.image_label.setText(f"❌  Camera Error:\n{e}")

    def stop(self):
        self._running = False
        self.timer.stop()
        self._fps_timer.stop()
        self.image_label.setText("📷  Camera Off")
        self.image_label.setPixmap(QPixmap())

    def _update_frame(self):
        if not self._running:
            return
        frame = self.camera.read_frame()
        if frame is None:
            return

        annotated, landmarks_list = self.detector.process_frame(frame)

        # Overlay FPS
        cv2.putText(annotated, f"FPS: {self._fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 212, 170), 2)
        cv2.putText(annotated, f"Hands: {len(landmarks_list)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (136, 153, 170), 2)

        # BGR → RGB → QPixmap
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,   # cheaper than SmoothTransformation at 30fps
        )
        self.image_label.setPixmap(pixmap)

        if landmarks_list:
            self.landmarks_detected.emit(landmarks_list)

        self._frame_count += 1

    def _tick_fps(self):
        self._fps = self._frame_count
        self._frame_count = 0

    def get_fps(self):
        return self._fps

    def is_running(self):
        return self._running
