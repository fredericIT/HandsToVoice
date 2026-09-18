"""
HandsToVoice — Loading Screen
Shown the instant "Get Started" is clicked, while the camera, hand
detector, and models load on a background thread. Without this, the
welcome window would appear to freeze for several seconds with no
feedback while those components initialize on the main thread.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.gui.animations import EqualizerBarsWidget


class LoadingScreen(QWidget):
    """Minimal, animated "please wait" screen shown during backend init."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandsToVoice — Loading")
        self.resize(420, 320)
        self.setStyleSheet("background-color: #0F1419;")
        self._equalizer = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(18)

        title = QLabel("HandsToVoice")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Rubik", 22, QFont.Bold))
        title.setStyleSheet("color:#00D4AA; background:transparent;")
        layout.addWidget(title)

        self._equalizer = EqualizerBarsWidget(bar_count=7, min_h=8, max_h=40)
        self._equalizer.setFixedHeight(46)
        layout.addWidget(self._equalizer, alignment=Qt.AlignCenter)

        status = QLabel("Starting camera, hand detector, and voice…")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet(
            "font-family:'Inter'; font-size:13px; color:#8899AA; background:transparent;"
        )
        layout.addWidget(status)

    def closeEvent(self, event):
        if self._equalizer:
            self._equalizer.stop()
        event.accept()
