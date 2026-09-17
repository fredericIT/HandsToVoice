"""
HandsToVoice — Welcome / Cover Screen
The mandatory entry point shown before the main recognition window. Its
hero is a single animated illustration: two people hold a real
back-and-forth sign-language conversation, then the scene shows the app
watching (camera brackets), understanding (processing ring), and
speaking it aloud (voice bubble) — tying the illustration directly to
what HandsToVoice does. Users cannot reach the main window without
passing through this screen.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QMovie

from src.logger import get_logger

logger = get_logger("gui.welcome_screen")

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")


class WelcomeScreen(QWidget):
    """Mandatory cover page shown before the main app.

    Emits `continue_requested` when the user clicks "Get Started" — main.py
    only builds and shows the real MainWindow after that signal fires, so
    closing this screen any other way (e.g. the OS close button) simply
    exits the app instead of falling through to the recognition system.
    """

    continue_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.movie = None
        self._entrance_widgets = []   # (widget, opacity_effect) pairs, faded in on show
        self._entrance_anims = []     # kept alive for the duration of the animation
        self._pulse_anim = None
        self._entrance_played = False
        self.setWindowTitle("HandsToVoice — Welcome")
        self.resize(880, 760)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(22)

        title = QLabel("HandsToVoice")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:34px; font-weight:800; color:#00D4AA;")
        root.addWidget(title)
        self._register_entrance(title)

        subtitle = QLabel("Kinyarwanda Sign Language Recognition & Voice Conversion")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:14px; color:#8899AA;")
        root.addWidget(subtitle)
        self._register_entrance(subtitle)

        hero_card = self._build_hero_card()
        root.addWidget(hero_card)
        self._register_entrance(hero_card)

        root.addStretch()

        self.start_btn = QPushButton("Get Started  →")
        self.start_btn.setObjectName("speakButton")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self._on_start)
        root.addWidget(self.start_btn)
        self._register_entrance(self.start_btn)

    def _register_entrance(self, widget):
        """Attach a QGraphicsOpacityEffect starting at 0 so this widget can
        be faded in once the window actually appears (see showEvent)."""
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)
        self._entrance_widgets.append((widget, effect))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._entrance_played:
            self._entrance_played = True
            self._play_entrance()

    def _play_entrance(self):
        """Fade each section in in sequence — genuine Qt property animation,
        not the pre-rendered illustration — so the page itself feels alive
        as it appears. Starts the Get Started button's attention pulse once
        it has fully faded in.
        """
        stagger_ms = 140
        for i, (widget, effect) in enumerate(self._entrance_widgets):
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(480)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._entrance_anims.append(anim)
            QTimer.singleShot(i * stagger_ms, anim.start)

        total_delay = len(self._entrance_widgets) * stagger_ms + 480
        QTimer.singleShot(total_delay, self._start_cta_pulse)

    def _start_cta_pulse(self):
        """A gentle, continuous breathing glow on the CTA button — draws
        the eye without being distracting. Loops until the window closes."""
        effect = self.start_btn.graphicsEffect()
        if effect is None:
            return
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(1100)
        anim.setStartValue(1.0)
        anim.setKeyValueAt(0.5, 0.72)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)
        anim.setLoopCount(-1)
        anim.start()
        self._pulse_anim = anim

    def _build_hero_card(self):
        card = QFrame()
        card.setObjectName("cardFrame")
        layout = QVBoxLayout(card)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        tagline = QLabel("One shared language")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet("font-size:18px; font-weight:700; color:#7C4DFF;")
        layout.addWidget(tagline)

        anim_path = os.path.join(ASSETS_DIR, "cover_conversation.gif")
        anim_label = QLabel()
        anim_label.setFixedSize(520, 260)
        anim_label.setAlignment(Qt.AlignCenter)
        anim_label.setStyleSheet("background:#1A2332; border-radius:14px;")
        if os.path.exists(anim_path):
            self.movie = QMovie(anim_path)
            anim_label.setMovie(self.movie)
            self.movie.start()
        else:
            anim_label.setText("Illustration unavailable")
            anim_label.setStyleSheet(
                "background:#1A2332; border-radius:14px; color:#8899AA;")
        anim_row = QVBoxLayout()
        anim_row.setAlignment(Qt.AlignCenter)
        anim_row.addWidget(anim_label, alignment=Qt.AlignCenter)
        layout.addLayout(anim_row)

        story = QLabel(
            "Watch two people sign to each other — then watch HandsToVoice "
            "see it, understand it, and speak it aloud. Built for the Deaf, "
            "Hard-of-Hearing, and speech-impaired community, so every "
            "conversation reaches everyone in the room."
        )
        story.setWordWrap(True)
        story.setAlignment(Qt.AlignCenter)
        story.setStyleSheet("font-size:15px; color:#E8ECF1; padding:4px 16px;")
        layout.addWidget(story)

        return card

    def _on_start(self):
        logger.info("[WelcomeScreen] User continued into the app.")
        if self.movie:
            self.movie.stop()
        if self._pulse_anim:
            self._pulse_anim.stop()
        self.continue_requested.emit()
        self.close()

    def closeEvent(self, event):
        if self.movie:
            self.movie.stop()
        if self._pulse_anim:
            self._pulse_anim.stop()
        event.accept()
