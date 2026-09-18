"""
HandsToVoice — Welcome / Cover Screen
The mandatory entry point shown before the main recognition window.
Three animations tie the page directly to what HandsToVoice does:
  1. The hero illustration (cover_conversation.gif) — two people signing,
     the app watching, understanding, and speaking it aloud.
  2. A looping Sign -> Understand -> Speak pipeline chase.
  3. A voice-equalizer bar animation representing spoken output.
Users cannot reach the main window without passing through this screen.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QVariantAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QMovie, QFont, QPixmap

from src.gui.animations import PipelineFlowWidget, EqualizerBarsWidget
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
        self._pipeline_anim = None
        self._equalizer_anim = None
        self.setWindowTitle("HandsToVoice — Welcome")
        self.resize(920, 820)
        self.setStyleSheet("background-color: #0F1419;")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Logo, pinned to the top-left, always visible (outside the
        # scroll area so it never scrolls away) ─────────────────────────────
        logo_bar = QHBoxLayout()
        logo_bar.setContentsMargins(24, 18, 24, 0)
        logo_path = os.path.join(ASSETS_DIR, "logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaledToHeight(44, Qt.SmoothTransformation))
            logo_bar.addWidget(logo_label, alignment=Qt.AlignLeft)
        logo_bar.addStretch()
        outer.addLayout(logo_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(44, 40, 44, 40)
        root.setSpacing(20)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("HandsToVoice")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Rubik", 40, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color:#00D4AA; background:transparent; letter-spacing:0.5px;")
        root.addWidget(title)
        self._register_entrance(title)

        subtitle = QLabel("Kinyarwanda Sign Language Recognition & Voice Conversion")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "font-family:'Inter'; font-size:15px; font-weight:500; color:#8899AA;"
            "background:transparent;"
        )
        root.addWidget(subtitle)
        self._register_entrance(subtitle)

        # ── Animation 3: voice equalizer, right under the subtitle ──────────
        # Not wrapped in an entrance-fade effect: its bars animate
        # continuously, and nesting a second QGraphicsOpacityEffect around
        # an already-animating child causes Qt offscreen-buffer conflicts
        # (QPainter "paint device can only be painted by one painter" spam).
        equalizer = EqualizerBarsWidget(bar_count=7, min_h=6, max_h=28)
        equalizer.setFixedHeight(34)
        self._equalizer_anim = equalizer
        eq_row = QHBoxLayout()
        eq_row.setAlignment(Qt.AlignCenter)
        eq_row.addWidget(equalizer)
        root.addLayout(eq_row)

        # ── Hero card (animation 1: conversation illustration) ──────────────
        hero_card = self._build_hero_card()
        root.addWidget(hero_card)
        self._register_entrance(hero_card)

        # ── Pipeline card (animation 2: Sign -> Understand -> Speak) ────────
        # Same reasoning as the equalizer above: PipelineFlowWidget's badges
        # each animate their own opacity continuously, so the card itself
        # is left out of the entrance-fade registration.
        pipeline_card = self._build_pipeline_card()
        root.addWidget(pipeline_card)

        root.addStretch()

        self.start_btn = QPushButton("Get Started  →")
        self.start_btn.setObjectName("speakButton")
        self.start_btn.setMinimumHeight(54)
        self.start_btn.setFont(QFont("Inter", 15, QFont.DemiBold))
        self.start_btn.clicked.connect(self._on_start)
        root.addWidget(self.start_btn)
        # Deliberately NOT wrapped in a QGraphicsOpacityEffect entrance-fade
        # like the other sections: a QGraphicsOpacityEffect on this native
        # QPushButton renders the whole button invisible on this system
        # (background, text, everything) — a known-flaky combination of
        # QGraphicsEffect with styled QPushButtons. It appears at full
        # opacity immediately instead; the pulse below uses a stylesheet
        # animation, not an effect, to stay safe.

        hint = QLabel("Camera and voice will load right after you continue.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(
            "font-family:'Inter'; font-size:11px; color:#5A6B7A; background:transparent;"
        )
        root.addWidget(hint)
        self._register_entrance(hint)

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
        stagger_ms = 120
        for i, (widget, effect) in enumerate(self._entrance_widgets):
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(460)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._entrance_anims.append(anim)
            QTimer.singleShot(i * stagger_ms, anim.start)

        total_delay = len(self._entrance_widgets) * stagger_ms + 460
        QTimer.singleShot(total_delay, self._start_cta_pulse)

    _BTN_BASE = "#00D4AA"
    _BTN_GLOW = "#00F5C8"

    def _start_cta_pulse(self):
        """A gentle, continuous breathing glow on the CTA button — draws
        the eye without being distracting. Loops until the window closes.
        Uses a QVariantAnimation driving the stylesheet color directly
        (not a QGraphicsEffect — see the note where start_btn is built)."""
        anim = QVariantAnimation(self)
        anim.setDuration(1100)
        anim.setStartValue(0.0)
        anim.setKeyValueAt(0.5, 1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)
        anim.setLoopCount(4)   # a few breaths to draw the eye, then settle
        anim.valueChanged.connect(self._apply_btn_glow)
        anim.finished.connect(lambda: self.start_btn.setStyleSheet(""))
        anim.start()
        self._pulse_anim = anim

    def _apply_btn_glow(self, t):
        color = self._lerp_color(self._BTN_BASE, self._BTN_GLOW, t)
        self.start_btn.setStyleSheet(
            "QPushButton#speakButton {"
            f" background-color: {color}; color: #0F1419;"
            " border: none; border-radius: 12px; }"
        )

    @staticmethod
    def _lerp_color(c1, c2, t):
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = round(r1 + (r2 - r1) * t)
        g = round(g1 + (g2 - g1) * t)
        b = round(b1 + (b2 - b1) * t)
        return f"#{r:02X}{g:02X}{b:02X}"

    def _build_hero_card(self):
        card = QFrame()
        card.setObjectName("cardFrame")
        layout = QVBoxLayout(card)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        tagline = QLabel("One shared language")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setFont(QFont("Inter", 16, QFont.DemiBold))
        tagline.setStyleSheet("color:#7C4DFF; background:transparent;")
        layout.addWidget(tagline)

        anim_path = os.path.join(ASSETS_DIR, "cover_conversation.gif")
        anim_label = QLabel()
        anim_label.setFixedSize(540, 260)
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
        story.setStyleSheet(
            "font-family:'Inter'; font-size:15px; color:#E8ECF1; background:transparent;"
            "padding:4px 16px;"
        )
        layout.addWidget(story)

        return card

    def _build_pipeline_card(self):
        card = QFrame()
        card.setObjectName("cardFrame")
        layout = QVBoxLayout(card)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 20, 24, 20)

        heading = QLabel("How it works")
        heading.setAlignment(Qt.AlignCenter)
        heading.setFont(QFont("Inter", 16, QFont.DemiBold))
        heading.setStyleSheet("color:#00D4AA; background:transparent;")
        layout.addWidget(heading)

        pipeline = PipelineFlowWidget()
        self._pipeline_anim = pipeline
        pipeline_row = QHBoxLayout()
        pipeline_row.setAlignment(Qt.AlignCenter)
        pipeline_row.addWidget(pipeline)
        layout.addLayout(pipeline_row)

        return card

    def _on_start(self):
        logger.info("[WelcomeScreen] User continued into the app.")
        self._stop_animations()
        self.continue_requested.emit()
        self.close()

    def _stop_animations(self):
        if self.movie:
            self.movie.stop()
        if self._pulse_anim:
            self._pulse_anim.stop()
        if self._pipeline_anim:
            self._pipeline_anim.stop()
        if self._equalizer_anim:
            self._equalizer_anim.stop()

    def closeEvent(self, event):
        self._stop_animations()
        event.accept()
