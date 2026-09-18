"""
HandsToVoice — Reusable content animations for the welcome/loading screens.

Two small, content-relevant animated widgets:
  - PipelineFlowWidget: a looping highlight chase across Sign -> Understand
    -> Speak, illustrating what the app actually does.
  - EqualizerBarsWidget: a looping voice-equalizer bar animation, standing
    in for HandsToVoice's spoken output.
"""

import random
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QVariantAnimation


class PipelineFlowWidget(QWidget):
    """Sign -> Understand -> Speak, with a looping highlight chase across
    the three stages."""

    STAGES = [("\U0001F44B", "Sign"), ("\U0001F9E0", "Understand"), ("\U0001F50A", "Speak")]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        self._badges = []
        self._effects = []
        for i, (icon, name) in enumerate(self.STAGES):
            if i > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet("color:#445566; font-size:20px; background:transparent;")
                layout.addWidget(arrow)

            badge = QLabel(f"{icon}\n{name}")
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(96, 68)
            badge.setStyleSheet(self._badge_style(False))
            effect = QGraphicsOpacityEffect(badge)
            effect.setOpacity(0.45)
            badge.setGraphicsEffect(effect)
            layout.addWidget(badge)
            self._badges.append(badge)
            self._effects.append(effect)

        self._active = -1
        self._anims = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(900)
        self._advance()

    def _badge_style(self, active):
        if active:
            return (
                "background:#1E2D3D; border:2px solid #00D4AA; border-radius:12px;"
                "font-size:12px; font-weight:600; color:#00D4AA;"
            )
        return (
            "background:#1A2332; border:2px solid #2A3A4A; border-radius:12px;"
            "font-size:12px; font-weight:600; color:#8899AA;"
        )

    def _advance(self):
        prev = self._active
        self._active = (self._active + 1) % len(self._badges)

        if prev >= 0:
            self._badges[prev].setStyleSheet(self._badge_style(False))
        self._badges[self._active].setStyleSheet(self._badge_style(True))

        for i, effect in enumerate(self._effects):
            target = 1.0 if i == self._active else 0.45
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(400)
            anim.setStartValue(effect.opacity())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            anim.start()
            self._anims.append(anim)
        self._anims = self._anims[-12:]

    def stop(self):
        self._timer.stop()


class EqualizerBarsWidget(QWidget):
    """A small looping voice-equalizer animation — bars rising and falling
    like an audio level meter."""

    def __init__(self, parent=None, bar_count=5, min_h=8, max_h=34):
        super().__init__(parent)
        self._min_h = min_h
        self._max_h = max_h
        self._anims = []
        self._running = True

        layout = QHBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)

        for i in range(bar_count):
            bar = QFrame()
            bar.setFixedWidth(6)
            bar.setFixedHeight(min_h)
            bar.setStyleSheet(
                "background: qlineargradient(x1:0,y1:1,x2:0,y2:0,"
                " stop:0 #00D4AA, stop:1 #7C4DFF); border-radius:3px;"
            )
            layout.addWidget(bar, alignment=Qt.AlignBottom)
            self._start_bar_loop(bar, delay=i * 120)

    def _start_bar_loop(self, bar, delay=0):
        def make_cycle():
            if not self._running:
                return
            target = random.randint(self._min_h, self._max_h)
            anim = QVariantAnimation(self)
            anim.setDuration(random.randint(260, 420))
            anim.setStartValue(bar.height())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.InOutSine)
            anim.valueChanged.connect(lambda v, b=bar: b.setFixedHeight(int(v)))
            anim.finished.connect(make_cycle)
            anim.start()
            self._anims.append(anim)
            self._anims[:] = self._anims[-40:]

        QTimer.singleShot(delay, make_cycle)

    def stop(self):
        self._running = False
        for anim in self._anims:
            anim.stop()
