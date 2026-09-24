"""
HandsToVoice — Main GUI Window
PyQt5 main window that integrates all system components.
Includes a sidebar with signing-state indicator.
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QFrame, QLabel, QPushButton, QProgressBar,
                             QScrollArea, QSplitter, QGroupBox, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont

from .camera_widget import CameraWidget
from .add_sign_dialog import AddSignDialog
from .voice_manager_dialog import VoiceManagerDialog
from .edit_sign_dialog import EditSignDialog
from .delete_sign_dialog import DeleteSignDialog
from .model_info_dialog import ModelInfoDialog
from .learn_sign_dialog import LearnSignDialog
from .sign_popup import SignPopup

from src.logger import get_logger
from src.speech_listener import SpeechListener
from src.model_info import load_lstm_metadata, summarize

logger = get_logger("gui.main_window")


# ── Signing-workflow states ──────────────────────────────────────────────────
STATE_IDLE = "idle"            # camera off
STATE_READY = "ready"          # user may sign
STATE_INTERPRETING = "interp"  # system is classifying — user must wait


class MainWindow(QMainWindow):
    """Main application window for HandsToVoice system."""

    # How long (ms) the system stays in INTERPRETING before going back to READY
    INTERP_DURATION_MS = 1200   # ms to lock out new signs after one fires

    # Grace period (ms) between a sign being detected and it actually being
    # spoken aloud — gives the user a window to hit "Undo Last Sign" and
    # cancel a wrong detection before it reaches the audience.
    CONFIRM_DELAY_MS = 1500

    def __init__(self, camera, detector, tts, vocabulary,
                 lstm_classifier=None, speech_listener=None):
        super().__init__()

        # System components
        self.camera = camera
        self.detector = detector
        self.tts = tts
        self.vocabulary = vocabulary
        self.lstm_classifier = lstm_classifier
        self.use_lstm = (lstm_classifier is not None and lstm_classifier.is_ready())

        # UI state
        self.current_sign = None
        self.current_confidence = 0.0
        self.sentence_history = []
        self.sentence_labels = []
        self.is_recording = False
        self._sign_state = STATE_IDLE
        self._last_detected_sign = None   # suppress consecutive duplicates

        self._lstm_sign_buffer = []       # consecutive overlapping-window consensus for LSTM
        self._LSTM_CONSENSUS_NEEDED = 2   # windows in a row to confirm an LSTM sign

        # Batch-speak: speak each sign as soon as it's recognized
        self._BATCH_SIZE = 3
        self._pending_batch = []      # labels waiting to be spoken

        # Confirm-before-speak: holds a detected sign during CONFIRM_DELAY_MS
        # so "Undo Last Sign" can cancel it before it's voiced
        self._pending_speak_labels = None
        self._confirm_timer = QTimer(self)
        self._confirm_timer.setSingleShot(True)
        self._confirm_timer.timeout.connect(self._confirm_pending_sign)

        # Setup UI
        self.setWindowTitle("HandsToVoice - Kinyarwanda Sign Language Recognition")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1100, 600)

        self.speech_listener = None
        self.sign_popup = SignPopup(self.vocabulary, self)

        self.setup_ui()
        self.setup_status_bar()
        self.setup_connections()
        self._apply_sign_state(STATE_IDLE)
        self.update_ui_state()
        self._start_listening(speech_listener)

    # ── UI construction ──────────────────────────────────────────────────────
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header
        main_layout.addWidget(self.create_header())

        # Missing-voice warning banner (shown when some signs have no recording)
        self.voice_banner = self._create_voice_banner()
        main_layout.addWidget(self.voice_banner)
        self._refresh_voice_banner()

        # Body = sidebar + camera + results
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        # ── LEFT SIDEBAR (sign-state indicator) ──
        body_layout.addWidget(self.create_sidebar())

        # ── CENTRE – camera feed ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_camera_section())
        splitter.addWidget(self.create_results_section())
        splitter.setSizes([720, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        body_layout.addWidget(splitter, stretch=1)

        main_layout.addLayout(body_layout)

        # Bottom controls
        main_layout.addWidget(self.create_control_section())

    # ── Sidebar ──────────────────────────────────────────────────────────────
    def create_sidebar(self):
        """Sidebar showing the current signing state."""
        sidebar = QFrame()
        sidebar.setObjectName("sidebarFrame")
        sidebar.setFixedWidth(200)
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(16)

        # Title
        title = QLabel("Signing Status")
        title.setObjectName("sidebarTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Big state icon
        self.state_icon_label = QLabel("📷")
        self.state_icon_label.setObjectName("stateIconLabel")
        self.state_icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_icon_label)

        # State text
        self.state_text_label = QLabel("Camera Off")
        self.state_text_label.setObjectName("stateTextLabel")
        self.state_text_label.setAlignment(Qt.AlignCenter)
        self.state_text_label.setWordWrap(True)
        layout.addWidget(self.state_text_label)

        # Instruction
        self.state_instruction_label = QLabel("Press Start to begin")
        self.state_instruction_label.setObjectName("stateInstructionLabel")
        self.state_instruction_label.setAlignment(Qt.AlignCenter)
        self.state_instruction_label.setWordWrap(True)
        layout.addWidget(self.state_instruction_label)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # Model mode indicator
        mode_header = QHBoxLayout()
        mode_header.addWidget(QLabel("Model Mode:"))
        mode_header.addStretch()
        self.model_info_btn = QPushButton("ℹ")
        self.model_info_btn.setFixedSize(22, 22)
        self.model_info_btn.setToolTip("What does this model actually know?")
        self.model_info_btn.setStyleSheet(
            "QPushButton { background:#1A2332; color:#8899AA; border:1px solid #2A3A4A;"
            "  border-radius:11px; font-weight:700; font-size:12px; min-width:0; padding:0; }"
            "QPushButton:hover { color:#7C4DFF; border-color:#7C4DFF; }")
        self.model_info_btn.clicked.connect(self.open_model_info)
        mode_header.addWidget(self.model_info_btn)
        layout.addLayout(mode_header)

        self.model_mode_label = QLabel("")
        self.model_mode_label.setAlignment(Qt.AlignCenter)
        self.model_mode_label.setWordWrap(True)
        layout.addWidget(self.model_mode_label)
        self._refresh_model_mode_label()

        # LSTM buffer fill bar (only visible in LSTM mode)
        self.lstm_fill_bar = QProgressBar()
        self.lstm_fill_bar.setRange(0, 100)
        self.lstm_fill_bar.setValue(0)
        self.lstm_fill_bar.setFormat("Buffer: %p%")
        self.lstm_fill_bar.setVisible(self.use_lstm)
        layout.addWidget(self.lstm_fill_bar)

        # Batch progress indicator
        layout.addWidget(QLabel("Batch Progress:"))
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, self._BATCH_SIZE)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"0 / {self._BATCH_SIZE} signs")
        self.batch_progress_bar.setStyleSheet(
            "QProgressBar { border-radius:6px; background:#1A2332; text-align:center;"
            " color:#FFFFFF; font-weight:600; font-size:11px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "  stop:0 #7C4DFF, stop:1 #00D4AA); border-radius:6px; }"
        )
        layout.addWidget(self.batch_progress_bar)

        # Last recognised sign (mini card)
        layout.addWidget(QLabel("Last Sign:"))
        self.sidebar_last_sign = QLabel("—")
        self.sidebar_last_sign.setObjectName("sidebarLastSign")
        self.sidebar_last_sign.setAlignment(Qt.AlignCenter)
        self.sidebar_last_sign.setWordWrap(True)
        layout.addWidget(self.sidebar_last_sign)

        layout.addStretch()
        return sidebar

    # ── Header ───────────────────────────────────────────────────────────────
    def create_header(self):
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("HandsToVoice")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel(
            "Real-time Kinyarwanda Sign Language Recognition & Voice Conversion"
        )
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return header_frame

    # ── Camera section ───────────────────────────────────────────────────────
    def create_camera_section(self):
        camera_frame = QFrame()
        camera_frame.setObjectName("cardFrame")
        layout = QVBoxLayout(camera_frame)

        title = QLabel("📷 Camera Feed")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        self.camera_widget = CameraWidget(self.camera, self.detector)
        layout.addWidget(self.camera_widget)
        return camera_frame

    # ── Results section ──────────────────────────────────────────────────────
    def create_results_section(self):
        results_frame = QFrame()
        results_frame.setObjectName("cardFrame")
        layout = QVBoxLayout(results_frame)

        title = QLabel("🔤 Recognition Results")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # Current sign
        self.sign_label = QLabel("No Sign Detected")
        self.sign_label.setObjectName("signLabel")
        self.sign_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sign_label)

        # Confidence
        confidence_group = QGroupBox("Confidence")
        cl = QVBoxLayout(confidence_group)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("0%")
        cl.addWidget(self.confidence_bar)
        layout.addWidget(confidence_group)

        # Translation
        self.translation_label = QLabel("")
        self.translation_label.setObjectName("translationLabel")
        self.translation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.translation_label)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # Sentence history
        history_group = QGroupBox("Sentence History")
        hl = QVBoxLayout(history_group)
        self.sentence_label = QLabel("")
        self.sentence_label.setObjectName("sentenceLabel")
        self.sentence_label.setWordWrap(True)
        hl.addWidget(self.sentence_label)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setMaximumHeight(200)
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_scroll.setWidget(self.history_widget)
        hl.addWidget(self.history_scroll)
        layout.addWidget(history_group)

        # Model status
        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("statusLabel")
        layout.addWidget(self.model_status_label)
        return results_frame

    # ── Missing-voice banner ──────────────────────────────────────────────────
    def _create_voice_banner(self):
        """Yellow warning banner shown when some signs still lack a custom recording."""
        banner = QFrame()
        banner.setObjectName("voiceBanner")
        banner.setStyleSheet(
            "QFrame#voiceBanner {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            "    stop:0 #332600, stop:1 #2a1f00);"
            "  border: 1px solid #FFD93D;"
            "  border-radius: 10px;"
            "  padding: 2px;"
            "}"
        )
        banner.setFixedHeight(52)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        # Warning icon + text
        self._banner_icon = QLabel("⚠️")
        self._banner_icon.setStyleSheet("font-size:22px; background:transparent;")
        layout.addWidget(self._banner_icon)

        self._banner_text = QLabel("")
        self._banner_text.setStyleSheet(
            "color:#FFD93D; font-size:13px; font-weight:600; background:transparent;"
        )
        layout.addWidget(self._banner_text, stretch=1)

        # Quick-action button
        self._banner_btn = QPushButton("🎙️ Record Missing Voices")
        self._banner_btn.setFixedHeight(34)
        self._banner_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #FFD93D; color: #0F1419;"
            "  border: none; border-radius: 8px;"
            "  font-weight: 700; font-size: 12px;"
            "  padding: 0 16px;"
            "}"
            "QPushButton:hover { background-color: #FFE566; }"
            "QPushButton:pressed { background-color: #E5C235; }"
        )
        self._banner_btn.clicked.connect(self._open_voice_manager_missing)
        layout.addWidget(self._banner_btn)

        return banner

    def _refresh_voice_banner(self):
        """Recalculate missing voices and show/hide banner accordingly."""
        missing = [
            lbl for lbl in self.vocabulary.signs
            if not self.tts._has_custom_audio(lbl)
        ]
        total = self.vocabulary.get_total_signs()
        recorded = total - len(missing)

        if missing:
            self._banner_text.setText(
                f"{len(missing)} sign(s) using system TTS voice — record custom audio for better quality. "
                f"({recorded}/{total} recorded)"
            )
            self.voice_banner.setVisible(True)
        else:
            self.voice_banner.setVisible(False)

    def _open_voice_manager_missing(self):
        """Open Voice Manager pre-filtered to show unrecorded signs first."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        dlg = VoiceManagerDialog(self.vocabulary, self.tts, missing_only=True, parent=self)
        dlg.voices_updated.connect(self._on_voices_updated)
        dlg.exec_()

        if was_running:
            self.start_recognition()

    # ── Controls ─────────────────────────────────────────────────────────────
    def create_control_section(self):
        controls_frame = QFrame()
        controls_frame.setObjectName("cardFrame")
        outer = QVBoxLayout(controls_frame)
        outer.setSpacing(12)

        # ── Row 1: live-session controls — used constantly while signing ──
        session_row = QHBoxLayout()
        session_row.setSpacing(10)

        self.start_stop_button = QPushButton("▶️ Start Recognition")
        self.start_stop_button.setObjectName("speakButton")
        self.start_stop_button.clicked.connect(self.toggle_recognition)
        session_row.addWidget(self.start_stop_button)

        self.speak_button = QPushButton("🔊 Speak")
        self.speak_button.setObjectName("speakButton")
        self.speak_button.clicked.connect(self.speak_current_sentence)
        self.speak_button.setEnabled(False)
        session_row.addWidget(self.speak_button)

        session_row.addStretch()

        # Neutral style: Undo/Clear are reversible, low-stakes actions —
        # they don't get the red "danger" treatment that Delete Sign does.
        self.undo_sign_button = QPushButton("↩️ Undo Last Sign")
        self.undo_sign_button.setObjectName("neutralButton")
        self.undo_sign_button.setToolTip(
            "Remove the last detected sign — use this if the system read the wrong sign,"
            " before it gets spoken to your audience."
        )
        self.undo_sign_button.clicked.connect(self.undo_last_sign)
        self.undo_sign_button.setEnabled(False)
        session_row.addWidget(self.undo_sign_button)

        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setObjectName("neutralButton")
        self.clear_button.clicked.connect(self.clear_sentence)
        session_row.addWidget(self.clear_button)

        self.listen_button = QPushButton("🎧 Listening: loading…")
        self.listen_button.setObjectName("neutralButton")
        self.listen_button.setToolTip(
            "When a hearing person says a vocabulary word, its sign video pops up. "
            "Click to turn microphone listening off or on.")
        self.listen_button.clicked.connect(self.toggle_listening)
        session_row.addWidget(self.listen_button)

        outer.addLayout(session_row)

        divider = QFrame()
        divider.setObjectName("separator")
        outer.addWidget(divider)

        # ── Row 2: vocabulary management — setup-time actions ─────────────
        manage_row = QHBoxLayout()
        manage_row.setSpacing(10)

        self.add_sign_button = QPushButton("➕ Add Sign")
        self.add_sign_button.setObjectName("addSignButton")
        self.add_sign_button.setToolTip("Define a new sign, record video clips, and retrain the LSTM model")
        self.add_sign_button.clicked.connect(self.open_add_sign_dialog)
        manage_row.addWidget(self.add_sign_button)

        self.edit_sign_button = QPushButton("✏️ Edit Sign")
        self.edit_sign_button.setObjectName("addSignButton")
        self.edit_sign_button.setToolTip("Edit an existing sign: change its name, translation, and re-record your voice")
        self.edit_sign_button.clicked.connect(self.open_edit_sign_dialog)
        manage_row.addWidget(self.edit_sign_button)

        self.manage_voices_button = QPushButton("🎙️ Manage Voices")
        self.manage_voices_button.setObjectName("addSignButton")
        self.manage_voices_button.setToolTip("Record, play, and delete custom voices for KSL signs")
        self.manage_voices_button.clicked.connect(self.open_voice_manager)
        manage_row.addWidget(self.manage_voices_button)

        self.learn_sign_button = QPushButton("📖 Learn a Sign")
        self.learn_sign_button.setObjectName("addSignButton")
        self.learn_sign_button.setToolTip(
            "Look up a word and watch a reference video of how to sign it")
        self.learn_sign_button.clicked.connect(self.open_learn_sign_dialog)
        manage_row.addWidget(self.learn_sign_button)

        manage_row.addStretch()

        # Kept visually separate (red/danger) at the end of its row — this
        # one really is destructive and permanent, unlike its neighbors.
        self.delete_sign_button = QPushButton("🗑️ Delete Sign")
        self.delete_sign_button.setObjectName("clearButton")
        self.delete_sign_button.setToolTip("Permanently remove a sign and all its associated data")
        self.delete_sign_button.clicked.connect(self.open_delete_sign_dialog)
        manage_row.addWidget(self.delete_sign_button)

        outer.addLayout(manage_row)

        return controls_frame

    # ── Status bar ───────────────────────────────────────────────────────────
    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.speech_privacy_label = QLabel(
            "🔒 While listening, speech is sent to Google to be recognized")
        self.speech_privacy_label.setToolTip(
            "Voice listening sends each spoken phrase to Google's speech service "
            "over the internet. Without internet, it is recognized on this "
            "computer instead and nothing leaves it. HandsToVoice itself never "
            "saves the audio. "
            "Turn listening off with the 🎧 button.")
        self.speech_privacy_label.hide()
        self.status_bar.addPermanentWidget(self.speech_privacy_label)
        self.update_status_bar()

    # ── Connections ──────────────────────────────────────────────────────────
    def setup_connections(self):
        self.camera_widget.landmarks_detected.connect(self.on_landmarks_detected)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui_state)
        self.update_timer.start(100)

        # Timer used to exit the INTERPRETING state after a delay
        self._interp_timer = QTimer()
        self._interp_timer.setSingleShot(True)
        self._interp_timer.timeout.connect(self._finish_interpreting)

    # ── Signing-state machine ────────────────────────────────────────────────
    def _apply_sign_state(self, state):
        self._sign_state = state

        if state == STATE_IDLE:
            self.state_icon_label.setText("📷")
            self.state_text_label.setText("Camera Off")
            self.state_instruction_label.setText("Press Start to begin")
            self.state_icon_label.setStyleSheet(
                "font-size:72px; color:#8899AA; background:transparent;")
            self.state_text_label.setStyleSheet(
                "font-size:16px; font-weight:700; color:#8899AA; background:transparent;")

        elif state == STATE_READY:
            self.state_icon_label.setText("✅")
            self.state_text_label.setText("READY\nSign Now!")
            self.state_instruction_label.setText(
                "Show your sign clearly in front of the camera.")
            self.state_icon_label.setStyleSheet(
                "font-size:72px; color:#00D4AA; background:transparent;")
            self.state_text_label.setStyleSheet(
                "font-size:16px; font-weight:700; color:#00D4AA; background:transparent;")

        elif state == STATE_INTERPRETING:
            self.state_icon_label.setText("⛔")
            self.state_text_label.setText("INTERPRETING\nPlease Wait...")
            self.state_instruction_label.setText(
                "Do NOT sign.\nThe system is processing your sign.")
            self.state_icon_label.setStyleSheet(
                "font-size:72px; color:#FF6B6B; background:transparent;")
            self.state_text_label.setStyleSheet(
                "font-size:16px; font-weight:700; color:#FF6B6B; background:transparent;")

    def _enter_interpreting(self):
        """Transition to INTERPRETING and schedule return to READY."""
        self._apply_sign_state(STATE_INTERPRETING)
        self._interp_timer.start(self.INTERP_DURATION_MS)

    def _finish_interpreting(self):
        """Return to READY after interpreting is done."""
        if self.is_recording:
            self._apply_sign_state(STATE_READY)

    @pyqtSlot(object, object)
    def on_landmarks_detected(self, landmarks_list, face_ref=None):
        if not self.is_recording:
            return
        # No hand visible — reset so the same sign can be spoken again next time
        if not landmarks_list:
            self._last_detected_sign = None
            return
        # Ignore new signs while the system is locked out after a detection
        if self._sign_state == STATE_INTERPRETING:
            return

        landmarks = landmarks_list[0]

        # ── Step 1: Always feed frames into LSTM buffer ──────────────────
        if self.use_lstm:
            self.lstm_classifier.push_frame(landmarks, face_ref)
            fill = self.lstm_classifier.buffer_fill()
            self.lstm_fill_bar.setValue(int(fill * 100))

        # ── Step 2: Try LSTM first when buffer is full ───────────────────
        sign_label = None
        confidence = 0.0
        source = None

        if self.use_lstm and self.lstm_classifier.buffer_fill() >= 1.0:
            raw_label, raw_conf = self.lstm_classifier.predict()
            if raw_label:
                # Require consensus across consecutive overlapping windows
                # (SLIDE_STEP means windows share most of their frames) before
                # accepting — cuts down on one-off spurious detections without
                # needing any more training data.
                self._lstm_sign_buffer.append(raw_label)
                self._lstm_sign_buffer = self._lstm_sign_buffer[-self._LSTM_CONSENSUS_NEEDED:]
                if (len(self._lstm_sign_buffer) == self._LSTM_CONSENSUS_NEEDED and
                        len(set(self._lstm_sign_buffer)) == 1):
                    sign_label = raw_label
                    confidence = raw_conf
                    source = "LSTM"
                    self.lstm_classifier.clear_buffer()
                    self.lstm_fill_bar.setValue(0)
                    self._lstm_sign_buffer.clear()
                else:
                    self.lstm_classifier.slide_buffer()
            else:
                # LSTM buffer is full but gave no result — slide and keep going
                self._lstm_sign_buffer.clear()
                self.lstm_classifier.slide_buffer()

        if sign_label:
            sign_label = str(sign_label)   # cast np.str_ → plain str at source

            # Only speak if this is a DIFFERENT sign from the last one
            if sign_label == self._last_detected_sign:
                # Same sign — reset buffer and wait; do not repeat audio
                self.lstm_classifier.clear_buffer()
                return

            logger.info(f"[Detection] {source} → '{sign_label}' ({confidence*100:.0f}%)")
            self._last_detected_sign = sign_label
            self.current_sign = sign_label
            self.current_confidence = confidence

            cp = int(confidence * 100)
            self.confidence_bar.setValue(cp)
            self.confidence_bar.setFormat(f"{cp}%")

            display = self.vocabulary.get_display_text(sign_label)
            self.sidebar_last_sign.setText(display)
            self.sign_label.setText(display)

            self.add_sign_to_sentence(sign_label)
            self.undo_sign_button.setEnabled(True)

            # ── Batch-speak: accumulate 3 signs, then read all together ──
            self._pending_batch.append(sign_label)
            batch_count = len(self._pending_batch)

            # Update batch progress bar
            self.batch_progress_bar.setValue(batch_count)
            self.batch_progress_bar.setFormat(
                f"{batch_count} / {self._BATCH_SIZE} signs"
                if batch_count < self._BATCH_SIZE
                else "🔊 Speaking!"
            )

            if batch_count >= self._BATCH_SIZE:
                # Hold the accumulated signs for a short grace period instead
                # of speaking immediately, so a wrong detection can be
                # cancelled via "Undo Last Sign" before it reaches the
                # audience.
                batch_labels = list(self._pending_batch)
                logger.info(
                    f"[Batch] Holding for confirm ({self.CONFIRM_DELAY_MS}ms): {batch_labels}"
                )
                self._pending_speak_labels = batch_labels
                self.undo_sign_button.setEnabled(True)
                self._confirm_timer.start(self.CONFIRM_DELAY_MS)

                # ── Full reset so next batch collects only NEW signs ──
                self._pending_batch.clear()
                self._last_detected_sign = None   # any sign is fresh for next batch
                if self.use_lstm:
                    self.lstm_classifier.clear_buffer()  # flush LSTM frame buffer
                logger.info("[Batch] Buffers flushed — ready to collect next batch")

                # Reset bar after a short delay so user sees the flash
                QTimer.singleShot(800, lambda: (
                    self.batch_progress_bar.setValue(0),
                    self.batch_progress_bar.setFormat(f"0 / {self._BATCH_SIZE} signs")
                ))
            else:
                logger.info(f"[Batch] Collected {batch_count}/{self._BATCH_SIZE} — waiting for more")

            self._enter_interpreting()
            return

    # ── Listening: speech in -> sign video out ───────────────────────────────
    def _start_listening(self, existing=None):
        """existing: an already-started SpeechListener (see main.py), which
        was kicked off in parallel with camera/model loading instead of
        sequentially after this window was built — otherwise its ~15-20s
        model load was added on top of camera/model init instead of
        overlapping it. Falls back to creating a fresh one (e.g. when
        re-enabling via the toolbar button after stop_listening())."""
        if self.speech_listener is not None:
            return
        if existing is not None:
            self.speech_listener = existing
            self.speech_listener.update_vocab(self._vocab_words())
        else:
            self.speech_listener = SpeechListener(self._vocab_words())
            self.speech_listener.start()
        self.speech_listener.words_heard.connect(self._on_words_heard)
        self.speech_listener.mic_silent.connect(self._on_mic_silent)
        self.speech_listener.model_status.connect(self._on_listen_model_status)
        # The listener may already have finished loading (or failed) before
        # we could connect just now, in which case that earlier model_status
        # signal was already emitted and missed — read its current state
        # directly instead of waiting for a signal that already fired.
        self._on_listen_model_status(self.speech_listener.status)

    def stop_listening(self):
        if self.speech_listener is not None:
            self.speech_listener.stop()
            self.speech_listener = None
        self.listen_button.setText("🎧 Listening: Off")
        self.speech_privacy_label.hide()

    def toggle_listening(self):
        if self.speech_listener is None:
            self.listen_button.setText("🎧 Listening: loading…")
            self._start_listening()
        else:
            self.stop_listening()

    def _vocab_words(self):
        return {label: info.get("kinyarwanda", label)
                for label, info in self.vocabulary.signs.items()}

    def _reload_listener_vocab(self):
        if self.speech_listener is not None:
            self.speech_listener.update_vocab(self._vocab_words())

    @pyqtSlot(str)
    def _on_listen_model_status(self, status):
        if status == "ready":
            self.listen_button.setText("🎧 Listening: On")
            self.speech_privacy_label.show()
        elif status == "failed":
            self.listen_button.setText("🎧 Listening: unavailable")
            self.listen_button.setEnabled(False)
            self.listen_button.setToolTip("The speech-recognition model could not be loaded.")

    @pyqtSlot(bool)
    def _on_mic_silent(self, silent):
        if self.speech_listener is None:
            return
        self.listen_button.setText("🎧 Mic is muted?" if silent else "🎧 Listening: On")

    def _update_listener_mute(self):
        """Don't listen while the system is speaking a sign aloud — the mic
        would hear the system's own voice and pop up that same sign again."""
        if self.speech_listener is not None:
            self.speech_listener.muted = self.tts.is_speaking()

    @pyqtSlot(list)
    def _on_words_heard(self, words):
        if QApplication.activeModalWidget() is not None:
            return          # a dialog is open (e.g. recording a voice) — not audience speech
        labels = [label for label, _ in words]
        logger.info(f"[Listen] showing signs for: {labels}")
        try:
            self.sign_popup.show_words(labels)
        except Exception:
            # PyQt slots invoked from a queued cross-thread signal (this one
            # fires from SpeechListener's thread) can swallow exceptions
            # silently instead of printing them — logging explicitly here
            # to actually see what's failing.
            logger.exception("[Listen] show_words() raised — popup not shown")

    # ── Confirm-before-speak / Undo ──────────────────────────────────────────
    def _confirm_pending_sign(self):
        """Grace period elapsed with no Undo — speak the held sign(s) now."""
        labels = self._pending_speak_labels
        self._pending_speak_labels = None
        self.undo_sign_button.setEnabled(False)
        if labels:
            logger.info(f"[Batch] Speaking (confirmed): {labels}")
            self.tts.speak(None, labels=labels)

    def undo_last_sign(self):
        """Remove the most recently detected sign — from wherever it
        currently lives, so a mistake anywhere in the batch of 3 can be
        corrected, not just the one that just finished the batch:

        - Still being collected (batch not yet at 3): drop it from the
          pending batch, nothing has been held for speech yet.
        - Held during its confirm grace period (batch just completed):
          drop just that one sign from what's about to be spoken — the
          rest of the batch still speaks normally when the timer fires.
        - Already spoken: nothing to un-speak, so this just corrects the
          sentence text.
        """
        if self._pending_batch:
            removed = self._pending_batch.pop()
            logger.info(f"[Undo] Removed from pending batch: {removed}")
            batch_count = len(self._pending_batch)
            self.batch_progress_bar.setValue(batch_count)
            self.batch_progress_bar.setFormat(f"{batch_count} / {self._BATCH_SIZE} signs")
        elif self._pending_speak_labels:
            removed = self._pending_speak_labels.pop()
            logger.info(f"[Undo] Removed before speaking: {removed}")
            if not self._pending_speak_labels:
                self._confirm_timer.stop()

        if self.sentence_labels:
            self.sentence_labels.pop()
            self.sentence_history.pop()
            self.update_sentence_display()

        self.undo_sign_button.setEnabled(bool(self.sentence_labels))
        self._last_detected_sign = None   # allow re-signing the same sign immediately

    # ── Sentence management ──────────────────────────────────────────────────
    def add_sign_to_sentence(self, sign_label):
        kw = self.vocabulary.get_kinyarwanda(sign_label)
        self.sentence_labels.append(sign_label)
        if not self.sentence_history:
            self.sentence_history.append(kw.capitalize())
        else:
            self.sentence_history.append(kw)
        self.update_sentence_display()

    def update_sentence_display(self):
        if self.sentence_history:
            self.sentence_label.setText(" ".join(self.sentence_history))
            self.speak_button.setEnabled(True)
        else:
            self.sentence_label.setText("")
            self.speak_button.setEnabled(False)

    # ── Add Sign Dialog ──────────────────────────────────────────────────────
    def open_add_sign_dialog(self):
        """Open the 4-step Add Sign wizard."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        # Release the main camera so the dialog can access /dev/video0
        self.camera.release()

        dlg = AddSignDialog(self.vocabulary,
                            camera_index=self.camera.camera_index,
                            parent=self)
        dlg.training_complete.connect(self.reload_model)
        dlg.exec_()

        # ── Full refresh after the dialog closes ──────────────────────────
        # Ensure the newly trained sign is immediately usable.
        self.reload_model()                # reload LSTM model + vocabulary
        self._reset_detection_state()      # clear buffers so new sign is detected fresh
        self._refresh_voice_banner()       # update missing-voice banner
        self.update_status_bar()

        # Re-open the main camera after the dialog closes
        try:
            self.camera.open()
        except RuntimeError as e:
            logger.warning(f"[MainWindow] Warning: could not re-open camera: {e}")

        if was_running:
            self.start_recognition()

    def open_edit_sign_dialog(self):
        """Open the Edit Sign dialog to rename, retranslate, and re-record a sign."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        dlg = EditSignDialog(self.vocabulary, self.tts, parent=self)
        dlg.sign_updated.connect(self._on_voices_updated)
        dlg.exec_()

        if was_running:
            self.start_recognition()

    def open_delete_sign_dialog(self):
        """Open the Delete Sign dialog."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        dlg = DeleteSignDialog(self.vocabulary, parent=self)
        dlg.sign_deleted.connect(self._on_sign_deleted)
        dlg.exec_()

        if was_running:
            self.start_recognition()

    def _on_sign_deleted(self):
        """Refresh state after a sign is deleted."""
        self.reload_model()
        self._reset_detection_state()
        self._refresh_voice_banner()
        self.update_status_bar()
        self.update_ui_state()
        self._reload_listener_vocab()
        logger.info("[MainWindow] State refreshed after sign deletion.")

    def open_voice_manager(self):
        """Open the custom voice manager dialog."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        dlg = VoiceManagerDialog(self.vocabulary, self.tts, parent=self)
        dlg.voices_updated.connect(self._on_voices_updated)
        dlg.exec_()

        if was_running:
            self.start_recognition()

    def open_learn_sign_dialog(self):
        """Open the Learn a Sign dialog: look up a word, watch its reference video."""
        was_running = self.is_recording
        if was_running:
            self.stop_recognition()

        dlg = LearnSignDialog(self.vocabulary, parent=self)
        dlg.exec_()

        if was_running:
            self.start_recognition()

    def _on_voices_updated(self):
        """Refresh sentence display, last sign label, and the missing-voice banner."""
        new_history = []
        for i, label in enumerate(self.sentence_labels):
            kw = self.vocabulary.get_kinyarwanda(label)
            if i == 0:
                new_history.append(kw.capitalize())
            else:
                new_history.append(kw)
        self.sentence_history = new_history
        self.update_sentence_display()
        self._refresh_voice_banner()
        self._reload_listener_vocab()

        if self.current_sign:
            display = self.vocabulary.get_display_text(self.current_sign)
            self.sidebar_last_sign.setText(display)
            self.sign_label.setText(display)

    def reload_model(self):
        """Hot-reload the LSTM classifier and vocabulary after a new sign is trained."""
        from src.classifier import LSTMClassifier

        # Reload vocabulary
        self.vocabulary.reload()

        # Reload LSTM model
        new_lstm = LSTMClassifier()
        if new_lstm.is_ready():
            self.lstm_classifier = new_lstm
            self.use_lstm = True
            self.lstm_fill_bar.setVisible(True)
            logger.info(f"[MainWindow] LSTM model reloaded. "
                  f"Classes: {new_lstm.get_num_classes()}, "
                  f"Labels: {new_lstm.labels}")
        else:
            logger.warning("[MainWindow] Warning: reloaded LSTM is not ready.")

        self._refresh_model_mode_label()
        self.update_status_bar()

    def _refresh_model_mode_label(self):
        """Set the sidebar Model Mode text from the model's real, honest
        metrics (trained class count, cross-validated accuracy) instead of
        a generic "LSTM (Video)" badge that overstates what's actually known.
        """
        if not self.use_lstm:
            self.model_mode_label.setText("⚠️ No model trained")
            self.model_mode_label.setStyleSheet(
                "font-size:13px; font-weight:700; color:#FF8A65;"
                "background:#1A2332; padding:8px; border-radius:8px;")
            return

        summary = summarize(load_lstm_metadata(), vocabulary=self.vocabulary)
        if summary and summary.get("accuracy") is not None:
            acc_pct = f"{summary['accuracy']:.0%}"
            tested_n = len(summary["tested_classes"])
            trained_n = summary["num_classes"]
            subtitle = f"{trained_n} signs · {acc_pct} acc ({tested_n}/{trained_n} tested)"
        else:
            subtitle = "accuracy not yet evaluated"

        self.model_mode_label.setText(f"🧠 LSTM (Video)\n{subtitle}")
        self.model_mode_label.setStyleSheet(
            "font-size:13px; font-weight:700; color:#7C4DFF;"
            "background:#1A2332; padding:8px; border-radius:8px;")

    def open_model_info(self):
        """Open the full model-info breakdown dialog."""
        dlg = ModelInfoDialog(self.vocabulary, parent=self)
        dlg.exec_()

    def _reset_detection_state(self):
        """Clear all detection buffers and state so fresh signs are accepted."""
        self.current_sign = None
        self.current_confidence = 0.0
        self._last_detected_sign = None
        self._lstm_sign_buffer.clear()
        self._pending_batch.clear()
        self._confirm_timer.stop()
        self._pending_speak_labels = None
        self.undo_sign_button.setEnabled(False)
        if self.use_lstm and self.lstm_classifier:
            self.lstm_classifier.clear_buffer()
            self.lstm_fill_bar.setValue(0)
        self.sidebar_last_sign.setText("—")
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"0 / {self._BATCH_SIZE} signs")
        logger.info("[MainWindow] Detection state reset — ready for new signs.")

    # ── Recognition toggle ───────────────────────────────────────────────────
    def toggle_recognition(self):
        if self.is_recording:
            self.stop_recognition()
        else:
            self.start_recognition()

    def start_recognition(self):
        if not self.use_lstm:
            self.show_model_warning()
            return
        self.is_recording = True
        self.camera_widget.start()
        self.start_stop_button.setText("⏸️ Stop Recognition")
        self._apply_sign_state(STATE_READY)
        self.update_status_bar()

    def stop_recognition(self):
        self.is_recording = False
        self.camera_widget.stop()
        self.start_stop_button.setText("▶️ Start Recognition")
        self._apply_sign_state(STATE_IDLE)
        self.update_status_bar()

    def speak_current_sentence(self):
        if self.sentence_history:
            self.tts.speak(" ".join(self.sentence_history), labels=self.sentence_labels)

    def clear_sentence(self):
        self.sentence_history.clear()
        self.sentence_labels.clear()
        self.current_sign = None
        self.current_confidence = 0.0
        self._last_detected_sign = None   # allow re-detecting same sign
        self._pending_batch.clear()       # discard any partial batch
        self._confirm_timer.stop()        # cancel any sign awaiting speech
        self._pending_speak_labels = None
        self.undo_sign_button.setEnabled(False)
        self.sidebar_last_sign.setText("—")
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"0 / {self._BATCH_SIZE} signs")
        self.update_sentence_display()
        self.update_ui_state()

    def show_model_warning(self):
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("No Model Loaded")
        msg.setText("No trained model found. Please train a model first.")
        msg.setInformativeText("Use ➕ Add Sign to record signs and train the model.")
        msg.exec_()

    # ── Periodic UI refresh ──────────────────────────────────────────────────
    def update_ui_state(self):
        self._update_listener_mute()
        if self.current_sign:
            display = self.vocabulary.get_display_text(self.current_sign)
            self.sign_label.setText(display)
            cp = int(self.current_confidence * 100)
            self.confidence_bar.setValue(cp)
            self.confidence_bar.setFormat(f"{cp}%")
            eng = self.vocabulary.get_english(self.current_sign)
            self.translation_label.setText(f"English: {eng}" if eng else "")
        else:
            self.sign_label.setText("No Sign Detected")
            self.confidence_bar.setValue(0)
            self.confidence_bar.setFormat("0%")
            self.translation_label.setText("")

    def update_status_bar(self):
        total_signs = self.vocabulary.get_total_signs()
        # When LSTM is active, "Signs" reflects what the model can actually
        # recognize, not the full vocabulary — a sign can exist in the
        # vocabulary (for TTS/display) without the model having been
        # trained on it yet.
        if self.use_lstm:
            trained_n = self.lstm_classifier.get_num_classes()
            signs_text = f"Signs: {trained_n}/{total_signs} trained" if trained_n != total_signs else f"Signs: {total_signs}"
        else:
            signs_text = f"Signs: {total_signs}"

        if self.is_recording:
            fps = self.camera_widget.get_fps()
            status = f"🔴 Recording | FPS: {fps:.0f} | {signs_text}"
        else:
            status = f"⏸️ Stopped | {signs_text}"
        if self.use_lstm:
            status += " | 🧠 LSTM Model"
        else:
            status += " | ⚠️ No Model"
        self.status_bar.showMessage(status)

    def closeEvent(self, event):
        self.stop_recognition()
        self.camera.release()
        if hasattr(self.detector, 'close'):
            self.detector.close()
        if hasattr(self.tts, 'cleanup'):
            self.tts.cleanup()
        event.accept()
