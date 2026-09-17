"""
HandsToVoice — Voice Manager Dialog
Allows users to record, play back, and delete custom audio pronunciations
for each sign, replacing default TTS with their own voice.
"""

import os
import time
import subprocess
import threading
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QComboBox, QProgressBar, QMessageBox, QWidget, QSplitter,
    QFormLayout, QDialogButtonBox, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

from src.logger import get_logger

logger = get_logger("gui.voice_manager_dialog")


class VoiceManagerDialog(QDialog):
    """Dialog to manage custom voice recordings for KSL signs."""
    
    voices_updated = pyqtSignal()

    def __init__(self, vocabulary, tts, parent=None, missing_only=False):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.tts = tts
        self.selected_label = None
        self.is_recording = False
        self._missing_only = missing_only  # when True, sort unrecorded signs to top
        self._record_proc = None         # arecord subprocess handle
        self._record_temp_file = None    # temp WAV path during recording
        self._record_dest_file = None    # final WAV path

        # Timer for duration and visual ticking
        self.record_timer = QTimer(self)
        self.record_timer.setSingleShot(True)
        self.record_timer.timeout.connect(self.stop_recording)

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._on_elapsed_tick)
        self.elapsed_start_time = 0
        
        self.setWindowTitle("HandsToVoice — Manage Custom Voices")
        self.setMinimumSize(850, 500)
        self.setModal(True)
        
        self._build_ui()
        self._populate_signs()
        self._populate_audio_devices()

        # In missing_only mode, select the first unrecorded sign automatically
        if self._missing_only:
            self._select_first_missing()
        elif self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("🎙️ Custom Voice Manager")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:700; color:#00D4AA; padding-bottom:8px;")
        layout.addWidget(title)
        
        # Subtitle
        if self._missing_only:
            sub_text = (
                "Signs with no recording are listed first. "
                "Record each one — the dialog will automatically advance to the next missing voice."
            )
        else:
            sub_text = (
                "Record your own voice for each sign below. "
                "When the sign is recognized, the system will play your recorded voice."
            )
        sub = QLabel(sub_text)
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#8899AA; margin-bottom:8px;")
        layout.addWidget(sub)
        
        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel — Sign List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        list_lbl = QLabel("KSL Vocabulary Signs:")
        list_lbl.setStyleSheet("font-weight:600; color:#8899AA;")
        left_layout.addWidget(list_lbl)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #1A2332; border: 1px solid #2A3A4A; border-radius: 12px; padding: 6px; }"
            "QListWidget::item { padding: 10px; border-bottom: 1px solid #2A3A4A; border-radius: 6px; }"
            "QListWidget::item:selected { background-color: #1E2D3D; color: #00D4AA; border: 1px solid #00D4AA; }"
            "QListWidget::item:hover { background-color: #1E2D3D; }"
        )
        self.list_widget.currentItemChanged.connect(self._on_sign_selected)
        left_layout.addWidget(self.list_widget)
        splitter.addWidget(left_widget)
        
        # Right Panel — Record Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.detail_group = QGroupBox("Voice Details")
        self.detail_group.setStyleSheet(
            "QGroupBox { background-color: #1A2332; border: 1px solid #2A3A4A; border-radius: 12px; margin-top: 0px; padding-top: 20px; }"
            "QGroupBox::title { color: #00D4AA; font-size: 13px; font-weight: bold; subcontrol-origin: margin; subcontrol-position: top left; padding: 4px 12px; }"
        )
        group_layout = QVBoxLayout(self.detail_group)
        group_layout.setSpacing(14)
        group_layout.setContentsMargins(16, 24, 16, 16)
        
        # Selected sign info
        self.lbl_sign_name = QLabel("Sign: —")
        self.lbl_sign_name.setStyleSheet("font-size: 18px; font-weight: bold; color: #E8ECF1;")
        group_layout.addWidget(self.lbl_sign_name)
        
        self.lbl_translation = QLabel("Translation: —")
        self.lbl_translation.setStyleSheet("font-size: 13px; color: #8899AA;")
        group_layout.addWidget(self.lbl_translation)
        
        # Separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2A3A4A;")
        group_layout.addWidget(sep)
        
        # Input device selector
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("🎤 Microphone:"))
        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet(
            "QComboBox { background:#0F1419; color:#E8ECF1; padding:6px; border:1px solid #2A3A4A; border-radius:6px; min-width: 200px; }"
        )
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addWidget(self.device_combo)
        group_layout.addLayout(device_layout)
        
        # Status
        self.lbl_status = QLabel("Status: Reverted to AI Text-to-Speech (Default)")
        self.lbl_status.setStyleSheet("color:#8899AA; font-weight: 500;")
        group_layout.addWidget(self.lbl_status)
        
        # Progress/timer bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #0F1419; border: 1px solid #2A3A4A; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #FF6B6B; border-radius: 4px; }"
        )
        group_layout.addWidget(self.progress_bar)
        
        # Control Buttons
        btn_layout = QHBoxLayout()
        self.btn_record = QPushButton("🎙️ Record Voice")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #FF6B6B; color: #0F1419; border: none; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #FF8585; }"
            "QPushButton:pressed { background-color: #E05252; }"
        )
        self.btn_record.clicked.connect(self.toggle_recording)
        btn_layout.addWidget(self.btn_record)
        
        self.btn_play = QPushButton("▶️ Play Preview")
        self.btn_play.clicked.connect(self.play_recording)
        self.btn_play.setEnabled(False)
        btn_layout.addWidget(self.btn_play)

        self.btn_save = QPushButton("💾 Save Audio")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #00D4AA; color: #0F1419; border: none; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #00EABB; }"
            "QPushButton:pressed { background-color: #00B899; }"
            "QPushButton:disabled { background-color: #1A2332; color: #445566; border: 1px solid #2A3A4A; }"
        )
        self.btn_save.clicked.connect(self.save_recording)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)

        self.btn_edit = QPushButton("✏️ Edit Details")
        self.btn_edit.clicked.connect(self.edit_sign_details)
        self.btn_edit.setEnabled(False)
        btn_layout.addWidget(self.btn_edit)

        self.btn_delete = QPushButton("🗑️ Delete Voice")
        self.btn_delete.setStyleSheet(
            "QPushButton { border: 1px solid #FF6B6B; color: #FF6B6B; background-color: transparent; }"
            "QPushButton:hover { background-color: rgba(255, 107, 107, 0.1); }"
        )
        self.btn_delete.clicked.connect(self.delete_recording)
        self.btn_delete.setEnabled(False)
        btn_layout.addWidget(self.btn_delete)
        
        group_layout.addLayout(btn_layout)
        group_layout.addStretch()
        
        right_layout.addWidget(self.detail_group)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([350, 500])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        
        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setFixedWidth(120)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(btn_close)
        layout.addLayout(bottom_layout)

    def _populate_signs(self):
        self.list_widget.clear()
        all_labels = sorted(self.vocabulary.signs.keys())

        if self._missing_only:
            # Split into unrecorded (top) and recorded (bottom)
            missing = [l for l in all_labels if not self.tts._has_custom_audio(l)]
            recorded = [l for l in all_labels if self.tts._has_custom_audio(l)]

            if missing:
                hdr = QListWidgetItem(f"── {len(missing)} still need recording ──")
                hdr.setFlags(Qt.NoItemFlags)  # not selectable
                hdr.setForeground(Qt.yellow)
                hdr.setFont(QFont("Arial", 9, QFont.Bold))
                self.list_widget.addItem(hdr)

            for label in missing:
                self._add_sign_item(label)

            if recorded:
                hdr2 = QListWidgetItem(f"── {len(recorded)} already recorded ──")
                hdr2.setFlags(Qt.NoItemFlags)
                hdr2.setForeground(Qt.darkGreen)
                hdr2.setFont(QFont("Arial", 9, QFont.Bold))
                self.list_widget.addItem(hdr2)

            for label in recorded:
                self._add_sign_item(label)
        else:
            for label in all_labels:
                self._add_sign_item(label)

    def _add_sign_item(self, label):
        """Add a single sign entry to the list widget."""
        info = self.vocabulary.signs[label]
        kiny = info["kinyarwanda"]
        has_voice = self.tts._has_custom_audio(label)
        status_text = "🔊 Recorded" if has_voice else "🎙️ No Voice"
        item = QListWidgetItem(f"{kiny}   ({status_text})")
        item.setData(Qt.UserRole, label)
        if has_voice:
            item.setForeground(Qt.white)
        else:
            item.setForeground(Qt.yellow)
        self.list_widget.addItem(item)

    def _select_first_missing(self):
        """Select the first item in the list that has no recording."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            label = item.data(Qt.UserRole)
            if label and not self.tts._has_custom_audio(label):
                self.list_widget.setCurrentRow(i)
                return
        # All recorded — select first item
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _populate_audio_devices(self):
        """List recording devices via arecord -l and populate the combo box."""
        self.device_combo.clear()
        devices = ["pulse (default)"]   # always add PulseAudio default first
        try:
            out = subprocess.check_output(
                ["arecord", "-l"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if line.startswith("card "):
                    # e.g. "card 0: PCH [HDA Intel PCH], device 0: ALC3246 ..."
                    parts = line.split(":")
                    if len(parts) >= 2:
                        card_info = parts[0].strip()   # "card 0"
                        card_num = card_info.split()[-1]
                        dev_part = parts[1].split(",")[0].strip()  # " PCH [HDA Intel PCH]"
                        devices.append(f"hw:{card_num},0  ({dev_part.strip()})")
        except Exception:
            pass

        self.device_combo.addItems(devices)
        self.device_combo.setCurrentIndex(0)  # always default to pulse

    def _on_device_changed(self, idx):
        pass  # device is resolved in start_recording from combo text

    def _on_sign_selected(self, current, previous):
        if self.is_recording:
            self.stop_recording()
            
        if not current:
            self.selected_label = None
            self.lbl_sign_name.setText("Sign: —")
            self.lbl_translation.setText("Translation: —")
            self.lbl_status.setText("Status: Reverted to AI Text-to-Speech (Default)")
            self.btn_record.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_edit.setEnabled(False)
            self.btn_delete.setEnabled(False)
            return
            
        label = current.data(Qt.UserRole)
        self.selected_label = label
        
        info = self.vocabulary.signs[label]
        kiny = info["kinyarwanda"]
        eng = info.get("english", "")
        
        self.lbl_sign_name.setText(f"Sign: {kiny}")
        self.lbl_translation.setText(f"English translation: {eng}")
        
        self.btn_record.setEnabled(self.device_combo.currentText() != "No microphone detected")
        self.btn_edit.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_preview_controls()

    def update_preview_controls(self):
        if not self.selected_label:
            return

        has_voice = self.tts._has_custom_audio(self.selected_label)
        if has_voice:
            self.lbl_status.setText("Status: Custom Voice Recorded (WAV)")
            self.lbl_status.setStyleSheet("color:#00D4AA; font-weight: bold;")
            self.btn_play.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.btn_delete.setEnabled(True)
        else:
            self.lbl_status.setText("Status: Using System TTS (Default)")
            self.lbl_status.setStyleSheet("color:#8899AA; font-weight: 500;")
            self.btn_play.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.btn_delete.setEnabled(False)

    def update_list_item_status(self, label):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == label:
                info = self.vocabulary.signs[label]
                kiny = info["kinyarwanda"]
                has_voice = self.tts._has_custom_audio(label)
                status_text = "🔊 Recorded" if has_voice else "🎙️ No Voice"
                item.setText(f"{kiny}   ({status_text})")
                item.setForeground(Qt.white if has_voice else Qt.yellow)
                break
        self.voices_updated.emit()

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    # ── Recording duration (seconds) ────────────────────────────────────────
    MAX_RECORD_SECS = 5

    def _arecord_device(self):
        """Return the ALSA device string for arecord from the combo selection."""
        text = self.device_combo.currentText()
        if text.startswith("hw:"):
            return text.split()[0]   # e.g. "hw:0,0"
        return "pulse"               # default: PulseAudio

    def start_recording(self):
        if not self.selected_label:
            return

        # Stop any active playback so the mixer doesn't lock the device
        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass

        # Paths: record to a temp file; only replace the real file on success
        dest = os.path.abspath(
            os.path.join(self.tts.audio_dir, f"{self.selected_label}.wav")
        )
        temp = dest + ".tmp"
        self._record_dest_file = dest
        self._record_temp_file = temp

        # Clean up any leftover temp file
        if os.path.exists(temp):
            try:
                os.unlink(temp)
            except Exception:
                pass

        # Launch arecord — writes a proper RIFF/WAV file reliably
        device = self._arecord_device()
        try:
            self._record_proc = subprocess.Popen(
                ["arecord", "-D", device,
                 "-f", "S16_LE",   # signed 16-bit little-endian
                 "-r", "22050",    # 22 kHz
                 "-c", "1",        # mono
                 temp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            self.lbl_status.setText("❌ arecord not found — install alsa-utils")
            self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            return
        except Exception as e:
            self.lbl_status.setText(f"❌ Recording error: {e}")
            self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            logger.error(f"[VoiceManager] arecord launch error: {e}")
            return

        self.is_recording = True
        self.btn_record.setText("⏹️ Stop Recording")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #FFD93D; color: #0F1419; border: none; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #FFE566; }"
        )
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_delete.setEnabled(False)

        # Start timers
        self.elapsed_start_time = time.time()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #0F1419; border: 1px solid #2A3A4A; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #00D4AA; border-radius: 4px; }"
        )
        self.lbl_status.setText("🔴 Recording… Speak now!")
        self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight: bold;")

        self.record_timer.start(self.MAX_RECORD_SECS * 1000)
        self.elapsed_timer.start(50)   # 20 ticks per second

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_timer.stop()
        self.elapsed_timer.stop()

        # Immediately reset button so the UI feels responsive
        self.btn_record.setText("🎙️ Record Voice")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #FF6B6B; color: #0F1419; border: none; font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #FF8585; }"
            "QPushButton:pressed { background-color: #E05252; }"
        )
        self.lbl_status.setText("⏳ Saving recording…")
        self.lbl_status.setStyleSheet("color:#FFD93D; font-weight:bold;")

        proc         = self._record_proc
        temp         = self._record_temp_file
        dest         = self._record_dest_file
        label        = self.selected_label
        missing_only = self._missing_only

        def _finish():
            """Run in a background thread: terminate arecord, verify, rename."""
            # Terminate arecord (SIGTERM lets it flush & close the WAV header)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            time.sleep(0.15)  # brief wait for OS to flush disk buffers

            ok = (temp and dest and
                  os.path.exists(temp) and
                  os.path.getsize(temp) > 4096)   # at least a few KB

            if ok:
                # Safe swap: only remove old file after new one is confirmed
                try:
                    if os.path.exists(dest):
                        os.unlink(dest)
                    os.rename(temp, dest)
                    logger.info(f"[VoiceManager] ✔ Saved: {dest}")
                except Exception as e:
                    logger.error(f"[VoiceManager] ✘ Rename failed: {e}")
                    ok = False
            else:
                logger.info("[VoiceManager] ✘ Recording empty/missing — old file kept")
                if temp and os.path.exists(temp):
                    try:
                        os.unlink(temp)
                    except Exception:
                        pass

            # Update UI back on the main thread via a queued timer
            QTimer.singleShot(0, lambda: self._on_recording_saved(label, ok, missing_only))

        threading.Thread(target=_finish, daemon=True).start()

    def _on_recording_saved(self, label, success, missing_only):
        """Called on the main thread once the background save thread finishes."""
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #0F1419; border: 1px solid #2A3A4A; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #FF6B6B; border-radius: 4px; }"
        )
        if success:
            self.lbl_status.setText("✅ Recording saved!")
            self.lbl_status.setStyleSheet("color:#00D4AA; font-weight:bold;")
        else:
            self.lbl_status.setText("❌ Recording failed — please try again")
            self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")

        self.update_list_item_status(label)
        self.update_preview_controls()

        if missing_only and label:
            QTimer.singleShot(600, self._advance_to_next_missing)

    def _on_elapsed_tick(self):
        elapsed = time.time() - self.elapsed_start_time
        if elapsed >= self.MAX_RECORD_SECS:
            self.stop_recording()
            return

        pct = int((elapsed / self.MAX_RECORD_SECS) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"🔴 Recording: {elapsed:.1f}s / {self.MAX_RECORD_SECS}.0s — Speak now!")

    def play_recording(self):
        if not self.selected_label:
            return

        # Resolve to ABSOLUTE path so subprocess never fails due to CWD
        rel_path = self.tts._resolve_path(self.selected_label)
        if not rel_path:
            self.lbl_status.setText("⚠️  No audio file found.")
            self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            return
        path = os.path.abspath(rel_path)
        if not os.path.exists(path):
            self.lbl_status.setText("⚠️  Audio file missing.")
            self.lbl_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            return

        logger.info(f"[VoiceManager] Preview: {path}")

        try:
            import wave
            with wave.open(path, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
        except Exception:
            duration = 2.0

        self.lbl_status.setText(f"▶️  Playing ({duration:.1f}s)...")
        self.lbl_status.setStyleSheet("color:#FFD93D; font-weight:bold;")

        def _play():
            # Use tts._play_file() — the exact same proven code path
            # that plays sign recognition audio successfully.
            try:
                self.tts._play_file(path)
                logger.info(f"[VoiceManager] Preview done: {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"[VoiceManager] Preview error: {e}")

        threading.Thread(target=_play, daemon=True).start()
        QTimer.singleShot(int(duration * 1000) + 300, self.update_preview_controls)

    def save_recording(self):
        """Export the recorded WAV to a user-chosen location."""
        if not self.selected_label:
            return

        src = self.tts._resolve_path(self.selected_label)
        if not src:
            QMessageBox.warning(self, "No Recording", "No audio file found for this sign.")
            return
        src = os.path.abspath(src)

        # Build a default filename: <kinyarwanda_word>.wav
        info = self.vocabulary.signs.get(self.selected_label, {})
        default_name = f"{info.get('kinyarwanda', self.selected_label)}.wav"

        # Open save-file dialog
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio Recording",
            os.path.join(os.path.expanduser("~"), default_name),
            "WAV Audio (*.wav);;All Files (*)"
        )
        if not dest:
            return  # user cancelled

        try:
            import shutil
            shutil.copy2(src, dest)
            self.lbl_status.setText(f"✅ Saved to: {os.path.basename(dest)}")
            self.lbl_status.setStyleSheet("color:#00D4AA; font-weight:bold;")
            logger.info(f"[VoiceManager] Audio exported → {dest}")
            QTimer.singleShot(3000, self.update_preview_controls)
        except Exception as e:
            logger.error(f"[VoiceManager] Save error: {e}")
            QMessageBox.critical(self, "Save Failed", f"Could not save file:\n{e}")




    def _advance_to_next_missing(self):
        """Select the next sign that still has no recording (missing_only mode)."""
        current_row = self.list_widget.currentRow()
        for i in range(current_row + 1, self.list_widget.count()):
            item = self.list_widget.item(i)
            label = item.data(Qt.UserRole)
            if label and not self.tts._has_custom_audio(label):
                self.list_widget.setCurrentRow(i)
                return
        # Also check from the top in case we wrapped
        for i in range(0, current_row):
            item = self.list_widget.item(i)
            label = item.data(Qt.UserRole)
            if label and not self.tts._has_custom_audio(label):
                self.list_widget.setCurrentRow(i)
                return

    def delete_recording(self):
        if not self.selected_label:
            return
            
        confirm = QMessageBox.question(
            self, "Delete Custom Voice",
            f"Are you sure you want to delete your custom voice recording for the sign '{self.selected_label}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return
            
        path = os.path.join(self.tts.audio_dir, f"{self.selected_label}.wav")
        if os.path.exists(path):
            try:
                import pygame
                # Stop channel playback to release file lock
                pygame.mixer.stop()
                os.unlink(path)
                
                self.lbl_status.setText("Status: Custom voice deleted (reverted to TTS)")
                self.progress_bar.setValue(0)
                
                self.update_list_item_status(self.selected_label)
                self.update_preview_controls()
            except Exception as e:
                logger.error(f"[VoiceManager] Delete error: {e}")
                QMessageBox.critical(self, "Delete Error", f"Could not delete audio file: {e}")

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()

    def edit_sign_details(self):
        if not self.selected_label:
            return
            
        info = self.vocabulary.signs[self.selected_label]
        kiny = info["kinyarwanda"]
        eng = info.get("english", "")
        cat = info.get("category", "other")
        
        dlg = EditSignDetailsDialog(self.selected_label, kiny, eng, cat, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new_kiny, new_eng, new_cat = dlg.get_details()
            if not new_kiny:
                QMessageBox.warning(self, "Validation Error", "Kinyarwanda word cannot be empty.")
                return
                
            if self.vocabulary.update_sign(self.selected_label, new_kiny, new_eng, new_cat):
                # Update UI elements
                self.lbl_sign_name.setText(f"Sign: {new_kiny}")
                self.lbl_translation.setText(f"English translation: {new_eng}")
                
                # Refresh list items
                self.update_list_item_status(self.selected_label)
                QMessageBox.information(self, "Success", "Sign details updated successfully!")


class EditSignDetailsDialog(QDialog):
    """Sub-dialog to edit Kinyarwanda word, English translation, and category of an existing sign."""
    
    def __init__(self, label, kinyarwanda, english, category, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Sign Details — {label}")
        self.setMinimumWidth(400)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Notice label
        info_lbl = QLabel(
            "Note: The internal sign label (slug) cannot be changed to prevent breaking the trained neural network model. "
            "You can edit the spoken Kinyarwanda word, English translation, and category below."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #8899AA; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(info_lbl)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.inp_kiny = QLineEdit(kinyarwanda)
        self.inp_kiny.setStyleSheet("background:#0F1419; color:#E8ECF1; padding:6px; border:1px solid #2A3A4A; border-radius:6px;")
        form.addRow("Kinyarwanda Word:", self.inp_kiny)
        
        self.inp_eng = QLineEdit(english)
        self.inp_eng.setStyleSheet(self.inp_kiny.styleSheet())
        form.addRow("English Meaning:", self.inp_eng)
        
        self.inp_cat = QComboBox()
        self.inp_cat.addItems(["greetings", "common", "emergency", "body", "alphabet", "other"])
        self.inp_cat.setCurrentText(category)
        self.inp_cat.setStyleSheet("QComboBox { background:#0F1419; color:#E8ECF1; padding:6px; border:1px solid #2A3A4A; border-radius:6px; }")
        form.addRow("Category:", self.inp_cat)
        
        layout.addLayout(form)
        
        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        self.buttons.setStyleSheet(
            "QPushButton { background-color: #1E2D3D; color: #E8ECF1; border: 1px solid #2A3A4A; padding: 6px 12px; min-width: 80px; }"
            "QPushButton:hover { border-color: #00D4AA; }"
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
    def get_details(self):
        return (
            self.inp_kiny.text().strip(),
            self.inp_eng.text().strip(),
            self.inp_cat.currentText()
        )
