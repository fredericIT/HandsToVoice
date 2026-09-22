"""
HandsToVoice — Edit Sign Dialog
Standalone dialog allowing users to edit the Kinyarwanda word, English translation,
category, and custom voice recording for any existing sign.
"""

import os
import time
import subprocess
import threading

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QFormLayout, QLineEdit, QComboBox,
    QProgressBar, QMessageBox, QWidget, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from src.logger import get_logger
from src.gui.voice_manager_dialog import _trim_silence, _is_clipped

logger = get_logger("gui.edit_sign_dialog")


class EditSignDialog(QDialog):
    """
    Dialog to edit existing sign details and re-record their custom voice.
    Accessible directly from the main window toolbar.
    """

    sign_updated = pyqtSignal()

    def __init__(self, vocabulary, tts, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.tts = tts
        self.selected_label = None

        # Voice recording — uses arecord directly (see start_recording),
        # not Qt Multimedia's QAudioRecorder: that backend depends on
        # GStreamer's PipeWire/PulseAudio input plugin being present and
        # correctly configured, which isn't guaranteed on this system, and
        # it failed silently (recorded pure silence with no error) rather
        # than surfacing the problem.
        self.is_recording = False
        self._record_proc = None
        self._record_temp_file = None
        self._record_dest_file = None

        self.record_timer = QTimer(self)
        self.record_timer.setSingleShot(True)
        self.record_timer.timeout.connect(self.stop_recording)

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._on_elapsed_tick)
        self.elapsed_start = 0

        self.setWindowTitle("HandsToVoice — Edit Existing Signs")
        self.setMinimumSize(900, 540)
        self.setModal(True)

        self._build_ui()
        self._populate_signs()
        self._populate_devices()

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        # Header
        title = QLabel("✏️  Edit Existing Signs")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:22px; font-weight:700; color:#7C4DFF; padding-bottom:6px;"
        )
        root.addWidget(title)

        sub = QLabel(
            "Select a sign from the list, update its details, re-record your voice, then click Save."
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#8899AA; margin-bottom:4px;")
        root.addWidget(sub)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: sign list ──────────────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("All Signs:")
        lbl.setStyleSheet("font-weight:600; color:#8899AA;")
        ll.addWidget(lbl)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:#1A2332; border:1px solid #2A3A4A;"
            " border-radius:12px; padding:6px; }"
            "QListWidget::item { padding:10px; border-bottom:1px solid #2A3A4A;"
            " border-radius:6px; }"
            "QListWidget::item:selected { background:#1E2D3D; color:#7C4DFF;"
            " border:1px solid #7C4DFF; }"
            "QListWidget::item:hover { background:#1E2D3D; }"
        )
        self.list_widget.currentItemChanged.connect(self._on_sign_selected)
        ll.addWidget(self.list_widget)
        splitter.addWidget(left)

        # ── Right: edit form ─────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)

        # Details group
        details_grp = QGroupBox("Sign Details")
        details_grp.setStyleSheet(
            "QGroupBox { background:#1A2332; border:1px solid #2A3A4A;"
            " border-radius:12px; margin-top:0px; padding-top:20px; }"
            "QGroupBox::title { color:#7C4DFF; font-size:13px; font-weight:bold;"
            " subcontrol-origin:margin; subcontrol-position:top left; padding:4px 12px; }"
        )
        form = QFormLayout(details_grp)
        form.setSpacing(10)
        form.setContentsMargins(16, 28, 16, 16)

        field_style = (
            "background:#0F1419; color:#E8ECF1; padding:8px;"
            " border:1px solid #2A3A4A; border-radius:6px; font-size:13px;"
        )

        self.lbl_slug = QLabel("—")
        self.lbl_slug.setStyleSheet("color:#4A5568; font-style:italic;")
        form.addRow("Internal label (read-only):", self.lbl_slug)

        self.inp_kiny = QLineEdit()
        self.inp_kiny.setPlaceholderText("Kinyarwanda word")
        self.inp_kiny.setStyleSheet(field_style)
        form.addRow("Kinyarwanda Word:*", self.inp_kiny)

        self.inp_eng = QLineEdit()
        self.inp_eng.setPlaceholderText("English meaning")
        self.inp_eng.setStyleSheet(field_style)
        form.addRow("English Meaning:", self.inp_eng)

        self.inp_cat = QComboBox()
        self.inp_cat.addItems(
            ["greetings", "common", "emergency", "body", "alphabet", "other"]
        )
        self.inp_cat.setStyleSheet(
            "QComboBox { background:#0F1419; color:#E8ECF1; padding:8px;"
            " border:1px solid #2A3A4A; border-radius:6px; }"
        )
        form.addRow("Category:", self.inp_cat)
        rl.addWidget(details_grp)

        # Voice group
        voice_grp = QGroupBox("Custom Voice Recording")
        voice_grp.setStyleSheet(details_grp.styleSheet())
        vl = QVBoxLayout(voice_grp)
        vl.setContentsMargins(16, 28, 16, 16)
        vl.setSpacing(10)

        # Microphone selector
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("🎤 Microphone:"))
        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet(self.inp_cat.styleSheet())
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        dev_row.addWidget(self.device_combo)
        vl.addLayout(dev_row)

        self.lbl_voice_status = QLabel("No custom voice recorded — will use AI TTS")
        self.lbl_voice_status.setStyleSheet("color:#8899AA;")
        vl.addWidget(self.lbl_voice_status)

        self.voice_progress = QProgressBar()
        self.voice_progress.setRange(0, 100)
        self.voice_progress.setValue(0)
        self.voice_progress.setTextVisible(False)
        self.voice_progress.setFixedHeight(8)
        self.voice_progress.setStyleSheet(
            "QProgressBar { background:#0F1419; border:1px solid #2A3A4A; border-radius:4px; }"
            "QProgressBar::chunk { background:#FF6B6B; border-radius:4px; }"
        )
        vl.addWidget(self.voice_progress)

        voice_btns = QHBoxLayout()
        self.btn_record = QPushButton("🎙️  Record Voice")
        self.btn_record.setStyleSheet(
            "QPushButton { background:#FF6B6B; color:#0F1419; border:none;"
            " font-weight:bold; padding:10px; border-radius:8px; }"
            "QPushButton:hover { background:#FF8585; }"
            "QPushButton:disabled { background:#2A3A4A; color:#4A5568; }"
        )
        self.btn_record.clicked.connect(self.toggle_recording)
        voice_btns.addWidget(self.btn_record)

        self.btn_play = QPushButton("▶️  Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_recording)
        voice_btns.addWidget(self.btn_play)

        self.btn_delete_voice = QPushButton("🗑️  Delete Voice")
        self.btn_delete_voice.setEnabled(False)
        self.btn_delete_voice.setStyleSheet(
            "QPushButton { border:1px solid #FF6B6B; color:#FF6B6B; background:transparent; border-radius:8px; }"
            "QPushButton:hover { background:rgba(255,107,107,0.1); }"
            "QPushButton:disabled { border-color:#2A3A4A; color:#4A5568; }"
        )
        self.btn_delete_voice.clicked.connect(self.delete_voice)
        voice_btns.addWidget(self.btn_delete_voice)

        vl.addLayout(voice_btns)
        rl.addWidget(voice_grp)

        # Save / Cancel
        bottom = QHBoxLayout()
        bottom.addStretch()

        self.btn_save = QPushButton("💾  Save Changes")
        self.btn_save.setEnabled(False)
        self.btn_save.setFixedWidth(160)
        self.btn_save.setStyleSheet(
            "QPushButton { background:#7C4DFF; color:white; font-weight:700;"
            " border:none; padding:11px; border-radius:10px; }"
            "QPushButton:hover { background:#9C7CFF; }"
            "QPushButton:disabled { background:#3A2A6A; color:#7A6A9A; }"
        )
        self.btn_save.clicked.connect(self.save_changes)
        bottom.addWidget(self.btn_save)

        btn_close = QPushButton("Close")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)

        rl.addLayout(bottom)
        splitter.addWidget(right)

        splitter.setSizes([320, 580])
        root.addWidget(splitter)

    # ── Data Population ──────────────────────────────────────────────────────

    def _populate_signs(self):
        self.list_widget.clear()
        for label in sorted(self.vocabulary.signs.keys()):
            info = self.vocabulary.signs[label]
            kiny = info["kinyarwanda"]
            has_voice = self.tts._has_custom_audio(label)
            icon = "🔊" if has_voice else "🎙️"
            item = QListWidgetItem(f"{icon}  {kiny}")
            item.setData(Qt.UserRole, label)
            item.setForeground(Qt.white if has_voice else Qt.gray)
            self.list_widget.addItem(item)

    def _populate_devices(self):
        """List recording devices via arecord -l — same source of truth as
        VoiceManagerDialog, so device selection behaves identically."""
        self.device_combo.clear()
        devices = ["default (system)"]
        try:
            out = subprocess.check_output(
                ["arecord", "-l"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if line.startswith("card "):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        card_num = parts[0].strip().split()[-1]
                        dev_part = parts[1].split(",")[0].strip()
                        devices.append(f"plughw:{card_num},0  ({dev_part})")
        except Exception:
            pass
        self.device_combo.addItems(devices)
        self.device_combo.setCurrentIndex(0)

    def _arecord_device(self):
        text = self.device_combo.currentText()
        if text.startswith("plughw:"):
            return text.split()[0]
        return "default"

    # ── Slot: sign selected ──────────────────────────────────────────────────

    def _on_sign_selected(self, current, _previous):
        if self.is_recording:
            self.stop_recording()

        if not current:
            self.selected_label = None
            self._clear_form()
            return

        label = current.data(Qt.UserRole)
        self.selected_label = label
        info = self.vocabulary.signs[label]

        self.lbl_slug.setText(label)
        self.inp_kiny.setText(info["kinyarwanda"])
        self.inp_eng.setText(info.get("english", ""))
        self.inp_cat.setCurrentText(info.get("category", "other"))

        self.btn_save.setEnabled(True)
        self.btn_record.setEnabled(
            self.device_combo.currentText() != "No microphone detected"
        )
        self.voice_progress.setValue(0)
        self._refresh_voice_status()

    def _on_device_changed(self, idx):
        pass  # device is resolved in start_recording from combo text

    def _clear_form(self):
        self.lbl_slug.setText("—")
        self.inp_kiny.clear()
        self.inp_eng.clear()
        self.inp_cat.setCurrentIndex(0)
        self.btn_save.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_delete_voice.setEnabled(False)
        self.lbl_voice_status.setText("Select a sign to begin editing")
        self.lbl_voice_status.setStyleSheet("color:#8899AA;")

    def _refresh_voice_status(self):
        if not self.selected_label:
            return
        has_voice = self.tts._has_custom_audio(self.selected_label)
        if has_voice:
            self.lbl_voice_status.setText("✅  Custom voice recorded (WAV)")
            self.lbl_voice_status.setStyleSheet("color:#00D4AA; font-weight:bold;")
            self.btn_play.setEnabled(True)
            self.btn_delete_voice.setEnabled(True)
        else:
            self.lbl_voice_status.setText("No custom voice — will use AI TTS")
            self.lbl_voice_status.setStyleSheet("color:#8899AA;")
            self.btn_play.setEnabled(False)
            self.btn_delete_voice.setEnabled(False)

    def _refresh_list_item(self, label):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == label:
                info = self.vocabulary.signs[label]
                kiny = info["kinyarwanda"]
                has_voice = self.tts._has_custom_audio(label)
                icon = "🔊" if has_voice else "🎙️"
                item.setText(f"{icon}  {kiny}")
                item.setForeground(Qt.white if has_voice else Qt.gray)
                break

    # ── Save ────────────────────────────────────────────────────────────────

    def save_changes(self):
        if not self.selected_label:
            return

        new_kiny = self.inp_kiny.text().strip()
        if not new_kiny:
            QMessageBox.warning(self, "Validation Error",
                                "Kinyarwanda word cannot be empty.")
            return

        new_eng = self.inp_eng.text().strip()
        new_cat = self.inp_cat.currentText()

        if self.vocabulary.update_sign(self.selected_label, new_kiny, new_eng, new_cat):
            self._refresh_list_item(self.selected_label)
            self.sign_updated.emit()
            QMessageBox.information(self, "Saved",
                                    f"Sign '{self.selected_label}' updated successfully!")
        else:
            QMessageBox.critical(self, "Error", "Could not save changes to labels.json.")

    # ── Voice recording ──────────────────────────────────────────────────────

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.selected_label:
            return

        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass

        dest = os.path.abspath(
            os.path.join(self.tts.audio_dir, f"{self.selected_label}.wav")
        )
        temp = dest + ".tmp"
        self._record_dest_file = dest
        self._record_temp_file = temp
        if os.path.exists(temp):
            try:
                os.unlink(temp)
            except Exception:
                pass

        device = self._arecord_device()
        try:
            self._record_proc = subprocess.Popen(
                ["arecord", "-D", device, "-f", "S16_LE", "-r", "22050", "-c", "1", temp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.lbl_voice_status.setText("❌ arecord not found — install alsa-utils")
            self.lbl_voice_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            return
        except Exception as e:
            self.lbl_voice_status.setText(f"❌ Recording error: {e}")
            self.lbl_voice_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
            logger.error(f"[EditSign] arecord launch error: {e}")
            return

        self.is_recording = True

        self.btn_record.setText("⏹️  Stop Recording")
        self.btn_record.setStyleSheet(
            "QPushButton { background:#FFD93D; color:#0F1419; border:none;"
            " font-weight:bold; padding:10px; border-radius:8px; }"
        )
        self.btn_play.setEnabled(False)
        self.btn_delete_voice.setEnabled(False)

        self.elapsed_start = time.time()
        self.voice_progress.setValue(0)
        self.voice_progress.setStyleSheet(
            "QProgressBar { background:#0F1419; border:1px solid #2A3A4A; border-radius:4px; }"
            "QProgressBar::chunk { background:#00D4AA; border-radius:4px; }"
        )
        self.lbl_voice_status.setText("🔴  Recording... Speak now!")
        self.lbl_voice_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")

        self.record_timer.start(3000)
        self.elapsed_timer.start(50)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.record_timer.stop()
        self.elapsed_timer.stop()

        self.btn_record.setText("🎙️  Record Voice")
        self.btn_record.setStyleSheet(
            "QPushButton { background:#FF6B6B; color:#0F1419; border:none;"
            " font-weight:bold; padding:10px; border-radius:8px; }"
            "QPushButton:hover { background:#FF8585; }"
        )
        self.lbl_voice_status.setText("⏳ Saving recording…")
        self.lbl_voice_status.setStyleSheet("color:#FFD93D; font-weight:bold;")

        proc = self._record_proc
        temp = self._record_temp_file
        dest = self._record_dest_file
        label = self.selected_label

        def _finish():
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            time.sleep(0.15)

            ok = temp and dest and os.path.exists(temp) and os.path.getsize(temp) > 4096
            no_voice = False
            clipped = False
            if ok:
                try:
                    if not _trim_silence(temp):
                        ok = False
                        no_voice = True
                except Exception as e:
                    logger.error(f"[EditSign] Trim error: {e}")

            if ok:
                try:
                    if os.path.exists(dest):
                        os.unlink(dest)
                    os.rename(temp, dest)
                    logger.info(f"[EditSign] ✔ Saved (trimmed): {dest}")
                    try:
                        clipped = _is_clipped(dest)
                    except Exception:
                        clipped = False
                except Exception as e:
                    logger.error(f"[EditSign] ✘ Rename failed: {e}")
                    ok = False
            else:
                logger.info(
                    f"[EditSign] ✘ {'No voice detected' if no_voice else 'Recording empty/missing'}"
                    " — old file kept"
                )
                if temp and os.path.exists(temp):
                    try:
                        os.unlink(temp)
                    except Exception:
                        pass

            QTimer.singleShot(0, lambda: self._on_recording_saved(label, ok, no_voice, clipped))

        threading.Thread(target=_finish, daemon=True).start()

    def _on_recording_saved(self, label, success, no_voice, clipped=False):
        self.voice_progress.setValue(100)
        self.voice_progress.setStyleSheet(
            "QProgressBar { background:#0F1419; border:1px solid #2A3A4A; border-radius:4px; }"
            "QProgressBar::chunk { background:#FF6B6B; border-radius:4px; }"
        )
        if success:
            self.lbl_voice_status.setText("✅ Recording saved! (trimmed to spoken audio)")
            self.lbl_voice_status.setStyleSheet("color:#00D4AA; font-weight:bold;")
        elif no_voice:
            self.lbl_voice_status.setText("⚠️ No voice detected — check mic and speak clearly, then retry")
            self.lbl_voice_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")
        else:
            self.lbl_voice_status.setText("❌ Recording failed — please try again")
            self.lbl_voice_status.setStyleSheet("color:#FF6B6B; font-weight:bold;")

        if label == self.selected_label:
            self._refresh_list_item(label)
            self._refresh_voice_status()

        # Set last: _refresh_voice_status() overwrites the status label.
        if success and clipped:
            self.lbl_voice_status.setText(
                "⚠️ Saved, but too loud — it will sound distorted. "
                "Move a little back from the mic and record again."
            )
            self.lbl_voice_status.setStyleSheet("color:#FFD93D; font-weight:bold;")

    def _on_elapsed_tick(self):
        elapsed = time.time() - self.elapsed_start
        if elapsed >= 3.0:
            self.stop_recording()
            return
        pct = int((elapsed / 3.0) * 100)
        self.voice_progress.setValue(pct)
        self.lbl_voice_status.setText(
            f"🔴  Recording: {elapsed:.1f}s / 3.0s — Speak now!"
        )

    def play_recording(self):
        if not self.selected_label:
            return
        path = os.path.join(self.tts.audio_dir, f"{self.selected_label}.wav")
        if not os.path.exists(path):
            QMessageBox.warning(self, "No Recording",
                                "No recording found. Please record your voice first.")
            return

        # Stop any previous preview playback
        if hasattr(self, '_preview_proc') and self._preview_proc:
            try:
                self._preview_proc.terminate()
            except Exception:
                pass

        try:
            import wave
            with wave.open(path, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
        except Exception:
            duration = 2.0

        self.lbl_voice_status.setText(f"▶️  Playing ({duration:.1f}s)...")
        self.lbl_voice_status.setStyleSheet("color:#FFD93D; font-weight:bold;")

        def _play():
            try:
                # pw-play works with PipeWire (the audio server on this system)
                self._preview_proc = subprocess.Popen(
                    ['pw-play', path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._preview_proc.wait()
            except FileNotFoundError:
                try:
                    # Fallback: aplay (ALSA direct)
                    self._preview_proc = subprocess.Popen(
                        ['aplay', '-q', path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self._preview_proc.wait()
                except Exception as e2:
                    logger.error(f"[EditSignDialog] Playback fallback failed: {e2}")
            except Exception as e:
                logger.error(f"[EditSignDialog] Playback error: {e}")

        t = threading.Thread(target=_play, daemon=True)
        t.start()

        QTimer.singleShot(int(duration * 1000) + 300, self._refresh_voice_status)

    def delete_voice(self):
        if not self.selected_label:
            return
        path = os.path.join(self.tts.audio_dir, f"{self.selected_label}.wav")
        if not os.path.exists(path):
            return
        confirm = QMessageBox.question(
            self, "Delete Voice",
            f"Delete the custom recording for '{self.selected_label}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            import pygame
            pygame.mixer.stop()
            os.unlink(path)
            self._refresh_list_item(self.selected_label)
            self._refresh_voice_status()
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", str(e))

    def done(self, result):
        # The Close/Save buttons call accept(), which never fires closeEvent.
        self.stop_recording()
        super().done(result)

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()
