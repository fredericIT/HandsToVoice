"""
HandsToVoice — Delete Sign Dialog
Allows the user to select and permanently delete an existing sign,
including all associated data files (videos, sequences, audio).
"""

import os
import csv
import shutil

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QFrame, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.logger import get_logger

logger = get_logger("gui.delete_sign_dialog")


VIDEO_DIR = "data/videos"
SEQUENCE_DIR = "data/sequences"
AUDIO_DIR = "data/audio"


class DeleteSignDialog(QDialog):
    """Dialog to delete an existing sign and all its associated data."""
    sign_deleted = pyqtSignal()

    def __init__(self, vocabulary, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.setWindowTitle("HandsToVoice — Delete Sign")
        self.setMinimumSize(520, 480)
        self.setModal(True)
        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # Title
        title = QLabel("🗑️ Delete Existing Sign")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:22px; font-weight:700; color:#FF6B6B; padding:8px;")
        root.addWidget(title)

        # Warning banner
        warning = QFrame()
        warning.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "  stop:0 #3D1F1F, stop:1 #2D1515);"
            "  border:1px solid #FF6B6B; border-radius:10px; padding:12px; }")
        wl = QHBoxLayout(warning)
        wl.setContentsMargins(12, 8, 12, 8)
        warn_icon = QLabel("⚠️")
        warn_icon.setStyleSheet("font-size:24px; background:transparent;")
        wl.addWidget(warn_icon)
        warn_text = QLabel(
            "This action is permanent. The sign, its videos, landmark sequences, "
            "and custom audio recording will be deleted from disk.")
        warn_text.setWordWrap(True)
        warn_text.setStyleSheet(
            "color:#FF6B6B; font-size:13px; font-weight:600; background:transparent;")
        wl.addWidget(warn_text, stretch=1)
        root.addWidget(warning)

        # Sign selector
        sel_group = QGroupBox("Select Sign to Delete")
        sl = QVBoxLayout(sel_group)
        self.sign_combo = QComboBox()
        self.sign_combo.setStyleSheet(
            "QComboBox { background:#1A2332; color:#E8ECF1; padding:10px;"
            "  border:1px solid #2A3A4A; border-radius:8px; font-size:14px; }")
        self._populate_combo()
        self.sign_combo.currentIndexChanged.connect(self._update_preview)
        sl.addWidget(self.sign_combo)
        root.addWidget(sel_group)

        # Preview group
        preview_group = QGroupBox("Sign Details")
        pl = QVBoxLayout(preview_group)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "color:#E8ECF1; font-size:13px; padding:4px;")
        pl.addWidget(self.info_label)

        self.files_log = QTextEdit()
        self.files_log.setReadOnly(True)
        self.files_log.setMaximumHeight(140)
        self.files_log.setStyleSheet(
            "background:#0F1419; color:#8899AA; border:1px solid #2A3A4A;"
            "border-radius:8px; padding:8px; font-family:monospace; font-size:11px;")
        pl.addWidget(self.files_log)
        root.addWidget(preview_group, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("clearButton")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()

        self.delete_btn = QPushButton("🗑️ Delete Permanently")
        self.delete_btn.setStyleSheet(
            "QPushButton { background:#FF4444; color:white; font-weight:bold;"
            "  padding:12px 24px; border:none; border-radius:8px; font-size:14px; }"
            "QPushButton:hover { background:#FF6666; }"
            "QPushButton:disabled { background:#3A2020; color:#666; }")
        self.delete_btn.clicked.connect(self._confirm_delete)
        btn_layout.addWidget(self.delete_btn)
        root.addLayout(btn_layout)

    def _populate_combo(self):
        """Fill the combo box with all sign labels."""
        self.sign_combo.clear()
        labels = self.vocabulary.get_all_labels()
        if not labels:
            self.sign_combo.addItem("(no signs available)")
            self.delete_btn.setEnabled(False)
        else:
            for label in labels:
                kiny = self.vocabulary.get_kinyarwanda(label)
                eng = self.vocabulary.get_english(label)
                display = f"{label}  —  {kiny}"
                if eng:
                    display += f" ({eng})"
                self.sign_combo.addItem(display, userData=label)

    def _update_preview(self):
        """Update the preview panel when the selected sign changes."""
        label = self.sign_combo.currentData()
        if not label:
            self.info_label.setText("No sign selected.")
            self.files_log.clear()
            self.delete_btn.setEnabled(False)
            return

        self.delete_btn.setEnabled(True)
        info = self.vocabulary.signs.get(label, {})
        kiny = info.get("kinyarwanda", "—")
        eng = info.get("english", "—")
        cat = info.get("category", "—")
        sign_id = info.get("id", "?")

        self.info_label.setText(
            f"<b>Label:</b> {label}  &nbsp;|&nbsp;  "
            f"<b>ID:</b> {sign_id}  &nbsp;|&nbsp;  "
            f"<b>Category:</b> {cat}<br>"
            f"<b>Kinyarwanda:</b> {kiny}  &nbsp;|&nbsp;  "
            f"<b>English:</b> {eng}")

        # List files that will be deleted
        files_to_delete = self._find_files(label)
        if files_to_delete:
            self.files_log.setPlainText(
                f"Files to be deleted ({len(files_to_delete)}):\n\n"
                + "\n".join(f"  • {f}" for f in files_to_delete))
        else:
            self.files_log.setPlainText("No associated data files found on disk.")

    def _find_files(self, label):
        """Find all data files associated with a sign label."""
        files = []

        # Videos folder
        video_dir = os.path.join(VIDEO_DIR, label)
        if os.path.isdir(video_dir):
            for f in os.listdir(video_dir):
                files.append(os.path.join(video_dir, f))
            files.append(f"{video_dir}/  (folder)")

        # Sequence .npy files
        if os.path.isdir(SEQUENCE_DIR):
            for f in os.listdir(SEQUENCE_DIR):
                if f.startswith(f"{label}_") and f.endswith(".npy"):
                    files.append(os.path.join(SEQUENCE_DIR, f))

        # Audio recording
        audio_file = os.path.join(AUDIO_DIR, f"{label}.wav")
        if os.path.exists(audio_file):
            files.append(audio_file)

        # Training data (.npy in data/processed)
        processed_dir = "data/processed"
        if os.path.isdir(processed_dir):
            for f in os.listdir(processed_dir):
                if f.startswith(f"{label}_") or f.startswith(f"{label}."):
                    files.append(os.path.join(processed_dir, f))

        return files

    def _delete_files(self, label):
        """Remove all data files associated with a sign label."""
        deleted = []

        # Videos folder
        video_dir = os.path.join(VIDEO_DIR, label)
        if os.path.isdir(video_dir):
            shutil.rmtree(video_dir, ignore_errors=True)
            deleted.append(video_dir)

        # Sequence .npy files
        if os.path.isdir(SEQUENCE_DIR):
            for f in os.listdir(SEQUENCE_DIR):
                if f.startswith(f"{label}_") and f.endswith(".npy"):
                    fpath = os.path.join(SEQUENCE_DIR, f)
                    try:
                        os.remove(fpath)
                        deleted.append(fpath)
                    except OSError:
                        pass

        # Clean labels.csv in sequences
        labels_csv = os.path.join(SEQUENCE_DIR, "labels.csv")
        if os.path.exists(labels_csv):
            try:
                with open(labels_csv, newline="") as f:
                    rows = list(csv.DictReader(f))
                rows = [r for r in rows if r.get("label") != label]
                with open(labels_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["file", "label"])
                    w.writeheader()
                    w.writerows(rows)
            except Exception as e:
                logger.error(f"[DeleteSign] Error updating labels.csv: {e}")

        # Audio recording
        audio_file = os.path.join(AUDIO_DIR, f"{label}.wav")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                deleted.append(audio_file)
            except OSError:
                pass

        # Training data in processed
        processed_dir = "data/processed"
        if os.path.isdir(processed_dir):
            for f in os.listdir(processed_dir):
                if f.startswith(f"{label}_") or f.startswith(f"{label}."):
                    fpath = os.path.join(processed_dir, f)
                    try:
                        os.remove(fpath)
                        deleted.append(fpath)
                    except OSError:
                        pass

        return deleted

    def _confirm_delete(self):
        """Ask for confirmation then delete the sign."""
        label = self.sign_combo.currentData()
        if not label:
            return

        kiny = self.vocabulary.get_kinyarwanda(label)

        reply = QMessageBox.warning(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete the sign "
            f"'{kiny}' ({label})?\n\n"
            f"This will remove all videos, sequences, audio, "
            f"and vocabulary entry. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete data files first
        deleted_files = self._delete_files(label)
        logger.info(f"[DeleteSign] Removed {len(deleted_files)} file(s)/folder(s) for '{label}'")

        # Delete from vocabulary
        success = self.vocabulary.delete_sign(label)

        if success:
            QMessageBox.information(
                self, "Sign Deleted",
                f"Sign '{kiny}' ({label}) has been deleted.\n\n"
                f"Removed {len(deleted_files)} file(s)/folder(s).\n\n"
                f"Note: The trained models (static & LSTM) still contain this sign. "
                f"Retrain to fully remove it from recognition.")
            self.sign_deleted.emit()
            self.accept()
        else:
            QMessageBox.critical(
                self, "Delete Failed",
                f"Failed to delete sign '{label}' from the vocabulary.")
