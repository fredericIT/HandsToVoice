"""
HandsToVoice — Model Info Dialog
Shows the user exactly what the trained LSTM model knows: which signs it
was trained on, which ones have an honest cross-validated accuracy figure
vs which ones don't have enough data to test, and the overall accuracy —
so the sidebar's "LSTM (Video)" badge isn't the only thing implying the
system's real reliability.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from src.model_info import load_lstm_metadata, summarize
from src.logger import get_logger

logger = get_logger("gui.model_info_dialog")

_GREEN = QColor("#00D4AA")
_YELLOW = QColor("#FFD93D")
_MUTED = QColor("#8899AA")


class ModelInfoDialog(QDialog):
    """Read-only dialog summarizing what the active LSTM model actually knows."""

    def __init__(self, vocabulary, parent=None):
        super().__init__(parent)
        self.vocabulary = vocabulary
        self.setWindowTitle("HandsToVoice — Model Info")
        self.setMinimumSize(560, 540)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("📊 What This Model Actually Knows")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:20px; font-weight:700; color:#7C4DFF; padding:8px;")
        root.addWidget(title)

        metadata = load_lstm_metadata()
        summary = summarize(metadata, vocabulary=self.vocabulary)

        if not summary:
            empty = QLabel(
                "No LSTM model has been trained yet.\n"
                "Run train_lstm_model.py after collecting sign videos.")
            empty.setAlignment(Qt.AlignCenter)
            empty.setWordWrap(True)
            empty.setStyleSheet("color:#8899AA; font-size:14px; padding:24px;")
            root.addWidget(empty)
            self._add_close_button(root)
            return

        root.addWidget(self._build_summary_card(summary))
        root.addWidget(self._build_table(summary), stretch=1)

        if summary["untested_classes"]:
            note = QLabel(
                f"⚠️ {len(summary['untested_classes'])} sign(s) have only 1 recorded "
                f"example each — not enough to measure real accuracy. Record more "
                f"videos for these signs for a trustworthy result.")
            note.setWordWrap(True)
            note.setStyleSheet(
                "color:#FFD93D; background: rgba(255,217,61,0.08); "
                "border:1px solid rgba(255,217,61,0.2); border-radius:8px; "
                "padding:10px; font-size:12px;")
            root.addWidget(note)

        self._add_close_button(root)

    def _build_summary_card(self, summary):
        card = QFrame()
        card.setObjectName("cardFrame")
        layout = QVBoxLayout(card)

        acc = summary.get("accuracy")
        acc_label = summary.get("accuracy_label", "accuracy")
        acc_n = summary.get("accuracy_n")
        if acc is not None:
            acc_text = f"{acc:.1%} {acc_label}"
            if acc_n:
                acc_text += f" (n={acc_n} held-out samples)"
        else:
            acc_text = "not yet evaluated"

        trained = summary["num_classes"]
        total_vocab = summary["total_vocab_signs"]
        coverage_text = f"{trained} sign(s) trained"
        if total_vocab is not None and total_vocab != trained:
            coverage_text += f" out of {total_vocab} in the vocabulary"

        tested_n = len(summary["tested_classes"])
        lines = [
            f"<b>Model type:</b> {summary['model_type'].upper()}",
            f"<b>Trained on:</b> {coverage_text}",
            f"<b>Reliably evaluated:</b> {tested_n} of {trained} signs",
            f"<b>Overall accuracy:</b> {acc_text}",
        ]
        if summary.get("timestamp"):
            lines.append(f"<b>Last trained:</b> {summary['timestamp'][:19].replace('T', ' ')}")

        info = QLabel("<br>".join(lines))
        info.setStyleSheet("font-size:13px; color:#E8ECF1;")
        info.setWordWrap(True)
        layout.addWidget(info)
        return card

    def _build_table(self, summary):
        classes = summary["classes"]
        untested = set(summary["untested_classes"])
        per_class = summary.get("per_class") or {}

        table = QTableWidget(len(classes), 4)
        table.setHorizontalHeaderLabels(["Sign", "Kinyarwanda", "Status", "Accuracy"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        for row, label in enumerate(sorted(classes)):
            kiny = self.vocabulary.get_kinyarwanda(label) if self.vocabulary else label
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem(kiny))

            if label in untested:
                status_item = QTableWidgetItem("⚠ 1 sample only")
                status_item.setForeground(_YELLOW)
                acc_item = QTableWidgetItem("—")
                acc_item.setForeground(_MUTED)
            else:
                status_item = QTableWidgetItem("✓ tested")
                status_item.setForeground(_GREEN)
                stats = per_class.get(label)
                if stats:
                    acc_item = QTableWidgetItem(f"{stats['recall']:.0%} recall")
                else:
                    acc_item = QTableWidgetItem("see overall")
                    acc_item.setForeground(_MUTED)
            table.setItem(row, 2, status_item)
            table.setItem(row, 3, acc_item)

        return table

    def _add_close_button(self, root):
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)
