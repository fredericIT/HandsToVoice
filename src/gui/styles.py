"""
HandsToVoice — GUI Styles Module
Dark-mode QSS stylesheet for the PyQt5 interface.
"""

# ─── Color Palette ────────────────────────────────────────────────────────────
# Primary Background:  #0F1419  (deep charcoal)
# Secondary BG:        #1A2332  (dark blue-grey)
# Card BG:             #1E2D3D  (elevated surface)
# Accent:              #00D4AA  (teal-green)
# Accent Hover:        #00F5C8  (bright teal)
# Accent Dark:         #009E7E  (muted teal)
# Text Primary:        #E8ECF1  (off-white)
# Text Secondary:      #8899AA  (muted grey-blue)
# Border:              #2A3A4A  (subtle divider)
# Danger:              #FF6B6B  (soft red)
# Warning:             #FFD93D  (warm yellow)
# Success:             #00D4AA  (same as accent)

DARK_THEME = """
/* ─── Global ─────────────────────────────────────────────── */
QWidget {
    background-color: #0F1419;
    color: #E8ECF1;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}

/* ─── Main Window ────────────────────────────────────────── */
QMainWindow {
    background-color: #0F1419;
}

/* ─── Labels ─────────────────────────────────────────────── */
QLabel {
    color: #E8ECF1;
    background: transparent;
}

QLabel#titleLabel {
    font-size: 22px;
    font-weight: 700;
    color: #00D4AA;
    padding: 8px 0;
}

QLabel#subtitleLabel {
    font-size: 12px;
    color: #8899AA;
    padding: 2px 0;
}

QLabel#signLabel {
    font-size: 48px;
    font-weight: 800;
    color: #00D4AA;
    padding: 16px;
    background-color: #1E2D3D;
    border-radius: 16px;
    border: 2px solid #2A3A4A;
}

QLabel#translationLabel {
    font-size: 20px;
    font-weight: 500;
    color: #8899AA;
    padding: 8px;
}

QLabel#statusLabel {
    font-size: 11px;
    color: #8899AA;
    padding: 4px 12px;
}

QLabel#sentenceLabel {
    font-size: 18px;
    font-weight: 600;
    color: #E8ECF1;
    padding: 12px 16px;
    background-color: #1A2332;
    border-radius: 12px;
    border: 1px solid #2A3A4A;
    min-height: 48px;
}

QLabel#cameraPlaceholder {
    font-size: 16px;
    color: #8899AA;
    background-color: #1A2332;
    border-radius: 16px;
    border: 2px dashed #2A3A4A;
}

QLabel#noModelLabel {
    font-size: 13px;
    color: #FFD93D;
    background-color: rgba(255, 217, 61, 0.08);
    border: 1px solid rgba(255, 217, 61, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
}

/* ─── Sidebar ────────────────────────────────────────────── */
QFrame#sidebarFrame {
    background-color: #131B25;
    border: 1px solid #2A3A4A;
    border-radius: 16px;
    padding: 16px 8px;
}

QLabel#sidebarTitle {
    font-size: 14px;
    font-weight: 700;
    color: #8899AA;
    padding-bottom: 8px;
    background: transparent;
}

QLabel#stateIconLabel {
    font-size: 72px;
    padding: 12px 0;
    background: transparent;
}

QLabel#stateTextLabel {
    font-size: 16px;
    font-weight: 700;
    padding: 4px 0;
    background: transparent;
}

QLabel#stateInstructionLabel {
    font-size: 11px;
    color: #8899AA;
    padding: 4px 8px;
    background: transparent;
}

QLabel#sidebarLastSign {
    font-size: 15px;
    font-weight: 600;
    color: #00D4AA;
    padding: 10px;
    background-color: #1E2D3D;
    border-radius: 10px;
    border: 1px solid #2A3A4A;
    min-height: 32px;
}

/* ─── Buttons ────────────────────────────────────────────── */
QPushButton {
    background-color: #1E2D3D;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #2A3A4A;
    border-color: #00D4AA;
    color: #00D4AA;
}

QPushButton:pressed {
    background-color: #009E7E;
    color: #0F1419;
}

QPushButton:disabled {
    background-color: #151C24;
    color: #4A5568;
    border-color: #1E2D3D;
}

QPushButton#speakButton {
    background-color: #00D4AA;
    color: #0F1419;
    font-size: 15px;
    font-weight: 700;
    border: none;
    padding: 14px 32px;
    border-radius: 12px;
    min-width: 140px;
}

QPushButton#speakButton:hover {
    background-color: #00F5C8;
    color: #0F1419;
}

QPushButton#speakButton:pressed {
    background-color: #009E7E;
}

QPushButton#speakButton:disabled {
    background-color: #1E2D3D;
    color: #4A5568;
}

QPushButton#clearButton {
    background-color: transparent;
    color: #FF6B6B;
    border: 1px solid #FF6B6B;
    min-width: 80px;
}

QPushButton#clearButton:hover {
    background-color: rgba(255, 107, 107, 0.1);
}

/* ─── Progress Bar (Confidence) ──────────────────────────── */
QProgressBar {
    background-color: #1A2332;
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    text-align: center;
    color: #E8ECF1;
    font-size: 11px;
    font-weight: 600;
    min-height: 20px;
    max-height: 20px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #009E7E, stop:0.5 #00D4AA, stop:1 #00F5C8);
    border-radius: 7px;
}

/* ─── Frames / Cards ─────────────────────────────────────── */
QFrame#cardFrame {
    background-color: #1A2332;
    border: 1px solid #2A3A4A;
    border-radius: 16px;
    padding: 16px;
}

QFrame#separator {
    background-color: #2A3A4A;
    max-height: 1px;
    min-height: 1px;
}

/* ─── Group Box ──────────────────────────────────────────── */
QGroupBox {
    background-color: #1A2332;
    border: 1px solid #2A3A4A;
    border-radius: 12px;
    margin-top: 16px;
    padding-top: 24px;
    font-weight: 600;
    color: #8899AA;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #00D4AA;
    font-size: 12px;
}

/* ─── Combo Box ──────────────────────────────────────────── */
QComboBox {
    background-color: #1A2332;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #00D4AA;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #1A2332;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    selection-background-color: #009E7E;
    selection-color: #0F1419;
    outline: none;
}

/* ─── Spin Box ───────────────────────────────────────────── */
QSpinBox {
    background-color: #1A2332;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 20px;
}

QSpinBox:hover {
    border-color: #00D4AA;
}

/* ─── Text Edit / Line Edit ──────────────────────────────── */
QTextEdit, QLineEdit {
    background-color: #1A2332;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #009E7E;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #00D4AA;
}

/* ─── Scroll Area ────────────────────────────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background-color: #1A2332;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #2A3A4A;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00D4AA;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

/* ─── Tooltip ────────────────────────────────────────────── */
QToolTip {
    background-color: #1E2D3D;
    color: #E8ECF1;
    border: 1px solid #2A3A4A;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ─── Add Sign Button ────────────────────────────────────── */
QPushButton#addSignButton {
    background-color: #7C4DFF;
    color: white;
    font-size: 13px;
    font-weight: 700;
    border: none;
    padding: 10px 24px;
    border-radius: 10px;
    min-width: 120px;
}

QPushButton#addSignButton:hover {
    background-color: #9C7CFF;
    color: white;
}

QPushButton#addSignButton:pressed {
    background-color: #5C2DE0;
}

QPushButton#addSignButton:disabled {
    background-color: #3A2A6A;
    color: #7A6A9A;
}

/* ─── Add Sign Dialog ────────────────────────────────────── */
QDialog {
    background-color: #0F1419;
    color: #E8ECF1;
}

QDialog QGroupBox {
    background-color: #1A2332;
    border: 1px solid #2A3A4A;
    border-radius: 12px;
    margin-top: 16px;
    padding-top: 24px;
    font-weight: 600;
    color: #8899AA;
}

QDialog QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #7C4DFF;
    font-size: 12px;
}
"""

# Convenience function
def get_stylesheet():
    """Return the dark theme QSS stylesheet."""
    return DARK_THEME
