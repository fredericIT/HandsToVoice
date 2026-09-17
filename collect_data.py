#!/usr/bin/env python3
"""
HandsToVoice — Data Collection Script
Collects hand landmark data for Kinyarwanda Sign Language training.
"""

import os
import sys
import csv
import argparse
from datetime import datetime
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QSpinBox, QTextEdit, QProgressBar, QMessageBox,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.capture import CameraCapture
from src.detector import HandDetector
from src.vocabulary import Vocabulary
from src.gui.styles import get_stylesheet

from src.logger import get_logger

logger = get_logger("collect_data")


class DataCollectorGUI(QMainWindow):
    """GUI for collecting KSL training data."""

    def __init__(self, camera_index=0, output_dir="data/raw"):
        super().__init__()
        
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.vocabulary = Vocabulary()
        
        # Initialize components
        self.camera = None
        self.detector = None
        
        # Collection state
        self.is_collecting = False
        self.current_sign = None
        self.samples_collected = 0
        self.target_samples = 30
        self.collected_data = []
        
        # Setup UI
        self.setup_ui()
        self.initialize_components()
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def setup_ui(self):
        """Setup the main UI."""
        self.setWindowTitle("HandsToVoice - Data Collection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left side - Camera feed
        left_widget = self.create_camera_section()
        main_layout.addWidget(left_widget, stretch=2)
        
        # Right side - Controls
        right_widget = self.create_controls_section()
        main_layout.addWidget(right_widget, stretch=1)

    def create_camera_section(self):
        """Create the camera feed section."""
        camera_frame = QGroupBox("📷 Camera Feed")
        camera_frame.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(camera_frame)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #2A3A4A;
                border: 2px solid #3A4A5A;
                border-radius: 8px;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("📷 Camera Off")
        layout.addWidget(self.camera_label)
        
        # Status info
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #00D4AA; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        return camera_frame

    def create_controls_section(self):
        """Create the controls section."""
        controls_frame = QGroupBox("🎛️ Data Collection Controls")
        controls_frame.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(controls_frame)
        
        # Sign selection
        sign_group = QGroupBox("Sign Selection")
        sign_layout = QVBoxLayout(sign_group)
        
        self.sign_combo = QComboBox()
        self.sign_combo.addItems(self.vocabulary.get_all_labels())
        sign_layout.addWidget(QLabel("Select Sign:"))
        sign_layout.addWidget(self.sign_combo)
        
        layout.addWidget(sign_group)
        
        # Collection settings
        settings_group = QGroupBox("Collection Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Target Samples:"), 0, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 200)
        self.samples_spin.setValue(30)
        settings_layout.addWidget(self.samples_spin, 0, 1)
        
        layout.addWidget(settings_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("0 / 0 samples collected")
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("▶️ Start Collection")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #00D4AA;
                color: #0F1419;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #00F5C8;
            }
            QPushButton:disabled {
                background-color: #4A5568;
                color: #1A2332;
            }
        """)
        self.start_button.clicked.connect(self.toggle_collection)
        button_layout.addWidget(self.start_button)
        
        self.save_button = QPushButton("💾 Save Data")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2A3A4A;
                color: #E8ECF1;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #3A4A5A;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3A4A5A;
                border-color: #00D4AA;
            }
        """)
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.clear_button = QPushButton("🗑️ Clear Data")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #FF6B6B;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #FF6B6B;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(255, 107, 107, 0.1);
            }
        """)
        self.clear_button.clicked.connect(self.clear_data)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QTextEdit()
        instructions_text.setReadOnly(True)
        instructions_text.setMaximumHeight(150)
        instructions_text.setPlainText("""
1. Select a sign from the dropdown
2. Set target number of samples
3. Click "Start Collection"
4. Perform the sign in front of camera
5. Each detected hand gesture adds a sample
6. Click "Save Data" when finished
7. Repeat for all signs

Tips:
- Keep hand in camera view
- Perform signs clearly and consistently
- Collect samples from different angles
- Ensure good lighting
        """)
        instructions_text.setStyleSheet("""
            QTextEdit {
                background-color: #1A2332;
                color: #E8ECF1;
                border: 1px solid #2A3A4A;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        instructions_layout.addWidget(instructions_text)
        
        layout.addWidget(instructions_group)
        
        return controls_frame

    def initialize_components(self):
        """Initialize camera and detector."""
        try:
            self.camera = CameraCapture(camera_index=self.camera_index)
            self.detector = HandDetector()
            self.camera.open()
            self.status_label.setText("Status: Camera Ready")
        except Exception as e:
            self.status_label.setText(f"Status: Camera Error - {e}")
            self.start_button.setEnabled(False)

    def update_frame(self):
        """Update camera feed and collect data."""
        if not self.camera or not self.camera.is_opened():
            return
        
        frame = self.camera.read_frame()
        if frame is None:
            return
        
        # Process frame with hand detector
        annotated, landmarks_list = self.detector.process_frame(frame)
        
        # Display frame
        self.display_frame(annotated)
        
        # Collect data if actively collecting
        if self.is_collecting and landmarks_list:
            self.collect_sample(landmarks_list[0])

    def display_frame(self, frame):
        """Display frame in the GUI."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        
        # Convert to QImage
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.camera_label.setPixmap(scaled_pixmap)

    def collect_sample(self, landmarks):
        """Collect a single sample of hand landmarks."""
        if self.samples_collected >= self.target_samples:
            return
        
        # Add sample with timestamp and sign label
        sample = {
            'timestamp': datetime.now().isoformat(),
            'sign': self.current_sign,
            'landmarks': landmarks.tolist()
        }
        
        self.collected_data.append(sample)
        self.samples_collected += 1
        
        # Update progress
        progress = (self.samples_collected / self.target_samples) * 100
        self.progress_bar.setValue(int(progress))
        self.progress_label.setText(f"{self.samples_collected} / {self.target_samples} samples collected")
        
        # Check if collection complete
        if self.samples_collected >= self.target_samples:
            self.stop_collection()
            QMessageBox.information(self, "Collection Complete", 
                                   f"Successfully collected {self.samples_collected} samples for '{self.current_sign}'")

    def toggle_collection(self):
        """Start or stop data collection."""
        if self.is_collecting:
            self.stop_collection()
        else:
            self.start_collection()

    def start_collection(self):
        """Start collecting data for the selected sign."""
        # Get selected sign
        self.current_sign = self.sign_combo.currentText()
        self.target_samples = self.samples_spin.value()
        
        # Reset counters
        self.samples_collected = 0
        self.collected_data = []
        
        # Update UI
        self.is_collecting = True
        self.start_button.setText("⏸️ Stop Collection")
        self.status_label.setText(f"Status: Collecting '{self.current_sign}'")
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"0 / {self.target_samples} samples collected")
        self.save_button.setEnabled(False)
        self.sign_combo.setEnabled(False)
        self.samples_spin.setEnabled(False)

    def stop_collection(self):
        """Stop data collection."""
        self.is_collecting = False
        self.start_button.setText("▶️ Start Collection")
        self.status_label.setText(f"Status: Collection stopped ({self.samples_collected} samples)")
        self.save_button.setEnabled(self.samples_collected > 0)
        self.sign_combo.setEnabled(True)
        self.samples_spin.setEnabled(True)

    def save_data(self):
        """Save collected data to CSV file."""
        if not self.collected_data:
            QMessageBox.warning(self, "No Data", "No data to save!")
            return
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_sign}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [f'landmark_{i}' for i in range(63)] + ['label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for sample in self.collected_data:
                row = {}
                landmarks = sample['landmarks']
                for i in range(63):
                    row[f'landmark_{i}'] = landmarks[i]
                row['label'] = sample['sign']
                writer.writerow(row)
        
        QMessageBox.information(self, "Data Saved", 
                               f"Successfully saved {len(self.collected_data)} samples to:\n{filepath}")
        
        # Clear data after saving
        self.clear_data()

    def clear_data(self):
        """Clear collected data."""
        self.collected_data = []
        self.samples_collected = 0
        self.progress_bar.setValue(0)
        self.progress_label.setText("0 / 0 samples collected")
        self.save_button.setEnabled(False)

    def closeEvent(self, event):
        """Handle window close event."""
        if self.camera:
            self.camera.release()
        if self.detector:
            self.detector.close()
        event.accept()


def main():
    """Main data collection script entry point."""
    parser = argparse.ArgumentParser(description="Collect KSL training data")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory (default: data/raw)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    logger.info("HandsToVoice - Data Collection")
    logger.info("University of Rwanda - College of Science and Technology")
    print("=" * 60)
    logger.info(f"Camera Index: {args.camera}")
    logger.info(f"Output Directory: {args.output}")
    print("=" * 60)
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("HandsToVoice Data Collection")
    app.setStyleSheet(get_stylesheet())

    # Create and show GUI
    window = DataCollectorGUI(camera_index=args.camera, output_dir=args.output)
    window.show()
    
    # Run application
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
