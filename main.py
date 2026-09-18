#!/usr/bin/env python3
"""
HandsToVoice — Main Application Entry Point
Real-time Kinyarwanda Sign Language Recognition and Voice Conversion System
"""

import sys
import os
import signal
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.styles import get_stylesheet, load_bundled_fonts
from src.capture import CameraCapture
from src.detector import HandDetector
from src.classifier import SignClassifier, LSTMClassifier
from src.tts import VoiceOutput
from src.vocabulary import Vocabulary
from src.gui.main_window import MainWindow
from src.gui.welcome_screen import WelcomeScreen
from src.gui.loading_screen import LoadingScreen

from src.logger import get_logger

logger = get_logger("main")


class _ComponentInitWorker(QThread):
    """Loads the camera, detector, classifiers, and voice engine off the
    GUI thread, so clicking "Get Started" shows a loading screen instantly
    instead of freezing the window for the several seconds this takes."""

    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        try:
            camera = CameraCapture(camera_index=self.camera_index)
            detector = HandDetector()
            classifier = SignClassifier()

            lstm_classifier = LSTMClassifier()
            if lstm_classifier.is_ready():
                logger.info("[System] LSTM model detected — sequence mode available")
            else:
                logger.info("[System] No LSTM model — using static classifier only")

            vocabulary = Vocabulary()
            tts = VoiceOutput(vocabulary=vocabulary)

            self.ready.emit({
                "camera": camera,
                "detector": detector,
                "classifier": classifier,
                "lstm_classifier": lstm_classifier,
                "vocabulary": vocabulary,
                "tts": tts,
            })
        except Exception as e:
            logger.error(f"[System] Error initializing components: {e}")
            self.failed.emit(str(e))


class HandsToVoiceApp:
    """Main application controller for HandsToVoice system."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.app = None
        self.main_window = None
        self.welcome_screen = None
        self.loading_screen = None
        self._init_worker = None
        self._signal_timer = None

        # Initialize core components
        self.camera = None
        self.detector = None
        self.classifier = None
        self.lstm_classifier = None
        self.tts = None
        self.vocabulary = None

    def create_gui(self):
        """Create and configure the main GUI window."""
        try:
            self.main_window = MainWindow(
                camera=self.camera,
                detector=self.detector,
                classifier=self.classifier,
                tts=self.tts,
                vocabulary=self.vocabulary,
                lstm_classifier=self.lstm_classifier
            )

            # Setup signal handling for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)

            # Handle window close
            self.main_window.closeEvent = self._on_close

            logger.info("[System] GUI created successfully")
            return True

        except Exception as e:
            logger.error(f"[System] Error creating GUI: {e}")
            return False

    def run(self):
        """Start the HandsToVoice application.

        Users cannot reach the recognition system directly — the welcome
        screen is a mandatory gate. Camera/model initialization is deferred
        until the user clicks through it, so startup is instant and no
        hardware is touched before they've seen who the app is for and how
        it works.
        """
        # Create QApplication
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("HandsToVoice")
        self.app.setApplicationVersion("1.0")
        load_bundled_fonts()
        self.app.setStyleSheet(get_stylesheet())

        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.continue_requested.connect(self._start_main_app)
        self.welcome_screen.show()

        logger.info("[System] Welcome screen shown")
        return self.app.exec_()

    def _start_main_app(self):
        """Called the instant "Get Started" is clicked.

        Shows a lightweight animated loading screen immediately and loads
        the camera/detector/models on a background thread — so the click
        feels instant instead of freezing the window for the several
        seconds real initialization takes.
        """
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()

        self._init_worker = _ComponentInitWorker(self.camera_index)
        self._init_worker.ready.connect(self._on_components_ready)
        self._init_worker.failed.connect(self._on_components_failed)
        self._init_worker.start()

    def _on_components_ready(self, components):
        """Called on the GUI thread once the background worker has built
        every component — safe to build QWidgets here."""
        self.camera = components["camera"]
        self.detector = components["detector"]
        self.classifier = components["classifier"]
        self.lstm_classifier = components["lstm_classifier"]
        self.vocabulary = components["vocabulary"]
        self.tts = components["tts"]
        logger.info("[System] All components initialized successfully")

        if not self.create_gui():
            self._show_error("Failed to create GUI")
            self.app.quit()
            return

        self.main_window.show()
        if self.loading_screen:
            self.loading_screen.close()
            self.loading_screen = None

        # Setup timer for signal handling — kept as an attribute so it isn't
        # garbage-collected once this method returns.
        self._signal_timer = QTimer()
        self._signal_timer.timeout.connect(lambda: None)
        self._signal_timer.start(100)

        logger.info("[System] HandsToVoice started successfully")

    def _on_components_failed(self, message):
        """Called on the GUI thread if the background worker raised."""
        if self.loading_screen:
            self.loading_screen.close()
            self.loading_screen = None
        self._show_error(f"Failed to initialize system components: {message}")
        self.app.quit()

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal for graceful shutdown."""
        logger.info("\n[System] Received interrupt signal, shutting down...")
        self.cleanup()
        sys.exit(0)

    def _on_close(self, event):
        """Handle main window close event."""
        self.cleanup()
        event.accept()

    def _show_error(self, message):
        """Show error dialog and exit."""
        if self.app:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("HandsToVoice - Error")
            msg.setText(message)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            logger.error(f"ERROR: {message}")

    def cleanup(self):
        """Clean up all resources."""
        try:
            if self.camera:
                self.camera.release()
            if self.detector:
                self.detector.close()
            if self.tts:
                self.tts.cleanup()
            logger.info("[System] Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"[System] Error during cleanup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HandsToVoice - Kinyarwanda Sign Language Recognition System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--version", action="version", version="HandsToVoice 1.0")
    
    args = parser.parse_args()
    
    # Print system info
    print("=" * 60)
    logger.info("HandsToVoice - Kinyarwanda Sign Language Recognition System")
    logger.info("University of Rwanda - College of Science and Technology")
    logger.info("Department of Information and Communication Technology")
    print("=" * 60)
    logger.info(f"Camera Index: {args.camera}")
    logger.info("Voice Mode:   Custom recordings + system TTS fallback")
    print("=" * 60)
    
    # Create and run application
    app = HandsToVoiceApp(camera_index=args.camera)
    exit_code = app.run()
    
    logger.info(f"[System] Application exited with code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
