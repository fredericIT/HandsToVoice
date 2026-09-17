"""
HandsToVoice — Webcam Capture Module
Wraps OpenCV's VideoCapture for real-time frame acquisition.
"""

import cv2
import time


class CameraCapture:
    """Manages the webcam lifecycle and provides frames as a generator."""

    def __init__(self, camera_index=0, width=640, height=480, fps_cap=30):
        """
        Initialize camera capture.

        Args:
            camera_index: Index of the camera device (0 = default webcam).
            width: Desired frame width in pixels.
            height: Desired frame height in pixels.
            fps_cap: Maximum frames per second to yield.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps_cap = fps_cap
        self.cap = None

    def open(self):
        """Open the camera device."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Please check your webcam connection."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_cap)
        return self

    def read_frame(self):
        """
        Read a single frame from the camera.

        Returns:
            BGR frame as a numpy array, or None if read fails.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Mirror the frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)
        return frame

    def frames(self):
        """
        Generator that yields BGR frames at the configured FPS cap.

        Yields:
            numpy.ndarray: BGR frame from the camera.
        """
        if self.cap is None:
            self.open()

        frame_interval = 1.0 / self.fps_cap
        last_time = 0

        while self.cap.isOpened():
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            frame = self.read_frame()
            if frame is None:
                break

            last_time = time.time()
            yield frame

    def release(self):
        """Release the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        """Check if the camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
