"""
HandsToVoice — Simple Hand Landmark Detector Module
A simplified version that works without requiring MediaPipe model files.
For testing and demonstration purposes.
"""

import cv2
import numpy as np


class HandDetector:
    """Simple hand landmark detector for testing purposes."""

    # MediaPipe returns 21 landmarks, each with x, y, z coordinates
    NUM_LANDMARKS = 21
    FEATURE_LENGTH = NUM_LANDMARKS * 3  # 63 features

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize the simple hand detector.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        
        # Simple background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # For testing, we'll generate random landmarks when motion is detected
        self.frame_count = 0

    def process_frame(self, bgr_frame):
        """
        Process a BGR frame and extract hand landmarks.

        Args:
            bgr_frame: Input BGR image (numpy array from OpenCV).

        Returns:
            tuple: (annotated_frame, landmarks_list)
                - annotated_frame: The input frame with hand skeleton drawn.
                - landmarks_list: List of normalized landmark arrays, one per
                  detected hand. Each array has shape (63,) representing
                  [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
                  Returns empty list if no hands detected.
        """
        self.frame_count += 1
        
        # Simple motion detection
        fg_mask = self.fgbg.apply(bgr_frame)
        
        # Find contours (potential hands)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated = bgr_frame.copy()
        landmarks_list = []
        
        # Filter contours by size (hands are typically a certain size)
        min_contour_area = 5000  # Adjust based on camera resolution
        max_contour_area = 50000
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_contour_area < area < max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Generate mock hand landmarks for testing
                if self.frame_count % 10 == 0:  # Generate landmarks every 10 frames
                    landmarks = self._generate_mock_landmarks(x, y, w, h)
                    landmarks_list.append(landmarks)
                    
                    # Draw some points to simulate hand landmarks
                    for i in range(0, 63, 3):  # Draw every 3rd point
                        lx = int(landmarks[i] * bgr_frame.shape[1])
                        ly = int(landmarks[i+1] * bgr_frame.shape[0])
                        cv2.circle(annotated, (lx, ly), 3, (0, 0, 255), -1)
                
                # Limit number of hands
                if len(landmarks_list) >= self.max_num_hands:
                    break
        
        # Add status text
        cv2.putText(annotated, f"Hands Detected: {len(landmarks_list)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {self.frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated, landmarks_list

    def _generate_mock_landmarks(self, bbox_x, bbox_y, bbox_w, bbox_h):
        """
        Generate mock hand landmarks for testing purposes.
        
        Args:
            bbox_x, bbox_y, bbox_w, bbox_h: Bounding box coordinates and size
            
        Returns:
            numpy.ndarray: Mock landmark coordinates of shape (63,)
        """
        # Generate realistic-looking hand landmarks within the bounding box
        landmarks = []
        
        # Wrist (landmark 0) - center bottom of bounding box
        wrist_x = (bbox_x + bbox_w / 2) / 640  # Normalize to [0, 1]
        wrist_y = (bbox_y + bbox_h * 0.8) / 480  # Normalize to [0, 1]
        landmarks.extend([wrist_x, wrist_y, 0.0])  # z = 0 for 2D
        
        # Generate other landmarks in a hand-like pattern
        # This is a simplified pattern - in reality, hand landmarks have specific positions
        for i in range(1, 21):
            # Create a spread pattern around the wrist
            angle = (i - 1) * (2 * np.pi / 20)
            radius = 0.05 + (i % 5) * 0.02  # Varying radius
            
            x = wrist_x + radius * np.cos(angle)
            y = wrist_y - radius * np.sin(angle) * 2  # Hands extend upward
            z = np.random.uniform(-0.1, 0.1)  # Small z variation
            
            # Ensure coordinates are within [0, 1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            landmarks.extend([x, y, z])
        
        return np.array(landmarks, dtype=np.float32)

    def close(self):
        """Release resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
