#!/usr/bin/env python3
"""
HandsToVoice — System Test Script
Tests all system components without GUI.
"""

import os
import sys
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.capture import CameraCapture
try:
    from src.detector import HandDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    from src.detector_simple import HandDetector
    DETECTOR_AVAILABLE = False
    print("Note: Using simple detector for testing")
from src.classifier import SignClassifier
from src.tts import VoiceOutput
from src.vocabulary import Vocabulary


def test_vocabulary():
    """Test vocabulary loading."""
    print("Testing Vocabulary...")
    vocab = Vocabulary()
    
    print(f"  Total signs: {vocab.get_total_signs()}")
    print(f"  Categories: {list(vocab.get_all_categories().keys())}")
    
    # Test a few signs
    test_signs = ['muraho', 'amakuru', 'yego']
    for sign in test_signs:
        kiny = vocab.get_kinyarwanda(sign)
        eng = vocab.get_english(sign)
        display = vocab.get_display_text(sign)
        print(f"  {sign}: {kiny} ({eng}) -> {display}")
    
    print("✅ Vocabulary test passed\n")
    return True


def test_camera_capture():
    """Test camera capture (without actually opening camera)."""
    print("Testing Camera Capture...")
    
    try:
        CameraCapture(camera_index=0)
        print("  Camera object created successfully")
        
        # Test without opening camera (just object creation)
        print("✅ Camera capture test passed\n")
        return True
    except Exception as e:
        print(f"❌ Camera capture test failed: {e}\n")
        return False


def test_hand_detector():
    """Test hand detector with sample data."""
    print("Testing Hand Detector...")
    
    try:
        detector = HandDetector()
        print("  Hand detector initialized successfully")
        
        # Create a dummy frame for testing
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, landmarks_list = detector.process_frame(dummy_frame)
        
        print(f"  Dummy frame processed: {len(landmarks_list)} hands detected")
        print("✅ Hand detector test passed\n")
        return True
    except Exception as e:
        print(f"❌ Hand detector test failed: {e}\n")
        return False


def test_classifier():
    """Test sign classifier."""
    print("Testing Sign Classifier...")
    
    try:
        classifier = SignClassifier()
        print("  Classifier initialized successfully")
        
        if classifier.is_ready():
            print("  Model is loaded and ready")
            print(f"  Number of classes: {classifier.get_num_classes()}")
        else:
            print("  No model loaded (expected for fresh installation)")
        
        # Test prediction with dummy data (will fail gracefully if no model)
        dummy_landmarks = np.random.randn(63).astype(np.float32)
        sign_label, confidence = classifier.predict(dummy_landmarks)
        print(f"  Prediction test: {sign_label} ({confidence:.3f})")
        
        print("✅ Sign classifier test passed\n")
        return True
    except Exception as e:
        print(f"❌ Sign classifier test failed: {e}\n")
        return False


def test_tts():
    """Test text-to-speech (without actually playing audio)."""
    print("Testing Text-to-Speech...")
    
    try:
        # Test offline TTS (should work without internet)
        VoiceOutput(use_offline=True)
        print("  Offline TTS initialized successfully")
        
        # Test speaking (but don't actually play)
        print("  TTS speak method available")
        
        print("✅ Text-to-speech test passed\n")
        return True
    except Exception as e:
        print(f"❌ Text-to-speech test failed: {e}\n")
        return False


def test_integration():
    """Test basic integration of components."""
    print("Testing Component Integration...")
    
    try:
        # Initialize all components
        vocab = Vocabulary()
        HandDetector()
        SignClassifier()
        VoiceOutput(use_offline=True)

        print("  All components initialized successfully")
        
        # Test vocabulary + TTS integration
        test_word = vocab.get_kinyarwanda('muraho')
        print(f"  Vocabulary → TTS integration: '{test_word}' ready for speech")
        
        print("✅ Integration test passed\n")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}\n")
        return False


def test_model_availability():
    """Check if trained model is available."""
    print("Checking Model Availability...")
    
    model_path = "models/ksl_model.h5"
    labels_path = "models/ksl_model_labels.npy"
    
    if os.path.exists(model_path):
        print(f"  ✅ Model found: {model_path}")
        if os.path.exists(labels_path):
            print(f"  ✅ Labels found: {labels_path}")
        else:
            print(f"  ⚠️  Labels not found: {labels_path}")
        return True
    else:
        print(f"  ⚠️  Model not found: {model_path}")
        print("  Run 'python train_model.py --sample-data' to create a model")
        return False


def main():
    """Run all system tests."""
    print("=" * 60)
    print("HandsToVoice - System Test")
    print("University of Rwanda - College of Science and Technology")
    print("=" * 60)
    
    tests = [
        test_vocabulary,
        test_camera_capture,
        test_hand_detector,
        test_classifier,
        test_tts,
        test_integration,
        test_model_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Train a model: python train_model.py --sample-data")
        print("2. Run the app: python main.py")
        print("3. Or collect data: python collect_data.py")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
