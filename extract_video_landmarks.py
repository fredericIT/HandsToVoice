#!/usr/bin/env python3
"""
HandsToVoice — Video Landmark Extraction Script
Processes video files and extracts MediaPipe hand landmark sequences
ready for LSTM model training.

Expected folder structure:
    data/videos/<sign_name>/video001.mp4
    data/videos/<sign_name>/video002.avi
    ...

Outputs:
    data/sequences/<sign_name>_<index>.npy   — shape (SEQUENCE_LENGTH, 63)
    data/sequences/labels.csv               — filename → label mapping
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import HandDetector
from src.normalize import normalize_landmarks

from src.logger import get_logger

logger = get_logger("extract_landmarks")

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30   # frames per sequence fed into LSTM
FEATURE_LENGTH   = 65   # 21 landmarks × 3 (x, y, z) + 2 face-relative (dx, dy)
VIDEO_DIR        = "data/videos"
SEQUENCE_DIR     = "data/sequences"
LABELS_CSV       = os.path.join(SEQUENCE_DIR, "labels.csv")
SUPPORTED_EXT    = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_or_trim(frames: list, target_len: int) -> np.ndarray:
    """
    Ensure a sequence has exactly `target_len` frames.
    - If too short → repeat the last frame.
    - If too long  → sample evenly across the full sequence.
    """
    arr = np.array(frames, dtype=np.float32)
    n   = len(arr)

    if n == 0:
        return np.zeros((target_len, FEATURE_LENGTH), dtype=np.float32)

    if n < target_len:
        # Pad by repeating last frame
        pad = np.tile(arr[-1], (target_len - n, 1))
        arr = np.vstack([arr, pad])
    elif n > target_len:
        # Evenly subsample
        indices = np.linspace(0, n - 1, target_len, dtype=int)
        arr = arr[indices]

    return arr  # shape: (target_len, 63)


def extract_landmarks_from_video(video_path: str, detector: HandDetector) -> np.ndarray | None:
    """
    Open a video file and extract one landmark vector per frame.
    Returns numpy array of shape (SEQUENCE_LENGTH, 63) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"  ⚠ Cannot open: {video_path}")
        return None

    frames_lm = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, landmarks_list = detector.process_frame(frame)
        frame_idx += 1

        if landmarks_list:
            face_ref = detector.detect_face_ref(frame)
            frames_lm.append(normalize_landmarks(landmarks_list[0], face_ref))
        else:
            # No hand in this frame — use zeros
            frames_lm.append(np.zeros(FEATURE_LENGTH, dtype=np.float32))

    cap.release()

    if frame_idx == 0:
        logger.warning(f"  ⚠ Empty video: {video_path}")
        return None

    if sum(1 for f in frames_lm if f.any()) < 1:
        logger.warning(f"  ⚠ No hand detections in: {video_path}")
        return None

    return pad_or_trim(frames_lm, SEQUENCE_LENGTH)


# ── Main extraction loop ──────────────────────────────────────────────────────

def run_extraction(video_dir: str, output_dir: str, sequence_length: int):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_dir):
        logger.info(f"[Extractor] Video directory not found: {video_dir}")
        logger.info("[Extractor] Please create it and add subfolders named after each sign.")
        logger.info(f"[Extractor] Example: {video_dir}/muraho/muraho_001.mp4")
        sys.exit(1)

    sign_dirs = [
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ]

    if not sign_dirs:
        logger.info(f"[Extractor] No sign subfolders found in {video_dir}")
        sys.exit(1)

    logger.info(f"[Extractor] Found {len(sign_dirs)} sign(s): {sign_dirs}")
    logger.info("[Extractor] Initializing MediaPipe hand detector...")
    detector = HandDetector(min_detection_confidence=0.4)

    rows         = []   # for labels.csv
    total_ok     = 0
    total_failed = 0

    for sign_name in sorted(sign_dirs):
        sign_folder = os.path.join(video_dir, sign_name)
        videos = [
            f for f in os.listdir(sign_folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]

        if not videos:
            logger.info(f"  [{sign_name}] No videos found (skipped)")
            continue

        logger.info(f"\n  [{sign_name}] Processing {len(videos)} video(s)...")

        for idx, vfile in enumerate(sorted(videos)):
            vpath    = os.path.join(sign_folder, vfile)
            out_name = f"{sign_name}_{idx:04d}.npy"
            out_path = os.path.join(output_dir, out_name)

            print(f"    → {vfile}", end=" ", flush=True)

            sequence = extract_landmarks_from_video(vpath, detector)
            if sequence is None:
                logger.error("❌ FAILED")
                total_failed += 1
                continue

            np.save(out_path, sequence)
            rows.append({"file": out_name, "label": sign_name})
            total_ok += 1
            logger.info(f"✅  shape={sequence.shape}")

    # Write labels CSV
    if rows:
        with open(LABELS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "label"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"\n[Extractor] Labels saved → {LABELS_CSV}")

    print("=" * 60)
    logger.error(f"[Extractor] Done!  ✅ {total_ok} sequences  ❌ {total_failed} failed")
    logger.info(f"[Extractor] Output directory: {output_dir}")
    logger.info("[Extractor] Now run:  python train_lstm_model.py")

    detector.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmark sequences from sign language videos"
    )
    parser.add_argument("--video-dir",  default=VIDEO_DIR,
                        help=f"Root video folder (default: {VIDEO_DIR})")
    parser.add_argument("--output-dir", default=SEQUENCE_DIR,
                        help=f"Output sequence folder (default: {SEQUENCE_DIR})")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH,
                        help=f"Frames per sequence (default: {SEQUENCE_LENGTH})")

    args = parser.parse_args()

    print("=" * 60)
    logger.info("HandsToVoice — Video Landmark Extractor")
    print("=" * 60)
    logger.info(f"  Video dir    : {args.video_dir}")
    logger.info(f"  Output dir   : {args.output_dir}")
    logger.info(f"  Seq length   : {args.sequence_length} frames")
    print("=" * 60)

    run_extraction(args.video_dir, args.output_dir, args.sequence_length)


if __name__ == "__main__":
    main()
