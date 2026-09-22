"""
HandsToVoice — Speech Listener
Listens on the microphone while the system runs. When a hearing person says
a vocabulary word, it is transcribed by src.speech_recognizer and matched
against the vocabulary, so the matching sign video can be shown to the signer.

Audio is cut into utterances at natural pauses, then each utterance is
transcribed and matched. Nothing is recorded to disk; audio is discarded
once processed.
"""

import subprocess
import time
from collections import deque

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.speech_recognizer import SAMPLE_RATE, SpeechRecognizer, fuzzy_match

logger = get_logger("speech_listener")

REPEAT_COOLDOWN = 1.5      # same word not shown again within this many seconds
ONLINE_RETRY_SECONDS = 60  # how often to re-check online recognition after it's disabled


class Segmenter:
    """Cuts a stream of 16-bit samples into utterances separated by pauses."""

    def __init__(self, sr=SAMPLE_RATE, frame_ms=20, pause_ms=220,
                 min_ms=200, max_ms=2500, start_frames=3, min_rms=350.0,
                 noise_factor=2.5):
        self.frame_len = int(sr * frame_ms / 1000)
        self.pause_frames = pause_ms // frame_ms
        self.min_frames = min_ms // frame_ms
        self.max_frames = max_ms // frame_ms
        self.start_frames = start_frames
        self.min_rms = min_rms
        self.noise_factor = noise_factor
        self.pad_frames = 5
        self.reset()

    def reset(self):
        self._leftover = np.zeros(0, dtype=np.int16)
        self._noise = None
        self._calibration = []
        self._preroll = deque(maxlen=self.pad_frames)
        self._frames = []
        self._speaking = False
        self._loud_run = 0
        self._quiet_run = 0

    def discard(self):
        """Drop any speech in progress but keep the learned room noise level,
        so listening resumes instantly (used while the system itself speaks)."""
        self._leftover = np.zeros(0, dtype=np.int16)
        self._preroll.clear()
        self._frames = []
        self._speaking = False
        self._loud_run = 0
        self._quiet_run = 0

    def feed(self, samples):
        """Consume int16 samples; return the utterances (float32) completed."""
        data = np.concatenate([self._leftover, samples])
        n = len(data) // self.frame_len
        self._leftover = data[n * self.frame_len:]
        done = []
        for k in range(n):
            frame = data[k * self.frame_len:(k + 1) * self.frame_len]
            utterance = self._step(frame)
            if utterance is not None:
                done.append(utterance)
        return done

    def _step(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

        if self._noise is None:                     # learn the room's noise level
            self._calibration.append(rms)
            if len(self._calibration) >= 25:
                self._noise = float(np.median(self._calibration))
            return None

        threshold = max(self._noise * self.noise_factor, self.min_rms)
        loud = rms > threshold

        if not self._speaking:
            if not loud:
                self._noise = 0.95 * self._noise + 0.05 * min(rms, self._noise * 3)
                # Hard ceiling: without this, anything that leaks the
                # system's own (much louder) voice into the mic while not
                # correctly muted — e.g. the _is_speaking race documented
                # in src/tts.py — can ratchet the floor up permanently,
                # after which normal human speech never reads as "loud"
                # again and nothing gets segmented for the rest of the run.
                self._noise = min(self._noise, self.min_rms * 4)
            self._preroll.append(frame)
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.start_frames:
                self._speaking = True
                self._frames = list(self._preroll)
                self._quiet_run = 0
            return None

        self._frames.append(frame)
        # A single loud frame (breath, trailing consonant) only costs a few
        # frames of progress rather than resetting the whole countdown to
        # 0 — otherwise one stray blip near the end of a word could force
        # waiting all the way out to max_ms before giving up on a pause.
        if rms > threshold * 0.7:
            self._quiet_run = max(0, self._quiet_run - 4)
        else:
            self._quiet_run += 1
        if self._quiet_run >= self.pause_frames or len(self._frames) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self):
        frames = self._frames
        trim = max(0, self._quiet_run - self.pad_frames)   # keep a short tail of the pause
        if trim:
            frames = frames[:-trim]
        self._frames = []
        self._speaking = False
        self._loud_run = 0
        self._quiet_run = 0
        self._preroll.clear()
        if len(frames) - self.pad_frames < self.min_frames:
            return None                             # too short: a click or cough
        return np.concatenate(frames).astype(np.float32) / 32768.0


class SpeechListener(QThread):
    """Background thread: microphone -> utterances -> matched vocabulary labels."""

    words_heard = pyqtSignal(list)      # [(label, ratio), ...]
    mic_silent = pyqtSignal(bool)       # True: the mic delivers digital silence (muted/off)
    model_status = pyqtSignal(str)      # "loading" | "ready" | "failed"

    def __init__(self, vocab_words):
        super().__init__()
        self._vocab_words = dict(vocab_words)   # {label: kinyarwanda text}
        self._running = False
        self._proc = None
        self.muted = False              # set while the system itself is speaking
        self.recognizer = SpeechRecognizer()
        self.segmenter = Segmenter()
        self._last_shown = {}
        self._silent_for = 0.0
        self._silent_reported = False

    def update_vocab(self, vocab_words):
        """Re-read the vocabulary (after signs are added/edited/deleted).
        No model reload needed — only the text to match against changes."""
        self._vocab_words = dict(vocab_words)

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.wait(2000)

    def run(self):
        self._running = True
        self.model_status.emit("loading")
        self.recognizer.load()
        self.model_status.emit("ready" if self.recognizer.ready else "failed")
        if not self.recognizer.ready:
            return

        failures = 0
        while self._running and failures < 5:
            try:
                self._proc = subprocess.Popen(
                    ["arecord", "-D", "default", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                     "-c", "1", "-t", "raw", "-q"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            except FileNotFoundError:
                logger.error("[Listen] arecord not found — install alsa-utils. "
                             "Voice listening is disabled.")
                return
            started = time.time()
            self._read_stream()
            if not self._running:
                break
            failures = failures + 1 if time.time() - started < 5 else 0
            self.segmenter.reset()
            time.sleep(2)
        if self._running:
            logger.error("[Listen] microphone stream keeps failing — listening stopped.")

    def _read_stream(self):
        chunk_bytes = int(SAMPLE_RATE * 0.1) * 2
        next_retry = time.time() + ONLINE_RETRY_SECONDS
        while self._running:
            data = self._proc.stdout.read(chunk_bytes)
            if not data:
                return
            self._watch_for_silence(data)
            if time.time() >= next_retry:
                self.recognizer.retry_online()
                next_retry = time.time() + ONLINE_RETRY_SECONDS
            if self.muted:
                self.segmenter.discard()
                continue
            samples = np.frombuffer(data[:len(data) // 2 * 2], dtype=np.int16)
            for utterance in self.segmenter.feed(samples):
                self._handle(utterance)

    def _watch_for_silence(self, data):
        """A live microphone always has some noise; exact digital silence
        means the capture is switched off or muted at system level. Say so —
        otherwise listening just appears to do nothing."""
        if any(data):
            self._silent_for = 0.0
            if self._silent_reported:
                self._silent_reported = False
                self.mic_silent.emit(False)
            return
        self._silent_for += len(data) / 2 / SAMPLE_RATE
        if self._silent_for >= 3.0 and not self._silent_reported:
            self._silent_reported = True
            logger.warning("[Listen] the microphone is delivering pure silence — "
                           "it is probably muted or switched off in the system sound settings.")
            self.mic_silent.emit(True)

    def _handle(self, utterance):
        text = self.recognizer.transcribe(utterance)
        if not text:
            return
        match = fuzzy_match(text, self._vocab_words)
        logger.info(f"[Listen] heard \"{text}\" ({len(utterance) / SAMPLE_RATE:.1f}s) | "
                    + (f"matched '{match[0]}' ({match[1]:.2f})" if match else "no match"))
        if not match:
            return
        label, ratio = match
        now = time.time()
        if now - self._last_shown.get(label, 0) > REPEAT_COOLDOWN:
            self._last_shown[label] = now
            self.words_heard.emit([(label, ratio)])
