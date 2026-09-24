"""
HandsToVoice — Speech Recognizer
Transcribes Kinyarwanda speech and matches it against the vocabulary, so a
hearing person's speech can be mapped to a sign.

Two engines, tried in order:
  1. Online — Google's public speech-recognition endpoint (via the
     SpeechRecognition library's recognize_google). No account or API key
     needed, about 1s/word, and measured at 89% correct on this project's
     own vocabulary. It needs internet, sends the audio to Google, and is
     an unofficial/undocumented endpoint — it can be rate-limited or slow,
     and Google gives no guarantee it keeps working.
  2. Offline — Meta's MMS model, running entirely on this machine. Slower
     to start (~15-20s model load) and to run (~0.5-2s/word), but needs no
     internet and never leaves this computer.

transcribe() tries online first and falls back to offline automatically,
so the feature keeps working without internet, just slower to first respond
and with the load delay while MMS spins up.
"""

import difflib
import wave

import numpy as np

from src.logger import get_logger

logger = get_logger("speech_recognizer")

SAMPLE_RATE = 16000
MODEL_ID = "facebook/mms-1b-all"
LANGUAGE = "kin"   # Kinyarwanda
ONLINE_LANGUAGE = "rw-RW"
ONLINE_TIMEOUT = 5   # seconds to wait for Google's endpoint before giving up

# Minimum text-similarity ratio to accept a match. Measured against real
# transcripts of this project's own recordings: every correct match scored
# at least 0.67, while 95% of wrong pairings scored under 0.48. 0.6 sits
# between the two with margin on both sides.
MATCH_THRESHOLD = 0.6

# A vocabulary word shorter than this many letters (e.g. "ni" - Kinyarwanda
# for "is") is far too short and common to safely spot embedded inside a
# natural sentence — ordinary grammar would trigger it on almost every
# sentence. Such short words are only matched when spoken alone as the
# whole utterance, never as a fragment picked out of a longer sentence.
MIN_EMBED_CHARS = 4

# Stricter threshold applied only when picking a vocabulary word out of a
# longer sentence rather than matching a single spoken word. With one word
# spoken alone, MATCH_THRESHOLD's calibration (against real, sometimes
# noisy transcriptions of that exact word) applies directly. Inside a full
# sentence, many unrelated words are being compared per utterance instead
# of one, so the odds of some unrelated word coincidentally overlapping a
# vocabulary word's letters (e.g. "amakuru" vs "marume", ratio 0.62) rise
# with sentence length — this higher bar keeps that noise out without
# touching the single-word calibration.
EMBED_MATCH_THRESHOLD = 0.75


def load_wav(path, target_sr=SAMPLE_RATE):
    """Read a 16-bit mono WAV as float32 in [-1, 1], resampled to target_sr."""
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    signal = raw.astype(np.float32) / 32768.0
    if rate != target_sr and len(signal) > 1:
        n = int(len(signal) * target_sr / rate)
        signal = np.interp(np.linspace(0, len(signal) - 1, n),
                           np.arange(len(signal)), signal).astype(np.float32)
    return signal


def _normalize(text):
    return "".join(text.lower().split())


def fuzzy_match(text, vocab_words, threshold=MATCH_THRESHOLD):
    """Best (label, ratio) from vocab_words = {label: kinyarwanda_text}
    whose text is close enough to `text`, or None if nothing is close enough.

    `text` may be a single isolated word (matched directly) or a full
    natural sentence with a vocabulary word or phrase embedded in it (e.g.
    "ndashaka kuvuga amazi") — every contiguous run of words up to the
    longest vocabulary phrase is tried as its own candidate, so a match is
    found regardless of what surrounds it in the sentence. Candidates
    always respect word boundaries (never the raw, unsegmented sentence
    compared as one long string) — a short word compared against a whole
    long sentence can score a deceptively high ratio by chance character
    overlap alone, with no relation to whether the word was actually said."""
    if not text.strip():
        return None
    words = text.lower().split()
    if not words:
        return None
    max_phrase_len = max((len(w.split()) for w in vocab_words.values()), default=1)

    windows = []
    for size in range(1, min(max_phrase_len, len(words)) + 1):
        for start in range(len(words) - size + 1):
            windows.append("".join(words[start:start + size]))

    best_label, best_ratio = None, 0.0
    for label, word in vocab_words.items():
        target_word = _normalize(word)
        if len(target_word) < MIN_EMBED_CHARS and len(words) > 1:
            continue   # too short/common a word to trust unless it's the whole utterance
        for candidate in windows:
            ratio = difflib.SequenceMatcher(None, candidate, target_word).ratio()
            if ratio > best_ratio:
                best_label, best_ratio = label, ratio
    effective_threshold = threshold if len(words) == 1 else EMBED_MATCH_THRESHOLD
    if best_ratio >= effective_threshold:
        return best_label, best_ratio
    return None


def _to_audio_data(signal, sr=SAMPLE_RATE):
    import speech_recognition as sr_lib
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16).tobytes()
    return sr_lib.AudioData(pcm, sr, 2)


class SpeechRecognizer:
    """Transcribes Kinyarwanda audio, online first then offline (see module
    docstring). The offline model's loading is deferred to the first call
    to load() (not __init__), so constructing this object is cheap and the
    expensive part can be run explicitly on a background thread with a
    clear "loading" state.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.ready = False               # offline model ready
        self._online_recognizer = None
        self._online_available = True    # set False after a failure, retried occasionally
        self._online_fail_count = 0

    def setup_online(self):
        try:
            import speech_recognition as sr_lib
            self._online_recognizer = sr_lib.Recognizer()
            # Must be set here: recognize_google passes this straight to
            # urlopen, and its default of None means "wait forever" —
            # overriding socket.setdefaulttimeout. One stalled request to
            # Google then froze the listener thread for good.
            self._online_recognizer.operation_timeout = ONLINE_TIMEOUT
        except Exception as e:
            logger.error(f"[SpeechRecognizer] Online recognition unavailable: {e}")
            self._online_recognizer = None

    def load(self):
        """Set up online recognition and load the offline fallback model."""
        self.setup_online()
        if self.ready:
            return
        try:
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            self._torch = torch
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, target_lang=LANGUAGE)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                MODEL_ID, target_lang=LANGUAGE, ignore_mismatched_sizes=True)
            self.model.load_adapter(LANGUAGE)
            self.model.eval()
            self.ready = True
            logger.info("[SpeechRecognizer] Offline (MMS) model loaded and ready")
        except Exception as e:
            logger.error(f"[SpeechRecognizer] Could not load offline model: {e}")
            self.ready = False

    def transcribe(self, signal):
        """signal: float32 numpy array at SAMPLE_RATE. Tries online first,
        falls back to offline. Returns text, or '' if both fail/unavailable."""
        if len(signal) < SAMPLE_RATE * 0.1:
            return ""
        if self._online_available:
            text = self._transcribe_online(signal)
            if text is not None:
                return text
        return self._transcribe_offline(signal)

    def _transcribe_online(self, signal):
        """Returns text (possibly ''), or None if the request itself failed
        (network/rate-limit/timeout) — None is what triggers the offline
        fallback; '' means Google understood nothing, which is a real answer."""
        if self._online_recognizer is None:
            return None
        import speech_recognition as sr_lib
        audio = _to_audio_data(signal)
        try:
            text = self._online_recognizer.recognize_google(audio, language=ONLINE_LANGUAGE)
            self._online_fail_count = 0
            return text
        except sr_lib.UnknownValueError:
            self._online_fail_count = 0
            return ""     # Google understood the request but heard no speech
        except Exception as e:
            self._online_fail_count += 1
            logger.warning(f"[SpeechRecognizer] Online recognition failed ({e}); "
                           f"using offline for this utterance.")
            if self._online_fail_count >= 3:
                # Several failures in a row: stop trying online for a while
                # instead of adding a multi-second timeout to every utterance.
                self._online_available = False
                logger.warning("[SpeechRecognizer] Online recognition disabled after repeated "
                               "failures — using offline only. Will retry periodically.")
            return None

    def retry_online(self):
        """Call periodically (e.g. every minute) to resume trying online
        recognition after it was disabled by repeated failures."""
        if not self._online_available:
            self._online_available = True
            self._online_fail_count = 0

    def _transcribe_offline(self, signal):
        if not self.ready:
            return ""
        try:
            inputs = self.processor(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with self._torch.no_grad():
                logits = self.model(**inputs).logits
            pred_ids = self._torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(pred_ids)[0]
        except Exception as e:
            logger.error(f"[SpeechRecognizer] Offline transcription error: {e}")
            return ""
