"""
HandsToVoice — Voice Output Module
Priority:
  1. Custom user-recorded audio file from data/audio/  (highest quality)
  2. pyttsx3 offline TTS fallback (English, no internet needed)
Signs without a recording will be spoken by the system TTS voice.
"""

import os
import time
import threading
import subprocess

from src.logger import get_logger

logger = get_logger("tts")


def _init_pygame_mixer():
    """
    Initialise pygame mixer using PulseAudio backend.

    PipeWire provides full PulseAudio compatibility, so SDL_AUDIODRIVER=pulse
    routes through the software mixer — multiple apps can share the audio
    device simultaneously (no 'device or resource busy' errors).
    """
    try:
        # Must be set BEFORE pygame is imported or mixer is initialised
        os.environ.setdefault("SDL_AUDIODRIVER", "pulse")

        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
        return True
    except Exception as e:
        logger.error(f"[Voice] pygame mixer init failed: {e}")
        return False


# Initialise once at module load so the driver is ready before anything else
_PYGAME_OK = _init_pygame_mixer()


class VoiceOutput:
    """Handles custom voice playback for recognised KSL signs."""

    def __init__(self, language="rw", use_offline=False, vocabulary=None,
                 audio_dir="data/audio"):
        self.language = language
        self.vocabulary = vocabulary
        self.audio_dir = audio_dir
        self._is_speaking = False
        self._active_speakers = 0     # see _speak_impl: _is_speaking must
                                       # stay True as long as ANY concurrent
                                       # speak() call is still running,
                                       # regardless of which one finishes first
        self._lock = threading.Lock()
        self._channel = None   # dedicated pygame channel for sign audio
        self._tts_engine = None  # pyttsx3 fallback engine

        os.makedirs(self.audio_dir, exist_ok=True)

        # Grab our dedicated pygame channel (channel 7)
        if _PYGAME_OK:
            try:
                import pygame
                self._channel = pygame.mixer.Channel(7)
            except Exception:
                pass

        # Initialise pyttsx3 TTS fallback (offline, no internet required)
        self._init_tts_engine()

    def _init_tts_engine(self):
        """Initialise pyttsx3 as a TTS fallback engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Use English (Great Britain) voice — clearest for Kinyarwanda words
            voices = engine.getProperty('voices')
            en_voice = next(
                (v for v in voices if 'en-gb' in str(v.languages) and 'scotland' not in str(v.id)),
                None
            )
            if en_voice is None:
                # Fallback to any English voice
                en_voice = next((v for v in voices if 'en' in str(v.languages)), None)
            if en_voice:
                engine.setProperty('voice', en_voice.id)
                logger.info(f"[Voice] TTS fallback: {en_voice.name}")
            engine.setProperty('rate', 140)    # slightly slower for clarity
            engine.setProperty('volume', 1.0)
            self._tts_engine = engine
        except Exception as e:
            logger.error(f"[Voice] pyttsx3 init failed (TTS fallback disabled): {e}")
            self._tts_engine = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _has_custom_audio(self, label):
        """Return True if a recording exists for this sign label."""
        if not label:
            return False
        slug = str(label).strip().lower().replace(" ", "_")
        for ext in (".wav", ".mp3", ".ogg"):
            p = os.path.join(self.audio_dir, f"{slug}{ext}")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return True
        return False

    def _resolve_path(self, label):
        """Return the audio file path for a label, or None if not found."""
        slug = str(label).strip().lower().replace(" ", "_")
        for ext in (".wav", ".mp3", ".ogg"):
            p = os.path.join(self.audio_dir, f"{slug}{ext}")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        return None

    # ── core playback ─────────────────────────────────────────────────────────

    def _play_file(self, path):
        """
        Play a single audio file and block until it finishes.

        Strategy:
          1. pygame.mixer.Sound on channel 7  (PulseAudio backend, shareable)
          2. aplay default device             (PipeWire ALSA plugin fallback)
        """
        # ── Method 1: pygame with PulseAudio backend ──────────────────────
        if _PYGAME_OK and self._channel is not None:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    _init_pygame_mixer()
                    self._channel = pygame.mixer.Channel(7)

                self._channel.stop()
                sound = pygame.mixer.Sound(path)
                self._channel.play(sound)

                # Wait for playback to finish
                while self._channel.get_busy() and self._is_speaking:
                    time.sleep(0.05)

                logger.info(f"[Voice] ✔ Played via pygame: {os.path.basename(path)}")
                return True
            except Exception as e:
                logger.error(f"[Voice] pygame error: {e} — trying aplay")

        # ── Method 2: aplay (default device → PipeWire ALSA plugin) ──────
        try:
            proc = subprocess.Popen(
                ["aplay", path],          # no -D flag → uses 'default' device
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            self._active_proc = proc
            _, err = proc.communicate()
            self._active_proc = None
            if proc.returncode != 0:
                logger.error(f"[Voice] aplay error: {err.decode().strip()}")
            else:
                logger.info(f"[Voice] ✔ Played via aplay: {os.path.basename(path)}")
            return proc.returncode == 0
        except FileNotFoundError:
            logger.info("[Voice] Neither pygame nor aplay available — no audio output")
        except Exception as e:
            logger.error(f"[Voice] aplay exception: {e}")

        return False

    def _play_custom_audio(self, label):
        """Resolve label to a file and play it."""
        path = self._resolve_path(label)
        if not path:
            return False
        logger.info(f"[Voice] ▶ {label}")
        return self._play_file(path)

    # ── public API ────────────────────────────────────────────────────────────

    def speak(self, text, label=None, labels=None, blocking=False):
        """
        Play pre-recorded audio for the given sign label(s).

        Args:
            text:     Kinyarwanda text (used only for vocabulary lookup when
                      no label is supplied).
            label:    Single sign label to play.
            labels:   List of sign labels to play sequentially.
            blocking: If True wait until playback finishes; otherwise run in
                      a background daemon thread.
        """
        if not text and not label and not labels:
            return

        if blocking:
            self._speak_impl(text, label, labels)
        else:
            t = threading.Thread(
                target=self._speak_impl, args=(text, label, labels), daemon=True
            )
            t.start()

    def _speak_impl(self, text, label=None, labels=None):
        """Internal: play audio on the calling thread.

        speak() starts a new daemon thread on every call without waiting
        for a previous one to finish (e.g. back-to-back batches), so calls
        can overlap in either order. Without the active-count tracking
        below, whichever call happened to finish first would clear
        _is_speaking even while another one was still actively playing —
        which let the speech listener un-mute mid-playback and pick up the
        system's own voice, corrupting its background-noise calibration so
        it stopped hearing real speech afterward.
        """
        with self._lock:
            self._active_speakers += 1
            self._is_speaking = True

        try:
            # Stop any currently playing audio first
            self._stop_playback()

            with self._lock:
                self._is_speaking = True   # re-set after stop

            # 1. List of labels — play each in sequence
            if labels:
                for lbl in labels:
                    lbl = str(lbl)   # cast np.str_ → plain str so file lookup works
                    if self._has_custom_audio(lbl):
                        self._play_custom_audio(lbl)
                        time.sleep(0.1)
                    else:
                        # No custom recording — fall back to TTS
                        kw = lbl
                        if self.vocabulary:
                            kw = self.vocabulary.get_kinyarwanda(lbl) or lbl
                        self._speak_tts(kw)
                        time.sleep(0.1)
                return

            # 2. Single label
            if label:
                label = str(label)   # cast np.str_ → plain str
                if self._has_custom_audio(label):
                    self._play_custom_audio(label)
                else:
                    # No custom recording — fall back to TTS
                    kw = text if text else label
                    if self.vocabulary:
                        kw = self.vocabulary.get_kinyarwanda(label) or kw
                    self._speak_tts(kw)
                return

            # 3. Plain text — try to match words to vocabulary labels
            if text and self.vocabulary:
                for word in text.strip().split():
                    clean = word.strip(".,!?").lower()
                    found = None
                    for sl, info in self.vocabulary.signs.items():
                        if info["kinyarwanda"].lower() == clean:
                            found = sl
                            break
                    if found:
                        if self._has_custom_audio(found):
                            self._play_custom_audio(found)
                            time.sleep(0.1)
                        else:
                            # No custom recording — fall back to TTS
                            kw = self.vocabulary.get_kinyarwanda(found) or clean
                            self._speak_tts(kw)
                            time.sleep(0.1)
                    else:
                        logger.warning(f"[Voice] ⚠ '{clean}' not in vocabulary")

        except Exception as e:
            logger.error(f"[Voice] Playback error: {e}")
        finally:
            with self._lock:
                self._active_speakers = max(0, self._active_speakers - 1)
                if self._active_speakers == 0:
                    self._is_speaking = False

    # ── TTS fallback ──────────────────────────────────────────────────────────

    def _speak_tts(self, text):
        """Speak text using pyttsx3 TTS fallback engine."""
        if not text:
            return
        if self._tts_engine is None:
            logger.warning(f"[Voice] ⚠ TTS engine not available — cannot speak '{text}'")
            return
        try:
            logger.info(f"[Voice] 🔊 TTS fallback: '{text}'")
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"[Voice] TTS error: {e}")

    # ── control ───────────────────────────────────────────────────────────────

    def _stop_playback(self):
        """Stop all active playback immediately."""
        # Stop pygame channel
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass

        # Kill any aplay subprocess
        proc = getattr(self, '_active_proc', None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            self._active_proc = None

    def stop(self):
        """Public stop — called when recognition stops or new sign fires."""
        self._stop_playback()
        with self._lock:
            self._is_speaking = False

    def is_speaking(self):
        with self._lock:
            return self._is_speaking

    def cleanup(self):
        self.stop()


