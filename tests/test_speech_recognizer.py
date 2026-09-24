import socket
import threading
import time
import unittest
import urllib.request
from unittest import mock

import numpy as np

from src import speech_recognizer
from src.speech_recognizer import SpeechRecognizer


class StalledServer:
    """Accepts connections but never answers — like a hung Google request."""

    def __enter__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def _accept(self):
        try:
            while True:
                self.conns.append(self.sock.accept())
        except OSError:
            pass

    def __exit__(self, *exc):
        self.sock.close()


class OnlineTimeoutTest(unittest.TestCase):
    def test_stalled_request_gives_up_and_falls_back(self):
        import speech_recognition.recognizers.google as google
        real_urlopen = urllib.request.urlopen
        with StalledServer() as srv, \
                mock.patch.object(speech_recognizer, "ONLINE_TIMEOUT", 1), \
                mock.patch.object(google, "urlopen",
                                  lambda req, timeout=None: real_urlopen(
                                      f"http://127.0.0.1:{srv.port}/", timeout=timeout)):
            rec = SpeechRecognizer()
            rec.setup_online()
            start = time.time()
            result = rec._transcribe_online(np.full(16000, 0.01, dtype=np.float32))
            elapsed = time.time() - start
        self.assertIsNone(result)          # None = use the offline model instead
        self.assertLess(elapsed, 5)

    def test_repeated_failures_disable_online_until_retry(self):
        rec = SpeechRecognizer()
        rec._online_recognizer = mock.Mock()
        rec._online_recognizer.recognize_google.side_effect = OSError("network down")
        audio = np.full(16000, 0.01, dtype=np.float32)
        for _ in range(3):
            rec._transcribe_online(audio)
        self.assertFalse(rec._online_available)
        rec.retry_online()
        self.assertTrue(rec._online_available)

    def test_very_short_audio_is_skipped(self):
        rec = SpeechRecognizer()
        rec._online_recognizer = mock.Mock()
        self.assertEqual(rec.transcribe(np.zeros(100, dtype=np.float32)), "")
        rec._online_recognizer.recognize_google.assert_not_called()


if __name__ == "__main__":
    unittest.main()
