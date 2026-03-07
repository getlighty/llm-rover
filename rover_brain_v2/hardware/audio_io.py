"""Thin wrapper around the existing audio helpers."""

from __future__ import annotations

import threading

import audio as legacy_audio


class AudioIO:
    def __init__(self):
        self.mic_device = None
        self.mic_card = None
        self.speaker_device = None
        self.ready = False

    def discover(self) -> bool:
        try:
            self.mic_device, self.mic_card = legacy_audio.find_mic()
            self.speaker_device = legacy_audio.find_speaker()
            self.ready = True
        except Exception:
            self.ready = False
        return self.ready

    def listen(self, abort_event: threading.Event | None = None):
        if not self.ready or not self.mic_device:
            return None
        return legacy_audio.listen(self.mic_device, abort_event=abort_event)

    def speak(self, text: str, tts_client, log_fn=None):
        if not self.ready or not self.speaker_device:
            return
        legacy_audio.speak(
            text,
            tts_client,
            self.speaker_device,
            self.mic_card,
            log_fn=log_fn,
        )

