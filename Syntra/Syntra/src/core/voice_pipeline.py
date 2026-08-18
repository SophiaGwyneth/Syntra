"""
voice_pipeline.py
------------------
Handles speech-to-text (STT) capture and text-to-speech (TTS) playback for
Syntra. TTS always runs on a background daemon thread so the GUI event loop
and terminal never freeze while Syntra is speaking - callers can optionally
supply `on_start` / `on_done` callbacks to reflect the "Speaking" state on
the GUI without blocking.
"""

import logging
import os
import threading

import pyttsx3
import speech_recognition as sr

logger = logging.getLogger("VirtualAssistant")


class VoicePipeline:
    def __init__(self, tts_rate: int = None):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        self.tts_rate = tts_rate or int(os.getenv("SYNTRA_TTS_RATE", "170"))
        self._tts_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Speech-to-Text
    # ------------------------------------------------------------------ #
    def listen(self) -> str:
        """Captures audio from the microphone and converts it to text."""
        try:
            with sr.Microphone() as source:
                logger.info("Microphone listening...")
                print("\n[Listening...] Speak into your microphone.")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                logger.info("Audio captured. Processing STT...")
                text = self.recognizer.recognize_google(audio)
                logger.info("STT Transcription: '%s'", text)
                return text

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out waiting for input.")
            return ""
        except sr.UnknownValueError:
            logger.warning("STT could not understand audio.")
            return ""
        except OSError as e:
            # Typically "no default microphone" in headless/CI environments
            logger.error("Microphone unavailable: %s", e)
            return ""
        except Exception:
            logger.exception("Unexpected STT error")
            return ""

    # ------------------------------------------------------------------ #
    # Text-to-Speech (always non-blocking)
    # ------------------------------------------------------------------ #
    def speak(self, text: str, on_start=None, on_done=None, block: bool = False):
        """
        Synthesizes `text` via pyttsx3 on a background thread so the caller
        (GUI or terminal loop) is never blocked.

        on_start()  -> called right before speech playback begins
        on_done()   -> called right after speech playback finishes
        block=True  -> optionally wait for the thread to finish before
                       returning (still runs the engine off the calling
                       thread's stack; useful for scripted/CLI flows).
        """
        if not text:
            return

        def _run():
            with self._tts_lock:
                try:
                    if on_start:
                        on_start()
                    logger.info("TTS Synthesizing: '%s'", text)
                    print(f"[Syntra Speaking]: {text}")

                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.tts_rate)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    logger.exception("TTS playback failed")
                finally:
                    if on_done:
                        on_done()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        if block:
            thread.join()
