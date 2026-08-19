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
import re
import threading

import pyttsx3
import speech_recognition as sr

logger = logging.getLogger("VirtualAssistant")

# --------------------------------------------------------------------------- #
# Voice gender selection (additive - does not affect STT, and TTS falls
# back to the system default voice if nothing matches).
# --------------------------------------------------------------------------- #
# pyttsx3 voices don't reliably expose a usable `.gender` field across
# platforms (SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on
# Linux), so gender is inferred from well-known voice names/ids instead.
# This covers the common installed voices out of the box; anything not
# recognized falls back to "unknown" and is simply skipped by gender
# filtering (still selectable directly via set_voice()).
_KNOWN_FEMALE_VOICE_HINTS = {
    "zira", "hazel", "susan", "samantha", "victoria", "karen", "moira",
    "tessa", "veena", "salli", "kendra", "joanna", "ivy", "kimberly",
    "female", "eva", "fiona", "amy", "emma",
}
_KNOWN_MALE_VOICE_HINTS = {
    "david", "mark", "alex", "fred", "daniel", "george", "james",
    "matthew", "joey", "justin", "male", "diego", "ryan",
}


def _voice_text_blob(voice) -> str:
    """Concatenates every string-ish field on a pyttsx3 voice object into
    one lowercase blob, for cheap keyword matching (name/id/languages all
    vary wildly in format between SAPI5/NSSS/espeak backends)."""
    parts = [str(getattr(voice, "id", "") or ""), str(getattr(voice, "name", "") or "")]
    for lang in (getattr(voice, "languages", None) or []):
        if isinstance(lang, bytes):
            try:
                lang = lang.decode("utf-8", errors="ignore")
            except Exception:
                lang = ""
        parts.append(str(lang))
    return " ".join(parts).lower()


def _infer_gender(voice) -> str:
    """Best-effort 'male' / 'female' / 'unknown' guess for a pyttsx3 voice."""
    declared = getattr(voice, "gender", None)
    if declared:
        declared = str(declared).lower()
        if "female" in declared:
            return "female"
        if "male" in declared:
            return "male"

    blob = _voice_text_blob(voice)
    if any(hint in blob for hint in _KNOWN_FEMALE_VOICE_HINTS):
        return "female"
    if any(hint in blob for hint in _KNOWN_MALE_VOICE_HINTS):
        return "male"

    # espeak-style ids sometimes encode gender directly, e.g. "en+f3"/"en+m3".
    if re.search(r"\+f\d*\b", blob) or blob.endswith("+f"):
        return "female"
    if re.search(r"\+m\d*\b", blob) or blob.endswith("+m"):
        return "male"

    return "unknown"


class VoicePipeline:
    def __init__(self, tts_rate: int = None, voice_gender: str = None):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        self.tts_rate = tts_rate or int(os.getenv("SYNTRA_TTS_RATE", "170"))
        self._tts_lock = threading.Lock()

        # --- Voice gender selection --------------------------------------- #
        # SYNTRA_VOICE_GENDER: "male" | "female"  (default "female")
        self.voice_gender = (voice_gender or os.getenv("SYNTRA_VOICE_GENDER", "female")).strip().lower()
        self.voice_id = None  # resolved pyttsx3 voice id, applied each time speak() runs
        self._voice_state_lock = threading.Lock()
        self.set_voice_gender(self.voice_gender)

    # ------------------------------------------------------------------ #
    # Voice gender selection
    # ------------------------------------------------------------------ #
    def list_available_voices(self) -> list:
        """Enumerates every TTS voice installed on this machine.

        Returns a list of dicts: {id, name, gender}, where `gender` is
        Syntra's best-effort "male"/"female"/"unknown" guess (useful for
        populating a GUI dropdown or debugging voice selection).
        """
        voices_info = []
        try:
            probe_engine = pyttsx3.init()
            for voice in probe_engine.getProperty("voices") or []:
                voices_info.append({
                    "id": voice.id,
                    "name": getattr(voice, "name", voice.id),
                    "gender": _infer_gender(voice),
                })
            probe_engine.stop()
        except Exception:
            logger.exception("Failed to enumerate TTS voices")
        return voices_info

    def set_voice(self, voice_id: str):
        """Directly selects a TTS voice by its pyttsx3 voice id."""
        with self._voice_state_lock:
            self.voice_id = voice_id
        logger.info("TTS voice explicitly set to id '%s'", voice_id)

    def get_voice_gender(self) -> str:
        return self.voice_gender

    def set_voice_gender(self, gender: str) -> bool:
        """
        Switches the active TTS voice to best match `gender`
        ("male"/"female"). Safe to call at any time, including
        mid-conversation, from a GUI setting, a voice/text command, or
        startup config.

        Returns True if a matching voice was found and applied, False if
        no installed voice matched (in which case the previous voice_id,
        if any, is left unchanged and pyttsx3's system default is used).
        """
        gender = (gender or "").strip().lower()
        if gender not in ("male", "female"):
            logger.warning("Ignoring invalid voice gender '%s' (expected male/female)", gender)
            return False

        voices = self.list_available_voices()
        if not voices:
            logger.warning("No TTS voices available on this system - keeping default voice")
            return False

        match = next((v for v in voices if v["gender"] == gender), None)
        if not match:
            # No voice explicitly tagged with this gender - fall back to
            # any voice with unknown gender metadata rather than give up,
            # since some systems only have one installed voice total.
            match = next((v for v in voices if v["gender"] == "unknown"), None)

        if not match:
            logger.warning(
                "No installed voice matched gender='%s' - voice unchanged", gender
            )
            return False

        with self._voice_state_lock:
            self.voice_gender = gender
            self.voice_id = match["id"]

        logger.info(
            "TTS voice set to gender='%s' -> voice id '%s' (%s)",
            gender, match["id"], match["name"],
        )
        return True

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
                    if self.voice_id:
                        try:
                            engine.setProperty("voice", self.voice_id)
                        except Exception:
                            logger.exception("Failed to apply TTS voice id '%s'", self.voice_id)
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