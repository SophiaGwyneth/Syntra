"""
main.py
-------
Entry point for Syntra, the AI Home Assistant.

Wires together four independent layers:
    src.core.ai_engine        -> natural language -> structured intent
    src.core.voice_pipeline   -> STT capture + non-blocking TTS playback
    src.simulator.home_simulator -> authoritative virtual device state
    src.gui.gui               -> dark-mode customtkinter dashboard

Each layer stays independently testable and swappable.
"""

import datetime
import logging
import os
import re
import sys
import threading
import time

from dotenv import load_dotenv

# Make "src" importable as a proper package regardless of the current
# working directory the app is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.ai_engine import AIEngine
from src.core.music_player import MusicPlayer
from src.core.time_utils import TimeQueryError, get_local_time, get_time_in_location
from src.core.voice_pipeline import VoicePipeline
from src.core.weather_alarm import AlarmClock, AlarmError, WeatherError, get_weather
from src.gui.gui import SyntraGUI
from src.simulator.home_simulator import HomeSimulator

load_dotenv()

LOG_FILE = os.getenv("SYNTRA_LOG_FILE", "assistant_execution.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("VirtualAssistant")

GREETING = "Hello! I am Syntra, your home assistant. How can I assist you today?"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default alarm sound. Place the file in the project's music/ folder (same
# folder MusicPlayer already reads from) so both features share one asset
# directory; override per-alarm by passing a different path to set_alarm().
# NOTE: this no longer has to be an exact path - AlarmClock now falls back
# to a recursive search under PROJECT_ROOT if the file has been moved to a
# different subfolder, so this filename alone is enough.
DEFAULT_ALARM_SOUND = os.path.join(
    PROJECT_ROOT,
    "alarm",
    "alarm.mp3",
)


class SyntraApp:
    def __init__(self):
        logger.info("Initializing Syntra AI Home Assistant...")

        self.ai_engine = AIEngine()
        # SYNTRA_VOICE_GENDER (.env) picks the initial TTS voice; it can
        # also be changed live via the GUI sidebar or a typed/spoken
        # command (see _process_command's voice-gender intercept).
        self.voice_pipeline = VoicePipeline()
        self.simulator = HomeSimulator(on_state_change=self._on_state_change)
        self.music_player = MusicPlayer(on_change=self._on_music_change)
        self.alarm_clock = AlarmClock(on_trigger=self._on_alarm_trigger, search_root=PROJECT_ROOT)
        self._current_alarm_time = None

        self.gui = SyntraGUI(
            on_text_command=self.handle_text_command,
            on_voice_command=self.handle_voice_command,
            on_device_toggle=self.handle_device_toggle,
            on_thermostat_adjust=self.handle_thermostat_adjust,
            on_lock_toggle=self.handle_lock_toggle,
            on_music_control=self.handle_music_control,
            on_alarm_set=self.handle_alarm_set,
            on_alarm_stop=self.handle_alarm_stop,
            on_voice_gender_change=self.handle_voice_gender_change,
        )

        # Prime the GUI with the simulator's + music player's initial state.
        self.gui.update_device_display(self.simulator.get_state())
        self.gui.update_music_display(self.music_player.get_now_playing())
        self.gui.update_alarm_display("No alarm set")
        self.gui.set_voice_gender_display(self.voice_pipeline.get_voice_gender())

    # ------------------------------------------------------------------ #
    # Dual output helper: EVERY response goes to terminal + GUI log + TTS
    # ------------------------------------------------------------------ #
    def _respond(self, text: str, sender: str = "Syntra", speak: bool = True):
        """Delivers a single message through all three channels at once:
        terminal stdout (via logger/print), the GUI chat log, and TTS audio
        (on its own background thread so nothing blocks)."""
        print(f"[{sender}] {text}")
        logger.info("%s response: %s", sender, text)
        self.gui.append_message(sender, text, tag="syntra")

        if speak:
            self.gui.set_status("speaking", "Speaking response...")
            self.voice_pipeline.speak(
                text,
                on_done=lambda: self.gui.set_status("idle", "Ready.")
            )
        else:
            self.gui.set_status("idle", "Ready.")

    def _on_state_change(self, state: dict, changed_target: str):
        """Callback from HomeSimulator -> keep the GUI device panel in sync,
        regardless of which thread triggered the change."""
        self.gui.update_device_display(state, changed_target)

    def _on_music_change(self, now_playing: dict):
        """Callback from MusicPlayer -> keep the GUI's Now Playing card in
        sync, regardless of which thread triggered the change (AI command,
        manual transport button, or the background 'track finished' watcher)."""
        self.gui.update_music_display(now_playing)

    def _on_alarm_trigger(self, sound_path: str):
        """Callback from AlarmClock's background thread the instant the
        alarm sound starts playing. Runs off the main thread, so it only
        does thread-safe work: log + dual-output response + popup."""
        logger.info("Alarm fired, playing '%s'", sound_path)
        self.gui.update_alarm_display("ALARM RINGING!", ringing=True)
        self.gui.show_alarm_popup("Wake up! Your alarm is going off.")
        self._respond("Wake up! Your alarm is going off.")

    # ------------------------------------------------------------------ #
    # Greeting
    # ------------------------------------------------------------------ #
    def greet(self):
        logger.info("Delivering session greeting.")
        self._respond(GREETING)

    # ------------------------------------------------------------------ #
    # Command handling (shared pipeline for both voice and typed text)
    # ------------------------------------------------------------------ #
    def _process_command(self, user_text: str):
        """Runs AI parsing + simulator updates + dual-output response.
        Executed entirely on a background thread; wrapped in a broad
        exception handler so a malformed command or a dead model server can
        never crash Syntra."""
        try:
            start_time = time.time()
            self.gui.set_status("processing", f'Processing: "{user_text}"...')

            lowered = user_text.lower().strip()

            # Direct Intercept: Agad na patayin ang alarm kapag sinabi ang turn off/stop
            # nang hindi na dumadaan sa AI o nagtatanong ng oras.
            alarm_stop_words = ["turn off", "stop", "cancel", "disable", "silence", "patayin", "off"]
            if "alarm" in lowered and any(word in lowered for word in alarm_stop_words):
                response = self._handle_direct_alarm_stop()
                self._respond(response)
                return

            # Direct Intercept: voice gender switch ("switch to male voice",
            # "use the female voice", "change your voice to male", etc.) -
            # handled deterministically here, same as the alarm intercept
            # above, so it never depends on the LLM parsing it correctly.
            gender_match = re.search(r'\b(male|female)\b', lowered)
            voice_change_words = ["switch", "change", "set", "use", "make"]
            if "voice" in lowered and gender_match and any(w in lowered for w in voice_change_words):
                response = self._handle_direct_voice_gender_change(gender_match.group(1))
                self._respond(response)
                return

            intent_res = self.ai_engine.parse_intent(user_text)

            applied_any = False
            simulator_rejected = False  # a home-device action was attempted
            music_feedback = []
            other_feedback = []
            for action_item in intent_res.actions:
                target = getattr(action_item, "target", None)
                if target == "music_player":
                    music_feedback.append(self._apply_music_action(action_item))
                    applied_any = True
                elif target == "weather":
                    other_feedback.append(self._apply_weather_action(action_item))
                    applied_any = True
                elif target == "alarm":
                    other_feedback.append(self._apply_alarm_action(action_item, user_text=user_text))
                    applied_any = True
                elif target == "time":
                    other_feedback.append(self._apply_time_action(action_item))
                    applied_any = True
                elif self.simulator.apply_action(action_item):
                    applied_any = True
                else:
                    # target/action were recognized shapes (e.g. thermostat
                    # set_temp) but the simulator declined to apply them -
                    # most commonly a missing/unparseable value. Don't let
                    # the LLM's own narrated spoken_response claim success
                    # in the chat log while the GUI silently stays stale.
                    simulator_rejected = True

            elapsed = round(time.time() - start_time, 2)
            logger.info("Command processed in %ss (actions applied: %s)", elapsed, applied_any)

            if music_feedback or other_feedback:
                final_response = " ".join(m for m in (music_feedback + other_feedback) if m)
            elif simulator_rejected and not applied_any:
                final_response = (
                    "I heard the command but couldn't tell what value to set - "
                    "could you repeat it with a specific number?"
                )
            else:
                final_response = intent_res.spoken_response
            if not intent_res.actions and not final_response:
                final_response = "I'm not sure how to help with that."
            self._respond(final_response)

        except Exception:
            logger.exception("Unhandled error while processing command: '%s'", user_text)
            self.gui.show_error("Something went wrong processing that command.")
            self._respond(
                "I'm sorry, something went wrong on my end. Please try again.",
            )
        finally:
            self.gui.reset_voice_button()

    def _apply_music_action(self, action_item) -> str:
        """Routes a single AI-issued music_player action to the MusicPlayer
        and returns its spoken-friendly confirmation string. Never raises."""
        act = getattr(action_item, "action", None)
        query = getattr(action_item, "query", None)
        value = getattr(action_item, "value", None)
        source = getattr(action_item, "source", None) or "local"
        try:
            if act == "play":
                return self.music_player.play(query, source=source)
            elif act == "pause":
                return self.music_player.pause()
            elif act == "resume":
                return self.music_player.resume()
            elif act == "stop":
                return self.music_player.stop()
            elif act == "next":
                return self.music_player.next()
            elif act == "previous":
                return self.music_player.previous()
            elif act == "set_volume" and value is not None:
                return self.music_player.set_volume(float(value))
            else:
                logger.warning("Ignoring unsupported music action: %s", act)
                return "I didn't recognize that music command."
        except Exception:
            logger.exception("Failed to apply music action: %s", action_item)
            return "Something went wrong with the music player."

    def _apply_weather_action(self, action_item) -> str:
        """Routes a 'weather' action to get_weather() and returns a
        spoken-friendly confirmation string. Never raises."""
        city = getattr(action_item, "query", None)
        if not city:
            return "Which city would you like the weather for?"
        try:
            report = get_weather(city)
            return report.spoken_summary()
        except WeatherError as err:
            logger.warning("Weather lookup failed for '%s': %s", city, err)
            return str(err)
        except Exception:
            logger.exception("Unexpected error fetching weather for '%s'", city)
            return "Something went wrong while checking the weather."

    # Words the LLM sometimes hallucinates into the 'query' field for a
    # time request even though they aren't a real place ("today", "now",
    # etc.) - treat any of these as "no location given" -> local time.
    _TIME_FILLER_WORDS = {
        "today", "now", "right now", "currently", "please", "exactly",
        "here", "this moment",
    }

    def _apply_time_action(self, action_item) -> str:
        """Routes a 'time' action to time_utils and returns a
        spoken-friendly response. Handles both local time ('what time is
        it?') and world time ('what time is it in South Korea?'). Never
        raises."""
        location = getattr(action_item, "query", None)
        if location and location.strip().lower() in self._TIME_FILLER_WORDS:
            location = None
        try:
            if location:
                return get_time_in_location(location)
            return get_local_time()
        except TimeQueryError as err:
            logger.warning("Time lookup failed for '%s': %s", location, err)
            return str(err)
        except Exception:
            logger.exception("Unexpected error fetching time for '%s'", location)
            return "Something went wrong while checking the time."

    # ------------------------------------------------------------------ #
    # Updated Alarm Action Handlers
    # ------------------------------------------------------------------ #
    def _handle_direct_alarm_stop(self) -> str:
        """Awtomatikong pinapatay ang tumutunog o nakatakdang alarm."""
        try:
            if hasattr(self.alarm_clock, "stop_alarm"):
                self.alarm_clock.stop_alarm()
            elif hasattr(self.alarm_clock, "cancel_alarm"):
                self.alarm_clock.cancel_alarm()
            elif hasattr(self.alarm_clock, "stop"):
                self.alarm_clock.stop()
            elif hasattr(self.alarm_clock, "off"):
                self.alarm_clock.off()
            self._current_alarm_time = None
            self.gui.update_alarm_display("No alarm set")
            self.gui.dismiss_alarm_popup()
            return "Alarm has been turned off."
        except Exception:
            logger.exception("Error while turning off the alarm")
            return "Could not turn off the alarm."

    # ------------------------------------------------------------------ #
    # Voice gender switching (typed/spoken command + GUI sidebar control)
    # ------------------------------------------------------------------ #
    def _handle_direct_voice_gender_change(self, gender: str) -> str:
        """Applies a male/female voice switch triggered by typed or spoken
        text (e.g. 'switch to male voice'). Never raises."""
        try:
            if self.voice_pipeline.set_voice_gender(gender):
                self.gui.set_voice_gender_display(gender)
                return f"Okay, switching to a {gender} voice."
            return f"Sorry, I couldn't find a {gender} voice installed on this system."
        except Exception:
            logger.exception("Failed to switch voice gender to '%s'", gender)
            return "Something went wrong while changing the voice."

    def handle_voice_gender_change(self, gender: str):
        """Callback from the GUI sidebar's Voice Gender selector, bypassing
        the AI entirely."""
        def task():
            try:
                if self.voice_pipeline.set_voice_gender(gender):
                    self._respond(f"Voice switched to {gender}.", speak=True)
                else:
                    self.gui.show_error(f"No {gender} voice is installed on this system.")
                    self.gui.set_voice_gender_display(self.voice_pipeline.get_voice_gender())
            except Exception:
                logger.exception("Failed to change voice gender to '%s'", gender)
                self.gui.show_error("Could not change the voice.")

        threading.Thread(target=task, daemon=True).start()

    def _apply_alarm_action(self, action_item, user_text: str = "") -> str:
        """Routes an 'alarm' action to AlarmClock."""
        act = getattr(action_item, "action", None) or getattr(action_item, "command", None)
        alarm_time = getattr(action_item, "query", None) or getattr(action_item, "time", None)

        full_text = f"{act} {alarm_time} {user_text}".lower()

        # Kung ang intent ay naglalaman ng pagpatay sa alarm, i-off agad
        if any(word in full_text for word in ["off", "stop", "cancel", "disable", "silence"]):
            return self._handle_direct_alarm_stop()

        # Extract/Format ng oras para sa pagse-set ng alarm
        parsed_time = self._parse_alarm_time_str(alarm_time) or self._parse_alarm_time_str(user_text)

        if not parsed_time:
            return "What time would you like the alarm set for?"

        try:
            self.alarm_clock.set_alarm(parsed_time, DEFAULT_ALARM_SOUND)
            self._current_alarm_time = parsed_time
            self.gui.update_alarm_display(f"Alarm set for {parsed_time}")
            return f"Alarm set for {parsed_time}."
        except AlarmError as err:
            logger.warning("Failed to set alarm for '%s': %s", parsed_time, err)
            return str(err)
        except Exception:
            logger.exception("Unexpected error setting alarm for '%s'", parsed_time)
            return "Something went wrong while setting the alarm."

    def _parse_alarm_time_str(self, text: str) -> str:
        """Helper na naghahanap ng time string (hal. '2:30 pm', '07:00 AM')."""
        if not text:
            return None
        match = re.search(r'(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def handle_text_command(self, text: str):
        threading.Thread(target=self._process_command, args=(text,), daemon=True).start()

    def handle_voice_command(self):
        def task():
            try:
                self.gui.set_status("listening", "Listening to voice command...")
                user_speech = self.voice_pipeline.listen()

                if not user_speech:
                    self.gui.set_status("idle", "Ready.")
                    self.gui.reset_voice_button()
                    self._respond("I couldn't hear or recognize that. Please try again.")
                    return

                self.gui.append_message("You", user_speech, tag="user")
                self._process_command(user_speech)

            except Exception:
                logger.exception("Unhandled error in voice command pipeline")
                self.gui.show_error("Voice input failed unexpectedly.")
                self.gui.reset_voice_button()
                self.gui.set_status("idle", "Ready.")

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Manual device quick-toggle controls (from the GUI, bypassing the AI)
    # ------------------------------------------------------------------ #
    def handle_device_toggle(self, target: str):
        try:
            if target in ("living_room_light", "kitchen_light", "humidifier", "air_conditioner"):
                self.simulator.toggle_light(target)
            elif target == "back_door":
                self.simulator.toggle_lock(target)
        except Exception:
            logger.exception("Failed to toggle device: %s", target)
            self.gui.show_error(f"Could not toggle {target}.")

    def handle_thermostat_adjust(self, delta: float):
        try:
            self.simulator.adjust_thermostat(delta)
        except Exception:
            logger.exception("Failed to adjust thermostat")
            self.gui.show_error("Could not adjust thermostat.")

    def handle_lock_toggle(self):
        try:
            self.simulator.toggle_lock()
        except Exception:
            logger.exception("Failed to toggle front door lock")
            self.gui.show_error("Could not toggle the front door lock.")

    def handle_alarm_set(self, time_str: str):
        """Manual alarm-set from the GUI sidebar's Alarm card, bypassing
        the AI entirely. Reuses the same flexible time parser the AI path
        uses so both 'HH:MM' and '7:30 PM'-style entries work."""
        def task():
            parsed_time = self._parse_alarm_time_str(time_str) or time_str.strip()
            try:
                self.alarm_clock.set_alarm(parsed_time, DEFAULT_ALARM_SOUND)
                self._current_alarm_time = parsed_time
                self.gui.update_alarm_display(f"Alarm set for {parsed_time}")
                self._respond(f"Alarm set for {parsed_time}.", speak=False)
            except AlarmError as err:
                logger.warning("Failed to set alarm for '%s': %s", parsed_time, err)
                self.gui.show_error(str(err))
            except Exception:
                logger.exception("Unexpected error setting alarm for '%s'", parsed_time)
                self.gui.show_error("Something went wrong while setting the alarm.")

        threading.Thread(target=task, daemon=True).start()

    def handle_alarm_stop(self):
        """Manual alarm stop/cancel from the GUI sidebar's Alarm card."""
        def task():
            response = self._handle_direct_alarm_stop()
            self._respond(response, speak=False)

        threading.Thread(target=task, daemon=True).start()

    def handle_music_control(self, action: str, value: float = None):
        """Manual transport controls from the GUI's Now Playing card
        (play/pause/stop/next/previous/volume), bypassing the AI entirely."""
        def task():
            try:
                if action == "play_pause":
                    now = self.music_player.get_now_playing()
                    if now["is_playing"] and not now["is_paused"]:
                        self.music_player.pause()
                    elif now["is_paused"]:
                        self.music_player.resume()
                    else:
                        self.music_player.play()
                elif action == "stop":
                    self.music_player.stop()
                elif action == "next":
                    self.music_player.next()
                elif action == "previous":
                    self.music_player.previous()
                elif action == "volume":
                    self.music_player.adjust_volume(value or 0.0)
            except Exception:
                logger.exception("Failed to handle manual music control: %s", action)
                self.gui.show_error("Could not control the music player.")

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------ #
    # App lifecycle
    # ------------------------------------------------------------------ #
    def run(self):
        logger.info("Launching Syntra GUI application...")
        # Greet shortly after the window is visible, not before, so the
        # user actually sees the greeting arrive in the chat log.
        self.gui.after(500, self.greet)
        self.gui.mainloop()


if __name__ == "__main__":
    try:
        app = SyntraApp()
        app.run()
    except Exception:
        logging.getLogger("VirtualAssistant").exception("Fatal error during startup")
        raise