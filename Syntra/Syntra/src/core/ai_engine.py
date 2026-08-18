"""
ai_engine.py
------------
Converts free-form natural language (from STT or typed text) into a
structured, validated set of smart-home actions using a local Ollama model,
backed by a programmatic normalization layer to prevent model hallucinations.
"""

import json
import logging
import os
import re
from typing import List, Optional

import ollama
from pydantic import BaseModel, Field

logger = logging.getLogger("VirtualAssistant")

# --------------------------------------------------------------------------- #
# Deterministic weather / alarm parsing helpers
# --------------------------------------------------------------------------- #

_WEATHER_RE = re.compile(
    r"\bweather\b.*?\b(?:in|for|at)\s+([a-zA-Z\s.\-]+?)\s*[?.!]?$"
    r"|\btemperature\b.*?\b(?:in|for|at)\s+([a-zA-Z\s.\-]+?)\s*[?.!]?$",
    re.IGNORECASE,
)
_WEATHER_BARE_RE = re.compile(r"\b(weather|temperature)\b", re.IGNORECASE)

_ALARM_RE = re.compile(
    r"\balarm\b|\bwake me up\b",
    re.IGNORECASE,
)

# Matches "what time is it", "current time", "what's the time in South
# Korea", or the more explicit "time in Tokyo" / "time for New York".
# Trailing filler words like "today"/"now"/"currently" are swallowed so
# they never get mistaken for a location. Deliberately does NOT match a
# bare "time" appearing elsewhere in a sentence (e.g. "wake me up at a
# good time"), to avoid stealing alarm commands.
_TIME_FILLER_WORDS = r"today|now|right\s+now|currently|please|exactly"
_TIME_RE = re.compile(
    r"\b(?:what(?:'s| is)?\s+(?:the\s+)?time(?:\s+is\s+it)?|current\s+time|time\s+right\s+now)\b"
    r"(?:\s+(?:is\s+it\s+)?(?:in|at|for)\s+(?P<location>[a-zA-Z\s.\-]+?))?"
    rf"(?:\s+(?:{_TIME_FILLER_WORDS}))*"
    r"\s*[?.!]?$"
    r"|\btime\s+(?:in|at|for)\s+(?P<location2>[a-zA-Z\s.\-]+?)\s*[?.!]?$",
    re.IGNORECASE,
)

# Generic/filler words that mean "just play something" rather than naming
# an actual song/artist - these must NEVER be treated as a literal track
# title to search for.
_GENERIC_MUSIC_TERMS = {
    "music", "song", "songs", "some music", "random music", "any music",
    "anything", "a song", "tunes", "tracks", "some songs", "something",
    "random", "random song", "random songs", "random tracks", "some tunes",
    "some song", "a track", "a tune", "the music", "some random music",
}
# Matches "7", "7:30", "07:30", optionally followed by am/pm.
_TIME_TOKEN_RE = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3])(?::(?P<minute>[0-5]\d))?\s*(?P<meridiem>am|pm)?\b",
    re.IGNORECASE,
)


def _parse_clock_time(text: str) -> Optional[str]:
    """Extracts the first clock-like time mention in `text` and normalizes
    it to a 24-hour 'HH:MM' string, or returns None if nothing plausible
    is found."""
    for match in _TIME_TOKEN_RE.finditer(text):
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        meridiem = (match.group("meridiem") or "").lower()

        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    return None


class SmartHomeAction(BaseModel):
    action: str = Field(
        description="Action to perform: 'turn_on', 'turn_off', 'set_temp', "
                    "'increase_temp', 'decrease_temp', 'lock', 'unlock', "
                    "'play', 'pause', 'resume', 'stop', 'next', 'previous', 'set_volume', "
                    "'get_weather', 'set_alarm', 'get_time'"
    )
    target: str = Field(
        description="Target device: 'living_room_light', 'kitchen_light', "
                    "'thermostat', 'front_door_lock', 'back_door', 'humidifier', "
                    "'air_conditioner', 'music_player', 'weather', 'alarm', 'time'"
    )
    value: Optional[float] = Field(
        default=None,
        description="Numeric value for temperature settings or music_player volume (0.0-1.0) if applicable",
    )
    query: Optional[str] = Field(
        default=None,
        description="Song/artist name for music_player 'play'; city name for target 'weather'; "
                    "'HH:MM' 24-hour time string for target 'alarm'; city/country name for target "
                    "'time' (leave empty for local time)",
    )
    source: Optional[str] = Field(
        default="local",
        description="Playback backend, ONLY used when target is 'music_player': "
                    "'local' (default - offline library), 'spotify', or 'youtube'.",
    )


class IntentResponse(BaseModel):
    actions: List[SmartHomeAction] = Field(description="List of commands to execute")
    spoken_response: str = Field(description="Natural spoken/text feedback response for the user")


class AIEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("SYNTRA_MODEL_NAME", "qwen2.5:1.5b")

    @staticmethod
    def _normalize_action(action_data: dict, text_input: str = "") -> dict:
        """Programmatically fixes common small-model string variations and hallucinations
        to guarantee exact compatibility with HomeSimulator.
        """
        text_lower = text_input.lower().strip()
        target = str(action_data.get("target", "")).lower().strip()
        action = str(action_data.get("action", "")).lower().strip()
        query = action_data.get("query")
        val = action_data.get("value")

        # 0. Global intent overrides based on raw text to prevent cross-device hallucinations
        both_doors = "both" in text_lower and any(d in text_lower for d in ["door", "lock"])
        if both_doors:
            # "lock/unlock both doors" mentions neither 'front' nor 'back',
            # so the single-target overrides below would collapse every
            # action in the list onto front_door_lock (or back_door),
            # silently dropping the other door. Skip the override here and
            # let each action's own 'target' field (from the LLM) drive
            # the per-action normalization in the sections below instead.
            pass
        elif "unlock" in text_lower and any(d in text_lower for d in ["door", "lock"]):
            target = "back_door" if "back" in text_lower else "front_door_lock"
            action = "unlock"
        elif "lock" in text_lower and any(d in text_lower for d in ["door", "lock"]) and "unlock" not in text_lower:
            target = "back_door" if "back" in text_lower else "front_door_lock"
            action = "lock"
        elif "humidifier" in text_lower and not (
            any(k in text_lower for k in ["air conditioner", "air conditioning", "aircon", "a/c"]) or " ac " in f" {text_lower} "
        ):
            # Guarded to single-device mentions only: if the sentence also
            # mentions the AC ("humidifier and aircon"), forcing every
            # action in the list to 'humidifier' here would silently eat
            # the air_conditioner action too. Compound commands instead
            # fall through to the per-action target normalization below,
            # which trusts each action's own 'target' field from the LLM.
            target = "humidifier"
            action = "turn_off" if any(k in text_lower for k in ["turn off", "switch off", "shut off", "stop"]) else "turn_on"
        elif (
            any(k in text_lower for k in ["air conditioner", "air conditioning", "aircon", "a/c"]) or " ac " in f" {text_lower} "
        ) and "humidifier" not in text_lower:
            target = "air_conditioner"
            action = "turn_off" if any(k in text_lower for k in ["turn off", "switch off", "shut off", "stop"]) else "turn_on"
        elif any(k in text_lower for k in ["stop music", "stop playing", "shut up", "turn off music"]) or text_lower == "stop":
            target = "music_player"
            action = "stop"
            action_data["query"] = None
        elif any(k in text_lower for k in ["volume", "sound", "louder", "quieter", "lower the volume", "turn up volume", "turn up the volume"]):
            target = "music_player"
            action = "set_volume"
        elif any(k in text_lower for k in ["next music", "next song", "next track", "skip track"]) or (action == "play" and query and "next" in str(query).lower()):
            target = "music_player"
            action = "next"
            action_data["query"] = None
        elif any(k in text_lower for k in ["previous music", "previous song", "previous track"]) or (action == "play" and query and any(q in str(query).lower() for q in ["prev", "back"])):
            target = "music_player"
            action = "previous"
            action_data["query"] = None
        elif text_lower.startswith("play ") or " play " in text_lower or text_lower == "play":
            target = "music_player"
            action = "play"
            if not query or str(query).lower().strip(" .!?") in _GENERIC_MUSIC_TERMS | {"none", "null"}:
                play_idx = text_lower.find("play ")
                extracted_query = None
                if play_idx != -1:
                    extracted_query = text_input[play_idx + 5:].strip(" .!?")
                # Only treat the extracted text as a real track/artist name
                # if it's NOT one of the generic "just play something"
                # filler phrases (e.g. "music", "some music", "anything").
                if extracted_query and extracted_query.lower() not in _GENERIC_MUSIC_TERMS:
                    action_data["query"] = extracted_query
                else:
                    action_data["query"] = None

        # 1. Normalize Target
        if any(k in target for k in ["living", "living_room", "living_light"]):
            target = "living_room_light"
        elif any(k in target for k in ["kitchen", "kitchen_light"]):
            target = "kitchen_light"
        elif any(k in target for k in ["thermostat", "temp", "temperature"]):
            target = "thermostat"
        elif any(k in target for k in ["humidifier", "humid"]):
            target = "humidifier"
        elif any(k in target for k in ["air_conditioner", "aircon", "air conditioner", "a/c"]) or target == "ac":
            target = "air_conditioner"
        elif "back" in target and any(k in target for k in ["door", "lock"]):
            target = "back_door"
        elif any(k in target for k in ["door", "lock", "front_door"]):
            target = "front_door_lock"
        elif any(k in target for k in ["music", "media", "player", "song", "spotify", "youtube", "kiss me", "speed demon", "i like me better"]) or action in ["play", "pause", "resume", "stop", "next", "previous", "set_volume"]:
            target = "music_player"

        # 2. Normalize Action based on Target
        if target == "front_door_lock":
            if "unlock" in text_lower or any(k in action for k in ["unlock", "open"]):
                action = "unlock"
            elif any(k in action for k in ["lock", "close"]):
                action = "lock"
        elif target == "back_door":
            if "unlock" in text_lower or any(k in action for k in ["unlock", "open"]):
                action = "unlock"
            elif any(k in action for k in ["lock", "close"]):
                action = "lock"
        elif target in ["living_room_light", "kitchen_light", "humidifier", "air_conditioner"]:
            if any(k in action for k in ["on", "turn_on", "enable", "toggle", "start", "run"]):
                action = "turn_on"
            elif any(k in action for k in ["off", "turn_off", "disable", "stop"]):
                action = "turn_off"
        elif target == "thermostat":
            if any(k in action for k in ["set", "change", "adjust", "temp"]):
                action = "set_temp"
            elif "increase" in action or "up" in action or "+" in action:
                action = "increase_temp"
            elif "decrease" in action or "down" in action or "-" in action:
                action = "decrease_temp"
        elif target == "music_player":
            if any(k in action for k in ["stop", "end", "off", "halt"]):
                action = "stop"
            elif any(k in action for k in ["volume", "vol"]):
                action = "set_volume"
            elif any(k in action for k in ["play", "start"]) and action != "set_volume":
                action = "play"
            elif any(k in action for k in ["pause"]):
                action = "pause"
            elif any(k in action for k in ["resume", "unpause"]):
                action = "resume"
            elif any(k in action for k in ["next", "skip"]):
                action = "next"
            elif any(k in action for k in ["previous", "prev", "back"]):
                action = "previous"

        # 3. Handle Volume Value Assignment (0.0 to 1.0)
        if target == "music_player" and action == "set_volume":
            numbers = re.findall(r"\b\d+\b", text_lower)
            if numbers:
                parsed_num = float(numbers[0])
                action_data["value"] = parsed_num / 100.0 if parsed_num > 1.0 else parsed_num
            elif val is None or val == 0.0:
                if any(w in text_lower for w in ["lower", "decrease", "down", "quieter", "soft", "softer"]):
                    action_data["value"] = 0.40
                elif any(w in text_lower for w in ["turn up", "increase", "louder", "raise", "up"]):
                    action_data["value"] = 0.80
                else:
                    action_data["value"] = 0.50

        # 3b. Handle Thermostat Value Assignment (target degrees / step delta).
        # The small local model reliably narrates a confirmation sentence
        # ("set to 19 degrees") but does NOT reliably populate the
        # structured 'value' field the simulator actually reads - so, just
        # like volume above, re-derive it deterministically from the raw
        # text instead of trusting the model's JSON. Without this, the
        # simulator's `val is not None` guard silently no-ops the action:
        # the chat/TTS response (built straight from the model's own
        # spoken_response text) claims success while HomeState never
        # changes and the GUI callback never fires.
        if target == "thermostat":
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text_lower)
            if action == "set_temp":
                if numbers:
                    action_data["value"] = float(numbers[0])
                elif val is None:
                    # No number anywhere and the model gave nothing usable -
                    # don't fabricate a target temperature.
                    action_data["value"] = None
            elif action in ("increase_temp", "decrease_temp"):
                if numbers:
                    action_data["value"] = abs(float(numbers[0]))
                elif val is None:
                    action_data["value"] = 1.0  # sensible default step

        action_data["target"] = target
        action_data["action"] = action
        return action_data

    def parse_intent(self, text_input: str) -> IntentResponse:
        text_lower = text_input.lower().strip()

        # Intercept identity & conversational queries to prevent smart home hallucinations
        identity_phrases = ["state your name", "who are you", "what is your name", "your name", "what can you do", "introduce yourself"]
        if any(p in text_lower for p in identity_phrases):
            return IntentResponse(
                actions=[],
                spoken_response="I am Syntra, your AI home assistant. My purpose is to help you control your smart home devices, manage media playback, adjust climate controls, and make your daily routines easier!"
            )

        # Intercept goodbyes and thank yous
        goodbye_phrases = ["goodbye", "bye", "thank you", "thanks", "see you", "bye bye", "farewell"]
        if any(p in text_lower for p in goodbye_phrases):
            return IntentResponse(
                actions=[],
                spoken_response="Goodbye! Have a wonderful day. Let me know if you need anything else!"
            )

        # Intercept time requests ("what time is it", "what's the time in
        # South Korea", "current time in New York") before weather/alarm,
        # since those also use 'time'-adjacent phrasing.
        time_match = _TIME_RE.search(text_input.strip())
        if time_match:
            location = time_match.group("location") or time_match.group("location2")
            if location:
                location = location.strip(" .!?")
            return IntentResponse(
                actions=[SmartHomeAction(action="get_time", target="time", query=location)],
                spoken_response="",  # filled in by main.py after the actual lookup
            )

        # Intercept weather requests ("what's the weather in Prague", "temperature in Tokyo?")
        if _WEATHER_BARE_RE.search(text_lower):
            match = _WEATHER_RE.search(text_input.strip())
            city = next((g for g in (match.groups() if match else []) if g), None)
            if city:
                city = city.strip(" .!?")
            return IntentResponse(
                actions=[SmartHomeAction(action="get_weather", target="weather", query=city)],
                spoken_response=(
                    f"Let me check the weather in {city}." if city
                    else "Which city would you like the weather for?"
                ),
            )

        # Intercept alarm requests ("set an alarm for 7:30am", "turn off the alarm", "wake me up at 6")
        if _ALARM_RE.search(text_lower):
            # Unahin i-check kung sinasabi nitong i-turn off / stop / cancel ang alarm
            if any(k in text_lower for k in ["off", "stop", "cancel", "disable", "turn off", "turn_off", "shut up"]):
                return IntentResponse(
                    actions=[SmartHomeAction(action="turn_off", target="alarm")],
                    spoken_response="Alarm has been turned off."
                )

            # Kung hindi naman turn off, saka pa lang ipa-parse ang oras para sa set_alarm
            alarm_time = _parse_clock_time(text_input)
            return IntentResponse(
                actions=[SmartHomeAction(action="set_alarm", target="alarm", query=alarm_time)]
                if alarm_time else [],
                spoken_response=(
                    f"Alarm set for {alarm_time}." if alarm_time
                    else "What time would you like the alarm set for?"
                ),
            )

        system_prompt = (
            "You are Syntra, an AI home assistant. Parse natural language user "
            "commands into structured JSON actions.\n\n"
            "EXACT ALLOWED TARGETS: 'living_room_light', 'kitchen_light', 'thermostat', 'front_door_lock', 'back_door', 'humidifier', 'air_conditioner', 'music_player', 'weather', 'alarm', 'time'\n"
            "EXACT ALLOWED ACTIONS:\n"
            "- Lights: 'turn_on', 'turn_off'\n"
            "- Thermostat: 'set_temp', 'increase_temp', 'decrease_temp'\n"
            "- Front Door: 'lock', 'unlock'\n"
            "- Back Door: 'lock', 'unlock'\n"
            "- Humidifier: 'turn_on', 'turn_off'\n"
            "- Air Conditioner: 'turn_on', 'turn_off'\n"
            "- Music Player: 'play', 'pause', 'resume', 'stop', 'next', 'previous', 'play music', 'set_volume'\n"
            "- Weather: 'get_weather' (query = city name)\n"
            "- Alarm: 'set_alarm' (query = 'HH:MM' 24-hour time)\n"
            "- Time: 'get_time' (query = city/country name, or empty for local time)\n\n"
            "STRICT RULES:\n"
            "1. 'unlock front door' MUST map to target 'front_door_lock' and action 'unlock'. NEVER set action to 'lock' when unlocking.\n"
            "2. Commands like 'stop', 'stop music', or 'stop playing' MUST map to action 'stop' and target 'music_player'.\n"
            "3. Commands like 'next music', 'next song', or 'next' MUST map to action 'next' and target 'music_player' (playing random/chronological songs locally). Do NOT put 'next music' in query.\n"
            "4. Commands like 'lower the volume', 'turn up volume', or 'lower the sound' MUST map to action 'set_volume' and target 'music_player'. Value MUST be between 0.0 and 1.0 (e.g. 0.4 for lower, 0.8 for turn up). NEVER map volume to lights, thermostat, or house temperature.\n"
            "5. Commands like 'play <song name>' (e.g., 'play kiss me') MUST map target to 'music_player', action to 'play', and query to '<song name>'.\n\n"
            "Return ONLY valid JSON matching this exact structure, without markdown backticks or commentary:\n"
            "{\n"
            '  "actions": [\n'
            '    {"action": "unlock", "target": "front_door_lock", "value": null, "query": null, "source": "local"}\n'
            "  ],\n"
            '  "spoken_response": "The front door has been unlocked."\n'
            "}"
        )

        try:
            logger.info("Sending prompt to local LLM (%s): '%s'", self.model_name, text_input)
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                format=IntentResponse.model_json_schema(),
                # --- Latency optimizations ---
                # keep_alive: keeps the model resident in memory between
                # calls so we never pay the multi-second cold-load cost on
                # every command (the biggest source of perceived lag).
                keep_alive="30m",
                options={
                    # Deterministic + short: intent parsing needs one
                    # small JSON object, not creative long-form text.
                    "temperature": 0.0,
                    "num_predict": 200,
                    # Smaller context window than Ollama's default (usually
                    # 2048-4096) speeds up prompt processing for these
                    # short, single-turn commands.
                    "num_ctx": 1024,
                },
            )

            raw_content = response["message"]["content"]
            logger.info("Raw LLM Response: %s", raw_content)

            parsed_json = json.loads(raw_content)

            # Post-process and normalize every action to prevent model hallucinations
            if "actions" in parsed_json and isinstance(parsed_json["actions"], list):
                parsed_json["actions"] = [
                    self._normalize_action(act, text_input) for act in parsed_json["actions"] if isinstance(act, dict)
                ]

            # Re-align spoken response for door, music stop, and volume commands
            for act in parsed_json.get("actions", []):
                target = act.get("target")
                action = act.get("action")
                val = act.get("value")

                if target == "front_door_lock":
                    spoken = parsed_json.get("spoken_response", "").lower()
                    if action == "unlock" and "unlocked" not in spoken:
                        parsed_json["spoken_response"] = "The front door has been unlocked."
                    elif action == "lock" and "locked" in spoken and "unlocked" in spoken:
                        parsed_json["spoken_response"] = "The front door has been locked."
                elif target == "music_player":
                    if action == "stop":
                        parsed_json["spoken_response"] = "Stopped the music."
                    elif action == "set_volume" and val is not None:
                        pct = int(round(val * 100))
                        parsed_json["spoken_response"] = f"Volume set to {pct} percent."

            validated_intent = IntentResponse(**parsed_json)
            return validated_intent

        except Exception:
            logger.exception("Failed to parse LLM response for input: '%s'", text_input)
            return IntentResponse(
                actions=[],
                spoken_response=(
                    "I'm sorry, I had trouble understanding that command. "
                    "Could you please rephrase it?"
                ),
            )