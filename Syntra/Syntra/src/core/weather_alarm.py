"""
weather_alarm.py
-----------------
Two independent features for Syntra, the AI Home Assistant:

  - get_weather(location)   -> live temperature + conditions for any city,
                                via the free Open-Meteo API (no API key
                                required). Geocoding + forecast are two
                                separate, chained HTTP calls.

  - AlarmClock / set_alarm  -> schedule an alarm for "HH:MM" that plays a
                                custom audio file through pygame's mixer,
                                on its own daemon thread so it never blocks
                                the GUI/event loop (matches the non-blocking
                                design of src/core/music_player.py).

Both are written to slot straight into the existing project layout as
src/core/weather_alarm.py: same logger name, same style of "never raise
out of a background thread" error handling, same Optional[Callable]
on_change/on_trigger hook pattern as MusicPlayer.

Dependencies (add to requirements.txt):
    requests>=2.31.0
    pygame>=2.5.2      # already a dependency of this project

No API key is needed for Open-Meteo. If you'd rather use OpenWeatherMap,
see the WEATHER_PROVIDER note near get_weather().
"""

from __future__ import annotations

import logging
import os
import datetime
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger("VirtualAssistant")

# --------------------------------------------------------------------------- #
# Weather (OpenWeatherMap Implementation)
# --------------------------------------------------------------------------- #

OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT = 8  # seconds

# Your active OpenWeatherMap API key integrated below:
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


class WeatherError(Exception):
    """Raised for invalid locations or unrecoverable API failures."""


@dataclass
class WeatherReport:
    location: str
    country: Optional[str]
    temperature_c: float
    feels_like_c: Optional[float]
    condition: str
    wind_kph: Optional[float]
    humidity_pct: Optional[int]

    def spoken_summary(self) -> str:
        """A short sentence suitable for TTS / chat."""
        place = f"{self.location}, {self.country}" if self.country else self.location
        return (
            f"It's currently {self.temperature_c:.0f}°C and {self.condition.lower()} "
            f"in {place}."
        )


def get_weather(location: str, api_key: Optional[str] = None) -> WeatherReport:
    """Fetches real-time temperature and conditions for `location` using OpenWeatherMap.

    Raises:
        WeatherError: on missing API key, empty input, unknown city, or network errors.
    """
    key = api_key or OPENWEATHER_API_KEY
    if not key or key == "YOUR_OPENWEATHERMAP_API_KEY":
        raise WeatherError("OpenWeatherMap API key is missing or invalid.")

    if not location or not location.strip():
        raise WeatherError("Please tell me which city you'd like the weather for.")

    query = location.strip()

    params = {
        "q": query,
        "appid": key,
        "units": "metric",  # Fetch temperature directly in Celsius
    }

    try:
        response = requests.get(OWM_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 404:
            raise WeatherError(f"I couldn't find a place called '{query}'.")
        elif response.status_code == 401:
            raise WeatherError("Invalid OpenWeatherMap API key provided.")

        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.exception("Weather request failed for '%s'", query)
        raise WeatherError(
            f"I couldn't reach the weather service to look up '{query}'."
        ) from exc

    data = response.json()

    # Parse response data
    resolved_name = data.get("name", query)
    country = data.get("sys", {}).get("country")

    main = data.get("main", {})
    temp_c = main.get("temp")
    feels_like_c = main.get("feels_like")
    humidity_pct = main.get("humidity")

    # Wind speed comes in m/s from OpenWeatherMap -> multiply by 3.6 to get km/h
    wind_mps = data.get("wind", {}).get("speed")
    wind_kph = (wind_mps * 3.6) if wind_mps is not None else None

    # Weather condition description (e.g., "clear sky", "light rain")
    weather_list = data.get("weather", [])
    condition = weather_list[0]["description"].capitalize() if weather_list else "Unknown conditions"

    report = WeatherReport(
        location=resolved_name,
        country=country,
        temperature_c=temp_c,
        feels_like_c=feels_like_c,
        condition=condition,
        wind_kph=wind_kph,
        humidity_pct=humidity_pct,
    )
    logger.info("Weather fetched for %s: %.1f°C, %s", resolved_name, report.temperature_c, condition)
    return report


# NOTE on OpenWeatherMap as an alternative provider:
#   If you'd rather use OpenWeatherMap (requires a free API key), swap the
#   two requests.get() calls above for a single call to:
#       https://api.openweathermap.org/data/2.5/weather
#           ?q={location}&appid={API_KEY}&units=metric
#   and read resp.json()["main"]["temp"] / resp.json()["weather"][0]["description"].
#   Store the key in .env as SYNTRA_OWM_API_KEY and load it with
#   os.getenv("SYNTRA_OWM_API_KEY") the same way main.py loads SYNTRA_MODEL_NAME.


# --------------------------------------------------------------------------- #
# Alarm clock
# --------------------------------------------------------------------------- #

class AlarmError(Exception):
    """Raised for invalid alarm times or missing/unplayable sound files."""


def _resolve_sound_path(sound_path: str, search_root: Optional[str] = None) -> Optional[Path]:
    """
    Flexibly resolves an alarm sound file even if it's been moved to a
    different folder or subfolder.

    Resolution order:
      1. The path exactly as given (absolute or relative to cwd).
      2. The path relative to `search_root` (defaults to this file's
         project root, i.e. the folder main.py lives in).
      3. A recursive filename search under `search_root` - if a file with
         the same name exists ANYWHERE under the project tree, use it.

    Returns a Path if found, or None if the file genuinely can't be
    located anywhere under the search root.
    """
    candidate = Path(sound_path)

    # 1. Exact path (absolute, or relative to current working directory).
    if candidate.is_file():
        return candidate

    if search_root is None:
        # This file lives at <project_root>/src/core/weather_alarm.py,
        # so walk up two levels to reach the project root.
        search_root = Path(__file__).resolve().parent.parent.parent
    else:
        search_root = Path(search_root)

    # 2. Path relative to the project root.
    relative_candidate = search_root / sound_path
    if relative_candidate.is_file():
        return relative_candidate

    # 3. Recursive filename search - handles the file having been moved
    #    to any subfolder (e.g. music/, alarm/, or a nested folder).
    target_name = Path(sound_path).name
    if target_name and search_root.exists():
        try:
            matches = list(search_root.rglob(target_name))
        except Exception:
            logger.exception("Recursive search for alarm sound '%s' failed", target_name)
            matches = []
        if matches:
            if len(matches) > 1:
                logger.info(
                    "Multiple files named '%s' found under '%s'; using the first match: %s",
                    target_name, search_root, matches[0],
                )
            return matches[0]

    return None


class AlarmClock:
    def __init__(self, on_trigger: Optional[Callable[[str], None]] = None, search_root: Optional[str] = None):
        self.on_trigger = on_trigger
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._mixer_ready = False
        # Root directory used as the fallback search base when the exact
        # alarm sound path can't be found (defaults to the project root).
        self.search_root = search_root

    def _ensure_mixer(self) -> None:
        if self._mixer_ready:
            return
        import pygame
        pygame.mixer.init()
        self._mixer_ready = True

    @staticmethod
    def _parse_time_string(alarm_time: str) -> tuple[int, int]:
        """Tumatanggap ng 12-hour (AM/PM) at 24-hour time formats."""
        clean_str = alarm_time.strip().upper()
        formats = [
            "%H:%M",       # 14:29
            "%I:%M %p",    # 02:29 PM
            "%I:%M%p",     # 02:29PM
            "%I %p",       # 2 PM
            "%H:%M:%S",    # 14:29:00
        ]
        
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(clean_str, fmt)
                return dt.hour, dt.minute
            except ValueError:
                pass

        raise AlarmError(
            f"'{alarm_time}' isn't a valid time. Please use formats like '14:29' or '2:29 PM'."
        )

    def set_alarm(self, alarm_time: str, sound_path: str) -> None:
        target_hour, target_minute = self._parse_time_string(alarm_time)

        sound = _resolve_sound_path(sound_path, search_root=self.search_root)
        if sound is None:
            raise AlarmError(
                f"Alarm sound file not found: {sound_path} "
                f"(searched the project folder and its subfolders)."
            )
        if sound.suffix.lower() not in {".mp3", ".wav", ".ogg"}:
            raise AlarmError(f"Unsupported audio format: {sound.suffix}")
        logger.info("Resolved alarm sound '%s' -> '%s'", sound_path, sound)

        self.cancel()
        self.stop_playback()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            args=(target_hour, target_minute, sound),
            daemon=True,
            name=f"AlarmClock-{target_hour:02d}:{target_minute:02d}",
        )
        self._thread.start()
        logger.info("Alarm set for %02d:%02d using '%s'", target_hour, target_minute, sound.name)

    def _seconds_until(self, hour: int, minute: int) -> float:
        now = time.localtime()
        target = time.struct_time(
            (now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1)
        )
        delta = time.mktime(target) - time.mktime(now)
        if delta <= -30:
            delta += 24 * 60 * 60
        return delta

    def _run(self, hour: int, minute: int, sound: Path) -> None:
        try:
            wait_seconds = self._seconds_until(hour, minute)
            logger.info("Alarm thread sleeping for %.0fs until %02d:%02d", wait_seconds, hour, minute)

            while wait_seconds > 0 and not self._stop_event.is_set():
                chunk = min(1.0, wait_seconds)
                time.sleep(chunk)
                wait_seconds -= chunk

            if self._stop_event.is_set():
                logger.info("Alarm cancelled before it triggered.")
                return

            self._play(sound)

        except Exception:
            logger.exception("Unhandled error in alarm thread")

    def _play(self, sound: Path) -> None:
        try:
            self._ensure_mixer()
            import pygame
            pygame.mixer.music.load(str(sound))
            pygame.mixer.music.play()
            logger.info("Alarm triggered: playing '%s'", sound.name)
            if self.on_trigger:
                self.on_trigger(str(sound))
        except Exception:
            logger.exception("Failed to play alarm sound '%s'", sound)

    def cancel(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("Alarm cancelled.")

    def stop_alarm(self) -> None:
        """Alias method na tinatawag ng main.py para ihinto ang alarm."""
        self.cancel()
        self.stop_playback()

    def stop_playback(self) -> None:
        if self._mixer_ready:
            import pygame
            pygame.mixer.music.stop()


def set_alarm(alarm_time: str, sound_path: str) -> AlarmClock:
    alarm = AlarmClock()
    alarm.set_alarm(alarm_time, sound_path)
    return alarm


# --------------------------------------------------------------------------- #
# Runnable example
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Weather demo
    for city in ["Prague", "Tokyo", "Manila"]:
        try:
            report = get_weather(city)
            print(report.spoken_summary())
        except WeatherError as err:
            print(f"[Weather error] {err}")

    # Alarm demo
    demo_time = time.strftime("%H:%M", time.localtime(time.time() + 65))
    sound_file = "denielcz-czechoslovakia-eas-alarm-1993-alt-new-earth-eas-alarm-youtube-463088.mp3"

    try:
        alarm = set_alarm(demo_time, sound_file)
        print(f"Alarm scheduled for {demo_time}. Waiting for it to trigger...")
        time.sleep(75)
        alarm.stop_playback()
    except AlarmError as err:
        print(f"[Alarm error] {err}")