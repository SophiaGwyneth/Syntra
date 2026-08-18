"""
geocoding.py
------------
Free, worldwide, no-API-key geocoding for Syntra.

Used as the last-resort fallback by both:
  - time_utils.py    (resolve any place on Earth -> lat/lon -> timezone)
  - weather_alarm.py (resolve any place OpenWeatherMap can't match directly
                       -> lat/lon -> weather at that point)

Backed by OpenStreetMap's Nominatim search API (https://nominatim.org).
Nominatim is free and keyless, but its usage policy requires:
  - a descriptive User-Agent identifying the application, and
  - no more than ~1 request/second.
Both are handled here: a fixed User-Agent is sent with every request, and
a simple in-memory cache means a place that's already been resolved this
run never triggers a second network call.

Public API:
    geocode(location)        -> (lat, lon, display_name) | None
    short_label(display_name) -> str
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger("VirtualAssistant")

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "Syntra-HomeAssistant/1.0 (contact: local-only project; no email on file)"
_REQUEST_TIMEOUT = 6  # seconds
_MIN_REQUEST_INTERVAL = 1.0  # seconds - Nominatim's usage-policy rate limit

# Simple in-memory cache: normalized query -> resolved (lat, lon, display_name)
# or None (a confirmed miss, so repeated typos/bad queries don't re-hit the
# network every time). Cleared automatically on process restart.
_cache: dict[str, Optional[Tuple[float, float, str]]] = {}
_cache_lock = threading.Lock()

# Throttling state shared across calls so we never exceed ~1 req/sec,
# regardless of which module (time or weather) is calling in.
_last_request_time = 0.0
_throttle_lock = threading.Lock()


def _throttle():
    """Blocks just long enough to keep requests at or under 1/second."""
    global _last_request_time
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_time
        wait = _MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _normalize(location: str) -> str:
    return " ".join(location.strip().lower().split())


def geocode(location: str) -> Optional[Tuple[float, float, str]]:
    """
    Resolves any free-form place name - city, province/state, country,
    or landmark - to coordinates.

    Returns:
        (latitude, longitude, display_name) on success, where display_name
        is Nominatim's full human-readable place name (e.g.
        "Bacoor, Cavite, Calabarzon, Philippines"), or None if the place
        can't be found or the lookup fails for any reason (network error,
        timeout, empty input, etc.). Never raises - callers treat None as
        "couldn't resolve this place" and turn it into a spoken message.
    """
    if not location or not location.strip():
        return None

    query = _normalize(location)

    with _cache_lock:
        if query in _cache:
            logger.info("Geocode cache hit for '%s'", query)
            return _cache[query]

    result: Optional[Tuple[float, float, str]] = None
    try:
        _throttle()
        response = requests.get(
            _NOMINATIM_URL,
            params={
                "q": location.strip(),
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 0,
            },
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()

        if results:
            top = results[0]
            lat = float(top["lat"])
            lon = float(top["lon"])
            display_name = top.get("display_name", location.strip())
            result = (lat, lon, display_name)
            logger.info("Geocoded '%s' -> (%.4f, %.4f) '%s'", location, lat, lon, display_name)
        else:
            logger.warning("Geocoding found no results for '%s'", location)

    except requests.exceptions.RequestException:
        logger.exception("Geocoding request failed for '%s'", location)
    except (KeyError, ValueError, TypeError):
        logger.exception("Geocoding returned unexpected data for '%s'", location)

    with _cache_lock:
        _cache[query] = result

    return result


def short_label(display_name: str) -> str:
    """
    Trims Nominatim's full comma-separated display_name down to something
    short and spoken-friendly.

    e.g. "Bacoor, Cavite, Calabarzon, 4102, Philippines"
         -> "Bacoor, Philippines"
         "Mount Everest, Solukhumbu, Koshi Province, Nepal"
         -> "Mount Everest, Nepal"

    Falls back to the original string if it doesn't look like a standard
    comma-separated Nominatim result.
    """
    if not display_name:
        return display_name

    parts = [p.strip() for p in display_name.split(",") if p.strip()]
    if len(parts) <= 2:
        return display_name.strip()

    # First segment is the place itself; last segment is typically the
    # country. Middle segments (postal codes, provinces, districts) add
    # little value to a spoken response, so they're dropped.
    place = parts[0]
    country = parts[-1]

    # Postal codes sometimes end up last (rare, malformed responses) -
    # if the "country" segment is mostly digits, prefer the segment before it.
    if country.replace(" ", "").isdigit() and len(parts) >= 3:
        country = parts[-2]

    if place == country:
        return place
    return f"{place}, {country}"