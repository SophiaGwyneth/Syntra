"""
time_utils.py
-------------
Real-time local & world time query handling for Syntra.

  - get_local_time()         -> a spoken-friendly string for "what time is
                                 it?" using the machine's local system clock.
  - get_time_in_location(x)  -> a spoken-friendly string for "what time is
                                 it in <city/country>?" using pytz. Cities
                                 and countries are resolved to an IANA
                                 timezone name via a curated alias table
                                 (fast, no network/API call needed) with a
                                 fallback fuzzy search across pytz's full
                                 timezone list.

Raises TimeQueryError (never a bare Exception) so callers can catch one
specific, well-known error type and turn it into a spoken response.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime
from typing import Optional

import pytz
from timezonefinder import TimezoneFinder

from geocoding import geocode, short_label

logger = logging.getLogger("VirtualAssistant")

# Built once at import time - TimezoneFinder loads its boundary data into
# memory, so a single shared instance is reused for every lookup rather
# than rebuilding it per call.
_tf = TimezoneFinder()


class TimeQueryError(Exception):
    """Raised when a location can't be resolved to a known timezone."""


# Common city/country/abbreviation aliases -> IANA timezone name. Covers
# the phrasing people actually use in voice/text commands; anything not
# listed here falls back to a fuzzy search over pytz.all_timezones.
_TIMEZONE_ALIASES = {
    # Countries / regions
    "south korea": "Asia/Seoul", "korea": "Asia/Seoul", "north korea": "Asia/Pyongyang",
    "japan": "Asia/Tokyo", "china": "Asia/Shanghai", "india": "Asia/Kolkata",
    "philippines": "Asia/Manila", "vietnam": "Asia/Ho_Chi_Minh", "thailand": "Asia/Bangkok",
    "singapore": "Asia/Singapore", "indonesia": "Asia/Jakarta", "malaysia": "Asia/Kuala_Lumpur",
    "taiwan": "Asia/Taipei", "hong kong": "Asia/Hong_Kong",
    "uk": "Europe/London", "united kingdom": "Europe/London", "england": "Europe/London",
    "britain": "Europe/London", "great britain": "Europe/London",
    "france": "Europe/Paris", "germany": "Europe/Berlin", "italy": "Europe/Rome",
    "spain": "Europe/Madrid", "portugal": "Europe/Lisbon", "netherlands": "Europe/Amsterdam",
    "russia": "Europe/Moscow", "greece": "Europe/Athens", "poland": "Europe/Warsaw",
    "sweden": "Europe/Stockholm", "switzerland": "Europe/Zurich", "ireland": "Europe/Dublin",
    "usa": "America/New_York", "united states": "America/New_York", "america": "America/New_York",
    "us": "America/New_York",
    "canada": "America/Toronto", "mexico": "America/Mexico_City", "brazil": "America/Sao_Paulo",
    "argentina": "America/Argentina/Buenos_Aires",
    "australia": "Australia/Sydney", "new zealand": "Pacific/Auckland",
    "uae": "Asia/Dubai", "united arab emirates": "Asia/Dubai", "dubai": "Asia/Dubai",
    "saudi arabia": "Asia/Riyadh", "israel": "Asia/Jerusalem", "turkey": "Europe/Istanbul",
    "egypt": "Africa/Cairo", "south africa": "Africa/Johannesburg", "nigeria": "Africa/Lagos",
    "pakistan": "Asia/Karachi", "bangladesh": "Asia/Dhaka",

    # Cities (only the ones with a country name that doesn't map 1:1 to a
    # single timezone, or that people commonly ask about by city alone)
    "new york": "America/New_York", "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "chicago": "America/Chicago", "houston": "America/Chicago",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "denver": "America/Denver", "phoenix": "America/Phoenix",
    "toronto": "America/Toronto", "vancouver": "America/Vancouver",
    "london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
    "rome": "Europe/Rome", "madrid": "Europe/Madrid", "amsterdam": "Europe/Amsterdam",
    "moscow": "Europe/Moscow", "istanbul": "Europe/Istanbul", "athens": "Europe/Athens",
    "tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul", "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai", "manila": "Asia/Manila", "bangkok": "Asia/Bangkok",
    "jakarta": "Asia/Jakarta", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "hanoi": "Asia/Ho_Chi_Minh", "kuala lumpur": "Asia/Kuala_Lumpur",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "auckland": "Pacific/Auckland", "cairo": "Africa/Cairo", "lagos": "Africa/Lagos",
    "johannesburg": "Africa/Johannesburg", "nairobi": "Africa/Nairobi",
    "sao paulo": "America/Sao_Paulo", "buenos aires": "America/Argentina/Buenos_Aires",
    "mexico city": "America/Mexico_City", "dubai": "Asia/Dubai",
}

# Philippine provinces, cities, and municipalities -> all map to the single
# Philippine timezone. This exists as its own table (rather than folding it
# into _TIMEZONE_ALIASES) because it's a flat one-timezone-fits-all list,
# and it's the category of query most likely to be asked locally and most
# likely to be missing from a generic aliases table.
_PH_LOCATIONS = {
    # Regions / general
    "philippines", "pilipinas", "ph",
    # Metro Manila cities
    "manila", "quezon city", "qc", "makati", "taguig", "bgc", "pasig",
    "mandaluyong", "san juan", "marikina", "paranaque", "las pinas",
    "muntinlupa", "pasay", "caloocan", "malabon", "navotas", "valenzuela",
    "pateros",
    # Cavite
    "cavite", "bacoor", "imus", "dasmarinas", "dasmariñas", "general trias",
    "tanza", "trece martires", "kawit", "noveleta", "rosario", "tagaytay",
    "silang", "carmona", "gma", "general mariano alvarez",
    # Laguna
    "laguna", "santa rosa", "calamba", "san pedro", "binan", "biñan",
    "los banos", "los baños", "cabuyao",
    # Bulacan
    "bulacan", "malolos", "meycauayan", "san jose del monte", "marilao",
    "santa maria", "baliuag", "baliwag",
    # Rizal
    "rizal", "antipolo", "cainta", "taytay",
    # Batangas / Cebu / Davao and other major cities
    "batangas", "batangas city", "lipa",
    "cebu", "cebu city", "mandaue", "lapu-lapu",
    "davao", "davao city",
    "iloilo", "iloilo city", "bacolod", "cagayan de oro", "zamboanga",
    "baguio", "angeles", "angeles city", "pampanga", "clark",
    "puerto princesa", "palawan", "tacloban", "general santos",
    "butuan", "dumaguete", "naga", "legazpi", "olongapo", "subic",
}


def _looks_like_ph_query(query: str) -> bool:
    """True if `query` matches a known PH location or a fuzzy variant of one."""
    if query in _PH_LOCATIONS:
        return True
    matches = difflib.get_close_matches(query, _PH_LOCATIONS, n=1, cutoff=0.8)
    return bool(matches)


def get_local_time() -> str:
    """Returns a spoken-friendly local time string, e.g.
    'It's 3:45 PM on Sunday, August 16.'"""
    now = datetime.now()
    return f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B %d')}."


def _normalize_location(location: str) -> str:
    """Strips, lowercases, and collapses whitespace/punctuation for matching."""
    query = location.strip().lower().strip(" .!?")
    # Collapse repeated whitespace so STT artifacts like "new  york" still match.
    query = " ".join(query.split())
    return query


def _find_timezone(location: str) -> Optional[str]:
    """Resolves a free-form city/country string to an IANA timezone name."""
    query = _normalize_location(location)
    if not query:
        return None

    if query in _TIMEZONE_ALIASES:
        return _TIMEZONE_ALIASES[query]

    # Philippine cities/provinces/municipalities all share one timezone.
    if _looks_like_ph_query(query):
        return "Asia/Manila"

    # Try a direct substring match against pytz's full timezone list
    # (e.g. "new_york" inside "America/New_York").
    normalized = query.replace(" ", "_")
    for tz in pytz.all_timezones:
        if normalized in tz.lower():
            return tz

    # Fuzzy match against both our alias keys and pytz's timezone segments,
    # to tolerate minor typos/mishearings from STT.
    alias_matches = difflib.get_close_matches(query, _TIMEZONE_ALIASES.keys(), n=1, cutoff=0.75)
    if alias_matches:
        return _TIMEZONE_ALIASES[alias_matches[0]]

    tz_leaf_names = {tz.split("/")[-1].replace("_", " ").lower(): tz for tz in pytz.all_timezones}
    leaf_matches = difflib.get_close_matches(query, tz_leaf_names.keys(), n=1, cutoff=0.75)
    if leaf_matches:
        return tz_leaf_names[leaf_matches[0]]

    return None


def _geocode_timezone(location: str) -> Optional[tuple[str, str]]:
    """
    Last-resort global fallback for anything the curated alias tables and
    pytz's own name list don't cover: states/provinces, countries, and
    landmarks (e.g. "Texas", "Cavite", "Mount Everest"). Geocodes the
    location to coordinates, then resolves those coordinates to an exact
    IANA timezone via timezonefinder - this works for any point on Earth,
    since it's boundary-based rather than a name lookup.

    Returns (tz_name, display_label) or None if either step fails.
    """
    geo = geocode(location)
    if geo is None:
        return None

    lat, lon, display_name = geo
    tz_name = _tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        logger.warning("timezonefinder found no timezone for (%.4f, %.4f)", lat, lon)
        return None

    return tz_name, short_label(display_name)


def get_time_in_location(location: str) -> str:
    """Returns a spoken-friendly time string for `location`, e.g.
    'It's 4:45 AM on Monday, August 17 in South Korea.'

    `location` can be a city, a state/province, a country, or a landmark.
    Common places resolve instantly via the curated alias table; anything
    else falls back to global geocoding + coordinate-based timezone lookup
    (see _geocode_timezone), giving effectively worldwide coverage.

    Raises:
        TimeQueryError: if `location` can't be resolved to a known timezone
            (empty input, or the place genuinely can't be found).
    """
    if not location or not location.strip():
        raise TimeQueryError("Which city or country would you like the time for?")

    display_label = location.strip()
    tz_name = _find_timezone(location)

    if not tz_name:
        # Not in the alias table or pytz's own names - try resolving it
        # from scratch via geocoding + coordinates.
        fallback = _geocode_timezone(location)
        if fallback is None:
            raise TimeQueryError(f"I don't know the timezone for '{location.strip()}'.")
        tz_name, display_label = fallback

    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        logger.exception("Failed to compute time for resolved timezone '%s'", tz_name)
        raise TimeQueryError(f"Something went wrong looking up the time in '{location.strip()}'.")

    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%A, %B %d")
    logger.info("Resolved '%s' -> timezone '%s' -> %s", location, tz_name, time_str)
    return f"It's {time_str} on {date_str} in {display_label}."