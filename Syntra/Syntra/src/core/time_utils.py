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

logger = logging.getLogger("VirtualAssistant")


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


def get_local_time() -> str:
    """Returns a spoken-friendly local time string, e.g.
    'It's 3:45 PM on Sunday, August 16.'"""
    now = datetime.now()
    return f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B %d')}."


def _find_timezone(location: str) -> Optional[str]:
    """Resolves a free-form city/country string to an IANA timezone name."""
    query = location.strip().lower().strip(" .!?")
    if not query:
        return None

    if query in _TIMEZONE_ALIASES:
        return _TIMEZONE_ALIASES[query]

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


def get_time_in_location(location: str) -> str:
    """Returns a spoken-friendly time string for `location`, e.g.
    'It's 4:45 AM on Monday, August 17 in South Korea.'

    Raises:
        TimeQueryError: if `location` can't be resolved to a known timezone.
    """
    if not location or not location.strip():
        raise TimeQueryError("Which city or country would you like the time for?")

    tz_name = _find_timezone(location)
    if not tz_name:
        raise TimeQueryError(f"I don't know the timezone for '{location.strip()}'.")

    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        logger.exception("Failed to compute time for resolved timezone '%s'", tz_name)
        raise TimeQueryError(f"Something went wrong looking up the time in '{location.strip()}'.")

    time_str = now.strftime("%I:%M %p").lstrip("0")
    date_str = now.strftime("%A, %B %d")
    logger.info("Resolved '%s' -> timezone '%s' -> %s", location, tz_name, time_str)
    return f"It's {time_str} on {date_str} in {location.strip()}."