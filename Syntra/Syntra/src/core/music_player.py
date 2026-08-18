"""
music_player.py
----------------
Playback engine for Syntra, with THREE selectable backends:

  - "local"   (default): real local-audio playback via pygame's mixer,
              reading .mp3/.wav/.ogg files from the `music/` folder.
              100% offline - this is what the project's core "on-premise,
              privacy-focused" pitch is built on, and it's what you get if
              you never say a service name out loud.
  - "spotify" (optional): streams via the Spotify Web API through spotipy.
              Requires a Spotify Developer app + Premium account + the
              Spotify desktop app already open (Spotify Connect needs an
              active device to hand playback to). NOT offline.
  - "youtube" (optional): searches YouTube via yt-dlp and streams the audio
              through python-vlc if it's installed; otherwise falls back to
              opening the result in your default browser. NOT offline.

Design note: adding streaming services is a genuine trade-off against this
project's stated "operates entirely offline" business requirement - keep
that in mind for your architecture write-up. Both extra backends are
strictly opt-in (only triggered when the user names the service) and fail
soft with a spoken explanation rather than crashing Syntra, so the graded,
offline local-playback path is never put at risk by missing credentials or
packages.

Only one source can be "active" (playing/paused) at a time; pause/resume/
stop/next/previous/set_volume are routed to whichever backend is currently
active. Track-finished detection for the local backend still runs on a
small watcher thread; the GUI is notified the same way regardless of which
backend produced the change.
"""

import difflib
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("VirtualAssistant")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}


class MusicPlayer:
    def __init__(self, music_dir: Optional[str] = None, on_change: Optional[Callable[[dict], None]] = None):
        # Dynamically locate the project root relative to this file (src/core/music_player.py)
        if music_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.music_dir = project_root / "music"
        else:
            self.music_dir = Path(music_dir)
            
        self.on_change = on_change

        # ---- local backend state ----
        self._mixer_ready = False
        self._library: List[Path] = []
        self._current_index: Optional[int] = None
        self._watch_thread = None
        self._watch_stop = threading.Event()

        # ---- shared / cross-backend state ----
        self._active_source = "local"  # "local" | "spotify" | "youtube"
        self._is_playing = False
        self._is_paused = False
        self._volume = 0.7
        self._current_title: Optional[str] = None

        # ---- spotify backend state (lazy) ----
        self._spotify_client = None

        # ---- youtube backend state (lazy) ----
        self._vlc_instance = None
        self._vlc_player = None

        self._init_mixer()
        self.refresh_library()

    # ------------------------------------------------------------------ #
    def _init_mixer(self):
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ready = True
            logger.info("Audio mixer initialized.")
        except Exception:
            logger.exception("Could not initialize audio mixer - local music playback will be disabled.")
            self._mixer_ready = False

    def refresh_library(self):
        """Rescans the music/ folder. Safe to call any time."""
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self._library = sorted(
            p for p in self.music_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("Music library loaded: %d track(s) in '%s'", len(self._library), self.music_dir)

    def _notify(self):
        if self.on_change:
            try:
                self.on_change(self.get_now_playing())
            except Exception:
                logger.exception("music on_change callback raised an exception")

    def _reset_playback_state(self, source: str, title: Optional[str], playing: bool):
        self._active_source = source
        self._current_title = title
        self._is_playing = playing
        self._is_paused = False

    # ================================================================== #
    # Public transport API - routes to whichever backend is active
    # (or, for `play`, to whichever backend the caller asks for)
    # ================================================================== #
    def play(self, query: Optional[str] = None, source: str = "local") -> str:
        source = (source or "local").lower()
        self._stop_watcher()
        if source == "spotify":
            return self._play_spotify(query)
        if source == "youtube":
            return self._play_youtube(query)
        return self._play_local(query)

    def pause(self) -> str:
        if self._active_source == "spotify":
            return self._spotify_call(lambda sp: sp.pause_playback(), "Paused Spotify.")
        if self._active_source == "youtube":
            return self._vlc_call(lambda p: p.pause(), "Paused.")
        return self._pause_local()

    def resume(self) -> str:
        if self._active_source == "spotify":
            return self._spotify_call(lambda sp: sp.start_playback(), "Resuming Spotify.")
        if self._active_source == "youtube":
            return self._vlc_call(lambda p: p.play(), "Resuming playback.")
        return self._resume_local()

    def stop(self) -> str:
        if self._active_source == "spotify":
            result = self._spotify_call(lambda sp: sp.pause_playback(), "Stopped Spotify.")
        elif self._active_source == "youtube":
            result = self._vlc_call(lambda p: p.stop(), "Stopped the music.")
        else:
            result = self._stop_local()
        self._is_playing = False
        self._is_paused = False
        self._notify()
        return result

    def next(self) -> str:
        if self._active_source == "spotify":
            return self._spotify_call(lambda sp: sp.next_track(), "Skipping to the next Spotify track.")
        if self._active_source == "youtube":
            return "Say the name of the next song to play it from YouTube."
        return self._next_local()

    def previous(self) -> str:
        if self._active_source == "spotify":
            return self._spotify_call(lambda sp: sp.previous_track(), "Going back a Spotify track.")
        if self._active_source == "youtube":
            return "Say the name of the song to play it again from YouTube."
        return self._previous_local()

    def set_volume(self, value: float) -> str:
        self._volume = max(0.0, min(1.0, value))
        if self._active_source == "local" and self._mixer_ready:
            try:
                import pygame
                pygame.mixer.music.set_volume(self._volume)
            except Exception:
                logger.exception("Failed to set local volume")
        elif self._active_source == "youtube" and self._vlc_player:
            try:
                self._vlc_player.audio_set_volume(int(self._volume * 100))
            except Exception:
                logger.exception("Failed to set YouTube volume")
        elif self._active_source == "spotify":
            self._spotify_call(lambda sp: sp.volume(int(self._volume * 100)), "")
        self._notify()
        return f"Volume set to {int(round(self._volume * 100))} percent."

    def adjust_volume(self, delta: float) -> str:
        return self.set_volume(self._volume + delta)

    # ================================================================== #
    # Local backend (offline, default)
    # ================================================================== #
    def _find_track(self, query: Optional[str]) -> Optional[int]:
        if not self._library:
            return None
        if not query:
            # No specific song requested ("play music", "play something",
            # etc.) -> pick a random track from the local library instead
            # of erroring out or always defaulting to the same track.
            return random.randrange(len(self._library))
        names = [p.stem for p in self._library]
        close = difflib.get_close_matches(query, names, n=1, cutoff=0.3)
        if close:
            return names.index(close[0])
        query_lower = query.lower()
        for i, name in enumerate(names):
            if query_lower in name.lower():
                return i
        return None

    def _play_local(self, query: Optional[str]) -> str:
        if not self._mixer_ready:
            return "Music playback isn't available right now - the audio system failed to start."
        self.refresh_library()
        if not self._library:
            return f"I couldn't find any tracks in the '{self.music_dir}' folder."

        idx = self._find_track(query)
        if idx is None:
            return f"I couldn't find a track matching '{query}'."

        try:
            import pygame
            track = self._library[idx]
            pygame.mixer.music.load(str(track))
            pygame.mixer.music.set_volume(self._volume)
            pygame.mixer.music.play()
            self._current_index = idx
            self._reset_playback_state("local", track.stem, playing=True)
            self._start_watcher()
            self._notify()
            return f"Now playing {track.stem}."
        except Exception:
            logger.exception("Failed to play local track: %s", query)
            return "Sorry, I ran into a problem trying to play that."

    def _pause_local(self) -> str:
        if not self._mixer_ready or not self._is_playing or self._is_paused:
            self._notify()
            return "Nothing is currently playing."
        try:
            import pygame
            pygame.mixer.music.pause()
            self._is_paused = True
            self._notify()
            return "Paused."
        except Exception:
            logger.exception("Failed to pause playback")
            return "I couldn't pause the music."

    def _resume_local(self) -> str:
        if not self._mixer_ready or self._current_index is None:
            return "There's nothing queued up to resume."
        try:
            import pygame
            pygame.mixer.music.unpause()
            self._is_paused = False
            self._is_playing = True
            self._notify()
            return "Resuming playback."
        except Exception:
            logger.exception("Failed to resume playback")
            return "I couldn't resume the music."

    def _stop_local(self) -> str:
        if not self._mixer_ready:
            return "Music playback isn't available."
        try:
            import pygame
            pygame.mixer.music.stop()
            self._stop_watcher()
            return "Stopped the music."
        except Exception:
            logger.exception("Failed to stop playback")
            return "I couldn't stop the music."

    def _next_local(self) -> str:
        if not self._library:
            return self._play_local(None)
        next_idx = 0 if self._current_index is None else (self._current_index + 1) % len(self._library)
        return self._play_local(self._library[next_idx].stem)

    def _previous_local(self) -> str:
        if not self._library:
            return self._play_local(None)
        prev_idx = 0 if self._current_index is None else (self._current_index - 1) % len(self._library)
        return self._play_local(self._library[prev_idx].stem)

    def _start_watcher(self):
        self._stop_watcher()
        self._watch_stop.clear()

        def _watch():
            try:
                import pygame
                while not self._watch_stop.is_set():
                    time.sleep(0.5)
                    if (self._active_source == "local" and self._is_playing
                            and not self._is_paused and not pygame.mixer.music.get_busy()):
                        self._is_playing = False
                        self._notify()
                        break
            except Exception:
                logger.exception("Music watcher thread crashed")

        self._watch_thread = threading.Thread(target=_watch, daemon=True)
        self._watch_thread.start()

    def _stop_watcher(self):
        self._watch_stop.set()

    # ================================================================== #
    # Spotify backend (optional, online, requires Premium + an open device)
    # ================================================================== #
    def _get_spotify_client(self):
        """Lazily builds a spotipy client from env-configured credentials.
        Returns None (never raises) if spotipy isn't installed or creds are
        missing, so this backend fails soft with a spoken explanation."""
        if self._spotify_client is not None:
            return self._spotify_client
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            client_id = os.getenv("SPOTIFY_CLIENT_ID")
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
            redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
            if not client_id or not client_secret:
                logger.warning("Spotify requested but SPOTIFY_CLIENT_ID/SECRET are not set in .env")
                return None

            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-modify-playback-state user-read-playback-state",
                cache_path=".spotify_token_cache",
                open_browser=True,
            )
            self._spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            return self._spotify_client
        except ImportError:
            logger.warning("spotipy isn't installed - run 'pip install spotipy' to enable Spotify.")
            return None
        except Exception:
            logger.exception("Failed to initialize Spotify client")
            return None

    def _play_spotify(self, query: Optional[str]) -> str:
        sp = self._get_spotify_client()
        if sp is None:
            return ("Spotify isn't set up yet - add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET "
                     "to your .env file (and make sure 'pip install spotipy' has been run).")
        if not query:
            return "Tell me a song or artist to play on Spotify."
        try:
            devices = sp.devices().get("devices", [])
            if not devices:
                return "I can't find an active Spotify device - open the Spotify app on this computer or phone first."
            device_id = next((d["id"] for d in devices if d.get("is_active")), devices[0]["id"])

            results = sp.search(q=query, type="track", limit=1)
            items = results.get("tracks", {}).get("items", [])
            if not items:
                return f"I couldn't find '{query}' on Spotify."
            track = items[0]

            sp.start_playback(device_id=device_id, uris=[track["uri"]])
            sp.volume(int(self._volume * 100), device_id=device_id)
            title = f"{track['name']} by {track['artists'][0]['name']}"
            self._reset_playback_state("spotify", title, playing=True)
            self._notify()
            return f"Now playing {title} on Spotify."
        except Exception:
            logger.exception("Spotify playback failed for query: %s", query)
            return "Something went wrong starting Spotify playback."

    def _spotify_call(self, action, success_message: str) -> str:
        sp = self._get_spotify_client()
        if sp is None:
            return "Spotify isn't connected right now."
        try:
            action(sp)
            if success_message:
                self._notify()
            return success_message
        except Exception:
            logger.exception("Spotify control call failed")
            return "That Spotify command didn't go through - make sure Spotify is open and active."

    # ================================================================== #
    # YouTube backend (optional, online, streams via python-vlc if present)
    # ================================================================== #
    def _get_vlc_player(self):
        if self._vlc_player is not None:
            return self._vlc_player
        try:
            import vlc
            self._vlc_instance = vlc.Instance("--no-video")
            self._vlc_player = self._vlc_instance.media_player_new()
            return self._vlc_player
        except ImportError:
            logger.warning("python-vlc isn't installed - YouTube audio will open in the browser instead.")
            return None
        except Exception:
            logger.exception("Failed to initialize VLC for YouTube playback")
            return None

    def _play_youtube(self, query: Optional[str]) -> str:
        if not query:
            return "Tell me a song or video to play on YouTube."
        try:
            import yt_dlp
        except ImportError:
            return "YouTube playback isn't set up yet - run 'pip install yt-dlp' to enable it."

        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "quiet": True,
                "noplaylist": True,
                "default_search": "ytsearch1",
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if "entries" in info:
                    info = info["entries"][0]
                stream_url = info["url"]
                title = info.get("title", query)
                video_id = info.get("id", "")
        except Exception:
            logger.exception("YouTube search/extraction failed for query: %s", query)
            return "Something went wrong searching YouTube for that."

        player = self._get_vlc_player()
        if player is None:
            try:
                import webbrowser
                webbrowser.open(f"https://www.youtube.com/watch?v={video_id}")
                self._reset_playback_state("youtube", title, playing=True)
                self._notify()
                return f"I opened '{title}' on YouTube in your browser (install python-vlc for hands-free playback)."
            except Exception:
                logger.exception("Failed to open browser fallback for YouTube")
                return "I found that on YouTube but couldn't start playback."

        try:
            import vlc
            media = self._vlc_instance.media_new(stream_url)
            player.set_media(media)
            player.audio_set_volume(int(self._volume * 100))
            player.play()
            self._reset_playback_state("youtube", title, playing=True)
            self._notify()
            return f"Now playing {title} from YouTube."
        except Exception:
            logger.exception("VLC playback failed for YouTube stream: %s", query)
            return "Something went wrong trying to stream that from YouTube."

    def _vlc_call(self, action, success_message: str) -> str:
        if self._vlc_player is None:
            return "There's no YouTube playback active right now."
        try:
            action(self._vlc_player)
            if "pause" in success_message.lower():
                self._is_paused = True
            elif "resum" in success_message.lower() or "play" in success_message.lower():
                self._is_paused = False
                self._is_playing = True
            self._notify()
            return success_message
        except Exception:
            logger.exception("VLC control call failed")
            return "That YouTube playback command didn't go through."

    # ------------------------------------------------------------------ #
    def get_now_playing(self) -> dict:
        track_name = self._current_title
        if self._active_source == "local" and track_name is None:
            if self._current_index is not None and self._current_index < len(self._library):
                track_name = self._library[self._current_index].stem
        return {
            "track": track_name,
            "source": self._active_source,
            "is_playing": self._is_playing,
            "is_paused": self._is_paused,
            "volume": self._volume,
            "library_size": len(self._library),
        }