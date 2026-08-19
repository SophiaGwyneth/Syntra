"""
gui.py
------
Dark-mode, futuristic customtkinter dashboard for Syntra.

The GUI is intentionally "dumb": it never talks to the AI engine, the voice
pipeline, or the simulator directly. It only:
  1. Renders state it's given (chat messages, device state, status).
  2. Forwards user intent (typed text, voice button press, device toggle)
     back up to whoever constructed it, via callbacks.

This keeps main.py as the single place that wires business logic together,
and lets the GUI be restyled/rebuilt without touching AI or simulator code.

All public methods that mutate widgets are safe to call from ANY thread -
they marshal themselves onto the Tkinter main thread via `self.after(0, ...)`.
"""

import datetime
import logging
import math

import customtkinter as ctk
import tkinter as tk

logger = logging.getLogger("VirtualAssistant")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ---------------------------------------------------------------------- #
# Palette - dark, futuristic / "HUD" inspired
# ---------------------------------------------------------------------- #
BG_MAIN = "#0b0f1a"
BG_PANEL = "#11172a"
BG_CARD = "#161d33"
BG_CARD_ACTIVE = "#1f2a4d"
FG_PRIMARY = "#e6ecff"
FG_MUTED = "#7c88ad"
ACCENT_CYAN = "#00e5ff"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#39ff9d"
ACCENT_AMBER = "#ffb347"
ACCENT_MAGENTA = "#ff5ac8"
ACCENT_RED = "#ff5470"

STATUS_COLORS = {
    "idle": ACCENT_BLUE,
    "listening": ACCENT_GREEN,
    "processing": ACCENT_AMBER,
    "speaking": ACCENT_MAGENTA,
}

STATUS_LABELS = {
    "idle": "IDLE",
    "listening": "LISTENING",
    "processing": "PROCESSING",
    "speaking": "SPEAKING",
}


class StatusOrb(tk.Canvas):
    """A glowing, gently pulsing status ring rendered on a plain tk.Canvas
    (customtkinter has no native canvas widget, so we embed one and match
    its background to the surrounding panel for a seamless look)."""

    SIZE = 180

    def __init__(self, master, **kwargs):
        super().__init__(
            master, width=self.SIZE, height=self.SIZE,
            bg=BG_PANEL, highlightthickness=0, **kwargs
        )
        self._state = "idle"
        self._phase = 0.0
        self._running = True
        self._draw()
        self._animate()

    def set_state(self, state: str):
        if state not in STATUS_COLORS:
            state = "idle"
        self._state = state

    def stop(self):
        self._running = False

    def _draw(self):
        self.delete("all")
        cx = cy = self.SIZE / 2
        color = STATUS_COLORS[self._state]

        # Pulse amount: idle barely breathes, active states pulse more
        pulse_amp = 3 if self._state == "idle" else 8
        pulse = pulse_amp * (0.5 + 0.5 * math.sin(self._phase))

        # Outer glow rings
        base_r = 55 + pulse
        for i, spread in enumerate([28, 18, 9]):
            r = base_r + spread
            self.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=color, width=1
            )

        # Core ring
        r = base_r
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=4)

        # Inner filled core
        inner_r = 30 + pulse * 0.4
        self.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
            fill=color, outline=""
        )

        # Status label
        self.create_text(
            cx, cy, text=STATUS_LABELS[self._state],
            fill=BG_MAIN, font=("Consolas", 11, "bold")
        )

    def _animate(self):
        if not self._running:
            return
        self._phase += 0.18
        self._draw()
        self.after(60, self._animate)


class DeviceCard(ctk.CTkFrame):
    """A single smart-home device row with a label, live state text, and a
    quick-toggle / adjust control."""

    def __init__(self, master, title: str, on_toggle=None, on_adjust=None, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=12, **kwargs)
        self.title = title
        self.on_toggle = on_toggle
        self.on_adjust = on_adjust

        self.grid_columnconfigure(0, weight=1)

        self.name_label = ctk.CTkLabel(
            self, text=title, font=("Segoe UI", 13, "bold"),
            text_color=FG_PRIMARY, anchor="w"
        )
        self.name_label.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))

        self.state_label = ctk.CTkLabel(
            self, text="--", font=("Consolas", 12),
            text_color=FG_MUTED, anchor="w"
        )
        self.state_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=8)

        if on_adjust:
            self.minus_btn = ctk.CTkButton(
                self.control_frame, text="-", width=32, height=32,
                fg_color=BG_CARD_ACTIVE, hover_color=ACCENT_BLUE,
                command=lambda: self.on_adjust(-1.0)
            )
            self.minus_btn.grid(row=0, column=0, padx=3)

            self.plus_btn = ctk.CTkButton(
                self.control_frame, text="+", width=32, height=32,
                fg_color=BG_CARD_ACTIVE, hover_color=ACCENT_BLUE,
                command=lambda: self.on_adjust(1.0)
            )
            self.plus_btn.grid(row=0, column=1, padx=3)

        if on_toggle:
            self.toggle_switch = ctk.CTkSwitch(
                self.control_frame, text="", width=44, command=self._handle_toggle,
                progress_color=ACCENT_GREEN, button_color=FG_PRIMARY
            )
            self.toggle_switch.grid(row=0, column=0, padx=3)

    def _handle_toggle(self):
        if self.on_toggle:
            self.on_toggle()

    def set_state_text(self, text: str, active: bool = False):
        self.state_label.configure(text=text, text_color=ACCENT_GREEN if active else FG_MUTED)
        self.configure(fg_color=BG_CARD_ACTIVE if active else BG_CARD)

    def set_switch(self, on: bool):
        if hasattr(self, "toggle_switch"):
            if on:
                self.toggle_switch.select()
            else:
                self.toggle_switch.deselect()


class MusicCard(ctk.CTkFrame):
    """Now Playing card with transport controls (play/pause, stop, next,
    previous, volume) for Syntra's local music player."""

    def __init__(self, master, on_play_pause=None, on_stop=None, on_next=None,
                 on_previous=None, on_volume=None, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=12, **kwargs)
        self.on_play_pause = on_play_pause
        self.on_stop = on_stop
        self.on_next = on_next
        self.on_previous = on_previous
        self.on_volume = on_volume

        self.grid_columnconfigure(0, weight=1)

        self.track_label = ctk.CTkLabel(
            self, text="Nothing playing", font=("Segoe UI", 13, "bold"),
            text_color=FG_PRIMARY, anchor="w", wraplength=220
        )
        self.track_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))

        self.status_label = ctk.CTkLabel(
            self, text="Say \"play some music\"", font=("Consolas", 11),
            text_color=FG_MUTED, anchor="w"
        )
        self.status_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        transport = ctk.CTkFrame(self, fg_color="transparent")
        transport.grid(row=2, column=0, pady=(0, 10))

        self.prev_btn = ctk.CTkButton(
            transport, text="⏮", width=36, height=32, fg_color=BG_CARD_ACTIVE,
            hover_color=ACCENT_BLUE, command=lambda: self.on_previous and self.on_previous()
        )
        self.prev_btn.grid(row=0, column=0, padx=3)

        self.play_pause_btn = ctk.CTkButton(
            transport, text="▶", width=44, height=32, fg_color=ACCENT_GREEN,
            text_color=BG_MAIN, hover_color=ACCENT_CYAN,
            command=lambda: self.on_play_pause and self.on_play_pause()
        )
        self.play_pause_btn.grid(row=0, column=1, padx=3)

        self.stop_btn = ctk.CTkButton(
            transport, text="⏹", width=36, height=32, fg_color=BG_CARD_ACTIVE,
            hover_color=ACCENT_RED, command=lambda: self.on_stop and self.on_stop()
        )
        self.stop_btn.grid(row=0, column=2, padx=3)

        self.next_btn = ctk.CTkButton(
            transport, text="⏭", width=36, height=32, fg_color=BG_CARD_ACTIVE,
            hover_color=ACCENT_BLUE, command=lambda: self.on_next and self.on_next()
        )
        self.next_btn.grid(row=0, column=3, padx=3)

        volume_row = ctk.CTkFrame(self, fg_color="transparent")
        volume_row.grid(row=3, column=0, pady=(0, 12))

        ctk.CTkButton(
            volume_row, text="🔉", width=32, height=28, fg_color=BG_CARD_ACTIVE,
            hover_color=ACCENT_BLUE, command=lambda: self.on_volume and self.on_volume(-0.1)
        ).grid(row=0, column=0, padx=3)

        self.volume_label = ctk.CTkLabel(
            volume_row, text="70%", font=("Consolas", 11), text_color=FG_MUTED, width=40
        )
        self.volume_label.grid(row=0, column=1, padx=6)

        ctk.CTkButton(
            volume_row, text="🔊", width=32, height=28, fg_color=BG_CARD_ACTIVE,
            hover_color=ACCENT_BLUE, command=lambda: self.on_volume and self.on_volume(0.1)
        ).grid(row=0, column=2, padx=3)

    def update_now_playing(self, now_playing: dict):
        track = now_playing.get("track")
        is_playing = now_playing.get("is_playing")
        is_paused = now_playing.get("is_paused")
        volume = now_playing.get("volume", 0.7)
        library_size = now_playing.get("library_size", 0)

        if track:
            self.track_label.configure(text=track)
        else:
            self.track_label.configure(text="Nothing playing")

        if is_paused:
            self.status_label.configure(text="Paused")
            self.play_pause_btn.configure(text="▶")
        elif is_playing:
            self.status_label.configure(text="Playing")
            self.play_pause_btn.configure(text="⏸")
        elif library_size == 0:
            self.status_label.configure(text="No tracks in music/ folder")
            self.play_pause_btn.configure(text="▶")
        else:
            self.status_label.configure(text="Say \"play some music\"")
            self.play_pause_btn.configure(text="▶")

        self.volume_label.configure(text=f"{int(round(volume * 100))}%")
        self.configure(fg_color=BG_CARD_ACTIVE if is_playing and not is_paused else BG_CARD)


class WheelColumn(ctk.CTkFrame):
    """
    A single vertical scroll-wheel spinner column (e.g. Hour, Minute, or
    AM/PM) - the previous and next values are shown faded above/below the
    bold, highlighted current value, mimicking a native mobile time-picker
    wheel. Scroll with the mouse wheel, or click a faded neighbor value to
    jump straight to it.
    """

    def __init__(self, master, values, on_change=None, width=64, **kwargs):
        super().__init__(master, fg_color="transparent", width=width, **kwargs)
        self.values = list(values)
        self.index = 0
        self.on_change = on_change

        self.grid_propagate(False)
        self.configure(height=156)
        self.grid_rowconfigure((1, 2, 3), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Up-arrow: click to scroll to the previous value.
        self.up_arrow = ctk.CTkLabel(
            self, text="▲", font=("Segoe UI", 10), text_color=FG_MUTED, cursor="hand2"
        )
        self.up_arrow.grid(row=0, column=0, pady=(2, 0))

        self.prev_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 14), text_color=FG_MUTED, cursor="hand2"
        )
        self.prev_label.grid(row=1, column=0, sticky="s", pady=(0, 2))

        self.current_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 26, "bold"), text_color=ACCENT_CYAN
        )
        self.current_label.grid(row=2, column=0)

        self.next_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 14), text_color=FG_MUTED, cursor="hand2"
        )
        self.next_label.grid(row=3, column=0, sticky="n", pady=(2, 0))

        # Down-arrow: click to scroll to the next value.
        self.down_arrow = ctk.CTkLabel(
            self, text="▼", font=("Segoe UI", 10), text_color=FG_MUTED, cursor="hand2"
        )
        self.down_arrow.grid(row=4, column=0, pady=(0, 2))

        # Mouse-wheel scrolling (Windows/macOS use <MouseWheel>; Linux uses
        # Button-4/Button-5) - bound on every sub-widget so hovering
        # anywhere in the column scrolls it.
        all_widgets = (
            self, self.up_arrow, self.prev_label, self.current_label,
            self.next_label, self.down_arrow,
        )
        for widget in all_widgets:
            widget.bind("<MouseWheel>", self._on_scroll)
            widget.bind("<Button-4>", lambda _e: self._step(-1))
            widget.bind("<Button-5>", lambda _e: self._step(1))

        # Explicit up/down arrows + clicking a faded neighbor value both
        # step the wheel - gives people without a scroll wheel (or who
        # just prefer clicking) an obvious way to change the value.
        self.up_arrow.bind("<Button-1>", lambda _e: self._step(-1))
        self.prev_label.bind("<Button-1>", lambda _e: self._step(-1))
        self.down_arrow.bind("<Button-1>", lambda _e: self._step(1))
        self.next_label.bind("<Button-1>", lambda _e: self._step(1))

        self._refresh()

    def _on_scroll(self, event):
        # event.delta is positive when scrolling up, negative scrolling down.
        self._step(-1 if event.delta > 0 else 1)

    def _step(self, direction: int):
        self.index = (self.index + direction) % len(self.values)
        self._refresh()
        if self.on_change:
            self.on_change(self.values[self.index])

    def _refresh(self):
        n = len(self.values)
        self.prev_label.configure(text=str(self.values[(self.index - 1) % n]))
        self.current_label.configure(text=str(self.values[self.index]))
        self.next_label.configure(text=str(self.values[(self.index + 1) % n]))

    def get(self) -> str:
        return self.values[self.index]

    def set(self, value):
        if value in self.values:
            self.index = self.values.index(value)
            self._refresh()


class AlarmCard(ctk.CTkFrame):
    """Sidebar card showing current alarm status, with a scroll-wheel time
    picker (Hour / Minute / AM-PM) to set a new alarm, and a button to
    stop/cancel the active one."""

    _HOURS = [f"{h:02d}" for h in range(1, 13)]
    _MINUTES = [f"{m:02d}" for m in range(60)]
    _MERIDIEMS = ["AM", "PM"]

    def __init__(self, master, on_set_alarm=None, on_stop_alarm=None, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=12, **kwargs)
        self.on_set_alarm = on_set_alarm
        self.on_stop_alarm = on_stop_alarm

        self.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self, text="No alarm set", font=("Segoe UI", 13, "bold"),
            text_color=FG_PRIMARY, anchor="w", wraplength=220
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))

        # --- Scroll-wheel picker: Hour | Minute | AM/PM --------------- #
        wheel_row = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=10)
        wheel_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        wheel_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.hour_wheel = WheelColumn(wheel_row, self._HOURS)
        self.hour_wheel.grid(row=0, column=0, sticky="ns", padx=(6, 0), pady=6)

        divider1 = ctk.CTkFrame(wheel_row, fg_color=BG_CARD_ACTIVE, width=1)
        divider1.grid(row=0, column=0, sticky="nse", pady=14)

        self.minute_wheel = WheelColumn(wheel_row, self._MINUTES)
        self.minute_wheel.grid(row=0, column=1, sticky="ns", pady=6)

        divider2 = ctk.CTkFrame(wheel_row, fg_color=BG_CARD_ACTIVE, width=1)
        divider2.grid(row=0, column=1, sticky="nse", pady=14)

        self.meridiem_wheel = WheelColumn(wheel_row, self._MERIDIEMS, width=52)
        self.meridiem_wheel.grid(row=0, column=2, sticky="ns", padx=(0, 6), pady=6)

        # Default to the current time (rounded to the current minute) so
        # the picker never opens on a nonsense value.
        now = datetime.datetime.now()
        self.meridiem_wheel.set("PM" if now.hour >= 12 else "AM")
        hour_12 = now.hour % 12
        hour_12 = 12 if hour_12 == 0 else hour_12
        self.hour_wheel.set(f"{hour_12:02d}")
        self.minute_wheel.set(f"{now.minute:02d}")

        self.set_btn = ctk.CTkButton(
            self, text="Set Alarm", height=32, corner_radius=16,
            fg_color=ACCENT_BLUE, hover_color=ACCENT_CYAN, text_color=BG_MAIN,
            command=self._handle_set
        )
        self.set_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.stop_btn = ctk.CTkButton(
            self, text="Stop / Cancel Alarm", height=32, corner_radius=16,
            fg_color=BG_CARD_ACTIVE, hover_color=ACCENT_RED,
            command=self._handle_stop
        )
        self.stop_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))

    def _handle_set(self):
        value = f"{self.hour_wheel.get()}:{self.minute_wheel.get()} {self.meridiem_wheel.get()}"
        if self.on_set_alarm:
            self.on_set_alarm(value)

    def _handle_stop(self):
        if self.on_stop_alarm:
            self.on_stop_alarm()

    def update_alarm_status(self, text: str, ringing: bool = False):
        self.status_label.configure(text=text, text_color=ACCENT_RED if ringing else FG_PRIMARY)
        self.configure(fg_color=BG_CARD_ACTIVE if ringing else BG_CARD)


class SyntraGUI(ctk.CTk):
    """
    Main application window.

    Constructor callbacks (all optional, all called with plain args - never
    with GUI objects):
        on_text_command(text: str)
        on_voice_command()
        on_device_toggle(target: str)
        on_thermostat_adjust(delta: float)
        on_lock_toggle()
        on_voice_gender_change(gender: str)  # "male" | "female"
    """

    def __init__(
        self,
        on_text_command=None,
        on_voice_command=None,
        on_device_toggle=None,
        on_thermostat_adjust=None,
        on_lock_toggle=None,
        on_music_control=None,
        on_alarm_set=None,
        on_alarm_stop=None,
        on_voice_gender_change=None,
    ):
        super().__init__()

        self.on_text_command = on_text_command
        self.on_voice_command = on_voice_command
        self.on_device_toggle = on_device_toggle
        self.on_thermostat_adjust = on_thermostat_adjust
        self.on_lock_toggle = on_lock_toggle
        self.on_music_control = on_music_control
        self.on_alarm_set = on_alarm_set
        self.on_alarm_stop = on_alarm_stop
        self.on_voice_gender_change = on_voice_gender_change
        self._alarm_popup = None

        self.title("SYNTRA — AI Home Assistant")
        self.geometry("980x640")
        self.minsize(860, 560)
        self.configure(fg_color=BG_MAIN)

        self._build_layout()

    # ------------------------------------------------------------------ #
    # Layout construction
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        # Converted left container to CTkScrollableFrame with custom theme styling
        panel = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_PANEL,
            corner_radius=0,
            scrollbar_button_color=BG_CARD_ACTIVE,
            scrollbar_button_hover_color=ACCENT_BLUE,
        )
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            panel, text="SYNTRA", font=("Segoe UI", 22, "bold"), text_color=ACCENT_CYAN
        )
        header.grid(row=0, column=0, pady=(24, 0))

        subheader = ctk.CTkLabel(
            panel, text="AI HOME ASSISTANT", font=("Consolas", 10),
            text_color=FG_MUTED
        )
        subheader.grid(row=1, column=0, pady=(0, 10))

        self.status_orb = StatusOrb(panel)
        self.status_orb.grid(row=2, column=0, pady=10)

        self.voice_button = ctk.CTkButton(
            panel, text="🎤  VOICE MODE", font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT_BLUE, hover_color=ACCENT_CYAN, text_color=BG_MAIN,
            height=44, corner_radius=22, command=self._handle_voice_button
        )
        self.voice_button.grid(row=3, column=0, sticky="ew", padx=24, pady=(6, 20))

        voice_settings_label = ctk.CTkLabel(
            panel, text="VOICE SETTINGS", font=("Consolas", 11, "bold"), text_color=FG_MUTED
        )
        voice_settings_label.grid(row=4, column=0, sticky="w", padx=24, pady=(0, 6))

        voice_settings_frame = ctk.CTkFrame(panel, fg_color=BG_CARD, corner_radius=12)
        voice_settings_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 16))
        voice_settings_frame.grid_columnconfigure(0, weight=1)

        gender_row_label = ctk.CTkLabel(
            voice_settings_frame, text="Voice Gender", font=("Segoe UI", 12),
            text_color=FG_PRIMARY, anchor="w"
        )
        gender_row_label.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self.voice_gender_selector = ctk.CTkSegmentedButton(
            voice_settings_frame, values=["Female", "Male"],
            selected_color=ACCENT_BLUE, selected_hover_color=ACCENT_CYAN,
            unselected_color=BG_CARD_ACTIVE, text_color=FG_PRIMARY,
            command=self._handle_voice_gender_change
        )
        self.voice_gender_selector.set("Female")
        self.voice_gender_selector.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

        devices_label = ctk.CTkLabel(
            panel, text="DEVICES", font=("Consolas", 11, "bold"), text_color=FG_MUTED
        )
        devices_label.grid(row=6, column=0, sticky="w", padx=24, pady=(0, 6))

        devices_frame = ctk.CTkFrame(panel, fg_color="transparent")
        devices_frame.grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 10))
        devices_frame.grid_columnconfigure(0, weight=1)

        self.lr_card = DeviceCard(
            devices_frame, "Living Room Light",
            on_toggle=lambda: self._handle_device_toggle("living_room_light")
        )
        self.lr_card.grid(row=0, column=0, sticky="ew", pady=6)

        self.kitchen_card = DeviceCard(
            devices_frame, "Kitchen Light",
            on_toggle=lambda: self._handle_device_toggle("kitchen_light")
        )
        self.kitchen_card.grid(row=1, column=0, sticky="ew", pady=6)

        self.thermo_card = DeviceCard(
            devices_frame, "Thermostat",
            on_adjust=self._handle_thermostat_adjust
        )
        self.thermo_card.grid(row=2, column=0, sticky="ew", pady=6)

        self.lock_card = DeviceCard(
            devices_frame, "Front Door Lock",
            on_toggle=self._handle_lock_toggle
        )
        self.lock_card.grid(row=3, column=0, sticky="ew", pady=6)

        self.humidifier_card = DeviceCard(
            devices_frame, "Humidifier",
            on_toggle=lambda: self._handle_device_toggle("humidifier")
        )
        self.humidifier_card.grid(row=4, column=0, sticky="ew", pady=6)

        self.ac_card = DeviceCard(
            devices_frame, "Air Conditioner",
            on_toggle=lambda: self._handle_device_toggle("air_conditioner")
        )
        self.ac_card.grid(row=5, column=0, sticky="ew", pady=6)

        self.back_door_card = DeviceCard(
            devices_frame, "Back Door Lock",
            on_toggle=lambda: self._handle_device_toggle("back_door")
        )
        self.back_door_card.grid(row=6, column=0, sticky="ew", pady=6)

        media_label = ctk.CTkLabel(
            panel, text="MEDIA", font=("Consolas", 11, "bold"), text_color=FG_MUTED
        )
        media_label.grid(row=8, column=0, sticky="w", padx=24, pady=(4, 6))

        self.music_card = MusicCard(
            panel,
            on_play_pause=lambda: self._handle_music_control("play_pause"),
            on_stop=lambda: self._handle_music_control("stop"),
            on_next=lambda: self._handle_music_control("next"),
            on_previous=lambda: self._handle_music_control("previous"),
            on_volume=lambda delta: self._handle_music_control("volume", delta),
        )
        self.music_card.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 16))

        alarm_label = ctk.CTkLabel(
            panel, text="ALARM", font=("Consolas", 11, "bold"), text_color=FG_MUTED
        )
        alarm_label.grid(row=10, column=0, sticky="w", padx=24, pady=(4, 6))

        self.alarm_card = AlarmCard(
            panel,
            on_set_alarm=self._handle_alarm_set,
            on_stop_alarm=self._handle_alarm_stop,
        )
        self.alarm_card.grid(row=11, column=0, sticky="ew", padx=16, pady=(0, 16))

    def _build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(right, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        top_bar.grid_columnconfigure(0, weight=1)

        self.status_text = ctk.CTkLabel(
            top_bar, text="Status: Ready.", font=("Consolas", 12, "italic"),
            text_color=FG_MUTED, anchor="w"
        )
        self.status_text.grid(row=0, column=0, sticky="w")

        self.chat_log = ctk.CTkTextbox(
            right, fg_color=BG_PANEL, corner_radius=14, font=("Consolas", 12),
            text_color=FG_PRIMARY, wrap="word", state="disabled"
        )
        self.chat_log.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)

        self.chat_log.tag_config("user", foreground=ACCENT_CYAN)
        self.chat_log.tag_config("syntra", foreground=ACCENT_GREEN)
        self.chat_log.tag_config("system", foreground=FG_MUTED)
        self.chat_log.tag_config("error", foreground=ACCENT_RED)
        self.chat_log.tag_config("timestamp", foreground=FG_MUTED)

        input_bar = ctk.CTkFrame(right, fg_color="transparent")
        input_bar.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_bar.grid_columnconfigure(0, weight=1)

        self.text_entry = ctk.CTkEntry(
            input_bar, placeholder_text="Type a command for Syntra...",
            fg_color=BG_CARD, border_color=ACCENT_BLUE, text_color=FG_PRIMARY,
            height=42, corner_radius=21
        )
        self.text_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.text_entry.bind("<Return>", lambda _event: self._handle_send())

        self.send_button = ctk.CTkButton(
            input_bar, text="Send", width=90, height=42, corner_radius=21,
            fg_color=ACCENT_BLUE, hover_color=ACCENT_CYAN, text_color=BG_MAIN,
            font=("Segoe UI", 13, "bold"), command=self._handle_send
        )
        self.send_button.grid(row=0, column=1)

    # ------------------------------------------------------------------ #
    # Internal event handlers (translate widget events -> callbacks)
    # ------------------------------------------------------------------ #
    def _handle_send(self):
        text = self.text_entry.get().strip()
        if not text:
            return
        self.text_entry.delete(0, "end")
        self.append_message("You", text, tag="user")
        if self.on_text_command:
            self.on_text_command(text)

    def _handle_voice_button(self):
        self.voice_button.configure(state="disabled", text="🎙  LISTENING...")
        self.set_status("listening", "Listening to voice command...")
        if self.on_voice_command:
            self.on_voice_command()

    def _handle_device_toggle(self, target: str):
        if self.on_device_toggle:
            self.on_device_toggle(target)

    def _handle_thermostat_adjust(self, delta: float):
        if self.on_thermostat_adjust:
            self.on_thermostat_adjust(delta)

    def _handle_lock_toggle(self):
        if self.on_lock_toggle:
            self.on_lock_toggle()

    def _handle_music_control(self, action: str, value: float = None):
        if self.on_music_control:
            self.on_music_control(action, value)

    def _handle_alarm_set(self, time_str: str):
        if self.on_alarm_set:
            self.on_alarm_set(time_str)

    def _handle_alarm_stop(self):
        if self.on_alarm_stop:
            self.on_alarm_stop()
        self.dismiss_alarm_popup()

    def _handle_voice_gender_change(self, value: str):
        if self.on_voice_gender_change:
            self.on_voice_gender_change(value.strip().lower())

    # ------------------------------------------------------------------ #
    # Thread-safe public API (safe to call from ANY thread)
    # ------------------------------------------------------------------ #
    def ui_call(self, func, *args, **kwargs):
        """Marshal a widget mutation onto the Tkinter main thread."""
        self.after(0, lambda: func(*args, **kwargs))

    def append_message(self, sender: str, text: str, tag: str = "system"):
        self.ui_call(self._append_message_impl, sender, text, tag)

    def _append_message_impl(self, sender: str, text: str, tag: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"[{timestamp}] ", "timestamp")
        self.chat_log.insert("end", f"{sender}: ", tag)
        self.chat_log.insert("end", f"{text}\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")

    def set_status(self, state: str, message: str = None):
        """state in {'idle', 'listening', 'processing', 'speaking'}"""
        self.ui_call(self._set_status_impl, state, message)

    def _set_status_impl(self, state: str, message: str):
        self.status_orb.set_state(state)
        if message:
            self.status_text.configure(text=f"Status: {message}")

    def reset_voice_button(self):
        self.ui_call(self._reset_voice_button_impl)

    def _reset_voice_button_impl(self):
        self.voice_button.configure(state="normal", text="🎤  VOICE MODE")

    def update_device_display(self, state: dict, changed_target: str = None):
        self.ui_call(self._update_device_display_impl, state)

    def _update_device_display_impl(self, state: dict):
        self.lr_card.set_state_text(
            "ON" if state["living_room_light"] else "OFF", active=state["living_room_light"]
        )
        self.lr_card.set_switch(state["living_room_light"])

        self.kitchen_card.set_state_text(
            "ON" if state["kitchen_light"] else "OFF", active=state["kitchen_light"]
        )
        self.kitchen_card.set_switch(state["kitchen_light"])

        self.thermo_card.set_state_text(f"{state['thermostat']:.1f}°C", active=True)

        locked = state["front_door_lock"]
        self.lock_card.set_state_text("LOCKED" if locked else "UNLOCKED", active=locked)
        self.lock_card.set_switch(locked)  # switch ON visually = locked

        self.humidifier_card.set_state_text(
            "ON" if state["humidifier"] else "OFF", active=state["humidifier"]
        )
        self.humidifier_card.set_switch(state["humidifier"])

        self.ac_card.set_state_text(
            "ON" if state["air_conditioner"] else "OFF", active=state["air_conditioner"]
        )
        self.ac_card.set_switch(state["air_conditioner"])

        back_locked = state["back_door"]
        self.back_door_card.set_state_text("LOCKED" if back_locked else "UNLOCKED", active=back_locked)
        self.back_door_card.set_switch(back_locked)  # switch ON visually = locked

    def update_music_display(self, now_playing: dict):
        self.ui_call(self.music_card.update_now_playing, now_playing)

    def update_alarm_display(self, text: str, ringing: bool = False):
        """Updates the sidebar Alarm card, e.g. 'Alarm set for 07:30',
        'No alarm set', or 'ALARM RINGING!' while it's going off."""
        self.ui_call(self.alarm_card.update_alarm_status, text, ringing)

    def show_alarm_popup(self, message: str = "Wake up! Your alarm is going off."):
        """Pops a prominent, always-on-top notification window the instant
        the alarm fires. Safe to call from any thread."""
        self.ui_call(self._show_alarm_popup_impl, message)

    def _show_alarm_popup_impl(self, message: str):
        # Avoid stacking multiple popups if triggered more than once.
        self.dismiss_alarm_popup()

        popup = ctk.CTkToplevel(self)
        self._alarm_popup = popup
        popup.title("⏰ Alarm")
        popup.geometry("380x200")
        popup.configure(fg_color=BG_PANEL)
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.after(50, lambda: (popup.lift(), popup.focus_force()))

        icon_label = ctk.CTkLabel(
            popup, text="⏰", font=("Segoe UI", 40)
        )
        icon_label.pack(pady=(20, 4))

        text_label = ctk.CTkLabel(
            popup, text=message, font=("Segoe UI", 15, "bold"),
            text_color=ACCENT_AMBER, wraplength=320, justify="center"
        )
        text_label.pack(expand=True, padx=20, pady=(0, 10))

        def _dismiss():
            self._handle_alarm_stop()

        dismiss_btn = ctk.CTkButton(
            popup, text="Dismiss & Stop Alarm", fg_color=ACCENT_RED, hover_color="#ff2f57",
            text_color=BG_MAIN, height=40, corner_radius=20, command=_dismiss
        )
        dismiss_btn.pack(pady=(0, 20))

        popup.protocol("WM_DELETE_WINDOW", _dismiss)

    def dismiss_alarm_popup(self):
        self.ui_call(self._dismiss_alarm_popup_impl)

    def _dismiss_alarm_popup_impl(self):
        if self._alarm_popup is not None:
            try:
                self._alarm_popup.destroy()
            except Exception:
                pass
            self._alarm_popup = None

    def show_error(self, message: str):
        self.append_message("System", message, tag="error")

    def set_voice_gender_display(self, gender: str):
        """Syncs the sidebar's Voice Gender selector to reflect the pipeline's
        actual current voice (e.g. after loading SYNTRA_VOICE_GENDER from
        .env at startup). Safe to call from any thread."""
        self.ui_call(self._set_voice_gender_display_impl, gender)

    def _set_voice_gender_display_impl(self, gender: str):
        label = "Male" if str(gender).strip().lower() == "male" else "Female"
        self.voice_gender_selector.set(label)