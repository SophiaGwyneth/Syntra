"""
home_simulator.py
------------------
A headless, dependency-free virtual smart-home device simulator.

This module intentionally contains NO GUI code. It owns the authoritative
state of every simulated smart-home device and exposes a small API that both
the AI engine (via structured actions) and the GUI (via manual quick-toggle
controls) can call. Keeping this logic separate from the GUI means the same
simulator could be reused with a different front-end (CLI, web, tests, etc.)
without any changes.

State changes are broadcast through an optional `on_state_change` callback,
which the GUI subscribes to in order to keep its widgets in sync.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("VirtualAssistant")

# Supported devices and the actions that apply to them. Kept here so both
# the AI engine's prompt and any future validation can stay aligned with the
# simulator's real capabilities.
SUPPORTED_DEVICES = {
    "living_room_light": ["turn_on", "turn_off"],
    "kitchen_light": ["turn_on", "turn_off"],
    "thermostat": ["set_temp", "increase_temp", "decrease_temp"],
    "front_door_lock": ["lock", "unlock"],
}

MIN_TEMP_C = 10.0
MAX_TEMP_C = 32.0


@dataclass
class HomeState:
    """Snapshot of every simulated device's current state."""
    living_room_light: bool = False
    kitchen_light: bool = False
    thermostat: float = 20.0
    front_door_lock: bool = True  # True = locked, False = unlocked

    def as_dict(self) -> dict:
        return {
            "living_room_light": self.living_room_light,
            "kitchen_light": self.kitchen_light,
            "thermostat": self.thermostat,
            "front_door_lock": self.front_door_lock,
        }


class HomeSimulator:
    """
    Owns and mutates the virtual smart-home state.

    Usage:
        sim = HomeSimulator(on_state_change=my_gui.update_device_display)
        sim.apply_action(action_item)          # from the AI engine
        sim.toggle_light("kitchen_light")       # from a manual GUI toggle
    """

    def __init__(self, on_state_change: Optional[Callable[[dict, str], None]] = None):
        self.state = HomeState()
        self.on_state_change = on_state_change

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _notify(self, changed_target: str):
        logger.info(
            "Simulator state updated: %s -> %s", changed_target, self.state.as_dict()
        )
        if self.on_state_change:
            try:
                self.on_state_change(self.state.as_dict(), changed_target)
            except Exception:
                logger.exception("on_state_change callback raised an exception")

    def _clamp_temp(self, value: float) -> float:
        return max(MIN_TEMP_C, min(MAX_TEMP_C, value))

    # ------------------------------------------------------------------ #
    # AI-driven structured actions
    # ------------------------------------------------------------------ #
    def apply_action(self, action_item) -> bool:
        """
        Applies a single SmartHomeAction (from ai_engine) to the simulator.
        Returns True if a recognized action was applied, False otherwise.
        Never raises - unrecognized targets/actions are logged and ignored
        so a bad/hallucinated AI response can never crash the app.
        """
        try:
            act = getattr(action_item, "action", None)
            target = getattr(action_item, "target", None)
            val = getattr(action_item, "value", None)

            if target not in SUPPORTED_DEVICES:
                logger.warning("Ignoring action for unknown target: %s", target)
                return False

            if act not in SUPPORTED_DEVICES[target]:
                logger.warning("Ignoring unsupported action '%s' for target '%s'", act, target)
                return False

            if target in ("living_room_light", "kitchen_light"):
                setattr(self.state, target, act == "turn_on")

            elif target == "thermostat":
                if act == "set_temp" and val is not None:
                    self.state.thermostat = self._clamp_temp(float(val))
                elif act == "increase_temp":
                    self.state.thermostat = self._clamp_temp(
                        self.state.thermostat + (float(val) if val else 1.0)
                    )
                elif act == "decrease_temp":
                    self.state.thermostat = self._clamp_temp(
                        self.state.thermostat - (float(val) if val else 1.0)
                    )

            elif target == "front_door_lock":
                self.state.front_door_lock = (act == "lock")

            self._notify(target)
            return True

        except Exception:
            logger.exception("Failed to apply action safely: %s", action_item)
            return False

    # ------------------------------------------------------------------ #
    # Manual quick-toggle controls (driven directly by the GUI)
    # ------------------------------------------------------------------ #
    def toggle_light(self, target: str):
        if target not in ("living_room_light", "kitchen_light"):
            logger.warning("toggle_light called with invalid target: %s", target)
            return
        current = getattr(self.state, target)
        setattr(self.state, target, not current)
        self._notify(target)

    def toggle_lock(self):
        self.state.front_door_lock = not self.state.front_door_lock
        self._notify("front_door_lock")

    def set_thermostat(self, value: float):
        self.state.thermostat = self._clamp_temp(float(value))
        self._notify("thermostat")

    def adjust_thermostat(self, delta: float):
        self.state.thermostat = self._clamp_temp(self.state.thermostat + float(delta))
        self._notify("thermostat")

    def get_state(self) -> dict:
        return self.state.as_dict()
