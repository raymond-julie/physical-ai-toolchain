"""Analog-sensor -> Robotiq position scaling.

The leader tool head carries an analog sensor (e.g. a squeeze/force pad wired to
a UR tool analog input). This module turns its raw reading into a Robotiq
position command (0 = open, 255 = closed).

Pipeline:

    raw --normalize--> [0,1] --invert?--> deadband/snap --> 0..255

The mapping is deliberately stateless and pure so it can be unit-tested with no
hardware.

This module is part of the shelved analog-teleop path and is not imported in
recording-only mode; it remains for when mirroring is re-enabled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalogGripperMap:
    """Maps a raw analog reading to a Robotiq 0..255 position.

    Args:
        analog_min: Raw value corresponding to *fully open*.
        analog_max: Raw value corresponding to *fully closed*.
        invert: If True, high signal = open instead of closed.
        open_band: Normalized fraction near 0 that snaps to fully open.
        close_band: Normalized fraction near 1 that snaps to fully closed.
    """

    analog_min: float = 0.0
    analog_max: float = 10.0
    invert: bool = False
    open_band: float = 0.03
    close_band: float = 0.03

    def normalize(self, raw: float) -> float:
        """Return the closed-fraction in [0, 1] for a raw analog value."""
        span = self.analog_max - self.analog_min
        if abs(span) < 1e-9:
            return 0.0
        frac = (raw - self.analog_min) / span
        frac = max(0.0, min(1.0, frac))
        if self.invert:
            frac = 1.0 - frac
        if frac <= self.open_band:
            frac = 0.0
        elif frac >= 1.0 - self.close_band:
            frac = 1.0
        return frac

    def to_position(self, raw: float) -> int:
        """Return a Robotiq position 0..255 for a raw analog value."""
        return round(self.normalize(raw) * 255.0)


# Map the human-facing analog-input name to candidate ur_rtde getter methods.
# UR teach-pendant "tool analog input 2/3" == analog_in[2]/[3] == tool0/tool1.
# Some ur_rtde builds expose tool analog inputs directly; where they do not, the
# LeaderInterface falls back to the official RTDE recipe (see ur_interface).
ANALOG_INPUT_GETTERS = {
    "tool0": ("getToolAnalogInput0",),
    "tool1": ("getToolAnalogInput1",),
    "tool2": ("getToolAnalogInput0",),  # pendant label alias
    "tool3": ("getToolAnalogInput1",),  # pendant label alias
    "standard0": ("getStandardAnalogInput0",),
    "standard1": ("getStandardAnalogInput1",),
}

# Matching field names in the official UR RTDE output recipe, used as a fallback
# when ur_rtde does not expose a getter for the requested input.
ANALOG_INPUT_RECIPE_FIELDS = {
    "tool0": "tool_analog_input0",
    "tool1": "tool_analog_input1",
    "tool2": "tool_analog_input0",
    "tool3": "tool_analog_input1",
    "standard0": "standard_analog_input0",
    "standard1": "standard_analog_input1",
}
