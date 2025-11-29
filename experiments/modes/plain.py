"""Plain pHash baseline mode."""

from .base_runner import ModeRunner


class PlainModeRunner(ModeRunner):
    name = "plain"
    description = "Plain pHash baseline without SIS or MPC."
