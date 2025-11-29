"""SIS selective reconstruction mode."""

from .base_runner import ModeRunner


class SISSelectiveModeRunner(ModeRunner):
    name = "sis_selective"
    description = "Selective SIS mode with partial reconstruction."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
