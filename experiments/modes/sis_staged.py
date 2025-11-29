"""SIS staged (multiple thresholds) mode."""

from .base_runner import ModeRunner


class SISStagedModeRunner(ModeRunner):
    name = "sis_staged"
    description = "Stage-wise SIS where reconstruction progresses with increasing k."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
