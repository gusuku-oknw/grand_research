"""Traditional SIS (dealer-based) naive mode."""

from .base_runner import ModeRunner


class SISNaiveModeRunner(ModeRunner):
    name = "sis_naive"
    description = "Standard SIS with simple reconstruction (dealer-based)."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
