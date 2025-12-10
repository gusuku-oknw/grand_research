"""Dealer-free staged mode that refines reconstruction thresholds."""

from .base_runner import ModeRunner


class SISClientPartialModeRunner(ModeRunner):
    name = "sis_client_partial"
    description = "Dealer-free staged mode with incremental reconstruction."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
