"""Dealer-free SIS mode with partial reconstruction."""

from .base_runner import ModeRunner


class SISClientDealerFreeModeRunner(ModeRunner):
    name = "sis_client_dealer_free"
    description = "Dealer-free selective reconstruction mode."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
