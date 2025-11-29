"""SIS MPC (simulated) mode."""

from .base_runner import ModeRunner


class SISMPCModeRunner(ModeRunner):
    name = "sis_mpc"
    description = "Simulated MPC mode that avoids reconstructing plain pHashes."
    def __init__(self, images, **kwargs):
        super().__init__(images, **kwargs)
