import pytest
import torch
from reversible_sd import get_pipeline, latent_to_image, image_to_latent

@pytest.fixture(scope="module")
def pipe():
    return get_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cpu")

def test_round_trip_latent(pipe):
    # Create deterministic latent
    torch.manual_seed(42)
    latent = torch.randn(1, pipe.vae.config.latent_channels,
                         pipe.vae.config.sample_size, pipe.vae.config.sample_size,
                         device=pipe.device, dtype=pipe.unet.dtype)
    img = latent_to_image(latent, pipe)
    latent2 = image_to_latent(img, pipe)
    # Check that round-trip error is within tolerance
    diff = torch.norm(latent - latent2) / torch.numel(latent)
    assert diff < 1e-2, f"Round-trip difference too high: {diff}"
