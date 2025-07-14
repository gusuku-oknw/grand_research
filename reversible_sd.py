# reversible_sd_fixed.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

LIGHT_MODELS = {
    "modi_diffusion": "nitrosocke/mo-di-diffusion",
    # 必要に応じて他の軽量モデルを追加
}

def get_pipeline(
    model_key: str = "modi_diffusion",
    device: str = "cuda",
    dtype=torch.float16,
) -> StableDiffusionPipeline:
    """
    device_map="auto" と low_cpu_mem_usage=True で、
    メモリ節約しつつモデルを正しくロードします。
    """
    model_id = LIGHT_MODELS[model_key]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    # もし meta テンソル関連の警告が出るなら、hf 環境変数で無効化も検討:
    # os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # attention slicing は残しても OK
    pipe.enable_attention_slicing()
    # safety checker 無効化
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    return pipe

def latent_to_image(latent: torch.FloatTensor, pipe: StableDiffusionPipeline) -> Image.Image:
    with torch.no_grad():
        image_tensor = pipe.vae.decode(latent).sample
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    arr = (image_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr[0])

def image_to_latent(image: Image.Image, pipe: StableDiffusionPipeline) -> torch.FloatTensor:
    latent_size = pipe.vae.config.sample_size  # 例: 64
    img_resolution = latent_size * 8  # → 512
    image = image.convert("RGB").resize(
            (img_resolution, img_resolution),
                resample = Image.LANCZOS
                )
    img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img_t = img_t.to(dtype=pipe.unet.dtype, device=pipe.device)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(img_t).latent_dist
        latent = latent_dist.sample() * pipe.vae.config.scaling_factor
    return latent

if __name__ == "__main__":
    pipe = get_pipeline("modi_diffusion", device="cuda")
    latent = torch.randn(
        1, pipe.vae.config.latent_channels,
        pipe.vae.config.sample_size, pipe.vae.config.sample_size,
        device=pipe.device, dtype=pipe.unet.dtype
    )
    img = latent_to_image(latent, pipe)
    latent2 = image_to_latent(img, pipe)
    diff = torch.norm(latent - latent2) / torch.numel(latent)
    print(f"Average L2 difference per element: {diff.item():.6f}")
