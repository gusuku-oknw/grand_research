# reversible_sd_steps_and_forest.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os

# --- モデル設定 ---
LIGHT_MODELS = {
    "modi_diffusion": "nitrosocke/mo-di-diffusion",
}

def get_pipeline(model_key="modi_diffusion", device="cuda", dtype=torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(
        LIGHT_MODELS[model_key],
        torch_dtype=dtype,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    return pipe

# --- ノイズ潜在の可視化（1チャンネルだけ抜き出して正規化） ---
def visualize_latent(latent: torch.FloatTensor) -> Image.Image:
    # latent: (1, C, H, W) からチャンネル 0 を抜き出し
    arr = latent[0, 0].cpu().numpy()
    # [-L, L] の値域を [0,255] に線形マッピング
    mn, mx = arr.min(), arr.max()
    img = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L").resize((512,512))

# --- latent → 画像 ---
def latent_to_image(latent, pipe):
    with torch.no_grad():
        img_t = pipe.vae.decode(latent).sample
    img_t = (img_t / 2 + 0.5).clamp(0,1)
    arr = (img_t.cpu().permute(0,2,3,1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr[0])

# --- 画像 → latent ---
def image_to_latent(image, pipe):
    latent_size = pipe.vae.config.sample_size
    img_res = latent_size * 8
    resized = image.convert("RGB").resize((img_res, img_res), resample=Image.LANCZOS)
    img_t = (torch.from_numpy(np.array(resized)).permute(2,0,1).unsqueeze(0) / 127.5 - 1.0)
    img_t = img_t.to(dtype=pipe.unet.dtype, device=pipe.device)
    with torch.no_grad():
        dist = pipe.vae.encode(img_t).latent_dist
        latent = dist.sample() * pipe.vae.config.scaling_factor
    return latent, resized

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    pipe = get_pipeline(device="cuda")

    # --- 1) ランダムノイズ latent を生成 & 可視化 ---
    latent_noise = torch.randn(
        1, pipe.vae.config.latent_channels,
        pipe.vae.config.sample_size, pipe.vae.config.sample_size,
        device=pipe.device, dtype=pipe.unet.dtype
    )
    noise_img = visualize_latent(latent_noise)
    noise_img.save("out/step1_noise.png")         # ノイズ画像

    # --- 2) ノイズ→デコード画像 ---
    decoded = latent_to_image(latent_noise, pipe)
    decoded.save("out/step2_decoded.png")          # デコード画像

    # --- 3) デコード→再エンコード & 可視化 ---
    latent2, resized_for_vae = image_to_latent(decoded, pipe)
    recon_noise_img = visualize_latent(latent2)
    recon_noise_img.save("out/step3_reencoded_noise.png")  # 再エンコードノイズ画像

    # --- 4) 森（forest）のテキストから生成 ---
    forest = pipe(
        "a photo of a lush forest, high resolution",
        num_inference_steps=30
    ).images[0]
    forest.save("out/forest.png")

    print("Saved images in ./out/:")
    print("  1) step1_noise.png")
    print("  2) step2_decoded.png")
    print("  3) step3_reencoded_noise.png")
    print("  4) forest.png")
