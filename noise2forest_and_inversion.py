import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    BitsAndBytesConfig,
    StableDiffusionPipeline,
    DDIMScheduler
)
# 4bit量子化用の設定クラス
try:
    from diffusers import PipelineQuantizationConfig
except ImportError:
    from diffusers.quantizers import PipelineQuantizationConfig

# — 定数定義 —
MODEL_ID = "nitrosocke/mo-di-diffusion"
DEVICE   = "cuda"            # GPU: "cuda" / CPU オフロード時: "cpu"
OUT_DIR  = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# — 環境変数によるメモリ断片化対策 —
# (シェルで export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 を事前に設定推奨)

# 1) 4bit量子化設定
quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",    # 量子化バックエンド名 :contentReference[oaicite:0]{index=0}
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16
    },
    components_to_quantize=["unet", "vae"]
)

# 2) パイプライン初期化 (max_memory で明示的に GPU/CPU メモリ割当) :contentReference[oaicite:1]{index=1}
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    scheduler=DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"),
    torch_dtype=torch.float16,
    device_map="balanced",
    max_memory={0: "5GB", "cpu": "16GB"},
    quantization_config=quant_config,
    safety_checker=None,
    low_cpu_mem_usage=True
)

# 3) メモリ節約オプション
pipe.enable_attention_slicing()              # Attention スライシング :contentReference[oaicite:2]{index=2}
pipe.enable_vae_slicing()                    # VAE スライシング :contentReference[oaicite:3]{index=3}
pipe.enable_vae_tiling()                     # VAE タイリング :contentReference[oaicite:4]{index=4}
try:
    pipe.enable_xformers_memory_efficient_attention()  # xFormers アテンション :contentReference[oaicite:5]{index=5}
except ImportError:
    pass

# モデルオフロード（必要に応じて有効化） :contentReference[oaicite:6]{index=6}
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

def encode_image_to_latent(img: Image.Image, pipe: StableDiffusionPipeline):
    """
    PIL Image → VAE 潜在(latent) に変換
    """
    latent_size = pipe.vae.config.sample_size
    res = latent_size * 8
    img_resized = img.convert("RGB").resize((res, res), Image.LANCZOS)
    img_t = torch.from_numpy(np.array(img_resized))
    img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img_t = img_t.to(DEVICE, dtype=pipe.unet.dtype)
    with torch.no_grad():
        lat_dist = pipe.vae.encode(img_t).latent_dist
        return lat_dist.sample() * pipe.vae.config.scaling_factor

# 4) ランダムノイズ潜在を用意
latents = torch.randn(
    1,
    pipe.vae.config.latent_channels,
    pipe.vae.config.sample_size,
    pipe.vae.config.sample_size,
    device=DEVICE,
    dtype=pipe.unet.dtype
)

# 5) ノイズ→森画像生成 (256×256, output_type="latent")
with torch.no_grad():
    out = pipe(
        prompt="a photo of a lush forest",
        num_inference_steps=30,
        guidance_scale=7.5,
        height=256,
        width=256,
        latents=latents,
        output_type="latent"
    )
forest_latent = out.latents
forest_img = pipe.decode_latents(forest_latent)[0]
forest_img.save(f"{OUT_DIR}/forest_from_noise_256.png")

# 6) 森画像→潜在エンコード
forest_latent_encoded = encode_image_to_latent(forest_img, pipe)

# 7) DDIM反転：最終潜在→初期ノイズ
scheduler = pipe.scheduler
scheduler.set_timesteps(30)
lat = forest_latent_encoded
prompt_embeds = pipe._encode_prompt("a photo of a lush forest", num_images_per_prompt=1)[0]
for t in scheduler.timesteps[::-1]:
    with torch.no_grad():
        eps = pipe.unet(lat, t, encoder_hidden_states=prompt_embeds).sample
    lat = scheduler.step(model_output=eps, timestep=t, sample=lat).prev_sample

# 8) 潜在ノイズを可視化し保存
arr = lat[0, 0].cpu().numpy()
mn, mx = arr.min(), arr.max()
noise_img = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
Image.fromarray(noise_img).convert("L") \
     .resize((512, 512)) \
     .save(f"{OUT_DIR}/noise_from_forest.png")

print("生成完了：")
print(f" - {OUT_DIR}/forest_from_noise_256.png")
print(f" - {OUT_DIR}/noise_from_forest.png")
