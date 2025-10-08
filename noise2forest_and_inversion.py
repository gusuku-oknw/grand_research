import os, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
try:
    from diffusers import PipelineQuantizationConfig          # v0.34+
except ImportError:
    from diffusers.quantizers import PipelineQuantizationConfig  # v0.33.x↓

# -------- 環境 ----------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"   # 断片化緩和:contentReference[oaicite:3]{index=3}
torch.cuda.set_per_process_memory_fraction(0.75, 0)              # 8 GB×75 %=6 GB :contentReference[oaicite:4]{index=4}

MODEL = "nitrosocke/mo-di-diffusion"
OUT   = "out"; os.makedirs(OUT, exist_ok=True)

# -------- 4-bit 量子化 ----------
quant = PipelineQuantizationConfig(                              # bitsandbytes 4bit:contentReference[oaicite:5]{index=5}
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
    components_to_quantize=["unet", "vae", "text_encoder"],
)

# -------- Pipeline 読み込み (sequential map) ----------
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL,
    scheduler=DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler"),
    torch_dtype=torch.float16,
    device_map="balanced",                                    # ← balanced ではなく sequential:contentReference[oaicite:6]{index=6}
    max_memory={0: "6GB", "cpu": "16GB"},
    quantization_config=quant,
    safety_checker=None,
    low_cpu_mem_usage=True,
)

# -------- メモリ節約 ----------
pipe.enable_attention_slicing()                                 # Slice Attention:contentReference[oaicite:8]{index=8}
pipe.enable_vae_slicing(); pipe.enable_vae_tiling()             # Slice & Tile VAE:contentReference[oaicite:9]{index=9}
try: pipe.enable_xformers_memory_efficient_attention()          # xFormers:contentReference[oaicite:10]{index=10}
except ImportError: pass

# -------- 勾配チェックポイント（任意：速度 ↔ メモリ） ----------
pipe.unet.enable_gradient_checkpointing()                       #:contentReference[oaicite:11]{index=11}
pipe.text_encoder.gradient_checkpointing_enable()               #:contentReference[oaicite:12]{index=12}

# -------- Helper ----------
def encode_to_latent(img):
    sz = pipe.vae.config.sample_size * 8
    arr = np.array(img.resize((sz, sz))).astype(np.float32)
    t   = torch.from_numpy(arr).permute(2,0,1)[None]/127.5 - 1
    with torch.no_grad():
        return pipe.vae.encode(t.to("cuda", dtype=pipe.unet.dtype)).latent_dist.sample()*pipe.vae.config.scaling_factor

# -------- 生成 ----------
lat = torch.randn(1, pipe.vae.config.latent_channels, pipe.vae.config.sample_size, pipe.vae.config.sample_size,
                  device="cuda", dtype=pipe.unet.dtype)

with torch.no_grad():
    out = pipe("a photo of a lush forest",
               height=128, width=128,            # 超低解像度:contentReference[oaicite:13]{index=13}
               num_inference_steps=20,           # 20 step
               guidance_scale=7.5,
               latents=lat, output_type="latent")
forest_lat = out.latents
forest_img = pipe.decode_latents(forest_lat)[0]
forest_img.save(f"{OUT}/forest.png")

# -------- 反転 (DDIM-INVERSE) ----------
sched = pipe.scheduler; sched.set_timesteps(20)
lat   = encode_to_latent(forest_img)
pe    = pipe._encode_prompt("a photo of a lush forest", 1)[0]
for t in sched.timesteps[::-1]:
    with torch.no_grad():
        eps = pipe.unet(lat, t, encoder_hidden_states=pe).sample
    lat = sched.step(eps, t, lat).prev_sample

narr = lat[0,0].cpu().numpy()
Image.fromarray(((narr-narr.min())/narr.ptp()*255).astype(np.uint8)).save(f"{OUT}/noise.png")
print("done")
