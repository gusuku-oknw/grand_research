"""
multi_secret_image_demo.py

Multi-secret・磯嚴螻､遘伜ｯ・・謨｣・峨ｒ菴ｿ縺｣縺ｦ
- 菴主刀雉ｪ縺ｼ縺九＠逕ｻ蜒・
- 鬮伜刀雉ｪ蜈・判蜒・
繧堤焚縺ｪ繧矩明蛟､ k 縺ｧ螳医ｋ繝・Δ繧ｹ繧ｯ繝ｪ繝励ヨ縲・

萓晏ｭ・
  pip install pillow cryptography
"""

import io
import os
import secrets
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageFilter
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ==============================
# 1. Shamir Secret Sharing 蝓ｺ譛ｬ
# ==============================

# 遘伜ｯ・ｼ・ES骰ｵ縺ｪ縺ｩ・峨ｒ謨ｴ謨ｰ縺ｫ縺励※蜿弱ａ繧区怏髯蝉ｽ・GF(p)
# 256bit骰ｵ繧偵◎縺ｮ縺ｾ縺ｾ蜈･繧後◆縺・・縺ｧ 2^256 繧医ｊ螟ｧ縺阪＞邏謨ｰ繧剃ｽｿ逕ｨ
P = 2**521 - 1  # 譌｢遏･縺ｮ繝｡繝ｫ繧ｻ繝ｳ繝檎ｴ謨ｰ・亥香蛻・､ｧ縺阪＞・・


def shamir_split(secret: int, k: int, n: int, p: int = P) -> List[Tuple[int, int]]:
    """
    Shamir k-of-n 遘伜ｯ・・謨｣
    secret : 0 <= secret < p 縺ｮ謨ｴ謨ｰ
    k      : 蠕ｩ蜈・↓蠢・ｦ√↑繧ｷ繧ｧ繧｢謨ｰ・磯明蛟､・・
    n      : 邱上す繧ｧ繧｢謨ｰ
    謌ｻ繧雁､ : [(x_i, y_i), ...]  i=1..n
    """
    assert 0 <= secret < p
    assert 1 < k <= n

    # 螟夐・ｼ・f(z) = a0 + a1 z + ... + a_{k-1} z^{k-1} (mod p)
    # a0 = secret, 莉悶・繝ｩ繝ｳ繝繝
    coeffs = [secret] + [secrets.randbelow(p) for _ in range(k - 1)]

    shares = []
    for x in range(1, n + 1):
        y = 0
        # Horner 豕輔〒隧穂ｾ｡
        for a in reversed(coeffs):
            y = (y * x + a) % p
        shares.append((x, y))
    return shares


def shamir_reconstruct(shares: List[Tuple[int, int]], p: int = P) -> int:
    """
    Shamir 蠕ｩ蜈・ｼ・agrange 陬憺俣・・
    shares : [(x_i, y_i), ...] 縺｡繧・≧縺ｩ k 蛟具ｼ医◎繧御ｻ･荳翫≠縺｣縺ｦ繧ょ・鬆ｭ k 蛟九ｒ菴ｿ縺・Φ螳夲ｼ・
    謌ｻ繧雁､ : secret・域紛謨ｰ・・
    """
    if len(shares) == 0:
        raise ValueError("shares is empty")

    secret = 0
    k = len(shares)
    for j, (xj, yj) in enumerate(shares):
        # ﾎｻ_j(0) = ﾎ_{m竕j} (0 - x_m) / (x_j - x_m)
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m == j:
                continue
            num = (num * (-xm)) % p
            den = (den * (xj - xm)) % p
        inv_den = pow(den, -1, p)  # 騾・・
        lj = num * inv_den % p
        secret = (secret + yj * lj) % p
    return secret


# =====================================
# 2. Multi-secret・磯嚴螻､遘伜ｯ・・謨｣・峨Λ繝・ヱ
# =====================================

@dataclass
class HierarchicalShare:
    """
    1莠ｺ縺ｮ蜿ょ刈閠・′謖√▽髫主ｱ､繧ｷ繧ｧ繧｢:
    - x: 隧穂ｾ｡轤ｹ
    - y_low : 縺ｼ縺醍判蜒冗畑骰ｵ K_low 縺ｮ Shamir 繧ｷ繧ｧ繧｢
    - y_high: 鬮伜刀雉ｪ逕ｻ蜒冗畑骰ｵ K_high 縺ｮ Shamir 繧ｷ繧ｧ繧｢
    """
    x: int
    y_low: int
    y_high: int


def multisecret_split_int(
    secret_low_int: int,
    secret_high_int: int,
    k_low: int,
    k_high: int,
    n: int,
    p: int = P,
) -> List[HierarchicalShare]:
    """
    謨ｴ謨ｰ secret_low, secret_high 繧・multi-secret・磯嚴螻､・峨〒蛻・淵縲・
    - k_low  繧ｷ繧ｧ繧｢縺ｧ secret_low 繧貞ｾｩ蜈・庄
    - k_high 繧ｷ繧ｧ繧｢縺ｧ secret_high 繧貞ｾｩ蜈・庄・・_high > k_low・・

    謌ｻ繧雁､: HierarchicalShare 縺ｮ繝ｪ繧ｹ繝茨ｼ磯聞縺・n・・
    """
    assert 1 < k_low <= k_high <= n

    shares_low = shamir_split(secret_low_int, k_low, n, p)
    shares_high = shamir_split(secret_high_int, k_high, n, p)

    result: List[HierarchicalShare] = []
    for (x1, y_low), (x2, y_high) in zip(shares_low, shares_high):
        assert x1 == x2
        result.append(HierarchicalShare(x=x1, y_low=y_low, y_high=y_high))
    return result


def reconstruct_low_key(shares: List[HierarchicalShare], k_low: int, p: int = P) -> int:
    """
    縺ｼ縺九＠逕ｻ蜒冗畑骰ｵ K_low 繧貞ｾｩ蜈・・
    shares  : HierarchicalShare 繧・k_low 蛟倶ｻ･荳頑ｸ｡縺・
    謌ｻ繧雁､  : K_low 繧定｡ｨ縺呎紛謨ｰ
    """
    if len(shares) < k_low:
        raise ValueError(f"need at least {k_low} shares to reconstruct low secret")
    subset = [(s.x, s.y_low) for s in shares[:k_low]]
    return shamir_reconstruct(subset, p)


def reconstruct_high_key(shares: List[HierarchicalShare], k_high: int, p: int = P) -> int:
    """
    鬮伜刀雉ｪ逕ｻ蜒冗畑骰ｵ K_high 繧貞ｾｩ蜈・・
    shares  : HierarchicalShare 繧・k_high 蛟倶ｻ･荳頑ｸ｡縺・
    謌ｻ繧雁､  : K_high 繧定｡ｨ縺呎紛謨ｰ
    """
    if len(shares) < k_high:
        raise ValueError(f"need at least {k_high} shares to reconstruct high secret")
    subset = [(s.x, s.y_high) for s in shares[:k_high]]
    return shamir_reconstruct(subset, p)


# ============================
# 3. AES-GCM 證怜捷蛹悶・繝ｫ繝代・
# ============================

def generate_aes_key(num_bytes: int = 32) -> bytes:
    """
    AES-GCM 逕ｨ骰ｵ繧堤函謌撰ｼ医ョ繝輔か繝ｫ繝・256bit・・
    """
    return secrets.token_bytes(num_bytes)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes | None = None) -> Tuple[bytes, bytes]:
    """
    AES-GCM縺ｧ證怜捷蛹・
    謌ｻ繧雁､: (nonce, ciphertext_with_tag)
    """
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)  # 96bit nonce 謗ｨ螂ｨ
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes | None = None) -> bytes:
    """
    AES-GCM縺ｧ蠕ｩ蜿ｷ
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# ============================
# 4. 逕ｻ蜒丞・逅・・繝ｫ繝代・
# ============================

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_low_res_blur(
    img: Image.Image,
    scale: float = 0.01,      # 繧ゅ→繧ゅ→ 0.25 竊・繧ゅ▲縺ｨ蟆上＆縺・
    blur_radius: float = 8.0, # 繧ゅ→繧ゅ→ 2.0 竊・繧ゅ▲縺ｨ蠑ｷ縺・
) -> Image.Image:
    """
    縺九↑繧頑ュ蝣ｱ繧定誠縺ｨ縺励◆菴手ｧ｣蜒丞ｺｦ・九⊂縺九＠逕ｻ蜒上ｒ逕滓・
    """
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # 邵ｮ蟆鞘・蜀肴僑螟ｧ竊貞ｼｷ繧√・縺ｼ縺九＠
    small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    back = small.resize((w, h), Image.Resampling.BILINEAR)
    blurred = back.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return blurred


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Pillow Image -> 繝舌う繝亥・
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """
    繝舌う繝亥・ -> Pillow Image
    """
    buf = io.BytesIO(data)
    return Image.open(buf).convert("RGB")


# ============================
# 5. 繝医ャ繝励Ξ繝吶Ν縺ｮ繝・Δ蜃ｦ逅・
# ============================

def demo_multi_secret_image(
    input_path: str,
    out_dir: str = "out_demo",
    n: int = 5,
    k_low: int = 2,
    k_high: int = 3,
) -> None:
    """
    Multi-secret 繧剃ｽｿ縺｣縺ｦ
    - 菴主刀雉ｪ縺ｼ縺九＠逕ｻ蜒・
    - 鬮伜刀雉ｪ蜈・判蜒・
    繧帝嚴螻､遘伜ｯ・・謨｣縺吶ｋ繝・Δ縲・

    out_dir 蜀・↓證怜捷蛹也判蜒・& 蠕ｩ蜈・判蜒上ｒ菫晏ｭ倥☆繧九・
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) 逕ｻ蜒剰ｪｭ縺ｿ霎ｼ縺ｿ & 縺ｼ縺九＠迚育函謌・
    img = load_image(input_path)
    low_img = make_low_res_blur(img)

    # 2) 逕ｻ蜒上ｒ繝舌う繝亥・縺ｫ
    high_bytes = image_to_bytes(img, fmt="PNG")
    low_bytes = image_to_bytes(low_img, fmt="PNG")

    # 3) 縺昴ｌ縺槭ｌ AES-GCM 縺ｧ證怜捷蛹・
    key_low = generate_aes_key(32)   # 256bit
    key_high = generate_aes_key(32)  # 256bit

    nonce_low, ct_low = aes_gcm_encrypt(key_low, low_bytes, aad=b"low_image")
    nonce_high, ct_high = aes_gcm_encrypt(key_high, high_bytes, aad=b"high_image")

    # ・亥ｮ滄°逕ｨ縺ｪ繧・ct_* 縺ｨ nonce_* 縺ｯ繧ｹ繝医Ξ繝ｼ繧ｸ縺ｫ菫晏ｭ倥☆繧区Φ螳夲ｼ・
    with open(os.path.join(out_dir, "enc_low.bin"), "wb") as f:
        f.write(nonce_low + ct_low)
    with open(os.path.join(out_dir, "enc_high.bin"), "wb") as f:
        f.write(nonce_high + ct_high)

    # 4) 骰ｵ繧呈紛謨ｰ遘伜ｯ・→縺励※ Multi-secret 縺ｧ蛻・淵
    key_low_int = int.from_bytes(key_low, "big")
    key_high_int = int.from_bytes(key_high, "big")

    shares = multisecret_split_int(
        secret_low_int=key_low_int,
        secret_high_int=key_high_int,
        k_low=k_low,
        k_high=k_high,
        n=n,
        p=P,
    )

    # ---- 縺薙％縺九ｉ蠕ｩ蜈・ョ繝｢ ----
    # A) k_low 蛟九□縺第戟縺｣縺ｦ縺・ｋ蝣ｴ蜷・竊・縺ｼ縺醍判蜒上□縺大ｾｩ蜈・〒縺阪ｋ
    subset_for_low = shares[:k_low]
    rec_low_int = reconstruct_low_key(subset_for_low, k_low, p=P)
    rec_key_low = rec_low_int.to_bytes(32, "big")

    # enc_low.bin 縺九ｉ隱ｭ縺ｿ謌ｻ縺・
    with open(os.path.join(out_dir, "enc_low.bin"), "rb") as f:
        enc_data = f.read()
    nonce_l = enc_data[:12]
    ct_l = enc_data[12:]

    dec_low_bytes = aes_gcm_decrypt(rec_key_low, nonce_l, ct_l, aad=b"low_image")
    dec_low_img = bytes_to_image(dec_low_bytes)
    dec_low_img.save(os.path.join(out_dir, "decoded_low_from_k_low.png"))

    # 鬮伜刀雉ｪ蛛ｴ縺ｯ縺ｾ縺蠕ｩ蜈・〒縺阪↑縺・ｼ磯嵯縺後ｏ縺九ｉ縺ｪ縺・ｼ峨・縺ｧ縲√％縺薙〒縺ｯ縺ゅ∴縺ｦ隧ｦ縺輔↑縺・

    # B) k_high 蛟九ｒ髮・ａ縺溷ｴ蜷・竊・縺ｼ縺醍判蜒・+ 鬮伜刀雉ｪ逕ｻ蜒上←縺｡繧峨・骰ｵ繧ょｾｩ蜈・〒縺阪ｋ
    subset_for_high = shares[:k_high]

    rec_high_int = reconstruct_high_key(subset_for_high, k_high, p=P)
    rec_key_high = rec_high_int.to_bytes(32, "big")

    # high 蛛ｴ縺ｮ decrypt
    with open(os.path.join(out_dir, "enc_high.bin"), "rb") as f:
        enc_data_h = f.read()
    nonce_h = enc_data_h[:12]
    ct_h = enc_data_h[12:]

    dec_high_bytes = aes_gcm_decrypt(rec_key_high, nonce_h, ct_h, aad=b"high_image")
    dec_high_img = bytes_to_image(dec_high_bytes)
    dec_high_img.save(os.path.join(out_dir, "decoded_high_from_k_high.png"))

    print("=== Demo finished ===")
    print(f"Shares (first few):")
    for s in shares:
        print(f"x={s.x}, y_low={s.y_low}, y_high={s.y_high}")
    print(f"Decoded images are saved under: {out_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multi_secret_image_demo.py path/to/image.png [out_dir] [k_low] [k_high] [n]")
        sys.exit(1)

    input_img_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "out_demo"

    # 隨ｬ3蠑墓焚莉･髯阪〒 k_low, k_high, n 繧呈欠螳壹〒縺阪ｋ繧医≧縺ｫ
    k_low = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    k_high = int(sys.argv[4]) if len(sys.argv) >= 5 else 3
    n = int(sys.argv[5]) if len(sys.argv) >= 6 else 5

    demo_multi_secret_image(
        input_path=input_img_path,
        out_dir=output_dir,
        n=n,
        k_low=k_low,
        k_high=k_high,
    )
