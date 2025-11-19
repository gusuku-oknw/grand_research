"""
multi_secret_image_demo.py

Multi-secret（階層秘密分散）を使って
- 低品質ぼかし画像
- 高品質元画像
を異なる閾値 k で守るデモスクリプト。

依存:
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
# 1. Shamir Secret Sharing 基本
# ==============================

# 秘密（AES鍵など）を整数にして収める有限体 GF(p)
# 256bit 鍵をそのまま入れたいので 2^256 より大きい素数を使用
P = 2**521 - 1  # 既知のメルセンヌ素数。十分大きい。


def shamir_split(secret: int, k: int, n: int, p: int = P) -> List[Tuple[int, int]]:
    """
    Shamir k-of-n 秘密分散
    secret : 0 <= secret < p の整数
    k      : 復元に必要なシェア数（閾値）
    n      : 総シェア数
    戻り値 : [(x_i, y_i), ...]  i=1..n
    """
    assert 0 <= secret < p
    assert 1 < k <= n

    # 多項式 f(z) = a0 + a1 z + ... + a_{k-1} z^{k-1} (mod p)
    # a0 = secret, 他はランダム
    coeffs = [secret] + [secrets.randbelow(p) for _ in range(k - 1)]

    shares = []
    for x in range(1, n + 1):
        y = 0
        # Horner 法で評価
        for a in reversed(coeffs):
            y = (y * x + a) % p
        shares.append((x, y))
    return shares


def shamir_reconstruct(shares: List[Tuple[int, int]], p: int = P) -> int:
    """
    Shamir 復元（Lagrange 補間）
    shares : [(x_i, y_i), ...] ちょうど k 個（それ以上あっても先頭 k 個を使う想定）
    戻り値 : secret（整数）
    """
    if len(shares) == 0:
        raise ValueError("shares is empty")

    secret = 0
    k = len(shares)
    for j, (xj, yj) in enumerate(shares):
        # λ_j(0) = Π_{m≠j} (0 - x_m) / (x_j - x_m)
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m == j:
                continue
            num = (num * (-xm)) % p
            den = (den * (xj - xm)) % p
        inv_den = pow(den, -1, p)  # 逆元
        lj = num * inv_den % p
        secret = (secret + yj * lj) % p
    return secret


# =====================================
# 2. Multi-secret（階層秘密分散）ラッパ
# =====================================

@dataclass
class HierarchicalShare:
    """
    1人の参加者が持つ階層シェア:
    - x      : 評価点
    - y_low  : ぼけ画像用鍵 K_low の Shamir シェア
    - y_high : 高品質画像用鍵 K_high の Shamir シェア
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
    整数 secret_low, secret_high を Multi-secret（階層方式）で分散する。
    - k_low  シェアで secret_low を復元可能
    - k_high シェアで secret_high を復元可能（k_high > k_low）

    戻り値: HierarchicalShare のリスト（長さ n）
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
    ぼかし画像用鍵 K_low を復元する。
    shares  : HierarchicalShare を k_low 個以上渡す
    戻り値  : K_low を表す整数
    """
    if len(shares) < k_low:
        raise ValueError(f"need at least {k_low} shares to reconstruct low secret")
    subset = [(s.x, s.y_low) for s in shares[:k_low]]
    return shamir_reconstruct(subset, p)


def reconstruct_high_key(shares: List[HierarchicalShare], k_high: int, p: int = P) -> int:
    """
    高品質画像用鍵 K_high を復元する。
    shares  : HierarchicalShare を k_high 個以上渡す
    戻り値  : K_high を表す整数
    """
    if len(shares) < k_high:
        raise ValueError(f"need at least {k_high} shares to reconstruct high secret")
    subset = [(s.x, s.y_high) for s in shares[:k_high]]
    return shamir_reconstruct(subset, p)


# ============================
# 3. AES-GCM 暗号化ヘルパ
# ============================

def generate_aes_key(num_bytes: int = 32) -> bytes:
    """
    AES-GCM 用鍵を生成（デフォルト 256bit）
    """
    return secrets.token_bytes(num_bytes)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes | None = None) -> Tuple[bytes, bytes]:
    """
    AES-GCM で暗号化
    戻り値: (nonce, ciphertext_with_tag)
    """
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)  # 96bit nonce 推奨
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes | None = None) -> bytes:
    """
    AES-GCM で復号
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# ============================
# 4. 画像処理ヘルパ
# ============================

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_low_res_blur(
    img: Image.Image,
    scale: float = 0.01,      # かなり小さく（情報を強く落とす）
    blur_radius: float = 8.0, # 強めのぼかし
) -> Image.Image:
    """
    かなり情報を落とした低解像度＋ぼかし画像を生成
    """
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # 縮小 → 再拡大 → 強めのぼかし
    small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    back = small.resize((w, h), Image.Resampling.BILINEAR)
    blurred = back.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return blurred


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Pillow Image -> バイト列
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """
    バイト列 -> Pillow Image
    """
    buf = io.BytesIO(data)
    return Image.open(buf).convert("RGB")


# ============================
# 5. トップレベルのデモ処理
# ============================

def demo_multi_secret_image(
    input_path: str,
    out_dir: str = "out_demo",
    n: int = 5,
    k_low: int = 2,
    k_high: int = 3,
) -> None:
    """
    Multi-secret を使って
    - 低品質ぼかし画像
    - 高品質元画像
    を階層秘密分散するデモ。

    out_dir 内に暗号化画像 & 復号画像を保存する。
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) 画像読み込み & ぼかし版生成
    img = load_image(input_path)
    low_img = make_low_res_blur(img)

    # 2) 画像をバイト列に
    high_bytes = image_to_bytes(img, fmt="PNG")
    low_bytes = image_to_bytes(low_img, fmt="PNG")

    # 3) それぞれ AES-GCM で暗号化
    key_low = generate_aes_key(32)   # 256bit
    key_high = generate_aes_key(32)  # 256bit

    nonce_low, ct_low = aes_gcm_encrypt(key_low, low_bytes, aad=b"low_image")
    nonce_high, ct_high = aes_gcm_encrypt(key_high, high_bytes, aad=b"high_image")

    # 実運用なら ct_* と nonce_* はストレージに保存する想定
    with open(os.path.join(out_dir, "enc_low.bin"), "wb") as f:
        f.write(nonce_low + ct_low)
    with open(os.path.join(out_dir, "enc_high.bin"), "wb") as f:
        f.write(nonce_high + ct_high)

    # 4) 鍵を整数秘密として Multi-secret で分散
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

    # ---- ここから復元デモ ----
    # A) k_low 個だけ持っている場合 → ぼけ画像だけ復元できる
    subset_for_low = shares[:k_low]
    rec_low_int = reconstruct_low_key(subset_for_low, k_low, p=P)
    rec_key_low = rec_low_int.to_bytes(32, "big")

    # enc_low.bin から読み戻し
    with open(os.path.join(out_dir, "enc_low.bin"), "rb") as f:
        enc_data = f.read()
    nonce_l = enc_data[:12]
    ct_l = enc_data[12:]

    dec_low_bytes = aes_gcm_decrypt(rec_key_low, nonce_l, ct_l, aad=b"low_image")
    dec_low_img = bytes_to_image(dec_low_bytes)
    dec_low_img.save(os.path.join(out_dir, "decoded_low_from_k_low.png"))

    # B) k_high 個を集めた場合 → ぼけ画像 + 高品質画像どちらの鍵も復元できる
    subset_for_high = shares[:k_high]

    rec_high_int = reconstruct_high_key(subset_for_high, k_high, p=P)
    rec_key_high = rec_high_int.to_bytes(32, "big")

    # high 側の decrypt
    with open(os.path.join(out_dir, "enc_high.bin"), "rb") as f:
        enc_data_h = f.read()
    nonce_h = enc_data_h[:12]
    ct_h = enc_data_h[12:]

    dec_high_bytes = aes_gcm_decrypt(rec_key_high, nonce_h, ct_h, aad=b"high_image")
    dec_high_img = bytes_to_image(dec_high_bytes)
    dec_high_img.save(os.path.join(out_dir, "decoded_high_from_k_high.png"))

    print("=== Demo finished ===")
    print("Shares (first few):")
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

    # 第3引数以降で k_low, k_high, n を指定できるように
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
