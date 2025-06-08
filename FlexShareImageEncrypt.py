# FlexShareImageEncrypt.py

import random
import secrets
from pathlib import Path
from sympy import mod_inverse
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image

# -------------------------------------------------------
# 1) 共有・有限体パラメータ
# -------------------------------------------------------
N_PARTICIPANTS = 5
T_INITIAL      = 4
T_NEW          = 2
P_PRIME = int("208351617316091241234326746312124448251235562226470460481")
Y0      = 123_456

# -------------------------------------------------------
# 2) FlexShare: シェア生成
# -------------------------------------------------------
def generate_shares(secret: int,
                    n: int = N_PARTICIPANTS,
                    t: int = T_INITIAL,
                    t_prime: int = T_NEW,
                    y0: int = Y0,
                    p: int = P_PRIME):
    a = {0: secret, **{k: random.randrange(0, p) for k in range(1, t_prime)}}
    b = {k: random.randrange(0, p) for k in range(t_prime, t)}
    shares = {}
    for i in range(1, n+1):
        sum_a = sum(a[k] * pow(i, k, p) for k in range(t_prime)) % p
        sum_b = sum(b[k] * pow(i, k, p) for k in range(t_prime, t)) % p
        u_i   = (sum_a - y0*sum_b) % p
        v_i   =  sum_b
        shares[i] = (u_i, v_i)
    return shares

# -------------------------------------------------------
# 3) Lagrange 補間 → 鍵再構成
# -------------------------------------------------------
def lagrange_univariate(xs, ys, x0, p):
    total = 0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num = den = 1
        for j, xj in enumerate(xs):
            if i == j: continue
            num = num * (x0 - xj) % p
            den = den * (xi - xj) % p
        total = (total + yi * num * mod_inverse(den, p)) % p
    return total

def reconstruct_key(shares, idx, new_threshold=True):
    xs = idx
    if new_threshold:
        ys = [(shares[i][0] + shares[i][1]*Y0) % P_PRIME for i in idx]
    else:
        ys = [shares[i][0] for i in idx]
    secret_int = lagrange_univariate(xs, ys, 0, P_PRIME)
    return secret_int.to_bytes(32, "big")

# -------------------------------------------------------
# 4) AES-GCM ヘルパ (必ず 12-byte nonce を利用)
# -------------------------------------------------------
def aes_gcm_encrypt(plain: bytes, key: bytes):
    """
    12-byte nonce を自前で生成して AES-GCM 暗号化。
    戻り値: (nonce:12B, tag:16B, ciphertext)
    """
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plain)
    return nonce, tag, ct

def aes_gcm_decrypt(nonce: bytes, tag: bytes, ct: bytes, key: bytes):
    """
    nonce(12B)、tag(16B)、ciphertext を受け取り AES-GCM 復号。
    """
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)

# -------------------------------------------------------
# 5) 画像暗号化＋シェア生成
# -------------------------------------------------------
def encrypt_image_with_flexshare(src_png: str,
                                 out_bin: str,
                                 meta_path: str,
                                 preview_bmp: str):
    # ① 元画像読込＆生データ取得
    img   = Image.open(src_png).convert("RGB")
    plain = img.tobytes()

    # ② 乱数鍵＋AES-GCM 暗号化
    key   = secrets.token_bytes(32)
    nonce, tag, ct = aes_gcm_encrypt(plain, key)

    # ③ 暗号文を生バイナリとして保存
    Path(out_bin).write_bytes(ct)

    # ④ nonce+tag をメタデータとして保存 (28B 固定)
    Path(meta_path).write_bytes(nonce + tag)

    # ⑤ ノイズ確認用 BMP プレビュー（復号には使わない）
    Image.frombytes(img.mode, img.size, ct).save(preview_bmp, format="BMP")

    # ⑥ FlexShare 用シェア生成
    secret_int = int.from_bytes(key, "big")
    shares = generate_shares(secret_int)
    return shares, key, img.size, img.mode

# -------------------------------------------------------
# 6) 画像復号
# -------------------------------------------------------
def decrypt_image_with_flexshare(bin_path: str,
                                 meta_path: str,
                                 dst_png: str,
                                 key: bytes,
                                 size, mode):
    # ① 暗号文とメタデータ読込
    ct   = Path(bin_path).read_bytes()
    meta = Path(meta_path).read_bytes()
    nonce, tag = meta[:12], meta[12:28]

    # ② 復号＆PNG 出力
    plain = aes_gcm_decrypt(nonce, tag, ct, key)
    Image.frombytes(mode, size, plain).save(dst_png, format="PNG")

# -------------------------------------------------------
# 7) デモ
# -------------------------------------------------------
if __name__ == "__main__":
    shares, key_bytes, img_size, img_mode = encrypt_image_with_flexshare(
        "input.png",   # 元画像
        "cipher.bin",  # 生バイナリ暗号文
        "meta.bin",    # nonce+tag
        "preview.bmp"  # ノイズ確認用
    )
    print("Secret(int) =", int.from_bytes(key_bytes, "big"))
    print("Shares:")
    for i, (u, v) in shares.items():
        print(f"  P{i}: u={u}, v={v}")

    # 閾値 t'=2 で鍵再構成
    rec_key = reconstruct_key(shares, [1,2], new_threshold=True)

    # 復号
    decrypt_image_with_flexshare(
        "cipher.bin",
        "meta.bin",
        "decrypted.png",
        rec_key,
        img_size,
        img_mode
    )
    print("復号成功！")
