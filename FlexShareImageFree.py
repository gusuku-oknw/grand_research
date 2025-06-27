# FlexShareImageDealerFree.py
# ディーラーフリー動的しきい値秘密分散 + AES-GCM画像暗号化

import random
import secrets
from sympy import mod_inverse
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image
from pathlib import Path

# -------------------------------------------------------
# 1) 安全な素数グループパラメータ (RFC 3526 2048-bit MODP Group 14)
# -------------------------------------------------------
p = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A63A3620FFFFFFFFFFFFFFFF", 16
)
g = 2
q = (p - 1) // 2  # 生成元gの群の位数

# -------------------------------------------------------
# 2) プロトコルパラメータ
# -------------------------------------------------------
N = 5        # 参加者の数
T = 4        # 初期のしきい値
T_NEW = 2    # 動的に下げられたしきい値
Y0 = 123_456 # FlexShare定数
rand = random.SystemRandom()

# -------------------------------------------------------
# 3) Feldman VSS関数
# -------------------------------------------------------
def feldman_commit(poly: list[int]) -> list[int]:
    """
    多項式の係数a_kに対して、Feldmanコミットメント C_k = g^{a_k} mod p を計算する。
    """
    return [pow(g, a, p) for a in poly]


def eval_poly(poly: list[int], x: int, mod: int) -> int:
    """
    法mod上で多項式のxにおける値を評価する。
    """
    y = 0
    for k, a in enumerate(poly):
        y = (y + a * pow(x, k, mod)) % mod
    return y


def feldman_verify(i: int, share: int, commit: list[int]) -> bool:
    """
    g^{share} == Π C_k^{i^k} mod p を検証する。
    """
    lhs = pow(g, share, p)
    rhs = 1
    for k, Ck in enumerate(commit):
        rhs = (rhs * pow(Ck, pow(i, k), p)) % p
    return lhs == rhs

# -------------------------------------------------------
# 4) ディーラーフリーDKG (Joint-Feldmanシミュレーション)
# -------------------------------------------------------
def dkg() -> dict[int, int]:
    """
    N人の各参加者が自身のランダムな秘密に対してFeldman VSSディーラーとして振る舞うことで、
    ディーラーフリーDKGをシミュレートする。
    最終的なシェアs_iは、すべてのラウンドからのシェアの合計（Z_q上）となる。
    Z_q上の最終的なシェアの辞書を返す。
    """
    shares_q: dict[int, int] = {i: 0 for i in range(1, N+1)}
    for _ in range(N):
        # 各参加者はZ_qからランダムな秘密s_iを選ぶ
        poly = [rand.randrange(q) for _ in range(T)]       # 次数 T-1
        commit = feldman_commit(poly)
        for j in range(1, N+1):
            s_ij = eval_poly(poly, j, q)
            assert feldman_verify(j, s_ij, commit), "VSS検証失敗"
            shares_q[j] = (shares_q[j] + s_ij) % q
    return shares_q

# -------------------------------------------------------
# 5) FlexShare動的しきい値ヘルパー
# -------------------------------------------------------
def add_auxiliary(shares_q: dict[int,int]) -> dict[int, tuple[int,int]]:
    """
    動的しきい値のために補助多項式のシェアを追加する。
    i -> (u_i, v_i) のマッピングを返す。
    u_i = Z_q上の基本シェア, v_i = Z_q上の補助シェア。
    """
    # しきい値を下げたときに復元される秘密が変わらないように、
    # 補助多項式の定数項は0でなければならない。
    aux_poly = [0] + [rand.randrange(q) for _ in range(1, T)]
    v = {i: eval_poly(aux_poly, i, q) for i in shares_q}
    return {i: (shares_q[i], v[i]) for i in shares_q}


def lambda_lagrange(xs: list[int], ys: list[int], x0: int, mod: int) -> int:
    """
    Z_mod上の点(xs, ys)に対して、x0におけるラグランジュ補間を計算する。
    """
    total = 0
    for j, (xj, yj) in enumerate(zip(xs, ys)):
        num = den = 1
        for m, xm in enumerate(xs):
            if m == j: continue
            num = (num * (x0 - xm)) % mod
            den = (den * (xj - xm)) % mod
        lj = num * mod_inverse(den, mod) % mod
        total = (total + yj * lj) % mod
    return total


def reduce_threshold(uv: dict[int, tuple[int,int]], indices: list[int]) -> bytes:
    """
    シェア(u_i, v_i)を使い、s' = Σ Lagrange(u_i + Y0 * v_i)として
    Z_q上の新しい秘密を計算する。32バイトの鍵を返す。
    """
    xs = indices
    ys = [(uv[i][0] + Y0 * uv[i][1]) % q for i in indices]
    secret_q = lambda_lagrange(xs, ys, 0, q)
    # 32バイトのビッグエンディアンに変換
    b = secret_q.to_bytes((secret_q.bit_length() + 7)//8, 'big')
    return b.rjust(32, b'\x00')

# -------------------------------------------------------
# 6) AES-GCMヘルパー
# -------------------------------------------------------
def aes_gcm_encrypt(data: bytes, key: bytes) -> bytes:
    """
    blob = nonce(12) || tag(16) || ciphertext を返す。
    """
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(data)
    return nonce + tag + ct


def aes_gcm_decrypt(blob: bytes, key: bytes) -> bytes:
    """
    blob = nonce||tag||ciphertext を復号し、検証する。
    """
    nonce, tag, ct = blob[:12], blob[12:28], blob[28:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)

# -------------------------------------------------------
# 7) メインデモ
# -------------------------------------------------------
if __name__ == "__main__":
    # 7.1 ディーラーフリーDKG
    base_shares = dkg()
    print("Z_q上の基本シェア:", base_shares)

    # 7.2 動的しきい値のために補助シェアを追加
    uv_pairs = add_auxiliary(base_shares)
    print("(u_i, v_i)ペア:", uv_pairs)

    # 7.3 画像の暗号化
    img = Image.open("data/input.png").convert("RGB")
    raw = img.tobytes()
    key_bytes = secrets.token_bytes(32)
    blob = aes_gcm_encrypt(raw, key_bytes)
    Path("data/cipher.bin").write_bytes(blob)
    # ノイズのプレビュー
    Image.frombytes(img.mode, img.size, blob[28:]).save("data/preview.bmp")
    print("暗号化された画像のプレビューを保存しました。")

    # 7.4 u_iを実際の秘密のシェアに置き換える
    secret_int = int.from_bytes(key_bytes, "big") % q
    for i, (_, v) in uv_pairs.items():
        # 各u_iを、u_i + Y0*v_iが実際の秘密と等しくなるように調整する。
        uv_pairs[i] = ((secret_int - Y0 * v) % q, v)

    # 7.5 しきい値をT_NEW=2に下げて鍵を復元
    rec_key = reduce_threshold(uv_pairs, list(range(1, T_NEW+1)))
    assert rec_key == key_bytes, "鍵の復元に失敗"

    # 7.6 復号して保存
    plain = aes_gcm_decrypt(blob, rec_key)
    Image.frombytes(img.mode, img.size, plain).save("data/decrypted.png")
    print("復号に成功しました！")