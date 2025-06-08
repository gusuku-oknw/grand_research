# FlexShareImageDealerFree.py
# Dealer-Free Dynamic Threshold Secret Sharing + AES-GCM Image Encryption

import random
import secrets
from sympy import mod_inverse
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image
from pathlib import Path

# -------------------------------------------------------
# 1) Safe-prime group parameters (RFC 3526 2048-bit MODP Group 14)
# -------------------------------------------------------
p = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A63A3620FFFFFFFFFFFFFFFF", 16
)
g = 2
q = (p - 1) // 2  # Group order for generator g

# -------------------------------------------------------
# 2) Protocol parameters
# -------------------------------------------------------
N = 5        # Number of participants
T = 4        # Initial threshold
T_NEW = 2    # Dynamic lowered threshold
Y0 = 123_456 # FlexShare constant
rand = random.SystemRandom()

# -------------------------------------------------------
# 3) Feldman VSS functions
# -------------------------------------------------------
def feldman_commit(poly: list[int]) -> list[int]:
    """
    Compute Feldman commitments C_k = g^{a_k} mod p
    for polynomial coefficients a_k.
    """
    return [pow(g, a, p) for a in poly]


def eval_poly(poly: list[int], x: int, mod: int) -> int:
    """
    Evaluate polynomial at x over modulus mod.
    """
    y = 0
    for k, a in enumerate(poly):
        y = (y + a * pow(x, k, mod)) % mod
    return y


def feldman_verify(i: int, share: int, commit: list[int]) -> bool:
    """
    Verify g^{share} == Π C_k^{i^k} mod p
    """
    lhs = pow(g, share, p)
    rhs = 1
    for k, Ck in enumerate(commit):
        rhs = (rhs * pow(Ck, pow(i, k), p)) % p
    return lhs == rhs

# -------------------------------------------------------
# 4) Dealer-Free DKG (Joint-Feldman simulation)
# -------------------------------------------------------
def dkg() -> dict[int, int]:
    """
    Simulate dealer-free DKG by having each of N participants
    act as a Feldman VSS dealer for their own random secret.
    Final share s_i is sum of shares from all runs, in Z_q.
    Returns dictionary of final shares in Z_q.
    """
    shares_q: dict[int, int] = {i: 0 for i in range(1, N+1)}
    for _ in range(N):
        # Each participant picks random secret s_i in Z_q
        poly = [rand.randrange(q) for _ in range(T)]       # degree T-1
        commit = feldman_commit(poly)
        for j in range(1, N+1):
            s_ij = eval_poly(poly, j, q)
            assert feldman_verify(j, s_ij, commit), "VSS verification failed"
            shares_q[j] = (shares_q[j] + s_ij) % q
    return shares_q

# -------------------------------------------------------
# 5) FlexShare dynamic threshold helpers
# -------------------------------------------------------
def add_auxiliary(shares_q: dict[int,int]) -> dict[int, tuple[int,int]]:
    """
    Add auxiliary polynomial shares for dynamic threshold.
    Returns mapping i -> (u_i, v_i).
    u_i = base share in Z_q, v_i = aux share in Z_q.
    """
    aux_poly = [rand.randrange(q) for _ in range(T)]
    v = {i: eval_poly(aux_poly, i, q) for i in shares_q}
    return {i: (shares_q[i], v[i]) for i in shares_q}


def lambda_lagrange(xs: list[int], ys: list[int], x0: int, mod: int) -> int:
    """
    Compute Lagrange interpolation at x0 for points (xs, ys) over Z_mod.
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
    Using shares (u_i, v_i), compute new secret in Z_q
    as s' = Σ Lagrange(u_i + Y0 * v_i). Returns 32-byte key.
    """
    xs = indices
    ys = [(uv[i][0] + Y0 * uv[i][1]) % q for i in indices]
    secret_q = lambda_lagrange(xs, ys, 0, q)
    # Convert to 32-byte big-endian
    b = secret_q.to_bytes((secret_q.bit_length() + 7)//8, 'big')
    return b.rjust(32, b'\x00')

# -------------------------------------------------------
# 6) AES-GCM helpers
# -------------------------------------------------------
def aes_gcm_encrypt(data: bytes, key: bytes) -> bytes:
    """
    Returns blob = nonce(12) || tag(16) || ciphertext.
    """
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(data)
    return nonce + tag + ct


def aes_gcm_decrypt(blob: bytes, key: bytes) -> bytes:
    """
    Decrypt blob = nonce||tag||ciphertext and verify.
    """
    nonce, tag, ct = blob[:12], blob[12:28], blob[28:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)

# -------------------------------------------------------
# 7) Main demo
# -------------------------------------------------------
if __name__ == "__main__":
    # 7.1 Dealer-free DKG
    base_shares = dkg()
    print("Base shares in Z_q:", base_shares)

    # 7.2 Add auxiliary shares for dynamic threshold
    uv_pairs = add_auxiliary(base_shares)
    print("(u_i, v_i) pairs:", uv_pairs)

    # 7.3 Image encryption
    img = Image.open("input.png").convert("RGB")
    raw = img.tobytes()
    key_bytes = secrets.token_bytes(32)
    blob = aes_gcm_encrypt(raw, key_bytes)
    Path("cipher.bin").write_bytes(blob)
    # Noise preview
    Image.frombytes(img.mode, img.size, blob[28:]).save("preview.bmp")
    print("Encrypted image preview saved.")

    # 7.4 Replace u_i with actual secret share
    secret_int = int.from_bytes(key_bytes, "big") % q
    for i in uv_pairs:
        uv_pairs[i] = (secret_int, uv_pairs[i][1])

    # 7.5 Reduce threshold to T_NEW=2 and reconstruct key
    rec_key = reduce_threshold(uv_pairs, list(range(1, T_NEW+1)))
    assert rec_key == key_bytes, "Key reconstruction failed"

    # 7.6 Decrypt and save
    plain = aes_gcm_decrypt(blob, rec_key)
    Image.frombytes(img.mode, img.size, plain).save("decrypted.png")
    print("Decryption successful!")
