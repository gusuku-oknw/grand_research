import random
import secrets
from sympy import mod_inverse
from PIL import Image

# -------------------------------------------------------
# Secret sharing parameters
# -------------------------------------------------------
N_PARTICIPANTS = 5  # number of participants
T_INITIAL = 4       # initial threshold
T_NEW = 2           # new threshold after update
P_PRIME = int(
    "208351617316091241234326746312124448251235562226470491514186331217050270460481"
)
Y0 = 123456

# -------------------------------------------------------
# Polynomial share generation based on FlexShare
# -------------------------------------------------------

def generate_shares(secret: int, n=N_PARTICIPANTS, t=T_INITIAL, t_prime=T_NEW, y0=Y0, p=P_PRIME):
    """Generate shares (u_i, v_i) for the given secret value."""
    # a_k for k < t_prime
    a = {}
    for k in range(t_prime):
        if k == 0:
            a[k] = secret
        else:
            a[k] = random.randrange(0, p)

    # b_k for t_prime <= k < t
    b = {}
    for k in range(t_prime, t):
        b[k] = random.randrange(0, p)

    shares = {}
    for i in range(1, n + 1):
        sum_a = sum(a[k] * pow(i, k, p) for k in range(t_prime)) % p
        sum_b_y0 = sum(b[k] * pow(i, k, p) for k in range(t_prime, t)) % p
        u_i = (sum_a - (y0 * sum_b_y0) % p) % p
        v_i = sum(b[k] * pow(i, k, p) for k in range(t_prime, t)) % p
        shares[i] = (u_i, v_i)
    return shares


def lagrange_univariate(x_list, y_list, x0, p):
    total = 0
    k = len(x_list)
    for i in range(k):
        xi, yi = x_list[i], y_list[i]
        num, den = 1, 1
        for j in range(k):
            if i == j:
                continue
            xj = x_list[j]
            num = (num * (x0 - xj)) % p
            den = (den * (xi - xj)) % p
        li = (num * mod_inverse(den, p)) % p
        total = (total + yi * li) % p
    return total

# -------------------------------------------------------
# Simple XOR-based image encryption
# -------------------------------------------------------

def xor_encrypt(data: bytes, key: bytes) -> bytes:
    encrypted = bytearray(len(data))
    for i, b in enumerate(data):
        encrypted[i] = b ^ key[i % len(key)]
    return bytes(encrypted)


# -------------------------------------------------------
# Demonstration for image encryption with FlexShare
# -------------------------------------------------------

def encrypt_image_with_flexshare(input_path: str, encrypted_path: str):
    """Encrypt an image and create key shares using FlexShare."""
    # Load image and obtain raw data
    img = Image.open(input_path)
    data = img.tobytes()

    # Generate random key (16 bytes)
    key_bytes = secrets.token_bytes(16)
    secret_int = int.from_bytes(key_bytes, "big")

    # Encrypt image data
    encrypted_data = xor_encrypt(data, key_bytes)
    enc_img = Image.frombytes(img.mode, img.size, encrypted_data)
    enc_img.save(encrypted_path)

    # Share the key
    shares = generate_shares(secret_int)

    return shares, secret_int


def reconstruct_key_from_shares(shares_dict, indices, use_new_threshold=False):
    x_list = indices
    if use_new_threshold:
        y_list = [(shares_dict[i][0] + shares_dict[i][1] * Y0) % P_PRIME for i in indices]
    else:
        y_list = [shares_dict[i][0] for i in indices]
    return lagrange_univariate(x_list, y_list, 0, P_PRIME)


if __name__ == "__main__":
    # Example usage
    INPUT_IMAGE = "input.png"       # Path to input image
    ENCRYPTED_IMAGE = "encrypted.png"  # Output encrypted image

    # Encrypt image and obtain shares
    shares, secret = encrypt_image_with_flexshare(INPUT_IMAGE, ENCRYPTED_IMAGE)
    print(f"Encryption key (secret) = {secret}")
    print("Shares:")
    for idx, pair in shares.items():
        print(f"Participant {idx}: {pair}")

    # Example reconstruction using the new threshold
    indices = list(range(1, T_NEW + 1))
    recovered = reconstruct_key_from_shares(shares, indices, use_new_threshold=True)
    recovered_key = recovered.to_bytes(16, "big")
    print(f"Recovered key = {recovered}")

    # Decrypt the image to verify
    enc_img = Image.open(ENCRYPTED_IMAGE)
    decrypted_data = xor_encrypt(enc_img.tobytes(), recovered_key)
    dec_img = Image.frombytes(enc_img.mode, enc_img.size, decrypted_data)
    dec_img.save("decrypted.png")
