# secure_search_sis_phash.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from PIL import Image
import os
import secrets

# =========================================================
# 1) pHash (64bit) 実装：32x32 → 2D-DCT → 上位8x8の中央値しきい
# =========================================================

def _dct_matrix(N: int) -> np.ndarray:
    """DCT-II の変換行列（直交正規化）"""
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    M = np.cos(np.pi * (n + 0.5) * k / N)
    M[0, :] *= np.sqrt(1/N)
    M[1:, :] *= np.sqrt(2/N)
    return M

def _dct2(x: np.ndarray) -> np.ndarray:
    """2D-DCT (type-II, orthonormal)"""
    N, M = x.shape
    Cn = _dct_matrix(N)
    Cm = _dct_matrix(M)
    return Cn @ x @ Cm.T

def phash_64(image_path: str) -> int:
    """
    画像ファイルから 64bit pHash を返す（整数）。
    - グレースケール
    - 32x32 にリサイズ
    - 2D-DCT → 左上 8x8（DC含む）からDCを除いた64要素を中央値しきい
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((32, 32), Image.BILINEAR)
    X = np.asarray(img, dtype=np.float32)
    D = _dct2(X)

    # 左上 8x8 を取得
    dct_low = D[:8, :8].copy()
    # DCを除いた残り64要素を使う（本来は 8x8=64 で DC除外→63だが、
    # 一般的pHashは 8x8 の中央値しきい（DC含）を用いる実装も多い。
    # ここでは“DC除外”で robust な方に寄せる → 64bitにするため DC以外の上位成分を1つ足す。
    # シンプルに 8x8 を中央値しきい（DC含）で 64bit とします。
    flat = dct_low.flatten()
    med = np.median(flat[1:])  # DC除外で中央値計算
    bits = (flat > med).astype(np.uint8)

    # 上位64bitを整数へ（8x8=64要素をそのまま利用）
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def hash64_to_bytes(h: int) -> List[int]:
    """64bit整数 → 8バイト（0..255のリスト）"""
    return [(h >> (8*(7-i))) & 0xFF for i in range(8)]

def bytes_to_hash64(bs: Iterable[int]) -> int:
    bs = list(bs)
    assert len(bs) == 8
    h = 0
    for b in bs:
        h = (h << 8) | (b & 0xFF)
    return h

# =========================================================
# 2) Shamir Secret Sharing（GF(257)）実装：k-of-n
#    - 各“バイト位置”ごとに多項式を作る素直実装（分かりやすさ優先）
# =========================================================

P = 257  # 素数。画素/バイト(0..255)をそのまま要素にできる。

def _mod_inv(a: int, p: int = P) -> int:
    return pow(a % p, p-2, p)

def _lagrange_basis_at_zero(xs: List[int], j: int, p: int = P) -> int:
    """
    ラグランジュ基底 l_j(0) = Π_{m≠j} (0 - x_m) / (x_j - x_m)  (mod p)
    xs: 使用する x のリスト（1..n の一部）
    j : 基底のインデックス（xs[j] を使う）
    """
    xj = xs[j] % p
    num, den = 1, 1
    for m, xm in enumerate(xs):
        xm %= p
        if m == j:
            continue
        num = (num * (-xm)) % p
        den = (den * (xj - xm)) % p
    return (num * _mod_inv(den, p)) % p

def shamir_share_bytes(secret_bytes: List[int], k: int, n: int, seed: Optional[int] = None
                      ) -> Dict[int, List[int]]:
    """
    秘密（バイト列）を Shamir で k-of-n に分散。
    戻り値: {x(=1..n): share_bytes(各バイト位置ごとに y=f(x)) }
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        def rand_byte():
            return int(rng.integers(0, P))
    else:
        def rand_byte():
            return secrets.randbelow(P)

    L = len(secret_bytes)
    # 係数: 各バイト位置 i について、k-1 個の乱数係数 a1..a_{k-1}
    coeffs = [[rand_byte() for _ in range(k-1)] for _ in range(L)]

    shares: Dict[int, List[int]] = {x: [0]*L for x in range(1, n+1)}
    for i, s in enumerate(secret_bytes):
        s %= P
        a = coeffs[i]
        # 評価
        for x in range(1, n+1):
            y = s
            t = 1
            for ai in a:
                t = (t * x) % P
                y = (y + ai * t) % P
            shares[x][i] = y
    return shares

def shamir_recover_bytes(subshares: Dict[int, List[int]]) -> List[int]:
    """
    任意の k 枚のシェアから秘密バイト列を復元。
    subshares: {x: share_bytes}
    """
    xs = sorted(subshares.keys())  # 使用する x のリスト
    k = len(xs)
    L = len(next(iter(subshares.values())))
    # 事前に基底係数 l_j(0) を計算
    Ljs = [_lagrange_basis_at_zero(xs, j) for j in range(k)]
    rec = [0]*L
    for i in range(L):
        s = 0
        for j, x in enumerate(xs):
            y = subshares[x][i] % P
            s = (s + y * Ljs[j]) % P
        # もともと 0..255 のはず
        if s == 256:  # ノイズがなければ出ない想定。健全性のためクリップ。
            s = 0
        rec[i] = int(s)
    return rec

# =========================================================
# 3) 検索可能な秘密画像格納：インデックス
#    - add(): pHash→SIS分散→「サーバごとの保管庫」に格納
#    - query(): k台のサーバが合意したら復元して距離計算
# =========================================================

@dataclass
class StoredShare:
    """サーバ上に置かれる1画像あたりのシェア（バイト列）"""
    image_id: str
    share_bytes: List[int]  # 8 バイト（pHash 64bitを8バイト化）

class SecureSearchIndex:
    """
    (k, n) Shamir + pHash(64bit) による検索可能“秘密”画像インデックス（概念実装）
    - server_shares[x][image_id] = StoredShare(...)
    """
    def __init__(self, k: int = 3, n: int = 5):
        assert 2 <= k <= n <= 20
        self.k = k
        self.n = n
        self.server_shares: Dict[int, Dict[str, StoredShare]] = {x: {} for x in range(1, n+1)}
        self._images: Dict[str, str] = {}  # image_id -> path（デモ用）

    def add_image(self, image_id: str, image_path: str) -> int:
        """
        画像を登録：pHash→SIS分散→各サーバへ保存。
        return: 登録した pHash(64bit)（デモ用：本来は破棄推奨）
        """
        h = phash_64(image_path)
        bs = hash64_to_bytes(h)  # 8要素
        shares = shamir_share_bytes(bs, self.k, self.n)
        for x, sbytes in shares.items():
            self.server_shares[x][image_id] = StoredShare(image_id=image_id, share_bytes=sbytes)
        self._images[image_id] = image_path
        return h

    def list_servers(self) -> List[int]:
        return sorted(self.server_shares.keys())

    def _reconstruct_hash_from_servers(self, image_id: str, servers: Iterable[int]) -> Optional[int]:
        """
        指定サーバ集合(>=k)から、image_id の pHash を復元。
        1台でも該当シェアが無い場合は None。
        """
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k:
            return None
        sub: Dict[int, List[int]] = {}
        for x in xs:
            item = self.server_shares.get(x, {}).get(image_id)
            if item is None:
                return None
            sub[x] = item.share_bytes
        bs = shamir_recover_bytes(sub)
        return bytes_to_hash64(bs)

    def query(self,
              query_image_path: str,
              servers_for_query: Iterable[int],
              topk: int = 10,
              max_hamming: Optional[int] = None
              ) -> List[Tuple[str, int]]:
        """
        クエリ画像で検索。
        - servers_for_query: 距離計算に協力するサーバ（>=k）。この集合が**合意**して初めて距離計算が可能。
        - 戻り値: [(image_id, hamming_distance), ...] を距離昇順で最大 topk 件
        """
        xs = sorted(set(int(s) for s in servers_for_query))
        if len(xs) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to compute distances.")

        hq = phash_64(query_image_path)
        results: List[Tuple[str, int]] = []
        # すべての登録画像について（デモ：実運用はインデックス圧縮/候補選別が必要）
        image_ids = list(self._images.keys())
        for image_id in image_ids:
            h_db = self._reconstruct_hash_from_servers(image_id, xs)
            if h_db is None:
                continue  # 当該サーバ集合では復元不可
            d = hamming_distance64(hq, h_db)
            if (max_hamming is None) or (d <= max_hamming):
                results.append((image_id, d))
        # 近い順に topk
        results.sort(key=lambda t: t[1])
        return results[:topk]
# =========================================================
# 4) デモ（読みやすいログ版＋改ざんテストD 追加）
# =========================================================

if __name__ == "__main__":
    """
    使い方（例）:
        python secure_search_sis_phash.py
    事前に ./images に jpg/png を数枚置いてください。
    """
    import glob
    import time

    def hr():
        print("-" * 72)

    # (k, n) を決める
    k, n = 3, 5
    index = SecureSearchIndex(k=k, n=n)

    # DBに登録
    os.makedirs("images", exist_ok=True)
    db_images = sorted(glob.glob("images/*"))

    print(f"[CONFIG] k={k}, n={n}")
    print(f"[INFO]   images dir = ./images  (found {len(db_images)} files)")
    hr()

    if len(db_images) == 0:
        print("Put some images into ./images/ then run again.")
        exit(0)

    t0 = time.perf_counter()
    for i, path in enumerate(db_images):
        img_id = f"img_{i:04d}"
        h = index.add_image(img_id, path)
        print(f"[ADD] {img_id:<12} file={os.path.basename(path):<40} pHash=0x{h:016x}")
    t1 = time.perf_counter()
    print(f"[DONE]   registered {len(db_images)} images in {t1 - t0:.3f}s")
    hr()

    # ===== ケースA: ちょうど k 台のサーバが合意（成功ケース） =====
    servers_ok = index.list_servers()[:k]   # 例: 1..k 台
    query_path = db_images[0]
    print(f"[QUERY-A] query={os.path.basename(query_path)}")
    print(f"[QUERY-A] servers used (k met) = {servers_ok}")
    t2 = time.perf_counter()
    hits = index.query(query_path, servers_for_query=servers_ok, topk=10, max_hamming=10)
    t3 = time.perf_counter()

    if len(hits) == 0:
        print("[RESULT-A] no matches within max_hamming=10")
    else:
        print("[RESULT-A] top matches (sorted by Hamming distance)")
        print(f"  {'image_id':<12} {'Hamming':<7}")
        for img_id, dist in hits:
            print(f"  {img_id:<12} {dist:<7d}")
    print(f"[TIME-A]  {t3 - t2:.3f}s (reconstruct + distance)")
    hr()

    # ===== ケースB: k-1 台のサーバしかない（合意不足で失敗） =====
    servers_insufficient = index.list_servers()[:max(1, k-1)]
    print(f"[QUERY-B] query={os.path.basename(query_path)}")
    print(f"[QUERY-B] servers used (k NOT met) = {servers_insufficient}")
    try:
        _ = index.query(query_path, servers_for_query=servers_insufficient, topk=5, max_hamming=10)
        print("[RESULT-B] (unexpected) query succeeded")
    except ValueError as e:
        print(f"[RESULT-B] FAILED: {e}")
    hr()

    # ===== ケースC: シェア欠落（1サーバの1画像シェアを削除） =====
    victim_img = "img_0000"
    kill_server = servers_ok[0]
    popped = index.server_shares[kill_server].pop(victim_img, None)
    print(f"[QUERY-C] simulate missing share: removed {victim_img} from server {kill_server}")
    print(f"[QUERY-C] servers used (k met)   = {servers_ok}")

    hits_c = index.query(query_path, servers_for_query=servers_ok, topk=10, max_hamming=10)
    listed = [img for img, _ in hits_c]
    if victim_img in listed:
        print("[RESULT-C] (unexpected) victim still present in results")
    else:
        print("[RESULT-C] victim image is skipped due to missing share (as expected)")
    print(f"[INFO]    results returned for {len(hits_c)} images (excluding missing-share targets)")
    hr()

    # 復元テスト: どの画像が現在のサーバ集合で復元できるか
    rec_ok, rec_ng = [], []
    for img_id in index._images.keys():
        h = index._reconstruct_hash_from_servers(img_id, servers_ok)
        (rec_ok if h is not None else rec_ng).append(img_id)
    print(f"[RECAP] reconstructable={len(rec_ok)} not_reconstructable={len(rec_ng)}")
    if rec_ng:
        print("        not reconstructable:", ", ".join(rec_ng))
    hr()

    # ===== ケースD: シェア改ざん（tampering）をシミュレート =====
    # 復元可能な画像の中から改ざん対象を選ぶ（Cで欠落させた victim_img は除外）
    victim_tamper = next((i for i in rec_ok if i != victim_img), rec_ok[0] if rec_ok else None)
    if victim_tamper is None:
        print("[QUERY-D] skip: no reconstructable image available for tamper test")
    else:
        print(f"[QUERY-D] simulate tampering: flip 1 byte in share of {victim_tamper} on server {servers_ok[0]}")
        # 改ざん前の復元ハッシュ
        h_before = index._reconstruct_hash_from_servers(victim_tamper, servers_ok)
        # 1バイト改ざん（GF(257) 上で +1）
        share = index.server_shares[servers_ok[0]][victim_tamper]
        share.share_bytes[0] = (share.share_bytes[0] + 1) % P
        # 改ざん後の復元ハッシュ
        h_after = index._reconstruct_hash_from_servers(victim_tamper, servers_ok)
        delta = hamming_distance64(h_before, h_after)
        print(f"[RESULT-D] reconstructed hash changed? {'YES' if h_before != h_after else 'NO'}  (Hamming(before,after)={delta})")

        # 改ざん後の状態で検索して、対象画像の距離がどう変わるかを確認（max_hammingを広めに）
        hits_d = index.query(query_path, servers_for_query=servers_ok, topk=10, max_hamming=64)
        d_victim = next((d for img, d in hits_d if img == victim_tamper), None)
        if d_victim is not None:
            print(f"[RESULT-D] victim {victim_tamper} now appears with Hamming={d_victim}")
        else:
            print(f"[RESULT-D] victim {victim_tamper} filtered out by max_hamming")
    hr()

    # ===== 補足説明 =====
    print("[GUIDE]")
    print("  - Hamming 距離: 0 が完全一致。小さいほど類似。")
    print("  - ケースA: k台の合意 → pHash 復元 → XOR+popcount で距離算出（通常運用）。")
    print("  - ケースB: k未満 → 復元不可（例外で明示）。")
    print("  - ケースC: シェア欠落 → 当該画像は距離計算に参加できず、結果一覧から自然に除外。")
    print("  - ケースD: シェア改ざん → 復元ハッシュが変化（ほぼ確実）。距離も変化して検知できる。")
