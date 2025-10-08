# searchable_sis_phash_selective_demo.py
# -*- coding: utf-8 -*-
"""
Searchable SIS with pHash, selective reconstruction demo
- pHash(64bit) を Shamir (k, n) で秘密分散
- サーバごとに "banded LSH" 風の HMACトークンを保持（pHashの一部を鍵付きで匿名化）
- クエリは pHash → 同じ鍵で bandトークンを作成 → サーバ側で「候補ID」を集合取得
- 候補だけを k台で復元してハミング距離を厳密計算（= 選択的復元）

依存: Pillow, numpy
    pip install Pillow numpy
"""
from __future__ import annotations

import os, glob, time, hmac, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
import numpy as np
from PIL import Image

# =========================================================
# 1) pHash (64bit): 32x32 → 2D-DCT → 8x8 → 平均(DC含)しきい
# =========================================================

def _dct_matrix(N: int) -> np.ndarray:
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    M = np.cos(np.pi * (n + 0.5) * k / N)
    M[0, :] *= np.sqrt(1/N)
    M[1:, :] *= np.sqrt(2/N)
    return M

def _dct2(x: np.ndarray) -> np.ndarray:
    N, M = x.shape
    Cn = _dct_matrix(N)
    Cm = _dct_matrix(M)
    return Cn @ x @ Cm.T

def phash64(image_path: str) -> int:
    img = Image.open(image_path).convert("L")
    img = img.resize((32, 32), Image.BILINEAR)
    X = np.asarray(img, dtype=np.float32)
    D = _dct2(X)
    dct_low = D[:8, :8]
    avg = dct_low.mean()  # DC含む平均（実用で一般的）
    bits = (dct_low.flatten() > avg).astype(np.uint8)  # 64要素
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def hash64_to_bytes(h: int) -> List[int]:
    return [(h >> (8*(7-i))) & 0xFF for i in range(8)]

def bytes_to_hash64(bs: Iterable[int]) -> int:
    bs = list(bs); assert len(bs) == 8
    h = 0
    for b in bs:
        h = (h << 8) | (b & 0xFF)
    return h

# =========================================================
# 2) Shamir Secret Sharing (GF(257)) で 8バイト(pHash)を分散
# =========================================================

P = 257

def _mod_inv(a: int, p: int = P) -> int:
    return pow(a % p, p-2, p)

def _lagrange_basis_at_zero(xs: List[int], j: int, p: int = P) -> int:
    xj = xs[j] % p
    num, den = 1, 1
    for m, xm in enumerate(xs):
        xm %= p
        if m == j: continue
        num = (num * (-xm)) % p
        den = (den * (xj - xm)) % p
    return (num * _mod_inv(den, p)) % p

def shamir_share_bytes(secret_bytes: List[int], k: int, n: int, rng=None) -> Dict[int, List[int]]:
    if rng is None:
        rng = np.random.default_rng()

    L = len(secret_bytes)
    coeffs = [[int(rng.integers(0, P)) for _ in range(k-1)] for _ in range(L)]
    out: Dict[int, List[int]] = {x: [0]*L for x in range(1, n+1)}
    for i, s in enumerate(secret_bytes):
        s %= P
        a = coeffs[i]
        for x in range(1, n+1):
            y = s
            t = 1
            for ai in a:
                t = (t * x) % P
                y = (y + ai * t) % P
            out[x][i] = y
    return out

def shamir_recover_bytes(subshares: Dict[int, List[int]]) -> List[int]:
    xs = sorted(subshares.keys())
    k = len(xs)
    L = len(next(iter(subshares.values())))
    Ljs = [_lagrange_basis_at_zero(xs, j) for j in range(k)]
    rec = [0]*L
    for i in range(L):
        s = 0
        for j, x in enumerate(xs):
            y = subshares[x][i] % P
            s = (s + y * Ljs[j]) % P
        if s == 256: s = 0
        rec[i] = int(s)
    return rec

# =========================================================
# 3) banded HMAC tokens（“選別用”匿名インデックス）
#    - 64bit pHash を B 本の band に分割（例: 8本×8bit）
#    - 各 server×band に固有鍵を持たせ、HMAC-SHA256(token) をキーに ID を束ねる
#    ⇒ サーバは band 一致だけ分かる（中身のbit値やpHash全体は分からない）
# =========================================================

def split_bands(h64: int, bands: int = 8) -> List[int]:
    """64bit を bands 本に等分（8本なら各8bit）。MSB側から順に返す"""
    assert 64 % bands == 0
    r = 64 // bands
    ms = (1 << r) - 1
    out = []
    for i in range(bands):
        shift = 64 - (i+1)*r
        out.append((h64 >> shift) & ms)
    return out  # 各要素は r-bit 値（例: 0..255）

def hmac_token(key: bytes, value_bytes: bytes, outlen: int = 8) -> bytes:
    mac = hmac.new(key, value_bytes, hashlib.sha256).digest()
    return mac[:outlen]  # 短縮トークン（衝突は実用上稀）

# =========================================================
# 4) インデックス：SIS + bandトークン（選択的復元つき）
# =========================================================

@dataclass
class StoredShare:
    image_id: str
    share_bytes: List[int]  # 8 bytes in GF(257)

class SearchableSISIndex:
    def __init__(self, k: int = 3, n: int = 5, bands: int = 8, token_len: int = 8, seed: int = 1234):
        assert 2 <= k <= n <= 20
        assert 64 % bands == 0
        self.k, self.n = k, n
        self.bands, self.token_len = bands, token_len
        # シェア保管
        self.server_shares: Dict[int, Dict[str, StoredShare]] = {x: {} for x in range(1, n+1)}
        # bandトークン: server -> band -> token(bytes) -> set(image_id)
        self.server_band_buckets: Dict[int, List[Dict[bytes, Set[str]]]] = {
            x: [dict() for _ in range(bands)] for x in range(1, n+1)
        }
        # server×band の鍵を作る（デモでは固定seedから決定）
        rng = np.random.default_rng(seed)
        self.hmac_keys: Dict[int, List[bytes]] = {
            x: [rng.integers(0, 256, size=32, dtype=np.uint8).tobytes() for _ in range(bands)]
            for x in range(1, n+1)
        }
        self._images: Dict[str, str] = {}  # id -> path（デモ用）

    # ---- 追加: bandトークン構築 ----
    def _index_band_tokens_for_image(self, image_id: str, h64: int):
        band_vals = split_bands(h64, bands=self.bands)  # 各 r-bit 値
        for x in range(1, self.n+1):
            for b_idx, v in enumerate(band_vals):
                key = self.hmac_keys[x][b_idx]
                token = hmac_token(key, v.to_bytes((64 // self.bands + 7)//8, 'big'), outlen=self.token_len)
                bucket = self.server_band_buckets[x][b_idx].setdefault(token, set())
                bucket.add(image_id)

    def add_image(self, image_id: str, image_path: str) -> int:
        h = phash64(image_path)
        self._images[image_id] = image_path
        # pHashを8バイト化してSIS分散
        bs = hash64_to_bytes(h)
        shares = shamir_share_bytes(bs, self.k, self.n)
        for x, sbytes in shares.items():
            self.server_shares[x][image_id] = StoredShare(image_id=image_id, share_bytes=sbytes)
        # 選別用の bandトークンをインデックス
        self._index_band_tokens_for_image(image_id, h)
        return h

    def list_servers(self) -> List[int]:
        return list(sorted(self.server_shares.keys()))

    def _reconstruct_hash_from_servers(self, image_id: str, servers: Iterable[int]) -> Optional[int]:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k: return None
        sub: Dict[int, List[int]] = {}
        for x in xs:
            item = self.server_shares.get(x, {}).get(image_id)
            if item is None: return None
            sub[x] = item.share_bytes
        bs = shamir_recover_bytes(sub)
        return bytes_to_hash64(bs)

    # ---- 選別：band一致で候補を集め、投票数（一致本数）でフィルタ ----
    def preselect_candidates(self,
                             query_h64: int,
                             servers_for_query: Iterable[int],
                             min_band_votes: int = 3) -> List[Tuple[str, int]]:
        """
        戻り値: [(image_id, votes)] を votes 降順で返す
        - servers_for_query に参加する各サーバで、各 band の HMACトークン一致 ID を取得
        - 全サーバ×全bandの一致票を集計し、min_band_votes 未満を除外
        """
        xs = sorted(set(int(s) for s in servers_for_query))
        band_vals = split_bands(query_h64, bands=self.bands)
        vote: Dict[str, int] = {}
        for x in xs:
            for b_idx, v in enumerate(band_vals):
                key = self.hmac_keys[x][b_idx]
                token = hmac_token(key, v.to_bytes((64 // self.bands + 7)//8, 'big'), outlen=self.token_len)
                ids = self.server_band_buckets[x][b_idx].get(token, set())
                for img_id in ids:
                    vote[img_id] = vote.get(img_id, 0) + 1
        # 閾値フィルタ
        cand = [(img_id, cnt) for img_id, cnt in vote.items() if cnt >= min_band_votes]
        cand.sort(key=lambda t: (-t[1], t[0]))
        return cand

    # ---- 最終ランキング：候補だけ復元して距離算出 ----
    def rank_candidates(self,
                        query_h64: int,
                        servers_for_query: Iterable[int],
                        candidates: Iterable[str],
                        topk: int = 10,
                        max_hamming: Optional[int] = None) -> List[Tuple[str, int]]:
        xs = sorted(set(int(s) for s in servers_for_query))
        if len(xs) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        out: List[Tuple[str, int]] = []
        for img_id in candidates:
            h_db = self._reconstruct_hash_from_servers(img_id, xs)
            if h_db is None:
                continue
            d = hamming64(query_h64, h_db)
            if (max_hamming is None) or (d <= max_hamming):
                out.append((img_id, d))
        out.sort(key=lambda t: (t[1], t[0]))
        return out[:topk]

    # ---- 一発API：選択的復元付き検索 ----
    def query_selective(self,
                        query_image_path: str,
                        servers_for_query: Iterable[int],
                        min_band_votes: int = 3,
                        topk: int = 5,
                        max_hamming: Optional[int] = 10) -> Dict[str, object]:
        hq = phash64(query_image_path)
        pre = self.preselect_candidates(hq, servers_for_query, min_band_votes=min_band_votes)
        pre_ids = [img_id for img_id, _ in pre]
        ranked = self.rank_candidates(hq, servers_for_query, pre_ids, topk=topk, max_hamming=max_hamming)
        return {
            "query_phash": f"0x{hq:016x}",
            "preselected": pre,           # [(image_id, votes)]
            "ranked": ranked,             # [(image_id, hamming)]
            "servers": list(sorted(set(int(s) for s in servers_for_query))),
            "params": {"min_band_votes": min_band_votes, "topk": topk, "max_hamming": max_hamming}
        }

# =========================================================
# 5) デモ
# =========================================================

def hr(): print("-"*72)

if __name__ == "__main__":
    # 設定
    k, n = 3, 5
    bands = 8          # 64bit を 8本×8bit に分割
    min_band_votes = 3 # 票がこの本数未満のIDは復元対象にしない（= 選択的復元）
    topk = 10
    max_hamming = 10

    index = SearchableSISIndex(k=k, n=n, bands=bands, token_len=8, seed=2025)

    # 画像登録
    os.makedirs("images", exist_ok=True)
    db_images = sorted(glob.glob("images/*"))
    if not db_images:
        print("Put some images into ./images/ and run again."); raise SystemExit

    print(f"[CONFIG] k={k}, n={n}, bands={bands}, min_band_votes={min_band_votes}, topk={topk}, max_hamming={max_hamming}")
    print(f"[INFO]   images=./images (found {len(db_images)})")
    hr()

    t0 = time.perf_counter()
    for i, path in enumerate(db_images):
        img_id = f"img_{i:04d}"
        h = index.add_image(img_id, path)
        print(f"[ADD] {img_id:<12} file={os.path.basename(path):<40} pHash=0x{h:016x}")
    t1 = time.perf_counter()
    print(f"[DONE] registered {len(db_images)} images in {t1 - t0:.3f}s")
    hr()

    # クエリ：最初の画像を使う例
    query_path = db_images[0]
    servers = index.list_servers()[:k]  # 例: ちょうどk台が合意
    print(f"[QUERY] query={os.path.basename(query_path)}  servers={servers}")
    res = index.query_selective(query_path, servers_for_query=servers,
                                min_band_votes=min_band_votes, topk=topk, max_hamming=max_hamming)

    # ログ表示
    print(f"[P-HASH] query = {res['query_phash']}")
    print("[PRESELECT] candidates by band tokens (image_id, votes)")
    if not res["preselected"]:
        print("  (no candidates passed min_band_votes)")
    else:
        for img_id, votes in res["preselected"]:
            print(f"  - {img_id:<12} votes={votes}")

    print("[RANK] top results after selective reconstruction (image_id, Hamming)")
    if not res["ranked"]:
        print("  (no matches within threshold)")
    else:
        for img_id, d in res["ranked"]:
            print(f"  - {img_id:<12} Hamming={d}")

    # query_selective() 呼び出し直後に追加
    pre_ids = [img_id for img_id, _ in res["preselected"]]
    ranked_ids = {img_id for img_id, _ in res["ranked"]}
    dropped = [i for i in pre_ids if i not in ranked_ids]
    if dropped:
        print("[DROP] preselected but filtered at RANK (distance > max_hamming):", ", ".join(dropped))
    hr()
    # ベンチ: 全件復元 vs 選択的復元（粗い比較）
    # （実運用では preselect の候補数が劇的に小さくなることを狙う）
    hq = int(res["query_phash"], 16)
    all_ids = list(index._images.keys())
    t_full0 = time.perf_counter()
    full = index.rank_candidates(hq, servers, all_ids, topk=topk, max_hamming=max_hamming)
    t_full1 = time.perf_counter()

    pre_ids = [img_id for img_id, _ in res["preselected"]]
    t_sel0 = time.perf_counter()
    sel = index.rank_candidates(hq, servers, pre_ids, topk=topk, max_hamming=max_hamming)
    t_sel1 = time.perf_counter()

    print("[BENCH] full-reconstruct ranking : {:.3f}s  ({} candidates)".format(t_full1 - t_full0, len(all_ids)))
    print("[BENCH] selective reconstructing: {:.3f}s  ({} candidates after preselect)".format(t_sel1 - t_sel0, len(pre_ids)))
    hr()

    print("[GUIDE]")
    print("  - PRESELECT はサーバ側の HMAC付き bandトークン一致だけで候補抽出（平文pHashは不要）")
    print("  - RANK は候補だけを k台で復元してハミング距離を計算（= 選択的復元）")
    print("  - min_band_votes を上げるほど候補が減り、復元コストが下がる（ただし見逃しリスク↑）")
