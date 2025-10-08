# searchable_sis_phash_image_sis_demo.py
# -*- coding: utf-8 -*-
"""
Searchable SIS (pHash) + Image SIS (Shamir) unified demo
- 画像の pHash(64bit) を Shamir (k,n) で分散し、HMACバンドトークンで“候補のみ”を選別
- 候補だけ k 台から pHash を復元してハミング距離を厳密判定（= 選択的復元）
- 同じ image_id で画像本体も Shamir (k,n) で分散保存
- 類似候補のみ画像を復元（k台合意）して ./recon_out/ に書き出し

依存:
  pip install numpy Pillow
"""

from __future__ import annotations
import os, glob, time, hmac, hashlib, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
from PIL import Image

# =========================================================
# 共通: pHash (64bit) 実装（32x32→DCT→8x8→平均しきい）
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
    avg = dct_low.mean()
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
# Shamir Secret Sharing（GF(257））: pHash用（8バイト）＆画像用（ベクトル化）
# =========================================================
P = 257

def _mod_inv(a: int, p: int = P) -> int:
    return pow(a % p, p-2, p)

def _lagrange_basis_at_zero(xs: List[int], p: int = P) -> np.ndarray:
    xs = [x % p for x in xs]
    k = len(xs)
    Ljs = np.zeros(k, dtype=np.int64)
    for j in range(k):
        num, den = 1, 1
        xj = xs[j]
        for m in range(k):
            xm = xs[m]
            if m == j: continue
            num = (num * (-xm)) % p
            den = (den * (xj - xm)) % p
        den_inv = pow(den % p, p-2, p)
        Ljs[j] = (num * den_inv) % p
    return Ljs.astype(np.uint16)

# ---- pHash(8B) 用：シンプル実装 ----
def shamir_share_bytes(secret_bytes: List[int], k: int, n: int, rng=None) -> Dict[int, List[int]]:
    if rng is None:
        rng = np.random.default_rng()
    L = len(secret_bytes)
    coeffs = [[int(rng.integers(0, P)) for _ in range(k-1)] for _ in range(L)]
    out: Dict[int, List[int]] = {x: [0]*L for x in range(1, n+1)}
    for i, s in enumerate(secret_bytes):
        s %= P
        for x in range(1, n+1):
            y = s
            t = 1
            for ai in coeffs[i]:
                t = (t * x) % P
                y = (y + ai * t) % P
            out[x][i] = y
    return out

def shamir_recover_bytes(subshares: Dict[int, List[int]]) -> List[int]:
    xs = sorted(subshares.keys())
    k = len(xs)
    L = len(next(iter(subshares.values())))
    Ljs = _lagrange_basis_at_zero(xs).astype(np.uint32)
    rec = [0]*L
    for i in range(L):
        s = 0
        for j, x in enumerate(xs):
            y = subshares[x][i] % P
            s = (s + y * Ljs[j]) % P
        if s == 256: s = 0
        rec[i] = int(s)
    return rec

# ---- 画像全体用（ベクトル化：高速） ----
def shamir_share_flat_bytes(flat_u8: np.ndarray, k: int, n: int, rng: np.random.Generator
                            ) -> Dict[int, np.ndarray]:
    L = flat_u8.size
    s = flat_u8.astype(np.uint16)
    coeffs = [rng.integers(0, P, size=L, dtype=np.uint16) for _ in range(k-1)]
    shares: Dict[int, np.ndarray] = {}
    for x in range(1, n+1):
        y = s.copy().astype(np.uint32)
        t = np.ones(L, dtype=np.uint32)
        for a in coeffs:
            t = (t * x) % P
            y = (y + (a.astype(np.uint32) * t)) % P
        shares[x] = y.astype(np.uint16)
    return shares

def shamir_recover_flat_bytes(subshares: Dict[int, np.ndarray]) -> np.ndarray:
    xs = sorted(int(x) for x in subshares.keys())
    ys = [subshares[x].astype(np.uint32) for x in xs]
    L = ys[0].size
    Ljs = _lagrange_basis_at_zero(xs).astype(np.uint32)
    acc = np.zeros(L, dtype=np.uint32)
    for j in range(len(xs)):
        acc = (acc + (ys[j] * Ljs[j])) % P
    acc = np.where(acc == 256, 0, acc)
    return acc.astype(np.uint8)

# =========================================================
# 画像I/O（RGB）
# =========================================================
def load_image_u8(path: str) -> Tuple[np.ndarray, Tuple[int,int,int]]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr, arr.shape

def save_image_u8(path: str, arr_u8: np.ndarray) -> None:
    Image.fromarray(arr_u8, mode="RGB").save(path)

# =========================================================
# pHash 検索: banded HMAC token（匿名プレセレクト）
# =========================================================
def split_bands(h64: int, bands: int = 8) -> List[int]:
    assert 64 % bands == 0
    r = 64 // bands
    ms = (1 << r) - 1
    out = []
    for i in range(bands):
        shift = 64 - (i+1)*r
        out.append((h64 >> shift) & ms)
    return out

def hmac_token(key: bytes, value_bytes: bytes, outlen: int = 8) -> bytes:
    return hmac.new(key, value_bytes, hashlib.sha256).digest()[:outlen]

# =========================================================
# 画像本体 Shamir ストア
# =========================================================
@dataclass
class ImageMeta:
    image_id: str
    shape: Tuple[int, int, int]
    filename: str

class ShamirImageStore:
    def __init__(self, k: int, n: int, shares_dir: str = "img_shares", meta_dir: str = "img_meta"):
        self.k, self.n = k, n
        self.shares_dir, self.meta_dir = shares_dir, meta_dir
        os.makedirs(self.meta_dir, exist_ok=True)
        for x in range(1, n+1):
            os.makedirs(self._server_dir(x), exist_ok=True)

    def _server_dir(self, x: int) -> str:
        return os.path.join(self.shares_dir, f"server_{x}")

    def add_image(self, image_id: str, image_path: str, rng_seed: Optional[int] = None) -> ImageMeta:
        arr, shape = load_image_u8(image_path)
        flat = arr.reshape(-1)
        rng = np.random.default_rng(rng_seed)
        shares = shamir_share_flat_bytes(flat, self.k, self.n, rng)
        for x, y in shares.items():
            np.savez_compressed(os.path.join(self._server_dir(x), f"{image_id}.npz"), share=y)
        np.savez_compressed(os.path.join(self.meta_dir, f"{image_id}.npz"),
                            shape=np.array(shape, dtype=np.int32),
                            filename=os.path.basename(image_path))
        return ImageMeta(image_id=image_id, shape=shape, filename=os.path.basename(image_path))

    def reconstruct(self, image_id: str, servers: Iterable[int], out_path: str) -> bool:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k:
            print(f"[ERR] Need at least k={self.k} servers to reconstruct.")
            return False
        mpath = os.path.join(self.meta_dir, f"{image_id}.npz")
        if not os.path.exists(mpath):
            print(f"[ERR] meta not found for {image_id}"); return False
        meta = np.load(mpath)
        shape = tuple(meta["shape"].tolist())
        sub: Dict[int, np.ndarray] = {}
        for x in xs:
            sp = os.path.join(self._server_dir(x), f"{image_id}.npz")
            if not os.path.exists(sp):
                print(f"[WARN] missing share: server_{x}/{image_id}.npz"); return False
            sub[x] = np.load(sp)["share"]
        flat = shamir_recover_flat_bytes(sub)
        arr = flat.reshape(shape)
        save_image_u8(out_path, arr)
        return True

# =========================================================
# 統合: pHash 検索 + 画像ストア
# =========================================================
@dataclass
class StoredShare:
    image_id: str
    share_bytes: List[int]  # 8 bytes in GF(257)

class SearchableSISWithImageStore:
    """
    - pHash: Shamir(k,n) + banded HMAC tokens（匿名プレセレクト）
    - Image: Shamir(k,n) full-image shares persisted as .npz (per server)
    """
    def __init__(self, k: int = 3, n: int = 5, bands: int = 8, token_len: int = 8, seed: int = 2025,
                 shares_dir: str = "img_shares", meta_dir: str = "img_meta"):
        assert 2 <= k <= n <= 20 and 64 % bands == 0
        self.k, self.n, self.bands, self.token_len = k, n, bands, token_len
        # pHash共有
        self.server_shares: Dict[int, Dict[str, StoredShare]] = {x: {} for x in range(1, n+1)}
        # band token buckets
        self.server_band_buckets: Dict[int, List[Dict[bytes, Set[str]]]] = {
            x: [dict() for _ in range(bands)] for x in range(1, n+1)
        }
        # server×band 鍵
        rng = np.random.default_rng(seed)
        self.hmac_keys: Dict[int, List[bytes]] = {
            x: [rng.integers(0, 256, size=32, dtype=np.uint8).tobytes() for _ in range(bands)]
            for x in range(1, n+1)
        }
        # image store
        self.store = ShamirImageStore(k=k, n=n, shares_dir=shares_dir, meta_dir=meta_dir)
        # メモ（デモ用）
        self._images: Dict[str, str] = {}  # image_id -> original path

    def list_servers(self) -> List[int]:
        return list(sorted(self.server_shares.keys()))

    # ---- 画像登録（pHash側＋画像側の両方を構築）
    def add_image(self, image_id: str, image_path: str) -> int:
        self._images[image_id] = image_path
        # 1) pHash
        h = phash64(image_path)
        #   a) pHashを8バイトに分割してShamir共有
        bs = hash64_to_bytes(h)
        shares = shamir_share_bytes(bs, self.k, self.n)
        for x, sbytes in shares.items():
            self.server_shares[x][image_id] = StoredShare(image_id=image_id, share_bytes=sbytes)
        #   b) band token を作って匿名インデックス
        band_vals = split_bands(h, bands=self.bands)
        rbits = 64 // self.bands
        bsize = (rbits + 7)//8
        for x in range(1, self.n+1):
            for b_idx, v in enumerate(band_vals):
                key = self.hmac_keys[x][b_idx]
                token = hmac_token(key, v.to_bytes(bsize, 'big'), outlen=self.token_len)
                bucket = self.server_band_buckets[x][b_idx].setdefault(token, set())
                bucket.add(image_id)
        # 2) 画像本体 Shamir 共有（永続化）
        self.store.add_image(image_id, image_path, rng_seed=None)
        return h

    # ---- pHash 復元（k台合意）
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

    # ---- プレセレクト（匿名トークン一致投票）
    def preselect_candidates(self, query_h64: int, servers: Iterable[int], min_band_votes: int = 3
                             ) -> List[Tuple[str,int]]:
        xs = sorted(set(int(s) for s in servers))
        band_vals = split_bands(query_h64, bands=self.bands)
        rbits = 64 // self.bands
        bsize = (rbits + 7)//8
        vote: Dict[str, int] = {}
        for x in xs:
            for b_idx, v in enumerate(band_vals):
                key = self.hmac_keys[x][b_idx]
                token = hmac_token(key, v.to_bytes(bsize, 'big'), outlen=self.token_len)
                ids = self.server_band_buckets[x][b_idx].get(token, set())
                for img_id in ids:
                    vote[img_id] = vote.get(img_id, 0) + 1
        cand = [(img_id, cnt) for img_id, cnt in vote.items() if cnt >= min_band_votes]
        cand.sort(key=lambda t: (-t[1], t[0]))
        return cand

    # ---- 候補だけ pHash を復元→距離
    def rank_candidates(self, query_h64: int, servers: Iterable[int], candidates: Iterable[str],
                        topk: int = 10, max_hamming: Optional[int] = 10) -> List[Tuple[str,int]]:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k: raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        out: List[Tuple[str,int]] = []
        for img_id in candidates:
            h_db = self._reconstruct_hash_from_servers(img_id, xs)
            if h_db is None:
                continue
            d = hamming64(query_h64, h_db)
            if (max_hamming is None) or (d <= max_hamming):
                out.append((img_id, d))
        out.sort(key=lambda t: (t[1], t[0]))
        return out[:topk]

    # ---- 一発API：選択的復元つき検索＋（任意で）画像復元
    def query_and_optionally_reconstruct(self, query_image_path: str, servers: Iterable[int],
                                         min_band_votes: int = 3, topk: int = 5, max_hamming: int = 10,
                                         reconstruct_top: int = 1, recon_dir: str = "recon_out") -> Dict[str,object]:
        hq = phash64(query_image_path)
        pre = self.preselect_candidates(hq, servers, min_band_votes=min_band_votes)
        pre_ids = [img_id for img_id, _ in pre]
        ranked = self.rank_candidates(hq, servers, pre_ids, topk=topk, max_hamming=max_hamming)

        recon_paths: List[Tuple[str,str]] = []
        if reconstruct_top and ranked:
            os.makedirs(recon_dir, exist_ok=True)
            for img_id, _ in ranked[:reconstruct_top]:
                out_path = os.path.join(recon_dir, f"reconstructed_{img_id}.png")
                ok = self.store.reconstruct(img_id, servers, out_path)
                if ok: recon_paths.append((img_id, out_path))

        return {
            "query_phash": f"0x{hq:016x}",
            "preselected": pre,              # [(image_id, votes)]
            "ranked": ranked,                # [(image_id, hamming)]
            "reconstructed": recon_paths,    # [(image_id, out_path)]
            "servers": list(sorted(set(int(s) for s in servers))),
            "params": {"min_band_votes":min_band_votes, "topk":topk, "max_hamming":max_hamming,
                       "reconstruct_top": reconstruct_top, "recon_dir": recon_dir}
        }

# =========================================================
# デモ
# =========================================================
def hr(): print("-"*72)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--min_band_votes", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_hamming", type=int, default=10)
    ap.add_argument("--reconstruct_top", type=int, default=1, help="上位何件 画像を復元するか（0で復元なし）")
    ap.add_argument("--images_dir", type=str, default="images")
    ap.add_argument("--recon_dir", type=str, default="recon_out")
    ap.add_argument("--query", type=str, default=None, help="クエリ画像のパス（未指定なら最初の画像）")
    args = ap.parse_args()

    k, n = args.k, args.n
    bands = args.bands
    min_band_votes = args.min_band_votes
    topk = args.topk
    max_hamming = args.max_hamming
    reconstruct_top = args.reconstruct_top
    images_dir = args.images_dir
    recon_dir = args.recon_dir

    os.makedirs(images_dir, exist_ok=True)
    db_images = sorted(glob.glob(os.path.join(images_dir, "*")))
    if not db_images:
        print("Put some images into ./images/ and run again."); return

    index = SearchableSISWithImageStore(k=k, n=n, bands=bands, token_len=8, seed=2025,
                                        shares_dir="img_shares", meta_dir="img_meta")

    print(f"[CONFIG] k={k}, n={n}, bands={bands}, min_band_votes={min_band_votes}, topk={topk}, max_hamming={max_hamming}, reconstruct_top={reconstruct_top}")
    print(f"[INFO]   images_dir={images_dir} (found {len(db_images)})")
    hr()

    # 登録
    t0 = time.perf_counter()
    ids = []
    for i, path in enumerate(db_images):
        img_id = f"img_{i:04d}"
        h = index.add_image(img_id, path)
        ids.append(img_id)
        print(f"[ADD] {img_id:<12} file={os.path.basename(path):<40} pHash=0x{h:016x}")
    t1 = time.perf_counter()
    print(f"[DONE] registered {len(ids)} images in {t1 - t0:.3f}s")
    hr()

    # クエリ
    query_path = args.query or db_images[0]
    servers = index.list_servers()[:k]  # 例：ちょうど k 台
    print(f"[QUERY] query={os.path.basename(query_path)}  servers={servers}")
    res = index.query_and_optionally_reconstruct(
        query_path, servers,
        min_band_votes=min_band_votes,
        topk=topk,
        max_hamming=max_hamming,
        reconstruct_top=reconstruct_top,
        recon_dir=recon_dir
    )

    print(f"[P-HASH] query = {res['query_phash']}")
    print("[PRESELECT] (image_id, votes)")
    if not res["preselected"]:
        print("  (no candidates passed min_band_votes)")
    else:
        for img_id, votes in res["preselected"]:
            print(f"  - {img_id:<12} votes={votes}")

    print("[RANK] (image_id, Hamming)")
    if not res["ranked"]:
        print("  (no matches within threshold)")
    else:
        for img_id, d in res["ranked"]:
            print(f"  - {img_id:<12} Hamming={d}")

    if res["reconstructed"]:
        print("[RECONSTRUCTED] written files:")
        for img_id, path in res["reconstructed"]:
            print(f"  - {img_id:<12} -> {path}")
    else:
        print("[RECONSTRUCTED] (none)")

    hr()
    # ベンチ（全件 vs 選択）
    hq = int(res["query_phash"], 16)
    all_ids = ids
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
    print(" - PRESELECT: HMAC付き band トークン一致だけで候補抽出（平文pHash不要）")
    print(" - RANK     : 候補のみ pHash を k台で復元して距離算出（選択的復元）")
    print(" - IMAGE    : 同じ image_id に対する画像本体 SIS を、上位 only で復元 ./recon_out/")
    print(" - パラメータ調整: --min_band_votes / --max_hamming / --reconstruct_top")

if __name__ == "__main__":
    main()
