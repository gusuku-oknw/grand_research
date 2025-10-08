# shamir_image_sis_store_demo.py
# -*- coding: utf-8 -*-
"""
Shamir-based (k,n) Secret Image Sharing for full images (not only pHash).
- 画像ピクセル（RGB 8bit）をそのまま GF(257) 上で Shamir 共有
- 各サーバの share は .npz として保存（uint16）
- 復元時は任意 k サーバの share を読み出して元画像を完全復元

依存: Pillow, numpy
    pip install Pillow numpy
"""
from __future__ import annotations

import os, glob, time
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
from PIL import Image

P = 257  # Shamir 用の素数（0..255 の画素をそのまま要素として扱える）

# -----------------------------
# 画像 I/O
# -----------------------------
def load_image_u8(path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """画像を RGB uint8 (H,W,3) で読み込む"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # H,W,3
    return arr, arr.shape

def save_image_u8(path: str, arr_u8: np.ndarray) -> None:
    Image.fromarray(arr_u8, mode="RGB").save(path)

# -----------------------------
# Shamir (k,n) ベクトル化実装
# -----------------------------
def shamir_share_flat_bytes(flat_u8: np.ndarray, k: int, n: int, rng: np.random.Generator
                            ) -> Dict[int, np.ndarray]:
    """
    入力: flat_u8 (L,) uint8  …… 画像の全バイト (H*W*3) を一次元化
    出力: {x: flat_share_uint16}  …… 各サーバ x の share（GF(257) 値を格納; dtype uint16）
    ※ ベクトル化：各 x で y = s + a1*x + a2*x^2 + ... を一括で計算
    """
    L = flat_u8.size
    s = flat_u8.astype(np.uint16)  # 0..255
    # 係数 a1..a_{k-1} を配列で用意（各バイト独立の乱数係数）
    coeffs = [rng.integers(0, P, size=L, dtype=np.uint16) for _ in range(k-1)]

    shares: Dict[int, np.ndarray] = {}
    for x in range(1, n+1):
        y = s.copy()
        t = np.ones(L, dtype=np.uint32)  # x^d の累積（オーバーフロー回避用に32bit）
        for a in coeffs:
            t = (t * x) % P
            y = (y.astype(np.uint32) + (a.astype(np.uint32) * t)) % P
        shares[x] = y.astype(np.uint16)  # 0..256 の可能性; 保存は uint16
    return shares

def _lagrange_basis_at_zero(xs: List[int]) -> np.ndarray:
    """
    Lagrange 基底係数 l_j(0) を j=0..k-1 について計算（mod P）
    戻り値: (k,) uint16
    """
    xs = [x % P for x in xs]
    k = len(xs)
    Ljs = np.zeros(k, dtype=np.int64)
    for j in range(k):
        num, den = 1, 1
        xj = xs[j]
        for m in range(k):
            xm = xs[m]
            if m == j: continue
            num = (num * (-xm)) % P
            den = (den * (xj - xm)) % P
        # 逆元
        den_inv = pow(den % P, P-2, P)
        Ljs[j] = (num * den_inv) % P
    return Ljs.astype(np.uint16)

def shamir_recover_flat_bytes(subshares: Dict[int, np.ndarray]) -> np.ndarray:
    """
    任意 k サーバの share（フラット配列）から元のバイト配列を復元（ベクトル化）
    subshares: {x: flat_share_uint16}, すべて同じ長さ L
    戻り値: flat_u8 (L,) …… 復元画素（uint8）
    """
    xs = sorted(int(x) for x in subshares.keys())
    ys = [subshares[x].astype(np.uint32) for x in xs]  # List[(L,)]
    L = ys[0].size
    k = len(xs)

    Ljs = _lagrange_basis_at_zero(xs).astype(np.uint32)  # (k,)
    # s = Σ_j y_j * l_j(0)  (mod P) …… 要素ごとにベクトル化
    acc = np.zeros(L, dtype=np.uint32)
    for j in range(k):
        acc = (acc + (ys[j] * Ljs[j])) % P

    # 理論上 0..255 のはず（mod 257）。安全のため 256→0 に丸め。
    acc = np.where(acc == 256, 0, acc)
    return acc.astype(np.uint8)

# -----------------------------
# ストレージ（.npz 保存/読込）
# -----------------------------
@dataclass
class ImageMeta:
    image_id: str
    shape: Tuple[int, int, int]  # (H,W,3)
    filename: str               # 元画像ファイル名（情報用）

class ShamirImageStore:
    """
    画像本体を Shamir (k,n) で分散/復元するストア。
    - shares_dir/server_{x}/{image_id}.npz として share を保存
    - meta_dir/{image_id}.npz に shape/元ファイル名を保存
    """
    def __init__(self, k: int = 3, n: int = 5, shares_dir: str = "img_shares", meta_dir: str = "img_meta"):
        assert 2 <= k <= n <= 20
        self.k, self.n = k, n
        self.shares_dir = shares_dir
        self.meta_dir = meta_dir
        os.makedirs(self.meta_dir, exist_ok=True)
        for x in range(1, n+1):
            os.makedirs(self._server_dir(x), exist_ok=True)

    def _server_dir(self, x: int) -> str:
        return os.path.join(self.shares_dir, f"server_{x}")

    def add_image(self, image_id: str, image_path: str, rng_seed: Optional[int] = None) -> ImageMeta:
        arr, shape = load_image_u8(image_path)  # (H,W,3) uint8
        flat = arr.reshape(-1)  # (L,)

        rng = np.random.default_rng(rng_seed)
        shares = shamir_share_flat_bytes(flat, self.k, self.n, rng)

        # 保存
        for x, y in shares.items():
            out_path = os.path.join(self._server_dir(x), f"{image_id}.npz")
            np.savez_compressed(out_path, share=y)

        meta_path = os.path.join(self.meta_dir, f"{image_id}.npz")
        np.savez_compressed(meta_path, shape=np.array(shape, dtype=np.int32), filename=os.path.basename(image_path))
        return ImageMeta(image_id=image_id, shape=shape, filename=os.path.basename(image_path))

    def reconstruct(self, image_id: str, servers: Iterable[int], out_path: str) -> bool:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k:
            print(f"[ERR] Need at least k={self.k} servers to reconstruct.")
            return False

        meta_path = os.path.join(self.meta_dir, f"{image_id}.npz")
        if not os.path.exists(meta_path):
            print(f"[ERR] meta not found for {image_id}")
            return False
        meta = np.load(meta_path)
        shape = tuple(meta["shape"].tolist())

        subshares: Dict[int, np.ndarray] = {}
        for x in xs:
            sp = os.path.join(self._server_dir(x), f"{image_id}.npz")
            if not os.path.exists(sp):
                print(f"[WARN] share missing: server_{x}/{image_id}.npz")
                return False
            y = np.load(sp)["share"]
            subshares[x] = y

        flat_u8 = shamir_recover_flat_bytes(subshares)
        arr = flat_u8.reshape(shape)
        save_image_u8(out_path, arr)
        return True

# -----------------------------
# 4) デモ
# -----------------------------
def hr(): print("-"*72)

if __name__ == "__main__":
    """
    使い方:
      1) ./images に jpg/png を入れる
      2) python shamir_image_sis_store_demo.py
    """
    # パラメータ
    k, n = 3, 5
    shares_dir = "img_shares"
    meta_dir = "img_meta"
    os.makedirs("images", exist_ok=True)

    db_images = sorted(glob.glob("images/*"))
    if not db_images:
        print("Put some images into ./images/ and run again.")
        raise SystemExit

    print(f"[CONFIG] k={k}, n={n}")
    print(f"[INFO]   images=./images (found {len(db_images)})  shares_dir={shares_dir}")
    hr()

    store = ShamirImageStore(k=k, n=n, shares_dir=shares_dir, meta_dir=meta_dir)

    # 画像を分散保存
    t0 = time.perf_counter()
    img_ids = []
    for i, path in enumerate(db_images):
        img_id = f"img_{i:04d}"
        _meta = store.add_image(img_id, path, rng_seed=None)
        img_ids.append(img_id)
        print(f"[ADD] {img_id:<12} <- {os.path.basename(path)}  (H,W,3)={_meta.shape}")
    t1 = time.perf_counter()
    print(f"[DONE] shared {len(img_ids)} images in {t1 - t0:.3f}s")
    hr()

    # 復元テスト（最初の画像）
    target_id = img_ids[0]
    servers_ok = [1,2,3]  # ちょうど k 台
    out_path = f"reconstructed_{target_id}.png"
    ok = store.reconstruct(target_id, servers_ok, out_path)
    print(f"[RECONSTRUCT] {target_id} with servers={servers_ok} -> {'OK' if ok else 'FAIL'}  out={out_path if ok else '-'}")

    # k-1 台での失敗を確認
    servers_ng = [1,2]
    out_path_ng = f"reconstructed_{target_id}_ng.png"
    ok2 = store.reconstruct(target_id, servers_ng, out_path_ng)
    print(f"[RECONSTRUCT] {target_id} with servers={servers_ng} -> {'OK' if ok2 else 'FAIL (expected)'}")
    hr()

    print("[GUIDE]")
    print(" - 画像本体を Shamir の (k,n) で分散し、各サーバに .npz share を保存します。")
    print(" - 復元は任意 k 台から。k 未満では復元できません。")
    print(" - pHash 検索フェーズと組み合わせる場合は、image_id を共通にして、")
    print("   類似候補の image_id だけ store.reconstruct() を呼べば “選択的復元” ができます。")
