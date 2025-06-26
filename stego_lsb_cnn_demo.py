# lsb_cnn_demo.py
import random, math, os, io
from typing import List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ──────────────────────────────
# 1. LSB 埋め込み／抽出ユーティリティ
# ──────────────────────────────
def str_to_bits(s: str) -> List[int]:
    return [int(b) for ch in s.encode() for b in f'{ch:08b}']

def bits_to_str(bits: List[int]) -> str:
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return bytes(int(''.join(map(str,c)), 2) for c in chars).decode(errors='ignore')

def lsb_embed(img: Image.Image, bits: List[int]) -> Image.Image:
    """青チャネルの LSB に bits を順に書き込む（不足分は 0 パディング）"""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    total = h * w
    if len(bits) > total:
        raise ValueError("メッセージが長すぎます")
    bits = bits + [0]*(total - len(bits))  # 余りを埋める
    flat = arr.reshape(-1, 3)
    for i, bit in enumerate(bits):
        flat[i][2] = (flat[i][2] & 0xFE) | bit    # Blue channel LSB
    arr_stego = flat.reshape(h, w, 3)
    return Image.fromarray(arr_stego.astype(np.uint8))

def lsb_extract(img: Image.Image, n_bits: int) -> List[int]:
    arr = np.array(img.convert("RGB"))
    flat_b = arr.reshape(-1, 3)[:, 2]
    return [px & 1 for px in flat_b[:n_bits]]

# ──────────────────────────────
# 2. データセット生成
# ──────────────────────────────
class BitDataset(Dataset):
    """
    ステガ画像を 8×8 RGB パッチに分割し，
    各パッチ中央ピクセルの Blue-LSB をラベルにする
    """
    def __init__(self, cover_imgs: List[Image.Image], bits_per_img: int = 2048):
        self.patches, self.labels = [], []
        tf = transforms.ToTensor()
        for img in cover_imgs:
            # ランダムビット列を埋め込み
            bits = [random.getrandbits(1) for _ in range(bits_per_img)]
            stego = lsb_embed(img, bits)
            arr = np.array(stego)
            # パッチ分割
            for y in range(0, arr.shape[0]-7, 8):
                for x in range(0, arr.shape[1]-7, 8):
                    patch = arr[y:y+8, x:x+8, :]
                    # 中央ピクセル (4,4) の LSB を教師ラベル
                    lbl = patch[4,4,2] & 1
                    self.patches.append(tf(Image.fromarray(patch)))
                    self.labels.append(lbl)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

# ──────────────────────────────
# 3. 簡易 CNN
# ──────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),        # → (32,1,1)
            nn.Flatten(),
            nn.Linear(32, 1),               # 2-class → σ
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ──────────────────────────────
# 4. 学習 & 推論
# ──────────────────────────────
def train_cnn(train_ds: Dataset, epochs: int = 3, batch=128) -> SimpleCNN:
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    model = SimpleCNN().cuda()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    for epoch in range(epochs):
        model.train(); pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        for x,y in pbar:
            x,y = x.cuda(), y.cuda().unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            pbar.set_postfix(loss=loss.item())
    return model.cpu().eval()

def predict_bits(model: SimpleCNN, stego: Image.Image, n_bits: int) -> List[int]:
    tf = transforms.ToTensor()
    arr = np.array(stego)
    preds = []
    with torch.no_grad():
        for idx in range(n_bits):
            y, x = divmod(idx, arr.shape[1])
            # 8×8 パッチ（境界はゼロパディング）
            y0, x0 = max(0, y-4), max(0, x-4)
            patch = arr[y0:y0+8, x0:x0+8, :]
            patch = tf(Image.fromarray(patch)).unsqueeze(0)
            p = model(patch)[0,0].item()
            preds.append(1 if p>0.5 else 0)
    return preds

# ──────────────────────────────
# 5. デモ
# ──────────────────────────────
if __name__ == "__main__":
    # カバー画像を用意（ここではフリー画像を簡単に生成）
    img = Image.new("RGB", (256,256), color=(200,220,242))
    # 学習データセット作成（ダミーカバー画像を複数用意しても良い）
    train_ds = BitDataset([img]*50, bits_per_img=2048)
    cnn = train_cnn(train_ds, epochs=2)

    # テスト：任意メッセージを埋め込み→復号
    secret_msg = "HELLO XOR-SIS!!"
    bits = str_to_bits(secret_msg)
    stego_img = lsb_embed(img, bits)
    # CNN でビット推定
    rec_bits = predict_bits(cnn, stego_img, len(bits))
    decoded = bits_to_str(rec_bits)
    # 直接 LSB 抽出した場合と比較
    direct_bits = lsb_extract(stego_img, len(bits))
    direct_decoded = bits_to_str(direct_bits)

    print("元メッセージ:", secret_msg)
    print("CNN 復号     :", decoded)
    print("直接 LSB 抽出 :", direct_decoded)
