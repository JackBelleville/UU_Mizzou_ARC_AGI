# train_cnn_with_crop.py
import os, json, random
from pathlib import Path
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# -------------- Config --------------
DATA_FOLDER = "../data/training"
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COLORS = 10  # classes 0..9
PAD_SIZE = 30
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------- Utilities --------------
def pad_to_30(grid):
    g = np.array(grid, dtype=np.int64)
    H,W = g.shape
    out = np.zeros((PAD_SIZE, PAD_SIZE), dtype=np.int64)
    out[:H, :W] = g
    return out

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def gather_train_pairs(folder: str):
    folder = Path(folder)
    pairs = []
    for p in folder.glob("*.json"):
        try:
            data = load_json(p)
            for ex in data.get("train", []):
                inp = pad_to_30(ex["input"])
                out = pad_to_30(ex["output"])
                pairs.append((inp, out))
        except Exception:
            continue
    return pairs

# -------------- Dataset --------------
class ARCDataset(Dataset):
    def __init__(self, pairs, augment=True):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def rand_color_map(self):
        perm = list(range(NUM_COLORS))
        random.shuffle(perm)
        return perm

    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        if self.augment and random.random() < 0.5:
            k = random.choice([0,1,2,3])
            inp = np.rot90(inp, k).copy()
            out = np.rot90(out, k).copy()
        if self.augment and random.random() < 0.5:
            if random.random() < 0.5:
                inp = np.fliplr(inp).copy()
                out = np.fliplr(out).copy()
            else:
                inp = np.flipud(inp).copy()
                out = np.flipud(out).copy()
        if self.augment and random.random() < 0.5:
            cmap = self.rand_color_map()
            inp = np.vectorize(lambda x: cmap[x])(inp)
            out = np.vectorize(lambda x: cmap[x])(out)

        x = torch.tensor(inp, dtype=torch.long)
        y = torch.tensor(out, dtype=torch.long)
        return x, y

# -------------- Model --------------
class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, base=64, n_classes=NUM_COLORS):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Embedding(NUM_COLORS, base))
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 2, stride=2), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 2, stride=2), nn.ReLU())
        self.out = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x_long):
        B,H,W = x_long.shape
        emb = self.enc1[0](x_long)          # (B, H, W, base)
        emb = emb.permute(0,3,1,2).float()  # (B, base, H, W)

        x = self.conv_in(emb)
        d1 = self.down1(x)
        d2 = self.down2(d1)

        # UPSAMPLE WITH CROPPING
        u1 = self.up1(d2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = u1[:, :, :d1.shape[2], :d1.shape[3]]
        u1 = u1 + d1

        u2 = self.up2(u1)
        if u2.shape[2:] != x.shape[2:]:
            u2 = u2[:, :, :x.shape[2], :x.shape[3]]
        u2 = u2 + x

        out = self.out(u2)
        return out

# -------------- Metrics --------------
def compute_metrics(pred_logits, target):
    preds = pred_logits.argmax(dim=1)
    pixel_acc = (preds == target).float().mean().item()
    B = target.shape[0]
    exact = sum(torch.equal(preds[i], target[i]) for i in range(B))
    exact_rate = exact / B
    return pixel_acc, exact_rate

# -------------- Training --------------
def train():
    pairs = gather_train_pairs(DATA_FOLDER)
    print("Loaded pairs:", len(pairs))
    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = ARCDataset(train_pairs, augment=True)
    val_ds = ARCDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SmallUNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
        running_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_pixel = 0.0
        val_exact = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                pa, er = compute_metrics(logits, yb)
                batch = xb.size(0)
                val_pixel += pa * batch
                val_exact += er * batch
                count += batch
        val_pixel /= count
        val_exact /= count
        print(f"Epoch {epoch+1}/{EPOCHS} loss={running_loss:.4f} val_pixel={val_pixel:.4f} val_exact={val_exact:.4f}")

        if val_exact > best_val:
            best_val = val_exact
            torch.save(model.state_dict(), "best_arc_cnn.pth")
            print("Saved best model.")

    return model

if __name__ == "__main__":
    train()
