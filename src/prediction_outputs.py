import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
from torch_model import SmallUNet, pad_to_30

# -------------------------- Config --------------------------
FOLDER = "../data/predictions"   # folder with prediction JSONs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_SIZE = 30
NUM_COLORS = 10

COLOR_MAP = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

# -------------------------- Load Model --------------------------
MODEL = SmallUNet().to(DEVICE)
MODEL.load_state_dict(torch.load("best_arc_cnn.pth", map_location=DEVICE))
MODEL.eval()

# -------------------------- Utility Functions --------------------------
def load_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()
    return files

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def draw_grid(ax, grid, title):
    ax.clear()
    ax.imshow(np.array(grid), cmap=plt.matplotlib.colors.ListedColormap(COLOR_MAP), interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

def predict_grid(model, grid):
    padded = pad_to_30(grid)
    x = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    # Crop to original size
    return pred[:len(grid), :len(grid[0])]

# -------------------------- Viewer Class --------------------------
class ARCViewer:
    def __init__(self, folder):
        self.folder = folder
        self.files = load_files(folder)
        if not self.files:
            raise FileNotFoundError(f"No JSON files found in {folder}")
        
        self.file_idx = 0
        self.sample_idx = 0
        self.data = None

        # Figure with 3 subplots
        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        self.ax_input = self.fig.add_subplot(gs[0])
        self.ax_pred  = self.fig.add_subplot(gs[1])
        self.ax_gt    = self.fig.add_subplot(gs[2])

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.load_and_draw()

    def load_and_draw(self):
        file_name = self.files[self.file_idx]
        path = os.path.join(self.folder, file_name)
        self.data = load_json(path)

        sample_list = self.data.get("train", [])
        self.sample_idx %= len(sample_list)
        sample = sample_list[self.sample_idx]

        inp = sample["input"]
        gt  = sample["output"]
        pred = predict_grid(MODEL, inp)

        draw_grid(self.ax_input, inp, "Input")
        draw_grid(self.ax_pred, pred, "Prediction")
        draw_grid(self.ax_gt, gt, "Ground Truth")

        self.fig.suptitle(
            f"File: {file_name} ({self.file_idx+1}/{len(self.files)}) | Sample: {self.sample_idx+1}/{len(sample_list)}",
            fontsize=16
        )
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "right":
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.sample_idx = 0
            self.load_and_draw()
        elif event.key == "left":
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.sample_idx = 0
            self.load_and_draw()
        elif event.key == "down":
            self.sample_idx += 1
            self.load_and_draw()
        elif event.key == "up":
            self.sample_idx -= 1
            self.load_and_draw()
        elif event.key == "q":
            plt.close(self.fig)

# -------------------------- Main --------------------------
if __name__ == "__main__":
    viewer = ARCViewer(FOLDER)
    plt.show()
