import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import keyboard
import time
import torch

# --------------------------
# IMPORT YOUR CNN MODEL
# --------------------------
from torch_model import SmallUNet  # adjust path if needed

# --------------------------
# CNN CONFIG
# --------------------------
NUM_COLORS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_SIZE = 30

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
MODEL = SmallUNet().to(DEVICE)
MODEL.load_state_dict(torch.load("best_arc_cnn.pth", map_location=DEVICE))
MODEL.eval()

# --------------------------
# Data folder
# --------------------------
FOLDER = "../data/training"

COLOR_MAP = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

# --------------------------
# Utility functions
# --------------------------
def load_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()
    return files

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def draw_grid(ax, grid, title):
    ax.imshow(np.array(grid), cmap=plt.matplotlib.colors.ListedColormap(COLOR_MAP), interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

def compute_correctness(pred_grid, true_grid):
    """
    Returns 1.0 if the prediction grid exactly matches the ground truth, else 0.0
    """
    pred = np.array(pred_grid)
    true = np.array(true_grid)
    H, W = true.shape
    pred = pred[:H, :W]
    return 1.0 if np.array_equal(pred, true) else 0.0

# --------------------------
# Make prediction using CNN
# --------------------------
def make_prediction(model, input_grid, output_grid):
    inp = np.array(input_grid, dtype=np.int64)
    H, W = inp.shape
    padded = np.zeros((PAD_SIZE, PAD_SIZE), dtype=np.int64)
    padded[:H, :W] = inp

    x = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1,H,W)

    with torch.no_grad():
        logits = model(x)  # (1, C, PAD_SIZE, PAD_SIZE)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    target_h, target_w = np.array(output_grid).shape
    return pred[:target_h, :target_w]

# --------------------------
# Show input / prediction / output
# --------------------------
def show_file(json_data, file_name, train_idx, file_idx, file_count, running_stats):
    trains = json_data["train"]
    sample = trains[train_idx]

    inp = sample["input"]
    out = sample["output"]
    pred = make_prediction(MODEL, inp, out)

    correctness = compute_correctness(pred, out)
    running_stats["total"] += 1
    running_stats["correct"] += correctness

    plt.clf()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
    draw_grid(plt.subplot(gs[0]), inp, "Input")
    draw_grid(plt.subplot(gs[1]), pred, "Prediction")
    draw_grid(plt.subplot(gs[2]), out, "Ground Truth")

    accuracy_percent = (running_stats["correct"] / running_stats["total"]) * 100

    plt.suptitle(
        f"File: {file_name} ({file_idx+1}/{file_count}) | "
        f"Sample: {train_idx+1}/{len(trains)} | "
        f"Exact match: {'Yes' if correctness==1.0 else 'No'} | "
        f"Running accuracy: {accuracy_percent:.1f}%",
        fontsize=15
    )
    plt.pause(0.01)

# --------------------------
# UI loop
# --------------------------
def main():
    files = load_files(FOLDER)
    file_idx = 0
    train_idx = 0

    running_stats = {"total": 0, "correct": 0}

    plt.figure(figsize=(14,5))
    plt.ion()

    while True:
        file_name = files[file_idx]
        path = os.path.join(FOLDER, file_name)
        data = load_json(path)

        train_idx %= len(data["train"])

        show_file(data, file_name, train_idx, file_idx, len(files), running_stats)

        # Keyboard navigation
        if keyboard.is_pressed("right"):
            file_idx = (file_idx + 1) % len(files)
            train_idx = 0
            time.sleep(0.2)

        elif keyboard.is_pressed("left"):
            file_idx = (file_idx - 1) % len(files)
            train_idx = 0
            time.sleep(0.2)

        elif keyboard.is_pressed("down"):
            train_idx += 1
            time.sleep(0.15)

        elif keyboard.is_pressed("up"):
            train_idx -= 1
            time.sleep(0.15)

        elif keyboard.is_pressed("q"):
            # Print final overall correctness
            if running_stats["total"] > 0:
                overall_accuracy = (running_stats["correct"] / running_stats["total"]) * 100
                print(f"\nFinal Overall Exact Match Accuracy: {overall_accuracy:.2f}% "
                      f"({int(running_stats['correct'])}/{running_stats['total']})")
            else:
                print("\nNo samples were evaluated.")
            plt.close()
            break

if __name__ == "__main__":
    main()
