import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import keyboard
import time

FOLDER = "ARC-AGI-2/data/evaluation"

COLOR_MAP = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

def load_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()
    return files

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def draw_grid(ax, grid, title):
    grid = np.array(grid)
    ax.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(COLOR_MAP), interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

def show_file(json_data, file_name, train_idx, file_idx, file_count):
    trains = json_data["train"]
    sample = trains[train_idx]

    plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    draw_grid(ax1, sample["input"], "Input")

    ax2 = plt.subplot(gs[1])
    draw_grid(ax2, sample["output"], "Output")

    plt.suptitle(
        f"File: {file_name} ({file_idx+1}/{file_count})   |   Train Sample: {train_idx+1}/{len(trains)}",
        fontsize=16
    )
    plt.pause(0.01)

def main():
    files = load_files(FOLDER)
    file_idx = 0
    train_idx = 0

    plt.figure(figsize=(10, 5))
    plt.ion()

    while True:
        file_name = files[file_idx]
        path = os.path.join(FOLDER, file_name)
        data = load_json(path)

        train_idx %= len(data["train"])

        show_file(data, file_name, train_idx, file_idx, len(files))

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
            plt.close()
            break

if __name__ == "__main__":
    main()
