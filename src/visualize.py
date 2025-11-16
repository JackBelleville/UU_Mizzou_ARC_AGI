import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys

FOLDER = "../data/example_submission"

COLOR_MAP = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

def load_files(folder):
    """Loads and sorts all .json files from a folder."""
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()
    return files

def load_json(path):
    """Loads a single JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def draw_grid(ax, grid, title):
    """Draws a single grid on the given matplotlib axis."""
    grid = np.array(grid)
    ax.clear()  # Clear the axis before drawing
    ax.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(COLOR_MAP), interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

class ARCViewer:
    """
    An interactive viewer for ARC dataset files.

    Navigation:
      - Left/Right: Change file
      - Up/Down:     Change training example within the file
      - Q:           Quit
    """
    def __init__(self, folder, files=None):
        self.folder = folder
        if not files:
            self.files = load_files(self.folder)
            if not self.files:
                raise FileNotFoundError(f"No .json files found in {self.folder}") 
        else:
            self.files = files
                

        self.file_idx = 0
        self.train_idx = 0
        self.data = None

        # Set up the figure and axes
        self.fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])

        # Connect the event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Load and display the first item
        self.load_and_draw()

    def load_and_draw(self):
        """Loads the current file's data and redraws the plot."""
        # Load new file data if the file index changed
        file_name = self.files[self.file_idx]
        path = os.path.join(self.folder, file_name)
        self.data = load_json(path)

        # Ensure train_idx is valid for the current file
        num_train = len(self.data["train"])
        self.train_idx = self.train_idx % num_train
        
        sample = self.data["train"][self.train_idx]

        # Draw the grids
        draw_grid(self.ax1, sample["input"], "Input")
        draw_grid(self.ax2, sample["output"], "Output")

        # Update the suptitle
        self.fig.suptitle(
            f"File: {file_name} ({self.file_idx + 1}/{len(self.files)}) | "
            f"Example: {self.train_idx + 1}/{num_train}",
            fontsize=16
        )

        # Redraw the canvas
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Handles key press events."""
        if event.key == 'right':
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.train_idx = 0  # Reset example index
            self.load_and_draw()
        
        elif event.key == 'left':
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.train_idx = 0
            self.load_and_draw()
            
        elif event.key == 'down':
            self.train_idx += 1
            # We only need to redraw, not reload the file
            self.load_and_draw()
            
        elif event.key == 'up':
            self.train_idx -= 1
            self.load_and_draw()
            
        elif event.key == 'q':
            plt.close(self.fig)

def main():
    viewer = None
    if not sys.argv[1:]:
        if not os.path.isdir(FOLDER):
            print(f"Error: Folder not found: {FOLDER}")
            print("Please ensure the 'FOLDER' variable points to your data directory.")
            return
        # Create the viewer instance
        viewer = ARCViewer(FOLDER)
    else:
        with open(sys.argv[1], "r") as id_list:
            ids = [line.strip() for line in id_list]
            viewer = ARCViewer(FOLDER, ids)

    
    # Show the plot and start the event loop
    plt.show()

if __name__ == "__main__":
    main()
