import json
import numpy as np
from pathlib import Path
from typing import List


####################################################
# Helpers for handling ARC-style rectangular grids
####################################################

def to_array(grid: List[List[int]]) -> np.ndarray:
    """Convert list-of-lists grid to numpy array."""
    return np.array(grid, dtype=np.int32)


def load_arc_json(path: str):
    """Load a single ARC JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["train"]  # your format


def load_all_arc_tasks(folder: str):
    """
    Load *all* ARC training examples from every JSON file in a folder.
    """
    folder = Path(folder)
    all_train_data = []

    for json_file in folder.glob("*.json"):
        print(f"Loading {json_file.name}")
        train_data = load_arc_json(json_file)
        all_train_data.extend(train_data)

    print(f"Total training samples loaded: {len(all_train_data)}")
    return all_train_data


####################################################
# Padding (NEW — fixes vstack dimension crash)
####################################################

def pad_to_30x30(grid: np.ndarray) -> np.ndarray:
    """
    Pad grid to fixed size 30x30 (max ARC size).
    """
    H, W = grid.shape
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[:H, :W] = grid
    return padded


####################################################
# Feature extraction (uses padding)
####################################################

def extract_features(grid: np.ndarray) -> np.ndarray:
    """
    Extract features from the ARC grid.
    Now uses padding to avoid size mismatch errors.
    """
    grid = pad_to_30x30(grid)
    return grid.flatten()  # 900 features always


####################################################
# Dummy model (replace later)
####################################################

class DummyModel:
    """
    A trivial model to show structure.
    Replace with:
      - XGBoost
      - sklearn MLP
      - PyTorch NN
      - Your ARC rule engine
    """

    def fit(self, X, Y):
        print("Training on", X.shape, "inputs and", Y.shape, "labels")
        self.mean_output = np.mean(Y, axis=0)

    def predict(self, X):
        return np.repeat(self.mean_output[None, :], X.shape[0], axis=0)


####################################################
# Prepare dataset for training
####################################################

def prepare_dataset(train_data):
    X_list = []
    Y_list = []

    for item in train_data:
        inp = to_array(item["input"])
        out = to_array(item["output"])

        X_list.append(extract_features(inp))
        Y_list.append(pad_to_30x30(out).flatten())

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    return X, Y


####################################################
# Full training pipeline
####################################################

def train_model(path: str):
    path = Path(path)

    if path.is_dir():
        train_data = load_all_arc_tasks(path)
    else:
        train_data = load_arc_json(path)

    X, Y = prepare_dataset(train_data)

    model = DummyModel()
    model.fit(X, Y)

    return model


####################################################
# Inference
####################################################

def solve(model, grid: List[List[int]]) -> np.ndarray:
    """Solve a new ARC input grid."""
    arr = to_array(grid)
    feat = extract_features(arr).reshape(1, -1)
    pred = model.predict(feat)
    return pred.reshape(30, 30)  # return full grid shape


####################################################
# Main
####################################################

if __name__ == "__main__":
    # Train on a folder containing many ARC JSON files
    model = train_model("../data/training")

    # Example test grid
    test_grid = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]

    result = solve(model, test_grid)

    print("Prediction shape:", result.shape)
    print(result)
