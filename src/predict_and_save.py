# predict_and_save.py
import json
from pathlib import Path
import torch
from torch_model import SmallUNet, pad_to_30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
def load_model(weights_path="best_arc_cnn.pth"):
    model = SmallUNet().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

# -----------------------------
# Predict a single input grid
# -----------------------------
def predict_grid(model, grid):
    padded = pad_to_30(grid)
    x = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
    return pred

# -----------------------------
# Save predictions for a single task
# -----------------------------
def save_predictions_per_file(inputs, predictions, truths=None, out_path=None):
    if out_path is None:
        raise ValueError("Must provide out_path for JSON file")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"train": [], "test": []}

    # If ground truth exists, save as train
    if truths is not None:
        for inp, true in zip(inputs, truths):
            data["train"].append({"input": inp, "output": true})

    # Save predictions as test
    for inp, pred in zip(inputs, predictions):
        data["test"].append({"input": inp, "output": pred})

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved prediction for {out_path.name}")

# -----------------------------
# Predict folder
# -----------------------------
def predict_folder(input_folder="../data/evaluation", output_folder="../data/predictions(evaluation)"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = load_model()

    for task_file in input_folder.glob("*.json"):
        try:
            with open(task_file, "r") as f:
                task = json.load(f)

            predictions = []
            inputs = []
            truths = []

            # Check if the task has "train" or "test" cases
            if "train" in task:
                for case in task["train"]:
                    inp = case["input"]
                    out = case["output"]
                    pred = predict_grid(model, inp)
                    predictions.append(pred.tolist())
                    inputs.append(inp)
                    truths.append(out)

            if "test" in task:
                for case in task["test"]:
                    inp = case["input"]
                    pred = predict_grid(model, inp)
                    predictions.append(pred.tolist())
                    inputs.append(inp)

            # Save predictions for this task as a separate JSON
            out_path = output_folder / task_file.name
            save_predictions_per_file(inputs, predictions, truths if truths else None, out_path)

        except Exception as e:
            print(f"Error on {task_file}: {e}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    predict_folder()
