import json
from pathlib import Path

def save_predictions_per_file(inputs, predictions, truths=None, out_path=None):
    """
    Save predictions for a single ARC task to JSON.
    Each call should correspond to one task/file.
    
    inputs: list of input grids (lists of lists)
    predictions: list of predicted output grids
    truths: optional list of true outputs
    out_path: Path or str to save the JSON file
    """
    if out_path is None:
        raise ValueError("Must provide out_path for JSON file")

    out_path = Path(out_path)
    data = {"train": [], "test": []}

    # If we have ground truth, save as train
    if truths is not None:
        for inp, true in zip(inputs, truths):
            data["train"].append({
                "input": inp,
                "output": true
            })

    # Save predictions as test samples
    for inp, pred in zip(inputs, predictions):
        data["test"].append({
            "input": inp,
            "output": pred
        })

    # Ensure parent folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved prediction for {out_path.name}")
