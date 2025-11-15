import os
import json
import numpy as np

FOLDER = "../data/training"

def load_json(path):
    """Loads a single JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def check_task_is_easy(task_data):
    """
    Applies a set of heuristics to determine if a task is "easy".
    Returns True if the task passes all checks, False otherwise.
    """
    try:
        # Check all training pairs
        for pair in task_data["train"]:
            input_grid = np.array(pair["input"])
            output_grid = np.array(pair["output"])

            # --- Heuristic 1: Identical Dimensions ---
            if input_grid.shape != output_grid.shape:
                return False

            # --- Heuristic 2: Small Grid ---
            # We'll set a max size of 15x15 (225 pixels)
            if input_grid.size > 225:
                return False

            # --- Heuristic 3: Low Color Count ---
            # Max 4 colors (e.g., background + 3 others)
            if len(np.unique(input_grid)) > 4:
                return False

            # --- Heuristic 4: No New Colors ---
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            if not output_colors.issubset(input_colors):
                return False

        # If we get here, all training pairs passed all checks
        return True

    except Exception as e:
        print(f"  Error processing task: {e}")
        return False

def main():
    files = [f for f in os.listdir(FOLDER) if f.endswith(".json")]
    files.sort()
    
    easy_task_files = []

    print(f"Scanning {len(files)} tasks...")

    for file_name in files:
        path = os.path.join(FOLDER, file_name)
        data = load_json(path)
        
        if check_task_is_easy(data):
            easy_task_files.append(file_name)
            
    print("\n--- Found Easy Tasks ---")
    for file_name in easy_task_files:
        print(file_name)
        
    print(f"\nTotal: {len(easy_task_files)} / {len(files)}")

    # Write the list to a file for your training script to read
    with open("easy_tasks.txt", "w") as f:
        for file_name in easy_task_files:
            f.write(f"{file_name}\n")
    
    print("\nSaved list to 'easy_tasks.txt'")

if __name__ == "__main__":
    main()
