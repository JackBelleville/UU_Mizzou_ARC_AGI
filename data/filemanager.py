# copys files from training and combines the easy ones from training_rotated and puts it into training_easy 

import os
import shutil

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
EASY_LIST_FILE = "../src/easy_tasks.txt"  # list of JSON names
TRAINING_DIR = "training"
TRAINING_ROTATED_DIR = "training_rotated"
OUTPUT_DIR = "training_easy"

# ---------------------------------------------
# LOAD EASY TASK NAMES
# ---------------------------------------------
with open(EASY_LIST_FILE, "r") as f:
    easy_tasks = [line.strip() for line in f.readlines() if line.strip()]

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------
# FUNCTION TO COPY FILE WITH OPTIONAL RENAME
# ---------------------------------------------
def safe_copy(src, dst_dir, rename_suffix=None):
    filename = os.path.basename(src)
    if rename_suffix:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{rename_suffix}{ext}"
    dst = os.path.join(dst_dir, filename)
    shutil.copy2(src, dst)
    return dst

# ---------------------------------------------
# MAIN PROCESSING LOGIC
# ---------------------------------------------
count_train = 0
count_rot = 0

for task in easy_tasks:
    found_in_training = False
    # 1. Search in training first
    train_path = os.path.join(TRAINING_DIR, task)
    if os.path.isfile(train_path):
        safe_copy(train_path, OUTPUT_DIR)
        found_in_training = True
        count_train += 1

    # 2. Then search in training_rotated
    rot_path = os.path.join(TRAINING_ROTATED_DIR, task)
    if os.path.isfile(rot_path):
        if found_in_training:
            safe_copy(rot_path, OUTPUT_DIR, rename_suffix="training_rotated")
        else:
            safe_copy(rot_path, OUTPUT_DIR)
        count_rot += 1

print(f"Copied {count_train} training files and {count_rot} training_rotated files into {OUTPUT_DIR}.")
