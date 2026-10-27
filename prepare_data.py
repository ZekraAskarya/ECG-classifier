import os
import shutil

# Path to your downloaded Kaggle dataset
KAGGLE_DIR = "Dataset_ECG"

# The exact folder names the pipeline expects
TARGET_MAP = {
    "MI": "data/ECG Images of Myocardial Infarction Patients",
    "HMI": "data/ECG Images of Patient that have History of MI",
    "AHB": "data/ECG Images of Patient that have abnormal heartbeat",
    "Normal": "data/Normal Person ECG Images"
}

# Create the target folders if they don't exist
for folder in TARGET_MAP.values():
    os.makedirs(folder, exist_ok=True)

# Loop through train, test, and valid
splits = ["train", "test", "valid"]
count = 0

print("Starting to copy images...")

for split in splits:
    split_path = os.path.join(KAGGLE_DIR, split)
    if not os.path.exists(split_path):
        print(f"⚠️ Folder not found: {split_path}")
        continue

    for class_folder, target_folder in TARGET_MAP.items():
        class_path = os.path.join(split_path, class_folder)
        if not os.path.exists(class_path):
            print(f"⚠️ Class folder not found: {class_path}")
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(class_path, filename)
                # Rename to avoid duplicate filename collisions
                dst = os.path.join(target_folder, f"{split}_{filename}")
                shutil.copy(src, dst)
                count += 1

print(f"\n✅ Successfully copied {count} images into the data/ folder!")
