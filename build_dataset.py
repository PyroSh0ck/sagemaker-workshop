"""
Assembles bone_dataset.csv from multiple datasets.

Label mapping:
  0 = Normal
  1 = Bone Cancer
  2 = Osteoporosis
  3 = Bone Tumor
  4 = Scoliosis
  5 = Arthritis
  6 = Fracture
  7 = Sprain (abnormal upper limb)

Dataset download instructions:
  data/bone_cancer/     <- kaggle datasets download -d ziya07/bone-cancer-detection-dataset
  data/osteoporosis/    <- kaggle datasets download -d mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset
  data/arthritis/       <- kaggle datasets download -d shashwatwork/knee-osteoarthritis-dataset-with-severity
  data/fractures/       <- kaggle datasets download -d bmadushanirodrigo/fracture-multi-region-x-ray-data
  data/bone_tumor/      <- kaggle datasets download -d hazemalaa14/bone-tumor-x-ray-for-yolo-object-detection
  data/scoliosis/       <- kaggle datasets download -d salmankey/scoliosis-yolov5-annotated-spine-x-ray-dataset
  data/elbow_xray/      <- kaggle datasets download -d ahmedahmoud/elbow-xray-dataset
"""

import os
import glob
import pandas as pd

DATA_ROOT = "data"

# (folder, label_int, positive_only)
SOURCES = [
    # Normal
    ("osteoporosis/OS Collected Data/Normal",        0, False),
    ("arthritis/train/0",                            0, False),
    ("fractures/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/not fractured", 0, False),
    ("elbow_xray/XR_ELBOW_ALL/XR_ELBOW_ALL",        0, False),  # negative images filtered below
    # Osteoporosis
    ("osteoporosis/OS Collected Data/Osteoporosis",  2, False),
    # Bone Tumor (YOLO dataset - all images are tumor)
    ("bone_tumor/images/train",                      3, False),
    ("bone_tumor/images/val",                        3, False),
    # Scoliosis
    ("scoliosis/scoliosis yolov5/train/images",      4, False),
    ("scoliosis/scoliosis yolov5/valid/images",      4, False),
    # Arthritis (KL grades 1-4)
    ("arthritis/train/1",                            5, False),
    ("arthritis/train/2",                            5, False),
    ("arthritis/train/3",                            5, False),
    ("arthritis/train/4",                            5, False),
    # Fracture
    ("fractures/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/fractured", 6, False),
    # Sprain (positive/abnormal upper limb X-rays from MURA-style dataset)
    ("elbow_xray/XR_ELBOW_ALL/XR_ELBOW_ALL",        7, True),
    ("elbow_xray/XR_WRIST_ALL/XR_WRIST_ALL",        7, True),
    ("elbow_xray/XR_SHOULDER_ALL/XR_SHOULDER_ALL",  7, True),
]

rows = []

# Bone cancer uses filename-based labels (no subfolders)
CANCER_KEYWORDS = ["osteosarcoma", "chondrosarcoma", "ewing", "metastasis", "fibrosarcoma", "bone-cancer"]
NORMAL_KEYWORDS = ["normal", "image-no"]
for split in ("train", "valid", "test"):
    bc_dir = os.path.join(DATA_ROOT, "bone_cancer", split)
    if not os.path.exists(bc_dir):
        continue
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in glob.glob(os.path.join(bc_dir, ext)):
            fname = os.path.basename(path).lower()
            if any(k in fname for k in CANCER_KEYWORDS):
                label = 1
            elif any(k in fname for k in NORMAL_KEYWORDS):
                label = 0
            else:
                continue
            rows.append({"image_path": path, "age": None, "bp_sys": None,
                         "bp_dia": None, "spo2": None, "calcium": None, "label": label})

# All other sources
for rel_folder, label, positive_only in SOURCES:
    folder = os.path.join(DATA_ROOT, rel_folder)
    if not os.path.exists(folder):
        print(f"[SKIP] {folder} not found")
        continue
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in glob.glob(os.path.join(folder, "**", ext), recursive=True):
            fname = os.path.basename(path).lower()
            is_positive = "positive" in fname
            if positive_only and not is_positive:
                continue
            if label == 0 and "elbow_xray" in rel_folder and is_positive:
                continue
            rows.append({"image_path": path, "age": None, "bp_sys": None,
                         "bp_dia": None, "spo2": None, "calcium": None, "label": label})

df = pd.DataFrame(rows)
df.to_csv("bone_dataset.csv", index=False)
print(f"Saved {len(df)} rows to bone_dataset.csv")
print(df["label"].value_counts().sort_index())
