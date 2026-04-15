# Bone Disease Classification — AWS SageMaker Workshop

A multimodal deep learning model that classifies bone diseases from X-ray images, built and trained on AWS SageMaker. Developed as a demo for an AWS club workshop.

---

## What It Does

The model takes an X-ray image (and optionally patient vitals) and classifies it into one of 8 categories:

| Label | Disease |
|-------|---------|
| 0 | Normal |
| 1 | Bone Cancer |
| 2 | Osteoporosis |
| 3 | Bone Tumor |
| 4 | Scoliosis |
| 5 | Arthritis |
| 6 | Fracture |
| 7 | Sprain |

---

## Architecture

The model is a **multimodal neural network** combining two branches:

- **Vision branch** — EfficientNet-V2-S pre-trained on ImageNet, then fine-tuned on the [MURA dataset](https://www.kaggle.com/datasets/cjinny/mura-v11) (36,808 musculoskeletal X-rays) to recognize abnormal bone structure
- **Tabular branch** — MLP that processes patient vitals (age, blood pressure, SpO2, calcium)
- Both branches are fused and passed through a classification head

---

## Project Structure

```
├── MURA/
│   ├── generate_mura_csvs.py     # Step 1a: prepares MURA image path CSVs
│   └── EfficientNetFineTune.py   # Step 1b: fine-tunes EfficientNet on MURA
├── data/
│   ├── arthritis/                # Knee Osteoarthritis Dataset (KL grading)
│   ├── bone_cancer/              # Bone Cancer Detection Dataset
│   ├── bone_tumor/               # Bone Tumor X-ray (YOLO)
│   ├── elbow_xray/               # MURA-style upper limb X-rays (sprains)
│   ├── fractures/                # Bone Fracture Binary Classification
│   ├── mura/                     # MURA-v1.1 (pre-training)
│   ├── osteoporosis/             # Multi-Class Knee Osteoporosis X-ray
│   └── scoliosis/                # Scoliosis YOLOv5 Annotated Spine X-ray
├── Models/                       # Output folder for saved models
├── build_dataset.py              # Step 2: assembles bone_dataset.csv
├── trainMultiModal.py            # Step 3: trains the full multimodal model
├── bone_dataset.csv              # Generated dataset index (37,198 images)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API
Create `~/.kaggle/kaggle.json`:
```json
{"username": "<your_username>", "key": "<your_api_key>"}
```
Get your key from [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token.

### 3. Download datasets
All datasets are already downloaded into `data/`. If you need to re-download:
```bash
python -c "
from kaggle import api; api.authenticate()
api.dataset_download_files('ziya07/bone-cancer-detection-dataset',                          path='data/bone_cancer',   unzip=True)
api.dataset_download_files('mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset',     path='data/osteoporosis',  unzip=True)
api.dataset_download_files('shashwatwork/knee-osteoarthritis-dataset-with-severity',        path='data/arthritis',     unzip=True)
api.dataset_download_files('bmadushanirodrigo/fracture-multi-region-x-ray-data',            path='data/fractures',     unzip=True)
api.dataset_download_files('hazemalaa14/bone-tumor-x-ray-for-yolo-object-detection',        path='data/bone_tumor',    unzip=True)
api.dataset_download_files('salmankey/scoliosis-yolov5-annotated-spine-x-ray-dataset',      path='data/scoliosis',     unzip=True)
api.dataset_download_files('ahmedahmoud/elbow-xray-dataset',                                path='data/elbow_xray',    unzip=True)
api.dataset_download_files('cjinny/mura-v11',                                               path='data/mura',          unzip=True)
"
```

---

## Training

Run the following steps in order.

### Step 1 — Fine-tune EfficientNet on MURA
Run from the `MURA/` directory:
```bash
cd MURA
python generate_mura_csvs.py    # fixes and copies MURA path CSVs
python EfficientNetFineTune.py  # trains for 5 epochs, saves ../Models/mura_efficientnet.keras
cd ..
```

### Step 2 — Build the dataset CSV
```bash
python build_dataset.py         # scans all data/ folders, outputs bone_dataset.csv
```

### Step 3 — Train the multimodal model
```bash
python trainMultiModal.py       # trains for up to 20 epochs, saves best_multimodal_model.keras
```

Training uses:
- **Class weighting** to handle imbalanced classes (Osteoporosis: 793 images vs Normal: 12,105)
- **Early stopping** with patience 3
- **ReduceLROnPlateau** to halve learning rate when val_loss stalls
- **Data augmentation** (random flip, brightness, contrast) on training set

---

## AWS Infrastructure

| Service | Purpose |
|---------|---------|
| **S3** | Stores datasets, training scripts, and saved models |
| **SageMaker Studio** | JupyterLab environment for running training notebooks |
| **SageMaker Training Jobs** | `ml.c5.4xlarge` ($0.816/hr) for model training |
| **Lambda + API Gateway** | Inference endpoint accessible by any application |
| **CloudWatch** | Usage monitoring and cost alerts |

### S3 Bucket Setup
Name your bucket with `sagemaker` in the name (e.g. `your-sagemaker-bone-xray`) — SageMaker automatically gets read/write access to any bucket with `sagemaker` in the name without extra IAM configuration.

### SageMaker Setup
1. Create an IAM user with least-privilege permissions (S3 read/write on your bucket, SageMaker full access)
2. Create a SageMaker Domain
3. Open SageMaker Studio → JupyterLab → Create Space → Run Space
4. Upload scripts and reference your S3 bucket:
```python
s3_path = 's3://your-sagemaker-bone-xray/data/bone_dataset.csv'
```

---

## Security Notes (HIPAA Considerations)

Since this handles medical imaging data, the following should be in place for any real deployment:

- **S3**: Enable SSE-KMS encryption, block all public access, enable versioning
- **SageMaker**: Run training instances inside a VPC (no public internet access)
- **IAM**: Least-privilege roles — scope S3 access to the specific bucket only, never use `s3:*`
- **API Gateway**: Use IAM auth or API keys on the inference endpoint, never leave it open
- **CloudWatch Logs**: Set retention policies so patient data doesn't persist indefinitely
- **Data**: All datasets used here are de-identified public datasets — no PHI. A real deployment would require a HIPAA BAA with AWS

---

## Dataset Summary

| Disease | Dataset | Images |
|---------|---------|--------|
| Normal | Multiple sources | 12,105 |
| Bone Cancer | ziya07/bone-cancer-detection-dataset | 2,915 |
| Osteoporosis | mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset | 793 |
| Bone Tumor | hazemalaa14/bone-tumor-x-ray-for-yolo-object-detection | 1,867 |
| Scoliosis | salmankey/scoliosis-yolov5-annotated-spine-x-ray-dataset | 1,635 |
| Arthritis | shashwatwork/knee-osteoarthritis-dataset-with-severity | 3,492 |
| Fracture | bmadushanirodrigo/fracture-multi-region-x-ray-data | 4,606 |
| Sprain | ahmedahmoud/elbow-xray-dataset (positive studies) | 9,785 |
| **Total** | | **37,198** |
