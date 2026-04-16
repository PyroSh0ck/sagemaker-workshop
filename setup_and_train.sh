#!/bin/bash
set -e

echo "=== Step 1: Installing dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Step 2: Downloading datasets from S3 ==="
aws s3 sync s3://sagemaker-bone-xray-baba/data/ data/
aws s3 cp s3://sagemaker-bone-xray-baba/bone_dataset.csv bone_dataset.csv

echo ""
echo "=== Step 3: Generating MURA CSVs ==="
cd MURA
python generate_mura_csvs.py

echo ""
echo "=== Step 4: Fine-tuning EfficientNet on MURA ==="
XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/conda python EfficientNetFineTune.py

cd ..

echo ""
echo "=== Step 5: Training multimodal model ==="
python trainMultiModal.py

echo ""
echo "=== Done! Model saved to Models/best_multimodal_model.keras ==="
