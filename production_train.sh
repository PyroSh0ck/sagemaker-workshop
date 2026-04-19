#!/bin/bash
set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   PRODUCTION TRAINING: Ensemble + Test-Time Augmentation      ║"
echo "║   Expected accuracy: 90%+ (vs 89.41% single model)            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: GPU Setup (same as before)
echo "=== GPU Detection & CUDA Setup ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU hardware detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "✗ No GPU detected - training will be slow"
fi

# Set CUDA library paths
export CUDA_HOME=/usr/local/cuda
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH

echo ""
echo "=== Step 1: Installing dependencies ==="
pip install -r requirements.txt -q

# Step 2: Data prep (same as before)
echo ""
echo "=== Step 2: Downloading datasets from S3 ==="
aws s3 sync s3://sagemaker-bone-xray-baba/data/ data/ --quiet
aws s3 cp s3://sagemaker-bone-xray-baba/bone_dataset.csv bone_dataset.csv --quiet

echo ""
echo "=== Step 3: Generating MURA CSVs ==="
cd MURA
python generate_mura_csvs.py -q 2>&1 | grep -v "^$" || true
python EfficientNetFineTune.py
cd ..

# Step 4: Train ensemble
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   TRAINING ENSEMBLE (3 models with different random seeds)    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
python train_ensemble.py

# Step 5: Ensemble inference with TTA
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   ENSEMBLE INFERENCE WITH TEST-TIME AUGMENTATION              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
python ensemble_inference.py

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✓ PRODUCTION TRAINING COMPLETE               ║"
echo "║                                                                ║"
echo "║   Ensemble models saved to: Models/ensemble_model_{1,2,3}.keras  ║"
echo "║   For deployment, use: python ensemble_inference.py           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
