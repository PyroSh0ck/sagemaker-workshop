#!/bin/bash
set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   PRODUCTION TRAINING: Ensemble + Test-Time Augmentation      ║"
echo "║   Expected accuracy: 90%+ (vs 89.41% single model)            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 0: CUDA Setup FIRST (before any Python execution)
echo "=== CUDA Library Path Configuration ==="
export CUDA_HOME=/usr/local/cuda
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Build LD_LIBRARY_PATH with all NVIDIA CUDA libraries from pip
export LD_LIBRARY_PATH="\
$CUDA_HOME/lib64:\
$SITE_PACKAGES/nvidia/cuda_runtime/lib:\
$SITE_PACKAGES/nvidia/cudnn/lib:\
$SITE_PACKAGES/nvidia/cublas/lib:\
$SITE_PACKAGES/nvidia/cusolver/lib:\
$SITE_PACKAGES/nvidia/cusparse/lib:\
$SITE_PACKAGES/nvidia/cufft/lib:\
$SITE_PACKAGES/nvidia/curand/lib:\
$SITE_PACKAGES/nvidia/nccl/lib:\
$LD_LIBRARY_PATH"

# Also set XLA and TensorFlow environment variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda --xla_gpu_enable_triton_gemm=false"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_ENABLE_ONEDNN_OPTS="0"

echo "CUDA_HOME: $CUDA_HOME"
echo "SITE_PACKAGES: $SITE_PACKAGES"
echo "LD_LIBRARY_PATH configured ✓"
echo ""

# Step 1: GPU Detection
echo "=== GPU Detection & Verification ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU hardware detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "✗ No nvidia-smi found - GPU may not be available"
fi

echo ""
echo "=== Step 2: Installing dependencies ==="
pip install -r requirements.txt -q

# Verify TensorFlow can see GPU
echo ""
echo "=== Verifying TensorFlow GPU Detection ==="
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow detects {len(gpus)} GPU(s)')
if len(gpus) > 0:
    print(f'  GPU 0: {gpus[0]}')
    print('  ✓ GPU properly configured!')
else:
    print('  ✗ WARNING: No GPUs detected by TensorFlow')
    print('  This may cause very slow training')
"

# Step 3: Data prep
echo ""
echo "=== Step 3: Downloading datasets from S3 ==="
aws s3 sync s3://sagemaker-bone-xray-baba/data/ data/ --quiet
aws s3 cp s3://sagemaker-bone-xray-baba/bone_dataset.csv bone_dataset.csv --quiet

echo ""
echo "=== Step 4: Generating MURA CSVs ==="
cd MURA
python generate_mura_csvs.py -q 2>&1 | grep -v "^$" || true
python EfficientNetFineTune.py
cd ..

# Step 5: Train ensemble
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   TRAINING ENSEMBLE (3 models with different random seeds)    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
python train_ensemble.py

# Step 6: Ensemble inference with TTA
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
