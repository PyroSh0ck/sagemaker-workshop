#!/bin/bash
set -e

echo "=== GPU Detection & CUDA Setup ==="
# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found. GPU hardware detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found. GPU support may not be available."
fi

# Set CUDA library paths for TensorFlow to find GPU libraries
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add NVIDIA CUDA packages installed by pip (from conda site-packages)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH

export CUDA_PATH=$CUDA_HOME
echo "Set CUDA paths:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  LD_LIBRARY_PATH updated with CUDA libraries"

echo ""
echo "=== Step 1: Installing dependencies (including NVIDIA CUDA libraries) ==="
pip install --no-cache-dir -r requirements.txt -q

# Install NVIDIA libraries separately to ensure they're properly linked
echo "Installing NVIDIA CUDA Runtime Libraries..."
pip install --no-cache-dir nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 -q

# Verify TensorFlow can detect GPU
echo ""
echo "=== Verifying TensorFlow GPU Support ==="
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow detects {len(gpus)} GPU(s)')
if len(gpus) > 0:
    print('  ✓ GPU properly configured!')
    for gpu in gpus:
        print(f'    - {gpu}')
else:
    print('  ✗ WARNING: No GPUs detected by TensorFlow')
"

echo ""
echo "=== Step 2: Downloading datasets from S3 ==="
aws s3 sync s3://sagemaker-bone-xray-baba/data/ data/
aws s3 cp s3://sagemaker-bone-xray-baba/bone_dataset.csv bone_dataset.csv

echo ""
echo "=== Step 3: Generating MURA CSVs ==="
cd MURA
python generate_mura_csvs.py -q 2>&1 | grep -v "^$" || true

echo ""
echo "=== Step 4: Fine-tuning EfficientNet on MURA ==="
python EfficientNetFineTune.py

cd ..

echo ""
echo "=== Step 5: Training multimodal model ==="
python trainMultiModal.py

echo ""
echo "=== Done! Model saved to Models/best_multimodal_model.keras ==="
