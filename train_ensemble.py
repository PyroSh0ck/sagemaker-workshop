"""
Train ensemble of 3 models with different random seeds for stronger predictions.
Each model gets different data shuffling and augmentation, reducing overfitting.
"""
import os
import sys
import subprocess

NUM_ENSEMBLE_MODELS = 3

print(f"\n{'='*60}")
print(f"ENSEMBLE TRAINING: Training {NUM_ENSEMBLE_MODELS} models")
print(f"{'='*60}")
print(f"\nCUDA Environment Check:")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:80]}...")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')[:60]}...")
print(f"\n{'='*60}\n")

for model_id in range(1, NUM_ENSEMBLE_MODELS + 1):
    print(f"\n{'='*60}")
    print(f"Training Ensemble Model {model_id}/{NUM_ENSEMBLE_MODELS}")
    print(f"{'='*60}\n")
    
    # Run trainMultiModal with custom seed for this model
    # Copy entire parent environment (including CUDA paths)
    env = os.environ.copy()
    env['ENSEMBLE_MODEL_ID'] = str(model_id)
    env['ENSEMBLE_SEED'] = str(42 + model_id * 10)  # Different seed per model
    
    # Ensure CUDA environment persists
    print(f"  Seed: {env['ENSEMBLE_SEED']}")
    print(f"  GPU config will inherit from parent environment\n")
    
    result = subprocess.run(
        [sys.executable, 'trainMultiModal.py'],
        env=env,
        cwd=os.getcwd()
    )
    
    if result.returncode != 0:
        print(f"\n✗ Model {model_id} training failed!")
        sys.exit(1)
    
    print(f"\n✓ Model {model_id} training completed!")

print(f"\n{'='*60}")
print(f"✓ All {NUM_ENSEMBLE_MODELS} ensemble models trained successfully!")
print(f"{'='*60}\n")
print("Next step: Run ensemble_inference.py to combine predictions\n")
