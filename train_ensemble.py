"""
Train a single model for bone disease classification.
"""
import os
import sys
import subprocess

NUM_ENSEMBLE_MODELS = 1

print(f"\n{'='*60}")
print(f"TRAINING: Training {NUM_ENSEMBLE_MODELS} model")
print(f"{'='*60}")
print(f"\nCUDA Environment Check:")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:80]}...")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'NOT SET')[:60]}...")
print(f"\n{'='*60}\n")

# Train single model
print(f"\n{'='*60}")
print(f"Training Model 1/1")
print(f"{'='*60}\n")

# Run trainMultiModal with custom seed
env = os.environ.copy()
env['ENSEMBLE_MODEL_ID'] = str(1)
env['ENSEMBLE_SEED'] = str(52)  # Fixed seed for reproducibility

print(f"  Seed: {env['ENSEMBLE_SEED']}")
print(f"  GPU config will inherit from parent environment\n")

result = subprocess.run(
    [sys.executable, 'trainMultiModal.py'],
    env=env,
    cwd=os.getcwd()
)

if result.returncode != 0:
    print(f"\n\u2717 Model training failed!")
    sys.exit(1)

print(f"\n\u2713 Model training completed!")

print(f"\n{'='*60}")
print(f"\u2713 Model trained successfully!")
print(f"{'='*60}\n")
print("Next step: Run ensemble_inference.py to evaluate accuracy\n")
