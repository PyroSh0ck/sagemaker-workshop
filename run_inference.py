#!/usr/bin/env python3
"""
Run ensemble inference on SageMaker.
Can be executed as a SageMaker training job or directly on notebook.
"""
import subprocess
import sys
import os

def main():
    print("\n" + "="*60)
    print("SINGLE MODEL INFERENCE RUNNER")
    print("="*60 + "\n")
    
    # Check if a usable model exists
    models_dir = "Models"
    model_candidates = [
        "ensemble_model_1.keras",
        "best_multimodal_model.keras",
        "multimodal_model.keras",
    ]
    
    print(f"Checking for model files in {models_dir}/...")
    found_model = None
    for model in model_candidates:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024**2)
            print(f"  ✓ {model} ({size_mb:.1f} MB)")
            if found_model is None:
                found_model = model
    
    if found_model is None:
        print("\n❌ ERROR: No supported model artifact found.")
        print(f"Expected one of: {model_candidates}")
        print("Run train_ensemble.py first to generate a model.")
        return 1

    print(f"\nUsing model: {found_model}")
    
    print("\n" + "-"*60)
    print("Starting inference...")
    print("-"*60 + "\n")
    
    # Run ensemble inference
    try:
        result = subprocess.run(
            [sys.executable, "ensemble_inference.py"],
            check=True
        )
        
        print("\n" + "="*60)
        print("✓ INFERENCE COMPLETE")
        print("="*60)
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Inference failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
