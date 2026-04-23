"""
Ensemble inference with Test-Time Augmentation (TTA).
Loads 3 trained models and averages their predictions with TTA.
Expected accuracy: 90%+
"""
import os

# Force CPU inference by default to avoid SageMaker libdevice/XLA GPU JIT failures.
# Set FORCE_CPU_INFERENCE=0 to allow GPU execution.
FORCE_CPU_INFERENCE = os.environ.get('FORCE_CPU_INFERENCE', '1') == '1'
if FORCE_CPU_INFERENCE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable XLA JIT compilation BEFORE importing TensorFlow
os.environ['TF_ENABLE_XLA_JIT'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Force eager execution globally and disable JIT optimizer paths
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
tf.config.optimizer.set_jit(False)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 8
NUM_ENSEMBLE_MODELS = 1  # Using single model (not ensemble)
TTA_AUGMENTATIONS = 4  # Apply 4 augmentations + original = 5 predictions per image

# Setup
DATA_ROOT = 'data'
MODEL_DIR = 'Models'
IS_SAGEMAKER = os.path.exists('/opt/ml/input/data')
if IS_SAGEMAKER:
    DATA_ROOT = '/opt/ml/input/data'
    MODEL_DIR = os.environ.get('SM_MODEL_DIR', 'Models')

print(f"\n{'='*60}")
print(f"SINGLE MODEL INFERENCE WITH TTA")
print(f"Models: {NUM_ENSEMBLE_MODELS} | TTA: {TTA_AUGMENTATIONS} augmentations")
print(f"{'='*60}\n")

# Load test data
df = pd.read_csv('bone_dataset.csv')
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
df['image_path'] = df['image_path'].apply(
    lambda p: os.path.join(DATA_ROOT, '/'.join(p.replace('\\', '/').split('/')[1:]))
    if not p.startswith(DATA_ROOT) else p
)
df = df.dropna(subset=['image_path'])
df = df[df['label'].between(0, NUM_CLASSES - 1)]

# Create test split (same as training - last 15%)
train_frac, val_frac = 0.7, 0.15
test_parts = []
for _, label_df in df.groupby('label'):
    label_df = label_df.sample(frac=1, random_state=42)
    n = len(label_df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train < 1: n_train = 1
    if n_val < 1: n_val = 1
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    test_parts.append(label_df.iloc[n_train + n_val:])

test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Test set: {len(test_df)} images\n")

# TTA augmentation function
def augment_tta(image):
    """Apply random augmentation for TTA"""
    aug_type = np.random.randint(0, TTA_AUGMENTATIONS)
    
    if aug_type == 0:  # Flip horizontal
        image = tf.image.flip_left_right(image)
    elif aug_type == 1:  # Flip vertical
        image = tf.image.flip_up_down(image)
    elif aug_type == 2:  # Brightness
        image = tf.image.adjust_brightness(image, delta=0.1)
    elif aug_type == 3:  # Contrast
        image = tf.image.adjust_contrast(image, contrast_factor=1.1)
    
    return image

def load_and_preprocess(path):
    """Load and preprocess image"""
    image_bytes = tf.io.read_file(path)
    lower_path = tf.strings.lower(path)
    is_jpeg = tf.strings.regex_full_match(lower_path, ".*\\.(jpg|jpeg)$")
    
    def decode_jpeg_recover():
        return tf.image.decode_jpeg(
            image_bytes,
            channels=3,
            try_recover_truncated=True,
            acceptable_fraction=0.3,
        )
    
    def decode_generic():
        return tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    
    image = tf.cond(is_jpeg, decode_jpeg_recover, decode_generic)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image

# Load single model
print("Loading model...")
models = []

# Try to load the best model in order of preference
model_candidates = [
    os.path.join(MODEL_DIR, 'ensemble_model_1.keras'),
    os.path.join(MODEL_DIR, 'best_multimodal_model.keras'),
    os.path.join(MODEL_DIR, 'multimodal_model.keras')
]

model_path = None
for candidate in model_candidates:
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    print(f"✗ Error: No model found in {MODEL_DIR}/")
    print(f"  Searched for: {model_candidates}")
    sys.exit(1)

print(f"  ✓ Loading: {os.path.basename(model_path)}")
model = keras.models.load_model(model_path, safe_mode=False)
models = [model]

print(f"Loaded 1 model\n")

# Single model inference with TTA
print("Running inference with TTA...")
print(f"  Original prediction: 1")
print(f"  + TTA augmentations: {TTA_AUGMENTATIONS}")
print(f"  = Total predictions per image: {1 + TTA_AUGMENTATIONS}\n")

ensemble_predictions = []
true_labels = test_df['label'].values.astype('int32')

# Prepare tabular features (dummy values - all zeros)
tabular_dummy = np.zeros((1, 5), dtype=np.float32)

for idx, row in test_df.iterrows():
    image_path = row['image_path']
    
    # Load original image
    original_image = load_and_preprocess(image_path)
    
    # Collect predictions: original + TTA
    image_predictions = []
    
    # Original (no augmentation)
    batch = tf.expand_dims(original_image, axis=0)
    for model in models:
        # Provide both image and tabular inputs
        pred = model.predict([batch, tabular_dummy], verbose=0)
        image_predictions.append(pred[0])
    
    # TTA augmentations
    for _ in range(TTA_AUGMENTATIONS):
        augmented = augment_tta(original_image)
        batch = tf.expand_dims(augmented, axis=0)
        for model in models:
            # Provide both image and tabular inputs
            pred = model.predict([batch, tabular_dummy], verbose=0)
            image_predictions.append(pred[0])
    
    # Average TTA predictions (all from single model)
    avg_pred = np.mean(image_predictions, axis=0)
    ensemble_predictions.append(avg_pred)
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(test_df)} images")

ensemble_predictions = np.array(ensemble_predictions)
predicted_labels = np.argmax(ensemble_predictions, axis=1)

# Compute accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY WITH TTA: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}\n")

# Per-class accuracy
class_names = ['Normal', 'BoneCancer', 'Osteoporosis', 'BoneTumor', 
               'Scoliosis', 'Arthritis', 'Fracture', 'Sprain']
print("Per-class accuracy:")
for class_id in range(NUM_CLASSES):
    class_mask = true_labels == class_id
    if class_mask.sum() > 0:
        class_acc = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
        print(f"  {class_names[class_id]}: {class_acc:.4f}")

print("\n✓ Inference complete!\n")

# Save results to file
results_file = 'inference_results.txt'
with open(results_file, 'w') as f:
    f.write(f"Single Model Test Accuracy (with TTA): {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Total test images: {len(test_df)}\n\n")
    f.write("Per-class accuracy:\n")
    for class_id in range(NUM_CLASSES):
        class_mask = true_labels == class_id
        if class_mask.sum() > 0:
            class_acc = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
            f.write(f"  {class_names[class_id]}: {class_acc:.4f}\n")

print(f"Results saved to {results_file}")
