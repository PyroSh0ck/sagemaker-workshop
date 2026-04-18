import os
import pandas as pd

# Work around SageMaker TF/CUDA image issues where XLA JIT looks for libdevice
# in the wrong location and crashes during graph compilation.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/conda --xla_gpu_enable_triton_gemm=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure GPU to use ~95% of available memory for maximum performance
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set virtual device limit to use ~95% of GPU memory
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=14650)]  # 95% of 15GB
            )
    except RuntimeError as e:
        print(f"GPU memory config error: {e}")

# Enable mixed precision training for faster computation
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64  # Balanced batch size for GPU memory
NUM_CLASSES = 2
HEAD_EPOCHS = 12
FINETUNE_EPOCHS = 20

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# Debug: Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'='*50}")
print(f"GPU DETECTION: {len(gpus)} GPU(s) found")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("  WARNING: No GPUs detected! Training will be VERY slow.")
print(f"{'='*50}\n")

# Retrieving data
train_df = pd.read_csv('train_image_paths.csv', header=None, names=['path'])
test_df = pd.read_csv('valid_image_paths.csv', header=None, names=['path'])

# Labels have a 1 if 'positive' is in the string (for the folder), otherwise its a 0
train_labels = (train_df['path'].str.contains('positive')).astype(int).values
test_labels = (test_df['path'].str.contains('positive')).astype(int).values
train_paths = train_df['path'].values
test_paths = test_df['path'].values

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # EfficientNetV2 with include_preprocessing=True expects pixel range [0, 255].
    image = tf.cast(image, tf.float32)
    return image, label

# Make the tf datasets
ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_train = ds_train.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)
ds_train = ds_train.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
ds_test = ds_test.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# apply augmentation and batching - more aggressive augmentation
img_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.25),
        layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15),
    ],
    name="img_augmentation",
)

# apply the augmentation and batching
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.map(lambda x, y: (img_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(8)  # Moderate prefetch for better GPU pipeline

ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
ds_test = ds_test.prefetch(4)  # Moderate prefetch for validation

# the actual model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

pretrained_model = keras.applications.EfficientNetV2M(
    include_top=False, # larger backbone for better feature extraction
    weights="imagenet",
    input_tensor=inputs
)

pretrained_model.trainable = False

# Rebuild the top (The Classification Head)
# this is basically taken straight from keras documentation
x = layers.GlobalAveragePooling2D(name="avg_pool")(pretrained_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4, name="top_dropout")(x)

# Final prediction layer
# num classes is gonna be 2 because we only determine whether its abnormal or normal
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

model = keras.Model(inputs, outputs, name="EfficientNet")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

# Inverse-frequency class weighting to reduce majority-class bias.
class_counts = pd.Series(train_labels).value_counts().to_dict()
total = len(train_labels)
class_weight = {
    0: total / (NUM_CLASSES * class_counts.get(0, 1)),
    1: total / (NUM_CLASSES * class_counts.get(1, 1)),
}

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="../Models/mura_efficientnet.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
    ),
]

print("Starting head training")
hist = model.fit(
    ds_train,
    epochs=HEAD_EPOCHS,
    validation_data=ds_test,
    class_weight=class_weight,
    callbacks=callbacks,
)

# Unfreeze the upper part of the backbone for a low-LR fine-tuning pass.
pretrained_model.trainable = True
for layer in pretrained_model.layers[:-40]:
    layer.trainable = False
for layer in pretrained_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

finetune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="../Models/mura_efficientnet.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
    ),
]

print("Starting backbone fine-tuning")
hist_finetune = model.fit(
    ds_train,
    initial_epoch=len(hist.history["loss"]),
    epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
    validation_data=ds_test,
    class_weight=class_weight,
    callbacks=finetune_callbacks,
)

os.makedirs("../Models", exist_ok=True)
# Save the final in-memory model as well; the best checkpoint is already saved above.
model.save("../Models/mura_efficientnet.keras")
