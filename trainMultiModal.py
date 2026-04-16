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

IMG_SIZE = 224
BATCH_SIZE = 32
# 0=Normal, 1=BoneCancer, 2=Osteoporosis, 3=BoneTumor, 4=Scoliosis, 5=Arthritis, 6=Fracture, 7=Sprain
NUM_CLASSES = 8
NUM_TABULAR_FEATURES = 5  # Age, BP_Sys, BP_Dia, SpO2, Calcium
HEAD_EPOCHS = 6
FINETUNE_EPOCHS = 10
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15

# On SageMaker, data is copied to /opt/ml/input/data/
# Locally it lives in the project root under data/
IS_SAGEMAKER = os.path.exists('/opt/ml/input/data')
DATA_ROOT = '/opt/ml/input/data' if IS_SAGEMAKER else 'data'
MODEL_DIR = os.environ.get('SM_MODEL_DIR', 'Models')

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

print(f"Running on {'SageMaker' if IS_SAGEMAKER else 'local'}")
print(f"Data root: {DATA_ROOT}")

# --- Data Loading & Cleaning ---
df = pd.read_csv('bone_dataset.csv')

# Remap local Windows paths to the correct data root for this environment
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
df['image_path'] = df['image_path'].apply(
    lambda p: os.path.join(DATA_ROOT, '/'.join(p.replace('\\', '/').split('/')[1:]))
    if not p.startswith(DATA_ROOT) else p
)

# Drop rows with missing image paths or invalid labels
df = df.dropna(subset=['image_path'])
df = df[df['label'].between(0, NUM_CLASSES - 1)]
df = df.drop_duplicates(subset='image_path')

# Fill missing tabular values with 0 (all datasets are image-only)
TABULAR_COLS = ['age', 'bp_sys', 'bp_dia', 'spo2', 'calcium']
df[TABULAR_COLS] = df[TABULAR_COLS].fillna(0).astype('float32')
HAS_TABULAR_SIGNAL = bool(df[TABULAR_COLS].to_numpy().any())

print(f"Using {'multimodal' if HAS_TABULAR_SIGNAL else 'image-only'} classifier head")

# Shuffle before split so classes aren't clustered
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 70/15/15 stratified split to preserve class ratios and keep a true held-out test set.
train_parts, val_parts, test_parts = [], [], []
for _, label_df in df.groupby('label'):
    label_df = label_df.sample(frac=1, random_state=42)
    n = len(label_df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # Ensure each split has at least 1 sample when possible.
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    train_parts.append(label_df.iloc[:n_train])
    val_parts.append(label_df.iloc[n_train:n_train + n_val])
    test_parts.append(label_df.iloc[n_train + n_val:])

train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Train label dist:\n", train_df['label'].value_counts().sort_index())

# --- Class Weights (inverse frequency) ---
label_counts = train_df['label'].value_counts().sort_index()
total = len(train_df)
label_indices = label_counts.index.to_numpy(dtype='int32')
label_values = label_counts.to_numpy(dtype='int32')
class_weights = {
    int(label): total / (NUM_CLASSES * int(count))
    for label, count in zip(label_indices, label_values)
}
print("Class weights:", class_weights)

# --- Dataset Pipeline ---
def preprocess_multimodal(image_path, tabular_stats, label):
    image_bytes = tf.io.read_file(image_path)
    lower_path = tf.strings.lower(image_path)
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
    # MURA backbone was trained with pixel range [0, 255]. Keep the same scale.
    image = tf.cast(image, tf.float32)
    # Keep label as integer — class_weight requires integer labels, not one-hot
    return {"vision_input": image, "tabular_input": tabular_stats}, label

def augment(inputs, label):
    image = inputs["vision_input"]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return {"vision_input": image, "tabular_input": inputs["tabular_input"]}, label

def create_dataset(dataframe, training=False):
    image_paths = dataframe['image_path'].values
    tabular_data = dataframe[TABULAR_COLS].values.astype('float32')
    labels = dataframe['label'].values.astype('int32')

    ds = tf.data.Dataset.from_tensor_slices((image_paths, tabular_data, labels))
    if training:
        ds = ds.shuffle(buffer_size=5000, seed=42)
    ds = ds.map(preprocess_multimodal, num_parallel_calls=tf.data.AUTOTUNE)
    # Skip any samples that still fail decode after JPEG recovery.
    ds = ds.ignore_errors()
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df, training=False)
test_ds = create_dataset(test_df, training=False)

# --- Model ---
def build_combo_model(vision_base, tabular_dim, use_tabular):
    img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="vision_input")
    tab_in = keras.Input(shape=(tabular_dim,), name="tabular_input")

    # training=False keeps BatchNorm in inference mode during feature extraction
    vision_features = vision_base(img_in, training=False)
    if len(vision_features.shape) == 4:
        vision_features = layers.GlobalAveragePooling2D()(vision_features)
    vision_features = layers.Dropout(0.2)(vision_features)

    if use_tabular:
        x_tab = layers.Dense(64, activation="relu")(tab_in)
        x_tab = layers.Dense(128, activation="relu")(x_tab)
        tabular_features = layers.BatchNormalization()(x_tab)
        fused = layers.Concatenate(axis=1)([vision_features, tabular_features])
    else:
        # Keep tabular input connected in image-only mode with a deterministic zero-output stub.
        tab_stub = layers.Dense(
            1,
            use_bias=False,
            kernel_initializer="zeros",
            trainable=False,
            name="tabular_stub",
        )(tab_in)
        fused = layers.Concatenate(axis=1)([vision_features, tab_stub])

    x = layers.Dense(256, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=[img_in, tab_in], outputs=outputs, name="Multimodal_Bone_Classifier")


mura_model = keras.models.load_model(os.path.join(MODEL_DIR, 'mura_efficientnet.keras'))
vision_extractor = keras.Model(
    inputs=mura_model.input,
    outputs=mura_model.get_layer("avg_pool").output,
    name="MURA_Feature_Extractor"
)
vision_extractor.trainable = False

model = build_combo_model(vision_extractor, NUM_TABULAR_FEATURES, HAS_TABULAR_SIGNAL)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",  # sparse = integer labels, works with class_weight
    metrics=["accuracy"],
    jit_compile=False,
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_multimodal_model.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    ),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    train_ds,
    epochs=HEAD_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine-tune the top of the vision backbone on the actual 8-class target task.
vision_extractor.trainable = True
for layer in vision_extractor.layers[:-30]:
    layer.trainable = False
for layer in vision_extractor.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_multimodal_model.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    ),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

history_finetune = model.fit(
    train_ds,
    initial_epoch=len(history.history['loss']),
    epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks
)

# Evaluate on held-out test set with best checkpoint.
# `safe_mode=False` allows loading older checkpoints that may contain Lambda layers.
best_model = keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_multimodal_model.keras'),
    safe_mode=False,
)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"Held-out test accuracy: {test_acc:.4f} | test loss: {test_loss:.4f}")

print("Training Complete. Model saved as best_multimodal_model.keras")
