import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

IMG_SIZE = 224
BATCH_SIZE = 32
# 0=Normal, 1=BoneCancer, 2=Osteoporosis, 3=BoneTumor, 4=Scoliosis, 5=Arthritis, 6=Fracture, 7=Sprain
NUM_CLASSES = 8
NUM_TABULAR_FEATURES = 5  # Age, BP_Sys, BP_Dia, SpO2, Calcium
HEAD_EPOCHS = 6
FINETUNE_EPOCHS = 10

# On SageMaker, data is copied to /opt/ml/input/data/
# Locally it lives in the project root under data/
IS_SAGEMAKER = os.path.exists('/opt/ml/input/data')
DATA_ROOT = '/opt/ml/input/data' if IS_SAGEMAKER else 'data'
MODEL_DIR = os.environ.get('SM_MODEL_DIR', 'Models')

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

# 80/20 stratified split to preserve class ratios
train_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))
test_df = df.drop(train_df.index)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
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
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
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
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_df, training=True)
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
        fused = vision_features

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
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_multimodal_model.keras'), save_best_only=True),
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    train_ds,
    epochs=HEAD_EPOCHS,
    validation_data=test_ds,
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
    metrics=["accuracy"]
)

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_multimodal_model.keras'), save_best_only=True),
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

history_finetune = model.fit(
    train_ds,
    initial_epoch=len(history.history['loss']),
    epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
    validation_data=test_ds,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks
)

print("Training Complete. Model saved as best_multimodal_model.keras")
