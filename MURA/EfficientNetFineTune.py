import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2

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

# apply augmentation and batching
img_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip("horizontal"),
    ],
    name="img_augmentation",
)

# apply the augmentation and batching
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.map(lambda x, y: (img_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# the actual model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

pretrained_model = keras.applications.EfficientNetV2S(
    include_top=False, # so we don't have to remove it manually for fine tuning
    weights="imagenet",
    input_tensor=inputs
)

pretrained_model.trainable = False

# Rebuild the top (The Classification Head)
# this is basically taken straight from keras documentation
x = layers.GlobalAveragePooling2D(name="avg_pool")(pretrained_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2, name="top_dropout")(x)

# Final prediction layer
# num classes is gonna be 2 because we only determine whether its abnormal or normal
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

model = keras.Model(inputs, outputs, name="EfficientNet")

# training
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
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
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
    ),
]

epochs = 5
print("Starting training")
hist = model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
    class_weight=class_weight,
    callbacks=callbacks,
)

import os
os.makedirs("../Models", exist_ok=True)
# SavedModel format (recommended over .h5 in Keras 3)
model.save("../Models/mura_efficientnet.keras")
