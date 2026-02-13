import tensorflow as tf
from tensorflow.keras import layers, models
import os
import shutil
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # Keep small for incremental training
IMAGE_DATA_DIR = "ImageData"
NEW_DATA_DIR = "NewData"
MODEL_DIR = "model"

# ----------------------------
# MERGE NEW DATA INTO TRAINING
# ----------------------------
def merge_new_data():
    if not os.path.exists(NEW_DATA_DIR):
        print("No NewData folder found.")
        return

    for class_name in os.listdir(NEW_DATA_DIR):
        new_class_path = os.path.join(NEW_DATA_DIR, class_name)

        if os.path.isdir(new_class_path):
            target_class_path = os.path.join(IMAGE_DATA_DIR, class_name)

            os.makedirs(target_class_path, exist_ok=True)

            for file in os.listdir(new_class_path):
                shutil.move(
                    os.path.join(new_class_path, file),
                    os.path.join(target_class_path, file)
                )

    print("New data merged successfully.")

# ----------------------------
# LOAD DATASET
# ----------------------------
def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        IMAGE_DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode="categorical"
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        IMAGE_DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode="categorical"
    )

    class_names = dataset.class_names
    print("Classes:", class_names)

    normalization = layers.Rescaling(1./255)

    dataset = dataset.map(lambda x, y: (normalization(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization(x), y))

    return dataset, val_dataset, class_names

# ----------------------------
# BUILD TRANSFER LEARNING MODEL
# ----------------------------
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ----------------------------
# SAVE MODEL WITH VERSIONING
# ----------------------------
def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"cnn_model_{version}.h5")

    model.save(model_path)
    model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))  # overwrite latest

    print(f"Model saved as {model_path}")

# ----------------------------
# SAVE CLASS NAMES
# ----------------------------
def save_class_names(class_names):
    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

# ----------------------------
# CLEAR NEW DATA
# ----------------------------
def clear_new_data():
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
        os.makedirs(NEW_DATA_DIR)
        print("NewData folder cleared.")

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    print("Merging new data...")
    merge_new_data()

    print("Loading dataset...")
    train_ds, val_ds, class_names = load_dataset()

    print("Building model...")
    model = build_model(len(class_names))

    print("Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    print("Saving model...")
    save_model(model)
    save_class_names(class_names)

    print("Cleaning new data...")
    clear_new_data()

    print("Retraining complete!")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
