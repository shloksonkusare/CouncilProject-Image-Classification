import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# -----------------------------
# LOAD DATASET (MULTI-CLASS)
# -----------------------------
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "ImageData/",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'   # 🔥 IMPORTANT CHANGE
)

# Save class names for Streamlit
class_names = train_data.class_names
print("Class Names:", class_names)

# -----------------------------
# NORMALIZATION
# -----------------------------
normalization = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization(x), y))

# -----------------------------
# CNN MODEL
# -----------------------------
model = models.Sequential([
    layers.Conv2D(128, 3, activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    # 🔥 OUTPUT LAYER FOR 2 CLASSES
    layers.Dense(len(class_names), activation='softmax')
])

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(train_data, epochs=EPOCHS)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("model/cnn_model.h5")

# OPTIONAL: Save class names for Streamlit
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
