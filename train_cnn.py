import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# -----------------------------
# 1. DATASET LOADING & SPLIT
# -----------------------------
data_dir = r"D:\FFPP\resized"

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -----------------------------
# 2. NORMALIZATION
# -----------------------------
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 3. CNN MODEL
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.build(input_shape=(None, 224, 224, 3))

model.summary()

# -----------------------------
# 4. COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)



# -----------------------------
# 5. TRAIN MODEL
# -----------------------------

#callbacks = [
    #tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss',
        #patience=12,
        #restore_best_weights=True
   # )
#]
class_weights = {
    0: 2.0,   # fake
    1: 1.0    # real
}

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=25,
    class_weight=class_weights
)

model.save("deepfake_model.keras")
print("Model Saved Successfully!")



# -----------------------------
# 6. EVALUATE MODEL
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# 7. CONFUSION MATRIX & METRICS
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predictions = (predictions > 0.4).astype(int).flatten()
    y_pred.extend(predictions)
    y_true.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    zero_division=0
))


cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# -----------------------------
# 8. PLOT ACCURACY & LOSS
# -----------------------------



cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


metrics = ["Precision", "Recall", "F1-score"]
values = [precision, recall, f1]
colors=["blue","red","purple"]

plt.figure(figsize=(6,4))
plt.bar(metrics, values,color=colors)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Precision, Recall and F1-score")
plt.tight_layout()


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.figure(figsize=(12,6))

for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int).flatten()

    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(
            f"Actual: {class_names[labels[i]]}\nPredicted: {class_names[predictions[i]]}"
        )
        plt.axis("off")

#plt.savefig("sample_predictions.png", dpi=300)


plt.tight_layout()
plt.show()



