import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)

SEED = 1000
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
CLASSES = {
    "Uninfected": 0,
    "Falciparum": 1,
    "Vivax": 2
}
IMAGE_DIR = r"D:\\" # dataset on hard drive
np.random.seed(SEED)

def load_images(image_dir, target_size, classes_dict):
    data, labels = [], []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    for label_name, label_idx in classes_dict.items():
        class_path = os.path.join(image_dir, label_name)

        for root, _, files in os.walk(class_path):  # Walk through subdirectories
            for img_name in files:
                if not img_name.lower().endswith(valid_extensions):
                    continue

                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)

                if img is None or len(img.shape) != 3 or img.shape[2] != 3:
                    print(f"[WARN] Skipping unreadable or grayscale image: {img_path}")
                    continue

                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img).resize(target_size)
                    data.append(np.array(img))
                    labels.append(label_idx)
                    print(f"[LOADED] image: {img_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to process image: {img_path}\n{e}")

    return np.array(data, dtype=np.float32), np.array(labels)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    fig, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training Performance', fontsize=14)

    acc_ax.plot(history.history['accuracy'], label='Train Acc')
    acc_ax.plot(history.history['val_accuracy'], label='Val Acc')
    acc_ax.set_title('Accuracy')
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.legend()

    loss_ax.plot(history.history['loss'], label='Train Loss')
    loss_ax.plot(history.history['val_loss'], label='Val Loss')
    loss_ax.set_title('Loss')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    print("[INFO] Loading images...")
    data, labels = load_images(IMAGE_DIR, IMAGE_SIZE, CLASSES)
    data /= 255.0  # Normalize

    labels = to_categorical(labels, num_classes=len(CLASSES))

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=SEED
    )

    print("[INFO] Building model...")
    model = build_model(input_shape=(64, 64, 3), num_classes=3)
    model.summary()

    print("[INFO] Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        shuffle=True
    )

    print("[INFO] Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=CLASSES.keys()))

    plot_history(history)
    model.save("malaria_model.h5")


#if __name__ == "__main__":
#    main()