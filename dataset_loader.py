### Data loading, preprocessing, splitting

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import IMG_SIZE, NUM_CLASSES

def load_dataset(directory):
    images, labels = [], []
    for label_folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, label_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = img / 255.0
                        images.append(img)
                        labels.append(ord(label_folder.upper()) - ord('A'))
    X = np.array(images)
    y = to_categorical(labels, num_classes=NUM_CLASSES)
    return X, y, labels  # labels (not one-hot) for visualization

