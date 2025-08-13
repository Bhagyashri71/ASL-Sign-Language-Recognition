###Training Logic: fitting model, callbacks ##

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from dataset_loader import load_dataset
from model import build_model
from config import TRAIN_DIR, MODEL_PATH
import matplotlib.pyplot as plt

X, y, labels = load_dataset(TRAIN_DIR)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

model = build_model()
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=15, batch_size=32, callbacks=[early_stop])

model.save(MODEL_PATH)

# Plot accuracy/loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()
plt.show()
