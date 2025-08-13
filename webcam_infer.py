import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import IMG_SIZE, MODEL_PATH
from utils import label_to_char

import mediapipe as mp

# Load model
model = load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Set HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            image_height, image_width, _ = frame.shape
            x_coords = [lm.x * image_width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * image_height for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Extract ROI and resize to model input
            x_min_crop = max(x_min - 20, 0)
            y_min_crop = max(y_min - 20, 0)
            x_max_crop = min(x_max + 20, image_width)
            y_max_crop = min(y_max + 20, image_height)

            roi = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_normalized = roi_resized / 255.0
            input_image = np.expand_dims(roi_normalized, axis=0)

            # Predict letter
            prediction = model.predict(input_image)
            class_index = np.argmax(prediction)
            predicted_letter = label_to_char(class_index)

            # Draw bounding box
            cv2.rectangle(frame, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop), (0, 255, 0), 2)

            # Draw label directly above the box
            label_position = (x_min_crop, y_min_crop - 10)
            cv2.putText(frame, f'{predicted_letter}', label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Show frame
    cv2.imshow("ASL Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
