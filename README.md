# ASL-Sign-Language-Recognition


########### ASL Hand Sign Recognition using CNN ###################

A computer vision project that recognizes American Sign Language (ASL) hand gestures using a Convolutional Neural Network (CNN).
It allows both image-based predictions and real-time gesture detection using your webcam.

## 📂 Project Structure

asl\_project/
├── dataset_loader.py         # Loads and preprocesses images
├── eda_visualization.py      # Generates graphs for data imbalance and insights
├── model_builder.py          # Defines the CNN model architecture
├── train.py                   # Trains the model with callbacks and saves it
├── evaluate.py             # Evaluates the model and prints classification report
├── webcam_infer.py         # Uses OpenCV for live webcam prediction
├── config.py                 # Global constants (like image size, paths)
├── utils.py                  # Helper functions like label mapping
├── asl_model.h5              # Trained model (binary file)
└── README.md                 # Project documentation

## 🧰 Requirements
 Install all required packages:
- Python 3.11
- TensorFlow
- NumPy
- OpenCV
- Mediapipe
- Matplotlib
- Scikit-learn
- Seaborn


## ⚙️ How to Use

### 1. 📥 Dataset Setup

Place your ASL dataset in the structure:

dataset/
├── A/
├── B/
├── C/
...
├── Z/

Make sure `dataset_loader.py` points to the correct path.

### 2. 🏗️ Train the Model

  python train_model.py

This will:

* Preprocess and augment data
* Train the model with early stopping
* Save the trained model as `asl_model.h5`


### 3. 🧪 Test the Model

  python test_model.py


This will:

* Load test data
* Evaluate the model and print the classification report and accuracy


### 4. 📸 Real-Time Webcam Prediction

  python webcam_predict.py

This will:

* Open your webcam
* Detect your hand sign
* Predict the class in real-time and show it on screen


##  Model Performance

| Metric    | Score |
| Accuracy  | 79 %  |
| Precision | 0.93  |
| Recall    | 0.87  |
| F1 Score  | 0.86  |




