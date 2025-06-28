
# 🎧 Audio Classification using Machine Learning

This project demonstrates how to classify environmental audio using **MFCC (Mel-Frequency Cepstral Coefficients)** features and a **neural network model** built with TensorFlow/Keras. The dataset used is [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), which contains labeled audio files of various urban sounds like sirens, dog barks, drilling, etc.

---

## 🚀 Project Workflow

### 1. 📦 Dataset & Files Setup
- **UrbanSound8K.zip** is extracted into proper `fold`-based folders.
- Metadata CSV (`UrbanSound8K.csv`) is used to map filenames to labels.
- Each `.wav` file is organized into `/content/audio/fold1`, `/fold2`, ..., `/fold10`.

### 2. 🔍 Audio Preprocessing
- Audio files are loaded using **librosa**.
- Both waveform plots and audio playback are used for visual/audio inspection.
- Audio is resampled to a consistent format (typically 22050Hz mono).

### 3. 🧠 Feature Extraction
- Features are extracted using **MFCC** (40 coefficients).
- Each audio sample is converted to a fixed-size vector using mean MFCCs.
- Features and labels are saved in a CSV file for later use.

### 4. ✂️ Dataset Preparation
- Features (`X`) and labels (`y`) are extracted from the DataFrame.
- Labels are one-hot encoded using `pandas.get_dummies`.
- Data is split into training and testing sets using `train_test_split`.

### 5. 🏗️ Model Building
- A **Sequential Neural Network** is created using Keras with:
  - Input layer: Dense(100) + ReLU + Dropout
  - Hidden layers: Dense(200) + ReLU + Dropout, Dense(100) + ReLU + Dropout
  - Output layer: Dense(num_classes) + Softmax
- Compiled using `categorical_crossentropy` and optimized with **Adam**.

### 6. 📈 Model Training
- Model is trained for up to 100 epochs with early stopping.
- Best model is saved using `ModelCheckpoint`.
- Training and validation accuracy/loss are monitored.

### 7. ✅ Model Evaluation
- Final accuracy is printed on test data.
- Duration of training is logged.

### 8. 🔊 Audio Prediction
- A sample `.wav` file is selected from the dataset.
- MFCC features are extracted.
- The trained model predicts the class of the audio.
- Predicted class label is displayed based on metadata mapping.

---



