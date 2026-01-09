# Emotion Recognition System for Virtual Classrooms (EduSense)

A deep learning–based facial emotion recognition system to classify student emotions (Happy, Sad, Angry, Neutral, Surprise) in real time, designed to improve engagement monitoring in virtual classrooms.

---

## Features

- Real-time facial emotion detection using webcam feed.  
- VGG16-based CNN model with custom layers for accurate emotion classification.  
- Training visualization: accuracy/loss plots, confusion matrix, and performance metrics.  
- Built with Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, and Seaborn.

---

## Requirements

- Python 3.x  
- TensorFlow / Keras  
- OpenCV (`opencv-python`)  
- NumPy, Pandas, Matplotlib, Seaborn  
- gdown (for downloading the model)

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Download Model:-

- The trained CNN model file can be downloaded automatically using Python:

import gdown

# Google Drive direct download link
url = "https://drive.google.com/uc?export=download&id=1SNKfLh7QV2QcvYpUBiLv7oXAqa5rqCfB"
output = "emotion_model.h5"  # File will be saved with this name
gdown.download(url, output, quiet=False)


Note: The model file is not included in the repository due to GitHub size limits. Use the code above to download it before running the project.

## How to Run:-

1. Install all dependencies (see Requirements above).

2. Download the model using the snippet above.

3. Run the main script for real-time emotion detection:
        python main.py

## Project Structure:-
/EduSense
│
├─ main.py             # Main script for real-time emotion detection
├─ emotion_model.h5    # CNN model file (download separately)
├─ requirements.txt    # Python dependencies
└─ README.md           # This file

