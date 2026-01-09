import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
import pandas as pd
import os

# --- Data Logging Setup (Yahan hum data ko save karne ka setup bana rahe hain) ---
LOG_FILE = 'emotion_log.csv'
LOG_INTERVAL_SECONDS = 5 # Har 5 second me data CSV file me save hoga
last_log_time = time.time()
emotion_counts = {emotion: 0 for emotion in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]}
# ------------------------

# 1. Model load karo (Dhyaan rakho ki 'emotion_model_edsense.h5' isi folder me ho)
try:
    model = load_model('emotion_model_edsense.h5') 
except Exception as e:
    print(f"ERROR: Model could not be loaded. Check file name and path: {e}")
    exit()

# 2. Emotion Labels (Ye index number ko emotion naam se map karta hai)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# 3. Face Detector load karo (Ye haarcascade file se face detect karega)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    print(f"ERROR: Face cascade file not loaded from: {cascade_path}")
    exit()

# 4. Camera start karo (0 ka matlab default webcam hota hai)
cap = cv2.VideoCapture(0)

print("Starting Webcam. Keep your face well-lit and centered.")
print("The CSV file will be created 5 seconds after the first face detection.")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("ERROR: Failed to read frame from camera. Is camera in use by another app?")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yahan hum face detect kar rahe hain
    faces = face_classifier.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    current_frame_emotion = 'Neutral' # Agar face detect nahi hota toh default emotion 'Neutral' rakha hai

    if len(faces) > 0:
        # If face detected
        # print(f"Face Detected! Current faces: {len(faces)}") 

       # Take the first (largest) face
        (x, y, w, h) = faces[0] 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        
        # Model ke liye image ko pre-process kar rahe hain (resize + normalize)
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cropped_img.astype('float32') / 255.0
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        
        # Yahan model se predict karte hain ki kaun sa emotion hai
        prediction = model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction[0]))
        current_frame_emotion = emotion_dict[maxindex]

        # Frame par emotion ka naam show karte hain
        cv2.putText(frame, current_frame_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # --- Data Logging Logic (Yahan hum emotions ko count aur save kar rahe hain) ---
# Har frame me detect hua emotion count kar rahe hain
    emotion_counts[current_frame_emotion] += 1
    
    # Agar 5 second ho gaye ho toh data save karenge
    if time.time() - last_log_time > LOG_INTERVAL_SECONDS:
        total_frames = sum(emotion_counts.values())
        
        # Ensure frames are captured
        if total_frames > 0:
            
            # Nai entry ko CSV me add karte hain
            new_log = pd.DataFrame([{
                'Timestamp': pd.Timestamp.now(),
                'Total_Frames': total_frames,
                'Dominant_Emotion': max(emotion_counts, key=emotion_counts.get),
                'Happy_Percent': (emotion_counts['Happy'] / total_frames) * 100,
                'Sad_Percent': (emotion_counts['Sad'] / total_frames) * 100,
                'Neutral_Percent': (emotion_counts['Neutral'] / total_frames) * 100,
            }])
            
            # Agar file pehli baar ban rahi hai toh header add hoga, warna data append hoga
            new_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            
            # Terminal me print karte hain ki data save ho gaya
            print(f"--- Data Logged! Dominant Emotion: {new_log['Dominant_Emotion'].iloc[0]} ---")
            
        # Reset counters and update time
        emotion_counts = {emotion: 0 for emotion in emotion_counts}
        last_log_time = time.time()
    # --------------------------

    # Output window (camera preview) dikhate hain
    cv2.imshow('EduSense Emotion Detector', frame)
    
    # 'q' dabane par program band ho jayega
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()