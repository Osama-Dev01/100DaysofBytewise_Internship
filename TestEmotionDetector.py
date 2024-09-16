# Import required packages
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Dictionary for emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the entire model from the saved file
model_path = 'emotion_model.h5'
emotion_model = load_model(model_path)
print("Loaded model from disk")

# Initialize video capture from webcam or video file
# Uncomment the following line for webcam capture
#cap = cv2.VideoCapture(0)

# Pass your video file path
video_path = "I:\\ved.mp4"  # Use double backslashes for Windows paths
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file at path {video_path}")
    exit()

# Start processing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading video file.")
        break

    frame = cv2.resize(frame, (1280, 720))
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
