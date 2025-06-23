import cv2
import numpy as np
from keras.models import model_from_json
from collections import deque, Counter
import time

# Load model
json_file = open('Emotion Detector.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('Emotion Detector.h5')

# Load Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Webcam setup
webcam = cv2.VideoCapture(0)
time.sleep(2)  # Warm-up

# Smoothing history
emotion_history = deque(maxlen=10)

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for speed
    im = cv2.resize(im, (640, 480))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        face_img = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        # Resize and predict
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        pred = model.predict(img)
        emotion_index = pred.argmax()

        # Add to history for smoothing
        emotion_history.append(emotion_index)
        most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
        prediction_label = labels[most_common_emotion]

        # Show emotion on screen
        cv2.putText(im, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection", im)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()