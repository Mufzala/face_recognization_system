# Facialytics (Advanced face recognition system with real Time detection )
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def load_models(face_model_path="facenet_model.h5", recog_model_path="face_recognition_model.pkl"):
    """Load the FaceNet model and trained recognition model."""
    face_model = load_model(face_model_path)
    with open(recog_model_path, "rb") as f:
        recognition_model = pickle.load(f)
    return face_model, recognition_model

def recognize_face(frame, face_model, recognition_model):
    """Detect and recognize faces in a given frame."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160)) / 255.0
        face_embedding = face_model.predict(np.expand_dims(face_resized, axis=0))[0]
        label = recognition_model.predict([face_embedding])
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(label[0]), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

def real_time_recognition():
    """Capture live video and recognize faces in real-time."""
    face_model, recognition_model = load_models()
    cam = cv2.VideoCapture(0)
    print("Starting real-time face recognition...")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        frame = recognize_face(frame, face_model, recognition_model)
        cv2.imshow("Real-Time Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("Face recognition session ended.")

if __name__ == "__main__":
    real_time_recognition()

