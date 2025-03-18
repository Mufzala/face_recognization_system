# Facialytics (Advanced face recognition system with real Time detection )
import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tensorflow.keras.models import load_model

def load_face_embeddings(embedding_path="embeddings.pkl"):
    """
    Load saved face embeddings and labels.
    """
    if os.path.exists(embedding_path):
        with open(embedding_path, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["labels"]
    else:
        return [], []

def train_recognition_model(embeddings, labels, model_path="face_recognition_model.pkl"):
    """
    Train and save a KNN classifier for face recognition.
    """
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(embeddings, labels)
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    print("Face recognition model trained and saved.")

def recognize_face(frame, face_model, recognition_model):
    """
    Detect and recognize faces from the frame.
    """
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

def live_face_recognition(face_model_path="facenet_model.h5", recog_model_path="face_recognition_model.pkl"):
    """
    Capture live video and recognize faces in real-time.
    """
    face_model = load_model(face_model_path)
    with open(recog_model_path, "rb") as f:
        recognition_model = pickle.load(f)
    
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
    embeddings, labels = load_face_embeddings()
    if embeddings and labels:
        train_recognition_model(embeddings, labels)
    live_face_recognition()
