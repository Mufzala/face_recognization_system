# Facialytics (Advanced face recognition system with real Time detection )
# hey this is the code for collecting the data for the face recognition system. This code will capture the images from the webcam and store them in the specified folder. The images are captured when the face is detected in the frame. The number of images to capture per person can be specified using the num_samples parameter. The images are resized to 160x160 pixels before saving.
import cv2
import os
import numpy as np
from PIL import Image

def collect_data(save_path="dataset", num_samples=100, cam_index=0):
    """
    Captures face images from webcam and stores them in the specified folder.
    :param save_path: Directory to save the captured images.
    :param num_samples: Number of images to capture per person.
    :param cam_index: Camera index for webcam (default is 0).
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(cam_index)
    count = 0
    
    print("Starting data collection. Look into the camera.")
    while count < num_samples:
        ret, frame = cam.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            img_path = os.path.join(save_path, f"face_{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

if __name__ == "__main__":
    collect_data()
