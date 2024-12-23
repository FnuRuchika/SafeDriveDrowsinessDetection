import cv2
import os

base_path = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(base_path, 'cascades/haarcascade_frontalface_default.xml'))

if face_cascade.empty():
    print("Error: Face cascade file not loaded!")
else:
    print("Face cascade loaded successfully!")

