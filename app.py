import os
import streamlit as st
import cv2
import numpy as np
import pygame
import dlib
import time

# Initialize pygame for sound
pygame.mixer.init()

# Constants for drowsiness detection
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.8  # Increased threshold for yawning
CONSECUTIVE_FRAMES = 15
ALERT_COOLDOWN = 5  # Cooldown time in seconds

blink_count = 0
yawn_count = 0
last_alert_time = 0

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download and provide this file
if not os.path.exists(predictor_path):
    raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
predictor = dlib.shape_predictor(predictor_path)

# Function to play alert sound
def play_alert_sound():
    try:
        pygame.mixer.music.load('alert.mp3')
        pygame.mixer.music.play()
    except pygame.error as e:
        st.error(f"Error: Unable to play alert sound. {e}")

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_points):
    vertical = np.linalg.norm(mouth_points[2] - mouth_points[10]) + np.linalg.norm(mouth_points[4] - mouth_points[8])
    horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
    mar = vertical / (2.0 * horizontal)
    return mar

# Function to get landmarks for a face
def get_landmarks(gray, face):
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

# Main drowsiness detection function
def detect_drowsiness():
    global blink_count, yawn_count, last_alert_time

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam. Make sure it is connected and not in use by another application.")
        return

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        for face in faces:
            landmarks = get_landmarks(gray, face)

            # Extract eye and mouth regions
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]

            # Calculate EAR and MAR
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            # EAR Check
            if ear < EAR_THRESHOLD:
                blink_count += 1
            else:
                blink_count = 0

            if blink_count >= CONSECUTIVE_FRAMES and time.time() - last_alert_time > ALERT_COOLDOWN:
                st.warning("Drowsiness Alert: Eyes closed for too long!")
                play_alert_sound()
                last_alert_time = time.time()

            # MAR Check
            if mar > MAR_THRESHOLD:
                yawn_count += 1
            else:
                yawn_count = 0

            if yawn_count >= CONSECUTIVE_FRAMES and time.time() - last_alert_time > ALERT_COOLDOWN:
                st.warning("Drowsiness Alert: Yawning detected!")
                play_alert_sound()
                last_alert_time = time.time()

            # Draw rectangles around eyes and mouth
            for (x, y) in np.vstack((left_eye, right_eye, mouth)):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the video feed in Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Main Streamlit app logic
if __name__ == "__main__":
    st.title("Safe Drive: Real-Time Drowsiness Detection")
    st.write("Press 'Start Detection' to activate the webcam and detect drowsiness.")
    if st.button("Start Detection"):
        detect_drowsiness()
