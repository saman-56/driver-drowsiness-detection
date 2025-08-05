import cv2
import mediapipe as mp
import pygame
import math
import speech_recognition as sr
import threading

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize Pygame mixer for alarm
pygame.mixer.init()
pygame.mixer.music.load("music.wav")  # Ensure this file is in the same folder

# EAR calculation
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_points):
    vertical1 = euclidean(eye_points[1], eye_points[5])
    vertical2 = euclidean(eye_points[2], eye_points[4])
    horizontal = euclidean(eye_points[0], eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Landmark indexes for eyes
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
counter = 0

# Alarm state tracking
alarm_playing = False
alarm_lock = threading.Lock()

def stop_alarm():
    global alarm_playing
    pygame.mixer.music.stop()
    with alarm_lock:
        alarm_playing = False

def voice_command_listener():
    global alarm_playing
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("[INFO] Listening for 'stop alarm' command...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            command = r.recognize_google(audio).lower()
            print(f"[VOICE] Heard: {command}")
            if "stop" in command or "alarm" in command:
                stop_alarm()
        except Exception as e:
            print(f"[VOICE ERROR] {e}")

# Start webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
CONSEC_FRAMES = int(fps * 2)

print(f"[INFO] FPS: {fps:.2f} | CONSEC_FRAMES for 2 sec: {CONSEC_FRAMES}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply night mode enhancement (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_frame = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDEXES]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDEXES]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                with alarm_lock:
                    if not alarm_playing:
                        pygame.mixer.music.play(-1)
                        alarm_playing = True
                        threading.Thread(target=voice_command_listener, daemon=True).start()
                cv2.putText(frame, "DROWSINESS DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            counter = 0
            if alarm_playing:
                pygame.mixer.music.stop()
                with alarm_lock:
                    alarm_playing = False

        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

