import cv2
import mediapipe as mp
import pygame
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize Pygame mixer for alarm
pygame.mixer.init()
pygame.mixer.music.load("music.wav")  # Make sure music.wav is in the same folder

# EAR calculation functions
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_points):
    vertical1 = euclidean(eye_points[1], eye_points[5])
    vertical2 = euclidean(eye_points[2], eye_points[4])
    horizontal = euclidean(eye_points[0], eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Eye landmark indexes for MediaPipe Face Mesh (LEFT and RIGHT)
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

# EAR threshold for detecting closed eyes
EAR_THRESHOLD = 0.25
counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

# Frame rate calculation (for 2-second threshold)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # Fallback if actual FPS is not available

CONSEC_FRAMES = int(fps * 2)  # 2 seconds worth of frames

print(f"[INFO] FPS: {fps:.2f} | CONSEC_FRAMES for 2 sec: {CONSEC_FRAMES}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDEXES]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDEXES]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if eyes are closed
        if avg_ear < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                cv2.putText(frame, "DROWSINESS DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            counter = 0
            pygame.mixer.music.stop()

        # Draw eye landmarks
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Show video feed
    cv2.imshow("Driver Drowsiness Detection", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
