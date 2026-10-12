import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LANDMARK_IDS = [168, 4, 33, 133, 362, 263]

# Konfiguracja Tasks API
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        face = result.face_landmarks[0]
        for idx in LANDMARK_IDS:
            pt = face[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("Krok 1 - Tasks API", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()

