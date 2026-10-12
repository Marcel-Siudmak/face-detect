import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LANDMARK_IDS = [168, 4, 33, 133, 362, 263]

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if result.face_landmarks:
            face = result.face_landmarks[0]
            points = {idx: (int(face[idx].x * w), int(face[idx].y * h)) for idx in LANDMARK_IDS}
            print(points)
except KeyboardInterrupt:
    pass
finally:
    detector.close()
    cap.release()
