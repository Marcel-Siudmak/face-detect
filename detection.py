import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model antropometryczny twarzy 3D w mm (układ twarzy: środek między oczami [168], +X prawo, +Y dół, +Z od kamery)
# Kluczowe punkty o pełnej rozpiętości wysokości (czoło-broda ~180mm) i głębi (nos-skronie)
LANDMARKS_DICT = {
    168: [  0.0,   0.0,   0.0],  # Nasada nosa (środek układu)
      4: [  0.0,  30.0, -22.0],  # Czubek nosa (wystaje do przodu)
     10: [  0.0, -75.0,  15.0],  # Czoło (góra twarzy - kotwica pionowa)
    152: [  0.0, 105.0,  10.0],  # Broda (dół twarzy - kotwica pionowa)
     33: [-46.0,   3.0,  18.0],  # Prawy kącik oka zewn. (viewer's left)
    133: [-16.0,   3.0,   4.0],  # Prawy kącik oka wewn.
    362: [ 16.0,   3.0,   4.0],  # Lewy kącik oka wewn.
    263: [ 46.0,   3.0,  18.0],  # Lewy kącik oka zewn. (viewer's right)
     61: [-28.0,  68.0,   8.0],  # Prawy kącik ust
    291: [ 28.0,  68.0,   8.0],  # Lewy kącik ust
}

LANDMARK_IDS = list(LANDMARKS_DICT.keys())
MODEL_3D = np.array(list(LANDMARKS_DICT.values()), dtype=np.float64)

# Środek prawego i lewego oka w układzie modelu głowy (anatomiczny rozstaw IPD = 63 mm)
EYE_RIGHT_HEAD = np.array([-31.5, 3.0, 8.0])  # viewer's left
EYE_LEFT_HEAD  = np.array([ 31.5, 3.0, 8.0])  # viewer's right

# Landmarki źrenic w MediaPipe FaceMesh
IRIS_RIGHT_ID = 468  # Prawa źrenica (viewer's left)
IRIS_LEFT_ID  = 473  # Lewa źrenica (viewer's right)

# Parametry ekranu MacBook Air 13.3" (M1):
# Ekran 13.3 cala Retina 16:10 -> szerokość: 286.5 mm, wysokość: 179.0 mm
# Kamerka FaceTime HD znajduje się symetrycznie w poziomie na górnej ramce (ok. 6.0 mm nad krawędzią ekranu)
SCREEN_WIDTH_MM = 286.5
SCREEN_HEIGHT_MM = 179.0
BEZEL_TOP_TO_SCREEN_MM = 6.0
CAMERA_TO_SCREEN_CENTER_Y_MM = BEZEL_TOP_TO_SCREEN_MM + (SCREEN_HEIGHT_MM / 2.0)  # ~95.5 mm w dół od kamery

# Domyślne parametry kamery FaceTime HD 720p (HFOV ~ 58.5°)
K = np.array([
    [1140.0,    0.0, 640.0],
    [   0.0, 1140.0, 360.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float64)
D = np.zeros((5, 1), dtype=np.float64)

# Konfiguracja Tasks API
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
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
        points_2d = []

        for idx in LANDMARK_IDS:
            pt = face[idx]
            x_f, y_f = pt.x * w, pt.y * h
            points_2d.append([x_f, y_f])
            cv2.circle(frame, (int(x_f), int(y_f)), 3, (0, 0, 255), -1)

        # Zaznaczanie źrenic na obrazie
        for iris_id, col in [(IRIS_RIGHT_ID, (255, 0, 0)), (IRIS_LEFT_ID, (0, 255, 255))]:
            if iris_id < len(face):
                iris_pt = face[iris_id]
                cv2.circle(frame, (int(iris_pt.x * w), int(iris_pt.y * h)), 4, col, -1)

        # Obliczenie pozycji 3D (solvePnP)
        image_points = np.array(points_2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            MODEL_3D, image_points, K, D, flags=cv2.SOLVEPNP_SQPNP
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t_head = tvec.ravel()

            # Precyzyjne pozycje 3D obu oczu w układzie kamery (w mm): P_cam = R * P_head + t
            eye_r_3d = R @ EYE_RIGHT_HEAD + t_head
            eye_l_3d = R @ EYE_LEFT_HEAD  + t_head
            ipd_dist = np.linalg.norm(eye_r_3d - eye_l_3d)

            # Obliczenie kąta obrotu głowy (Yaw)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            yaw = np.degrees(np.arctan2(R[2, 1], R[2, 2])) if sy < 1e-6 else np.degrees(np.arctan2(-R[2, 0], sy))

            # Konwersja do układu współrzędnych środka ekranu MacBooka (w mm):
            # Środek (0,0,0) = idealny środek ekranu
            # +X = w prawo, +Y = w górę (oko wyżej -> Y rośnie), +Z = odległość od płaszczyzny ekranu
            eye_r_screen = np.array([
                eye_r_3d[0],
                CAMERA_TO_SCREEN_CENTER_Y_MM - eye_r_3d[1],
                eye_r_3d[2]
            ])
            eye_l_screen = np.array([
                eye_l_3d[0],
                CAMERA_TO_SCREEN_CENTER_Y_MM - eye_l_3d[1],
                eye_l_3d[2]
            ])

            # Wyświetlanie koordynatów oczu na ekranie (układ środka ekranu)
            t_r = f"Ekran: Prawe (R): X:{eye_r_screen[0]:6.1f} Y:{eye_r_screen[1]:6.1f} Z:{eye_r_screen[2]:6.1f} mm"
            t_l = f"Ekran: Lewe  (L): X:{eye_l_screen[0]:6.1f} Y:{eye_l_screen[1]:6.1f} Z:{eye_l_screen[2]:6.1f} mm"
            t_ipd = f"Rozstaw: {ipd_dist:5.1f} mm | Yaw: {yaw:5.1f} deg | Srodek ekranu (0,0)"

            cv2.putText(frame, t_r, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, t_l, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, t_ipd, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Precyzyjne koordynaty 3D oczu (Układ Ekranu)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()

