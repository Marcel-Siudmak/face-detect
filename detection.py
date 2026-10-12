import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model antropometryczny twarzy 3D w mm
LANDMARKS_DICT = {
    168: [  0.0,   0.0,   0.0],  # Nasada nosa
      4: [  0.0,  30.0, -22.0],  # Czubek nosa
     10: [  0.0, -75.0,  15.0],  # Czoło
    152: [  0.0, 105.0,  10.0],  # Broda
     33: [-46.0,   3.0,  18.0],  # Prawy kącik oka zewn.
    133: [-16.0,   3.0,   4.0],  # Prawy kącik oka wewn.
    362: [ 16.0,   3.0,   4.0],  # Lewy kącik oka wewn.
    263: [ 46.0,   3.0,  18.0],  # Lewy kącik oka zewn.
     61: [-28.0,  68.0,   8.0],  # Prawy kącik ust
    291: [ 28.0,  68.0,   8.0],  # Lewy kącik ust
}

LANDMARK_IDS = list(LANDMARKS_DICT.keys())
MODEL_3D = np.array(list(LANDMARKS_DICT.values()), dtype=np.float64)

EYE_RIGHT_HEAD = np.array([-31.5, 3.0, 8.0])  # viewer's left
EYE_LEFT_HEAD  = np.array([ 31.5, 3.0, 8.0])  # viewer's right

IRIS_RIGHT_ID = 468
IRIS_LEFT_ID  = 473

# Parametry MacBook Air 13.3" (M1)
SCREEN_WIDTH_MM = 286.5
SCREEN_HEIGHT_MM = 179.0
BEZEL_TOP_TO_SCREEN_MM = 6.0
CAMERA_TO_SCREEN_CENTER_Y_MM = BEZEL_TOP_TO_SCREEN_MM + (SCREEN_HEIGHT_MM / 2.0)  # ~95.5 mm

K = np.array([
    [1140.0,    0.0, 640.0],
    [   0.0, 1140.0, 360.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float64)
D = np.zeros((5, 1), dtype=np.float64)

# --- MARTWE STREFY I PROGI CZUŁOŚCI ---
DEADZONE_X_MM = 30.0        # 3.0 cm martwej strefy poziomej od centrum
MAX_DIST_X_MM = 135.0       # Pełna intencja pozioma przy 13.5 cm
DEADZONE_ANGLE_SIN = 0.12   # ~7 stopni martwej strefy skrętu twarzy
MAX_ANGLE_SIN = 0.38        # Pełna intencja kątowa przy ~22 stopniach

DEADZONE_Y_MM = 25.0        # 2.5 cm martwej strefy pionowej
MAX_DIST_Y_MM = 90.0        # Pełna intencja pionowa przy 9 cm
TILT_DEADZONE_MM = 8.0      # Martwa strefa przechyłu oczu w pionie
TILT_MAX_MM = 22.0          # Maksymalny przechył do wyboru oka

# --- STAŁE FILTRACJI CZASOWEJ (PŁYWANIE I STABILNOŚĆ) ---
ALPHA_SMOOTH_RATE = 4.5     # Szybkość przepływu wagi kamery (mniejsza = bardziej leniwe pływanie)
POS_SMOOTH_RATE   = 14.0    # Szybkość wygładzania drgań pozycji 3D z PnP

y_neutral_mm = CAMERA_TO_SCREEN_CENTER_Y_MM

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


def smoothstep(edge0, edge1, x):
    """Krzywa nieliniowa zapewniająca gładki start (zerową pochodną na krawędziach)."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def draw_hud(img, eye_l, eye_r, cam, normal, alpha, mode_text, y_rel, tilt_y):
    panel_w, panel_h = 380, 680
    px, py = img.shape[1] - panel_w - 20, 20

    sub = img[py:py+panel_h, px:px+panel_w]
    dark_rect = np.full_like(sub, 20)
    img[py:py+panel_h, px:px+panel_w] = cv2.addWeighted(sub, 0.25, dark_rect, 0.75, 0)
    cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    # 1. Suwak wagi z pozycją smoothed
    cv2.putText(img, "TRYB INTENCJI:", (px + 15, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, mode_text, (px + 15, py + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    bar_x, bar_y, bar_w = px + 35, py + 85, 300
    cv2.line(img, (bar_x, bar_y), (bar_x + bar_w, bar_y), (100, 100, 100), 4)
    cv2.circle(img, (bar_x, bar_y), 5, (255, 200, 0), -1)
    cv2.circle(img, (bar_x + bar_w // 2, bar_y), 4, (180, 180, 180), -1)
    cv2.circle(img, (bar_x + bar_w, bar_y), 5, (0, 255, 0), -1)

    slider_pos = int(bar_x + (bar_w // 2) + alpha * (bar_w // 2))
    cv2.circle(img, (slider_pos, bar_y), 8, (0, 140, 255), -1)
    cv2.circle(img, (slider_pos, bar_y), 10, (0, 200, 255), 2)

    cv2.putText(img, "LEWE (L)", (bar_x - 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1)
    cv2.putText(img, "SRODEK", (bar_x + bar_w // 2 - 20, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(img, "PRAWE (R)", (bar_x + bar_w - 25, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

    t_diag = f"Y_rel: {y_rel:+4.0f}mm | Tilt: {tilt_y:+3.0f}mm | 'c'=kalibracja"
    cv2.putText(img, t_diag, (px + 15, py + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # 2. Widok z góry (X - Z)
    cv2.putText(img, "WIDOK Z GORY (X - Z)", (px + 15, py + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    radar_z_top = py + 185
    radar_cx = px + panel_w // 2
    scale_xz = 0.32

    scr_half_w = int((SCREEN_WIDTH_MM / 2.0) * scale_xz)
    cv2.line(img, (radar_cx - scr_half_w, radar_z_top), (radar_cx + scr_half_w, radar_z_top), (255, 255, 255), 3)

    def to_radar_xz(pt):
        return (int(radar_cx + pt[0] * scale_xz), int(radar_z_top + pt[2] * scale_xz))

    p_el_xz = to_radar_xz(eye_l)
    p_er_xz = to_radar_xz(eye_r)
    p_cam_xz = to_radar_xz(cam)

    cv2.line(img, p_el_xz, p_er_xz, (120, 120, 120), 2)
    cv2.circle(img, p_el_xz, 5, (255, 200, 0), -1)
    cv2.circle(img, p_er_xz, 5, (0, 255, 0), -1)

    p_head_xz = ((p_el_xz[0] + p_er_xz[0]) // 2, (p_el_xz[1] + p_er_xz[1]) // 2)
    cv2.arrowedLine(img, p_head_xz,
                    (int(p_head_xz[0] + normal[0] * 35), int(p_head_xz[1] - normal[2] * 35)),
                    (0, 255, 255), 2, tipLength=0.25)

    cv2.line(img, p_cam_xz, (radar_cx - scr_half_w, radar_z_top), (0, 80, 180), 1, cv2.LINE_AA)
    cv2.line(img, p_cam_xz, (radar_cx + scr_half_w, radar_z_top), (0, 80, 180), 1, cv2.LINE_AA)
    cv2.circle(img, p_cam_xz, 6, (0, 140, 255), -1)

    # 3. Widok od przodu (X - Y)
    cv2.putText(img, "WIDOK OD PRZODU (X - Y)", (px + 15, py + 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    front_cx = px + panel_w // 2
    front_cy = py + 550
    scale_xy = 0.55

    rect_w = int(SCREEN_WIDTH_MM * scale_xy)
    rect_h = int(SCREEN_HEIGHT_MM * scale_xy)
    cv2.rectangle(img,
                  (front_cx - rect_w // 2, front_cy - rect_h // 2),
                  (front_cx + rect_w // 2, front_cy + rect_h // 2),
                  (140, 140, 140), 1)

    def to_front_xy(pt):
        return (int(front_cx + pt[0] * scale_xy), int(front_cy - pt[1] * scale_xy))

    p_el_xy = to_front_xy(eye_l)
    p_er_xy = to_front_xy(eye_r)
    p_cam_xy = to_front_xy(cam)

    cv2.line(img, p_el_xy, p_er_xy, (120, 120, 120), 2)
    cv2.circle(img, p_el_xy, 5, (255, 200, 0), -1)
    cv2.circle(img, p_er_xy, 5, (0, 255, 0), -1)
    cv2.circle(img, p_cam_xy, 7, (0, 140, 255), -1)
    cv2.circle(img, p_cam_xy, 9, (0, 255, 255), 2)


# Zmienne stanu filtrów czasowych
alpha_smooth = 0.0
cam_smooth = None
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    dt = np.clip(curr_time - prev_time, 0.001, 0.1)
    prev_time = curr_time

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

        for iris_id, col in [(IRIS_RIGHT_ID, (255, 0, 0)), (IRIS_LEFT_ID, (0, 255, 255))]:
            if iris_id < len(face):
                iris_pt = face[iris_id]
                cv2.circle(frame, (int(iris_pt.x * w), int(iris_pt.y * h)), 4, col, -1)

        image_points = np.array(points_2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            MODEL_3D, image_points, K, D, flags=cv2.SOLVEPNP_SQPNP
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t_head = tvec.ravel()

            eye_r_3d = R @ EYE_RIGHT_HEAD + t_head
            eye_l_3d = R @ EYE_LEFT_HEAD  + t_head

            eye_r_screen = np.array([
                -eye_r_3d[0],
                CAMERA_TO_SCREEN_CENTER_Y_MM - eye_r_3d[1],
                eye_r_3d[2]
            ])
            eye_l_screen = np.array([
                -eye_l_3d[0],
                CAMERA_TO_SCREEN_CENTER_Y_MM - eye_l_3d[1],
                eye_l_3d[2]
            ])

            head_center = (eye_l_screen + eye_r_screen) * 0.5
            eye_base_3d = eye_r_screen - eye_l_screen

            cam_forward = -R[:, 2]
            face_normal_screen = np.array([-cam_forward[0], -cam_forward[1], cam_forward[2]])

            delta_eye_y = eye_r_screen[1] - eye_l_screen[1]
            abs_tilt_y = abs(delta_eye_y)
            y_rel = head_center[1] - y_neutral_mm

            # --- 1. INTENCJA POZIOMA (ZAUŁEK) Z MARTWĄ STREFĄ I KRZYWĄ GŁADKĄ ---
            abs_x = abs(head_center[0])
            factor_pos_x = smoothstep(DEADZONE_X_MM, MAX_DIST_X_MM, abs_x)

            # Intencja aktywuje się wyłącznie, gdy kąt obrotu twarzy jest przeciwny do pozycji
            sign_pos_x = np.sign(head_center[0]) if abs_x > 1e-3 else 0.0
            opposing_rot_x = -sign_pos_x * face_normal_screen[0]
            factor_rot_x = smoothstep(DEADZONE_ANGLE_SIN, MAX_ANGLE_SIN, opposing_rot_x)

            intent_x = factor_pos_x * factor_rot_x
            target_alpha_x = -sign_pos_x * intent_x

            # --- 2. INTENCJA PIONOWA (MUREK / DASZEK) ---
            tilt_strength = smoothstep(TILT_DEADZONE_MM, TILT_MAX_MM, abs_tilt_y)

            higher_eye_dir = 1.0 if delta_eye_y > 0 else -1.0
            lower_eye_dir  = -higher_eye_dir

            # Murek: głowa uniesiona + wzrok w dół + przechył -> WYŻSZE OKO
            factor_pos_top = smoothstep(DEADZONE_Y_MM, MAX_DIST_Y_MM, y_rel)
            pitch_down = -face_normal_screen[1]
            factor_rot_top = smoothstep(DEADZONE_ANGLE_SIN, MAX_ANGLE_SIN, pitch_down)
            intent_top = factor_pos_top * factor_rot_top * tilt_strength

            # Daszek: głowa opuszczona + przechył -> NIŻSZE OKO
            factor_pos_bottom = smoothstep(DEADZONE_Y_MM, MAX_DIST_Y_MM, -y_rel)
            intent_bottom = factor_pos_bottom * tilt_strength

            # --- 3. WYZNACZENIE CELU WAGI (TARGET ALPHA) ---
            if intent_top > intent_x and intent_top > intent_bottom and intent_top > 0.02:
                target_alpha = higher_eye_dir * intent_top
                mode_str = f"MUREK -> {'PRAWE' if higher_eye_dir > 0 else 'LEWE'} (wyzsze)"
            elif intent_bottom > intent_x and intent_bottom > intent_top and intent_bottom > 0.02:
                target_alpha = lower_eye_dir * intent_bottom
                mode_str = f"DASZEK -> {'PRAWE' if lower_eye_dir > 0 else 'LEWE'} (nizsze)"
            elif intent_x > 0.02:
                target_alpha = target_alpha_x
                mode_str = f"ZAULEK -> {'LEWE' if target_alpha_x < 0 else 'PRAWE'}"
            else:
                target_alpha = 0.0
                mode_str = "NEUTRALNY (SRODEK)"

            # --- 4. FILTRACJA CZASOWA: PŁYWANIE WAGI KAMERY ---
            alpha_blend = 1.0 - np.exp(-ALPHA_SMOOTH_RATE * dt)
            alpha_smooth += (target_alpha - alpha_smooth) * alpha_blend

            # Surowa pozycja kamery na szynie ocznej
            raw_cam_screen = head_center + alpha_smooth * (eye_base_3d * 0.5)

            # --- 5. FILTRACJA CZASOWA: TŁUMIENIE DRGAŃ POZYCJI 3D ---
            if cam_smooth is None:
                cam_smooth = raw_cam_screen.copy()
            else:
                pos_blend = 1.0 - np.exp(-POS_SMOOTH_RATE * dt)
                cam_smooth += (raw_cam_screen - cam_smooth) * pos_blend

            draw_hud(frame, eye_l_screen, eye_r_screen, cam_smooth,
                     face_normal_screen, alpha_smooth, mode_str, y_rel, delta_eye_y)

            t_r   = f"Prawe: X:{eye_r_screen[0]:4.0f} Y:{eye_r_screen[1]:4.0f} Z:{eye_r_screen[2]:4.0f}"
            t_l   = f"Lewe : X:{eye_l_screen[0]:4.0f} Y:{eye_l_screen[1]:4.0f} Z:{eye_l_screen[2]:4.0f}"
            t_cam = f"KAMERA: X:{cam_smooth[0]:4.0f} Y:{cam_smooth[1]:4.0f} Z:{cam_smooth[2]:4.0f} (a:{alpha_smooth:+0.2f})"
            cv2.putText(frame, t_r,   (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, t_l,   (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(frame, t_cam, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    cv2.imshow("Tracking 3D - Peeking Biomechanics", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and 'head_center' in locals():
        y_neutral_mm = head_center[1]

detector.close()
cap.release()
cv2.destroyAllWindows()

