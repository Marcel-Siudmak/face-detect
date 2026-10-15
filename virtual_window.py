import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# PARAMETRY SPRZĘTOWE: MacBook Air 13.3" (M1)
# ==============================================================================
# Fizyczny rozmiar aktywnej matrycy Retina (16:10)
SCREEN_WIDTH_MM = 286.5
SCREEN_HEIGHT_MM = 179.0
BEZEL_TOP_TO_SCREEN_MM = 6.0
CAMERA_TO_SCREEN_CENTER_Y_MM = BEZEL_TOP_TO_SCREEN_MM + (SCREEN_HEIGHT_MM / 2.0)  # ~95.5 mm

# Rozdzielczość okna wirtualnego (dokładnie proporcje 16:10)
WIN_W = 1280
WIN_H = 800

# Parametry kamery FaceTime HD 720p
K = np.array([
    [1140.0,    0.0, 640.0],
    [   0.0, 1140.0, 360.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float64)
D = np.zeros((5, 1), dtype=np.float64)

# ==============================================================================
# MODEL ANTROPOMETRYCZNY TWARZY (10 punktów)
# ==============================================================================
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

# Parametry wygładzania
ALPHA_SMOOTH_RATE = 5.0
POS_SMOOTH_RATE   = 16.0

DEADZONE_X_MM = 30.0
MAX_DIST_X_MM = 135.0
DEADZONE_ANGLE_SIN = 0.12
MAX_ANGLE_SIN = 0.38
DEADZONE_Y_MM = 25.0
MAX_DIST_Y_MM = 90.0
TILT_DEADZONE_MM = 8.0
TILT_MAX_MM = 22.0

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ==============================================================================
# GEOMETRIA SCENY: PŁASKI STÓŁ I KOSTKI RUBIKA (1:1 W MILIMETRACH)
# ==============================================================================
HW = SCREEN_WIDTH_MM / 2.0  # 143.25 mm
HH = SCREEN_HEIGHT_MM / 2.0  # 89.50 mm
TABLE_DEPTH_MM = 500.0       # Głębokość stołu za ekranem

# Standardowe wymiary kostki Rubika: 57 mm x 57 mm x 57 mm
RUBIK_SIZE = 57.0
s = RUBIK_SIZE / 2.0

# Poziom blatu stołu: idealnie dolna krawędź ekranu
TABLE_Y = -HH  # -89.5 mm
# Wysokość środka kostki spoczywającej na stole
CUBE_Y = TABLE_Y + s  # -61.0 mm

# 1. KOSTKA CENTRALNA: przednia ścianka dotyka dokładnie ekranu (Z = 0)
# Środek kostki w osi Z to -s = -28.5 mm (przednia ściana Z = 0, tylna Z = -57 mm)
CUBE_CENTER_POS = np.array([0.0, CUBE_Y, -s])

# 2. KOSTKA LEWA I PRAWA: odsunięte w bok tak, aby stojąc na wprost (oko X=0, Z=500)
# nie były widoczne przez ramę okna, a odsłaniały się po wychyleniu głowy.
# Promień ze środka (0, 0, 500) przez krawędź okna (HW = 143.25) na głębokości Z = -160 mm
# wyznacza granicę widoczności na X = 143.25 * (500 + 160) / 500 = 189 mm.
# Umieszczamy kostki boczne na X = +/-205 mm, Z = -160 mm.
SIDE_Z = -160.0
CUBE_LEFT_POS  = np.array([-205.0, CUBE_Y, SIDE_Z])
CUBE_RIGHT_POS = np.array([ 205.0, CUBE_Y, SIDE_Z])

# Kolory ścian kostki Rubika (standardowy układ kolorów w BGR)
# Przód (+Z): Niebieski, Tył (-Z): Zielony, Góra (+Y): Żółty, Dół (-Y): Biały, Prawa (+X): Czerwony, Lewa (-X): Pomarańczowy
COLOR_BLUE   = (210,  50,  30)
COLOR_GREEN  = ( 40, 180,  40)
COLOR_YELLOW = ( 30, 220, 240)
COLOR_WHITE  = (230, 230, 230)
COLOR_RED    = ( 30,  30, 220)
COLOR_ORANGE = ( 20, 120, 235)
COLOR_BLACK  = ( 20,  20,  20)


def build_rubik_mesh(center_pos, size=RUBIK_SIZE):
    """
    Tworzy geometrię kostki Rubika z podziałem 3x3 naklejek na każdej ściance.
    Zwraca listę ścian bazowych i 54 naklejki (sub-faces).
    """
    cx, cy, cz = center_pos
    hs = size / 2.0

    # Definicje 6 głównych ścian: (baza, wektor_u, wektor_v, wektor_normalny, kolor_naklejek)
    face_configs = [
        # Przód (+Z)
        (np.array([cx - hs, cy - hs, cz + hs]), np.array([size, 0, 0]), np.array([0, size, 0]), np.array([0, 0, 1.0]), COLOR_BLUE),
        # Tył (-Z)
        (np.array([cx + hs, cy - hs, cz - hs]), np.array([-size, 0, 0]), np.array([0, size, 0]), np.array([0, 0, -1.0]), COLOR_GREEN),
        # Prawa (+X)
        (np.array([cx + hs, cy - hs, cz + hs]), np.array([0, 0, -size]), np.array([0, size, 0]), np.array([1.0, 0, 0]), COLOR_RED),
        # Lewa (-X)
        (np.array([cx - hs, cy - hs, cz - hs]), np.array([0, 0, size]), np.array([0, size, 0]), np.array([-1.0, 0, 0]), COLOR_ORANGE),
        # Góra (+Y)
        (np.array([cx - hs, cy + hs, cz + hs]), np.array([size, 0, 0]), np.array([0, 0, -size]), np.array([0, 1.0, 0]), COLOR_YELLOW),
        # Dół (-Y)
        (np.array([cx - hs, cy - hs, cz - hs]), np.array([size, 0, 0]), np.array([0, 0, size]), np.array([0, -1.0, 0]), COLOR_WHITE),
    ]

    elements = []
    gap_ratio = 0.10  # Czarna fuga między naklejkami (10%)

    for origin, u_vec, v_vec, normal, sticker_col in face_configs:
        # Główny czarny plastik podkładowy pod ścianką
        base_quad = np.array([
            origin,
            origin + u_vec,
            origin + u_vec + v_vec,
            origin + v_vec
        ])
        center_face = origin + 0.5 * u_vec + 0.5 * v_vec

        # Generuj siatkę 3x3 małych naklejek
        u_step = u_vec / 3.0
        v_step = v_vec / 3.0
        gap_u = u_step * gap_ratio
        gap_v = v_step * gap_ratio

        stickers = []
        for i in range(3):
            for j in range(3):
                p0 = origin + i * u_step + j * v_step + gap_u * 0.5 + gap_v * 0.5
                p1 = p0 + u_step * (1.0 - gap_ratio)
                p2 = p1 + v_step * (1.0 - gap_ratio)
                p3 = p0 + v_step * (1.0 - gap_ratio)
                stickers.append(np.array([p0, p1, p2, p3]))

        elements.append({
            'normal': normal,
            'center': center_face,
            'base_quad': base_quad,
            'stickers': stickers,
            'sticker_col': sticker_col
        })

    return elements


# ==============================================================================
# RZUTOWANIE OFF-AXIS PERSPECTIVE (RAY - SCREEN INTERSECTION)
# ==============================================================================
def project_off_axis(points_3d, eye_pos):
    xe, ye, ze = eye_pos
    ze = max(ze, 80.0)

    denominator = (ze - points_3d[:, 2])
    denominator = np.where(np.abs(denominator) < 1e-4, 1e-4, denominator)
    t = ze / denominator

    x_scr = xe + t * (points_3d[:, 0] - xe)
    y_scr = ye + t * (points_3d[:, 1] - ye)

    u = ((x_scr + HW) / SCREEN_WIDTH_MM) * WIN_W
    v = ((HH - y_scr) / SCREEN_HEIGHT_MM) * WIN_H

    return np.column_stack((u, v)).astype(np.int32)


# ==============================================================================
# GŁÓWNA PĘTLA APLIKACJI
# ==============================================================================
def main():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Inicjalizacja kamery: automatyczny wybór dostępnego indeksu (0 lub 1)
    cap = None
    for cam_idx in [0, 1]:
        temp_cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"[+] Camera initialized successfully on index {cam_idx}")
                break
            temp_cap.release()

    if cap is None:
        raise RuntimeError("Nie znaleziono działającej kamery! Sprawdź uprawnienia do kamery w macOS.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    default_eye = np.array([0.0, 0.0, 500.0], dtype=np.float64)
    cam_smooth = default_eye.copy()
    alpha_smooth = 0.0
    prev_time = time.time()
    y_neutral_mm = CAMERA_TO_SCREEN_CENTER_Y_MM

    cv2.namedWindow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", WIN_W, WIN_H)

    # Budujemy siatki dla 3 kostek Rubika
    center_cube_mesh = build_rubik_mesh(CUBE_CENTER_POS)
    left_cube_mesh   = build_rubik_mesh(CUBE_LEFT_POS)
    right_cube_mesh  = build_rubik_mesh(CUBE_RIGHT_POS)

    print("=== Wirtualne Okno: Mierzalna Scena z Kostkami Rubika ===")
    print("Centralna kostka 57mm lezy na stole od razu przy ekranie (Z=0).")
    print("Dwie boczne kostki sa schowane za ramka - wychyl sie w prawo/lewo, aby je dojrzec!")
    print("'c' = kalibracja neutralnej wysokosci | 'q' = wyjscie")

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

        face_detected = False

        if result.face_landmarks:
            face = result.face_landmarks[0]
            points_2d = []

            for idx in LANDMARK_IDS:
                pt = face[idx]
                x_f, y_f = pt.x * w, pt.y * h
                points_2d.append([x_f, y_f])

            image_points = np.array(points_2d, dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(
                MODEL_3D, image_points, K, D, flags=cv2.SOLVEPNP_SQPNP
            )

            if success:
                face_detected = True
                R, _ = cv2.Rodrigues(rvec)
                t_head = tvec.ravel()

                eye_r_3d = R @ EYE_RIGHT_HEAD + t_head
                eye_l_3d = R @ EYE_LEFT_HEAD  + t_head

                # Prawidłowa orientacja osi X (usunięcie efektu lustra)
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

                # Intencja pozioma
                abs_x = abs(head_center[0])
                factor_pos_x = smoothstep(DEADZONE_X_MM, MAX_DIST_X_MM, abs_x)
                sign_pos_x = np.sign(head_center[0]) if abs_x > 1e-3 else 0.0
                opposing_rot_x = -sign_pos_x * face_normal_screen[0]
                factor_rot_x = smoothstep(DEADZONE_ANGLE_SIN, MAX_ANGLE_SIN, opposing_rot_x)
                intent_x = factor_pos_x * factor_rot_x
                target_alpha_x = -sign_pos_x * intent_x

                # Intencja pionowa
                tilt_strength = smoothstep(TILT_DEADZONE_MM, TILT_MAX_MM, abs_tilt_y)
                higher_eye_dir = 1.0 if delta_eye_y > 0 else -1.0
                lower_eye_dir  = -higher_eye_dir

                factor_pos_top = smoothstep(DEADZONE_Y_MM, MAX_DIST_Y_MM, y_rel)
                pitch_down = -face_normal_screen[1]
                factor_rot_top = smoothstep(DEADZONE_ANGLE_SIN, MAX_ANGLE_SIN, pitch_down)
                intent_top = factor_pos_top * factor_rot_top * tilt_strength

                factor_pos_bottom = smoothstep(DEADZONE_Y_MM, MAX_DIST_Y_MM, -y_rel)
                intent_bottom = factor_pos_bottom * tilt_strength

                if intent_top > intent_x and intent_top > intent_bottom and intent_top > 0.02:
                    target_alpha = higher_eye_dir * intent_top
                elif intent_bottom > intent_x and intent_bottom > intent_top and intent_bottom > 0.02:
                    target_alpha = lower_eye_dir * intent_bottom
                elif intent_x > 0.02:
                    target_alpha = target_alpha_x
                else:
                    target_alpha = 0.0

                alpha_blend = 1.0 - np.exp(-ALPHA_SMOOTH_RATE * dt)
                alpha_smooth += (target_alpha - alpha_smooth) * alpha_blend

                raw_cam_screen = head_center + alpha_smooth * (eye_base_3d * 0.5)

                pos_blend = 1.0 - np.exp(-POS_SMOOTH_RATE * dt)
                cam_smooth += (raw_cam_screen - cam_smooth) * pos_blend

        if not face_detected:
            cam_smooth += (default_eye - cam_smooth) * (1.0 - np.exp(-2.0 * dt))

        # ======================================================================
        # RENDEROWANIE SCENY MIERZALNEJ
        # ======================================================================
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        # 1. Ściany pokoju / tło za stołem
        ROOM_BACK_Z = -TABLE_DEPTH_MM
        pts_room_back = project_off_axis(np.array([
            [-HW * 1.5, HH, ROOM_BACK_Z], [ HW * 1.5, HH, ROOM_BACK_Z],
            [ HW * 1.5, TABLE_Y, ROOM_BACK_Z], [-HW * 1.5, TABLE_Y, ROOM_BACK_Z]
        ]), cam_smooth)

        # Ciemnoszara ściana za stołem
        cv2.fillPoly(canvas, [pts_room_back], (30, 32, 36))

        # 2. Płaski stół (od Z=0 na dolnej krawędzi ekranu do Z=-500 mm w głąb)
        # Narożniki stołu:
        table_front_left  = np.array([-HW * 2.0, TABLE_Y, 0.0])
        table_front_right = np.array([ HW * 2.0, TABLE_Y, 0.0])
        table_back_left   = np.array([-HW * 2.0, TABLE_Y, ROOM_BACK_Z])
        table_back_right  = np.array([ HW * 2.0, TABLE_Y, ROOM_BACK_Z])

        pts_table = project_off_axis(np.array([
            table_front_left, table_front_right, table_back_right, table_back_left
        ]), cam_smooth)

        # Kolor blatu stołu (elegancki grafit / ciemne drewno)
        cv2.fillPoly(canvas, [pts_table], (48, 50, 56))

        # Metryczna siatka pomiarowa na stole (linie co 50 mm)
        # Linie wzdłuż osi X (równoległe do ekranu co 50 mm w głąb)
        for z_grid in np.arange(0.0, -TABLE_DEPTH_MM - 1.0, -50.0):
            g_line = np.array([[-HW * 2.0, TABLE_Y, z_grid], [HW * 2.0, TABLE_Y, z_grid]])
            pts_gl = project_off_axis(g_line, cam_smooth)
            col_line = (120, 120, 130) if z_grid == 0.0 else (68, 72, 80)
            thick = 2 if z_grid == 0.0 else 1
            cv2.line(canvas, tuple(pts_gl[0]), tuple(pts_gl[1]), col_line, thick, cv2.LINE_AA)

        # Linie wzdłuż osi Z (prostopadłe do ekranu co 50 mm)
        for x_grid in np.arange(-250.0, 251.0, 50.0):
            g_line = np.array([[x_grid, TABLE_Y, 0.0], [x_grid, TABLE_Y, ROOM_BACK_Z]])
            pts_gl = project_off_axis(g_line, cam_smooth)
            col_line = (90, 100, 115) if abs(x_grid) < 1e-3 else (64, 68, 76)
            thick = 2 if abs(x_grid) < 1e-3 else 1
            cv2.line(canvas, tuple(pts_gl[0]), tuple(pts_gl[1]), col_line, thick, cv2.LINE_AA)

        # 3. Zbieranie wszystkich ścian 3 kostek Rubika do posortowania wg głębi
        render_polys = []

        all_cubes = [
            ("Centralna", center_cube_mesh),
            ("Lewa", left_cube_mesh),
            ("Prawa", right_cube_mesh)
        ]

        left_visible = False
        right_visible = False

        for cube_name, cube_mesh in all_cubes:
            for face_data in cube_mesh:
                normal = face_data['normal']
                center = face_data['center']
                view_vec = cam_smooth - center

                # Backface culling w 3D: rysuj tylko ściany zwrócone ku oku
                if np.dot(normal, view_vec) > 0:
                    dist = np.linalg.norm(view_vec)

                    # Rzutuj czarną podstawę ścianki
                    proj_base = project_off_axis(face_data['base_quad'], cam_smooth)

                    # Rzutuj 9 naklejek na ściance
                    proj_stickers = [project_off_axis(st, cam_smooth) for st in face_data['stickers']]

                    # Sprawdź widoczność bocznych kostek w granicach okna
                    if cube_name == "Lewa":
                        for pt in proj_base:
                            if 0 <= pt[0] < WIN_W and 0 <= pt[1] < WIN_H:
                                left_visible = True
                    elif cube_name == "Prawa":
                        for pt in proj_base:
                            if 0 <= pt[0] < WIN_W and 0 <= pt[1] < WIN_H:
                                right_visible = True

                    render_polys.append({
                        'dist': dist,
                        'base_poly': proj_base,
                        'stickers': proj_stickers,
                        'sticker_col': face_data['sticker_col']
                    })

        # Sortowanie ścian od najdalszych do najbliższych (Painter's Algorithm)
        render_polys.sort(key=lambda item: item['dist'], reverse=True)

        for poly_item in render_polys:
            # 1. Rysuj czarną bryłę ścianki (plastik)
            cv2.fillPoly(canvas, [poly_item['base_poly']], COLOR_BLACK)
            cv2.polylines(canvas, [poly_item['base_poly']], isClosed=True, color=(10, 10, 10), thickness=2, lineType=cv2.LINE_AA)

            # 2. Rysuj 9 kolorowych naklejek na ściance z czarnymi obwódkami
            for st_poly in poly_item['stickers']:
                cv2.fillPoly(canvas, [st_poly], poly_item['sticker_col'])
                cv2.polylines(canvas, [st_poly], isClosed=True, color=(15, 15, 15), thickness=1, lineType=cv2.LINE_AA)

        # 4. Krawędź fizycznego okna (rama ekranu MacBooka)
        cv2.rectangle(canvas, (0, 0), (WIN_W - 1, WIN_H - 1), (180, 180, 180), 3)

        # Znacznik linii Z = 0 na dolnej krawędzi (krawędź styku stołu z ekranem)
        cv2.line(canvas, (0, WIN_H - 2), (WIN_W, WIN_H - 2), (0, 240, 255), 4)

        # 5. Mierzalny HUD diagnostyczny
        cv2.putText(canvas, "SCENA MIERZALNA: KOSTKI RUBIKA 57mm (1:1)", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        t_eye = f"Pozycja oka: X: {cam_smooth[0]:+5.1f} mm | Y: {cam_smooth[1]:+5.1f} mm | Z: {cam_smooth[2]:5.1f} mm"
        cv2.putText(canvas, t_eye, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

        # Status widoczności kostek
        t_center = "Kostka centralna (Z=0mm przy szybie): WIDOCZNA"
        cv2.putText(canvas, t_center, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 2, cv2.LINE_AA)

        col_l = (100, 255, 100) if left_visible else (140, 140, 150)
        status_l = "WIDOCZNA!" if left_visible else "UKRYTA (wychyl sie w PRAWO!)"
        cv2.putText(canvas, f"Lewa kostka (X=-205mm): {status_l}", (25, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_l, 2, cv2.LINE_AA)

        col_r = (100, 255, 100) if right_visible else (140, 140, 150)
        status_r = "WIDOCZNA!" if right_visible else "UKRYTA (wychyl sie w LEWO!)"
        cv2.putText(canvas, f"Prawa kostka (X=+205mm): {status_r}", (25, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_r, 2, cv2.LINE_AA)

        # 6. Podgląd kamery w prawym dolnym rogu (Picture-in-Picture)
        pip_w, pip_h = 240, 135
        pip_frame = cv2.flip(cv2.resize(frame, (pip_w, pip_h)), 1)
        canvas[WIN_H - pip_h - 15:WIN_H - 15, WIN_W - pip_w - 15:WIN_W - 15] = pip_frame
        cv2.rectangle(canvas, (WIN_W - pip_w - 15, WIN_H - pip_h - 15),
                      (WIN_W - 15, WIN_H - 15), (0, 240, 255), 1)

        cv2.imshow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and 'head_center' in locals():
            y_neutral_mm = head_center[1]

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
