from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import (
    ALPHA_ATTACK_RATE,
    ALPHA_CENTER_RATE,
    ALPHA_RELEASE_RATE,
    CAMERA_DISTORTION_D,
    CAMERA_HEIGHT,
    CAMERA_INTRINSIC_K,
    CAMERA_TO_SCREEN_CENTER_Y_MM,
    CAMERA_WIDTH,
    DEFAULT_MODEL_PATH,
    EAR_ASYMMETRY_MIN,
    EAR_SQUINT_THRESHOLD,
    EYE_L_INNER_ID,
    EYE_L_LOWER_ID,
    EYE_L_OUTER_ID,
    EYE_L_UPPER_ID,
    EYE_LEFT_HEAD,
    EYE_R_INNER_ID,
    EYE_R_LOWER_ID,
    EYE_R_OUTER_ID,
    EYE_R_UPPER_ID,
    EYE_RIGHT_HEAD,
    GAZE_CONVERGENCE_BOOST,
    GAZE_DIRECT_PEEK_GAIN,
    GAZE_EYE_WEIGHT,
    IRIS_LEFT_ID,
    IRIS_RIGHT_ID,
    LANDMARK_IDS,
    LOOK_AWAY_THRESHOLD,
    MODEL_3D,
    MP_DETECT_HEIGHT,
    MP_DETECT_WIDTH,
    PEEK_CURVE_EXP,
    PEEK_DEADZONE_MM,
    PEEK_FULL_DIST_MM,
    POS_FAST_RATE,
    POS_SMOOTH_RATE,
    VELOCITY_LEAD_GAIN,
    VELOCITY_SMOOTH_RATE,
    VERTICAL_TILT_GAIN,
)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Nieliniowa funkcja sigmoidalna o zerowych pochodnych na krawędziach."""
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def compute_eye_gaze_and_ear(
    face_pts_2d: Dict[int, Tuple[float, float]],
) -> Tuple[float, float, float, float, float]:
    """Oblicza znormalizowany kierunek spojrzenia źrenic (gaze_x, gaze_y),
    współczynniki otwarcia oczu (ear_left, ear_right) oraz wagę przymrużenia (wink_alpha).

    Konwencja osi ekranu:
    - gaze_x > 0: spojrzenie w prawo ekranu (+X), gaze_x < 0: w lewo (-X)
    - gaze_y > 0: spojrzenie w górę (+Y), gaze_y < 0: w dół (-Y)
    - wink_alpha > 0: lewe oko przymknięte -> wybierz prawe oko (+1.0)
    - wink_alpha < 0: prawe oko przymknięte -> wybierz lewe oko (-1.0)
    """
    required_ids = (
        EYE_R_OUTER_ID,
        EYE_R_INNER_ID,
        EYE_R_UPPER_ID,
        EYE_R_LOWER_ID,
        EYE_L_INNER_ID,
        EYE_L_OUTER_ID,
        EYE_L_UPPER_ID,
        EYE_L_LOWER_ID,
        IRIS_RIGHT_ID,
        IRIS_LEFT_ID,
    )
    if not all(idx in face_pts_2d for idx in required_ids):
        return 0.0, 0.0, 0.3, 0.3, 0.0

    p_r_out = np.array(face_pts_2d[EYE_R_OUTER_ID], dtype=np.float64)
    p_r_in = np.array(face_pts_2d[EYE_R_INNER_ID], dtype=np.float64)
    p_r_up = np.array(face_pts_2d[EYE_R_UPPER_ID], dtype=np.float64)
    p_r_lo = np.array(face_pts_2d[EYE_R_LOWER_ID], dtype=np.float64)
    p_iris_r = np.array(face_pts_2d[IRIS_RIGHT_ID], dtype=np.float64)

    p_l_in = np.array(face_pts_2d[EYE_L_INNER_ID], dtype=np.float64)
    p_l_out = np.array(face_pts_2d[EYE_L_OUTER_ID], dtype=np.float64)
    p_l_up = np.array(face_pts_2d[EYE_L_UPPER_ID], dtype=np.float64)
    p_l_lo = np.array(face_pts_2d[EYE_L_LOWER_ID], dtype=np.float64)
    p_iris_l = np.array(face_pts_2d[IRIS_LEFT_ID], dtype=np.float64)

    # Poziomy wektor w stronę +X ekranu (od lewej strony użytkownika do prawej):
    # Prawe oko: od kącika wewnętrznego (133) do zewnętrznego (33)
    # Lewe oko: od kącika zewnętrznego (263) do wewnętrznego (362)
    vec_r_x = p_r_out - p_r_in
    vec_l_x = p_l_in - p_l_out

    norm_r_x_sq = float(np.dot(vec_r_x, vec_r_x))
    norm_l_x_sq = float(np.dot(vec_l_x, vec_l_x))

    ratio_r_x = float(np.dot(p_iris_r - p_r_in, vec_r_x) / (norm_r_x_sq + 1e-6))
    ratio_l_x = float(np.dot(p_iris_l - p_l_out, vec_l_x) / (norm_l_x_sq + 1e-6))

    gaze_r_x = float(np.clip((ratio_r_x - 0.5) * 4.0, -1.0, 1.0))
    gaze_l_x = float(np.clip((ratio_l_x - 0.5) * 4.0, -1.0, 1.0))
    gaze_x = 0.5 * (gaze_r_x + gaze_l_x)

    # Pionowy wektor w stronę +Y ekranu (od dolnej powieki do górnej)
    vec_r_y = p_r_up - p_r_lo
    vec_l_y = p_l_up - p_l_lo

    norm_r_y_sq = float(np.dot(vec_r_y, vec_r_y))
    norm_l_y_sq = float(np.dot(vec_l_y, vec_l_y))

    ratio_r_y = float(np.dot(p_iris_r - p_r_lo, vec_r_y) / (norm_r_y_sq + 1e-6))
    ratio_l_y = float(np.dot(p_iris_l - p_l_lo, vec_l_y) / (norm_l_y_sq + 1e-6))

    gaze_r_y = float(np.clip((ratio_r_y - 0.5) * 4.0, -1.0, 1.0))
    gaze_l_y = float(np.clip((ratio_l_y - 0.5) * 4.0, -1.0, 1.0))
    gaze_y = 0.5 * (gaze_r_y + gaze_l_y)

    # Współczynnik otwarcia powiek (Eye Aspect Ratio)
    width_r = float(np.sqrt(norm_r_x_sq))
    width_l = float(np.sqrt(norm_l_x_sq))
    height_r = float(np.sqrt(norm_r_y_sq))
    height_l = float(np.sqrt(norm_l_y_sq))

    ear_right = height_r / (width_r + 1e-6)
    ear_left = height_l / (width_l + 1e-6)

    # Detekcja asymetrycznego przymrużenia (mrugnięcie obu oczu ma ~0 asymetrii)
    squint_l = smoothstep(EAR_SQUINT_THRESHOLD, EAR_SQUINT_THRESHOLD * 0.55, ear_left) * smoothstep(
        EAR_ASYMMETRY_MIN, EAR_ASYMMETRY_MIN * 1.8, ear_right - ear_left
    )
    squint_r = smoothstep(EAR_SQUINT_THRESHOLD, EAR_SQUINT_THRESHOLD * 0.55, ear_right) * smoothstep(
        EAR_ASYMMETRY_MIN, EAR_ASYMMETRY_MIN * 1.8, ear_left - ear_right
    )
    wink_alpha = float(np.clip(squint_l - squint_r, -1.0, 1.0))

    return gaze_x, gaze_y, ear_left, ear_right, wink_alpha


def compute_hybrid_intent(
    head_center: np.ndarray,
    eye_l_screen: np.ndarray,
    eye_r_screen: np.ndarray,
    face_normal_screen: np.ndarray,
    x_neutral_mm: float = 0.0,
    y_neutral_mm: float = 0.0,
    gaze_x: float = 0.0,
    gaze_y: float = 0.0,
    wink_alpha: float = 0.0,
    vel_proj_mm_s: float = 0.0,
    enable_horizontal_intent: bool = True,
    enable_vertical_intent: bool = True,
    enable_gaze_intent: bool = True,
) -> Dict[str, Union[float, str]]:
    """Hybrydowy model intencji wyboru oka (360° Vector + True Gaze + Velocity Lead + EAR Wink)."""
    x_rel = float(head_center[0] - x_neutral_mm) if enable_horizontal_intent else 0.0
    y_rel = float(head_center[1] - y_neutral_mm) if enable_vertical_intent else 0.0

    eye_vec_xy = np.array(
        [eye_r_screen[0] - eye_l_screen[0], eye_r_screen[1] - eye_l_screen[1]],
        dtype=np.float64,
    )
    eye_norm_xy = float(np.linalg.norm(eye_vec_xy))
    if eye_norm_xy > 1e-6:
        u_eyes = eye_vec_xy / eye_norm_xy
    else:
        u_eyes = np.array([1.0, 0.0], dtype=np.float64)

    # Zunifikowany rzut wektora wychylenia na oś oczu (360°)
    s_eff = float(x_rel * u_eyes[0] + (VERTICAL_TILT_GAIN * y_rel) * u_eyes[1])
    abs_s = abs(s_eff)
    sign_s = float(np.sign(s_eff)) if abs_s > 1e-3 else 0.0

    # Warstwa 1: Krzywa bazowa 360°
    base_mag = float(smoothstep(PEEK_DEADZONE_MM, PEEK_FULL_DIST_MM, abs_s) ** PEEK_CURVE_EXP)
    vec_alpha = sign_s * base_mag

    # Warstwa 2 i 3: Fuzja spojrzenia (True Gaze + VOR) i wyprzedzenie prędkościowe
    conv_boost = 0.0
    direct_gaze_mag = 0.0
    look_away_gate = 1.0
    vel_term = 0.0

    if enable_gaze_intent:
        combined_gaze_x = float(face_normal_screen[0] + GAZE_EYE_WEIGHT * gaze_x)
        if abs_s > 2.0:
            opposing_gaze = -sign_s * combined_gaze_x
            if opposing_gaze > 0.0:
                conv_boost = (
                    GAZE_CONVERGENCE_BOOST
                    * float(np.clip(opposing_gaze / 0.30, 0.0, 1.0))
                    * smoothstep(2.0, PEEK_FULL_DIST_MM * 0.6, abs_s)
                )
                direct_gaze_mag = (
                    GAZE_DIRECT_PEEK_GAIN
                    * float(np.clip(opposing_gaze / 0.35, 0.0, 1.0))
                    * smoothstep(2.0, PEEK_DEADZONE_MM * 2.5, abs_s)
                )
            look_away_gate = 1.0 - smoothstep(
                LOOK_AWAY_THRESHOLD, LOOK_AWAY_THRESHOLD + 0.25, -opposing_gaze
            )

        vel_gate = smoothstep(PEEK_DEADZONE_MM * 0.5, PEEK_DEADZONE_MM * 1.5, abs_s)
        vel_term = float(np.clip(VELOCITY_LEAD_GAIN * vel_proj_mm_s, -0.35, 0.35)) * vel_gate

    combined_mag = float(np.clip(base_mag + conv_boost + direct_gaze_mag, 0.0, 1.0))
    target_alpha = float(np.clip((sign_s * combined_mag + vel_term) * look_away_gate, -1.0, 1.0))

    # Warstwa 4: Nadpisanie przez przymrużenie oka (EAR Wink Override)
    wink_active = enable_gaze_intent and abs(wink_alpha) > 0.05
    if wink_active:
        wink_strength = min(abs(wink_alpha), 1.0)
        target_alpha = float(np.clip((1.0 - wink_strength) * target_alpha + wink_alpha, -1.0, 1.0))

    # Czytelny opis aktywnego trybu
    if wink_active:
        eye_name = "PRAWE" if wink_alpha > 0 else "LEWE"
        mode_str = f"WINK -> {eye_name}"
    elif abs(target_alpha) > 0.02:
        eye_name = "PRAWE" if target_alpha > 0 else "LEWE"
        horiz_part = abs(x_rel * u_eyes[0])
        vert_part = abs((VERTICAL_TILT_GAIN * y_rel) * u_eyes[1])
        if vert_part > horiz_part * 1.25:
            tilt_type = "GORA" if y_rel > 0 else "DOL"
            mode_str = f"PION/ROLL ({tilt_type}) -> {eye_name}"
        elif enable_gaze_intent and (conv_boost + direct_gaze_mag) > 0.03:
            mode_str = f"WEKTOR+GAZE -> {eye_name}"
        else:
            mode_str = f"WEKTOR 360 -> {eye_name}"
    else:
        mode_str = "NEUTRALNY (SRODEK)"

    return {
        "target_alpha": target_alpha,
        "mode_str": mode_str,
        "s_eff": s_eff,
        "vec_alpha": vec_alpha,
        "gaze_boost": conv_boost + direct_gaze_mag,
        "x_rel": x_rel,
        "y_rel": y_rel,
    }


@dataclass
class PoseResult:
    """Wynik estymacji pozycji głowy i wirtualnej kamery w milimetrach."""

    is_valid: bool = False
    # Pozycja oka/kamery po wygładzeniu i biomechanice (mm): X: +prawo, Y: +góra, Z: +do użytkownika
    cam_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 500.0], dtype=np.float64))
    head_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    eye_left: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    eye_right: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    face_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Kąty obrotu głowy w stopniach
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    # Diagnostyka hybrydowego systemu intencji
    alpha_smooth: float = 0.0
    target_alpha: float = 0.0
    mode_str: str = "NEUTRALNY"
    x_rel: float = 0.0
    y_rel: float = 0.0
    s_eff: float = 0.0
    delta_eye_y: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    ear_left: float = 0.3
    ear_right: float = 0.3
    wink_alpha: float = 0.0

    # Punkty 2D na klatce wideo do wizualizacji
    landmarks_2d: List[Tuple[float, float]] = field(default_factory=list)
    iris_2d: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)

    # Wydajność przetwarzania
    fps: float = 0.0
    frame_time_ms: float = 0.0


class FaceTracker:
    """Moduł śledzenia twarzy (Head & Eye Tracking) bazujący na MediaPipe i solvePnP."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        model_path_str = str(model_path)
        if not Path(model_path_str).exists():
            raise FileNotFoundError(
                f"Nie znaleziono modelu MediaPipe pod ścieżką: {model_path_str}. "
                f"Upewnij się, że plik face_landmarker.task znajduje się w katalogu models/."
            )

        base_options = python.BaseOptions(model_asset_path=model_path_str)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Stan filtrów wygładzających i kalibracji
        self.head_smooth: Optional[np.ndarray] = None
        self.eye_base_smooth: Optional[np.ndarray] = None
        self.alpha_smooth: float = 0.0
        self.x_neutral_mm: float = 0.0
        self.y_neutral_mm: float = 0.0
        self.prev_time: Optional[float] = None
        self.last_timestamp_ms: int = 0
        self.prev_s_eff: Optional[float] = None
        self.vel_proj_smooth: float = 0.0
        self.last_head_center: Optional[np.ndarray] = None
        self.fps_smooth: float = 0.0
        self.frame_time_ms: float = 0.0

        # Flagi diagnostyczne warstw intencji
        self.enable_horizontal_intent: bool = True  # Oś pozioma X
        self.enable_vertical_intent: bool = True    # Oś pionowa Y + przechył Roll
        self.enable_gaze_intent: bool = True        # Gaze (źrenice) + Velocity Lead + EAR Wink

    def toggle_horizontal_intent(self) -> bool:
        """Przełącza udział wychylenia poziomego w modelu wektorowym."""
        self.enable_horizontal_intent = not self.enable_horizontal_intent
        state_str = "WŁĄCZONA" if self.enable_horizontal_intent else "WYŁĄCZONA"
        print(f"[*] Intencja pozioma (X): {state_str}")
        return self.enable_horizontal_intent

    def toggle_vertical_intent(self) -> bool:
        """Przełącza udział wychylenia pionowego z przechyłem Roll."""
        self.enable_vertical_intent = not self.enable_vertical_intent
        state_str = "WŁĄCZONA" if self.enable_vertical_intent else "WYŁĄCZONA"
        print(f"[*] Intencja pionowa (Y/Roll): {state_str}")
        return self.enable_vertical_intent

    def toggle_gaze_intent(self) -> bool:
        """Przełącza warstwę Gaze (źrenice), Velocity Lead i przymrużenia oka (EAR Wink)."""
        self.enable_gaze_intent = not self.enable_gaze_intent
        state_str = "WŁĄCZONA" if self.enable_gaze_intent else "WYŁĄCZONA"
        print(f"[*] Warstwa Gaze + Velocity + EAR Wink: {state_str}")
        return self.enable_gaze_intent

    def calibrate_neutral_y(self) -> None:
        """Ustawia aktualną wysokość głowy jako poziom neutralny."""
        if self.last_head_center is not None:
            self.y_neutral_mm = float(self.last_head_center[1])
            print(f"[*] Skalibrowano neutralne Y: {self.y_neutral_mm:.1f} mm")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> PoseResult:
        """Przetwarza pojedynczą klatkę wideo BGR i zwraca oszacowaną pozycję głowy."""
        t_start = time.perf_counter()
        if self.prev_time is None:
            dt = 0.033
        else:
            dt = float(np.clip(t_start - self.prev_time, 0.001, 0.2))
        self.prev_time = t_start

        inst_fps = 1.0 / dt
        if self.fps_smooth <= 0.0:
            self.fps_smooth = inst_fps
        else:
            self.fps_smooth += (inst_fps - self.fps_smooth) * 0.15

        if timestamp_ms is None:
            timestamp_ms = int(t_start * 1000)
        timestamp_ms = max(int(timestamp_ms), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        h, w, _ = frame.shape
        if w > MP_DETECT_WIDTH or h > MP_DETECT_HEIGHT:
            detect_frame = cv2.resize(
                frame,
                (MP_DETECT_WIDTH, MP_DETECT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            detect_frame = frame

        rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            self.frame_time_ms = (time.perf_counter() - t_start) * 1000.0
            return PoseResult(
                is_valid=False,
                fps=self.fps_smooth,
                frame_time_ms=self.frame_time_ms,
            )

        face = result.face_landmarks[0]
        pnp_points_2d = []
        vis_points_2d = []
        for idx in LANDMARK_IDS:
            pt = face[idx]
            pnp_points_2d.append([pt.x * CAMERA_WIDTH, pt.y * CAMERA_HEIGHT])
            vis_points_2d.append((pt.x * w, pt.y * h))

        eye_landmark_ids = (
            EYE_R_OUTER_ID,
            EYE_R_INNER_ID,
            EYE_R_UPPER_ID,
            EYE_R_LOWER_ID,
            EYE_L_INNER_ID,
            EYE_L_OUTER_ID,
            EYE_L_UPPER_ID,
            EYE_L_LOWER_ID,
            IRIS_RIGHT_ID,
            IRIS_LEFT_ID,
        )
        face_pts_2d = {
            idx: (face[idx].x * w, face[idx].y * h)
            for idx in eye_landmark_ids
            if idx < len(face)
        }

        iris_points = []
        for iris_id in [IRIS_RIGHT_ID, IRIS_LEFT_ID]:
            if iris_id in face_pts_2d:
                iris_points.append((iris_id, face_pts_2d[iris_id]))

        image_points = np.array(pnp_points_2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            MODEL_3D,
            image_points,
            CAMERA_INTRINSIC_K,
            CAMERA_DISTORTION_D,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            self.frame_time_ms = (time.perf_counter() - t_start) * 1000.0
            return PoseResult(
                is_valid=False,
                fps=self.fps_smooth,
                frame_time_ms=self.frame_time_ms,
            )

        R, _ = cv2.Rodrigues(rvec)
        t_head = tvec.ravel()

        eye_r_3d = R @ EYE_RIGHT_HEAD + t_head
        eye_l_3d = R @ EYE_LEFT_HEAD  + t_head

        # Przekształcenie do układu współrzędnych ekranu:
        # X: +prawo, Y: +góra od środka ekranu, Z: +do użytkownika
        eye_r_screen = np.array([
            -eye_r_3d[0],
            CAMERA_TO_SCREEN_CENTER_Y_MM - eye_r_3d[1],
            eye_r_3d[2],
        ])
        eye_l_screen = np.array([
            -eye_l_3d[0],
            CAMERA_TO_SCREEN_CENTER_Y_MM - eye_l_3d[1],
            eye_l_3d[2],
        ])

        head_center = (eye_l_screen + eye_r_screen) * 0.5
        self.last_head_center = head_center.copy()
        eye_base_3d = eye_r_screen - eye_l_screen

        cam_forward = -R[:, 2]
        face_normal_screen = np.array([-cam_forward[0], -cam_forward[1], cam_forward[2]])

        delta_eye_y = float(eye_r_screen[1] - eye_l_screen[1])

        # --- 1. EKSTRAKCJA GAZE (ŹRENICE) I EAR (OTWARCIE POWIEK) ---
        gaze_x, gaze_y, ear_left, ear_right, wink_alpha = compute_eye_gaze_and_ear(face_pts_2d)

        # --- 2. ESTYMACJA PRĘDKOŚCI WZDŁUŻ OSI OCZU (VELOCITY LEAD) ---
        eye_vec_xy = np.array(
            [eye_r_screen[0] - eye_l_screen[0], eye_r_screen[1] - eye_l_screen[1]],
            dtype=np.float64,
        )
        eye_norm_xy = float(np.linalg.norm(eye_vec_xy))
        u_eyes = eye_vec_xy / eye_norm_xy if eye_norm_xy > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
        x_rel_raw = float(head_center[0] - self.x_neutral_mm) if self.enable_horizontal_intent else 0.0
        y_rel_raw = float(head_center[1] - self.y_neutral_mm) if self.enable_vertical_intent else 0.0
        s_eff_curr = float(x_rel_raw * u_eyes[0] + (VERTICAL_TILT_GAIN * y_rel_raw) * u_eyes[1])

        if self.prev_s_eff is not None:
            raw_vel = (s_eff_curr - self.prev_s_eff) / dt
            vel_blend = 1.0 - np.exp(-VELOCITY_SMOOTH_RATE * dt)
            self.vel_proj_smooth += (raw_vel - self.vel_proj_smooth) * vel_blend
        self.prev_s_eff = s_eff_curr

        # --- 3. HYBRYDOWY SYSTEM INTENCJI ---
        intent = compute_hybrid_intent(
            head_center=head_center,
            eye_l_screen=eye_l_screen,
            eye_r_screen=eye_r_screen,
            face_normal_screen=face_normal_screen,
            x_neutral_mm=self.x_neutral_mm,
            y_neutral_mm=self.y_neutral_mm,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            wink_alpha=wink_alpha,
            vel_proj_mm_s=self.vel_proj_smooth,
            enable_horizontal_intent=self.enable_horizontal_intent,
            enable_vertical_intent=self.enable_vertical_intent,
            enable_gaze_intent=self.enable_gaze_intent,
        )
        target_alpha = float(intent["target_alpha"])
        mode_str = str(intent["mode_str"])
        s_eff = float(intent["s_eff"])
        x_rel = float(intent["x_rel"])
        y_rel = float(intent["y_rel"])

        # --- 4. ASYMETRYCZNA FILTRACJA CZASOWA WAGI KAMERY (BEZ OPÓŹNIEŃ W RUCHU) ---
        if (
            abs(target_alpha) > abs(self.alpha_smooth)
            or target_alpha * self.alpha_smooth < 0.0
            or abs(self.vel_proj_smooth) > 18.0
        ):
            rate = ALPHA_ATTACK_RATE
        elif abs(s_eff) < PEEK_DEADZONE_MM and (not self.enable_gaze_intent or abs(wink_alpha) < 0.05):
            rate = ALPHA_CENTER_RATE
        else:
            rate = ALPHA_RELEASE_RATE

        alpha_blend = 1.0 - np.exp(-rate * dt)
        self.alpha_smooth += (target_alpha - self.alpha_smooth) * alpha_blend

        # --- 5. ADAPTACYJNA FILTRACJA POZYCJI 3D (BRAK PODWÓJNEGO FILTROWANIA) ---
        if self.head_smooth is None or self.eye_base_smooth is None:
            self.head_smooth = head_center.copy()
            self.eye_base_smooth = eye_base_3d.copy()
        else:
            head_speed = float(np.linalg.norm(head_center - self.head_smooth) / dt)
            speed_factor = smoothstep(20.0, 140.0, head_speed)
            pos_rate = POS_SMOOTH_RATE + (POS_FAST_RATE - POS_SMOOTH_RATE) * speed_factor
            pos_blend = 1.0 - np.exp(-pos_rate * dt)
            self.head_smooth += (head_center - self.head_smooth) * pos_blend
            self.eye_base_smooth += (eye_base_3d - self.eye_base_smooth) * pos_blend

        cam_pos = self.head_smooth + self.alpha_smooth * (self.eye_base_smooth * 0.5)

        # Obliczenie kątów głowy
        delta_eye_x = eye_r_screen[0] - eye_l_screen[0]
        roll_deg = float(np.degrees(np.arctan2(delta_eye_y, delta_eye_x)))
        yaw_deg = float(np.degrees(np.arctan2(face_normal_screen[0], face_normal_screen[2])))
        pitch_deg = float(np.degrees(np.arctan2(face_normal_screen[1], face_normal_screen[2])))

        self.frame_time_ms = (time.perf_counter() - t_start) * 1000.0

        return PoseResult(
            is_valid=True,
            cam_pos=cam_pos,
            head_center=head_center,
            eye_left=eye_l_screen,
            eye_right=eye_r_screen,
            face_normal=face_normal_screen,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            alpha_smooth=self.alpha_smooth,
            target_alpha=target_alpha,
            mode_str=mode_str,
            x_rel=x_rel,
            y_rel=y_rel,
            s_eff=s_eff,
            delta_eye_y=delta_eye_y,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            ear_left=ear_left,
            ear_right=ear_right,
            wink_alpha=wink_alpha,
            landmarks_2d=vis_points_2d,
            iris_2d=iris_points,
            fps=self.fps_smooth,
            frame_time_ms=self.frame_time_ms,
        )

    def close(self) -> None:
        """Zwalnia zasoby detektora MediaPipe."""
        try:
            self.detector.close()
        except Exception:
            pass
