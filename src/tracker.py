from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import (
    INTENT_POS_X_EXP,
    INTENT_POS_Y_EXP,
    ALPHA_SMOOTH_RATE,
    CAMERA_DISTORTION_D,
    CAMERA_INTRINSIC_K,
    CAMERA_TO_SCREEN_CENTER_Y_MM,
    DEFAULT_MODEL_PATH,
    EYE_LEFT_HEAD,
    EYE_RIGHT_HEAD,
    IRIS_LEFT_ID,
    IRIS_RIGHT_ID,
    LANDMARK_IDS,
    MAX_ANGLE_SIN,
    MAX_DIST_X_MM,
    MAX_DIST_Y_MM,
    MODEL_3D,
    POS_SMOOTH_RATE,
    TILT_DEADZONE_MM,
    TILT_MAX_MM,
)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Nieliniowa funkcja sigmoidalna o zerowych pochodnych na krawędziach."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


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

    # Diagnostyka biomechaniczna intencji (Peeking)
    alpha_smooth: float = 0.0
    target_alpha: float = 0.0
    mode_str: str = "NEUTRALNY"
    y_rel: float = 0.0
    delta_eye_y: float = 0.0

    # Punkty 2D na klatce wideo do wizualizacji
    landmarks_2d: List[Tuple[float, float]] = field(default_factory=list)
    iris_2d: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)


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
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Stan filtrów wygładzających
        self.cam_smooth: Optional[np.ndarray] = None
        self.alpha_smooth: float = 0.0
        self.y_neutral_mm: float = 0.0  # Stały punkt odniesienia (środek)
        self.prev_time: Optional[float] = None
        self.last_head_center: Optional[np.ndarray] = None

        # Flagi diagnostyczne intencji
        self.enable_horizontal_intent: bool = True  # Pkt 4 i 5 (Zaułek X/Yaw)
        self.enable_vertical_intent: bool = True    # Pkt 6 (Murek/Daszek Y/Roll)

    def toggle_horizontal_intent(self) -> bool:
        """Przełącza intencję poziomą (zaułek: translacja X + rotacja Yaw)."""
        self.enable_horizontal_intent = not self.enable_horizontal_intent
        state_str = "WŁĄCZONA" if self.enable_horizontal_intent else "WYŁĄCZONA"
        print(f"[*] Intencja pozioma (Pkt 4 i 5): {state_str}")
        return self.enable_horizontal_intent

    def toggle_vertical_intent(self) -> bool:
        """Przełącza intencję pionową (przechył Z Roll + translacja Y)."""
        self.enable_vertical_intent = not self.enable_vertical_intent
        state_str = "WŁĄCZONA" if self.enable_vertical_intent else "WYŁĄCZONA"
        print(f"[*] Intencja pionowa (Pkt 6): {state_str}")
        return self.enable_vertical_intent

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
        curr_time = time.time()
        if self.prev_time is None:
            dt = 0.033
        else:
            dt = float(np.clip(curr_time - self.prev_time, 0.001, 0.1))
        self.prev_time = curr_time

        if timestamp_ms is None:
            timestamp_ms = int(curr_time * 1000)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return PoseResult(is_valid=False)

        face = result.face_landmarks[0]
        points_2d = []
        for idx in LANDMARK_IDS:
            pt = face[idx]
            points_2d.append([pt.x * w, pt.y * h])

        iris_points = []
        for iris_id in [IRIS_RIGHT_ID, IRIS_LEFT_ID]:
            if iris_id < len(face):
                iris_pt = face[iris_id]
                iris_points.append((iris_id, (iris_pt.x * w, iris_pt.y * h)))

        image_points = np.array(points_2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            MODEL_3D,
            image_points,
            CAMERA_INTRINSIC_K,
            CAMERA_DISTORTION_D,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            return PoseResult(is_valid=False)

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

        delta_eye_y = eye_r_screen[1] - eye_l_screen[1]

        # --- 1. INTENCJA POZIOMA (Pkt 4 i 5: Zaułek X/Yaw) ---
        target_alpha_x = 0.0
        intent_x_mag = 0.0
        if self.enable_horizontal_intent:
            abs_x = abs(head_center[0])
            # Nieliniowa translacja (progresywna potęga)
            t_pos_x = float(np.clip(abs_x / MAX_DIST_X_MM, 0.0, 1.0))
            factor_pos_x = t_pos_x ** INTENT_POS_X_EXP

            sign_pos_x = float(np.sign(head_center[0])) if abs_x > 1e-3 else 0.0
            # Przeciwny zwrot: ruch w prawo (+) wymaga rotacji w lewo (face_normal < 0)
            opposing_rot_x = -sign_pos_x * face_normal_screen[0]

            if opposing_rot_x > 0.0:
                # Liniowa zależność od rotacji
                factor_rot_x = float(np.clip(opposing_rot_x / MAX_ANGLE_SIN, 0.0, 1.0))
                intent_x_mag = factor_pos_x * factor_rot_x
                # Ruch w prawo (+) -> prawe oko (+), ruch w lewo (-) -> lewe oko (-)
                target_alpha_x = sign_pos_x * intent_x_mag

        # --- 2. INTENCJA PIONOWA Z PRZECHYŁEM ROLL (Pkt 6: Murek / Daszek Y/Roll) ---
        target_alpha_y = 0.0
        intent_y_mag = 0.0
        if self.enable_vertical_intent:
            y_val = head_center[1]  # punkt odniesienia sztywno na 0.0
            abs_y = abs(y_val)
            # Nieliniowa translacja pionowa (taka sama charakterystyka jak X)
            t_pos_y = float(np.clip(abs_y / MAX_DIST_Y_MM, 0.0, 1.0))
            factor_pos_y = t_pos_y ** INTENT_POS_Y_EXP

            # Liniowy udział przechyłu w osi Z (różnica wysokości oczu)
            tilt_factor = float(np.clip(abs(delta_eye_y) / TILT_MAX_MM, 0.0, 1.0))

            if tilt_factor > 0.02 and abs_y > 1e-3:
                sign_y = float(np.sign(y_val))
                sign_delta = float(np.sign(delta_eye_y))
                # Znak intencji:
                # Dół (Y < 0) + Prawe niżej (delta < 0) -> Prawe oko (+)
                # Dół (Y < 0) + Lewe niżej (delta > 0) -> Lewe oko (-)
                # Góra (Y > 0) + Prawe wyżej (delta > 0) -> Prawe oko (+)
                # Góra (Y > 0) + Lewe wyżej (delta < 0) -> Lewe oko (-)
                intent_y_sign = sign_y * sign_delta
                intent_y_mag = factor_pos_y * tilt_factor
                target_alpha_y = intent_y_sign * intent_y_mag

        # --- 3. WYBÓR INTENCJI DOMINUJĄCEJ ---
        if intent_x_mag >= intent_y_mag and intent_x_mag > 0.02:
            target_alpha = target_alpha_x
            eye_name = "PRAWE" if target_alpha_x > 0 else "LEWE"
            mode_str = f"ZAULEK -> {eye_name}"
        elif intent_y_mag > intent_x_mag and intent_y_mag > 0.02:
            target_alpha = target_alpha_y
            eye_name = "PRAWE" if target_alpha_y > 0 else "LEWE"
            tilt_type = "GORA/WYZEJ" if head_center[1] > 0 else "DOL/NIZSZE"
            mode_str = f"PION ({tilt_type}) -> {eye_name}"
        else:
            target_alpha = 0.0
            mode_str = "NEUTRALNY (SRODEK)"

        # --- 4. FILTRACJA CZASOWA: PŁYWANIE WAGI KAMERY ---
        alpha_blend = 1.0 - np.exp(-ALPHA_SMOOTH_RATE * dt)
        self.alpha_smooth += (target_alpha - self.alpha_smooth) * alpha_blend

        raw_cam_screen = head_center + self.alpha_smooth * (eye_base_3d * 0.5)

        # --- 5. FILTRACJA CZASOWA: TŁUMIENIE DRGAŃ POZYCJI 3D ---
        if self.cam_smooth is None:
            self.cam_smooth = raw_cam_screen.copy()
        else:
            pos_blend = 1.0 - np.exp(-POS_SMOOTH_RATE * dt)
            self.cam_smooth += (raw_cam_screen - self.cam_smooth) * pos_blend

        # Obliczenie kątów głowy
        delta_eye_x = eye_r_screen[0] - eye_l_screen[0]
        roll_deg = float(np.degrees(np.arctan2(delta_eye_y, delta_eye_x)))
        yaw_deg = float(np.degrees(np.arctan2(face_normal_screen[0], face_normal_screen[2])))
        pitch_deg = float(np.degrees(np.arctan2(face_normal_screen[1], face_normal_screen[2])))

        return PoseResult(
            is_valid=True,
            cam_pos=self.cam_smooth.copy(),
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
            y_rel=y_val,
            delta_eye_y=delta_eye_y,
            landmarks_2d=[(p[0], p[1]) for p in points_2d],
            iris_2d=iris_points,
        )

    def close(self) -> None:
        """Zwalnia zasoby detektora MediaPipe."""
        try:
            self.detector.close()
        except Exception:
            pass
