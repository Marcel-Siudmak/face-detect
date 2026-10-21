from typing import Sequence
import cv2
from src.config import CAMERA_WIDTH, CAMERA_HEIGHT


def init_camera(
    indices: Sequence[int] = (0, 1),
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> cv2.VideoCapture:
    """Inicjalizuje pierwszą dostępną kamerę (optymalizowane pod macOS AVFoundation)."""
    for cam_idx in indices:
        # Najpierw próba z backendem AVFoundation dla macOS
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                print(f"[+] Kamera zainicjalizowana pomyślnie (indeks {cam_idx})")
                return cap
            cap.release()

    raise RuntimeError(
        "Nie znaleziono działającej kamery! Sprawdź uprawnienia do kamery w Ustawieniach systemowych macOS."
    )
