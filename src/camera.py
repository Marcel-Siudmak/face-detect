import threading
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import CAMERA_HEIGHT, CAMERA_WIDTH


class ThreadedCamera:
    """Bezlagowy chwytacz klatek w osobnym wątku, eliminujący kolejkowanie w buforze AVFoundation."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._cond = threading.Condition()
        self._running = True
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._last_returned_seq = 0
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while self._running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                with self._cond:
                    self._running = False
                    self._cond.notify_all()
                break
            with self._cond:
                self._frame = frame
                self._frame_seq += 1
                self._cond.notify_all()

    def isOpened(self) -> bool:
        return self._running and self._cap.isOpened()

    def read(self, timeout: float = 0.05) -> Tuple[bool, Optional[np.ndarray]]:
        with self._cond:
            if self._frame_seq == self._last_returned_seq and self._running:
                self._cond.wait(timeout=timeout)
            if self._frame is None:
                return False, None
            self._last_returned_seq = self._frame_seq
            return True, self._frame.copy()

    def release(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._cap.release()


def init_camera(
    indices: Sequence[int] = (0, 1),
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> ThreadedCamera:
    """Inicjalizuje pierwszą dostępną kamerę z odczytem w wątku (optymalizacja pod macOS AVFoundation)."""
    for cam_idx in indices:
        # Najpierw próba z backendem AVFoundation dla macOS
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 60)

            ret, _ = cap.read()
            if ret:
                print(f"[+] Kamera zainicjalizowana pomyślnie (indeks {cam_idx}, wątek bezlagowy)")
                return ThreadedCamera(cap)
            cap.release()

    raise RuntimeError(
        "Nie znaleziono działającej kamery! Sprawdź uprawnienia do kamery w Ustawieniach systemowych macOS."
    )
