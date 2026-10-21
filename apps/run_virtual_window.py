import argparse
from pathlib import Path
import sys

# Upewnij się, że katalog główny projektu jest w sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.camera import init_camera
from src.config import WIN_H, WIN_W
from src.protocol import HeadTrackerUDPClient
from src.renderer_offaxis import VirtualWindowRenderer
from src.tracker import FaceTracker


def main():
    parser = argparse.ArgumentParser(
        description="Wirtualne Okno 3D (Fish-tank VR) z rzutowaniem off-axis i śledzeniem głowy."
    )
    parser.add_argument("--udp", action="store_true", help="Włącz równoczesne nadawanie pozy przez UDP (127.0.0.1:4242)")
    parser.add_argument("--port", type=int, default=4242, help="Port UDP (domyślnie 4242)")
    parser.add_argument("--no-pip", action="store_true", help="Ukryj podgląd kamery PiP w rogu okna")
    args = parser.parse_args()

    print("=== Inicjalizacja Wirtualnego Okna 3D ===")
    cap = init_camera()
    tracker = FaceTracker()
    renderer = VirtualWindowRenderer(win_w=WIN_W, win_h=WIN_H)

    udp_client = None
    if args.udp:
        udp_client = HeadTrackerUDPClient(port=args.port)
        print(f"[+] Nadawanie UDP aktywne na 127.0.0.1:{args.port}")

    cv2.namedWindow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", WIN_W, WIN_H)

    default_eye = np.array([0.0, 0.0, 500.0], dtype=np.float64)

    print("\n[Sterowanie]")
    print(" 'c' - kalibracja neutralnej wysokości oczu")
    print(" 'q' / ESC - wyjście z programu\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose = tracker.process_frame(frame)

            eye_pos = pose.cam_pos if pose.is_valid else default_eye

            if udp_client is not None:
                if pose.is_valid:
                    udp_client.send_pose(
                        x_mm=float(pose.cam_pos[0]),
                        y_mm=float(pose.cam_pos[1]),
                        z_mm=float(pose.cam_pos[2]),
                        yaw=pose.yaw_deg,
                        pitch=pose.pitch_deg,
                        roll=pose.roll_deg,
                        is_valid=True,
                    )
                else:
                    udp_client.send_pose(is_valid=False)

            pip_img = None if args.no_pip else frame
            canvas = renderer.render(eye_pos=eye_pos, camera_frame=pip_img)

            cv2.imshow("Virtual Window - Mierzalna Scena 3D (Kostki Rubika)", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' lub ESC
                break
            elif key == ord('c'):
                tracker.calibrate_neutral_y()
    finally:
        if udp_client is not None:
            udp_client.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
