import argparse
from pathlib import Path
import sys

# Upewnij się, że katalog główny projektu jest w sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.camera import init_camera
from src.hud import draw_hud
from src.protocol import HeadTrackerUDPClient
from src.tracker import FaceTracker


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostyczny wizualizator śledzenia głowy (HUD + estymacja 3D + biomechanika)."
    )
    parser.add_argument("--udp", action="store_true", help="Nadawaj współrzędne 3D przez UDP (127.0.0.1:4242)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host docelowy UDP (domyślnie 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4242, help="Port UDP (domyślnie 4242)")
    args = parser.parse_args()

    print("=== Inicjalizacja Trackera Twarzy (Tryb Diagnostyczny HUD) ===")
    cap = init_camera()
    tracker = FaceTracker()

    udp_client = None
    udp_info_str = None
    if args.udp:
        udp_client = HeadTrackerUDPClient(host=args.host, port=args.port)
        udp_info_str = f"{args.host}:{args.port}"
        print(f"[+] Nadawanie UDP aktywne na {udp_info_str}")

    print("\n[Sterowanie]")
    print(" 'c' - kalibracja neutralnej wysokości oczu")
    print(" 'q' / ESC - wyjście z programu\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose = tracker.process_frame(frame)

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

            draw_hud(frame, pose, udp_info=udp_info_str)

            cv2.imshow("Tracking 3D - Peeking Biomechanics (HUD)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
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
