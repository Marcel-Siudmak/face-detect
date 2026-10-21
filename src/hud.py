from typing import Optional
import cv2
import numpy as np

from src.config import SCREEN_HEIGHT_MM, SCREEN_WIDTH_MM
from src.tracker import PoseResult


def draw_hud(
    img: np.ndarray,
    pose: PoseResult,
    udp_info: Optional[str] = None,
) -> None:
    """Rysuje panel diagnostyczny HUD, widoki rzutu (radar X-Z, widok X-Y) oraz punkty twarzy."""
    h, w, _ = img.shape

    # Rysuj punkty 2D na twarzy
    for pt in pose.landmarks_2d:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    for iris_id, iris_pt in pose.iris_2d:
        col = (255, 0, 0) if iris_id == 468 else (0, 255, 255)
        cv2.circle(img, (int(iris_pt[0]), int(iris_pt[1])), 4, col, -1)


     # Wizualizacja wirtualnej kamery na odcinku między źrenicami w zależności od intencji
    iris_dict = dict(pose.iris_2d)
    if 468 in iris_dict and 473 in iris_dict:
        pt_r = np.array(iris_dict[468], dtype=np.float64)  # Prawe oko (anatomicznie)
        pt_l = np.array(iris_dict[473], dtype=np.float64)  # Lewe oko (anatomicznie)

        # 1. Odcinek referencyjny łączący obie źrenice
        p_l_int = (int(round(pt_l[0])), int(round(pt_l[1])))
        p_r_int = (int(round(pt_r[0])), int(round(pt_r[1])))
        cv2.line(img, p_l_int, p_r_int, (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Pozycja kamery zależna od intencji (alpha_smooth w zakresie od -1.0 do +1.0)
        # alpha = -1.0 -> lewe oko, alpha = 0.0 -> środek, alpha = +1.0 -> prawe oko
        center = (pt_l + pt_r) * 0.5
        half_vec = (pt_r - pt_l) * 0.5
        cam_2d = center + pose.alpha_smooth * half_vec

        cam_x, cam_y = int(round(cam_2d[0])), int(round(cam_2d[1]))

        # 3. Punkt kamery (pomarańczowy okrąg z obwódką)
        cv2.circle(img, (cam_x, cam_y), 6, (0, 140, 255), -1)
        cv2.circle(img, (cam_x, cam_y), 8, (0, 220, 255), 2, cv2.LINE_AA)

    if not pose.is_valid:
        cv2.putText(
            img,
            "BRAK TWARZY W KADRZE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    # Panel boczny
    panel_w, panel_h = 380, 680
    px, py = w - panel_w - 20, 20

    sub = img[py:py + panel_h, px:px + panel_w]
    dark_rect = np.full_like(sub, 20)
    img[py:py + panel_h, px:px + panel_w] = cv2.addWeighted(sub, 0.25, dark_rect, 0.75, 0)
    cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)


    t_diag = f"Y_rel: {pose.y_rel:+4.0f}mm | Tilt: {pose.delta_eye_y:+3.0f}mm | 'c'=kalibracja"
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

    p_el_xz = to_radar_xz(pose.eye_left)
    p_er_xz = to_radar_xz(pose.eye_right)
    p_cam_xz = to_radar_xz(pose.cam_pos)

    cv2.line(img, p_el_xz, p_er_xz, (120, 120, 120), 2)
    cv2.circle(img, p_el_xz, 5, (255, 200, 0), -1)
    cv2.circle(img, p_er_xz, 5, (0, 255, 0), -1)

    p_head_xz = ((p_el_xz[0] + p_er_xz[0]) // 2, (p_el_xz[1] + p_er_xz[1]) // 2)
    cv2.arrowedLine(
        img,
        p_head_xz,
        (int(p_head_xz[0] + pose.face_normal[0] * 35), int(p_head_xz[1] - pose.face_normal[2] * 35)),
        (0, 255, 255),
        2,
        tipLength=0.25,
    )

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
    cv2.rectangle(
        img,
        (front_cx - rect_w // 2, front_cy - rect_h // 2),
        (front_cx + rect_w // 2, front_cy + rect_h // 2),
        (140, 140, 140),
        1,
    )

    def to_front_xy(pt):
        return (int(front_cx + pt[0] * scale_xy), int(front_cy - pt[1] * scale_xy))

    p_el_xy = to_front_xy(pose.eye_left)
    p_er_xy = to_front_xy(pose.eye_right)
    p_cam_xy = to_front_xy(pose.cam_pos)

    cv2.line(img, p_el_xy, p_er_xy, (120, 120, 120), 2)
    cv2.circle(img, p_el_xy, 5, (255, 200, 0), -1)
    cv2.circle(img, p_er_xy, 5, (0, 255, 0), -1)
    cv2.circle(img, p_cam_xy, 7, (0, 140, 255), -1)
    cv2.circle(img, p_cam_xy, 9, (0, 255, 255), 2)

    # Teksty diagnostyczne po lewej
    t_r   = f"Prawe: X:{pose.eye_right[0]:4.0f} Y:{pose.eye_right[1]:4.0f} Z:{pose.eye_right[2]:4.0f}"
    t_l   = f"Lewe : X:{pose.eye_left[0]:4.0f} Y:{pose.eye_left[1]:4.0f} Z:{pose.eye_left[2]:4.0f}"
    t_cam = f"KAMERA: X:{pose.cam_pos[0]:4.0f} Y:{pose.cam_pos[1]:4.0f} Z:{pose.cam_pos[2]:4.0f} (a:{pose.alpha_smooth:+0.2f})"
    cv2.putText(img, t_r,   (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, t_l,   (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(img, t_cam, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    if udp_info:
        cv2.putText(
            img,
            f"UDP: {udp_info}",
            (30, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
