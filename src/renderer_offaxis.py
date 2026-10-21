from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    CUBE_CENTER_POS,
    CUBE_LEFT_POS,
    CUBE_RIGHT_POS,
    HH,
    HW,
    ROOM_BACK_Z,
    RUBIK_SIZE,
    SCREEN_HEIGHT_MM,
    SCREEN_WIDTH_MM,
    TABLE_DEPTH_MM,
    TABLE_Y,
    WIN_H,
    WIN_W,
)


def build_rubik_mesh(center_pos: np.ndarray, size: float = RUBIK_SIZE) -> List[dict]:
    """Tworzy geometrię kostki Rubika z podziałem 3x3 naklejek na każdej ściance.

    Zwraca listę 6 ścian bazowych, z których każda zawiera 9 naklejek (sub-faces).
    """
    cx, cy, cz = center_pos
    hs = size / 2.0

    # Definicje 6 głównych ścian: (baza, wektor_u, wektor_v, normalna, kolor_naklejek)
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
        base_quad = np.array([
            origin,
            origin + u_vec,
            origin + u_vec + v_vec,
            origin + v_vec,
        ])
        center_face = origin + 0.5 * u_vec + 0.5 * v_vec

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
            'sticker_col': sticker_col,
        })

    return elements


def project_off_axis(
    points_3d: np.ndarray,
    eye_pos: np.ndarray,
    win_w: int = WIN_W,
    win_h: int = WIN_H,
    screen_w_mm: float = SCREEN_WIDTH_MM,
    screen_h_mm: float = SCREEN_HEIGHT_MM,
) -> np.ndarray:
    """Rzutowanie perspektywiczne off-axis (przecięcie promienia oko -> punkt ze szklaną szybą Z = 0)."""
    xe, ye, ze = eye_pos
    ze = max(float(ze), 80.0)

    denominator = (ze - points_3d[:, 2])
    denominator = np.where(np.abs(denominator) < 1e-4, 1e-4, denominator)
    t = ze / denominator

    x_scr = xe + t * (points_3d[:, 0] - xe)
    y_scr = ye + t * (points_3d[:, 1] - ye)

    hw = screen_w_mm / 2.0
    hh = screen_h_mm / 2.0

    u = ((x_scr + hw) / screen_w_mm) * win_w
    v = ((hh - y_scr) / screen_h_mm) * win_h

    return np.column_stack((u, v)).astype(np.int32)


class VirtualWindowRenderer:
    """Silnik rzutowania off-axis symulujący wirtualne okno 3D ze sceną kostek Rubika."""

    def __init__(
        self,
        win_w: int = WIN_W,
        win_h: int = WIN_H,
        screen_w_mm: float = SCREEN_WIDTH_MM,
        screen_h_mm: float = SCREEN_HEIGHT_MM,
    ):
        self.win_w = win_w
        self.win_h = win_h
        self.screen_w_mm = screen_w_mm
        self.screen_h_mm = screen_h_mm
        self.hw = screen_w_mm / 2.0
        self.hh = screen_h_mm / 2.0

        # Siatki 3 kostek Rubika
        self.center_cube_mesh = build_rubik_mesh(CUBE_CENTER_POS)
        self.left_cube_mesh   = build_rubik_mesh(CUBE_LEFT_POS)
        self.right_cube_mesh  = build_rubik_mesh(CUBE_RIGHT_POS)

    def project(self, points_3d: np.ndarray, eye_pos: np.ndarray) -> np.ndarray:
        """Rzutuje punkty 3D na piksele okna w oparciu o pozycję oka."""
        return project_off_axis(
            points_3d=points_3d,
            eye_pos=eye_pos,
            win_w=self.win_w,
            win_h=self.win_h,
            screen_w_mm=self.screen_w_mm,
            screen_h_mm=self.screen_h_mm,
        )

    def render(
        self,
        eye_pos: np.ndarray,
        camera_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Renderuje pełną klatkę wirtualnego okna ze sceną 3D, stołem i opcjonalnym PiP."""
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        canvas[:] = (35, 38, 44)  # Ciemnografitowe tło pokoju

        # 1. Tylna ściana pokoju
        room_w = 400.0
        wall_pts = np.array([
            [-room_w, TABLE_Y, ROOM_BACK_Z],
            [ room_w, TABLE_Y, ROOM_BACK_Z],
            [ room_w, TABLE_Y + 300.0, ROOM_BACK_Z],
            [-room_w, TABLE_Y + 300.0, ROOM_BACK_Z],
        ])
        pts_wall = self.project(wall_pts, eye_pos)
        cv2.fillPoly(canvas, [pts_wall], (28, 30, 35))

        # 2. Siatka stołu
        # Linie poprzeczne (wzdłuż osi X)
        for z_grid in np.arange(0.0, -TABLE_DEPTH_MM - 1.0, -50.0):
            g_line = np.array([[-300.0, TABLE_Y, z_grid], [300.0, TABLE_Y, z_grid]])
            pts_gl = self.project(g_line, eye_pos)
            col_line = (120, 130, 145) if abs(z_grid) < 1e-3 else (50, 54, 62)
            thick = 2 if abs(z_grid) < 1e-3 else 1
            cv2.line(canvas, tuple(pts_gl[0]), tuple(pts_gl[1]), col_line, thick, cv2.LINE_AA)

        # Linie wzdłuż osi Z
        for x_grid in np.arange(-250.0, 251.0, 50.0):
            g_line = np.array([[x_grid, TABLE_Y, 0.0], [x_grid, TABLE_Y, ROOM_BACK_Z]])
            pts_gl = self.project(g_line, eye_pos)
            col_line = (90, 100, 115) if abs(x_grid) < 1e-3 else (64, 68, 76)
            thick = 2 if abs(x_grid) < 1e-3 else 1
            cv2.line(canvas, tuple(pts_gl[0]), tuple(pts_gl[1]), col_line, thick, cv2.LINE_AA)

        # 3. Zbieranie wszystkich ścian kostek i sortowanie algorytmem malarza
        render_polys = []
        all_cubes = [
            ("Centralna", self.center_cube_mesh),
            ("Lewa", self.left_cube_mesh),
            ("Prawa", self.right_cube_mesh),
        ]

        left_visible = False
        right_visible = False

        for cube_name, cube_mesh in all_cubes:
            for face_data in cube_mesh:
                normal = face_data['normal']
                center = face_data['center']
                view_vec = eye_pos - center

                # Backface culling: rysuj tylko ściany zwrócone ku oku
                if np.dot(normal, view_vec) > 0:
                    dist = float(np.linalg.norm(view_vec))
                    proj_base = self.project(face_data['base_quad'], eye_pos)
                    proj_stickers = [self.project(st, eye_pos) for st in face_data['stickers']]

                    if cube_name == "Lewa":
                        for pt in proj_base:
                            if 0 <= pt[0] < self.win_w and 0 <= pt[1] < self.win_h:
                                left_visible = True
                    elif cube_name == "Prawa":
                        for pt in proj_base:
                            if 0 <= pt[0] < self.win_w and 0 <= pt[1] < self.win_h:
                                right_visible = True

                    render_polys.append({
                        'dist': dist,
                        'base_poly': proj_base,
                        'stickers': proj_stickers,
                        'sticker_col': face_data['sticker_col'],
                    })

        # Sortowanie od najdalszych do najbliższych
        render_polys.sort(key=lambda item: item['dist'], reverse=True)

        for poly_item in render_polys:
            # Czarny plastik bryły
            cv2.fillPoly(canvas, [poly_item['base_poly']], COLOR_BLACK)
            cv2.polylines(canvas, [poly_item['base_poly']], isClosed=True, color=(10, 10, 10), thickness=2, lineType=cv2.LINE_AA)

            # 9 kolorowych naklejek z czarnymi obwódkami
            for st_poly in poly_item['stickers']:
                cv2.fillPoly(canvas, [st_poly], poly_item['sticker_col'])
                cv2.polylines(canvas, [st_poly], isClosed=True, color=(15, 15, 15), thickness=1, lineType=cv2.LINE_AA)

        # 4. Krawędź fizycznego okna (rama monitora)
        cv2.rectangle(canvas, (0, 0), (self.win_w - 1, self.win_h - 1), (180, 180, 180), 3)
        # Linia styku ekranu ze stołem (Z = 0)
        cv2.line(canvas, (0, self.win_h - 2), (self.win_w, self.win_h - 2), (0, 240, 255), 4)

        # 5. Mierzalny HUD diagnostyczny
        cv2.putText(
            canvas,
            "SCENA MIERZALNA: KOSTKI RUBIKA 57mm (1:1)",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        t_eye = f"Pozycja oka: X: {eye_pos[0]:+5.1f} mm | Y: {eye_pos[1]:+5.1f} mm | Z: {eye_pos[2]:5.1f} mm"
        cv2.putText(canvas, t_eye, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

        # Status widoczności kostek
        cv2.putText(
            canvas,
            "Kostka centralna (Z=0mm przy szybie): WIDOCZNA",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (100, 255, 100),
            2,
            cv2.LINE_AA,
        )

        col_l = (100, 255, 100) if left_visible else (140, 140, 150)
        status_l = "WIDOCZNA!" if left_visible else "UKRYTA (wychyl sie w PRAWO!)"
        cv2.putText(
            canvas,
            f"Lewa kostka (X=-205mm): {status_l}",
            (25, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            col_l,
            2,
            cv2.LINE_AA,
        )

        col_r = (100, 255, 100) if right_visible else (140, 140, 150)
        status_r = "WIDOCZNA!" if right_visible else "UKRYTA (wychyl sie w LEWO!)"
        cv2.putText(
            canvas,
            f"Prawa kostka (X=+205mm): {status_r}",
            (25, 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            col_r,
            2,
            cv2.LINE_AA,
        )

        # 6. Miniaturowy podgląd kamery PiP
        if camera_frame is not None:
            pip_w, pip_h = 240, 135
            pip_frame = cv2.flip(cv2.resize(camera_frame, (pip_w, pip_h)), 1)
            canvas[self.win_h - pip_h - 15:self.win_h - 15, self.win_w - pip_w - 15:self.win_w - 15] = pip_frame
            cv2.rectangle(
                canvas,
                (self.win_w - pip_w - 15, self.win_h - pip_h - 15),
                (self.win_w - 15, self.win_h - 15),
                (0, 240, 255),
                1,
            )

        return canvas
