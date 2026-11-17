import time
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
            'stickers_arr': np.array(stickers, dtype=np.float64),
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

        # Prekalkulacja geometrii siatki stołu (jedno rzutowanie dla wszystkich odcinków)
        grid_pts = []
        grid_styles = []
        for z_grid in np.arange(0.0, -TABLE_DEPTH_MM - 1.0, -50.0):
            grid_pts.append([-300.0, TABLE_Y, z_grid])
            grid_pts.append([300.0, TABLE_Y, z_grid])
            col_line = (120, 130, 145) if abs(z_grid) < 1e-3 else (50, 54, 62)
            thick = 2 if abs(z_grid) < 1e-3 else 1
            grid_styles.append((col_line, thick))

        for x_grid in np.arange(-250.0, 251.0, 50.0):
            grid_pts.append([x_grid, TABLE_Y, 0.0])
            grid_pts.append([x_grid, TABLE_Y, ROOM_BACK_Z])
            col_line = (90, 100, 115) if abs(x_grid) < 1e-3 else (64, 68, 76)
            thick = 2 if abs(x_grid) < 1e-3 else 1
            grid_styles.append((col_line, thick))

        self.grid_lines_3d = np.array(grid_pts, dtype=np.float64)
        self.grid_styles = grid_styles
        self._last_render_t: Optional[float] = None
        self._fps_smooth: float = 0.0

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
        fps: Optional[float] = None,
        frame_time_ms: Optional[float] = None,
    ) -> np.ndarray:
        """Renderuje pełną klatkę wirtualnego okna ze sceną 3D, stołem, licznikiem FPS i opcjonalnym PiP."""
        now = time.perf_counter()
        if self._last_render_t is not None:
            dt_r = max(now - self._last_render_t, 1e-4)
            inst_fps = 1.0 / dt_r
            if self._fps_smooth <= 0.0:
                self._fps_smooth = inst_fps
            else:
                self._fps_smooth += (inst_fps - self._fps_smooth) * 0.15
        self._last_render_t = now
        display_fps = fps if (fps is not None and fps > 0.0) else self._fps_smooth

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

        # 2. Siatka stołu (zwektoryzowane rzutowanie jednym wywołaniem)
        pts_gl = self.project(self.grid_lines_3d, eye_pos)
        for idx, (col_line, thick) in enumerate(self.grid_styles):
            p0 = (int(pts_gl[2 * idx, 0]), int(pts_gl[2 * idx, 1]))
            p1 = (int(pts_gl[2 * idx + 1, 0]), int(pts_gl[2 * idx + 1, 1]))
            cv2.line(canvas, p0, p1, col_line, thick, cv2.LINE_AA)

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
                    proj_stickers = list(
                        self.project(face_data['stickers_arr'].reshape(-1, 3), eye_pos).reshape(9, 4, 2)
                    )

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

            # 9 kolorowych naklejek narysowanych jednym wywołaniem fillPoly + polylines
            cv2.fillPoly(canvas, poly_item['stickers'], poly_item['sticker_col'])
            cv2.polylines(canvas, poly_item['stickers'], isClosed=True, color=(15, 15, 15), thickness=1, lineType=cv2.LINE_AA)

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

        # Wskaźnik klatkarzu (FPS) w prawym górnym rogu okna 3D
        fps_col = (80, 255, 80) if display_fps >= 45.0 else ((0, 220, 255) if display_fps >= 25.0 else (60, 60, 255))
        fps_text = (
            f"FPS: {display_fps:4.1f} ({frame_time_ms:4.1f} ms)"
            if frame_time_ms is not None and frame_time_ms > 0.0
            else f"FPS: {display_fps:4.1f}"
        )
        cv2.putText(canvas, fps_text, (self.win_w - 280, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_col, 2, cv2.LINE_AA)

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
