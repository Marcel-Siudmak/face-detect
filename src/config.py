from pathlib import Path
import numpy as np

# ==============================================================================
# ŚCIEŻKI PROJEKTU
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# ==============================================================================
# PARAMETRY SPRZĘTOWE: MacBook Air 13.3" (M1)
# ==============================================================================
SCREEN_WIDTH_MM = 286.5
SCREEN_HEIGHT_MM = 179.0
BEZEL_TOP_TO_SCREEN_MM = 6.0
CAMERA_TO_SCREEN_CENTER_Y_MM = BEZEL_TOP_TO_SCREEN_MM + (SCREEN_HEIGHT_MM / 2.0)  # ~95.5 mm

HW = SCREEN_WIDTH_MM / 2.0  # 143.25 mm
HH = SCREEN_HEIGHT_MM / 2.0  # 89.50 mm

# Domyślna rozdzielczość wirtualnego okna (16:10)
WIN_W = 1280
WIN_H = 800

# Parametry kamery FaceTime HD 720p
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Rozdzielczość wejściowa dla detektora MediaPipe (przyspieszenie CPU bez utraty precyzji)
MP_DETECT_WIDTH = 640
MP_DETECT_HEIGHT = 360

CAMERA_INTRINSIC_K = np.array([
    [1140.0,    0.0, 640.0],
    [   0.0, 1140.0, 360.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float64)

CAMERA_DISTORTION_D = np.zeros((5, 1), dtype=np.float64)

# ==============================================================================
# MODEL ANTROPOMETRYCZNY TWARZY (10 punktów 3D w mm)
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

# Pozycja oczu w układzie głowy (mm)
EYE_RIGHT_HEAD = np.array([-31.5, 3.0, 8.0])  # viewer's left
EYE_LEFT_HEAD  = np.array([ 31.5, 3.0, 8.0])  # viewer's right

IRIS_RIGHT_ID = 468
IRIS_LEFT_ID  = 473

EYE_R_OUTER_ID = 33
EYE_R_INNER_ID = 133
EYE_R_UPPER_ID = 159
EYE_R_LOWER_ID = 145

EYE_L_INNER_ID = 362
EYE_L_OUTER_ID = 263
EYE_L_UPPER_ID = 386
EYE_L_LOWER_ID = 374

# ==============================================================================
# PARAMETRY HYBRYDOWEGO SYSTEMU INTENCJI (360° VECTOR + GAZE + EAR WINK)
# ==============================================================================
PEEK_DEADZONE_MM = 6.0          # Martwa strefa centralna (mm)
PEEK_FULL_DIST_MM = 52.0        # Dystans pełnego przejścia na oko wiodące (mm)
PEEK_CURVE_EXP = 1.15           # Wykładnik krzywej smoothstep
VERTICAL_TILT_GAIN = 2.2        # Wzmocnienie osi pionowej Y przy przechyle głowy (Roll)

GAZE_EYE_WEIGHT = 0.65          # Waga wychylenia źrenic względem kąta Yaw głowy
GAZE_CONVERGENCE_BOOST = 0.45   # Wzmocnienie intencji, gdy wzrok kontruje wychylenie
GAZE_DIRECT_PEEK_GAIN = 0.30    # Bezpośredni wkład spojrzenia przy małym wychyleniu
LOOK_AWAY_THRESHOLD = 0.22      # Próg wygaszania, gdy głowa i wzrok uciekają poza ekran

VELOCITY_LEAD_GAIN = 0.0030     # Wyprzedzenie prędkościowe (s / mm)
VELOCITY_SMOOTH_RATE = 12.0     # Filtr dolnoprzepustowy pochodnej prędkości

EAR_SQUINT_THRESHOLD = 0.165    # Próg przymknięcia powieki (EAR)
EAR_ASYMMETRY_MIN = 0.055       # Minimalna różnica EAR między oczami dla trybu Wink

# ==============================================================================
# STAŁE FILTRACJI CZASOWEJ (ASYMETRYCZNE PŁYWANIE I TŁUMIENIE DRGAŃ)
# ==============================================================================
ALPHA_ATTACK_RATE  = 16.0       # Błyskawiczna reakcja przy wychylaniu i ruchu głowy
ALPHA_RELEASE_RATE = 6.5        # Płynne podtrzymanie przy nieruchomej głowie
ALPHA_CENTER_RATE  = 12.0       # Szybki powrót do środka w strefie neutralnej
ALPHA_SMOOTH_RATE  = 12.0       # Kompatybilność wsteczna
POS_SMOOTH_RATE    = 14.0       # Bazowe wygładzanie drgań PnP w spoczynku
POS_FAST_RATE      = 36.0       # Szybkie śledzenie pozycji 3D podczas ruchu głowy

# ==============================================================================
# PARAMETRY SCENY 3D (KOSTKI RUBIKA I STÓŁ)
# ==============================================================================
TABLE_DEPTH_MM = 500.0
ROOM_BACK_Z = -500.0
RUBIK_SIZE = 57.0

# Poziom blatu stołu: idealnie dolna krawędź ekranu
TABLE_Y = -HH  # -89.5 mm
# Wysokość środka kostki spoczywającej na stole
CUBE_Y = TABLE_Y + (RUBIK_SIZE / 2.0)  # -61.0 mm

# Kostka centralna: przednia ścianka dotyka Z = 0
CUBE_CENTER_POS = np.array([0.0, CUBE_Y, -(RUBIK_SIZE / 2.0)])

# Kostki lewa i prawa
SIDE_Z = -160.0
CUBE_LEFT_POS  = np.array([-205.0, CUBE_Y, SIDE_Z])
CUBE_RIGHT_POS = np.array([ 205.0, CUBE_Y, SIDE_Z])

# Kolory BGR kostki Rubika
COLOR_BLUE   = (210,  50,  30)
COLOR_GREEN  = ( 40, 180,  40)
COLOR_YELLOW = ( 30, 220, 240)
COLOR_WHITE  = (230, 230, 230)
COLOR_RED    = ( 30,  30, 220)
COLOR_ORANGE = ( 20, 120, 235)
COLOR_BLACK  = ( 20,  20,  20)
