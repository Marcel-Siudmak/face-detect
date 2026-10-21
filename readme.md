# Wirtualne Okno 3D (Head & Eye Tracking z rzutowaniem Off-Axis)

Projekt realizujący symulację **Wirtualnego Okna (Fish-tank VR)** przy użyciu kamery internetowej (zoptymalizowany pod MacBooka z matrycą Retina i kamerą FaceTime HD). Obraz 3D jest rzutowany w perspektywie *off-axis*, dając złudzenie zaglądania za ekran fizycznego monitora.

---

## 🚀 Kluczowe możliwości

* **Precyzyjne śledzenie pozycji głowy w 3D:** MediaPipe FaceLandmarker + algorytm `solvePnP` z 10-punktowym modelem antropometrycznym twarzy (w milimetrach).
* **Biomechanika wychylania (*Peeking*):** Nieliniowy model intencji bazujący na `smoothstep` (tryby: *Zaułek* – wyglądanie zza węgła, *Murek* – wyglądanie z góry, *Daszek* – spoglądanie z dołu) z dynamicznym wyborem oka dominującego.
* **Filtracja czasowa:** Tłumienie drgań PnP oraz płynne przejścia wagi (*exponential blend*).
* **Rzutowanie Off-Axis (1:1):** Perspektywa uwzględniająca fizyczne wymiary ekranu (MacBook Air 13.3" M1) oraz odległość oka użytkownika od płaszczyzny $Z=0$.
* **Mierzalna scena testowa:** Wirtualny stół i kostki Rubika w skali 1:1 (57 mm) ze sprawdzaniem widoczności przy wychyleniu głowy.
* **Opcjonalna transmisja UDP:** Możliwość wysyłania 32-bajtowych pakietów binarnych do zewnętrznych silników (Unity, Unreal Engine, Blender, OpenTrack).

---

## 📁 Struktura projektu

```text
face_detect/
├── apps/
│   ├── run_virtual_window.py   # Główna aplikacja: Wirtualne okno 3D (kamera + tracker + renderer + PiP)
│   └── run_tracker.py          # Diagnostyczny podgląd trackera z panelem HUD (radary X-Z, X-Y, suwak wagi)
├── src/
│   ├── config.py               # Parametry fizyczne ekranu, macierz K kamery, model 3D twarzy, progi czułości
│   ├── camera.py               # Inicjalizacja kamery (AVFoundation dla macOS / fallback)
│   ├── tracker.py              # Klasa FaceTracker (estymacja 3D, biomechanika, wygładzanie drgań)
│   ├── renderer_offaxis.py     # Klasa VirtualWindowRenderer (geometria kostek Rubika, stół, rzutowanie off-axis)
│   ├── hud.py                  # Rysowanie panelu diagnostycznego HUD i widoków rzutu
│   └── protocol.py             # Protokół binarny UDP (HeadTrackerUDPClient i HeadTrackerUDPReceiver)
├── models/
│   └── face_landmarker.task    # Wagi modelu MediaPipe FaceLandmarker
├── experiments/
│   └── simplest_detection.py   # Prosty prototyp testowy detekcji punktów twarzy
├── requirements.txt            # Zależności Python (opencv-python, mediapipe, numpy)
└── README.md
```

---

## 🛠️ Wymagania i instalacja

Zalecany Python: **3.10** lub **3.11** (MediaPipe wymaga Pythona $\le$ 3.11).

```bash
# 1. Klonowanie i wejście do katalogu
git clone <adres-repozytorium>
cd face_detect

# 2. Utworzenie i aktywacja środowiska wirtualnego
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Instalacja bibliotek
pip install -r requirements.txt
```

---

## 🎮 Uruchomienie

### 1. Wirtualne okno 3D (Główna aplikacja)
Uruchamia pełną scenę 3D z podglądem kamery w prawym dolnym rogu (PiP):
```bash
python apps/run_virtual_window.py
```
Opcjonalne flagi:
* `--no-pip` – wyłącza podgląd z kamery w rogu okna.
* `--udp` – włącza jednoczesne rozsyłanie pakietów pozy przez UDP na `127.0.0.1:4242`.

### 2. Samodzielny podgląd diagnostyczny (HUD)
Wyświetla obraz z kamery wraz z nakładką HUD, wektorem normalnym twarzy, suwakiem intencji oraz rzutami radaru:
```bash
python apps/run_tracker.py
```

### Sterowanie podczas działania:
* **`c`** – kalibracja neutralnej wysokości oczu względem środka ekranu.
* **`q`** lub **`ESC`** – bezpieczne wyjście z programu.
