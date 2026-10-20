# Minecraft HeadTracker 6-DoF & Off-Axis Perspective Portal

System śledzenia pozycji i orientacji głowy (6-DoF) z kamery internetowej (MediaPipe + OpenCV) zintegrowany z Minecraftem (Fabric) za pośrednictwem lokalnego protokołu binarnego UDP.

---

## 🎮 Wymagania

- **Minecraft**: 26.3 (Fabric Loader >= 0.19.0, Fabric API)
- **Java**: 25+
- **Python**: 3.11+ z zainstalowanymi zależnościami projektu (`.venv`)
- **Kamera internetowa** (np. wbudowana kamera MacBooka M1)

---

## 🚀 Krok po kroku: Instalacja i Uruchomienie

### 1. Instalacja Modu w Minecraft

1. Upewnij się, że masz zainstalowany **Fabric Loader** dla Minecrafta 26.3 oraz **Fabric API** w folderze modów.
2. Skopiuj zbudowany plik `.jar` modu do folderu `mods`:
   - **macOS**:
     ```bash
     mkdir -p "$HOME/Library/Application Support/minecraft/mods"
     cp mc_mod/build/libs/headtracker-1.0.0.jar "$HOME/Library/Application Support/minecraft/mods/"
     ```
   - **Windows**:
     ```cmd
     copy mc_mod\build\libs\headtracker-1.0.0.jar "%APPDATA%\.minecraft\mods\"
     ```
   - **Linux**:
     ```bash
     mkdir -p ~/.minecraft/mods
     cp mc_mod/build/libs/headtracker-1.0.0.jar ~/.minecraft/mods/
     ```

### 2. Uruchomienie Trackera Twarzy w Pythonie

Przed dołączeniem do gry (lub w jej trakcie) uruchom tracker twarzy:

```bash
./run_tracker.sh
```

Alternatywnie bezpośrednio przez interpreter wirtualnego środowiska:
```bash
.venv/bin/python detection.py
```

Tracker:
- Otworzy okno podglądu z wizualizacją biomechaniki oka i głowy.
- Wyświetla status transmisji UDP na `127.0.0.1:4242`.
- Klawisz `q` w oknie podglądu zamyka tracker.
- Klawisz `c` w oknie podglądu pozwala skalibrować neutralną wysokość Y.

---

## 🕹️ Sterowanie w Grze (Keybinds)

W grze zarejestrowane są dedykowane skróty klawiszowe w menu **Opcje -> Sterowanie -> Head Tracker**:

| Klawisz | Funkcja | Opis działania |
| :---: | :---: | :--- |
| **`H`** | **Toggle Tracking** | Włącza / wyłącza śledzenie głowy w czasie rzeczywistym. Przy wyłączeniu kamera natychmiast wraca do standardowego widoku gracza. Na ActionBarze pojawia się komunikat o stanie. |
| **`J`** | **Recenter (Kalibracja)** | Ustawia bieżącą pozycję głowy przed ekranem jako punkt bazowy (0, 0, 0). Usiądź w wygodnej pozycji i naciśnij `J`. |

---

## 🌟 Jak to Działa?

1. **6-DoF Leaning / Peeking (Kamera)**:
   - Wychylenie głowy w prawo przesuwa perspektywę gracza w prawo (np. wychylenie się zza ściany lub narożnika).
   - Wychylenie w lewo, w górę, w dół oraz pochylenie w przód/tył są precyzyjnie nanoszone na kamerę z zachowaniem standardowego hitboxa i pozycji postaci gracza.
   - Offset kamery jest automatycznie rotowany zgodnie z bieżącym kierunkiem patrzenia gracza (Yaw/Pitch), dzięki czemu ruch w prawo zawsze odpowiada wychyleniu w prawo w Twoim polu widzenia.

2. **Off-Axis Perspective Projection (Efekt Portalu / Okna 3D)**:
   - Obraz na ekranie MacBooka zachowuje się jak fizyczne okno do świata Minecrafta.
   - Gdy zbliżasz się lub przesuwasz względem ekranu, macierz rzutowania dynamicznie dostosowuje asymetrię frustuma, tworząc złudzenie holograficznej głębi.

3. **Bezpieczeństwo i Płynność (Fail-safe)**:
   - Odbiornik UDP w modzie działa asynchronicznie w osobnym wątku i nie blokuje pętli renderowania gry.
   - W przypadku utraty wykrycia twarzy lub zamknięcia trackera (>300 ms), offset płynnie powraca do pozycji neutralnej (brak szarpnięć czy zacięć).
