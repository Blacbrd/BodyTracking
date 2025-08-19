# Minecraft Body & Hand Tracking Controller

<a href="https://www.youtube.com/watch?v=QgRdTzry0po">
<img width="720" height="405" alt="YouTube thumbnail" href src="https://github.com/user-attachments/assets/13077323-d19f-4348-96ad-5209aefb8188" />
</a>

A Python program that maps body/hand movement + simple voice commands to Minecraft controls (mouse and keyboard).
It uses MediaPipe for pose/hand tracking and Vosk for offline speech commands, plus `pyautogui`/`pynput` to control the game.

> **Warning:** This program *controls your mouse and keyboard*. Run it when Minecraft is focused (or otherwise in a safe environment) and be ready to kill the script if it acts unexpectedly (press `q` in the OpenCV window or use your OS-level input lock).

---
## Usage of program

* If you want to use this program for a video, fork it, make sure to credit both my GitHub and YouTube!

---

## YouTube showcase

* Check out the YouTube video related to this project for a showcase of how I made it/use it!:
* YouTube link: https://www.youtube.com/watch?v=QgRdTzry0po

## Features

* Punch / mining detection (single click / hold for consecutive punches)
* Walking with alternating knee raise pattern
* Jumping and crouch with body posture detection
* Block placement when wrists cross
* Inventory opening with wrist-to-face gesture; in-menu finger-as-mouse hand mode
* Voice commands (Vosk) for recalibration and switching modes
* Head-turn → camera movement (mouse) support
* Basic calibration via voice commands: `calibrate arms`, `calibrate legs`, `calibrate head`, `calibrate box`
* Mode switching via voice: `change hand`, `change body`

---

## Requirements (software & hardware)

* **Python 3.12.x** (the project was tested with Python 3.12).
* Camera + Microphone.
* Laptop that can run Minecraft along with other apps.
* Minecraft client (and Vivecraft if using VR).
* Preferably 2 monitors; works with 1.

---

## Files

* `bodyTracker.py` — Body controller
* `bodyTrackerVR.py` — Body controller with VR
* `requirements.txt` — Python dependencies (see below).
* `README.md` — this file.

---

## Quick setup (safe, cross-platform)

> General idea: create and activate a venv, upgrade pip/tools, install requirements, download a Vosk model, set `vosk_model_path` in the script, then run `bodyTracker.py`.

### 1) Create & activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

**macOS / Linux (bash / zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> If `PyAudio` fails on your platform see **Troubleshooting** below.

### 3) Download a Vosk model

Download a Vosk model and set `vosk_model_path` (absolute path) in the script. The official model list and downloads are here: 
https://alphacephei.com/vosk/models

For example, a small English model:

* `vosk-model-small-en-us-0.15` or `vosk-model-en-us-0.22` (larger / more accurate).

> I used vosk-model-en-us-0.22-lgraph

Unzip the model and point `vosk_model_path` to the folder:

```python
vosk_model_path = r"/absolute/path/to/vosk-model-small-en-us-0.15"
```

### 4) Minecraft settings (required)

* Pause Game → Options → Controls → Mouse Settings: `Raw mouse input = OFF`.
<img width="1775" height="626" alt="image" src="https://github.com/user-attachments/assets/bb8ae948-6110-4eb9-9856-8e0dc12ce668" />


> If you don't do this the camera turning won't work!
* If using VR, install the Vivecraft mod (or any other compatible VR mod that allows movement with keyboard and mouse) (Can be installed using Cursed Forge)

> Tutorial on how to install Vivecraft: 
> (Cursed Forge) https://www.youtube.com/watch?v=N_77a6dXMJ0

* Pause Game → Options → Controls → Mouse Settings: set look-left / look-right to `m` and `n` if using VR/vivecraft.

### 5) Run

```bash
python bodyTracker.py
```

* The OpenCV window shows the camera feed and landmark overlays. Press `q` inside the OpenCV window to quit.

---

## Voice commands

Supported words are limited by the small recognizer grammar. Useful commands you can say while the program runs:

* `calibrate arms` → Dummy function, doesn't do anything
* `calibrate legs` → sets knee threshold
* `calibrate head` → recalibrates base head rotation
* `calibrate box` → repositions the inventory box for hand mode
* `change hand` → switch to hand mode (inventory / menus)
* `change body` → switch back to body mode

---

## Configurable constants (top of `main.py`)

* `SMALL_ANGLE_THRESHOLD`, `LARGE_ANGLE_THRESHOLD` — punch detection angles
* `CONSECUTIVE_PUNCH_THRESHOLD` — time window for “mining” detection
* `KNEE_OFFSET` — how far above knees the purple threshold line is drawn (pixels)
* `WALK_THRESHOLD_ENABLED`, `CONSECUTIVE_WALK_THRESHOLD` — walking behaviour
* `DOMINANT_HAND` — `"Right"` or `"Left"`
* `RECTANGLE_WIDTH` / `HEIGHT` — hand-mode pointing box size
* `MONITOR` selection is automatic from `screeninfo`

---

## Safety notes (IMPORTANT)

* This program will click and press keys. Keep a spare keyboard/mouse or an OS-level input lock if you need to regain control quickly.
* Test with Minecraft already opened and in focus to prevent the mouse from drifting onto other programs.
* Do not run on an account or system where unexpected keypresses could be dangerous.
* Make sure stretch/warm up before using the program, and take regular breaks.
* The head turning feature is very finecky and hard to get used to, make sure to avoid fast head movements as these can cause neck strain (trust me I've been there).

---

## Troubleshooting & platform notes

### MediaPipe / Python 3.12

MediaPipe currently publishes Python wheels and supports desktop Python versions including Python 3.12; if you see an error when `pip install mediapipe`, please ensure `pip` is up to date using `pip install --upgrade pip` and try again.

### PyAudio install tips

* **Windows:** use `pipwin` or a prebuilt wheel if `pip install PyAudio` fails.
* **macOS:** `brew install portaudio` then `pip install pyaudio`.
* **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev python3-dev` then `pip install pyaudio`.


### Vosk & models

Vosk itself installs via `pip install vosk`. Models are downloaded separately (alpha-cephei models page and mirrors). Point `vosk.Model()` at the unzipped model directory. 


