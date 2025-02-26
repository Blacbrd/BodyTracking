import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import json
import vosk
import pyaudio
from playsound import playsound
import time
import pyautogui

# -------------------- Global Setup --------------------

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Punch State Machine Variables ---
left_ready = True
right_ready = True

SMALL_ANGLE_THRESHOLD = 45    # Degrees: elbow is bent (ready for a punch)
LARGE_ANGLE_THRESHOLD = 120   # Degrees: elbow is extended (punch thrown)

# Add threshold for when arms just go down, since I don't want the player to accidentally punch while resting


left_punch_display_counter = 0
right_punch_display_counter = 0
DISPLAY_FRAMES = 15           # Number of frames to display the "Punch!" label

# --- Walking (Forward Movement) Variables ---
WALK_THRESHOLD_ENABLED = True  # Set to False to disable the threshold line feature.
knee_threshold = None          # Will store the y coordinate of the horizontal threshold line.

# If both knees are raised between the CONSECUTIVE_WALK_THRESHOLD, then continuously walk
CONSECUTIVE_WALK_THRESHOLD = 0.5  
walk_last_time = 0
walk_hold_active = False
walk_timer = None

# Flag to ensure only one walk event is triggered per knee raise.
walk_triggered = False

# --- Jumping Variables ---
jump_ready = True

# --- Block Placement Variables ---
# This flag will be True when the wrists are "uncrossed" (ready to place a block again).
placement_ready = True

# -------------------- Punch to Minecraft Automation --------------------

# This helps determine whether the last punch and current punch were within the CONSECUTIVE_THRESHOLD time
punch_last_time = 0

punch_hold_active = False
punch_timer = None

# Amount of seconds. If 2 punches fall within this time frame, the program takes it as mining
CONSECUTIVE_THRESHOLD = 1 

def execute_single_punch():

    # Makes sure that these variables are updated throughout the program
    global punch_timer, punch_hold_active

    # If there is not consecutive punches, only press the mouse button
    if not punch_hold_active:
        pyautogui.click(button='left')
        print("Single punch executed: left click")
    
    # Reset timer
    punch_timer = None

def handle_punch():
    global punch_last_time, punch_hold_active, punch_timer

    # Gets the current time
    now = time.time()

    # If there was no timer before
    if punch_timer is None:
        punch_last_time = now

        # This will set a delay for "CONSECUTIVE_THRESHOLD" seconds (which could be 1 second)
        # If a second punch is recognised within this time, the else block will execute, which will cancel this thread
        # If not, then the execute_single_punch() function will run
        punch_timer = threading.Timer(CONSECUTIVE_THRESHOLD, execute_single_punch)
        punch_timer.daemon = True

        # Starts the thread
        punch_timer.start()
    else:

        # If a second punch is detected (since punch_timer is not None), the delay and therefore function call will be cancelled
        punch_timer.cancel()
        punch_timer = None

        # If the punch IS active, then there's no need to press the left mouse button again, it just keeps holding it
        if not punch_hold_active:
            pyautogui.mouseDown(button='left')
            punch_hold_active = True
            print("Consecutive punches detected: holding left mouse button for mining")
        
        # Checks for more consecutive punches
        punch_last_time = now

# -------------------- Walking Automation Functions --------------------

# If walking has ended 
def release_walk():
    global walk_hold_active
    pyautogui.keyUp('w')
    print("Single walk ended: 'w' key released")
    walk_hold_active = False

def execute_single_walk():
    global walk_timer, walk_hold_active
    if not walk_hold_active:
        pyautogui.keyDown('w')
        print("Single walk executed: 'w' key held for 0.1 seconds")
        t = threading.Timer(0.1, release_walk)
        t.daemon = True
        t.start()
    walk_timer = None

def handle_walk_event():
    global walk_last_time, walk_hold_active, walk_timer
    now = time.time()
    if walk_timer is None:
        walk_last_time = now
        walk_timer = threading.Timer(CONSECUTIVE_WALK_THRESHOLD, execute_single_walk)
        walk_timer.daemon = True
        walk_timer.start()
    else:
        walk_timer.cancel()
        walk_timer = None
        if not walk_hold_active:
            pyautogui.keyDown('w')
            walk_hold_active = True
            print("Consecutive walking detected: holding 'w' continuously")
        walk_last_time = now

# -------------------- Jumping Automation --------------------
def execute_jump():
    pyautogui.keyDown('space')
    time.sleep(0.1)
    pyautogui.keyUp('space')

def handle_jump_event():
    global jump_ready
    if jump_ready:
        if not walk_hold_active:
            def jump_forward():
                pyautogui.keyDown('w')
                time.sleep(0.05)
                pyautogui.keyDown('space')
                time.sleep(0.1)
                pyautogui.keyUp('space')
                pyautogui.keyUp('w')
                print("Jump executed with forward momentum (non-walking)")
            threading.Thread(target=jump_forward, daemon=True).start()
        else:
            threading.Thread(target=execute_jump, daemon=True).start()
            print("Jump while walking executed: space pressed with 'w' held")
        jump_ready = False

# -------------------- Utility Functions --------------------
def calculate_angle(a, b, c):
    a2d = np.array(a[:2])
    b2d = np.array(b[:2])
    c2d = np.array(c[:2])
    ba = a2d - b2d
    bc = c2d - b2d
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# This function now only plays the sound for arms calibration.
def calibrate(current_wristL_z, current_wristR_z):
    playsound(r"C:\Users\blacb\Downloads\fart-with-reverb.mp3")
    print("Calibration of arms complete")

def find_horizontal_threshold(left_knee_y, offset=25):
    return left_knee_y - offset

# -------------------- Vosk Audio Setup --------------------
vosk_model_path = r"C:\Users\blacb\Downloads\vosk-model-en-us-0.22-lgraph\vosk-model-en-us-0.22-lgraph"
model = vosk.Model(vosk_model_path)

# These are the only words that will be recognised
words = ["recalibrate", "calibrate", "arms", "arm", "legs", "leg", "inventory", "open", "close", "opened", "closed"]
grammar = json.dumps(words)

audio_queue = queue.Queue()
command_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status_flags):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def speech_recognition_worker():
    recognizer = vosk.KaldiRecognizer(model, 16000, grammar)
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result_json = recognizer.Result()
            result = json.loads(result_json)
            text = result.get("text", "")
        else:
            partial_json = recognizer.PartialResult()
            result = json.loads(partial_json)
            text = result.get("partial", "")

        if "calibrate" in text.lower() and ("arm" in text.lower() or "leg" in text.lower()):
            print("Voice command detected:", text)
            command_queue.put("calibrate_arms" if "arm" in text else "calibrate_legs")
            time.sleep(0.2)
            recognizer.Reset()
        
        # # If "recalibrate arms" is spoken, queue the arms calibration command.
        # if "recalibrate" in text.lower() and ("arm" in text.lower() or "arms" in text.lower() or "hand" in text.lower() or "hands" in text.lower() or "ones" in text.lower()):
        #     print("Voice command detected: recalibrate arms")
        #     command_queue.put("calibrate_arms")

        #     # Small delay
        #     time.sleep(0.2)
        #     recognizer.Reset()
        # # If "recalibrate legs" is spoken, queue the legs recalibration command.
        # elif "recalibrate" in text.lower() and ("leg" in text.lower() or "legs" in text.lower() or "feet" in text.lower()):
        #     print("Voice command detected: recalibrate legs")
        #     command_queue.put("calibrate_legs")

        #     # Small delay before reset
        #     time.sleep(0.2)
        #     recognizer.Reset()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=3000,
                stream_callback=audio_callback)
stream.start_stream()

speech_thread = threading.Thread(target=speech_recognition_worker, daemon=True)
speech_thread.start()

# -------------------- Main Loop with OpenCV & Mediapipe --------------------
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            def get_landmark_point(landmark):
                return [int(landmark.x * frame.shape[1]),
                        int(landmark.y * frame.shape[0]),
                        landmark.z]

            # Left-side landmarks
            shoulderL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            elbowL    = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            wristL    = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

            # Right-side landmarks
            shoulderR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            elbowR    = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            wristR    = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

            # Calculate elbow angles for each arm.
            angleL = int(calculate_angle(shoulderL, elbowL, wristL))
            angleR = int(calculate_angle(shoulderR, elbowR, wristR))

            cv2.putText(image, f'{angleL}', (elbowL[0] - 30, elbowL[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'{angleR}', (elbowR[0] - 30, elbowR[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --------- Punch Detection State Machine ----------
            if angleL < SMALL_ANGLE_THRESHOLD:
                left_ready = True
            if left_ready and angleL > LARGE_ANGLE_THRESHOLD:
                print("Left punch detected!")
                left_ready = False
                left_punch_display_counter = DISPLAY_FRAMES
                handle_punch()

            if angleR < SMALL_ANGLE_THRESHOLD:
                right_ready = True
            if right_ready and angleR > LARGE_ANGLE_THRESHOLD:
                print("Right punch detected!")
                right_ready = False
                right_punch_display_counter = DISPLAY_FRAMES
                handle_punch()

            # --------- Block Placement Feature ----------
            if wristR[0] < wristL[0]:
                if placement_ready:
                    pyautogui.click(button='right')
                    print("Block placed via wrist crossing!")
                    placement_ready = False
            else:
                placement_ready = True

            # --------- Walking (Forward Movement) Feature ----------
            left_knee = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])

            if WALK_THRESHOLD_ENABLED:
                if knee_threshold is None:
                    knee_threshold = find_horizontal_threshold(left_knee[1])
                cv2.line(image, (0, knee_threshold), (frame.shape[1], knee_threshold), (255, 0, 255), 2)
                
                left_above = left_knee[1] < knee_threshold
                right_above = right_knee[1] < knee_threshold
                single_knee_event = (left_above != right_above)
                
                if single_knee_event and not walk_triggered:
                    walk_triggered = True
                    pyautogui.keyDown('w')
                    print("Walking triggered: 'w' key pressed")
                    t = threading.Timer(0.5, lambda: (pyautogui.keyUp('w'), print("Walking ended: 'w' key released")))
                    t.daemon = True
                    t.start()
                    cv2.putText(image, "WALKING", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                
                if not (left_above or right_above):
                    if walk_triggered:
                        print("Knees below threshold: resetting walk trigger")
                    walk_triggered = False

            # --------- Jumping Feature ----------
            if left_knee[1] < knee_threshold and right_knee[1] < knee_threshold:
                if jump_ready:
                    handle_jump_event()
            else:
                jump_ready = True

        except Exception as e:
            print("Error processing pose:", e)

        if punch_hold_active and (time.time() - punch_last_time > CONSECUTIVE_THRESHOLD):
            pyautogui.mouseUp(button='left')
            punch_hold_active = False
            print("Consecutive punches ended: released left mouse button")

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if not command_queue.empty():
            command = command_queue.get()
            if command == "calibrate_arms":
                try:
                    calib_left = wristL[2]
                    calib_right = wristR[2]
                    threading.Thread(target=calibrate, args=(calib_left, calib_right), daemon=True).start()
                except Exception as e:
                    print("Calibration error:", e)
            elif command == "calibrate_legs":
                try:
                    if 'left_knee' in locals() and left_knee is not None:
                        knee_threshold = find_horizontal_threshold(left_knee[1])
                        print("Knee threshold recalibrated. New threshold:", knee_threshold)
                    else:
                        print("Knee landmarks not detected. Cannot recalibrate threshold.")
                except Exception as e:
                    print("Recalibration error:", e)

        cv2.imshow("Body Recognition", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
p.terminate()
