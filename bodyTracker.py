import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import queue
import json
import vosk
import pyaudio
from playsound import playsound
import time
import pyautogui
import math

# -------------------- Global Setup --------------------

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Punch State Machine Variables ---
left_ready = True
right_ready = True

SMALL_ANGLE_THRESHOLD = 45    # Degrees: elbow is bent (ready for a punch)
LARGE_ANGLE_THRESHOLD = 120   # Degrees: elbow is extended (punch thrown)

# Threshold for when arms just go down, since I don't want the player to accidentally punch while resting
arm_threshold = None # Set to just above the elbows. Wrists will have to go above this to activate it


left_punch_display_counter = 0
right_punch_display_counter = 0
DISPLAY_FRAMES = 15           # Number of frames to display the "Punch!" label

# --- Walking (Forward Movement) Variables ---
WALK_THRESHOLD_ENABLED = True  # Set to False to disable the threshold line feature.
knee_threshold = None          # Will store the y coordinate of the horizontal threshold line.

# Walking state machine variables
left_knee_was_up = False
right_knee_was_up = False
left_knee_is_up = False
right_knee_is_up = False

# If both knees are raised between the CONSECUTIVE_WALK_THRESHOLD, then continuously walk
CONSECUTIVE_WALK_THRESHOLD = 0.7
walk_last_time = 0
walk_hold_active = False
walk_timer = None

# For tracking alternating knee steps
last_knee_raised = None  # Will be "left" or "right"
step_count = 0
step_time_threshold = 1.0  # Time window to detect continuous walking pattern
last_step_time = 0

# --- Jumping Variables ---
jump_ready = True

# --- Block Placement Variables ---
# This flag will be True when the wrists are "uncrossed" (ready to place a block again).
placement_ready = True

# --- Camera movement Variables ---
# Only need the left ear as if it moves closer to the right ear it'll just be >= 100%
base_rotation = None

# -------------------- Punch to Minecraft Automation --------------------

# This helps determine whether the last punch and current punch were within the CONSECUTIVE_THRESHOLD time
punch_last_time = 0

punch_hold_active = False
punch_timer = None

# Amount of seconds. If 2 punches fall within this time frame, the program takes it as mining
CONSECUTIVE_THRESHOLD = 0.5

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

# Function to release the walk key when walking ends
def release_walk():
    global walk_hold_active
    if walk_hold_active:
        pyautogui.keyUp('w')
        print("Walking ended: 'w' key released")
        walk_hold_active = False

# Function to execute a single short walk
def execute_single_walk():
    global walk_timer, walk_hold_active
    if not walk_hold_active:
        pyautogui.keyDown('w')
        print("Single walk step: 'w' key held for 0.2 seconds")
        walk_hold_active = True
        t = threading.Timer(0.2, release_walk) # Timer used to delay function calling. Similar to time.sleep() except it doesn't block the main thread
        t.daemon = True
        t.start()
    walk_timer = None

# Function to handle the walking state and transitions
def handle_walking(left_knee_position, right_knee_position, knee_threshold):
    global left_knee_is_up, right_knee_is_up, left_knee_was_up, right_knee_was_up
    global last_knee_raised, step_count, last_step_time, walk_hold_active
    
    # Update current knee positions
    left_knee_is_up = left_knee_position < knee_threshold
    right_knee_is_up = right_knee_position < knee_threshold
    
    current_time = time.time()
    
    # Detect a new step (knee goes from down to up)
    if left_knee_is_up and not left_knee_was_up:
        handle_knee_step("left", current_time)
    elif right_knee_is_up and not right_knee_was_up:
        handle_knee_step("right", current_time)
    
    # If both knees are down, we can prepare for the next step
    if not left_knee_is_up and not right_knee_is_up:
        left_knee_was_up = False
        right_knee_was_up = False
    
    # Update previous state
    left_knee_was_up = left_knee_is_up
    right_knee_was_up = right_knee_is_up
    
    # Check for walking timeout (if user stopped walking)
    if walk_hold_active and current_time - last_step_time > step_time_threshold:
        release_walk()
        step_count = 0
        print("Walking timeout detected, released walk key")

# Function to handle a single knee step
def handle_knee_step(knee, current_time):
    global last_knee_raised, step_count, last_step_time, walk_hold_active
    
    # Check if this is an alternating step pattern
    if last_knee_raised is not None and last_knee_raised != knee:
        # Alternating step detected
        time_diff = current_time - last_step_time
        
        if time_diff < step_time_threshold:
            # Increment step counter if steps are happening quickly enough
            step_count += 1
            print(f"Alternating step detected ({knee}), step count: {step_count}")
            
            # After 2 alternating steps, switch to continuous walking
            if step_count >= 2 and not walk_hold_active:
                pyautogui.keyDown('w')
                walk_hold_active = True
                print("Continuous walking activated: holding 'w' key")
            # If already walking, just update the last step time
            elif walk_hold_active:
                last_step_time = current_time
        else:
            # Too much time between steps, reset counter
            step_count = 1
            print(f"New walking sequence started with {knee} knee")
            
            # If continuous walking was active, release it
            if walk_hold_active:
                release_walk()
            
            # Execute a single walk step
            execute_single_walk()
    else:
        # First step or same knee again
        if last_knee_raised != knee or current_time - last_step_time > step_time_threshold:
            # New walking sequence or different knee after timeout
            step_count = 1
            execute_single_walk()
            print(f"Single step with {knee} knee")
    
    # Update state for next detection
    last_knee_raised = knee
    last_step_time = current_time

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
                time.sleep(0.01)
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

# If we have 2 quaternions being applied one after the other, we can represent it as one
# Combined rotation
def multiply_quat(q1, q2):

    # (w, x, y, z)

    q = (q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
         q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
         q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
         q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0])
    
    return q

# rm stands for rotation matrix
def find_quat_from_matrix(rm):

    m = np.array(rm)
    tr = m[0][0] + m[1][1] + m[2][2]

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m[2][1] - m[1][2]) / S
        qy = (m[0][2] - m[2][0]) / S
        qz = (m[1][0] - m[0][1]) / S

    elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
        S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2  # S=4*qx
        qw = (m[2][1] - m[1][2]) / S
        qx = 0.25 * S
        qy = (m[0][1] + m[1][0]) / S
        qz = (m[0][2] + m[2][0]) / S

    elif m[1][1] > m[2][2]:
        S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2  # S=4*qy
        qw = (m[0][2] - m[2][0]) / S
        qx = (m[0][1] + m[1][0]) / S
        qy = 0.25 * S
        qz = (m[1][2] + m[2][1]) / S

    else:
        S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2  # S=4*qz
        qw = (m[1][0] - m[0][1]) / S
        qx = (m[0][2] + m[2][0]) / S
        qy = (m[1][2] + m[2][1]) / S
        qz = 0.25 * S

    # Normalize the quaternion
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return (qw/norm, qx/norm, qy/norm, qz/norm)

def inverse_quat(q):
    return (q[0], -q[1], -q[2], -q[3])

# Since I'm working with xyz coordinates, 
def find_yaw_angle(rotation):
    global base_rotation

    # Relative rotation
    rr_quat = multiply_quat(find_quat_from_matrix(rotation), inverse_quat(find_quat_from_matrix(base_rotation)))
    w, x, y, z = rr_quat
    # Pitch (around Y) calculation
    sin_pitch = 2 * (w * y - x * z)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Ensure valid arcsin input
    pitch = math.asin(sin_pitch)
    return pitch

def find_pitch_angle(rotation):
    global base_rotation

    # Relative rotation
    rr_quat = multiply_quat(find_quat_from_matrix(rotation), inverse_quat(find_quat_from_matrix(base_rotation)))
    w, x, y, z = rr_quat
    # Roll (around X) calculation
    sin_roll = 2 * (w * x + y * z)
    sin_roll = np.clip(sin_roll, -1.0, 1.0)
    roll = math.asin(sin_roll)
    return roll

# -------------------- Camera Movement --------------------
def set_base_rotation(rotation):
    global base_rotation
    base_rotation = rotation

def handle_look(rotation):
    global base_rotation

    if base_rotation is None:
        return

    # If yaw angle > or < than some degree, turn left/right
    # If pitch angle > or < than some degree, turn up/down
    # See if you can do it as the same time..?

    yaw_angle = find_yaw_angle(rotation)
    pitch_angle = find_pitch_angle(rotation)

    if yaw_angle > math.radians(10):
        print("Turned head right!")
    elif yaw_angle < math.radians(-10):
        print("Turned head left!")

    print(f"Yaw angle: {yaw_angle}")
    
    # Check for nodding (yes)
    if pitch_angle > math.radians(10):
        print("Nodded down!")
    elif pitch_angle < math.radians(-10):
        print("Nodded up!")

    

# This function now only plays the sound for arms calibration.
# What I want it to do, is to find the middle of the body, and if the arms go above that threshold, then they can punch
# The reason I want to do this, is because when the user relaxes their hands, it counts as a punch, however that is incorrect, so I want to remove that
def calibrate_arms(current_wristL_z, current_wristR_z):
    playsound(r"C:\Users\blacb\Downloads\fart-with-reverb.mp3")
    print("Calibration of arms complete")

def find_horizontal_threshold(left_knee_y, offset=15):
    return left_knee_y - offset

# -------------------- Vosk Audio Setup --------------------
vosk_model_path = r"C:\Users\blacb\Downloads\vosk-model-en-us-0.22-lgraph\vosk-model-en-us-0.22-lgraph"
model = vosk.Model(vosk_model_path)

# These are the only words that will be recognised
words = ["recalibrate", "calibrate", "arms", "arm", "legs", "leg", "inventory", "open", "close", "opened", "closed", "head"]
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
                    
                # Draw the threshold line
                cv2.line(image, (0, knee_threshold), (frame.shape[1], knee_threshold), (255, 0, 255), 2)
                
                # Handle the walking based on knee positions
                handle_walking(left_knee[1], right_knee[1], knee_threshold)
                
                # Display walking status
                if walk_hold_active:
                    cv2.putText(image, "WALKING", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                elif left_knee_is_up or right_knee_is_up:
                    cv2.putText(image, "STEP", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # --------- Jumping Feature ----------
            if left_knee[1] < knee_threshold and right_knee[1] < knee_threshold:
                if jump_ready:
                    handle_jump_event()
            else:
                jump_ready = True
            
            # --------- Looking Feature ----------

            # Need normalised coordinates instead of pixel coordinates
            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            left_ear = np.array([left_ear.x, left_ear.y, left_ear.z])
            right_ear = np.array([right_ear.x, right_ear.y, right_ear.z])
            nose = np.array([nose.x, nose.y, nose.z])

            # This gives x plane
            ear_to_ear_vector = right_ear - left_ear

            # Normalises so that magnitude len is 1
            X_vector = ear_to_ear_vector / np.linalg.norm(ear_to_ear_vector)

            # This gives z plane
            midpoint = (left_ear + right_ear) / 2
            nose_to_mid_vector = midpoint - nose
            Z_vector = nose_to_mid_vector / np.linalg.norm(nose_to_mid_vector)

            # This gives y plane
            face_vector = np.cross(Z_vector, X_vector)
            Y_vector = face_vector / np.linalg.norm(face_vector)

            # Gives a more accurate, 100% perpendicular z plane 
            # Since the z plane we calculated from coordinates may have noise
            Z_vector = np.cross(X_vector, Y_vector)
            Z_vector = Z_vector / np.linalg.norm(Z_vector)

            rotation_matrix = np.column_stack((X_vector, Y_vector, Z_vector))
            
            # If the base rotation hasn't been set, set it
            if base_rotation is None:
                base_rotation = rotation_matrix
            
            handle_look(rotation_matrix)

        except Exception as e:
            print("Error processing pose:", e)

        if punch_hold_active and (time.time() - punch_last_time > CONSECUTIVE_THRESHOLD):
            pyautogui.mouseUp(button='left')
            punch_hold_active = False
            print("Consecutive punches ended: released left mouse button")

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # This handles all of the calibration
        if not command_queue.empty():
            command = command_queue.get()
            if command == "calibrate_arms":
                try:
                    calib_left = wristL[2]
                    calib_right = wristR[2]
                    threading.Thread(target=calibrate_arms, args=(calib_left, calib_right), daemon=True).start()
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