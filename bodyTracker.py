import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import json
import vosk
import pyaudio
from playsound import playsound

# Comment for git commit

# -------------------- Global Setup --------------------

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calibration variables for wrist depth
calibrated = False
calib_wristL_z = None
calib_wristR_z = None

# --- Punch State Machine Variables ---
left_ready = True
right_ready = True

SMALL_ANGLE_THRESHOLD = 45    # Degrees: elbow is bent (ready for a punch)
LARGE_ANGLE_THRESHOLD = 120   # Degrees: elbow is extended (punch thrown)

left_punch_display_counter = 0
right_punch_display_counter = 0
DISPLAY_FRAMES = 15           # Number of frames to display the "Punch!" label

# Threshold for using the z-axis (if needed) can be set here.
Z_THRESHOLD_RATIO = 0.2      # (Not used in the state-machine logic below)

# --- Walking (Forward Movement) Variables ---
walk_ready = True           # Flag so a new step is only triggered once the knees have dropped below the line.
WALK_DURATION_FRAMES = 15   # Duration (in frames) to display the "walking" feedback.
walk_display_counter = 0    # Counter for visual feedback.
WALK_THRESHOLD_ENABLED = True  # <--- Set to False (or comment out the block below) to disable the threshold line feature.
knee_threshold = None       # This will store the y coordinate of the horizontal threshold line.

# -------------------- Utility Functions --------------------

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle (ignoring z) at point b given three points a, b, and c.
    """
    a2d = np.array(a[:2])
    b2d = np.array(b[:2])
    c2d = np.array(c[:2])
    ba = a2d - b2d
    bc = c2d - b2d
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calibrate(current_wristL_z, current_wristR_z):
    playsound(r"C:\Users\blacb\Downloads\fart-with-reverb.mp3")
    global calibrated, calib_wristL_z, calib_wristR_z
    calib_wristL_z = current_wristL_z
    calib_wristR_z = current_wristR_z
    calibrated = True
    print(f"Calibration complete. Left wrist z: {calib_wristL_z}, Right wrist z: {calib_wristR_z}")

# This is where you can set how high the line is
def find_horizontal_threshold(left_knee_y, offset=25):
    """
    Calculates a horizontal threshold line based on the left knee's y coordinate.
    The threshold is shifted up (i.e. a smaller y value) by the specified offset.
    """
    return left_knee_y - offset

# -------------------- Vosk Audio Setup --------------------

vosk_model_path = r"C:\Users\blacb\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
model = vosk.Model(vosk_model_path)

audio_queue = queue.Queue()
command_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status_flags):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def speech_recognition_worker():
    recognizer = vosk.KaldiRecognizer(model, 16000)
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
            
        # Check for "recalibrate" first so that it doesn’t get caught by the "calibrate" check.
        if "recalibrate" in text.lower() and ("leg" in text.lower() or "legs" in text.lower() or "feet" in text.lower() or "one" in text.lower()):
            print("Voice command detected: recalibrate")
            command_queue.put("recalibrate")
            recognizer.Reset()
        if "recalibrate" in text.lower() and ("hand" in text.lower() or "hands" in text.lower() or "arm" in text.lower() or "arms" in text.lower() or "two" in text.lower()):
            print("Voice command detected: calibrate")
            command_queue.put("calibrate")
            recognizer.Reset()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000,
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

        # Convert frame to RGB for Mediapipe processing
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

            # Display the elbow angles for debugging.
            cv2.putText(image, f'{angleL}', (elbowL[0] - 30, elbowL[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'{angleR}', (elbowR[0] - 30, elbowR[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --------- Punch Detection State Machine ----------

            # For left arm:
            if angleL < SMALL_ANGLE_THRESHOLD:
                left_ready = True
            if left_ready and angleL > LARGE_ANGLE_THRESHOLD:
                print("Left punch detected!")
                left_ready = False
                left_punch_display_counter = DISPLAY_FRAMES

            # For right arm:
            if angleR < SMALL_ANGLE_THRESHOLD:
                right_ready = True
            if right_ready and angleR > LARGE_ANGLE_THRESHOLD:
                print("Right punch detected!")
                right_ready = False
                right_punch_display_counter = DISPLAY_FRAMES

            # (Optional) Show the delta z values for debugging (if calibration is used)
            if calibrated:
                delta_z_left = wristL[2] - calib_wristL_z
                delta_z_right = wristR[2] - calib_wristR_z
                cv2.putText(image, f'dZ_L: {delta_z_left:.2f}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'dZ_R: {delta_z_right:.2f}', (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            # --------- Walking (Forward Movement) Feature ----------
            # This entire section is controlled by WALK_THRESHOLD_ENABLED.
            # To disable the walking feature, you can set WALK_THRESHOLD_ENABLED = False.
            #
            # When both knees are detected, the function find_horizontal_threshold() is used to
            # calculate a horizontal line (based on the left knee's y coordinate, shifted upward by an offset).
            # Every time either knee moves above this line (i.e. y coordinate becomes less than the threshold),
            # and if walk_ready is True, the player will "move forwards" for one second (simulated here with
            # a print and visual feedback). The knees then have to drop back below the line (i.e. y becomes greater)
            # before a new forward move is registered.

            # Get knee landmarks (landmark indices 25 and 26)
            left_knee = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])

            if WALK_THRESHOLD_ENABLED:
                # Initialize the knee threshold if not already set.
                if knee_threshold is None:
                    knee_threshold = find_horizontal_threshold(left_knee[1])
                # Draw the horizontal threshold line. (Comment out this block to disable drawing.)
                cv2.line(image, (0, knee_threshold), (frame.shape[1], knee_threshold), (255, 0, 255), 2)

                # Check if either knee has moved above the threshold.
                if walk_ready and (left_knee[1] < knee_threshold or right_knee[1] < knee_threshold):
                    print("Walking forwards!")
                    walk_display_counter = WALK_DURATION_FRAMES
                    walk_ready = False

                # Reset the walk_ready flag when both knees are below (under) the threshold.
                if left_knee[1] > knee_threshold and right_knee[1] > knee_threshold:
                    walk_ready = True

                # Display visual feedback for walking.
                if walk_display_counter > 0:
                    cv2.putText(image, "WALKING FORWARDS", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    walk_display_counter -= 1

        except Exception as e:
            print("Error processing pose:", e)

        # Draw pose landmarks.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --------- Voice Command Handling ----------
        if not command_queue.empty():
            command = command_queue.get()
            if command == "calibrate":
                try:
                    # Calibrate using the current wrist positions.
                    calib_left = wristL[2]
                    calib_right = wristR[2]
                    threading.Thread(target=calibrate, args=(calib_left, calib_right), daemon=True).start()
                except Exception as e:
                    print("Calibration error:", e)
            elif command == "recalibrate":
                try:
                    # Recalibrate the knee threshold using the current left knee y coordinate.
                    # (If the knees are not visible, the threshold will remain None and nothing is drawn.)
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
