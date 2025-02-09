import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import json
import vosk
import pyaudio
from playsound import playsound

# -------------------- Global Setup --------------------

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calibration variables
calibrated = False
calib_wristL_z = None
calib_wristR_z = None

# --- Punch State Machine Variables ---
# For each arm, we use a "ready" flag that is True when the elbow is bent (small angle)
# and ready to register a punch once the elbow is extended.
left_ready = True
right_ready = True

# These thresholds define when an elbow is considered bent vs extended.
SMALL_ANGLE_THRESHOLD = 90    # Degrees: elbow is bent (ready for a punch)
LARGE_ANGLE_THRESHOLD = 160   # Degrees: elbow is extended (punch thrown)

# These counters are used for visual feedback.
left_punch_display_counter = 0
right_punch_display_counter = 0
DISPLAY_FRAMES = 15           # Number of frames to display the "Punch!" label

# Threshold for using the z-axis (if needed) can be set here.
Z_THRESHOLD_RATIO = 0.2      # (Not used in the state-machine logic below)

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
            
        if "calibrate" in text.lower():
            print("Voice command detected:", text)
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
            # If the elbow is bent (angle below small threshold), mark as ready.
            if angleL < SMALL_ANGLE_THRESHOLD:
                left_ready = True
            # If the elbow is extended (angle above large threshold) and the arm is ready,
            # register a punch and reset the ready flag.
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

            # (Optional) You could incorporate the z-axis change here as an additional criterion.
            # For example:
            # if calibrated:
            #     delta_z_left = wristL[2] - calib_wristL_z
            #     if left_ready and delta_z_left > abs(calib_wristL_z)*Z_THRESHOLD_RATIO:
            #         # Register left punch...
            #         pass

            # --------- Visual Feedback for Punches ----------
            # If a punch was recently detected (counter > 0), draw a label near the wrist.
            if left_punch_display_counter > 0:
                cv2.putText(image, "LEFT PUNCH!", (wristL[0]-50, wristL[1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                left_punch_display_counter -= 1

            if right_punch_display_counter > 0:
                cv2.putText(image, "RIGHT PUNCH!", (wristR[0]-50, wristR[1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                right_punch_display_counter -= 1

            # (Optional) Show the delta z values for debugging (if calibration is used)
            if calibrated:
                delta_z_left = wristL[2] - calib_wristL_z
                delta_z_right = wristR[2] - calib_wristR_z
                cv2.putText(image, f'dZ_L: {delta_z_left:.2f}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'dZ_R: {delta_z_right:.2f}', (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error processing pose:", e)

        # Draw pose landmarks.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if any command has been queued.
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

        cv2.imshow("Body Recognition", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
p.terminate()
