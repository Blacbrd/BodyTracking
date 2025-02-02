import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import json
import vosk
import pyaudio

# Set up for mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clips to reduce noise
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calibrate():
    print("Calibrating... Please stand still.")

# Set-up for vosk voice recognition
vosk_model_path = r"C:\Users\blacb\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
model = vosk.Model(vosk_model_path)

# Queue for audio data and commands
audio_queue = queue.Queue()
command_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status_flags):
    """
    This callback is called by PyAudio when new audio data is available.
    It puts the raw audio data into a thread-safe queue.
    """
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def speech_recognition_worker():
    """
    This worker thread reads audio data from audio_queue, feeds it to the Vosk recognizer,
    and if the word 'calibrate' is detected, puts a signal in command_queue.
    After triggering calibration, the recognizer's state is reset so that the calibrate
    keyword is cleared from the recognizer's buffer.
    """
    recognizer = vosk.KaldiRecognizer(model, 16000)
    while True:
        data = audio_queue.get()
        # Try to get a full result first.
        if recognizer.AcceptWaveform(data):
            result_json = recognizer.Result()
            result = json.loads(result_json)
            text = result.get("text", "")
        else:
            # Otherwise, use the partial result for faster feedback.
            partial_json = recognizer.PartialResult()
            result = json.loads(partial_json)
            text = result.get("partial", "")
            
        # Check for the calibrate command in the recognized text.
        if "calibrate" in text.lower():
            print("Voice command detected:", text)
            command_queue.put("calibrate")
            # Reset the recognizer's state to clear the accumulated audio.
            recognizer.Reset()

# Start PyAudio stream with a smaller buffer for faster processing
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000,  # Reduced buffer size for lower latency
                stream_callback=audio_callback)
stream.start_stream()

# Start the speech recognition worker thread
speech_thread = threading.Thread(target=speech_recognition_worker, daemon=True)
speech_thread.start()

# Set-up openCV
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

            # Gets x and y coordinates relative to the screen
            def get_landmark_point(landmark):
                return [int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])]

            shoulderL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            elbowL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            wristL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

            shoulderR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            elbowR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            wristR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

            angleL = int(calculate_angle(shoulderL, elbowL, wristL))
            angleR = int(calculate_angle(shoulderR, elbowR, wristR))

            cv2.putText(image, f'{angleL}', (elbowL[0] - 30, elbowL[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'{angleR}', (elbowR[0] - 30, elbowR[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error processing pose:", e)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if any command has been queued.
        if not command_queue.empty():
            command = command_queue.get()
            if command == "calibrate":
                threading.Thread(target=calibrate, daemon=True).start()

        cv2.imshow("Body Recognition", image)

        # Break the loop when 'q' is pressed.
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Clean up the audio stream
stream.stop_stream()
stream.close()
p.terminate()