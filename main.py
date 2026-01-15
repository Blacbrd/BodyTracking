import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import time
import cv2
import pygame
import screeninfo

from state import create_default_state
import audio
import tracking

load_dotenv()

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "")

# Initialise pygame mixer
pygame.mixer.init()

def main():
    state = create_default_state()

    monitors = screeninfo.get_monitors()
    if len(monitors) >= 2:
        monitor = monitors[1]
    else:
        monitor = monitors[0]

    state.MONITOR_WIDTH = monitor.width
    state.MONITOR_HEIGHT = monitor.height
    state.MONITOR_WIDTH_OFFSET = monitor.x
    state.MONITOR_CENTRE = (state.MONITOR_WIDTH // 2, state.MONITOR_HEIGHT // 2)

    print(f"Amount of monitors detected: {len(monitors)}")
    print(f"Monitor width: {state.MONITOR_WIDTH}")
    print(f"Monitor height: {state.MONITOR_HEIGHT}")

    # Start audio (vosk + pyaudio). This will spawn the speech recognition thread.
    p_audio, audio_stream = audio.start_audio(state, VOSK_MODEL_PATH)

    # Setup OpenCV capture
    cap = cv2.VideoCapture(0)

    with tracking.mp_pose_body.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose, \
        tracking.mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            if state.mode == "Body":
                state.mode, output = tracking.body_tracking(pose, image, state)

            else:
                state.mode, output = tracking.hand_tracking(hands, image, state)

            h, w = output.shape[:2]
            output_big = cv2.resize(output, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Minecraft body tracking", output_big)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    audio_stream.stop_stream()
    audio_stream.close()
    p_audio.terminate()

    sys.exit(0)

if __name__ == "__main__":
    main()
