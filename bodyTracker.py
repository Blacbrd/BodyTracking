import cv2
import mediapipe as mp
import numpy as np

# Gives us drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Pose estimation model
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:

    # Video feed
    while cap.isOpened():

        ret, frame = cap.read()

        # Recolour to RGB as openCV is BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolour back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Body Recognition", image)

        # If q pressed, end
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


cap.release()