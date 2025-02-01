import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using the dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clip to prevent NaN errors
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  

    # Convert to degrees
    angle = np.degrees(angle)

    return angle


# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Pose estimation model
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolour to RGB as OpenCV uses BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolour back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Reduces need for repeat code
            def get_landmark_point(landmark):
                return [int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])]

            # Get coordinates of joints
            shoulderL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            elbowL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            wristL = get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

            shoulderR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            elbowR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            wristR = get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

            # Calculate angles
            angleL = int(calculate_angle(shoulderL, elbowL, wristL))
            angleR = int(calculate_angle(shoulderR, elbowR, wristR))

            print(f"Left: {angleL}°")
            print(f"Right: {angleR}°")

            # Display angles on screen
            cv2.putText(image, f'{angleL}', (elbowL[0] - 30, elbowL[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, f'{angleR}', (elbowR[0] - 30, elbowR[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Body Recognition", image)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
