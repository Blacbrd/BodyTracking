import cv2
import numpy as np
import threading
import time
import pyautogui
from pynput.mouse import Controller
import mediapipe as mp

from configurables import (SMALL_ANGLE_THRESHOLD, LARGE_ANGLE_THRESHOLD, DISPLAY_FRAMES,
                        CONSECUTIVE_PUNCH_THRESHOLD, KNEE_OFFSET, RECTANGLE_WIDTH, RECTANGLE_HEIGHT,
                        DOMINANT_HAND, WALK_THRESHOLD_ENABLED)
import utils

mp_drawing = mp.solutions.drawing_utils
mp_pose_body = mp.solutions.pose
mp_hands = mp.solutions.hands

mouse = Controller()
pyautogui.FAILSAFE = False

# Helpers using dependency-injected state

def handle_punch(state):
    """Schedules single click or holds mouse depending on consecutive punches.
        Uses nested function to capture 'state' in the Timer closure."""
    now = time.time()

    def execute_single_punch():
        # closure uses state
        with state.lock:
            if not state.punch_hold_active:
                pyautogui.click(button='left')
                print("Single punch executed: left click")

            state.punch_timer = None

    with state.lock:
        if state.punch_timer is None:
            state.punch_last_time = now
            state.punch_timer = threading.Timer(CONSECUTIVE_PUNCH_THRESHOLD, execute_single_punch)
            state.punch_timer.daemon = True
            state.punch_timer.start()

        else:
            # second punch within threshold
            state.punch_timer.cancel()
            state.punch_timer = None

            if not state.punch_hold_active:
                pyautogui.mouseDown(button='left')
                state.punch_hold_active = True
                print("Consecutive punches detected: holding left mouse button for mining")

            state.punch_last_time = now

def release_walk(state):
    with state.lock:
        if state.walk_hold_active:
            pyautogui.keyUp('w')
            print("Walking ended: 'w' key released")
            state.walk_hold_active = False

def execute_single_walk(state):
    with state.lock:
        if not state.walk_hold_active:
            pyautogui.keyDown('w')
            print("Single walk step: 'w' key held for 0.2 seconds")
            state.walk_hold_active = True
            t = threading.Timer(0.2, lambda: release_walk(state))
            t.daemon = True
            t.start()

        state.walk_timer = None

def handle_knee_step(state, knee, current_time):
    # No lock for read-mostly; lock around writes that must be atomic
    if state.last_knee_raised is not None and state.last_knee_raised != knee:
        time_diff = current_time - state.last_step_time

        if time_diff < state.step_time_threshold:
            state.step_count += 1
            print(f"Alternating step detected ({knee}), step count: {state.step_count}")

            if state.step_count >= 2 and not state.walk_hold_active:
                pyautogui.keyDown('w')
                state.walk_hold_active = True
                print("Continuous walking activated: holding 'w' key")

            elif state.walk_hold_active:
                state.last_step_time = current_time

        else:
            state.step_count = 1
            print(f"New walking sequence started with {knee} knee")
            if state.walk_hold_active:
                release_walk(state)

            execute_single_walk(state)

    else:
        if state.last_knee_raised != knee or current_time - state.last_step_time > state.step_time_threshold:
            state.step_count = 1
            execute_single_walk(state)
            print(f"Single step with {knee} knee")

    state.last_knee_raised = knee
    state.last_step_time = current_time

def handle_walking(state, left_knee_position, right_knee_position, knee_threshold):
    state.left_knee_is_up = left_knee_position < knee_threshold
    state.right_knee_is_up = right_knee_position < knee_threshold
    current_time = time.time()

    if state.left_knee_is_up and not state.left_knee_was_up:
        handle_knee_step(state, "left", current_time)

    elif state.right_knee_is_up and not state.right_knee_was_up:
        handle_knee_step(state, "right", current_time)

    if not state.left_knee_is_up and not state.right_knee_is_up:
        state.left_knee_was_up = False
        state.right_knee_was_up = False

    state.left_knee_was_up = state.left_knee_is_up
    state.right_knee_was_up = state.right_knee_is_up

    if state.walk_hold_active and current_time - state.last_step_time > state.step_time_threshold:
        release_walk(state)
        state.step_count = 0
        print("Walking timeout detected, released walk key")

def reset_crouch(state):
    with state.lock:
        state.crouch_ready = True

def execute_jump():
    pyautogui.keyDown('space')
    time.sleep(0.1)
    pyautogui.keyUp('space')

def handle_jump_event(state):
    if state.jump_ready:
        if not state.walk_hold_active:

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

        state.jump_ready = False

def find_horizontal_threshold(left_knee_y, offset=KNEE_OFFSET):
    return left_knee_y - offset

def set_index_finger_pos(state, x, y):
    state.reference_index_finger_x = x
    state.reference_index_finger_y = y

def calibrate_arms(current_wristL_z, current_wristR_z):
    print("Calibration of arms complete")

# Body tracking (mirrors original logic, takes state)
def body_tracking(pose, frame, state):
    results = pose.process(frame)
    frame.flags.writeable = True
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        # If no landmarks available, results.pose_landmarks may be None
        if results.pose_landmarks is None:
            raise ValueError("No pose landmarks")

        landmarks = results.pose_landmarks.landmark

        def get_landmark_point(landmark):
            return [int(landmark.x * image.shape[1]),
                    int(landmark.y * image.shape[0]),
                    landmark.z]

        shoulderL = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.LEFT_SHOULDER])
        elbowL    = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.LEFT_ELBOW])
        wristL    = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.LEFT_WRIST])
        shoulderR = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.RIGHT_SHOULDER])
        elbowR    = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.RIGHT_ELBOW])
        wristR    = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.RIGHT_WRIST])
        hipL = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.LEFT_HIP])
        hipR = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.RIGHT_HIP])
        kneeL = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.LEFT_KNEE])
        kneeR = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.RIGHT_KNEE])
        nose = get_landmark_point(landmarks[mp_pose_body.PoseLandmark.NOSE])

        mid_point_shoulders = utils.calculate_midpoint(shoulderL, shoulderR)
        mid_point_hips = utils.calculate_midpoint(hipL, hipR)
        mid_point_knees = utils.calculate_midpoint(kneeL, kneeR)

        angle_back = utils.calculate_angle(mid_point_shoulders, mid_point_hips, mid_point_knees)
        angleL = int(utils.calculate_angle(shoulderL, elbowL, wristL))
        angleR = int(utils.calculate_angle(shoulderR, elbowR, wristR))

        cv2.putText(image, f'{angleL}', (elbowL[0] - 30, elbowL[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image, f'{angleR}', (elbowR[0] - 30, elbowR[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Punch detection
        if angleL < SMALL_ANGLE_THRESHOLD:
            state.left_ready = True

        if state.left_ready and angleL > LARGE_ANGLE_THRESHOLD:
            print("Left punch detected!")
            state.left_ready = False
            state.left_punch_display_counter = DISPLAY_FRAMES
            handle_punch(state)

        if angleR < SMALL_ANGLE_THRESHOLD:
            state.right_ready = True

        if state.right_ready and angleR > LARGE_ANGLE_THRESHOLD:
            print("Right punch detected!")
            state.right_ready = False
            state.right_punch_display_counter = DISPLAY_FRAMES
            handle_punch(state)

        # Block placement via wrist crossing
        if wristR[0] > wristL[0]:
            if state.placement_ready:
                pyautogui.click(button='right')
                print("Block placed via wrist crossing!")
                state.placement_ready = False

        else:
            state.placement_ready = True

        # Walking
        if WALK_THRESHOLD_ENABLED:
            if state.knee_threshold is None:
                state.knee_threshold = find_horizontal_threshold(kneeL[1])
            cv2.line(image, (0, state.knee_threshold), (image.shape[1], state.knee_threshold), (255, 0, 255), 2)
            handle_walking(state, kneeL[1], kneeR[1], state.knee_threshold)

            if state.walk_hold_active:
                cv2.putText(image, "WALKING", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

            elif state.left_knee_is_up or state.right_knee_is_up:
                cv2.putText(image, "STEP", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Jumping
        if kneeL[1] < state.knee_threshold and kneeR[1] < state.knee_threshold:
            if state.jump_ready:
                handle_jump_event(state)

        else:
            state.jump_ready = True

        # Crouch
        if angle_back < 145 and state.crouch_ready:
            pyautogui.keyDown("c")
            time.sleep(0.1)
            pyautogui.keyUp("c")
            state.crouch_ready = False
            t = threading.Timer(1.0, lambda: reset_crouch(state))
            t.daemon = True
            t.start()

        # Inventory open via wrist near face
        dist_wristR_face = np.hypot(wristR[0] - nose[0], wristR[1] - nose[1])
        dist_wristL_face = np.hypot(wristL[0] - nose[0], wristL[1] - nose[1])
        if dist_wristL_face <= 15 or dist_wristR_face <= 15:
            pyautogui.keyDown("e")
            time.sleep(0.1)
            pyautogui.keyUp("e")
            return "Hand", image

        # Looking feature - compute normalized vectors
        left_ear_lm = results.pose_landmarks.landmark[mp_pose_body.PoseLandmark.LEFT_EAR]
        right_ear_lm = results.pose_landmarks.landmark[mp_pose_body.PoseLandmark.RIGHT_EAR]
        nose_lm = results.pose_landmarks.landmark[mp_pose_body.PoseLandmark.NOSE]

        left_ear = np.array([left_ear_lm.x, left_ear_lm.y, left_ear_lm.z])
        right_ear = np.array([right_ear_lm.x, right_ear_lm.y, right_ear_lm.z])
        nose_n = np.array([nose_lm.x, nose_lm.y, nose_lm.z])

        ear_to_ear_vector = right_ear - left_ear
        X_vector = ear_to_ear_vector / np.linalg.norm(ear_to_ear_vector)
        earMidpoint = (left_ear + right_ear) / 2
        nose_to_mid_vector = earMidpoint - nose_n
        Z_vector = nose_to_mid_vector / np.linalg.norm(nose_to_mid_vector)
        face_vector = np.cross(Z_vector, X_vector)
        Y_vector = face_vector / np.linalg.norm(face_vector)
        Z_vector = np.cross(X_vector, Y_vector)
        Z_vector = Z_vector / np.linalg.norm(Z_vector)

        rotation_matrix = np.column_stack((X_vector, Y_vector, Z_vector))

        if state.base_rotation is None:
            state.base_rotation = rotation_matrix

        # mouse movement based on yaw/pitch
        yaw_angle = utils.find_yaw_angle(rotation_matrix, state.base_rotation)
        pitch_angle = utils.find_pitch_angle(rotation_matrix, state.base_rotation)
        MOUSE_MOVEMENT = 20

        if yaw_angle > np.radians(45):
            mouse.move(MOUSE_MOVEMENT, 0)
            print("Turned head right!")

        elif yaw_angle < np.radians(-40):
            mouse.move(-MOUSE_MOVEMENT, 0)
            print("Turned head left!")

        elif pitch_angle > np.radians(5):
            mouse.move(0, MOUSE_MOVEMENT)
            print("Nodded down!")

        elif pitch_angle < np.radians(-3):
            mouse.move(0, -MOUSE_MOVEMENT)
            print("Nodded up!")

    except Exception as e:
        print("Error processing pose:", e)

    # Release mouse if consecutive punching ended
    if state.punch_hold_active and (time.time() - state.punch_last_time > CONSECUTIVE_PUNCH_THRESHOLD):
        pyautogui.mouseUp(button='left')
        state.punch_hold_active = False
        print("Consecutive punches ended: released left mouse button")

    # Draw pose landmarks if present
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose_body.POSE_CONNECTIONS)

    # Handle queued commands
    if not state.command_queue.empty():
        command = state.command_queue.get()

        if command == "calibrate_arms":

            try:
                calib_left = wristL[2]
                calib_right = wristR[2]
                threading.Thread(target=calibrate_arms, args=(calib_left, calib_right), daemon=True).start()

            except Exception as e:
                print("Calibration error:", e)

        elif command == "calibrate_legs":

            try:

                if kneeL is not None:
                    state.knee_threshold = find_horizontal_threshold(kneeL[1])
                    print("Knee threshold recalibrated. New threshold:", state.knee_threshold)

                elif kneeR is not None:
                    state.knee_threshold = find_horizontal_threshold(kneeR[1])
                    print("Knee threshold recalibrated. New threshold:", state.knee_threshold)

                else:
                    print("Knee landmarks not detected. Cannot recalibrate threshold.")

            except Exception as e:
                print("Recalibration error:", e)

        elif command == "calibrate_head":

            try:
                state.base_rotation = rotation_matrix
                print(f"Successfully calibrated head, new rotation: {state.base_rotation}")

            except Exception as e:
                print("Calibration error:", e)

        elif command == "change_hand":
            return "Hand", image

    return "Body", image

# Hand tracking (takes state)
def hand_tracking(hands, frame, state):
    results = hands.process(frame)
    frame.flags.writeable = True
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    index_finger = None
    non_dominant_wrist = None
    non_dominant_thumb = None
    non_dominant_index = None
    non_dominant_middle = None
    non_dominant_middle_base = None

    if results.multi_hand_landmarks and results.multi_handedness:
        h, w, _ = image.shape

        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
            ):

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

            for idx, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.putText(image, str(idx), (px - 10, py + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        label = hand_handedness.classification[0].label

        if label == DOMINANT_HAND:
            index_finger = results.multi_hand_landmarks[0].landmark[8]

            if len(results.multi_hand_landmarks) > 1:
                non_dominant_wrist = results.multi_hand_landmarks[1].landmark[0]
                non_dominant_thumb = results.multi_hand_landmarks[1].landmark[4]
                non_dominant_index = results.multi_hand_landmarks[1].landmark[8]
                non_dominant_middle_base = results.multi_hand_landmarks[1].landmark[9]
                non_dominant_middle = results.multi_hand_landmarks[1].landmark[12]

        else:

            if len(results.multi_hand_landmarks) > 1:
                index_finger = results.multi_hand_landmarks[1].landmark[8]
                non_dominant_wrist = results.multi_hand_landmarks[0].landmark[0]
                non_dominant_thumb = results.multi_hand_landmarks[0].landmark[4]
                non_dominant_index = results.multi_hand_landmarks[0].landmark[8]
                non_dominant_middle_base = results.multi_hand_landmarks[0].landmark[9]
                non_dominant_middle = results.multi_hand_landmarks[0].landmark[12]

            else:
                index_finger = results.multi_hand_landmarks[0].landmark[8]

        if index_finger:
            cur_index_finger_x = int(index_finger.x * w)
            cur_index_finger_y = int(index_finger.y * h)

        else:
            cur_index_finger_x = None
            cur_index_finger_y = None

        if non_dominant_wrist:
            non_dominant_wrist_x = int(non_dominant_wrist.x * w)
            non_dominant_wrist_y = int(non_dominant_wrist.y * h)

        else:
            non_dominant_wrist_x = non_dominant_wrist_y = None

        if non_dominant_thumb:
            non_dominant_thumb_x = int(non_dominant_thumb.x * w)
            non_dominant_thumb_y = int(non_dominant_thumb.y * h)

        else:
            non_dominant_thumb_x = non_dominant_thumb_y = None

        if non_dominant_index:
            non_dominant_index_x = int(non_dominant_index.x * w)
            non_dominant_index_y = int(non_dominant_index.y * h)

        else:
            non_dominant_index_x = non_dominant_index_y = None

        if non_dominant_middle:
            non_dominant_middle_x = int(non_dominant_middle.x * w)
            non_dominant_middle_y = int(non_dominant_middle.y * h)

        else:
            non_dominant_middle_x = non_dominant_middle_y = None

        if non_dominant_middle_base:
            non_dominant_middle_base_x = int(non_dominant_middle_base.x * w)
            non_dominant_middle_base_y = int(non_dominant_middle_base.y * h)

        else:
            non_dominant_middle_base_x = non_dominant_middle_base_y = None

        if state.reference_index_finger_x is None or state.reference_index_finger_y is None:
            state.reference_index_finger_x = cur_index_finger_x
            state.reference_index_finger_y = cur_index_finger_y

        rx1 = state.reference_index_finger_x - RECTANGLE_WIDTH // 2
        ry1 = state.reference_index_finger_y - RECTANGLE_HEIGHT // 2

        if rx1 < 0:
            rx1 = 0

        elif rx1 + RECTANGLE_WIDTH > w:
            rx1 = w - RECTANGLE_WIDTH

        if ry1 < 0:
            ry1 = 0

        elif ry1 + RECTANGLE_HEIGHT > h:
            ry1 = h - RECTANGLE_HEIGHT

        rx2 = rx1 + RECTANGLE_WIDTH
        ry2 = ry1 + RECTANGLE_HEIGHT

        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        multiply_x = state.MONITOR_WIDTH / RECTANGLE_WIDTH
        multiply_y = state.MONITOR_HEIGHT / RECTANGLE_HEIGHT

        if cur_index_finger_x is not None and rx1 <= cur_index_finger_x <= rx2 and ry1 <= cur_index_finger_y <= ry2:
            mouse_x = int((rx2 - cur_index_finger_x) * multiply_x)
            mouse_y = int((cur_index_finger_y - ry1) * multiply_y)
            mouse_x += state.MONITOR_WIDTH_OFFSET
            mouse.position = (mouse_x, mouse_y)

        dist_thumb_index = dist_thumb_middle = reference_distance = dist_two_hands = None

        if (non_dominant_index_x is not None and non_dominant_middle_x is not None
                and non_dominant_thumb_x is not None and non_dominant_middle_base_x is not None):

            dist_thumb_index = np.hypot(non_dominant_thumb_x - non_dominant_index_x, non_dominant_thumb_y - non_dominant_index_y)
            dist_thumb_middle = np.hypot(non_dominant_thumb_x - non_dominant_middle_x, non_dominant_thumb_y - non_dominant_middle_y)
            reference_distance = np.hypot(non_dominant_wrist_x - non_dominant_middle_base_x, non_dominant_wrist_y - non_dominant_middle_base_y)

            if cur_index_finger_x and cur_index_finger_y:
                dist_two_hands = np.hypot(non_dominant_index_x - cur_index_finger_x, non_dominant_index_y - cur_index_finger_y)

        if dist_two_hands is not None and dist_two_hands < 10:
            pyautogui.keyDown("e")
            time.sleep(0.1)
            pyautogui.keyUp("e")
            return "Body", image

        if dist_thumb_index is not None and dist_thumb_middle is not None and reference_distance is not None:

            if dist_thumb_index > reference_distance * 0.2:
                state.can_press_index = True

            if dist_thumb_middle > reference_distance * 0.2:
                state.can_press_middle = True

            if state.can_press_index and dist_thumb_index < 12:
                state.can_press_index = False
                pyautogui.click(button="left")

            if state.can_press_middle and dist_thumb_middle < 12:
                state.can_press_middle = False
                pyautogui.click(button="right")

    # Handle queued commands
    if not state.command_queue.empty():
        command = state.command_queue.get()

        if command == "change_body":
            return "Body", image

        elif command == "calibrate_box":
            if index_finger:
                set_index_finger_pos(state, int(index_finger.x * w), int(index_finger.y * h))

    return "Hand", image
