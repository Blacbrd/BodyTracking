# ---- Punching ----
SMALL_ANGLE_THRESHOLD = 45    # Degrees: elbow is bent (ready for a punch)
LARGE_ANGLE_THRESHOLD = 120   # Degrees: elbow is extended (punch thrown)
DISPLAY_FRAMES = 15           # Number of frames to display the "Punch!" label
CONSECUTIVE_PUNCH_THRESHOLD = 0.5 # Amount of seconds. If 2 punches fall within this time frame, the program takes it as mining
KNEE_OFFSET = 22    # Amount of pixels the purple line appears above your knees

# ---- Walking ----
WALK_THRESHOLD_ENABLED = True    # Set to False to disable the threshold line feature.
CONSECUTIVE_WALK_THRESHOLD = 0.7 # If both knees are raised between the CONSECUTIVE_WALK_THRESHOLD, then continuously walk
STEP_TIME_THRESHOLD = 1.0        # Timing for steps

# Depending on which hand is dominant depends what each of them do
# Type "Right" or "Left"
DOMINANT_HAND = "Right"
NON_DOMINANT_HAND = "Left"

# ---- Inventory/hand model management ----
RECTANGLE_WIDTH = 200
RECTANGLE_HEIGHT = 100
