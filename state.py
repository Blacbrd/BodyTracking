from dataclasses import dataclass, field
import queue
import threading
from typing import Optional, Tuple, Any

@dataclass
class State:
    # Punch state
    left_ready: bool = True
    right_ready: bool = True
    left_punch_display_counter: int = 0
    right_punch_display_counter: int = 0
    punch_last_time: float = 0.0
    punch_hold_active: bool = False
    punch_timer: Optional[threading.Timer] = None

    # Arm / crouch / jump / placement
    arm_threshold: Optional[float] = None
    crouch_ready: bool = True
    jump_ready: bool = True
    placement_ready: bool = True

    # Walking / knees
    knee_threshold: Optional[int] = None
    left_knee_was_up: bool = False
    right_knee_was_up: bool = False
    left_knee_is_up: bool = False
    right_knee_is_up: bool = False
    walk_last_time: float = 0.0
    walk_hold_active: bool = False
    walk_timer: Optional[threading.Timer] = None

    # Step tracking
    last_knee_raised: Optional[str] = None
    step_count: int = 0
    step_time_threshold: float = 1.0
    last_step_time: float = 0.0

    # Camera rotation
    base_rotation: Optional[Any] = None

    # Mode
    mode: str = "Body"

    # Monitor info (set by main)
    MONITOR_WIDTH: Optional[int] = None
    MONITOR_HEIGHT: Optional[int] = None
    MONITOR_WIDTH_OFFSET: Optional[int] = None
    MONITOR_CENTRE: Optional[Tuple[int,int]] = None

    # Reference finger position
    reference_index_finger_x: Optional[int] = None
    reference_index_finger_y: Optional[int] = None

    # Finger press debounce
    can_press_index: bool = True
    can_press_middle: bool = True

    # Queues for audio threading
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    command_queue: queue.Queue = field(default_factory=queue.Queue)

    # Lock for thread-safety on writes
    lock: threading.Lock = field(default_factory=threading.Lock)

def create_default_state() -> State:
    return State()
