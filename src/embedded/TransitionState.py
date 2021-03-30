from datetime import datetime
from typing import Optional


class TransitionState:
    """A utility object for tracking transition states across calls to DataCoordinator.finalize_predictions()"""
    start_time: Optional[datetime]
    num_consecutive_ones: Optional[int]

    def __init__(self, start_time: Optional[datetime], num_consecutive_ones: Optional[int]):
        self.start_time = start_time
        self.num_consecutive_ones = num_consecutive_ones

    def is_detected(self):
        return self.start_time is not None


def non_detected_transition_state():
    return TransitionState(None, None)
