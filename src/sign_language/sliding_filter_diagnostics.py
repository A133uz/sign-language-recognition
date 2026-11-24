"""Matched experimental filters; these do not alter the webcam recognizer."""

import math


class CandidateFilters:
    """Compare a fixed confidence threshold with two consecutive agreeing windows."""

    def __init__(self, threshold=.60, max_gap_seconds=2.0):
        if not 0 <= threshold <= 1 or max_gap_seconds <= 0:
            raise ValueError('Invalid filter settings')
        self.threshold = threshold
        self.max_gap_seconds = max_gap_seconds
        self.previous = None
        self.last_time = None

    def update(self, label, score, timestamp):
        if not label or not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(timestamp):
            raise ValueError('Invalid prediction')
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Prediction timestamps must increase')
        if self.last_time is not None and timestamp - self.last_time > self.max_gap_seconds:
            self.previous = None
        confident = label if score >= self.threshold else None
        stable = label if confident is not None and confident == self.previous else None
        self.previous = confident
        self.last_time = timestamp
        return {'raw': label, 'threshold': confident, 'threshold_and_stability': stable}

    def reset(self):
        self.previous = None
        self.last_time = None
