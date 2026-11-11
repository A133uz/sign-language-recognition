"""Average-rate capture budget; tolerate jitter without inventing frames."""
import math


class FrameRateLimiter:
    def __init__(self, fps=30.):
        if not math.isfinite(fps) or not 1 <= fps <= 60:
            raise ValueError('Capture FPS must be finite and between 1 and 60')
        self.fps = float(fps)
        self.interval = 1 / self.fps
        # Retain fractional credit instead of requiring each individual interval
        # to exceed 1/fps. Small arrival jitter must not halve a native 30-Hz feed.
        # Capacity two permits bounded jitter recovery; a long stall resets the
        # budget so it cannot generate a catch-up burst.
        self.credit = 1.
        self.last_seen = None
        self.accepted = self.dropped = 0

    def accept(self, seconds):
        if (not math.isfinite(seconds) or seconds < 0 or
                (self.last_seen is not None and seconds < self.last_seen)):
            raise ValueError('Capture times must be finite, nonnegative and ordered')
        if self.last_seen is not None:
            elapsed = seconds-self.last_seen
            self.credit = 1. if elapsed > 2*self.interval+1e-9 else min(2.,self.credit+elapsed*self.fps)
        self.last_seen = seconds
        # Up to 0.05 frame of temporary debt covers sub-millisecond phase jitter;
        # it is repaid by later arrivals, rather than resetting the budget.
        if self.credit + .05 + 1e-9 < 1.:
            self.dropped += 1
            return False
        self.credit -= 1.
        self.accepted += 1
        return True


def capture_rate_metrics(timestamps, source_frames, source_seconds, fps):
    count = len(timestamps)
    duration = (timestamps[-1] - timestamps[0]) / 1000 if count > 1 else 0.
    return {'capture_fps_limit': float(fps), 'source_frame_count': source_frames,
            'fps_limit_dropped_frames': source_frames - count,
            'source_capture_fps': (source_frames-1)/source_seconds if source_seconds > 0 and source_frames > 1 else None,
            'effective_capture_fps': (count-1)/duration if duration > 0 else None}
