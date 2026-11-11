import math
import unittest

from sign_language.frame_rate import FrameRateLimiter, capture_rate_metrics


class FrameRateTests(unittest.TestCase):
    def test_fast_stream_is_sampled_without_duplicate_frames_or_falsified_time(self):
        limiter=FrameRateLimiter(30)
        times=[i/60 for i in range(121)]
        selected=[t for t in times if limiter.accept(t)]
        self.assertEqual(len(selected),61)
        self.assertEqual(selected,times[::2])
        metrics=capture_rate_metrics([round(t*1000) for t in selected],len(times),times[-1],30)
        self.assertEqual(metrics['effective_capture_fps'],30)
        self.assertEqual(metrics['source_capture_fps'],60)
        self.assertEqual(metrics['fps_limit_dropped_frames'],60)

    def test_jittery_30_fps_stream_is_not_accidentally_halved(self):
        limiter=FrameRateLimiter(30)
        times=[i/30+.0007*math.sin(i) for i in range(301)]
        self.assertTrue(all(limiter.accept(t) for t in times))

    def test_windows_quantized_clock_does_not_halve_native_30fps(self):
        limiter=FrameRateLimiter(30)
        times=[math.floor((i/30)/.015625)*.015625 for i in range(301)]
        kept=sum(limiter.accept(t) for t in times)
        self.assertGreaterEqual(kept,300)

    def test_larger_camera_jitter_keeps_nearly_all_native_frames(self):
        limiter=FrameRateLimiter(30)
        times=[i/30+.004*math.sin(i) for i in range(301)]
        self.assertGreaterEqual(sum(limiter.accept(t) for t in times),299)

    def test_slow_camera_retains_every_frame_without_upsampling(self):
        limiter=FrameRateLimiter(30)
        times=[i/15 for i in range(46)]
        self.assertTrue(all(limiter.accept(t) for t in times))
        self.assertEqual(limiter.accepted,len(times))
        self.assertEqual(limiter.dropped,0)

    def test_stall_cannot_create_a_catchup_burst(self):
        limiter=FrameRateLimiter(30)
        self.assertTrue(limiter.accept(0))
        self.assertTrue(limiter.accept(2))
        self.assertFalse(limiter.accept(2.001))
        self.assertFalse(limiter.accept(2.002))
        self.assertTrue(limiter.accept(2+1/30))

    def test_new_capture_resets_the_sampling_budget(self):
        first=FrameRateLimiter(30);first.accept(10)
        second=FrameRateLimiter(30)
        self.assertTrue(second.accept(0))

    def test_invalid_rates_and_time_order_are_rejected(self):
        for fps in (0,61,float('nan'),float('inf')):
            with self.assertRaises(ValueError):FrameRateLimiter(fps)
        limiter=FrameRateLimiter(30);limiter.accept(1)
        for now in (.9,-1,float('nan')):
            with self.assertRaises(ValueError):limiter.accept(now)


if __name__=='__main__':unittest.main()
