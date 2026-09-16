import unittest
from analysis_timing import AnalysisTiming


class TimingTests(unittest.TestCase):
    def test_loading_is_not_batch_time(self):
        now=[0.0];timing=AnalysisTiming(lambda:now[0])
        now[0]=30;timing.start_detection()
        now[0]=32;timing.batch(1,10)
        self.assertIsNone(timing.snapshot('running',True)['detection_remaining_seconds'])
        now[0]=36;timing.batch(3,10)
        self.assertEqual(timing.snapshot('running',True),dict(elapsed_seconds=36,detection_remaining_seconds=14))
        now[0]=40
        self.assertEqual(timing.snapshot('running',True)['detection_remaining_seconds'],10)
        self.assertIsNone(timing.snapshot('stopping',True)['detection_remaining_seconds'])
        self.assertIsNone(timing.snapshot('running',False)['detection_remaining_seconds'])
        now[0]=55
        self.assertIsNone(timing.snapshot('running',True)['detection_remaining_seconds'])
        timing.batch(4,10)
        self.assertGreater(timing.snapshot('running',True)['detection_remaining_seconds'],0)
        timing.batch(10,10)
        self.assertIsNone(timing.snapshot('running',True)['detection_remaining_seconds'])

    def test_terminal_elapsed_stops_and_new_job_resets(self):
        now=[0.0];timing=AnalysisTiming(lambda:now[0]);now[0]=12
        self.assertEqual(timing.snapshot('complete',False)['elapsed_seconds'],12)
        now[0]=45
        self.assertEqual(timing.snapshot('complete',False)['elapsed_seconds'],12)
        fresh=AnalysisTiming(lambda:now[0])
        self.assertEqual(fresh.snapshot('running',False)['elapsed_seconds'],0)


if __name__=='__main__':unittest.main()
