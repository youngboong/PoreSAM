"""Live SAM batch timing, separate from model loading and final output writing."""
import math
import time


class AnalysisTiming:
    def __init__(self,clock=time.monotonic):
        self.clock=clock
        self.started=clock()
        self.finished=None
        self.detection_started=None
        self.estimated_end=None

    def start_detection(self):
        if self.detection_started is None:self.detection_started=self.clock()

    def batch(self,completed,total):
        if self.detection_started is not None and 3<=completed<total:
            now=self.clock()
            per_batch=(now-self.detection_started)/completed
            self.estimated_end=now+per_batch*(total-completed)
        elif completed>=total:self.estimated_end=None

    def snapshot(self,status,detecting):
        now=self.clock()
        if status in ('complete','failed','cancelled') and self.finished is None:self.finished=now
        elapsed=max(0,(self.finished if self.finished is not None else now)-self.started)
        remaining=None
        if status=='running' and detecting and self.estimated_end is not None and self.estimated_end>now:
            remaining=math.ceil(self.estimated_end-now)
        return dict(elapsed_seconds=int(elapsed),detection_remaining_seconds=remaining)
