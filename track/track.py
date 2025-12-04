import numpy as np
from collections import deque
from .kalman_filter import KalmanFilterSimple

def xyah_to_xyxy(xyah):
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return np.array([x1, y1, x2, y2], dtype=float)

class Track:
    def __init__(self, mean, covariance, track_id, n_init=3, max_age=30, score=0.0):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'Tentative'  # or 'Confirmed', 'Deleted'
        self.n_init = n_init
        self.max_age = max_age
        self.score = score
        self.kf = KalmanFilterSimple()
        self.history = deque(maxlen=30)

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.to_tlbr())

    def update(self, detection_measurement, score=None):
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection_measurement)
        self.hits += 1
        self.time_since_update = 0
        if score is not None:
            self.score = score
        if self.state == 'Tentative' and self.hits >= self.n_init:
            self.state = 'Confirmed'

    def mark_missed(self):
        if self.state == 'Tentative':
            self.state = 'Deleted'
        elif self.time_since_update > self.max_age:
            self.state = 'Deleted'

    def is_deleted(self):
        return self.state == 'Deleted'

    def is_confirmed(self):
        return self.state == 'Confirmed'

    def to_tlbr(self):
        return xyah_to_xyxy(self.mean[:4])
