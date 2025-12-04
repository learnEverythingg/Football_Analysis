import numpy as np
from .track import Track
from .kalman_filter import KalmanFilterSimple
from scipy.optimize import linear_sum_assignment

def xyxy_to_xyah(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=float)

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0

def iou_cost(tracks, detections):
    C = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        tb = t.to_tlbr()
        for j, d in enumerate(detections):
            C[i, j] = 1 - iou(tb, d['bbox'])
    return C

def min_cost_matching(cost, thresh=0.7):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row, col = linear_sum_assignment(cost)
    matches, u_t, u_d = [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    for r, c in zip(row, col):
        if cost[r, c] <= 1 - thresh:
            matches.append((r, c))
            u_t.remove(r)
            u_d.remove(c)
    return matches, u_t, u_d

class ByteTrack:
    def __init__(self, high_thresh=0.6, low_thresh=0.1, iou_threshold=0.3, n_init=3, max_age=30):
        self.tracks = []
        self._next_id = 1
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.kf = KalmanFilterSimple()
        self.n_init = n_init
        self.max_age = max_age

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, detections):
        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]
        active_tracks = [t for t in self.tracks if not t.is_deleted()]

        cost = iou_cost(active_tracks, high_dets)
        matches, u_t, u_d = min_cost_matching(cost, self.iou_threshold)

        for ti, di in matches:
            t = active_tracks[ti]
            d = high_dets[di]
            t.update(xyxy_to_xyah(d['bbox']), d['score'])

        unmatched_tracks = [active_tracks[i] for i in u_t]
        if unmatched_tracks and low_dets:
            cost2 = iou_cost(unmatched_tracks, low_dets)
            matches2, u_t2, u_d2 = min_cost_matching(cost2, self.iou_threshold)
            for ti, di in matches2:
                unmatched_tracks[ti].update(xyxy_to_xyah(low_dets[di]['bbox']), low_dets[di]['score'])
            unmatched_tracks = [unmatched_tracks[i] for i in u_t2]

        for t in unmatched_tracks:
            t.time_since_update += 1
            t.mark_missed()

        unmatched_high = [high_dets[i] for i in u_d]
        for det in unmatched_high:
            mean, cov = self.kf.initiate(xyxy_to_xyah(det['bbox']))
            self.tracks.append(Track(mean, cov, self._next_id, n_init=self.n_init, max_age=self.max_age, score=det['score']))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def step(self, detections):
        self.predict()
        self.update(detections)
        return [
            {'id': t.track_id, 'bbox': t.to_tlbr(), 'score': t.score}
            for t in self.tracks if t.is_confirmed() and t.time_since_update == 0
        ]
