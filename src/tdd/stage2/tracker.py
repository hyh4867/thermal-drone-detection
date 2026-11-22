"""Image-plane multi-target tracker for detector outputs (stage 2 / Q2).

Design source: ``projects/radar-track-lab/src/radar_track_lab/tracker.py`` on branch
hyh4867/Radar (commit c4ed4d7) of this repository - Kalman filter + chi-square
Mahalanobis gating + global-nearest-neighbour association solved with the Hungarian
algorithm, confirmed tracks associated first, then tentative tracks, leftover
detections start tentative tracks; M-of-N confirmation and K-consecutive-miss
deletion; track ids never reused. That project tracks radar plots with an EKF in
polar measurement space; this file is a NEW, much smaller implementation for box
centres in the image plane (linear measurement), written for this project. No code
was copied.

Units: pixels of the 640x512 reference frame, seconds.
State: [x, y, vx, vy] (model "cv": nearly-constant velocity, white-noise acceleration
with PSD ``q`` px^2/s^3) or [x, y] (model "rw": random walk with PSD ``q`` px^2/s).
Evaluation only: the tracker reports where boxes are; it assigns no priority, threat
level or any other judgement to a track.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1.0e9
CHI2_2DOF = {0.95: 5.991, 0.99: 9.210, 0.999: 13.816}
TENTATIVE, CONFIRMED = 0, 1


@dataclass(frozen=True)
class TrackerParams:
    model: str = "cv"            # "cv" or "rw"
    q: float = 100.0             # process-noise PSD
    sigma_meas: float = 2.0      # measurement std of a box centre [px]
    gate_probability: float = 0.99
    init_speed_sigma: float = 60.0   # px/s, prior std of the unknown initial velocity
    confirm_m: int = 2
    confirm_n: int = 3
    delete_misses: int = 3
    det_threshold: float = 0.1   # detections below this score are not given to the tracker
    coast_outputs: int = 0       # missed frames for which a confirmed track still outputs its predicted box
    size_alpha: float = 0.5      # exponential smoothing of the box width / height

    @property
    def gate(self) -> float:
        return CHI2_2DOF[self.gate_probability]


class MofN:
    """M-of-N confirmation / K-miss deletion bookkeeping for one track."""

    def __init__(self, m: int, n: int, k: int):
        if not 1 <= m <= n:
            raise ValueError("need 1 <= M <= N")
        self.m, self.n, self.k = m, n, k
        self.status, self.hits, self.scans, self.misses = TENTATIVE, 1, 1, 0

    def step(self, hit: bool) -> bool:
        """Advance one frame; returns False when the track must be deleted."""
        self.hits += int(hit)
        self.scans += 1
        self.misses = 0 if hit else self.misses + 1
        if self.status == TENTATIVE:
            if self.hits >= self.m and self.scans <= self.n:
                self.status = CONFIRMED
            elif (self.scans - self.hits) > (self.n - self.m):
                return False                       # M hits within N frames no longer possible
        if self.status == CONFIRMED and self.misses >= self.k:
            return False
        return True


class Track:
    def __init__(self, tid, det, t, p: TrackerParams):
        cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
        r = p.sigma_meas ** 2
        if p.model == "cv":
            self.x = np.array([cx, cy, 0.0, 0.0])
            self.P = np.diag([r, r, p.init_speed_sigma ** 2, p.init_speed_sigma ** 2])
        else:
            self.x = np.array([cx, cy])
            self.P = np.diag([r, r])
        self.id, self.t = tid, t
        self.w, self.h = det[2] - det[0], det[3] - det[1]
        self.mofn = MofN(p.confirm_m, p.confirm_n, p.delete_misses)
        self.score_sum, self.score_n = float(det[4]), 1
        self.last_box = np.asarray(det[:4], dtype=float)
        self.updated = True
        self.confirmed_at = None

    @property
    def score(self) -> float:
        return self.score_sum / self.score_n

    def predict(self, t, p: TrackerParams, shift=(0.0, 0.0)):
        """Propagate to time t. ``shift`` = global image translation since the last
        frame (camera-motion compensation), added to the predicted position."""
        dt = t - self.t
        if p.model == "cv":
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            q11, q12, q22 = p.q * dt ** 3 / 3, p.q * dt ** 2 / 2, p.q * dt
            Q = np.array([[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]])
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
        else:
            self.P = self.P + np.eye(2) * p.q * dt
        self.x[0] += shift[0]
        self.x[1] += shift[1]
        self.t = t

    def innovation(self, det, p: TrackerParams):
        z = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
        S = self.P[:2, :2] + np.eye(2) * p.sigma_meas ** 2
        y = z - self.x[:2]
        return y, S, float(y @ np.linalg.solve(S, y))

    def update(self, det, p: TrackerParams):
        y, S, _ = self.innovation(det, p)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ np.eye(2, len(self.x))) @ self.P
        a = p.size_alpha
        self.w = a * (det[2] - det[0]) + (1 - a) * self.w
        self.h = a * (det[3] - det[1]) + (1 - a) * self.h
        self.score_sum += float(det[4])
        self.score_n += 1
        self.last_box = np.asarray(det[:4], dtype=float)

    def predicted_box(self):
        cx, cy = self.x[0], self.x[1]
        return np.array([cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2])


def associate(tracks, dets, p: TrackerParams):
    """Hungarian assignment of detections to tracks inside the chi-square gate.
    Returns {track_index: det_index}. (Vectorised form of Track.innovation.)"""
    if not tracks or len(dets) == 0:
        return {}
    pos = np.array([tr.x[:2] for tr in tracks])                       # (n, 2)
    S = np.array([tr.P[:2, :2] for tr in tracks]) + np.eye(2) * p.sigma_meas ** 2
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    z = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)   # (m, 2)
    y = z[None] - pos[:, None]                                        # (n, m, 2)
    d2 = (S[:, 1, 1, None] * y[..., 0] ** 2 - (S[:, 0, 1, None] + S[:, 1, 0, None]) * y[..., 0] * y[..., 1]
          + S[:, 0, 0, None] * y[..., 1] ** 2) / det[:, None]
    cost = np.where(d2 <= p.gate, d2, BIG)
    if not (cost < BIG).any():
        return {}
    r, c = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, c) if cost[i, j] < BIG}


class Tracker:
    """Feed frames in time order with ``step``; each call returns the output records
    of that frame: (box [x1,y1,x2,y2], track score, track id, updated_this_frame)."""

    def __init__(self, params: TrackerParams):
        self.p = params
        self.tracks: list[Track] = []
        self.next_id = 1

    def step(self, t: float, dets, shift=(0.0, 0.0)):
        p = self.p
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        dets = dets[dets[:, 4] >= p.det_threshold]
        for tr in self.tracks:
            tr.predict(t, p, shift)
            tr.updated = False
        free = np.ones(len(dets), bool)
        for status in (CONFIRMED, TENTATIVE):          # confirmed tracks choose first
            group = [tr for tr in self.tracks if tr.mofn.status == status]
            idx = np.flatnonzero(free)
            for i, j in associate(group, dets[idx], p).items():
                group[i].update(dets[idx[j]], p)
                group[i].updated = True
                free[idx[j]] = False
        keep = []
        for tr in self.tracks:
            was = tr.mofn.status
            if tr.mofn.step(tr.updated):
                if was == TENTATIVE and tr.mofn.status == CONFIRMED:
                    tr.confirmed_at = t
                keep.append(tr)
        self.tracks = keep
        for d in dets[free]:
            self.tracks.append(Track(self.next_id, d, t, p))
            self.next_id += 1
        out = []
        for tr in self.tracks:
            if tr.mofn.status != CONFIRMED:
                continue
            if tr.updated:
                out.append((tr.last_box.copy(), tr.score, tr.id, True))
            elif tr.mofn.misses <= p.coast_outputs:
                out.append((tr.predicted_box(), tr.score, tr.id, False))
        return out


def track_video(times, dets_per_frame, params: TrackerParams, shifts=None):
    """Runs one video. Returns per-frame lists of [x1, y1, x2, y2, track_score, track_id, updated]."""
    trk = Tracker(params)
    out = []
    for i, (t, d) in enumerate(zip(times, dets_per_frame)):
        recs = trk.step(float(t), d, shifts[i] if shifts is not None else (0.0, 0.0))
        out.append([[*b.tolist(), float(s), int(tid), bool(u)] for b, s, tid, u in recs])
    return out
