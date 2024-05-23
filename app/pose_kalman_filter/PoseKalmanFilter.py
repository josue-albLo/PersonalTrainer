from filterpy.kalman import KalmanFilter
import numpy as np


class PoseKalmanFilter:
    def __init__(self, n_landmarks, dim=2):
        self.n_landmarks = n_landmarks
        self.dim = dim
        self.filters = [self._create_kalman_filter() for _ in range(n_landmarks * dim)]

    def _create_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.zeros((4, 1))
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R = np.eye(2) * 0.01
        kf.Q = np.eye(4) * 0.01
        return kf

    def predict(self):
        for kf in self.filters:
            kf.predict()

    def update(self, landmarks):
        for i, lm in enumerate(landmarks.landmark):
            idx = i * self.dim
            measurement = np.array([[lm.x], [lm.y]])
            self.filters[idx].update(measurement[:2])
            lm.x, lm.y = self.filters[idx].x[:2].flatten()

    def apply_filter(self, landmarks):
        self.predict()
        self.update(landmarks)
