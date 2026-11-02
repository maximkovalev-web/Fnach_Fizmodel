import numpy as np
from scipy.spatial import cKDTree

class Detector:
    def __init__(self, R=15.0, N=2000, r=0.1, a=0.5, seed=None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.rng = np.random.default_rng(seed)

        indices = np.arange(N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / N)
        theta = np.pi * (1 + 5**0.5) * indices

        x = R * np.cos(theta) * np.sin(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(phi)

        self.pmt_positions = np.column_stack((x, y, z))
        
        self.tree = cKDTree(self.pmt_positions)

    def detect(self, point, atol=1e-9):
        point = np.asarray(point, dtype=float)
        if not np.isclose(np.linalg.norm(point), self.R, atol=atol):
            return False, None


        dist, idx = self.tree.query(point)
        
        if dist <= self.r:

            registered = self.rng.random() < self.a
            return registered, idx
        return False, None
