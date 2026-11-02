import numpy as np
import k3d

class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, seed=None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.rng = np.random.default_rng(seed)

        # Spiral Fibonachi
        indices = np.arange(N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / N)
        theta = np.pi * (1 + 5**0.5) * indices

        x = R * np.cos(theta) * np.sin(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(phi)

        self.pmt_positions = np.column_stack((x, y, z))
        self._pmt_unit = self.pmt_positions / R

    # Geometry
    def _nearest_pmt(self, point):
        point = np.asarray(point, dtype=float)
        dists = np.linalg.norm(self.pmt_positions - point, axis=1)
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def is_point_on_pmt(self, point, atol=1e-9):
        point = np.asarray(point, dtype=float)
        if not np.isclose(np.linalg.norm(point), self.R, atol=atol):
            return False
        _, dmin = self._nearest_pmt(point)
        return dmin <= self.r

    # Visual
    def visualize(self):
        plot = k3d.plot()
        plot += k3d.points(
            positions=self.pmt_positions.astype(np.float32),
            point_size=0.15, color=0xff0000, shader='3d'
        )
        theta_mesh, phi_mesh = np.mgrid[0:2*np.pi:60j, 0:np.pi:60j]
        x_m = self.R * np.cos(theta_mesh) * np.sin(phi_mesh)
        y_m = self.R * np.sin(theta_mesh) * np.sin(phi_mesh)
        z_m = self.R * np.cos(phi_mesh)

        vertices = np.column_stack(
            (x_m.flatten(), y_m.flatten(), z_m.flatten())
        ).astype(np.float32)

        indices = []
        for i in range(59):
            for j in range(59):
                k = i * 60 + j
                indices.append([k, k+1, k+60])
                indices.append([k+1, k+61, k+60])
        indices = np.array(indices, dtype=np.uint32)

        plot += k3d.mesh(vertices, indices, color=0x333333, opacity=0.3)
        plot.display()


# Demonstration

det = Detector(R=2.0, N=40, r=0.1, a=0.5, seed=42)
p_hit = det.pmt_positions[0]            
p_miss = np.array([0.0, 0.0, det.R])    
print("Геометрия:")
print(det.is_point_on_pmt(p_hit))
print(det.is_point_on_pmt(p_miss))

    

