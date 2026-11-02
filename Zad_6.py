import numpy as np
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, L=8.0, seed=None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.L = L
        self.rng = np.random.default_rng(seed)

        indices = np.arange(N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / N)
        theta = np.pi * (1 + 5**0.5) * indices

        x = R * np.cos(theta) * np.sin(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(phi)

        self.pmt_positions = np.column_stack((x, y, z))
        self._pmt_unit = self.pmt_positions / R
        self._cos_thr = 1.0 - 0.5 * (r / R) ** 2

    def _nearest_pmt(self, point):
        point = np.asarray(point, dtype=float)
        dists = np.linalg.norm(self.pmt_positions - point, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def is_point_on_pmt(self, point, atol=1e-9):
        point = np.asarray(point, dtype=float)
        if not np.isclose(np.linalg.norm(point), self.R, atol=atol):
            return False
        _, dmin = self._nearest_pmt(point)
        return dmin <= self.r

    def trace(self, p0, n):
        p0 = np.asarray(p0, dtype=float)
        n = np.asarray(n, dtype=float)

        b = float(p0 @ n)
        c = float(p0 @ p0) - self.R**2
        disc = b * b - c
        if disc < 0:
            return -1, 0.0

        t = -b + np.sqrt(disc)
        if t <= 0:
            return -1, 0.0

        p_exit = p0 + t * n
        u = p_exit / np.linalg.norm(p_exit)

        cos_i = self._pmt_unit @ u
        j = int(np.argmax(cos_i))
        if cos_i[j] >= self._cos_thr:
            return j, t
        return -1, t

    def sample_photoelectrons(self, a):
        if a <= 0:
            return 0
        return int(self.rng.poisson(a))

    def simulate(self, p0, n_photons, attenuation=False):
        p0 = np.asarray(p0, dtype=float)

        cos_t = self.rng.uniform(-1.0, 1.0, n_photons)
        sin_t = np.sqrt(1.0 - cos_t**2)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, n_photons)

        dirs = np.column_stack((
            sin_t * np.cos(phi),
            sin_t * np.sin(phi),
            cos_t,
        ))

        hits = np.zeros(self.N, dtype=float)
        for n in dirs:
            i, dist = self.trace(p0, n)
            if i >= 0:
                if attenuation:
                    hits[i] += np.exp(-dist / self.L)
                else:
                    hits[i] += 1.0

        Q = sum(self.sample_photoelectrons(h * self.a) for h in hits)
        return Q


def run_events(det, p0, E, Y=1e4, n_events=1000, attenuation=False, seed=0):
    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, dtype=float)
    Q_arr = np.empty(n_events)
    for k in range(n_events):
        N_phot = int(rng.poisson(Y * E))
        Q_arr[k] = det.simulate(p0, N_phot, attenuation)
    return Q_arr


if __name__ == "__main__":
    det = Detector(R=2.0, N=40, r=0.1, a=0.5, L=8.0, seed=42)
    E = 1.0
    z_positions = np.linspace(-det.R * 0.95, det.R * 0.95, 10)
    
    results_no_att = []
    results_att = []
    
    for i, z in enumerate(z_positions):
        p0 = np.array([0.0, 0.0, z])
        Q_no = run_events(det, p0, E, n_events=1000, attenuation=False, seed=100+i)
        Q_att = run_events(det, p0, E, n_events=1000, attenuation=True, seed=200+i)
        results_no_att.append(Q_no.mean())
        results_att.append(Q_att.mean())

    plt.figure(figsize=(8, 5))
    plt.plot(z_positions, results_no_att, 'o-', label='Без затухания')
    plt.plot(z_positions, results_att, 's--', label='С затуханием (L=8 м)')
    plt.xlabel('z, м')
    plt.ylabel('Средний собранный заряд ⟨Q⟩')
    plt.title('Энергетический отклик детектора (E = 1 МэВ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('attenuation_comparison.png', dpi=130)
    plt.show()
