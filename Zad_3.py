import numpy as np
import k3d


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, seed=None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.rng = np.random.default_rng(seed)
        #Spiral Fibanscchi
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
            return -1

        t = -b + np.sqrt(disc)                  
        if t <= 0:
            return -1

        p_exit = p0 + t * n
        u = p_exit / np.linalg.norm(p_exit)      

        cos_i = self._pmt_unit @ u
        j = int(np.argmax(cos_i))
        if cos_i[j] >= self._cos_thr:
            return j
        return -1                                

    
    def sample_photoelectrons(self, a):
        if a <= 0:
            return 0
        return int(self.rng.poisson(a))


    def simulate(self, p0, n_photons):
        p0 = np.asarray(p0, dtype=float)

       
        cos_t = self.rng.uniform(-1.0, 1.0, n_photons)
        sin_t = np.sqrt(1.0 - cos_t**2)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, n_photons)

        dirs = np.column_stack((
            sin_t * np.cos(phi),
            sin_t * np.sin(phi),
            cos_t,
        ))

        hits = np.zeros(self.N, dtype=int)
        for n in dirs:
            i = self.trace(p0, n)
            if i >= 0:
                hits[i] += 1

        Q = sum(self.sample_photoelectrons(h * self.a) for h in hits)
        return Q, hits






det = Detector(R=2.0, N=40, r=0.1, a=0.5, seed=42)



E, Y = 1.0, 1e4
rng_ev = np.random.default_rng(0)
N_phot = int(rng_ev.poisson(Y * E))

Q, hits = det.simulate(np.zeros(3), N_phot)

frac_area = det.N * det.r**2 / (4 * det.R**2)
Q_expected = det.a * N_phot * frac_area

print("\nСобытие 1 МэВ в центре:")
print(f"  испущено фотонов      : {N_phot}")
print(f"  попало в ФЭУ          : {hits.sum()}  ")
print(f"(доля {hits.sum()/N_phot:.4f}, ожидание {frac_area:.4f})")
print(f"  полный заряд Q        : {Q}")
print(f"  ожидание Q_ожид       : {Q_expected:.1f}")
print(f"  отношение Q/Q_ожид    : {Q/Q_expected:.3f}")


