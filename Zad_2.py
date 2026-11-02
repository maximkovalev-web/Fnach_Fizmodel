import numpy as np


class Source:
    def __init__(self, Y: float = 10000.0):
        self.Y = Y

    def sample_directions(self, E: float):
        N = int(np.random.poisson(self.Y * E))   

        cos_t = np.random.uniform(-1.0, 1.0, N)
        sin_t = np.sqrt(1.0 - cos_t**2)
        phi = np.random.uniform(0.0, 2.0 * np.pi, N)

        nx = sin_t * np.cos(phi)
        ny = sin_t * np.sin(phi)
        nz = cos_t
        return np.column_stack((nx, ny, nz))



np.random.seed(42)

src = Source(Y=10000.0)
E = 1.0                                   
dirs = src.sample_directions(E)

print(f"E = {E} МэВ, разыграно направлений: {len(dirs)}, shape = {dirs.shape}")
print("Первые 10 направлений (nx, ny, nz):")
print(np.round(dirs[:10], 4))
