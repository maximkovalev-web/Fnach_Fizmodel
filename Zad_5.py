import numpy as np


class Source:
    def __init__(self, Y: float = 10000.0, batch: int = 100):
        self.Y = Y          
        self.batch = batch  

    def n_photons(self, E: float) -> int:
        return int(np.random.poisson(self.Y * E))

    def generate(self, E: float):

        N = self.n_photons(E)
        emitted = 0

        while emitted < N:
            k = min(self.batch, N - emitted)

            cos_t = np.random.uniform(-1.0, 1.0, k)
            sin_t = np.sqrt(1.0 - cos_t**2)
            phi = np.random.uniform(0.0, 2.0 * np.pi, k)

            batch = np.column_stack((sin_t * np.cos(phi),
                                     sin_t * np.sin(phi),
                                     cos_t))
            emitted += k
            yield batch



np.random.seed(42)

src = Source(Y=10000.0, batch=100)
E = 1.0   

total = 0
n_batches = 0
for batch in src.generate(E):
    n_batches += 1
    total += len(batch)
    if n_batches <= 3:
        print(f"батч #{n_batches}: shape = {batch.shape}, "
                  f"первые 2 направления:\n{np.round(batch[:2], 4)}")

print(f"\nВсего фотонов: {total} (ожидалось ~{src.Y * E:.0f})")
print(f"Батчей: {n_batches}, размер батча: {src.batch}")
