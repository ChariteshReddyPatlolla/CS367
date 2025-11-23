import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
def bipolar_rand(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Return random bipolar vector of length n: +1 / -1."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        return np.where(rng.rand(n) > 0.5, 1, -1)
    return np.where(np.random.rand(n) > 0.5, 1, -1)


def flip_fraction(pattern: np.ndarray, fraction: float, seed: Optional[int] = None) -> np.ndarray:
    
    rng = np.random.RandomState(seed)
    noisy = pattern.copy()
    k = int(round(fraction * noisy.size))
    if k <= 0:
        return noisy
    idx = rng.choice(noisy.size, size=k, replace=False)
    noisy[idx] *= -1
    return noisy

class Hopfield:
    

    def _init_(self, N: int):
        self.N = int(N)
        self.W = np.zeros((self.N, self.N), dtype=float)

    def train(self, patterns: np.ndarray) -> None:
        
        P = np.atleast_2d(patterns)
        assert P.shape[1] == self.N, "Patterns must have length N"
        W = np.zeros((self.N, self.N), dtype=float)
        for p in P:
            p = p.reshape(self.N, 1)
            W += p @ p.T
        self.W = W / float(self.N)
        np.fill_diagonal(self.W, 0.0)

    def recall(self, state: np.ndarray, max_iter: int = 200, synchronous: bool = True) -> np.ndarray:
       
        s = state.copy().astype(int)
        for it in range(max_iter):
            if synchronous:
                s_new = np.where(self.W @ s >= 0, 1, -1)
            else:
                # asynchronous: single random neuron update per iteration
                s_new = s.copy()
                i = np.random.randint(0, self.N)
                hi = (self.W[i] @ s)
                s_new[i] = 1 if hi >= 0 else -1
            if np.array_equal(s_new, s):
                break
            s = s_new
        return s

    def energy(self, s: np.ndarray) -> float:

        return float(-0.5 * s.T @ self.W @ s)

def associative_memory_demo(N: int = 100, P: int = 10, noise_frac: float = 0.15, seed: Optional[int] = 0):
    np.random.seed(seed)
    print(" demo")
    print(f"N={N}, P={P}, noise_frac={noise_frac}")

    # generate random bipolar patterns
    patterns = np.array([bipolar_rand(N) for _ in range(P)])
    net = Hopfield(N)
    net.train(patterns)

    test_idx = 0
    original = patterns[test_idx].copy()
    noisy_input = flip_fraction(original, noise_frac, seed=seed + 1 if seed is not None else None)
    recalled = net.recall(noisy_input, max_iter=500, synchronous=True)

    success = np.array_equal(recalled, original)
    print("Recovery success:", success)
    print("Hamming distance (before):", np.sum(noisy_input != original) // 2 if success else np.sum(noisy_input != original))
    print("Energy before:", net.energy(noisy_input))
    print("Energy after :", net.energy(recalled))
    return net, patterns, original, noisy_input, recalled



# --------------Capacity test------------------

def capacity_test(N: int = 100, max_p: int = 30, trials: int = 20, noise_frac: float = 0.10, seed: Optional[int] = 0, plot: bool = True):
   
    rng = np.random.RandomState(seed)
    results: Dict[int, float] = {}
    print(f"Running capacity test N={N}, trials={trials}, noise={noise_frac}")
    for p in range(1, max_p + 1):
        success = 0
        for t in range(trials):
            patterns = np.array([np.where(rng.rand(N) > 0.5, 1, -1) for _ in range(p)])
            net = Hopfield(N)
            net.train(patterns)
            noisy = flip_fraction(patterns[0], noise_frac, seed=rng.randint(1_000_000))
            recalled = net.recall(noisy, max_iter=500)
            if np.array_equal(recalled, patterns[0]):
                success += 1
        rate = (success / trials) * 100.0
        results[p] = rate
        print(f"P={p:2d} -> success {rate:.1f}%")
    if plot:
        plt.figure()
        plt.plot(list(results.keys()), list(results.values()), marker='o')
        plt.xlabel("Number of patterns stored (P)")
        plt.ylabel("Recall success rate (%)")
        plt.title("Hopfield capacity test")
        plt.grid(True)
        plt.show()
    return results



# ------------ Error-correcting capability vs noise-------------------------------

def error_correction_demo(net: Hopfield, base_pattern: np.ndarray, trials: int = 100, plot: bool = True):
    
    noise_levels = np.linspace(0.0, 0.5, 11)
    success_rates = []
    print("Running error-correction sweep (noise 0%..50%)")
    for frac in noise_levels:
        succ = 0
        for _ in range(trials):
            noisy = flip_fraction(base_pattern, frac)
            recalled = net.recall(noisy, max_iter=500)
            if np.array_equal(recalled, base_pattern):
                succ += 1
        rate = (succ / trials) * 100.0
        success_rates.append(rate)
        print(f"Noise {int(frac*100):>3}% -> success {rate:5.1f}%")
    if plot:
        plt.figure()
        plt.plot(noise_levels * 100, success_rates, marker='o')
        plt.xlabel("Noise level (%)")
        plt.ylabel("Recall success rate (%)")
        plt.title("Hopfield error-correction vs noise")
        plt.grid(True)
        plt.show()
    return noise_levels, success_rates


=
def sigmoid(x: np.ndarray, gain: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-gain * x))


def solve_eight_rook(A: float = 600.0, B: float = 600.0, gain: float = 10.0, dt: float = 0.1, steps: int = 4000, seed: Optional[int] = 0) -> np.ndarray:
   
    rng = np.random.RandomState(seed)
    n = 8
    u = 0.1 * rng.randn(n, n)
    v = sigmoid(u, gain)
    for it in range(steps):
        row_err = v.sum(axis=1)[:, None] - 1.0   # each row should sum to 1
        col_err = v.sum(axis=0)[None, :] - 1.0   # each column should sum to 1
        # gradient descent style update
        u -= dt * (A * row_err + B * col_err)
        v = sigmoid(u, gain)
    binary = (v > 0.5).astype(int)
    return binary


#----------Example ---------------
def hopfield_tsp(n: int = 10, steps: int = 5000, seed: Optional[int] = 0):
   
    rng = np.random.RandomState(seed)
    coords = rng.rand(n, 2)
    # precompute symmetric distance matrix
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))

  
    A = 500.0
    B = 500.0
    C = 200.0
    D = 0.02
    gain = 500.0
    dt = 0.01

    u = 0.1 * rng.randn(n, n)
    v = sigmoid(u, gain)

    for it in range(steps):
        row = v.sum(axis=1)[:, None] - 1.0     
        col = v.sum(axis=0)[None, :] - 1.0     

        tour = np.zeros_like(v)
        for p in range(n):
            tour[:, p] = dist @ v[:, (p + 1) % n]

        u -= dt * (A * row + B * col + C * tour + D)
        v = sigmoid(u, gain)


    x = np.zeros_like(v, dtype=int)
    used = set()
    for p in range(n):
        order = np.argsort(-v[:, p])  # descending
        for c in order:
            if c not in used:
                x[c, p] = 1
                used.add(c)
                break
   
    remaining = [c for c in range(n) if c not in used]
    for p in range(n):
        if x[:, p].sum() == 0 and remaining:
            x[remaining.pop(), p] = 1

    route = [int(np.where(x[:, p] == 1)[0][0]) for p in range(n)]
    length = float(sum(dist[route[i], route[(i + 1) % n]] for i in range(n)))

    weights_needed = (n * n * (n * n - 1)) // 2
    return route, length, weights_needed, coords

def run_all():
    
    net, patterns, orig, noisy, recalled = associative_memory_demo()
    
    cap = capacity_test(plot=True)
    
    noise_levels, rates = error_correction_demo(net, patterns[0], plot=True)
    
    rooks = solve_eight_rook()
    print("\nEight-rook solution (row sums, col sums):")
    print(rooks.sum(axis=1), rooks.sum(axis=0))
    print(rooks)
    route, length, weights, coords = hopfield_tsp()
    print("\nTSP result:")
    print("Route:", route)
    print("Tour length:", round(length, 4))
    print("Weights needed (meta):", weights)


if __name__ == "__main__":
    main()
