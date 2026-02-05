import numpy as np
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TSPInstance:
    kind: str                 # "circle" or "random"
    N: int
    sigma: Optional[float]
    coords: np.ndarray
    D: np.ndarray


def generate_dataset_per_N(
    N: int,
    sigmas: Tuple[float, float, float] = (0.6, 1.0, 1.4),
    n_circle_total: int = 150,
    n_random: int = 50,
    seed: int = 42,
    max_trials_per_bucket: int = 100000
) -> List[TSPInstance]:

    assert n_circle_total % len(sigmas) == 0, "150 circle instances need to be evenly distributed across σ."
    per_sigma = n_circle_total // len(sigmas)
    rng = np.random.default_rng(seed)
    dataset: List[TSPInstance] = []

    # circle instances per sigma
    for sigma in sigmas:
        kept = 0
        trials = 0

        while kept < per_sigma:
            trials += 1

            if trials > max_trials_per_bucket:
                raise RuntimeError(f"Too many tries for circle σ={sigma} with N={N}. Increase max_trials_per_bucket.")

            inst = generate_cities(N, sigma=sigma, seed=int(rng.integers(1, 2**31-1)))
            l_star, _ = brute_force(inst.D)

            if greedy_filter(inst.D, l_star):
                dataset.append(inst)
                kept += 1

    # random instances
    kept = 0
    trials = 0

    while kept < n_random:
        trials += 1

        if trials > max_trials_per_bucket:
            raise RuntimeError(f"Too many tries for random with N={N}. Increase max_trials_per_bucket.")

        inst = generate_random_cities(N, seed=int(rng.integers(1, 2**31-1)))
        l_star, _ = brute_force(inst.D)

        if greedy_filter(inst.D, l_star):
            dataset.append(inst)
            kept += 1

    assert len(dataset) == n_circle_total + n_random
    return dataset

def generate_cities(N, sigma, seed=None) -> TSPInstance:
    if seed is not None:
        np.random.seed(seed)

    R = N / (2 * np.pi)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_circle = R * np.cos(angles)
    y_circle = R * np.sin(angles)

    rand_phi = np.random.uniform(0, 2 * np.pi, N)
    rand_r = np.random.uniform(0, sigma, N)

    dx = rand_r * np.cos(rand_phi)
    dy = rand_r * np.sin(rand_phi)

    x_perturbed = x_circle + dx
    y_perturbed = y_circle + dy

    coords = np.vstack((x_perturbed, y_perturbed)).T

    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    return TSPInstance(kind="circle", N=N, sigma=sigma, coords=coords, D=D)


def generate_random_cities(N, seed=None) -> TSPInstance:
    if seed is not None:
        np.random.seed(seed)

    coords = np.random.rand(N, 2)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    return TSPInstance(kind="random", N=N, sigma=None, coords=coords, D=D)


def brute_force(D) -> Tuple[float, float]:
    n = D.shape[0]
    best, worst = np.inf, -np.inf

    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm + (0,)
        length = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        best = min(best, length)
        worst = max(worst, length)
    return best, worst


def greedy_algortihm(D: np.ndarray, start: int) -> float:  # nearest neighbour
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    cur = start
    tour_len = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour_len += D[cur, nxt]
        cur = nxt
        unvisited.remove(nxt)
    tour_len += D[cur, start]
    return float(tour_len)


def greedy_filter(D: np.ndarray, l_star: float, atol: float = 1e-9) -> bool:
    n = D.shape[0]
    for s in range(n):
        if np.isclose(greedy_algortihm(D, s), l_star, atol=atol):
            return False
    return True



def save_dataset_npz(path: str, dataset: List[TSPInstance]) -> None:
    kinds = np.array([inst.kind for inst in dataset], dtype=object)
    Ns = np.array([inst.N for inst in dataset], dtype=np.int32)
    sigmas = np.array([(-1.0 if inst.sigma is None else inst.sigma) for inst in dataset], dtype=np.float64)
    coords_list = [inst.coords for inst in dataset]
    D_list = [inst.D for inst in dataset]
    np.savez_compressed(path, kinds=kinds, Ns=Ns, sigmas=sigmas, coords=coords_list, distances=D_list)


if __name__ == "__main__":
    """
    for N in [4, 5, 6, 7]:
        print(f"Erzeuge Datensatz für N={N} (200 Instanzen, Greedy-gefiltert) ...")
        ds = generate_dataset_per_N(N)
        print(f" -> fertig: {len(ds)} Instanzen")
        #save_dataset_npz(f"tsp_ref77_N{N}_greedyfiltered.npz", ds)
        #print(f" -> gespeichert: tsp_ref77_N{N}_greedyfiltered.npz")
    """
    circle_cities = generate_cities(4, 0.0, seed=42)
