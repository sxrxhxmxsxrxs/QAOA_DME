import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import pennylane as qml
from tqdm import tqdm

from generate_dataset import generate_dataset_per_N, brute_force


def get_tsp_instance_data(n_cities, distance_matrix):
    cities_to_permute = list(range(1, n_cities))
    valid_tours = list(itertools.permutations(cities_to_permute))

    M = len(valid_tours)
    n_qubits = math.ceil(math.log2(M))
    dim = 2 ** n_qubits

    state_F = np.zeros(dim, dtype=complex)
    state_F[:M] = 1.0 / np.sqrt(M)

    costs = np.zeros(dim)
    for idx, perm in enumerate(valid_tours):
        tour = [0] + list(perm) + [0]
        costs[idx] = sum(
            distance_matrix[tour[i]][tour[i + 1]]
            for i in range(len(tour) - 1)
        )

    return state_F, costs, n_qubits, M


def make_qnode(n_qubits, p):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circuit(state_F, costs, params):
        gammas = params[:p]
        betas = params[p:]

        qml.StatePrep(state_F, wires=range(n_qubits))

        dim = len(costs)
        rho = np.outer(state_F, state_F.conj())
        identity = np.eye(dim, dtype=complex)

        for k in range(p):
            qml.DiagonalQubitUnitary(
                np.exp(1j * gammas[k] * costs),
                wires=range(n_qubits)
            )

            U_mixer = identity + (np.exp(-1j * betas[k]) - 1.0) * rho
            qml.QubitUnitary(U_mixer, wires=range(n_qubits))

        return qml.probs(wires=range(n_qubits))

    return circuit


def calculate_error_from_probs(probs, shifted_costs):
    return probs[: len(shifted_costs)] @ shifted_costs


def create_tsp_error_plot(plot_data, num_cities):
    depths = np.array(plot_data["depths"])
    avg_errors = np.array(plot_data["avg_errors"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, avg_errors, marker="o")

    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels(depths.astype(int))

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Solution Quality Error $e_L$")
    ax.set_xlabel("depth $p$")
    ax.set_title(f"$k+1 = {num_cities}$ (Fixed Params)")

    ax.axhline(0.5, linestyle="--", color="black")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"final_dme_tsp_error_plot_{num_cities}_cities_fixed.png")
    plt.show()


if __name__ == "__main__":
    num_cities = 9
    depths = [2, 4, 8, 16, 32]
    plot_data = {"depths": [], "avg_errors": []}

    benchmark_dataset = generate_dataset_per_N(
        N=num_cities,
        n_circle_total=150,
        n_random=50,
        seed=42
    )


    instance_cache = []
    for inst in tqdm(benchmark_dataset, desc="Precomputing instances"):
        state_F, costs, n_qubits, M = get_tsp_instance_data(num_cities, inst.D)
        l_star, l_worst = brute_force(inst.D)

        shifted_costs = (costs[:M] - l_star) / (l_worst - l_star)

        instance_cache.append({
            "state_F": state_F,
            "costs": costs,
            "shifted_costs": shifted_costs,
        })


    for p in depths:
        errors = []

        k = np.arange(p)
        gammas = k / (p - 1)
        betas = 1.0 - k / (p - 1)
        fixed_params = np.concatenate([gammas, betas])

        circuit = make_qnode(n_qubits, p)

        for data in tqdm(
            instance_cache,
            desc=f"Instances processed (p={p})"
        ):
            probs = circuit(
                data["state_F"],
                data["costs"],
                fixed_params
            )

            e_L = calculate_error_from_probs(
                probs,
                data["shifted_costs"]
            )
            errors.append(e_L)

        avg_error = np.mean(errors)
        plot_data["depths"].append(p)
        plot_data["avg_errors"].append(avg_error)

        print(f"p={p}: mean e_L = {avg_error:.6f}")

    print("\nBenchmark completed.")
    create_tsp_error_plot(plot_data, num_cities)
