import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import pennylane as qml
from scipy.optimize import minimize
from generate_dataset import generate_dataset_per_N, brute_force



def get_tsp_instance_data(n_cities, distance_matrix):
    cities_to_permute = list(range(1, n_cities))
    valid_tours = list(itertools.permutations(cities_to_permute))

    M = len(valid_tours)
    n_qubits = math.ceil(math.log2(M))
    dim = 2 ** n_qubits

    # |F> state
    state_F = np.zeros(dim, dtype=complex)
    state_F[:M] = 1.0 / np.sqrt(M)

    costs = np.zeros(dim)
    for idx in range(M):
        tour = [0] + list(valid_tours[idx]) + [0]
        costs[idx] = sum(
            distance_matrix[tour[i]][tour[i + 1]]
            for i in range(len(tour) - 1)
        )

    return state_F, costs, n_qubits, M


def make_qnode(state_F, costs, n_qubits, p):
    dim = len(costs)
    rho = np.outer(state_F, state_F.conj())
    identity = np.eye(dim, dtype=complex)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        gammas = params[:p]
        betas = params[p:]

        qml.StatePrep(state_F, wires=range(n_qubits))

        for k in range(p):
            # Cost Unitary: diag(exp(-i * gamma_k * costs[0]???))
            U_cost = np.diag(np.exp(-1j * gammas[k] * costs))
            qml.QubitUnitary(U_cost, wires=range(n_qubits))

            # Mixer Unitary: exp(-i * beta_k * H_initial)
            # Rechen-Trick: U_M = I + (exp(-1j * beta_k) - 1) * rho
            U_mixer = identity + (np.exp(-1j * betas[k]) - 1.0) * rho
            qml.QubitUnitary(U_mixer, wires=range(n_qubits))

        return qml.probs(wires=range(n_qubits))

    return circuit


def make_objective(circuit, costs, M, l_star, l_worst):
    def objective(params):
        probs = circuit(params)
        return np.dot(probs[:M], costs[:M] - l_star) / (l_worst - l_star)
    return objective


def create_tsp_error_plot(plot_data, num_cities):
    depths = np.array(plot_data['depths'])
    avg_errors = np.array(plot_data['avg_errors'])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, avg_errors, marker='o')

    ax.set_xscale('log', base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels(depths.astype(int))

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Solution Quality Error $e_L$')
    ax.set_xlabel('depth $p$')
    ax.set_title(f'$k+1 = {num_cities}$')

    ax.axhline(0.5, linestyle='--', color='black')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"dme_tsp2_error_plot_{num_cities}_cities.png")
    plt.show()


if __name__ == "__main__":

    num_cities = 7
    depths = [2, 4, 8, 16, 32]
    plot_data = {"depths": [], "avg_errors": []}
    benchmark_dataset = generate_dataset_per_N(
        N=num_cities,
        n_circle_total=150,
        n_random=50,
        seed=42
    )

    instance_cache = {}

    for inst in benchmark_dataset:
        D = inst.D
        state_F, costs, n_qubits, M = get_tsp_instance_data(num_cities, D)
        l_star, l_worst = brute_force(D)

        instance_cache[id(D)] = {
            "state_F": state_F,
            "costs": costs,
            "n_qubits": n_qubits,
            "M": M,
            "l_star": l_star,
            "l_worst": l_worst
        }

    for p in depths:
        print(f"\nBenchmarking for p =", p, " and ", num_cities, " cities ===")
        errors = []

        # beta_k = (1 - s(t_k)) * delta, gamma_k = s(t_k) * delta
        s_T = np.linspace(0, 1, p)
        init_params = np.concatenate([s_T, 1 - s_T])

        for inst in benchmark_dataset:
            data = instance_cache[id(inst.D)]

            circuit = make_qnode(
                data["state_F"],
                data["costs"],
                data["n_qubits"],
                p
            )

            objective = make_objective(
                circuit,
                data["costs"],
                data["M"],
                data["l_star"],
                data["l_worst"]
            )

            res = minimize(
                objective,
                x0=init_params,
                method="L-BFGS-B"
            )

            errors.append(res.fun)

        avg_error = np.mean(errors)
        plot_data["depths"].append(p)
        plot_data["avg_errors"].append(avg_error)

        print(f"Mean Solution Quality Error for p={p}: e_L={avg_error:.6f}")

    print("\nBenchmark completed.")
    create_tsp_error_plot(plot_data, num_cities)
