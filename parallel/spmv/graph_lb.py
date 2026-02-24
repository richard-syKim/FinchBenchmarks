import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph/loadbalance"
RUNTIME_FOLDER = "runtime"
SPEEDUP_FOLDER = "speedup"
RESULTS_FOLDER = "results/loadbalance"

NTHREADS = [i + 1 for i in range(14)]

DEFAULT_METHOD = "optimal"
METHODS = [
    "static",
    "greedy",
    "julia",
    "optimal",
]

DATASETS = {
    "log_skewed": ["e^(-x/40K) Decay of nnz over rows"],
}
NUM_MATRICES = sum([len(matrices) for matrices in DATASETS.values()])

COLORS = [
    "gray",
    "cadetblue",
    "saddlebrown",
    "navy",
    "black",
    "orange",
    "green",
    "red",
    "purple",
]


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spmv_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            matrix = (result["matrix_type"].replace("/", "-"))
            combine_results[matrix][result["method"]][result["n_threads"]] = result["time"]
            
    return combine_results


def plot_runtime_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 10))
    for method, color in zip(METHODS, COLORS):
        y_values = [
            results[matrix][method][n_thread]
            for n_thread in NTHREADS
        ]

        plt.plot(
            NTHREADS,
            y_values,
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Total Runtime for {dataset}: {matrix}")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Total Runtime (in seconds)")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


def plot_speedup_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 10))
    for method, color in zip(METHODS, COLORS):
        if method != DEFAULT_METHOD:
            y_values = [
                results[matrix][DEFAULT_METHOD][n_thread]
                / results[matrix][method][n_thread]
                for n_thread in NTHREADS
            ]

            plt.plot(
                NTHREADS,
                y_values,
                label=method,
                color=color,
                marker="o",
                linestyle="-",
                linewidth=1,
            )

    plt.title(f"Speedup of Runtime for {dataset}: {matrix} (with respect to {DEFAULT_METHOD})")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(GRAPH_FOLDER, RUNTIME_FOLDER), exist_ok=True)

    results = load_json()
    for dataset, matrices in DATASETS.items():
        for matrix in matrices:
            plot_runtime_result(
                results,
                dataset,
                matrix,
                os.path.join(GRAPH_FOLDER, RUNTIME_FOLDER, f"{dataset}-{matrix}.png"),
            )

            plot_speedup_result(
                results,
                dataset,
                matrix,
                os.path.join(GRAPH_FOLDER, SPEEDUP_FOLDER, f"{dataset}-{matrix}.png"),
            )
