import json
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"

NTHREADS = [2**i for i in range(6)]

DEFAULT_METHOD = "serial_default_implementation"
SHARD_METHOD = "shard_implementation"
METHODS = [
    DEFAULT_METHOD,
    # "parallel_col_separate_sparselist_results",
    # "separated_memory_concatenate_results",
    SHARD_METHOD,
]

DATASETS = [
    {"uniform": ["1024-0.1", "2048-0.1", "4096-0.1", "8192-0.1", "16384-0.1", "32768-0.1"]},
    {"FEMLAB": ["FEMLAB-poisson3Da", "FEMLAB-poisson3Db"]},
]

COLORS = ["red", "gray", "cadetblue", "saddlebrown", "navy", "black"]


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spadd_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            matrix = (
                result["matrix"].replace("/", "-")
                if result["dataset"] != "uniform"
                else f"{result['matrix']['size']}-{result['matrix']['sparsity']}"
            )
            combine_results[result["dataset"]][matrix][result["method"]][
                result["n_threads"]
            ] = result["time"]

    return combine_results


def plot_speedup_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [
                results[dataset][matrix][DEFAULT_METHOD][n_thread]
                / results[dataset][matrix][method][n_thread]
                for n_thread in NTHREADS
            ],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(
        f"SpAdd - Speedup for {dataset}: {matrix} (with respect to {DEFAULT_METHOD})"
    )
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)


def plot_runtime_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [results[dataset][matrix][method][n_thread] for n_thread in NTHREADS],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"SpAdd - Runtime for {dataset}: {matrix}")
    plt.xscale("log", base=2)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Runtime (in seconds)")

    plt.legend()
    plt.savefig(save_location)


def weak_scaling_plot(results, dataset, save_location):
    plt.figure(figsize=(10, 6))
    plt.plot(
        NTHREADS,
        [
            results[dataset][f"{1024 * n_thread}-0.1"][SHARD_METHOD][n_thread]
            for n_thread in NTHREADS
        ],
        label="shard_implementation",
        color="grey",
        marker="o",
        linestyle="-",
        linewidth=1,
    )

    # for color in COLORS:
    #     plt.plot(
    #         NTHREADS,
    #         [
    #             results[dataset][matrix][SHARD_METHOD][n_thread]
    #             / results[dataset][matrix][SHARD_METHOD][1]
    #             for n_thread in NTHREADS
    #         ],
    #         label="uniform",
    #         color=color,
    #         marker="o",
    #         linestyle="-",
    #         linewidth=1,
    #     )

    plt.title(f"SpAdd - Weak Scaling for 1000 x X matrix of 0.1 sparsity")
    plt.xscale("log", base=2)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Runtime (in seconds)")

    plt.legend()
    plt.savefig(save_location)
    


if __name__ == "__main__":
    results = load_json()
    for datasets in DATASETS:
        for dataset, matrices in datasets.items():
            for matrix in matrices:
                # plot_speedup_result(
                #     results,
                #     dataset,
                #     matrix,
                #     f"{GRAPH_FOLDER}/{SPEEDUP_FOLDER}/{dataset}-{matrix}.png",
                # )
                plot_runtime_result(
                    results,
                    dataset,
                    matrix,
                    f"{GRAPH_FOLDER}/{RUNTIME_FOLDER}/{dataset}-{matrix}.png",
                )
            
            if dataset == "uniform":
                weak_scaling_plot(
                    results,
                    dataset,
                    f"{GRAPH_FOLDER}/{RUNTIME_FOLDER}/weak_scaling_{dataset}.png",
                )
