import json
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"

NTHREADS = [2**i for i in range(6)] # Modify based on how many threads were tested

DEFAULT_METHOD = "serial_default_implementation"
SHARD_METHOD = "shard_implementation"
METHODS = [
    DEFAULT_METHOD,
    # "parallel_col_separate_sparselist_results",
    "separated_memory_concatenate_results",
    SHARD_METHOD,
]

DATASETS = [
    # {"uniform_a": [
    #     "1024-0.1", "2048-0.1", "4096-0.1", "8192-0.1", "16384-0.1", "32768-0.1", 
    #     "65536-0.1", "131072-0.1"
    # ]},
    # {"uniform_b": ["10000-0.00001", "10000-0.1"]},
    {"uniform_a": ["1024-0.1", "131072-0.1"]},
    {"uniform_b": ["10000-0.00001", "10000-0.1"]},
    {"FEMLAB": ["FEMLAB-poisson3Da", "FEMLAB-poisson3Db"]},
]

COLORS = ["red", "gray", "cadetblue", "saddlebrown", "navy", "orange","black"]


def format_sparsity(x: float) -> str:
    # remove scientific notation, remove trailing zeros
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return s


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spadd_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            m = result["matrix"]

            if isinstance(m, str):
                matrix = m.replace("/", "-")
            elif isinstance(m, dict):
                matrix = f"{m['size']}-{format_sparsity(m['sparsity'])}"
            else:
                raise TypeError(f"Unknown matrix format: {type(m)}")

            # matrix = (
            #     result["matrix"].replace("/", "-")
            #     if (result["dataset"] != "uniform_a" or result["dataset"] != "uniform_b")
            #     else f"{result['matrix']['size']}-{result['matrix']['sparsity']}"
            # )
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


# def weak_scaling_plot(results, dataset, save_location):
#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         NTHREADS,
#         [
#             results[dataset][f"{4096 * n_thread}-0.1"][SHARD_METHOD][n_thread]
#             for n_thread in NTHREADS
#         ],
#         label="shard_implementation",
#         color="grey",
#         marker="o",
#         linestyle="-",
#         linewidth=1,
#     )

#     plt.title(f"SpAdd - Weak Scaling with 10,000 x 4096 matrix per thread for {dataset}")
#     plt.xscale("log", base=2)
#     plt.xticks(NTHREADS)
#     plt.xlabel("Number of Threads")
#     plt.ylabel(f"Runtime (in seconds)")

#     plt.legend()
#     plt.savefig(save_location)
    


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
            
            # if dataset == "uniform_a":
            #     weak_scaling_plot(
            #         results,
            #         dataset,
            #         f"{GRAPH_FOLDER}/{RUNTIME_FOLDER}/weak_scaling_{dataset}.png",
            #     )
