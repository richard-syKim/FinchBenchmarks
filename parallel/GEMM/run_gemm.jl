using Base: nothing_sentinel
#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Random

Random.seed!(1234)

# Parsing Arguments
s = ArgParseSettings("Run Parallel GEMM Experiments.")
@add_arg_table! s begin
   "--ncpu"
    help = "number of CPUs"
    arg_type = Int
    "--output", "-o"
    arg_type = String
    help = "output file path"
    "--dataset", "-d"
    arg_type = String
    help = "dataset keyword"
    "--method", "-m"
    arg_type = String
    help = "method keyword"
    "--accuracy-check", "-a"
    action = :store_true
    help = "check method accuracy"
end
parsed_args = parse_args(ARGS, s)

# Mapping from dataset types to datasets
datasets = Dict(
    "uniform_a" => [
        OrderedDict("size" => 1024, "sparsity" => 0.01),
        # OrderedDict("size" => 2048, "sparsity" => 0.01),
        # OrderedDict("size" => 4096, "sparsity" => 0.01),
        # OrderedDict("size" => 8192, "sparsity" => 0.01),
        OrderedDict("size" => 16384, "sparsity" => 0.01),
    ],
    "uniform_b" => [
        OrderedDict("size" => 10_000, "sparsity" => 0.000001),
        # OrderedDict("size" => 10_000, "sparsity" => 0.00001),
        # OrderedDict("size" => 10_000, "sparsity" => 0.0001),
        # OrderedDict("size" => 10_000, "sparsity" => 0.001),
        OrderedDict("size" => 10_000, "sparsity" => 0.01),
    ],
    "FEMLAB" => [
        "FEMLAB/poisson3Da",
    ],
    # potentially add another matrix from matrixdepot
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
include("handtuned_threads_implementation.jl")
include("finch_single_parallel_implementation.jl")
include("shard_implementation.jl")


methods = OrderedDict(
    # "serial_default_implementation" => serial_default_implementation_mul,
    # "handtuned_threads_implementation" => handtuned_threads_implementation_mul,
    # "finch_single_parallel_implementation" => finch_single_parallel_implementation_mul,
    "shard_implementation" => shard_implementation_mul,
)

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        # Get relevant matrix
        if dataset == "uniform_a"
            A = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
            B = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
            C = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
        elseif dataset == "uniform_b"
            A = fsprand(5_000, 5_000, mtx["sparsity"])
            B = fsprand(5_000, 5_000, mtx["sparsity"])
            C = fsprand(5_000, 5_000, mtx["sparsity"])
        elseif dataset == "FEMLAB"
            A = matrixdepot(mtx)
            row_permutation = randperm(size(A, 1))
            col_permutation = randperm(size(A, 2))
            B = A[row_permutation, col_permutation]
            # potentially change C to something else
            C = matrixdepot(mtx)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, B, C, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the multiplication
                serial_default_implementation_result = serial_default_implementation_mul(A, B, C, ncpu)
                @assert result.C == serial_default_implementation_result.C "Incorrect result for $key"
            end

            # Write result
            time = result.time
            @info "result for $key on $mtx" time
            push!(results, OrderedDict(
                "time" => time,
                "n_threads" => Threads.nthreads(),
                "method" => key,
                "dataset" => dataset,
                "matrix" => mtx,
            ))

            if isnothing(parsed_args["output"])
                write("results/gemm_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
            else
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end

results = []
if isnothing(parsed_args["dataset"])
    for (dataset, mtxs) in datasets
        calculate_results(dataset, mtxs, results)
    end
else
    dataset = parsed_args["dataset"]
    mtxs = datasets[dataset]
    calculate_results(dataset, mtxs, results)
end
