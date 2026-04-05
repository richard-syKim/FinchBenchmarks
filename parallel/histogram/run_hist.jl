using Base: nothing_sentinel
#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Images
using Colors
using OrderedCollections

# Parsing Arguments
s = ArgParseSettings("Run Histogram Experiments.")
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
    "image" => [
        OrderedDict("path" => "img/sc1.jpg"),
        OrderedDict("path" => "img/sc2.png"),
    ],
)

include("coalesce_impl.jl")



methods = OrderedDict(
    "coalesce_impl" => coalesce_impl,


if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        if dataset == "image"
            m = load_image_as_rgb_matrix(mtx["path"])
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(m, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the sum
                coalesce_impl_result = coalesce_impl(v, ncpu)
                @assert result.s == coalesce_impl_result.s "Incorrect result for $key: $result : $coalesce_impl_result"
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
                write("results/spadd_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
            else
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end

function load_image_as_rgb_matrix(filepath::String)
    img = load(filepath)
    img_rgb = RGB.(img)
    
    height, width = size(img_rgb)
    rgb_matrix = [(
        round(Int, red(pixel) * 255),
        round(Int, green(pixel) * 255),
        round(Int, blue(pixel) * 255)
    ) for pixel in img_rgb]
    
    return reshape(rgb_matrix, height, width)
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


