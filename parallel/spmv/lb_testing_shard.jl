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
using Random
using Finch
using SparseArrays


function spmv_static(A, x, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)

    cpu_dev = cpu(:id, num_cpu)
    _y = Tensor(Dense(Shard(cpu_dev, Element(0.0))))

    time = @belapsed begin
        (_A, _x, _y, cpu_dev) = $(_A, _x, _y, cpu_dev)

        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu_dev, static_schedule())
                for i = _
                    _y[j] += _A[i, j] * _x[i]
                end
            end
        end
    end

    @finch mode = :fast begin
        _y .= 0
        for j = parallel(_, cpu_dev, static_schedule())
            for i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end

    return (; comp_time=time, tot_time=time, y=_y)
end


function spmv_greedy(A, x, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)

    cpu_dev = cpu(:id, num_cpu)
    _y = Tensor(Dense(Shard(cpu_dev, Element(0.0))))

    time = @belapsed begin
        (_A, _x, _y, cpu_dev) = $(_A, _x, _y, cpu_dev)

        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu_dev, greedy_schedule())
                for i = _
                    _y[j] += _A[i, j] * _x[i]
                end
            end
        end
    end

    @finch mode = :fast begin
        _y .= 0
        for j = parallel(_, cpu_dev, greedy_schedule())
            for i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end

    return (; comp_time=time, tot_time=time, y=_y)
end


function spmv_julia(A, x, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)

    cpu_dev = cpu(:id, num_cpu)
    _y = Tensor(Dense(Shard(cpu_dev, Element(0.0))))

    time = @belapsed begin
        (_A, _x, _y, cpu_dev) = $(_A, _x, _y, cpu_dev)

        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu_dev, julia_schedule())
                for i = _
                    _y[j] += _A[i, j] * _x[i]
                end
            end
        end
    end

    @finch mode = :fast begin
        _y .= 0
        for j = parallel(_, cpu_dev, julia_schedule())
            for i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end

    return (; comp_time=time, tot_time=time, y=_y)
end


function spmv_lb(A, x, num_cpu)
    _A_t = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)

    cpu_dev = cpu(:id, num_cpu)
    _, n = size(A)
    _y = Tensor(Dense(Shard(cpu_dev, Element(0.0))), n)

    tot_time = @belapsed begin
        (_A_t, _x, _y, cpu_dev, num_cpu) = $(_A_t, _x, _y, cpu_dev, num_cpu)

        _A = loadbalanced_mat(_A_t, num_cpu)

        @finch mode = :fast begin
            for j = parallel(_, cpu_dev)
                for i = _
                    _y[j] += _A[i, j] * _x[i]
                end
            end
        end

        _y_r = loadbalanced_vec(_y, num_cpu, cpu_dev)
    end

    _A = loadbalanced_mat(_A_t, num_cpu)

    comp_time = @belapsed begin
        (_A, _x, _y, cpu_dev, num_cpu) = $(_A, _x, _y, cpu_dev, num_cpu)

        @finch mode = :fast begin
            for j = parallel(_, cpu_dev)
                for i = _
                    _y[j] += _A[i, j] * _x[i]
                end
            end
        end
    end

    @finch mode = :fast begin
        for j = parallel(_, cpu_dev)
            for i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end
    _y_r = loadbalanced_vec(_y, num_cpu, cpu_dev)

    return (; comp_time=comp_time, tot_time=tot_time, y=_y_r)
end


function loadbalanced_mat(A, num_cpu)
    m, n = size(A)

    @assert n % num_cpu == 0 "Must be divisible by num_cpu"

    block = div(n, num_cpu)

    A_lb = Tensor(Dense(SparseList(Element(0.0))), m, n)

    @finch mode = :fast begin
        for j in _
            for i in _
                A_lb[i, ((j - 1) % num_cpu) * block + div(j - 1, num_cpu) + 1] = A[i, j]
            end
        end
    end

    return A_lb
end


function loadbalanced_vec(y, num_cpu, cpu_dev)
    n, = size(y)
    y_org = Tensor(Dense(Element(0.0)), n)

    @assert n % num_cpu == 0 "Must be divisible by num_cpu"

    block = div(n, num_cpu)

    @finch mode = :fast begin
        for j = parallel(_, cpu_dev)
            y_org[j] = y[((j - 1) % block) * num_cpu + div(j - 1, block) + 1]
        end
    end

    return y_org
end


function main()
    Random.seed!(1234)

    s = ArgParseSettings("spmv")
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

    ncpu = parsed_args["ncpu"]

    # Transposed matrix
    block_cols = 10_000
    ncols = block_cols * ncpu
    nrows = block_cols

    A = spzeros(nrows, ncols)


    for j = 1:ncols
        col_density = exp(-(div(j, ncpu) * ncpu) / ncols)
        nnz_col = max(1, round(Int, col_density * nrows * 0.125))

        for i = 1:nnz_col
            A[rand(1:nrows), j] = rand()
        end
    end

    x = rand(nrows)

    methods = OrderedDict(
        "static" => spmv_static,
        "greedy" => spmv_greedy,
        "julia" => spmv_julia,
        "load_balanced" => spmv_lb
    )

    results = []

    for (key, method) in methods

        result = method(A, x, ncpu)

        if parsed_args["accuracy-check"]
            ref = spmv_static(A, x, ncpu)
            @assert isapprox(result.y, ref.y) "Incorrect result for $key"
        end

        push!(results, OrderedDict(
            "computation_time" => result.comp_time,
            "total_time" => result.tot_time,
            "n_threads" => ncpu,
            "method" => key,
            "matrix_type" => "log_skewed",
            "rows" => nrows,
            "cols" => ncols,
        ))

        @info "Result for $key" result.comp_time result.tot_time
    end

    output_file = isnothing(parsed_args["output"]) ?
        "results/loadbalance/spmv_$(ncpu)_threads.json" :
        parsed_args["output"]

    write(output_file, JSON.json(results, 4))
end

main()
