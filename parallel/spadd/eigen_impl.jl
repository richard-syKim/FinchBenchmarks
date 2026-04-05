using SparseArrays

const EIGEN_LIB = joinpath(@__DIR__, "libeigen_kernel.so")

function eigen_impl(A, B, num_cpu)

    _a = collect(transpose(Matrix{Float64}(A)))
    _b = collect(transpose(Matrix{Float64}(B)))
    rows, cols = Int32(size(_a, 1)), Int32(size(_a, 2))
    n_threads  = Int32(num_cpu)

    c_outer = Ref{Ptr{Int32}}(C_NULL)
    c_inner = Ref{Ptr{Int32}}(C_NULL)
    c_vals  = Ref{Ptr{Float64}}(C_NULL)
    c_nnz   = Ref{Int32}(0)

    elapsed = @ccall EIGEN_LIB.eigen_add(
        _a      :: Ptr{Float64},
        _b      :: Ptr{Float64},
        rows    :: Int32,
        cols    :: Int32,
        c_outer :: Ptr{Ptr{Int32}},
        c_inner :: Ptr{Ptr{Int32}},
        c_vals  :: Ptr{Ptr{Float64}},
        c_nnz   :: Ptr{Int32},
        n_threads :: Int32
    ) :: Float64

    # Eigen's CSR to SparseMatrix CSR
    nnz      = Int(c_nnz[])
    outer    = unsafe_wrap(Array, c_outer[], rows + 1)   # 0-based row offsets
    inner    = unsafe_wrap(Array, c_inner[], nnz)        # 0-based col indices
    vals_arr = unsafe_wrap(Array, c_vals[],  nnz)

    I_idx = Int32[]
    J_idx = Int32[]
    V     = Float64[]
    for i in 1:Int(rows)
        for p in (outer[i] + 1) : outer[i + 1]
            push!(I_idx, Int32(i))
            push!(J_idx, Int32(inner[p] + 1))   # 0-based → 1-based
            push!(V,     vals_arr[p])
        end
    end
    C_sparse = sparse(I_idx, J_idx, V, Int(rows), Int(cols))

    return (; time=elapsed, C=C_sparse)
end