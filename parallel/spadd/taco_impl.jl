using SparseArrays

const TACO_LIB = joinpath(@__DIR__, "libtaco_kernel.so")

function taco_impl(A, B, num_cpu)

    _a = collect(transpose(Matrix{Float64}(A)))
    _b = collect(transpose(Matrix{Float64}(B)))
    rows, cols = Int32(size(_a, 1)), Int32(size(_a, 2))
    n_threads  = Int32(num_cpu)

    c_pos  = Ref{Ptr{Int32}}(C_NULL)
    c_crd  = Ref{Ptr{Int32}}(C_NULL)
    c_vals = Ref{Ptr{Float64}}(C_NULL)

    elapsed = @ccall TACO_LIB.taco_add(
        _a     :: Ptr{Float64},   # const double** a  (row-major)
        _b     :: Ptr{Float64},   # const double** b
        rows   :: Int32,
        cols   :: Int32,
        c_pos  :: Ptr{Ptr{Int32}},
        c_crd  :: Ptr{Ptr{Int32}},
        c_vals :: Ptr{Ptr{Float64}},
        n_threads :: Int32
    ) :: Float64

    n_rows = Int(rows)
    pos_arr  = unsafe_wrap(Array, c_pos[],  n_rows + 1)  # 0-based offsets
    nnz      = Int(pos_arr[end])
    crd_arr  = unsafe_wrap(Array, c_crd[],  nnz)
    vals_arr = unsafe_wrap(Array, c_vals[], nnz)

    I_idx = Int32[]
    J_idx = Int32[]
    V     = Float64[]
    for i in 1:n_rows
        for p in (pos_arr[i] + 1) : pos_arr[i + 1]   # 1-based range into crd/vals
            push!(I_idx, Int32(i))
            push!(J_idx, Int32(crd_arr[p] + 1))       # 0-based col → 1-based
            push!(V,     vals_arr[p])
        end
    end
    C_sparse = sparse(I_idx, J_idx, V, n_rows, Int(cols))

    return (; time = elapsed, C = C_sparse)
end
