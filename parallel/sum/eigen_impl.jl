const EIGEN_LIB = joinpath(@__DIR__, "libeigen_kernel.so")

function eigen_impl(v, num_cpu)

    # converting to 0-based Int32
    _v = Array(v)
    dim = Int32(length(_v))
    out_sum = Ref{Float64}(0.0)
    n_threads = Int32(num_cpu)

    elapsed = @ccall EIGEN_LIB.eigen_sum(_v :: Ptr{Float64}, dim :: Int32, out_sum :: Ptr{Float64}, n_threads :: Int32) :: Float64

    return (; time = elapsed, s = out_sum[])
end