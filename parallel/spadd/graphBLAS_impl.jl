using SuiteSparseGraphBLAS
using SparseArrays
using BenchmarkTools

function graphblas_impl(A, B, num_cpu)

    # SuiteSparse's internal thread pool
    SuiteSparseGraphBLAS.gbset(:nthreads, num_cpu)

    _A = GBMatrix(Matrix{Float64}(A))
    _B = GBMatrix(Matrix{Float64}(B))

    _C = GBMatrix{Float64}

    time = @belapsed begin
        (_A, _B, _C) = $(_A, _B, _C)
        
        # element-wise addition
        _C = eadd(_A, _B, +)
    end
    
    _C = eadd(_A, _B, +)
    C_sparse = SparseMatrixCSC(_C)

    return (; time=time, C=C_sparse)
end