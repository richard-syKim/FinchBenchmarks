using Finch
using BenchmarkTools


function finch_single_parallel_implementation_mul(A, B, C, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    _C = Tensor(Dense(Dense(Element(0.0))), C)

    time = @belapsed begin
        (_A, _B, _C) = $(_A, _B, _C)
        @finch_code mode = :fast begin
            for j = parallel(_), k = _, i = _
                _C[i, j] += _A[i, k] * _B[k, j]
            end
        end
    end

    return (; time=time, C=_C)
end
