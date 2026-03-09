using Finch
using BenchmarkTools
using SparseArrays


function parallel_mul_jdim_shd(A, B, C, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    B_t = transpose(Array(B))
    _B = Tensor(Dense(SparseList(Element(0.0))), B_t)

    dev_j = cpu(:j, num_cpu)
    _C = Tensor(Dense(Shard(dev_j, HashFormat(0.0))))

    time = @belapsed begin
        (_A, _B, _C, dev_j) = $(_A, _B, _C, dev_j)
        @finch mode = :fast begin
            _C .= 0
            for k = _, j = parallel(_, dev_j), i = _
                _C[i, j] += _A[i, k] * _B[j, k]
            end
        end
    end

    @finch mode = :fast begin
        _C .= 0
        for k = _, j = parallel(_, dev_j), i = _
            _C[i, j] += _A[i, k] * _B[j, k]
        end
    end

    return (; time=time, C=_C)
end

function parallel_mul_jdim_col(A, B, C, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    B_t = transpose(Array(B))
    _B = Tensor(Dense(SparseList(Element(0.0))), B_t)


    dev_j = cpu(:j, num_cpu)
    _C = Tensor(Dense(Coalesce(dev_j, HashFormat(0.0))))

    time = @belapsed begin
        (_A, _B, _C, dev_j) = $(_A, _B, _C, dev_j)
        @finch mode = :fast begin
            _C .= 0
            for k = _, j = parallel(_, dev_j), i = _
                _C[i, j] += _A[i, k] * _B[j, k]
            end
        end
    end

    @finch mode = :fast begin
        _C .= 0
        for k = _, j = parallel(_, dev_j), i = _
            _C[i, j] += _A[i, k] * _B[j, k]
        end
    end

    return (; time=time, C=_C)
end

# not to be tested yet
# or error in use case?
function parallel_mul_kdim(A, B, C, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    B_t = transpose(Array(B))
    _B = Tensor(Dense(SparseList(Element(0.0))), B_t)

    dev_k = cpu(:k, num_cpu)
    _C = Tensor(Coalesce(dev_k, Dense(HashFormat(0.0))))
    

    time = @belapsed begin
        (_A, _B, _C, dev_j) = $(_A, _B, _C, dev_j)
        @finch mode = :fast begin
            _C .= 0
            for k = parallel(_, dev_k), j = _, i = _
                _C[i, j] += _A[i, k] * _B[j, k]
            end
        end
    end

    @finch mode = :fast begin
        _C .= 0
        for k = parallel(_, dev_k), j = _, i = _
            _C[i, j] += _A[i, k] * _B[j, k]
        end
    end

    return (; time=time, C=_C)
end

