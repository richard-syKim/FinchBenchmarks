using Finch
using BenchmarkTools
using Base.Threads

function shard_add(A, B, num_cpu)
    # change num of cpus
    ncpu = cpu(num_cpu)
    _A = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size(A)...)
    @finch begin
        _A .= 0
        for j in parallel(_, ncpu)
            for i in _
                _A[i, j] = A[i, j]
            end
        end
    end
    _B = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size(B)...)
    @finch begin
        _B .= 0
        for j in parallel(_, ncpu)
            for i in _
                _B[i, j] = B[i, j]
            end
        end
    end
    
    _C = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size(A)...)

    time = @belapsed begin
        (_A, _B, _C, ncpu) = $(_A, _B, _C, ncpu)
        @finch mode = :fast begin
            _C .= 0
            for j in parallel(_, ncpu)
                for i = _
                    _C[i, j] = _A[i, j] + _B[i, j]
                end
            end
        end
    end

    # C_n = Tensor(Dense(SparseList(Element(0.0))))
    # @finch begin
    #     C_n .= 0
    #       for i in _, j in _
    #          C_n[i, j] = _C[i, j]
    #       end
    # end
    return (; time=time, C=_C)
end
