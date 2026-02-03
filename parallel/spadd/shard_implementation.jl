using Finch
using BenchmarkTools
using Base.Threads


function shard_add(A, B, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    time = @belapsed begin
        cpu_dev = cpu(:id, $num_cpu)
        _C = Tensor(Dense(Shard(cpu_dev, SparseList(Element(0.0)))))

        kernel!($_A, $_B, _C, cpu_dev)
    end

    cpu_dev = cpu(:id, num_cpu)
    _C = Tensor(Dense(Shard(cpu_dev, SparseList(Element(0.0)))))

    kernel!(_A, _B, _C, cpu_dev)

    return (; time=time, C=_C)
end

function kernel!(A, B, C, cpu_dev)
    @finch mode = :fast begin
        C .= 0
        for j = parallel(_, cpu_dev)
            for i = _
                C[i, j] = A[i, j] + B[i, j]
            end
        end
    end
end
