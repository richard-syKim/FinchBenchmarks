using Finch
using BenchmarkTools
using SparseArrays


function shard_implementation_mul(A, B, C, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    dev_j = cpu(:j, num_cpu)
    _C = Tensor(Dense(Shard(dev_j, SparseList(Element(0.0)))))

    @finch begin
        _C .= 0
        for j = parallel(_, dev_j), i = _
            _C[i, j] = C[i, j]
        end
    end

    # parallel on one axis
    time = @belapsed begin
        (_A, _B, _C, C, dev_j) = $(_A, _B, _C, C, dev_j)
        @finch mode = :fast begin
            for j = parallel(_, dev_j), k = _, i = _
                _C[i, j] += _A[i, k] * _B[k, j]
            end
        end
    end

    @finch mode = :fast begin
        for j = parallel(_, dev_j), k = _, i = _
            _C[i, j] += _A[i, k] * _B[k, j]
        end
    end


    # dev_i = cpu(:i, num_cpu - div(num_cpu, 2))
    # dev_j = cpu(:j, div(num_cpu, 2))
    # _C = Tensor(Dense(Shard(dev_j, SparseList(Element(0.0)))))

    # @finch begin
    #     _C .= 0
    #     for i = parallel(_, dev_i)
    #         for j = parallel(_, dev_j)
    #             _C[i, j] = C[i, j]
    #         end
    #     end
    # end

    # # parallel on two axis
    # time = @belapsed begin
    #     (_A, _B, _C, C, dev_i, dev_j) = $(_A, _B, _C, C, dev_i, dev_j)
    #     @finch mode = :fast begin
    #         for i = parallel(_, dev_i)
    #             for j = parallel(_, dev_j)
    #                 for k = _
    #                     _C[i, j] += _A[i, k] * _B[k, j]
    #                 end
    #                 _C[i, j] += C[i, j]
    #             end
    #         end
    #     end
    # end

    # @finch mode = :fast begin
    #     for i = parallel(_, dev_i)
    #         for j = parallel(_, dev_j)
    #             for k = _
    #                 _C[i, j] += _A[i, k] * _B[k, j]
    #             end
    #             _C[i, j] += C[i, j]
    #         end
    #     end
    # end

    return (; time=time, C=_C)
end

# REPL
# A = sprand(10000, 10000, 0.01)
# B = sprand(10000, 10000, 0.01)
# C = sprand(10000, 10000, 0.01)
# _A = Tensor(Dense(SparseList(Element(0.0))), A)
# _B = Tensor(Dense(SparseList(Element(0.0))), B)
# dev_j = cpu(:j, 2)
# _C = Tensor(Dense(Shard(dev_j, SparseList(Element(0.0)))))

# @finch begin
#     dev_i = cpu(:i, 2)
#     _C .= 0
#     for j = parallel(_, dev_j)
#         for i = parallel(_, dev_i)
#             _C[i, j] = C[i, j]
#         end
#     end
# end
