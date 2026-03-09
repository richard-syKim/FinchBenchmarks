using Finch
using BenchmarkTools
using SparseArrays


function coalesce_impl(R, G, B, num_cpu)
    # in the future, assumed stored in runlength encoding
    _R = Tensor(Dense(SparseList(Element(0.0))), R)
    _G = Tensor(Dense(SparseList(Element(0.0))), G)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    # adjust thread numbers
    # dev_r = cpu(:t, 2)
    # dev_g = cpu(:g, 2)
    # dev_b = cpu(:b, 2)
    dev = cpu(:t, num_cpu)
    
    # coalesce should be on the outer level but its not supported yet?
    _H = Tensor(Coalesce(dev, SparseList(SparseList(SparseList(Element(0))))))

    # time = @belapsed begin
    #     (_R, _G, _B, _H, dev) = $(_R, _G, _B, _H, dev)
    #     @finch mode = :fast begin
    #         _H .= 0
    #         for r = parallel(1:256, dev_r), g = parallel(1:256, dev_g), b = parallel(1:256, dev_b)
    #             for x = _, y = _
    #                 if _R[x, y] == r && _G[x, y] == g && _B[x, y] == b
    #                     _H[r, g, b] += 1
    #                 end
    #             end
    #         end
    #     end
    # end

    time = @belapsed begin
        (_R, _G, _B, _H, dev) = $(_R, _G, _B, _H, dev)
        @finch mode = :fast begin
            _H .= 0
            for r = parallel(1:256, dev_r), g = 1:256, b = 1:256
                for x = _, y = _
                    if _R[x, y] == r && _G[x, y] == g && _B[x, y] == b
                        _H[r, g, b] += 1
                    end
                end
            end
        end
    end

    @finch mode = :fast begin
        _H .= 0
        for r = parallel(1:256, dev_r), g = 1:256, b = 1:256
            for x = _, y = _
                if _R[x, y] == r && _G[x, y] == g && _B[x, y] == b
                    _H[r, g, b] += 1
                end
            end
        end
    end

    return (; time=time, H=_H)
end