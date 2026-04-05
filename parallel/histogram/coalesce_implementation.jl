using Finch
using BenchmarkTools
using SparseArrays


function coalesce_impl(I, num_cpu)
    _R = Tensor(Dense(Dense(Element(Int64(0)))))
    _G = Tensor(Dense(Dense(Element(Int64(0)))))
    _B = Tensor(Dense(Dense(Element(Int64(0)))))

    @finch begin
        for i = _, j = _
            _R[i, j] = I[i, j][1]
            _G[i, j] = I[i, j][2]
            _B[i, j] = I[i, j][3]
        end
    end

    dev = cpu(:t, num_cpu)
    h = Tensor(Coalesce(dev, Dense(Dense(HashFormat(Int64(0))))))

    @finch mode = :fast begin
        h .= 0
        for j = parallel(_, dev)
            for i = _
                let r = _R[i, j], g = _G[i, j], b = _B[i, j]
                    h[r, g, b] += 1
                end
            end
        end
    end

    @finch mode = :fast begin
        h .= 0
        for j = parallel(_, dev)
            for i = _
                let r = I[i, j][1], g = I[i, j][2], b = I[i, j][3]
                    h[r, g, b] += 1
                end
            end
        end
    end

    
# @finch begin
#   h .= 0  
#   for j = parallel(_)
#     for i = _
#       let rgb = img[vi,j]
#         h[rgb[1],rgb[2],rgb[3]] += 1
#       end
#     end
#   end
# end


    # in the future, assumed stored in runlength encoding
    # _R = Tensor(Dense(SparseList(Element(0.0))), R)
    # _G = Tensor(Dense(SparseList(Element(0.0))), G)
    # _B = Tensor(Dense(SparseList(Element(0.0))), B)

    # adjust thread numbers
    # dev_r = cpu(:t, 2)
    # dev_g = cpu(:g, 2)
    # dev_b = cpu(:b, 2)
    # dev = cpu(:t, num_cpu)
    
    # coalesce should be on the outer level but its not supported yet?
    # _H = Tensor(Coalesce(dev, SparseList(SparseList(SparseList(Element(0))))))

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

    # time = @belapsed begin
    #     (_R, _G, _B, _H, dev) = $(_R, _G, _B, _H, dev)
    #     @finch mode = :fast begin
    #         _H .= 0
    #         for r = parallel(1:256, dev_r), g = 1:256, b = 1:256
    #             for x = _, y = _
    #                 if _R[x, y] == r && _G[x, y] == g && _B[x, y] == b
    #                     _H[r, g, b] += 1
    #                 end
    #             end
    #         end
    #     end
    # end

    # @finch mode = :fast begin
    #     _H .= 0
    #     for r = parallel(1:256, dev_r), g = 1:256, b = 1:256
    #         for x = _, y = _
    #             if _R[x, y] == r && _G[x, y] == g && _B[x, y] == b
    #                 _H[r, g, b] += 1
    #             end
    #         end
    #     end
    # end

    return (; time=time, H=_H)
end

#######################

    # Finch limitation where it doesn't support tuples within tensors?
    I = [(1, 2, 3) (4, 5, 6); (7, 8, 9) (10, 11, 12)]

    dev = cpu(:t, num_cpu)
    h = Tensor(Coalesce(dev, Dense(Dense(HashFormat(Int64(0))))))

    @finch mode = :fast begin
        h .= 0
        for j = parallel(_, dev)
            for i = _
                let r = I[i, j][1], g = I[i, j][2], b = I[i, j][3]
                    h[r, g, b] += 1
                end
            end
        end
    end


    # coalesce merge error?
    I = [(1, 2, 3) (4, 5, 6)
        (7, 8, 9) (10, 11, 12)]
    
    row, col = size(I)

    R_I = [I[i,j][1] for i in 1:row, j in 1:col]
    G_I = [I[i,j][2] for i in 1:row, j in 1:col]
    B_I = [I[i,j][3] for i in 1:row, j in 1:col]

    _R = Tensor(Dense(Dense(Element(0))), R_I)
    _G = Tensor(Dense(Dense(Element(0))), G_I)
    _B = Tensor(Dense(Dense(Element(0))), B_I)


    dev = cpu(:t, num_cpu)
    h = Tensor(Coalesce(dev, Dense(Dense(HashFormat(Int64(0))))))

    @finch mode = :fast begin
        h .= 0
        for j = parallel(_, dev)
            for i = _
                let r = _R[i, j], g = _G[i, j], b = _B[i, j]
                    h[r, g, b] += 1
                end
            end
        end
    end