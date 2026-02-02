using Finch
using BenchmarkTools
using Base.Threads

function shard_add(A, B, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    
    ncpu = cpu(1, num_cpu)
    _C = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size(A)...)

    time = @belapsed begin
        ncpu = cpu(1, $(num_cpu))
        _C = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size($(_A))...)
        kernel($(_A), $(_B), _C, ncpu)
    end

    ncpu = cpu(1, num_cpu)
    _C = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), size(A)...)
    kernel(_A, _B, _C, ncpu)

    return (; time=time, C=_C)
end

function kernel(A, B, C::Tensor, ncpu)
    @finch_code mode = :fast begin
        C .= 0
        for j in parallel(_, ncpu, static_schedule())
            for i = _
                C[i, j] = A[i, j] + B[i, j]
            end
        end
    end
end

# function kernel_mod(A, B, C, n)
#     @finch mode = :fast begin
#         C .= 0
#         for j in parallel(_, cpu(1, n), static_schedule())
#             for i = _
#                 C[i, j] = A[i, j] + B[i, j]
#             end
#         end
#     end
# end

# from parallel_col_separate_sparselist_results.jl
# function parallel_col(A, B, C)
#     @finch_code mode = :fast begin
#         C .= 0
#         for j = parallel(_), i = _
#             C[i, j] = A[i, j] + B[i, j]
#         end
#     end
# end

# generated (finch_code):
#     C_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
#     C_lvl_2 = C_lvl.lvl
#     C_lvl_2_ptr = C_lvl_2.ptr
#     C_lvl_2_task = C_lvl_2.task
#     C_lvl_2_qos_fill = C_lvl_2.used
#     C_lvl_2_qos_stop = C_lvl_2.alloc
#     n = C_lvl_2.device.n
#     C_lvl_3 = C_lvl_2.lvl
#     C_lvl_3_ptr = C_lvl_3.ptr
#     C_lvl_3_idx = C_lvl_3.idx
#     C_lvl_3_val = C_lvl_3.val
#     C_lvl_3_tbl = C_lvl_3.tbl
#     C_lvl_3_pool = C_lvl_3.pool
#     C_lvl_4 = C_lvl_3.lvl
#     C_lvl_4_val = C_lvl_4.val
#     n_2 = (((ex.bodies[1]).bodies[2]).ext.args[2]).bind.n
#     A = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[1]).tns.bind
#     A_m = A.m
#     A_n = A.n
#     A_ptr = A.colptr
#     A_idx = A.rowval
#     A_val = A.nzval
#     B = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[2]).tns.bind
#     B_m = B.m
#     B_n = B.n
#     B_ptr = B.colptr
#     B_idx = B.rowval
#     B_val = B.nzval
#     B_m == A_m || throw(DimensionMismatch("mismatched dimension limits ($(B_m) != $(A_m))"))
#     A_n == B_n || throw(DimensionMismatch("mismatched dimension limits ($(A_n) != $(B_n))"))
#     C_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_4_val)
#     C_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_3_ptr)
#     C_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_3_idx)
#     C_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_3_val)
#     C_lvl_3_tbl_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_3_tbl)
#     C_lvl_3_pool_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_3_pool)
#     C_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_2_qos_fill)
#     C_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n)), C_lvl_2_qos_stop)
#     Threads.@threads :dynamic for tid = 1:n
#             Finch.@barrier begin
#                     @inbounds @fastmath(begin
#                                 alloced_pos = C_lvl_2_qos_stop_2[tid]
#                                 C_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_4_val)
#                                 C_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_3_ptr)
#                                 C_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_3_idx)
#                                 C_lvl_3_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_3_val)
#                                 C_lvl_3_tbl_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_3_tbl)
#                                 C_lvl_3_pool_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n), n), (Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)())), C_lvl_3_pool)
#                                 C_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)()), C_lvl_2_qos_fill)
#                                 C_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{$(QuoteNode(4))}(n), (Finch.SerialTask)()), C_lvl_2_qos_stop)
#                                 empty!(C_lvl_3_tbl_3)
#                                 empty!(C_lvl_3_pool_3)
#                                 resize!(C_lvl_3_ptr_3, alloced_pos + 1)
#                                 C_lvl_3_ptr_3[1] = 1
#                                 Finch.fill_range!(C_lvl_3_ptr_3, 0, 2, alloced_pos + 1)
#                                 pdx_tmp = Vector{Int64}(undef, length(C_lvl_3_tbl_3))
#                                 resize!(C_lvl_3_idx_3, length(C_lvl_3_tbl_3))
#                                 resize!(C_lvl_3_val_3, length(C_lvl_3_tbl_3))
#                                 idx_tmp = Vector{Int64}(undef, length(C_lvl_3_tbl_3))
#                                 val_tmp = Vector{Int64}(undef, length(C_lvl_3_tbl_3))
#                                 q = 0
#                                 for entry = pairs(C_lvl_3_tbl_3)
#                                     sugar_2 = entry[1]
#                                     p_2 = sugar_2[1]
#                                     i_5 = sugar_2[2]
#                                     v = entry[2]
#                                     q += 1
#                                     idx_tmp[q] = i_5
#                                     val_tmp[q] = v
#                                     pdx_tmp[q] = p_2
#                                     C_lvl_3_ptr_3[p_2 + 1] += 1
#                                 end
#                                 for p_2 = 2:alloced_pos + 1
#                                     C_lvl_3_ptr_3[p_2] += C_lvl_3_ptr_3[p_2 - 1]
#                                 end
#                                 perm = sortperm(idx_tmp)
#                                 ptr_2 = copy(C_lvl_3_ptr_3)
#                                 for q = perm
#                                     p_2 = pdx_tmp[q]
#                                     r = ptr_2[p_2]
#                                     C_lvl_3_idx_3[r] = idx_tmp[q]
#                                     C_lvl_3_val_3[r] = val_tmp[q]
#                                     ptr_2[p_2] += 1
#                                 end
#                                 qos_stop = C_lvl_3_ptr_3[alloced_pos + 1] - 1
#                                 resize!(C_lvl_4_val_3, qos_stop)
#                                 C_lvl_2_qos_fill_3[tid] = 0
#                                 C_lvl_2_qos_stop_3[tid] = alloced_pos
#                             end)
#                     nothing
#                 end
#         end
#     C_lvl_4_val = (Finch).transfer(C_lvl_4_val, C_lvl_4_val_2)
#     C_lvl_3_ptr = (Finch).transfer(C_lvl_3_ptr, C_lvl_3_ptr_2)
#     C_lvl_3_idx = (Finch).transfer(C_lvl_3_idx, C_lvl_3_idx_2)
#     C_lvl_3_val = (Finch).transfer(C_lvl_3_val, C_lvl_3_val_2)
#     C_lvl_3_tbl = (Finch).transfer(C_lvl_3_tbl, C_lvl_3_tbl_2)
#     C_lvl_3_pool = (Finch).transfer(C_lvl_3_pool, C_lvl_3_pool_2)
#     C_lvl_2_qos_fill = (Finch).transfer(C_lvl_2_qos_fill, C_lvl_2_qos_fill_2)
#     C_lvl_2_qos_stop = (Finch).transfer(C_lvl_2_qos_stop, C_lvl_2_qos_stop_2)
#     Finch.resize_if_smaller!(C_lvl_2_task, A_n)
#     Finch.resize_if_smaller!(C_lvl_2_ptr, A_n)
#     Finch.fill_range!(C_lvl_2_ptr, 0, 1, A_n)
#     C_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_4_val)
#     C_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_3_ptr)
#     C_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_3_idx)
#     C_lvl_3_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_3_val)
#     C_lvl_3_tbl_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_3_tbl)
#     C_lvl_3_pool_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_3_pool)
#     C_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_2_ptr)
#     C_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_2_task)
#     C_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_2_qos_fill)
#     C_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{$(QuoteNode(4))}(n_2)), C_lvl_2_qos_stop)
#     Threads.@threads :dynamic for tid_2 = 1:n_2
#             Finch.@barrier begin
#                     @inbounds @fastmath(begin
#                                 C_lvl_2_qos_fill_5 = C_lvl_2_qos_fill_4[tid_2]
#                                 C_lvl_2_qos_stop_5 = C_lvl_2_qos_stop_4[tid_2]
#                                 C_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_4_val_4)
#                                 C_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_3_ptr_4)
#                                 C_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_3_idx_4)
#                                 C_lvl_3_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_3_val_4)
#                                 C_lvl_3_tbl_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_3_tbl_4)
#                                 C_lvl_3_pool_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{$(QuoteNode(4))}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())), C_lvl_3_pool_4)
#                                 C_lvl_3_qos_stop = C_lvl_3_ptr_5[C_lvl_2_qos_stop_5 + 1] - 1
#                                 C_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)()), C_lvl_2_ptr_2)
#                                 C_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)()), C_lvl_2_task_2)
#                                 (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())
#                                 (Finch.CPUThread)(tid_2, Finch.CPU{$(QuoteNode(4))}(n_2), (Finch.SerialTask)())
#                                 res_26 = begin
#                                         phase_start_2 = max(1, 1 + fld(A_n * (tid_2 + -1), n_2))
#                                         phase_stop_2 = min(A_n, fld(A_n * tid_2, n_2))
#                                         if phase_stop_2 >= phase_start_2
#                                             for j_7 = phase_start_2:phase_stop_2
#                                                 C_lvl_q = (1 - 1) * A_n + j_7
#                                                 qos = C_lvl_2_ptr_3[C_lvl_q]
#                                                 if qos == 0
#                                                     qos = (C_lvl_2_qos_fill_5 += 1)
#                                                     C_lvl_2_task_3[C_lvl_q] = tid_2
#                                                     C_lvl_2_ptr_3[C_lvl_q] = C_lvl_2_qos_fill_5
#                                                     if C_lvl_2_qos_fill_5 > C_lvl_2_qos_stop_5
#                                                         C_lvl_2_qos_stop_5 = max(C_lvl_2_qos_stop_5 << 1, 1)
#                                                     end
#                                                 end
#                                                 A_q = A_ptr[j_7]
#                                                 A_q_stop = A_ptr[j_7 + 1]
#                                                 if A_q < A_q_stop
#                                                     A_i1 = A_idx[A_q_stop - 1]
#                                                 else
#                                                     A_i1 = 0
#                                                 end
#                                                 B_q = B_ptr[j_7]
#                                                 B_q_stop = B_ptr[j_7 + 1]
#                                                 if B_q < B_q_stop
#                                                     B_i1 = B_idx[B_q_stop - 1]
#                                                 else
#                                                     B_i1 = 0
#                                                 end
#                                                 phase_stop_3 = min(B_m, A_i1, B_i1)
#                                                 if phase_stop_3 >= 1
#                                                     i = 1
#                                                     if A_idx[A_q] < 1
#                                                         A_q = Finch.scansearch(A_idx, 1, A_q, A_q_stop - 1)
#                                                     end
#                                                     if B_idx[B_q] < 1
#                                                         B_q = Finch.scansearch(B_idx, 1, B_q, B_q_stop - 1)
#                                                     end
#                                                     while i <= phase_stop_3
#                                                         A_i = A_idx[A_q]
#                                                         B_i = B_idx[B_q]
#                                                         phase_stop_4 = min(B_i, phase_stop_3, A_i)
#                                                         if A_i == phase_stop_4 && B_i == phase_stop_4
#                                                             A_val_2 = A_val[A_q]
#                                                             B_val_2 = B_val[B_q]
#                                                             C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, phase_stop_4), 0)
#                                                             if C_lvl_3_qos == 0
#                                                                 if !(isempty(C_lvl_3_pool_5))
#                                                                     C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                 else
#                                                                     C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                     if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                         C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                         Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                     end
#                                                                 end
#                                                                 C_lvl_3_tbl_5[(qos, phase_stop_4)] = C_lvl_3_qos
#                                                             end
#                                                             C_lvl_4_val_5[C_lvl_3_qos] = B_val_2 + A_val_2
#                                                             C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                             A_q += 1
#                                                             B_q += 1
#                                                         elseif B_i == phase_stop_4
#                                                             B_val_2 = B_val[B_q]
#                                                             C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, phase_stop_4), 0)
#                                                             if C_lvl_3_qos == 0
#                                                                 if !(isempty(C_lvl_3_pool_5))
#                                                                     C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                 else
#                                                                     C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                     if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                         C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                         Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                     end
#                                                                 end
#                                                                 C_lvl_3_tbl_5[(qos, phase_stop_4)] = C_lvl_3_qos
#                                                             end
#                                                             C_lvl_4_val_5[C_lvl_3_qos] = B_val_2
#                                                             C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                             B_q += 1
#                                                         elseif A_i == phase_stop_4
#                                                             A_val_2 = A_val[A_q]
#                                                             C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, phase_stop_4), 0)
#                                                             if C_lvl_3_qos == 0
#                                                                 if !(isempty(C_lvl_3_pool_5))
#                                                                     C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                 else
#                                                                     C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                     if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                         C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                         Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                     end
#                                                                 end
#                                                                 C_lvl_3_tbl_5[(qos, phase_stop_4)] = C_lvl_3_qos
#                                                             end
#                                                             C_lvl_4_val_5[C_lvl_3_qos] = A_val_2
#                                                             C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                             A_q += 1
#                                                         end
#                                                         i = phase_stop_4 + 1
#                                                     end
#                                                 end
#                                                 phase_start_5 = max(1, 1 + A_i1)
#                                                 phase_stop_5 = min(B_m, B_i1)
#                                                 if phase_stop_5 >= phase_start_5
#                                                     if B_idx[B_q] < phase_start_5
#                                                         B_q = Finch.scansearch(B_idx, phase_start_5, B_q, B_q_stop - 1)
#                                                     end
#                                                     while true
#                                                         B_i = B_idx[B_q]
#                                                         if B_i < phase_stop_5
#                                                             B_val_2 = B_val[B_q]
#                                                             C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, B_i), 0)
#                                                             if C_lvl_3_qos == 0
#                                                                 if !(isempty(C_lvl_3_pool_5))
#                                                                     C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                 else
#                                                                     C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                     if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                         C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                         Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                     end
#                                                                 end
#                                                                 C_lvl_3_tbl_5[(qos, B_i)] = C_lvl_3_qos
#                                                             end
#                                                             C_lvl_4_val_5[C_lvl_3_qos] = B_val_2
#                                                             C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                             B_q += 1
#                                                         else
#                                                             phase_stop_7 = min(B_i, phase_stop_5)
#                                                             if B_i == phase_stop_7
#                                                                 B_val_2 = B_val[B_q]
#                                                                 C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, phase_stop_7), 0)
#                                                                 if C_lvl_3_qos == 0
#                                                                     if !(isempty(C_lvl_3_pool_5))
#                                                                         C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                     else
#                                                                         C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                         if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                             C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                             Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                             Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                             Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                             Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         end
#                                                                     end
#                                                                     C_lvl_3_tbl_5[(qos, phase_stop_7)] = C_lvl_3_qos
#                                                                 end
#                                                                 C_lvl_4_val_5[C_lvl_3_qos] = B_val_2
#                                                                 C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                                 B_q += 1
#                                                             end
#                                                             break
#                                                         end
#                                                     end
#                                                 end
#                                                 phase_start_7 = max(1, 1 + B_i1)
#                                                 phase_stop_8 = min(B_m, A_i1)
#                                                 if phase_stop_8 >= phase_start_7
#                                                     if A_idx[A_q] < phase_start_7
#                                                         A_q = Finch.scansearch(A_idx, phase_start_7, A_q, A_q_stop - 1)
#                                                     end
#                                                     while true
#                                                         A_i = A_idx[A_q]
#                                                         if A_i < phase_stop_8
#                                                             A_val_2 = A_val[A_q]
#                                                             C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, A_i), 0)
#                                                             if C_lvl_3_qos == 0
#                                                                 if !(isempty(C_lvl_3_pool_5))
#                                                                     C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                 else
#                                                                     C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                     if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                         C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                         Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                         Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                     end
#                                                                 end
#                                                                 C_lvl_3_tbl_5[(qos, A_i)] = C_lvl_3_qos
#                                                             end
#                                                             C_lvl_4_val_5[C_lvl_3_qos] = A_val_2
#                                                             C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                             A_q += 1
#                                                         else
#                                                             phase_stop_10 = min(A_i, phase_stop_8)
#                                                             if A_i == phase_stop_10
#                                                                 A_val_2 = A_val[A_q]
#                                                                 C_lvl_3_qos = get(C_lvl_3_tbl_5, (qos, phase_stop_10), 0)
#                                                                 if C_lvl_3_qos == 0
#                                                                     if !(isempty(C_lvl_3_pool_5))
#                                                                         C_lvl_3_qos = pop!(C_lvl_3_pool_5)
#                                                                     else
#                                                                         C_lvl_3_qos = length(C_lvl_3_tbl_5) + 1
#                                                                         if C_lvl_3_qos > C_lvl_3_qos_stop
#                                                                             C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
#                                                                             Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
#                                                                             Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                             Finch.resize_if_smaller!(C_lvl_3_val_5, C_lvl_3_qos_stop)
#                                                                             Finch.fill_range!(C_lvl_3_val_5, 0, C_lvl_3_qos, C_lvl_3_qos_stop)
#                                                                         end
#                                                                     end
#                                                                     C_lvl_3_tbl_5[(qos, phase_stop_10)] = C_lvl_3_qos
#                                                                 end
#                                                                 C_lvl_4_val_5[C_lvl_3_qos] = A_val_2
#                                                                 C_lvl_3_val_5[C_lvl_3_qos] = C_lvl_3_qos
#                                                                 A_q += 1
#                                                             end
#                                                             break
#                                                         end
#                                                     end
#                                                 end
#                                             end
#                                         end
#                                         phase_start_10 = max(1, 1 + fld(A_n * tid_2, n_2))
#                                         if A_n >= phase_start_10
#                                             A_n + 1
#                                         end
#                                     end
#                                 resize!(C_lvl_3_ptr_5, C_lvl_2_qos_stop_5 + 1)
#                                 C_lvl_3_ptr_5[1] = 1
#                                 Finch.fill_range!(C_lvl_3_ptr_5, 0, 2, C_lvl_2_qos_stop_5 + 1)
#                                 pdx_tmp_2 = Vector{Int64}(undef, length(C_lvl_3_tbl_5))
#                                 resize!(C_lvl_3_idx_5, length(C_lvl_3_tbl_5))
#                                 resize!(C_lvl_3_val_5, length(C_lvl_3_tbl_5))
#                                 idx_tmp_2 = Vector{Int64}(undef, length(C_lvl_3_tbl_5))
#                                 val_tmp_2 = Vector{Int64}(undef, length(C_lvl_3_tbl_5))
#                                 q_2 = 0
#                                 for entry_2 = pairs(C_lvl_3_tbl_5)
#                                     sugar_4 = entry_2[1]
#                                     p_5 = sugar_4[1]
#                                     i_6 = sugar_4[2]
#                                     v_2 = entry_2[2]
#                                     q_2 += 1
#                                     idx_tmp_2[q_2] = i_6
#                                     val_tmp_2[q_2] = v_2
#                                     pdx_tmp_2[q_2] = p_5
#                                     C_lvl_3_ptr_5[p_5 + 1] += 1
#                                 end
#                                 for p_5 = 2:C_lvl_2_qos_stop_5 + 1
#                                     C_lvl_3_ptr_5[p_5] += C_lvl_3_ptr_5[p_5 - 1]
#                                 end
#                                 perm_2 = sortperm(idx_tmp_2)
#                                 ptr_3 = copy(C_lvl_3_ptr_5)
#                                 for q_2 = perm_2
#                                     p_5 = pdx_tmp_2[q_2]
#                                     r_2 = ptr_3[p_5]
#                                     C_lvl_3_idx_5[r_2] = idx_tmp_2[q_2]
#                                     C_lvl_3_val_5[r_2] = val_tmp_2[q_2]
#                                     ptr_3[p_5] += 1
#                                 end
#                                 qos_stop_2 = C_lvl_3_ptr_5[C_lvl_2_qos_stop_5 + 1] - 1
#                                 resize!(C_lvl_4_val_5, qos_stop_2)
#                                 res_26
#                             end)
#                     nothing
#                 end
#         end
#     (C = Tensor((DenseLevel){Int64}((ShardLevel)(Finch.CPU{$(QuoteNode(4))}(n), (SparseDictLevel){Int64}(ElementLevel{0.0, Float64, Int64}(C_lvl_4_val_4), B_m, C_lvl_3_ptr_4, C_lvl_3_idx_4, C_lvl_3_val_4, C_lvl_3_tbl_4, C_lvl_3_pool_4), C_lvl_2_ptr_2, C_lvl_2_task_2, C_lvl_2_qos_fill_4, C_lvl_2_qos_stop_4, C_lvl_2.schedule), A_n)),)
# end

