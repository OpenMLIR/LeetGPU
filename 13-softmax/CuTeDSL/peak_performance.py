import cutlass.cute as cute
from cutlass.utils import SmemAllocator

@cute.kernel
def softmax_log_sum_exp_kernel(input: cute.Tensor, output: cute.Tensor, s_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    gdimx, _, _ = cute.arch.grid_dim()

    local_max = float('-inf')
    local_sum = 0.0
    for i in range(tidx, input.shape[0], bdimx):
        x = input[i]
        new_max = max(local_max, x)
        new_sum = 0.0
        if local_max == float('-inf'):
            new_sum = cute.exp(x - new_max)
        else:
            new_sum = cute.exp(local_max - new_max) * local_sum + cute.exp(x - new_max)
        local_max = new_max
        local_sum = new_sum

    smem = SmemAllocator()
    smax = smem.allocate_tensor(input.element_type, s_layout, 16)
    ssum = smem.allocate_tensor(input.element_type, s_layout, 16)
    smax[tidx] = local_max
    ssum[tidx] = local_sum
    cute.arch.barrier()

    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            a_max, b_max = smax[tidx], smax[tidx + stride]
            a_sum, b_sum = ssum[tidx], ssum[tidx + stride]
            new_max, new_sum = max(a_max, b_max), 0.0
            if a_max == float('-inf'):
                new_sum = b_sum
            elif b_max == float('-inf'):
                new_sum = a_sum
            else:
                new_sum = cute.exp(a_max - new_max) * a_sum + cute.exp(b_max - new_max) * b_sum;
            smax[tidx], ssum[tidx] = new_max, new_sum
        cute.arch.barrier()
        stride = stride // 2

    max_val, sum_exp = smax[0], ssum[0]
    cute.arch.barrier()

    idx = bidx * bdimx + tidx
    for i in range(idx, input.shape[0], bdimx * gdimx):
        output[i] = cute.exp(input[i] - max_val) / sum_exp

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    block_dim = 1024, 1, 1
    grid_dim = 148, 1, 1
    s_layout = cute.make_layout((1024), stride=(1))
    softmax_log_sum_exp_kernel(input, output, s_layout).launch(grid=grid_dim, block=block_dim)
