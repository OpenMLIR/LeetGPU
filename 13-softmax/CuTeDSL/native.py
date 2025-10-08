import cutlass.cute as cute
from cutlass.utils import SmemAllocator

@cute.kernel
def softmax_kernel(input: cute.Tensor, output: cute.Tensor, s_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bdimx, _, _ = cute.arch.block_dim()

    smem = SmemAllocator()

    max_shared = smem.allocate_tensor(input.element_type, s_layout, 16)
    sum_shared = smem.allocate_tensor(input.element_type, s_layout, 16)

    local_max = input[tidx]
    for i in range(tidx + bdimx, input.shape[0], bdimx):
        local_max = max(local_max, input[i])
    max_shared[tidx] = local_max
    cute.arch.barrier()

    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            v = max_shared[tidx + stride]
            if v > max_shared[tidx]:
                max_shared[tidx] = v
        cute.arch.barrier()
        stride = stride // 2
    max_val = max_shared[0]
    cute.arch.barrier()
    local_sum = 0.0
    for i in range(tidx, input.shape[0], bdimx):
        local_sum += cute.exp(input[i] - max_val)
    sum_shared[tidx] = local_sum
    cute.arch.barrier()

    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            sum_shared[tidx] = sum_shared[tidx] + sum_shared[tidx + stride]
        cute.arch.barrier()
        stride = stride // 2

    sum_exp = sum_shared[0]
    cute.arch.barrier()

    for i in range(tidx, input.shape[0], bdimx):
        output[i] = cute.exp(input[i] - max_val) / sum_exp

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    block_dim = 1024, 1, 1
    grid_dim = 1, 1, 1
    s_layout = cute.make_layout((1024), stride=(1))
    softmax_kernel(input, output, s_layout).launch(grid=grid_dim, block=block_dim)
