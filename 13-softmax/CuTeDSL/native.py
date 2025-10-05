import cutlass
import cutlass.cute as cute

from cutlass.utils import SmemAllocator

@cute.kernel
def naive_softmax_kernel(
    inp: cute.Tensor,
    out: cute.Tensor,
    N: cute.Int32
):
    tidx, _, _ = cute.arch.thread_idx()
    _, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    smem = SmemAllocator()
    s_layout = cute.make_layout((1024), stride=(1))
    max_shared = smem.allocate_tensor(inp.element_type, s_layout, 16)
    sum_shared = smem.allocate_tensor(inp.element_type, s_layout, 16)

    local_max = inp[tidx]
    for i in range(tidx + bdimx, N, bdimx):
        local_max = max(local_max, inp[i])

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
    for i in range(tidx, N, bdimx):
        local_sum += cute.exp(inp[i] - max_val)

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

    for i in range(tidx, N, bdimx):
        out[i] = cute.exp(inp[i] - max_val) / sum_exp

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    num_threads_per_block = 1024
    kernel = naive_softmax_kernel(input, output, N)
    kernel.launch(grid=((N + 1023) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))
