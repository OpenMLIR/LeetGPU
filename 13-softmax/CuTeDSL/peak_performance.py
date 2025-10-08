import cutlass.cute as cute
from cutlass.utils import SmemAllocator
import cutlass
from typing import Callable, Optional
import math
import operator

@cute.jit
def warp_reduce(
    val: cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    col_idx = warp_idx
    if lane_idx == 0:
        reduction_buffer[col_idx] = val
    cute.arch.barrier()
    block_reduce_val = reduction_buffer[lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.kernel
def softmax_kernel(input: cute.Tensor, output: cute.Tensor, s_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bdimx, _, _ = cute.arch.block_dim()

    smem = SmemAllocator()

    shared = smem.allocate_tensor(input.element_type, s_layout, 16)

    local_max = input[tidx]
    for i in range(tidx + bdimx, input.shape[0], bdimx):
        local_max = max(local_max, input[i])

    max_val = block_reduce(local_max, cute.arch.fmax, shared, init_val=float('-inf'))

    local_sum = 0.0
    for i in range(tidx, input.shape[0], bdimx):
        local_sum += cute.exp(input[i] - max_val)

    local_sum = warp_reduce(local_sum, operator.add)
    sum_exp = block_reduce(local_sum, operator.add, shared)

    for i in range(tidx, input.shape[0], bdimx):
        output[i] = cute.exp(input[i] - max_val) / sum_exp

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    block_dim = 1024, 1, 1
    grid_dim = 1, 1, 1
    s_layout = cute.make_layout((1024), stride=(1))
    softmax_kernel(input, output, s_layout).launch(grid=grid_dim, block=block_dim)
