import cutlass
import cutlass.cute as cute


@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    N: cute.Uint64
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        gC[thread_idx] = gA[thread_idx] + gB[thread_idx]

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint64):
    num_threads_per_block = 1024
    kernel = naive_elementwise_add_kernel(A, B, C, N)
    kernel.launch(grid=((N + 1023) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))
