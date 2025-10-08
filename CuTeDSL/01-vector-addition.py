import cutlass.cute as cute

@cute.kernel
def vector_add_fragment_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    idx = bidx * bdimx + tidx
    if idx >= (N >> 4):
        if idx == (N >> 4):
            start = N & (~15)
            for i in range(start, N):
                C[i] = A[i] + B[i]
    else:
        a_val = A[(None, idx)].load()
        b_val = B[(None, idx)].load()
        C[(None, idx)] = a_val + b_val


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    num_threads_per_block = 512
    gA = cute.flat_divide(A, (16,))
    gB = cute.flat_divide(B, (16,))
    gC = cute.flat_divide(C, (16,))
    grid_dim = cute.ceil_div((N + 15) >> 4, num_threads_per_block), 1, 1
    vector_add_fragment_kernel(gA, gB, gC, N).launch(grid=grid_dim, block=(num_threads_per_block, 1, 1))
