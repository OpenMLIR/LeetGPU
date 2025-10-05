import cutlass
import cutlass.cute as cute

@cute.kernel
def naive_matmul_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    row = bidy * bdimy + tidy
    col = bidx * bdimx + tidx
    if row < M and col < K:
        acc = 0.0
        for i in range(N):
            acc += A[row, i] * B[i, col]
        C[row, col] = acc


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32):
    num_threads_per_block = 32
    kernel = naive_matmul_kernel(A, B, C, M, N, K)
    kernel.launch(grid=((K + 31) // num_threads_per_block, (M + 31) // num_threads_per_block, 1),
                  block=(num_threads_per_block, num_threads_per_block, 1))
