import cutlass
import cutlass.cute as cute
import cutlass
import cutlass.cute as cute

@cute.kernel
def naive_mat_trans_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    rows: cute.Int32,
    cols: cute.Int32
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    row = bidy * bdimy + tidy
    col = bidx * bdimx + tidx
    if row < rows and col < cols:
        output[col, row] = input[row, col]


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    num_threads_per_block = 32
    kernel = naive_mat_trans_kernel(input, output, rows, cols)
    kernel.launch(grid=((cols + 31) // num_threads_per_block, (rows + 31) // num_threads_per_block, 1),
                  block=(num_threads_per_block, num_threads_per_block, 1))
