import cutlass.cute as cute

@cute.kernel
def naive_mat_trans_kernel(input: cute.Tensor, output: cute.Tensor,):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    row = bidy * bdimy + tidy
    col = bidx * bdimx + tidx
    if row < input.shape[0] and col < input.shape[1]:
        output[col, row] = input[row, col]

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    block_dim = 32, 32, 1
    grid_dim = cute.ceil_div((cols, rows, 1), block_dim)
    naive_mat_trans_kernel(input, output).launch(grid=grid_dim, block=block_dim)
