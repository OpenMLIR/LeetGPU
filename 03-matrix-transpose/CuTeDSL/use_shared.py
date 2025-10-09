import cutlass.cute as cute
from cutlass.utils import SmemAllocator

TILE_DIM = 32
BLOCK_ROWS = 2

@cute.kernel
def mat_trans_kernel(input: cute.Tensor, output: cute.Tensor, s_layout):
    width, height = output.shape
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    smem = SmemAllocator()
    tile = smem.allocate_tensor(input.element_type, s_layout, 16)
    xIndex = bidx * TILE_DIM + tidx
    yIndex = bidy * TILE_DIM + tidy

    for i in range(0, TILE_DIM, BLOCK_ROWS):
        y = yIndex + i
        if y < height and xIndex < width:
            tile[tidy + i, tidx] = input[y, xIndex]

    cute.arch.barrier()
    xIndex = bidy * TILE_DIM + tidx
    yIndex = bidx * TILE_DIM + tidy
    for i in range(0, TILE_DIM, BLOCK_ROWS):
        y = yIndex + i
        if y < width and xIndex < height:
            output[y, xIndex] = tile[tidx, tidy + i]

# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    block_dim = TILE_DIM, BLOCK_ROWS, 1
    s_layout = cute.make_layout((TILE_DIM, TILE_DIM), stride=(1, TILE_DIM+1))
    grid_dim = cute.ceil_div((cols, rows, 1), block_dim)
    mat_trans_kernel(input, output, s_layout).launch(grid=grid_dim, block=block_dim)
