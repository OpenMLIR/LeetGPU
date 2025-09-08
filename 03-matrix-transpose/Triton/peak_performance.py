import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols):
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)
    offs_row = row_index * 32 + tl.arange(0, 32)
    offs_col = col_index * 32 + tl.arange(0, 32)

    old_offs = offs_row[:, None] * cols + offs_col[None, :]
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
    block = tl.load(input_ptr + old_offs, mask=mask)

    transposed_block = tl.trans(block)

    new_block = offs_col[:, None] * rows + offs_row[None, :]
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols
    )