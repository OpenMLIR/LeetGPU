# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261

    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)

    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val

@triton.jit
def fnv1a_hash_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_rounds,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.int32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint32))
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    result = input_vals.to(tl.uint32)
    for _ in range(n_rounds):
        result = fnv1a_hash(result)

    tl.store(output_ptr + offsets, result, mask=mask)

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](
        input_ptr,
        output_ptr,
        N,
        R,
        BLOCK_SIZE
    )