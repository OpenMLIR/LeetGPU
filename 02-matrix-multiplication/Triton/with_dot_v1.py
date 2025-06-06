# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # initialize pointers for a and b
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # accumulate along the N dimension
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # load current blocks of a and b with boundary check
        a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        # compute matrix multiplication and accumulate
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn

    # write result back to c
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    matrix_multiplication_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_K=16,
        BLOCK_SIZE_N=16
    )
