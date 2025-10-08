from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from memory import UnsafePointer
from buffer import NDBuffer
from math import ceildiv, exp
from gpu.memory import AddressSpace
from algorithm._gpu.reduction import block_reduce

@parameter
fn softmax_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    alias BLOCK_SIZE: Int = 1024
    var tid = thread_idx.x
    if tid == 0:
        output[0] = 1
    var max_buf = NDBuffer[
        DType.float32, 1, MutableAnyOrigin, 1, address_space = AddressSpace.SHARED
    ].stack_allocation()
    var sum_buf = NDBuffer[
        DType.float32, 1, MutableAnyOrigin, 1, address_space = AddressSpace.SHARED
    ].stack_allocation()

    # Step 1: compute max
    var local_max = Scalar[DType.float32](input[tid])
    for i in range(tid + BLOCK_SIZE, N, BLOCK_SIZE):
        local_max = max(local_max, input[i])

    @parameter
    @always_inline
    fn _max[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return max(x,y)

    var block_max = block_reduce[BLOCK_SIZE, _max](local_max, 0)

    if tid == 0:
        max_buf[0] = block_max
    barrier()

    # Step 2: out[i] = exp(in[i] - max) and compute sum of out[i]
    var local_sum = Scalar[DType.float32](0)
    for i in range(tid, N, BLOCK_SIZE):
        local_sum += exp(input[i] - max_buf[0])

    @parameter
    @always_inline
    fn _sum[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x+y

    var block_sum = block_reduce[BLOCK_SIZE, _sum](local_sum, 0)

    if tid == 0:
        sum_buf[0] = block_sum
    barrier()

    # Step 3: Normalize output
    for i in range(tid, N, BLOCK_SIZE):
        output[i] = exp(input[i] - max_buf[0]) / sum_buf[0]

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 1024
    var ctx = DeviceContext()
    var num_blocks = 1

    ctx.enqueue_function[softmax_kernel](
        input, output, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()
