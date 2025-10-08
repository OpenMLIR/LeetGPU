#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // dynamic shared memory for reduction
    extern __shared__ float shared_mem[];
    float* max_shared = shared_mem;
    float* sum_shared = &shared_mem[blockDim.x];

    int tid = threadIdx.x;

    // compute maximum value
    float local_max = input[tid];
    for (int i = tid + blockDim.x; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    max_shared[tid] = local_max;
    __syncthreads();

    // block reduction to find maximum
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = max_shared[0];
    __syncthreads();

    // compute exponential sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    sum_shared[tid] = local_sum;
    __syncthreads();

    // block reduction to compute sum
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (tid < i) {
            sum_shared[tid] += sum_shared[tid + i];
        }
        __syncthreads();
    }
    float sum_exp = sum_shared[0];
    __syncthreads();

    // compute final results
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 1024;
    size_t sharedSize = 2 * threadsPerBlock * sizeof(float);
    softmax_kernel<<<1, threadsPerBlock, sharedSize>>>(input, output, N);
    cudaDeviceSynchronize();
}
