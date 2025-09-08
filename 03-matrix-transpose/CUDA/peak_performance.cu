// solution.cu, https://leetgpu.com/profile?display_name=EpicSamurai464
#include <cuda_runtime.h>
#define TILE_DIM    32
#define BLOCK_ROWS   2

__global__ void transpose_cp_async_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    int width, int height)  // width=cols, height=rows
{
    // 消除 bank conflict，多一列
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    const float* srcPtr = input + yIndex * width + xIndex;


    // fallback：普通拷贝
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < height && xIndex < width)
            tile[threadIdx.y + i][threadIdx.x] =
                input[y * width + xIndex];
    }

    __syncthreads();

    // 写回全局内存，注意转置坐标
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < width && xIndex < height) {
            output[y * height + xIndex] =
                tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

extern "C" void solve(
    const float* input, float* output,
    int rows, int cols)
{
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
              (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_cp_async_kernel
        <<<grid, threads>>>(input, output, cols, rows);
    //cudaDeviceSynchronize();
}
