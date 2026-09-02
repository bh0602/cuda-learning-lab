#include <stdio.h>        // 提供 printf
#include <stdlib.h>       // 提供 exit 和 EXIT_FAILURE
#include <cuda_runtime.h> // 提供 CUDA Runtime API
#include<math.h>

#define CUDA_CHECK(call)                                                                              \
    do                                                                                                \
    {                                                                                                 \
        cudaError_t err = call;                                                                       \
        if (err != cudaSuccess)                                                                       \
        {                                                                                             \
            fprintf(stderr, "CUDA error at %s:%d:%s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

__global__ void warp_reduction_kernel(float *warpResults)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = tid % 32; // 计算当前线程在自己 Warp 内的编号
    int warpId = tid / 32; // 计算当前线程属于全局第几个 Warp
    float val = (float)laneId;
    float results = warp_reduce_sum(val);
    if (laneId == 0)
    {
        warpResults[warpId] = results;
    }
}

int main()
{
    int block_per_grid = 1;
    int thread_per_block = 64;
    int totalThreads = block_per_grid * thread_per_block;
    int totalWarps = totalThreads / 32;
    float *d_warpResults;
    CUDA_CHECK(cudaMalloc(&d_warpResults, totalWarps * sizeof(float))); // 因为这里只需要两个float结果，所以大小为totalWarps * sizeof(float)
    warp_reduction_kernel<<<block_per_grid, thread_per_block>>>(d_warpResults);
    CUDA_CHECK(cudaGetLastError());      // 检查 Kernel 启动是否出现错误
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 Kernel 执行完成
    // 需要在cpu上创建数组用来接收gpu计算结果
    float h_warpResults[2];
    CUDA_CHECK(cudaMemcpy(h_warpResults, d_warpResults, totalWarps * sizeof(float), cudaMemcpyDeviceToHost));

    // 预期结果
    float expected = 496.0f;

    // 记录错误结果
    int errors = 0;
    for (int i = 0; i < totalWarps; i++)
    {
        printf("Warp %d result: %f\n",i,h_warpResults[i]);
        if (h_warpResults[i] != expected)
        {
            errors++;
        }
    }

    if(errors ==0){
        printf("verification passed!\n");
    }else{
        printf("verification failed!\n");
    }

    CUDA_CHECK(cudaFree(d_warpResults));
    
    return 0;
}