#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = (call);                                                                      \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

// 测试四种Warp Shuffle指令
__global__ void testShuffleKernel(int *normalResults, int *upResults, int *downResults, int *xorResults) // 保存 __shfl_sync ,__shfl_up_sync ,__shfl_down_sync,__shfl_xor_sync的结果的结果,
{
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 因为本小节测试的只有1个warp
    int laneId = threadIdx.x % 32; // 获取线程在warp中的ID
    // 初始化laneId
    int val = laneId;

    // 32位全为1，表示当前warp的32个线程都参与Shuffle
    unsigned int mask = 0xffffffff;

    // 1、普通shuffle：所有线程都读取lane 0的值
    int normal_val = __shfl_sync(mask, val, 0); // val指的是每个线程都提供自己的val，0指的是所有线程都读取lane0

    // 2、Shuffle up：读取低位线程的值
    int up_val = __shfl_up_sync(mask, val, 1);

    int down_val = __shfl_down_sync(mask, val, 1);

    int xor_val = __shfl_xor_sync(mask, val, 1);

    // 将每个线程得到的结果写入 GPU 全局内存
    normalResults[laneId] = normal_val;
    upResults[laneId] = up_val;
    downResults[laneId] = down_val;
    xorResults[laneId] = xor_val;
}

int main()
{
    const int warpSize = 32;
    const size_t bytes = warpSize * sizeof(int); // 一个结果数组需要保存32个 int

    int *d_normalResults, *d_upResults, *d_downResults, *d_xorResults;
    CUDA_CHECK(cudaMalloc(&d_normalResults, bytes));
    CUDA_CHECK(cudaMalloc(&d_upResults, bytes));
    CUDA_CHECK(cudaMalloc(&d_downResults, bytes));
    CUDA_CHECK(cudaMalloc(&d_xorResults, bytes));

    testShuffleKernel<<<1, 32>>>(d_normalResults, d_upResults, d_downResults, d_xorResults);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 在 CPU 上创建四个结果数组
    int h_normalResults[warpSize], h_upResults[warpSize], h_downResults[warpSize], h_xorResults[warpSize];

    CUDA_CHECK(cudaMemcpy(h_normalResults, d_normalResults, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_upResults, d_upResults, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_downResults, d_downResults, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_xorResults, d_xorResults, bytes, cudaMemcpyDeviceToHost));

    // 打印结果
    printf("Lane | Input | shufl(0) | shfl_up(1) | shfl_down(1) | shfl_xor(1)\n");

    printf("---------------------------------------------------------------\n");

    for (int laneId = 0; laneId < warpSize; laneId++)
    {
        printf(
            "%4d | %5d | %7d | %10d | %12d | %11d\n",
            laneId,                 // 当前 Lane 编号
            laneId,                 // 当前线程的初始值
            h_normalResults[laneId], // 普通 Shuffle 结果
            h_upResults[laneId],     // Shuffle Up 结果
            h_downResults[laneId],   // Shuffle Down 结果
            h_xorResults[laneId]     // Shuffle XOR 结果
        );
    }

    // 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_normalResults));
    CUDA_CHECK(cudaFree(d_upResults));
    CUDA_CHECK(cudaFree(d_downResults));
    CUDA_CHECK(cudaFree(d_xorResults));

    return 0;
}