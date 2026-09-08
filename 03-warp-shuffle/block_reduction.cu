#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess)                                                                   \
        {                                                                                         \
            fprintf(stderr, "CUDA err at %s:%d:%s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    }while (0)

    __device__ float warp_reduce_sum(float val)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

__global__ void block_reduce_kernel(float *input, float *output, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float val = (tid < N) ? input[tid] : 0.0f; // 初始化val;

    // 第一步：Warp内归约
    val = warp_reduce_sum(val);

    // 第二步：每个warp的lane 0将结果写入共享内存
    __shared__ float warp_sum[32]; // 最多32个Warp（1024 / 32）
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (laneId == 0)
    {
        warp_sum[warpId] = val;
    }

    __syncthreads();

    // 第三步：第一个warp对warp_sums做最终归约
    int numWarps = blockDim.x / 32; // 计算当前block实际包含多少个Warp

    // 当前Block的前numWarps个线程读取共享内存
    //
    // thread 0读取warpSums[0]
    // thread 1读取warpSums[1]
    // ...
    //
    // 第一个 Warp 中剩余的线程使用0
    val = (threadIdx.x < numWarps) ? warp_sum[threadIdx.x] : 0.0f;

    if (warpId == 0)
    {
        val = warp_reduce_sum(val);
    }

    // lane 0 写出本block的归约结果
    // threadIdx.x等于0，同时意味着：
    //
    // warpId = 0
    // laneId = 0
    //
    // 所以这个线程拥有完整的 Block 求和结果
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = val;
    }
}

int main()
{
    const int N = 512;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    const size_t inputBytes = N * sizeof(float);
    // 输出数组每个 Block 对应一个float
    // 因此输出数组有blocksPerGrid个元素
    const size_t outputBytes = blocksPerGrid * sizeof(float);

    // --------------------------------------------------------
    // 1. 打印启动配置
    // --------------------------------------------------------

    printf("Input elements: %d\n", N);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Warps per block: %d\n", threadsPerBlock / 32);

    // 分配cpu内存
    float *h_input = (float *)malloc(inputBytes);
    float *h_output = (float *)malloc(outputBytes);

    // 初始化数据
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }

    // 分配GPU内存
    float *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    // 将输入数据从 CPU 复制到 GPU
    cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);

    // 启动kernel
    block_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将数据复制到cpu
    cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);

    // 验证每个 Block 的结果
    int errors = 0;

    for (int blockId = 0; blockId < blocksPerGrid; blockId++)
    {
        int start = blockId * threadsPerBlock;
        int end = start + threadsPerBlock;

        // 防止最后一个 Block 超出N
        if (end > N)
        {
            end = N;
        }

        // 当前 Block 实际处理的元素数量
        //
        // 因为所有输入都是1，所以理论和等于元素数量
        float expected = (float)(end - start);

        // 打印当前 Block 的结果
        printf("block %d result: %.1f, expected: %.1f\n", blockId, h_output[blockId], expected);

        // 使用误差范围比较浮点数
        if (fabsf(h_output[blockId] - expected) > 1e-5f)
        {
            errors++;
        }
    }

    // 10. CPU 汇总所有 Block 的部分和
    float totalSum = 0.0f;
    for (int blockId = 0; blockId < blocksPerGrid; blockId++)
    {
        totalSum += h_output[blockId];
    }
    printf("Final sum: %.1f, expected: %.1f\n", totalSum, (float)N);
    // 检查最终结果
    if (fabsf(totalSum - (float)N) > 1e-5f)
    {
        errors++;
    }
    // 11. 输出验证结果
    if (errors == 0)
    {
        printf("Verification PASSED!\n");
    }
    else
    {
        printf(
            "Verification FAILED: %d errors\n",
            errors);
    }

    // 12. 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // 13. 释放 CPU 内存
    free(h_input);
    free(h_output);

    return 0;
}