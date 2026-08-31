#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

// ============================================================
// Kernel 1：同一个Warp内，相邻线程走不同分支
// ============================================================
__global__ void
divergentKernel(const float *input, float *output, int N, int iterations)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N)
    {
        float value = input[tid];
        if (tid % 2 == 0)
        {
            // 路径A
            for (int i = 0; i < iterations; i++)
            {
                value += 1.0f;
            }
        }
        else
        {
            for (int i = 0; i < iterations; i++)
            {
                value -= 1.0f;
            }
        }
        output[tid] = value;
    }
}

// ================================================================
// Kernel 2：同一个Warp中的线程走相同分支
// ================================================================
__global__ void warpAlignedKernel(const float *input, float *output, int N, int iterations)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N)
    {
        float value = input[tid];
        int warpId = tid / 32;
        if (warpId % 2 == 0)
        {
            // 路径A
            for (int i = 0; i < iterations; i++)
            {
                value += 1.0f;
            }
        }
        else
        {
            for (int i = 0; i < iterations; i++)
            {
                value -= 1.0f;
            }
        }
        output[tid] = value;
    }
}

int main()
{
    const int N = 1 << 20;
    const int iterations = 100; // 每个线程执行100次循环
    const int warmup = 5;       // 预热五次
    const int blocksize = 256;
    const int repeats = 100; // 重复执行次数
    int gridsize = (N + blocksize - 1) / blocksize;
    size_t size = N * sizeof(float); // 计算一个数组的字节数

    // ============================================================
    // 1. 分配CPU内存
    // ============================================================

    float *h_input = (float *)malloc(size);     // cpu输入数组
    float *h_divergent = (float *)malloc(size); // 第一个kernel的cpu结果数组
    float *h_aligned = (float *)malloc(size);   // 第二个kernel的cpu结果数组

    if (h_input == NULL || h_divergent == NULL || h_aligned == NULL)
    {
        printf("Host momory allocation failed\n");
        free(h_input);
        free(h_divergent);
        free(h_aligned);
        return EXIT_FAILURE; // 结束程序
    }

    // ============================================================
    // 2. 初始化输入数组
    // ============================================================

    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f; // 所有输入都设置成1
    }

    // ============================================================
    // 3. 分配GPU显存
    // ============================================================

    float *d_input, *d_divergent, *d_aligned;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_divergent, size));
    CUDA_CHECK(cudaMalloc(&d_aligned, size));

    // ============================================================
    // 4. 将输入从CPU复制到GPU
    // ============================================================
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice)); //(GPU目标地址, CPU源地址, 复制字节数, 复制方向)

    // ============================================================
    // 5. 创建CUDA Event
    // ============================================================
    cudaEvent_t start, stop;             // 定义开始和结束事件
    CUDA_CHECK(cudaEventCreate(&start)); // 创建开始事件
    CUDA_CHECK(cudaEventCreate(&stop));  // 创建结束事件

    // ============================================================
    // 6. 第一个Kernel预热5次
    // ============================================================
    for (int i = 0; i < warmup; i++)
    {
        divergentKernel<<<gridsize, blocksize>>>(d_input, d_divergent, N, iterations);
    }

    CUDA_CHECK(cudaGetLastError());      // 检查kernel启动是否成功
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待5次预热kernel执行完成

    // ============================================================
    // 7. 第一个Kernel正式运行100次
    // ============================================================

    CUDA_CHECK(cudaEventRecord(start)); // 记录开始位置
    for (int i = 0; i < repeats; i++)
    {
        divergentKernel<<<gridsize, blocksize>>>(d_input, d_divergent, N, iterations); // 正式执行100次kernel
    }
    CUDA_CHECK(cudaGetLastError());         // 检查kernel启动是否成功
    CUDA_CHECK(cudaEventRecord(stop));      // 记录结束位置
    CUDA_CHECK(cudaEventSynchronize(stop)); // 等待kernel执行完成
    float divergentTotalMs = 0.0f;          // 保存100次总时间
    CUDA_CHECK(cudaEventElapsedTime(
        &divergentTotalMs,                                 // 接收计时结果
        start,                                             // 开始事件
        stop));                                            // 结束事件
    float divergentAverageMs = divergentTotalMs / repeats; // 计算平均时间

    // ============================================================
    // 8. 第二个Kernel预热5次
    // ============================================================
    for (int i = 0; i < warmup; i++)
    {
        warpAlignedKernel<<<gridsize, blocksize>>>(d_input, d_aligned, N, iterations);
    }

    CUDA_CHECK(cudaGetLastError());      // 检查kernel启动是否成功
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待5次预热kernel执行完成

    // ============================================================
    // 9. 第二个Kernel正式运行100次
    // ============================================================

    CUDA_CHECK(cudaEventRecord(start)); // 记录开始位置
    for (int i = 0; i < repeats; i++)
    {
        warpAlignedKernel<<<gridsize, blocksize>>>(d_input, d_aligned, N, iterations); // 正式执行100次kernel
    }
    CUDA_CHECK(cudaGetLastError());                                 // 检查kernel启动是否成功
    CUDA_CHECK(cudaEventRecord(stop));                              // 记录结束位置
    CUDA_CHECK(cudaEventSynchronize(stop));                         // 等待kernel执行完成
    float alignedTotalMs = 0.0f;                                    // 保存100次总时间
    CUDA_CHECK(cudaEventElapsedTime(&alignedTotalMs, start, stop)); // 计算时间差
    float alignedAverageMs = alignedTotalMs / repeats;              // 计算平均时间

    // ============================================================
    // 10. 将两个结果复制回CPU
    // ============================================================

    cudaMemcpy(h_divergent, d_divergent, size, cudaMemcpyDeviceToHost); // 将第一个kernel的结果复制回CPU

    cudaMemcpy(h_aligned, d_aligned, size, cudaMemcpyDeviceToHost); // 将第二个kernel的结果复制回CPU

    // ============================================================
    // 11. 验证第一个Kernel
    // ============================================================

    int divergentErrors = 0; // 记录错误数量

    for (int tid = 0; tid < N; tid++) // 检查所有元素
    {
        float expected;   // CPU期望结果
        if (tid % 2 == 0) // 偶数线程走路径A
        {
            expected = // 初始值1加100次1
                1.0f + iterations;
        }
        else // 奇数线程走路径B
        {
            expected = // 初始值1减100次1
                1.0f - iterations;
        }
        if (fabsf(h_divergent[tid] - expected) > 1e-5f) // 计算GPU与CPU结果差值判断误差是否过大
        {
            if (divergentErrors < 5) // 最多打印5个错误
            {
                printf(
                    "Divergent mismatch at %d: "
                    "GPU=%f, expected=%f\n",
                    tid,
                    h_divergent[tid],
                    expected);
            }
            divergentErrors++; // 错误数量增加1
        }
    }

    // ============================================================
    // 12. 验证第二个Kernel
    // ============================================================

    int alignedErrors = 0; // 记录错误数量

    for (int tid = 0; tid < N; tid++) // 检查所有元素
    {
        int warpId = tid / 32; // 计算线程所属Warp

        float expected; // CPU期望结果

        if (warpId % 2 == 0) // 偶数Warp走路径A
        {
            expected = 1.0f + iterations; // 初始值1加100次1
        }
        else // 奇数Warp走路径B
        {
            expected = 1.0f - iterations; // 初始值1减100次1
        }

        if (fabsf(h_aligned[tid] - expected) > 1e-5f) // 计算GPU与CPU结果差值判断误差是否过大
        {
            if (alignedErrors < 5) // 最多打印5个错误
            {
                printf(
                    "Aligned mismatch at %d: "
                    "GPU=%f, expected=%f\n",
                    tid,
                    h_aligned[tid],
                    expected);
            }
            alignedErrors++; // 错误数量增加1
        }
    }

    // ============================================================
    // 13. 输出验证和计时结果
    // ============================================================

    printf("\nVerification:\n"); // 输出验证标题

    printf("Divergent kernel: %s\n", divergentErrors == 0 ? "PASSED" : "FAILED"); // 没有错误就输出PASSED

    printf("Warp-aligned kernel: %s\n", alignedErrors == 0 ? "PASSED" : "FAILED"); // 没有错误就输出PASSED

    printf("\nAverage time over 100 runs:\n"); // 输出计时标题

    printf("Divergent kernel   : %.6f ms\n", divergentAverageMs); // 第一个Kernel平均时间

    printf("Warp-aligned kernel: %.6f ms\n", alignedAverageMs); // 第二个Kernel平均时间

    printf("Time ratio         : %.3fx\n", divergentAverageMs / alignedAverageMs); // 两个Kernel时间之比

    // ============================================================
    // 14. 释放资源
    // ============================================================

    CUDA_CHECK(cudaEventDestroy(start)); // 销毁开始事件

    CUDA_CHECK(cudaEventDestroy(stop)); // 销毁结束事件

    CUDA_CHECK(cudaFree(d_input)); // 释放GPU输入数组

    CUDA_CHECK(cudaFree(d_divergent)); // 释放第一个GPU结果数组

    CUDA_CHECK(cudaFree(d_aligned)); // 释放第二个GPU结果数组

    free(h_input); // 释放CPU输入数组

    free(h_divergent); // 释放第一个CPU结果数组

    free(h_aligned); // 释放第二个CPU结果数组

    return 0; // 程序正常结束
}