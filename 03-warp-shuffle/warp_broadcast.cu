#include <stdio.h>
#include <cuda_runtime.h>

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

__global__ void warp_broadcast_kernel(float *result) // 保存每个线程广播后的结果
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 计算线程在当前warp中的lane编号
    //int laneId = threadIdx.x % 32;
    // 每个线程都有自己的局部变量，tid为0~63，所以my_val也是0~63
    float my_val = (float)tid;//每个 Warp 都读取自己 lane 0 的值
    // 当前warp中的所有线程都读取Lane 0的my_val值，并将其广播给所有线程
    float broadcast_val =
        __shfl_sync(
            0xFFFFFFFF, // 32个Lane全部参与
            my_val,     // 每个线程提供自己的局部变量
            0,          // 所有线程读取Lane 0
            32
        );
        //将广播结果写入global memory，便于主机端读取
        result[tid] = broadcast_val;
}

int main(){
    int block_per_grid = 1;//启动一个block
    int thread_per_block = 64;//每个block启动64个线程
    int N = 64; //结果数组包含64个float元素
    float * d_results;//定义GPU结果数组指针
    cudaMalloc(&d_results, N * sizeof(float));//在GPU上分配64个float的显存
    warp_broadcast_kernel<<<block_per_grid,thread_per_block>>>(d_results);
    float h_results[64];//定义CPU结果数组
    cudaMemcpy(h_results,d_results,N*sizeof(float),cudaMemcpyDeviceToHost);//将GPU结果拷贝到CPU
    //打印warp0中的部分结果
    printf("Warp 0:\n");
    printf("Lane 0 result = %.1f\n",h_results[0]);
    printf("Lane 1 result = %.1f\n",h_results[1]);
    printf("Lane 31 result = %.1f\n",h_results[31]);
    printf("\n Warp 1:\n");
    printf("Lane 0 result = %.1f\n",h_results[32]);
    printf("Lane 1 result = %.1f\n",h_results[33]);
    printf("Lane 31 result = %.1f\n",h_results[63]);
    //释放GPU显存
    cudaFree(d_results);

    return 0;
}