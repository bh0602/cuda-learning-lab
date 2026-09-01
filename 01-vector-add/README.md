#该实验解决了什么样的问题；
#CUDA代码执行流程；
#如何编译运行；
#如何验证与计时
#实际实验结果和结论
# 01 - CUDA Vector Addition

## 实验目标

使用 CUDA 实现向量加法：

```text
C[i] = A[i] + B[i]
```

并完成以下对比：

- CPU 串行计算时间；
- GPU Kernel 计算时间；
- GPU 端到端时间（包含数据传输）；
- CPU 与 GPU 结果验证；
- Kernel 有效显存带宽计算。

## 实验配置

- 向量长度：`2^24 = 16,777,216`
- 数据类型：`float`
- 单个向量大小：64 MiB
- Block Size：256
- 输入数据：
  - `A[i] = sin(i × 0.001)`
  - `B[i] = cos(i × 0.001)`

## 编译运行

```bash
cd ~/cuda-learning-lab/01-vector-add
nvcc -O3 vector_add.cu -o vector_add
./vector_add
```

## 实验结果

| 指标 | 结果 |
|---|---:|
| CPU time | 33.4576 ms |
| GPU Kernel time | 0.9588 ms |
| GPU total time | 45.8416 ms |
| Kernel-only speedup | 34.9× |
| End-to-end speedup | 0.7× |
| Effective bandwidth | 209.97 GB/s |
| Verification | PASSED |

> 运行时间会受到 GPU 频率、温度和系统负载影响，不同运行之间可能存在波动。

## 结果分析

GPU Kernel 本身比 CPU 串行计算快约 34.9 倍，说明大量线程并行执行向量加法可以显著缩短纯计算时间。

但是，包含两次 Host-to-Device 复制和一次 Device-to-Host 复制后，GPU 端到端时间反而超过 CPU 时间，端到端加速比只有 0.7×。

这说明：

- Kernel 快不代表整个程序一定快；
- CPU 与 GPU 之间的数据传输可能成为主要开销；
- 简单算子计算量较小，可能无法抵消数据传输成本；
- 实际应用应尽量让数据留在 GPU 上，连续执行多个算子。

向量加法每个元素读取两个 `float`、写入一个 `float`，共访问12字节，但只完成一次加法，因此算术强度较低，属于典型的 Memory-bound Kernel。

## 当前不足

当前程序只测量了一次运行，结果存在一定波动。后续可以加入：

- Kernel 预热；
- 正式运行100次并计算平均时间；
- 不同 Block Size 对比；
- 使用 Nsight Compute 分析显存吞吐量。