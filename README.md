# CUDA Learning Lab

本仓库用于记录CUDA、GPU性能优化和AI推理部署的学习过程。

## 学习目标

- 掌握CUDA编程模型和内存模型
- 独立实现并优化基础GPU算子
- 学会进行正确性验证和性能测试
- 为AI Infra和模型部署实习准备项目

## 实验环境

- OS：WSL Ubuntu
- GPU：NVIDIA GeForce RTX 2080 Ti
- Language：C++ / CUDA
- Compiler：nvcc

## 学习进度

| 编号 | 主题 | 状态 | 主要产物 |
|---|---|---|---|
| 01 | Vector Add | 进行中 | 基础实现与性能测试 |
| 02 | Memory Access | 未开始 | 连续与跨步访问对比 |
| 03 | Warp Shuffle | 进行中 | Shuffle广播与Warp Reduce |
| 04 | Reduce | 未开始 | Shared与Shuffle版本 |
| 05 | Softmax | 未开始 | 基础与优化实现 |
| 06 | GEMM | 未开始 | 基础矩阵乘法 |

## 仓库规范

每个主题需要包含：

1. 可运行的CUDA代码
2. CPU参考实现与正确性验证
3. CUDA Event性能测试
4. 实验结果表格
5. 原理和结果分析README