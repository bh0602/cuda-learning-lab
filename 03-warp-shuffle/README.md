# Warp Divergence实验

## 学习范围

AIInfraGuide 2.1 Warp与执行模型：
3.3 Divergence量化、3.4规避策略、3.5独立线程调度。

## 两个Kernel

- Divergent：相邻线程进入不同分支
- Warp Aligned：同一个Warp内线程进入相同分支

## 测试方法

- 预热5次
- 正式运行100次
- 使用CUDA Event计时
- 使用CPU或逐元素检查验证结果

## 实验结果

填写两个Kernel的平均时间和正确性。

## 当前理解

解释为什么tid % 2会产生Warp Divergence，
而按warpId进行分支可以减少Divergence。

## Warp Broadcast 实验

### 核心接口

`__shfl_sync(mask, value, src_lane, width)` 可以让一个 warp
中的线程直接读取指定 lane 的寄存器值。

### 测试配置

- block size：32
- warp 数量：1
- 测试 src_lane：0、5、31
- 编译器：nvcc
- GPU：填写实际型号

### 测试结果

| src_lane | 预期结果 | 实际结果 | 是否正确 |
|---|---|---|---|
| 0 | 所有线程获得 lane 0 的值 | 填写结果 | 是/否 |
| 5 | 所有线程获得 lane 5 的值 | 填写结果 | 是/否 |
| 31 | 所有线程获得 lane 31 的值 | 填写结果 | 是/否 |

### 原理解释

- 数据交换发生在同一个 warp 内。
- 数据直接来自线程的寄存器。
- 不需要先把数据写入 shared memory。
- mask 表示参与本次操作的线程集合。
- src_lane 表示数据来源线程。
- width 用于划分逻辑子 warp。