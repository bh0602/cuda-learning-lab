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