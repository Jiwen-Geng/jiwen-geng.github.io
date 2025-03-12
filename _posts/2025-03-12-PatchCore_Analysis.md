---
title: PatchCore 算法分析
date: 2025-03-12 23:11:11 +0800
categories: [Animal, Insect]
tags: [bee]
math: true
---

# Patchcore Algorithm Analysis

参考论文：[Toward Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)


## 1. Locally aware patch features
正常图像的数据集合记为$\mathcal{X}_N$，其中N为正常数据个数，定义其标签为：
$$
\forall{x}\in{\mathcal{X}_N}: y_{x}=0
$$
其中标签为$y_x\in{\{0,1\}}$. $y=0$表示正常样本，$y=1$表示异常样本。 定义图像$x_i\in{\mathcal{X}}$在第$j$层预训练网络$\phi$中的特征为：
$$
\phi_{i,j}=\phi_{j}(x_i)
$$
其中$j$为特征层的下标索引。特征表示网络与ResNet相关，如`ResNet-50`, `WideResnet-50`等等。$j\in{\{1,2,3,4\}}$表示最终输出的分辨率层。

定义特征图 $\phi_{i,j}\in{\mathbb{R}}^{c^*\times h^*\times w^*}$ 为三维张量。其中 $c^*$ 为深度，$h^*$ 为高度，$w^*$ 为宽度。定义与之相关的**深度**切片为：
$$
\phi_{i,j}(h,w) = \phi_j(x_i,h,w) \in{\mathbb{R}}^{c^*}
$$
其中 $h\in{\{1,2,...,h^*\}}$，$w\in{\{1,2,...,w^*\}}$. 

定义尺寸为p的图像块集合为：
$$
\mathcal{N}_p^{h,w} = \{(a,b)|a\in{[h-\lfloor p/2\rfloor,...,h+\lfloor p/2\rfloor]},b\in{[w-\lfloor p/2\rfloor,...,w+\lfloor p/2\rfloor]}\}
$$
其中p为奇数。并且位置$(h,w)$处的局部显著特征为：
$$
\phi_{i,j}(\mathcal{N}_p^{(h,w)}) = f_{agg}(\{\phi_{i,j}(a,b)|(a,b)\in{\mathcal{N}_p^{(h,w)}}\})
$$
其中$f_{agg}$特征矢量的累积函数。在`PatchCore`中采用的是**adaptive average pooling**.类似于局部平均。
对于一个特征图张量$\phi_{i,j}$，它的局部显著特征集合为：
$$
\mathcal{P}_{s,p}(\phi_{i,j}) = \{ \phi_{i,j}(\mathcal{N}_p^{(h,w)}) | h,w \ mod \ s = 0, 
h<h^*,w<w^*,h,w\in{\mathbb{N}}\}
$$
上式中$s$为步长。该式的含义是按照步长$s$选择一系列特征块。

累积不同层级的patch特征是很有用的。`PatchCore`采用了两个中间特征层$j$和$j+1$, 即$\mathcal{P}_{s,p}(\phi_{i,j})$ 和$\mathcal{P}_{s,p}(\phi_{i,j+1})$。 尺寸不一致，采用双线性插值解决。
最终，对于正常的训练图像 $x_i\in{\mathcal{X}_N}$， `PatchCore` 内存集合 $\mathcal{M}$ 定义为：
$$
\mathcal{M} = \bigcup_{x_i\in{\mathcal{X}_N}}
\mathcal{P}_{s,p}(\phi_j(x_i))
$$

## 2. Coreset-reduced patch-feature memory bank
随着正样本 $\mathcal{X}_N$ 的个数增加，$\mathcal{M}$ 会变得越来越大，因此需要降低。这里采用了一种叫coreset subsampling mechanism 去降低 $\mathcal{M}$.  coreset selection通常希望找一个子集 $\mathcal{S}\in{\mathcal{A}}$, 使得在 $\mathcal{A}$上的解尽可能地与在 $\mathcal{S}$ [1]. 具体采用了**minimax facility location**, 即
$$
\mathcal{M}_C^* = \mathop{\arg\min}\limits_{\mathcal{M}_C\subset\mathcal{M}}\max\limits_{m\in{\mathcal{M}}}\min\limits_{n\in{\mathcal{M}_C}}\|m-n\|_2
$$
该式表明，子集 $\mathcal{M}_C$ 是原始集合 $\mathcal{M}$ 中最为集中的部分。这里采用贪婪算法求解[2]. 为了进一步降低coreset selection时间，这里采用**Johnson-Lindenstrauss Theorem**，采用随机线性投影进行降维: $\psi: \mathbb{R}^d \rightarrow \mathbb{R}^{d^{*}} $,其中 $d^*<d$. 

## 3. Anomaly Detection with PatchCore
计算测试图像 $x_{test}$, 计算正常图像的特征块构成的内存池 $\mathcal{M}$ 中的异常值 $s\in{\mathbb{R}}$. 通过计算测试图像特征块的集合与现有内存池的最大距离判断，即：
$$
m^{test,*},m^*=\mathop{\arg\max}\limits_{m^{test}\in{\mathcal{P}(x^{test})}}\mathop{\arg\min}\limits_{m\in{\mathcal{M}}}\|m^{test}-m\|^2
$$
$$
s^*=\|m^{test,*}-m^*\|_2
$$
从上式中可看出，计算过程如下：
- 先见测试图像 $x^{test}$ 变换成内存块 $\mathcal{P}(x^{test})$;
- 再计算测试图像内存块 $m^{test}$与现有内存池中最近的内存块 $m^*$;
- 最后再测试图像内存块 $\mathcal{P}(x^{test})$ 找到与 $m^*$ 最远的块，记为 $m^{test,*}$;
- 则最终的异常值为这两个块的距离。

最终的输出值需要对上述求解的 $s^*$ 归一化。其归一化方程为：
$$
s=\left(1-\frac{exp\|m^{test,*}-m^*\|_2}{\mathop{\sum{}}\limits_{m\in{\mathcal{N}_b(m^*)}}exp\|m^{test,*}-m\|_2}\right)\cdot\ s^*
$$
其中 $\mathcal{N}_b(m^*)$为测试图像块 $m^*$ 在内存池 $\mathcal{M}$中的 $b$ 个最近的邻域块.






## Reference
[1] Agarwal, Pankaj K., Sariel Har-Peled and Kasturi R. Varadarajan. “Geometric Approximation via Coresets.” (2007).

[2] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018. 2, 4