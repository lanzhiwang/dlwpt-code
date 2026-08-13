# PyTorch Distributed Overview
PyTorch 分布式概述

* https://docs.pytorch.org/tutorials/beginner/dist_overview.html

Created On: Jul 28, 2020 | Last Updated: Jul 20, 2025 | Last Verified: Nov 05, 2024
创建日期: 2020年7月28日 | 最后更新日期: 2025年7月20日 | 最后验证日期: 2024年11月5日

Author: [Will Constable](https://github.com/wconstab/), [Wei Feng](https://github.com/weifengpy)

> Note
> View and edit this tutorial in [github](https://github.com/pytorch/tutorials/blob/main/beginner_source/dist_overview.rst).
>

This is the overview page for the `torch.distributed` package. The goal of this page is to categorize documents into different topics and briefly describe each of them. If this is your first time building distributed training applications using PyTorch, it is recommended to use this document to navigate to the technology that can best serve your use case.
这是 `torch.distributed` 包的概览页面. 本页面旨在将文档按不同主题分类, 并简要介绍每个主题. 如果您是第一次使用 PyTorch 构建分布式训练应用程序, 建议您参考本文档, 找到最适合您用例的技术.

## Introduction
介绍

The PyTorch Distributed library includes a collective of parallelism modules, a communications layer, and infrastructure for launching and debugging large training jobs.
PyTorch 分布式库包含一系列并行模块、通信层以及用于启动和调试大型训练作业的基础架构.

### Parallelism APIs
并行 API

These Parallelism Modules offer high-level functionality and compose with existing models:
这些并行模块提供高级功能, 并可与现有模型组合使用:

- [Distributed Data-Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
  分布式数据并行(DDP)

- [Fully Sharded Data-Parallel Training (FSDP2)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
  全分片数据并行训练(FSDP2)

- [Tensor Parallel (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
  张量并行(TP)

- [Pipeline Parallel (PP)](https://pytorch.org/docs/main/distributed.pipelining.html)
  流水线并行(PP)

### Sharding primitives
分片原语

`DTensor` and `DeviceMesh` are primitives used to build parallelism in terms of sharded or replicated tensors on N-dimensional process groups.
`DTensor` 和 `DeviceMesh` 是用于在 N 维进程组上以分片或复制张量的方式构建并行性的原语.

- [DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md) represents a tensor that is sharded and/or replicated, and communicates automatically to reshard tensors as needed by operations.
  [DTensor]() 表示分片和/或复制的张量, 并根据操作需要自动通信以重新分片张量.

- [DeviceMesh](https://pytorch.org/docs/stable/distributed.html#devicemesh) abstracts the accelerator device communicators into a multi-dimensional array, which manages the underlying `ProcessGroup` instances for collective communications in multi-dimensional parallelisms. Try out our [Device Mesh Recipe](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) to learn more.
  [DeviceMesh]() 将加速器设备通信器抽象为一个多维数组, 该数组管理底层 `ProcessGroup` 实例, 用于在多维并行环境中进行集体通信. 试用我们的 [Device Mesh Recipe]() 了解更多信息.


### Communications APIs
通信 API

The [PyTorch distributed communication layer (C10D)](https://pytorch.org/docs/stable/distributed.html) offers both collective communication APIs (e.g., [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) and [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)) and P2P communication APIs (e.g., [send](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send) and [isend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)), which are used under the hood in all of the parallelism implementations. [Writing Distributed Applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) shows examples of using c10d communication APIs.
[PyTorch 分布式通信层(C10D)]() 提供了集体通信 API(例如, [all_reduce)]() 和 [all_gather]()) 以及 P2P 通信 API(例如, [send]() 和 [isend]()), 它们在所有并行实现中都被底层使用. [使用 PyTorch 编写分布式应用程序]() 展示使用 c10d 通信 API 的示例.

### Launcher
启动器

[torchrun](https://pytorch.org/docs/stable/elastic/run.html) is a widely-used launcher script, which spawns processes on the local and remote machines for running distributed PyTorch programs.
[torchrun]() 是一个广泛使用的启动脚本, 它在本地和远程机器上生成进程以运行分布式 PyTorch 程序.

## Applying Parallelism To Scale Your Model
应用并行性来扩展模型

Data Parallelism is a widely adopted single-program multiple-data training paradigm where the model is replicated on every process, every model replica computes local gradients for a different set of input data samples, gradients are averaged within the data-parallel communicator group before each optimizer step.
数据并行是一种广泛采用的单程序多数据训练范例, 其中模型在每个过程中复制, 每个模型副本为不同的输入数据样本集计算局部梯度, 在每个优化器步骤之前在数据并行通信器组内对梯度进行平均.

Model Parallelism techniques (or Sharded Data Parallelism) are required when a model doesn't fit in GPU, and can be combined together to form multi-dimensional (N-D) parallelism techniques.
当模型不适合 GPU 时, 需要模型并行技术(或分片数据并行), 并且可以组合在一起形成多维(ND)并行技术.

When deciding what parallelism techniques to choose for your model, use these common guidelines:
在决定为您的模型选择哪种并行技术时, 请使用以下通用准则:

1. Use [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html), if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
  如果您的模型适合单个 GPU, 但您想使用多个 GPU 轻松扩展训练, 请使用 [DistributedDataParallel (DDP)]().

    - Use [torchrun](https://pytorch.org/docs/stable/elastic/run.html), to launch multiple pytorch processes if you are using more than one node.
      如果您使用多个节点, 请使用 [torchrun]() 启动多个 pytorch 进程.

    - See also: [Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
      另请参阅: [分布式数据并行入门]()

2. Use [FullyShardedDataParallel (FSDP2)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) when your model cannot fit on one GPU.
  当您的模型无法在一个 GPU 上安装时, 请使用 [FullyShardedDataParallel (FSDP2)]().

    - See also: [Getting Started with FSDP2](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
      另请参阅: [FSDP2 入门]()

3. Use [Tensor Parallel (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) and/or [Pipeline Parallel (PP)](https://pytorch.org/docs/main/distributed.pipelining.html) if you reach scaling limitations with FSDP2.
  如果达到 FSDP2 的扩展限制, 请使用[张量并行 (TP)]() 和/或[流水线并行 (PP)]().

    - Try our [Tensor Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
      尝试我们的[张量并行教程]()

    - See also: [TorchTitan end to end example of 3D parallelism](https://github.com/pytorch/torchtitan)
      另请参阅: [TorchTitan 3D 并行端到端示例]()

4. Model Parallelism
5. 组合 DDP, FSDP2, TP, PP
6. 专家并行
7. 序列并行/上下文并行(SP/CP)


> Note
> Data-parallel training also works with [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus).
> 数据并行训练也适用于[自动混合精度(AMP)]().
>

## PyTorch Distributed Developers
PyTorch 分布式开发者

If you'd like to contribute to PyTorch Distributed, refer to our [Developer Guide](https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md).
如果您想为 PyTorch Distributed 做贡献, 请参阅我们的 [开发者指南]().
