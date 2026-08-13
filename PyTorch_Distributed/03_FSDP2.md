# Getting Started with Fully Sharded Data Parallel (FSDP2)
完全分片数据并行 (FSDP2) 入门

-----------------------------------------------------------------------------------------

Q: 假设你是一位精通 LLM 和 PyTorch 的高级研究人员, 我查看了 PyTorch Distributed 的相关文档, 知道当您的模型无法在一个 GPU 上安装时, 请使 `FullyShardedDataParallel(FSDP2)`.
1、请问什么是 FullyShardedDataParallel(FSDP2), 他的具体含义是什么?
2、为什么需要 FullyShardedDataParallel(FSDP2)?
3、FullyShardedDataParallel(FSDP2) 和 DDP, 以及 TP, PP 的区别是什么?
4、请使用代码示例具体说明 FullyShardedDataParallel(FSDP2) 应该如何使用

你好! 作为 PyTorch 和 LLM 大模型分布式训练方向的研究人员, 针对你提到的 FullyShardedDataParallel (FSDP2), 我将从定义概念、核心动机、技术对比、代码实战四个维度为你进行深度拆解.

---

### 1. 什么是 FSDP2? 它的具体含义是什么?

FSDP2(Fully Sharded Data Parallel v2) 是 PyTorch 2.4+ 推出的新一代`完全切片数据并行`架构(底层重构自 `torch.distributed.fsdp.fully_shard`). 它是对微软 DeepSpeed ZeRO-3 思想在 PyTorch 原生生态中的高性能实现.

#### 核心含义(ZeRO-3 原理):

在传统数据并行(DDP)中, 每个 GPU 都要完整保存一份模型参数、梯度和优化器状态. 而 FSDP2 则将这三者在数据并行维度(DP Group)上进行完全切片(Sharding):

1. 模型参数(Parameters): 每个 GPU 只持有 $1/N$ 的参数切片($N$ 为 GPU 卡数).

2. 梯度(Gradients): 每个 GPU 只计算并保留自己负责的那 $1/N$ 参数对应的梯度.

3. 优化器状态(Optimizer States): 如 Adam 的一阶/二阶动量, 每个 GPU 只更新和存储属于自己的 $1/N$ 切片.

#### FSDP2 与 FSDP1 的本质区别:

* FSDP1(旧版): 基于 `FlatParameter` 机制. 它将一个 Module 内的所有参数铺平拼接成一个巨大的 1维 Tensor, 导致无法独立访问单个参数, 冻结部分参数(Freeze)、混合精度转换、量化(QLoRA)以及与张量并行(TP)组合时非常繁琐.

* FSDP2(新版): 基于 `DTensor`(Distributed Tensor)按参数粒度(Per-Parameter)切片. 每个参数仍然保持其原始维度(如 2D 的 Linear 矩阵), 但底层类型变为 `DTensor`(在 `dim=0` 上分片). 这使得它具有极强的可组合性、可观测性, 并且显著降低了显存开销与调度延迟.

---

### 2. 为什么需要 FSDP2? (解决什么痛点)

大语言模型(LLM)的显存占用主要由 `静态显存(参数 + 梯度 + 优化器状态)` 和 `动态显存(激活值 Activation + 临时缓冲区)` 组成.

#### 痛点 1: 单卡显存爆满(OOM)

以一个 8B 参数的模型(FP16/BF16 训练)为例:

* 参数量: 80 亿 $\times$ 2 bytes = 16 GB

* 梯度: 80 亿 $\times$ 2 bytes = 16 GB

* AdamW 优化器状态(FP32 主权重 + 动量 $m$ + 方差 $v$): 80 亿 $\times$ 12 bytes = 96 GB

* 仅静态显存就已经高达 128 GB! 即使是 80GB 的 A100/H100, 连模型都装不下, 更别说运行 Forward/Backward 过程了.

#### 解决思路: 切片大幅降低显存

使用 FSDP2 在 8 卡($N=8$)集群上时, 静态显存开销会被直接除以 8, 单卡静态显存从 128 GB 降至 16 GB, 使单卡顺利运行大模型训练成为可能.

#### 选择 FSDP2 而非 FSDP1 的原因:

1. 彻底解决显存碎片化与内存泄漏: 弃用了容易导致显存不可控膨胀的 `recordStream`, 显存开销极低且完全可预测.

2. 原生支持 2D/3D 混合并行: 可以非常方便地与张量并行(TP)、流水线并行(PP)叠加组合(如 HSDP: 节点内 TP + 节点间 FSDP2).

3. 更好支持 `torch.compile` 与 Eager 模式: 执行效率(MFU)更高.

---

### 3. FSDP2 与 DDP、TP、PP 的区别是什么?

在 LLM 分布式训练中, 这四种技术处于不同的平行维度, 对比总结如下表:

| 技术名称                      | 类别                     | 拆分对象                                                     | 通信模式                                                         | 网络带宽需求                        | 适用场景                                     |
| :---------------------------- | :----------------------- | :----------------------------------------------------------- | :--------------------------------------------------------------- | :---------------------------------- | :------------------------------------------- |
| DDP <br> (Data Parallel)      | 数据并行                 | 数据 Batch 拆分 <br> (模型每卡复制一份)                      | 后向传播时做 `All-Reduce` 同步梯度                               | 低 ~ 中                             | 模型能够装入单卡显存, 仅需加速训练           |
| FSDP2 <br> (Fully Sharded DP) | 数据并行 <br> (切片型)   | 数据 Batch + 模型参数/梯度/优化器状态切片                    | 前后向 `All-Gather` 临时收集参数; 后向 `Reduce-Scatter` 聚合梯度 | 高 (需要高带宽 NVLink / InfiniBand) | 模型无法装入单卡, 且拥有高速互联网络         |
| TP <br> (Tensor Parallel)     | 模型并行 <br> (层内切分) | 单个 Linear/Attention 矩阵内部按行列切分                     | 每一层内部多次 `All-Reduce` / `All-Gather`                       | 极高 (通常仅限节点内 NVLink)        | 超大模型(如 70B+), 配合 FSDP2 在单节点内使用 |
| PP <br> (Pipeline Parallel)   | 模型并行 <br> (层间切分) | 模型按 Layer 深度切分到不同卡 (如 0-15 层卡 0, 16-31 层卡 1) | 节点间点对点(P2P)传输 Activation                                 | 低                                  | 跨节点/跨机房训练超大模型, 减少通信瓶颈      |

> 研究员建议的工业界常见组合策略(以 70B LLM 为例):
> * 单节点(8卡): TP=2/4 + FSDP2=4/2
> * 多节点(64卡): TP=8(节点内 NVLink) + FSDP2=8(跨节点) 或 PP=4 + TP=4 + FSDP2=4
>

---

### 4. 代码示例: 如何使用 PyTorch FSDP2

在 PyTorch 2.4+ 中, FSDP2 的使用体验大幅简化, 主要依靠 `from torch.distributed.fsdp import fully_shard` API.

#### 黄金法则: 自底向上(Bottom-Up)应用 `fully_shard`

为了让 FSDP2 实现通信与计算的最佳重叠(Overlap), 我们必须先对子模块(如 Transformer Block)调用 `fully_shard`, 最后再对根模型(Root Model)调用 `fully_shard`.

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist

# 1. 导入 FSDP2 的核心 API
from torch.distributed.fsdp import fully_shard


# ----------------- 定义一个简易 LLM 结构 -----------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ----------------- 分布式初始化与训练逻辑 -----------------
def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # 1. 实例化模型(可在 meta device 或直接在 GPU 实例化)
    model = SimpleLLM(dim=4096, num_layers=8).to(device)

    # 2. [核心步骤]: 自底向上应用 FSDP2 (fully_shard)
    # A. 先对每一个 Transformer Block 进行分片
    for layer in model.layers:
        fully_shard(layer)

    # B. 最后对根模型进行分片(自动处理 Embeddings 和 Head 等剩余参数)
    fully_shard(model)

    # 3. [注意]: 优化器必须在应用 fully_shard 之后创建!
    # 因为此时 model.parameters() 已经被转换为了 DTensor 切片
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 4. 模拟训练数据与 Forward / Backward Loop
    inputs = torch.randint(
        0, 10000, (4, 128), device=device
    )  # Batch_size=4, Seq_len=128
    targets = torch.randint(0, 10000, (4, 128), device=device)
    criterion = nn.CrossEntropyLoss()

    for step in range(5):
        optimizer.zero_grad()

        # 前向传播: FSDP2 自动通过 Hook 触发 All-Gather 临时收集当前 Block 参数
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 10000), targets.view(-1))

        # 反向传播: FSDP2 自动计算梯度并通过 Reduce-Scatter 将梯度切片分发
        loss.backward()

        # 优化器更新: 每个卡只更新自己拥有的参数切片和优化器状态
        optimizer.step()

        if local_rank == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    # 启动命令(假设单机 4 卡):
    # torchrun --nproc_per_node=4 train_fsdp2.py
    main()

```

### 总结关键点:

1. 机制: FSDP2 以 `DTensor` 为核心, 实现逐参数(Per-Parameter)切片, 消除了 FSDP1 `FlatParameter` 的不可组合弊端.

2. 应用顺序: 务必遵守自底向上(Bottom-Up)调用 `fully_shard`.

3. 优化器构建: 必须在 `fully_shard(model)` 之后实例化 Optimizer, 否则优化器无法跟踪切片后的 `DTensor` 参数.

-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`单机多卡 FSDP2`, 使用标准的 Python 命令执行训练脚本, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本
```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

这里为你提供一份完全按照 PyTorch 2.4+ 标准 FSDP2(`fully_shard`)规范编写的单机 8 卡训练脚本.

该脚本包含了分布式环境初始化、自底向上的 FSDP2 模型分片、模拟数据训练、基于分布式 Checkpoint(DCP)的高效模型保存以及最后的资源回收.

---

### 1. 完整训练脚本 (`train_fsdp2.py`)

将以下代码保存为 `train_fsdp2.py`:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 FSDP2 核心 API
from torch.distributed.fsdp import fully_shard

# 导入 PyTorch Distributed Checkpoint (DCP) 模块用于分片保存模型
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# ================= 1. 模型定义 =================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ================= 2. 分布式环境管理 =================
def setup_distributed():
    """初始化 NCCL 分布式进程组"""
    dist.init_process_group(backend="nccl")

    # 从 torchrun 环境变量中获取 rank 和 local_rank
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置当前进程绑定的 GPU
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    """资源回收: 销毁分布式进程组"""
    dist.barrier()  # 等待所有卡完成收尾
    dist.destroy_process_group()


# ================= 3. 主训练函数 =================
def main():
    # 初始化分布式环境
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[INFO] 训练开始: 成功初始化分布式环境, 共 {world_size} 张 GPU.")

    # 配置模型参数
    vocab_size = 10000
    dim = 2048
    num_layers = 4

    # 1. 实例化模型并推送到当前卡
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)

    # 2. 应用 FSDP2 (`fully_shard`) -- 必须遵守"自底向上 (Bottom-Up)"原则
    # 第一步: 先对每个内层的 TransformerBlock 进行分片
    for layer in model.layers:
        fully_shard(layer)
    # 第二步: 对外层的根模型进行分片(处理剩余的 embed 和 head 属性)
    fully_shard(model)

    # 3. 构造优化器([注意]: 必须在 fully_shard 之后创建, 此时参数已被转为 DTensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    model.train()

    # 4. 模拟训练过程
    epochs = 5
    batch_size = 4
    seq_len = 128

    for epoch in range(1, epochs + 1):
        # 每张 GPU 随机生成当前 Batch 的输入数据
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        # 前向传播: FSDP2 自动通过通信收集该 Block 的全量权重
        outputs = model(inputs)  # Shape: [batch_size, seq_len, vocab_size]

        # 计算 Loss
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # 反向传播: 计算梯度并自动执行 Reduce-Scatter 分发回各卡
        loss.backward()

        # 优化器更新(每张卡只更新属于自己的那 1/N 状态)
        optimizer.step()

        if rank == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

    # 5. 模型与优化器状态保存 (使用 PyTorch 推荐的 Distributed Checkpoint - DCP)
    checkpoint_dir = "./fsdp2_checkpoint"

    # 主进程清理旧的保存目录
    if rank == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"\n[INFO] 正在保存模型 Checkpoint 到 {checkpoint_dir} ...")

    dist.barrier()  # 等待目录清理完毕

    # 获取支持分布式切片的状态字典 (get_state_dict 原生支持 FSDP2 的 DTensor)
    model_state, optim_state = get_state_dict(model, optimizer)
    state_dict = {
        "model": model_state,
        "optimizer": optim_state,
    }

    # DCP 会让 8 张 GPU 并行将各自的切片写盘, 效率极高且不会造成主卡 OOM
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    dist.barrier()
    if rank == 0:
        print(f"[SUCCESS] 训练完成! Checkpoint 已成功保存.")

    # 6. 销毁进程组, 回收资源
    cleanup()


if __name__ == "__main__":
    main()

```

---

### 2. 如何运行脚本?

在 PyTorch 分布式训练中, 不推荐直接使用 `python train_fsdp2.py`, 因为单进程无法自动拉起多卡环境.

标准的方式是使用 PyTorch 原生的多进程拉起工具 `torchrun`.

#### 执行命令:

打开终端, 直接运行以下命令:

```bash
torchrun --nproc_per_node=8 train_fsdp2.py
```

#### 命令参数解析:

* `torchrun`: PyTorch 官方提供的分布式拉起工具(替代了旧版的 `python -m torch.distributed.launch`).
* `--nproc_per_node=8`: 指定在当前单台机器(Node)上启动 8 个进程, 自动绑定机器上的 8 张 GPU(卡 0 ~ 卡 7).
* `train_fsdp2.py`: 我们要执行的训练脚本文件名.

---

### 3. 代码关键细节拆解说明

1. 分布式环境绑定 (`setup_distributed`):

   `torchrun` 会自动在后台注入环境变量 `RANK`(全局进程 ID)、`LOCAL_RANK`(单机内的卡号)和 `WORLD_SIZE`(总卡数). 代码中使用 `torch.cuda.set_device(local_rank)` 确保每个进程绑定对应的 GPU.

2. 自底向上封装 (`fully_shard`):

   ```python
   for layer in model.layers:
       fully_shard(layer)
   fully_shard(model)
   ```

   这是 FSDP2 的黄金准则. 通过颗粒度更细的模块封装, 使得前向和后向传播时, 通信(All-Gather)与计算可以被重叠(Overlap), 达到最高的计算效率(MFU).

3. 优化器的创建时机:

   必须在 `fully_shard(model)` 之后创建 `AdamW` 优化器. 因为 FSDP2 会在原地将模型参数封装为 `DTensor`, 如果在分片前创建优化器, 优化器引用的将是普通的 Tensor, 导致切片状态失效.

4. 分布式 Checkpoint 保存 (DCP):

   脚本使用了 `torch.distributed.checkpoint` (DCP) 结合 `get_state_dict` 来保存权重. 相比传统的 `torch.save(model.state_dict())`(需要先把所有卡权重通信收集到 Rank 0 显存中, 极易导致 OOM), DCP 会让 8 张 GPU 并行写盘, 将各自的 `DTensor` 切片保存, 既省显存又极其快速.

5. 显存与资源回收 (`cleanup`):

   在脚本末尾调用 `dist.barrier()` 确保 8 张卡全部写入数据完成后, 再调用 `dist.destroy_process_group()` 优雅释放 NCCL 通信句柄与显存资源.

-----------------------------------------------------------------------------------------

Q: 假设我现在有 2 台机器, 每台机器上有 8 张 GPU, 现在我想模拟`多机多卡 FSDP2`, 使用标准的 Python 命令执行训练脚本, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本
```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

在 PyTorch 分布式设计中, 从单机多卡扩展到多机多卡, Python 训练脚本的核心逻辑几乎是 100% 兼容且通用的. PyTorch 的 `torchrun` 会自动将环境变量(如 `RANK`、`WORLD_SIZE`)注入到不同机器的进程中.

以下是针对 2 台机器 $\times$ 8 张 GPU(共 16 张 GPU) 的完整多机多卡 FSDP2 训练脚本及启动命令.

---

### 1. 完整多机多卡训练脚本 (`train_fsdp2_multinode.py`)

创建脚本文件 `train_fsdp2_multinode.py`:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 FSDP2 核心 API
from torch.distributed.fsdp import fully_shard

# 导入 PyTorch Distributed Checkpoint (DCP) 模块
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# ================= 1. 模型定义 =================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ================= 2. 多机分布式环境初始化 =================
def setup_distributed():
    """初始化多机 NCCL 通信组"""
    dist.init_process_group(backend="nccl")

    # 获取分布式节点信息
    rank = int(os.environ["RANK"])  # 全局唯一 Rank ID (0 ~ 15)
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前节点内的 GPU 卡号 (0 ~ 7)
    world_size = int(os.environ["WORLD_SIZE"])  # 全局总卡数 (16)

    # 核心: 必须绑定当前进程到本机对应的 local_rank 卡上, 而不是全局 rank
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    """多机资源回收"""
    dist.barrier()  # 强同步, 等待所有机器的卡执行完毕
    dist.destroy_process_group()


# ================= 3. 主训练函数 =================
def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 仅在主节点的主卡(Global Rank 0)打印日志
    if rank == 0:
        print(f"[INFO] 成功启动多机多卡训练! 总卡数 WORLD_SIZE = {world_size}")

    vocab_size = 10000
    dim = 2048
    num_layers = 4

    # 1. 实例化模型并移动到本机的对应 GPU
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)

    # 2. 应用 FSDP2 (`fully_shard`) -- 跨两台机器(16卡)切片模型参数
    # 自底向上封装: 先封装子模块 TransformerBlock
    for layer in model.layers:
        fully_shard(layer)
    # 再封装根模型 SimpleLLM
    fully_shard(model)

    # 3. 构造优化器(必须在 fully_shard 之后)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    model.train()

    epochs = 5
    batch_size = 4
    seq_len = 128

    for epoch in range(1, epochs + 1):
        # 16 张 GPU 会各自独立生成随机样本(相当于全球总 Batch Size = 4 * 16 = 64)
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        # 前向传播: 跨 16 卡收集临时全量参数
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # 反向传播: 跨 16 卡执行 Reduce-Scatter 切分梯度
        loss.backward()

        # 优化器更新各自负责的 1/16 状态
        optimizer.step()

        if rank == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

    # 4. 多机 Checkpoint 保存 (使用分布式 Checkpoint - DCP)
    # 注意: 在多机环境下, 推荐将 checkpoint 保存到两台机器都能访问到的共享文件系统(如 NFS/Ceph/Lustre)
    # 如果没有共享存储, DCP 也会在每台机器的本地路径各自写入属于该机器上 8 张卡的状态切片
    checkpoint_dir = "./fsdp2_multinode_checkpoint"

    if rank == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"\n[INFO] 正在跨 16 张 GPU 分布式保存 Checkpoint 到: {checkpoint_dir}")

    dist.barrier()  # 确保主节点准备好目录

    # 获取包含 DTensor 切片的状态字典
    model_state, optim_state = get_state_dict(model, optimizer)
    state_dict = {
        "model": model_state,
        "optimizer": optim_state,
    }

    # DCP 16 卡并行写入自己的切片文件
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    dist.barrier()
    if rank == 0:
        print(f"[SUCCESS] 2机16卡分布式 Checkpoint 保存完成!")

    # 5. 回收分布式资源
    cleanup()


if __name__ == "__main__":
    main()

```

---

### 2. 如何在 2 台机器上运行脚本?

假设你的两台机器环境如下:

* 机器 0 (主节点 Master Node): IP 为 `192.168.1.100`, 内有 8 张 GPU.

* 机器 1 (工作节点 Worker Node): IP 为 `192.168.1.101`, 内有 8 张 GPU.

* 通信端口: 主节点上开放端口(例如 `29500`), 确保两台机器网络互通.

#### 步骤 1: 在两台机器上放置相同的代码文件

确保机器 0 和机器 1 的相同目录下都有 `train_fsdp2_multinode.py`.

#### 步骤 2: 在机器 0(主节点)终端执行以下命令

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train_fsdp2_multinode.py
```

#### 步骤 3: 在机器 1(工作节点)终端执行以下命令

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train_fsdp2_multinode.py
```

---

### 3. 多机训练的关键参数说明

| 参数说明           | 机器 0 (Master) 的设置 | 机器 1 (Worker) 的设置 | 含义说明                                          |
| :----------------- | :--------------------- | :--------------------- | :------------------------------------------------ |
| `--nnodes`         | `2`                    | `2`                    | 参与训练的总机器数(Node 数)                       |
| `--nproc_per_node` | `8`                    | `8`                    | 每台机器上的 GPU 数量                             |
| `--node_rank`      | `0`                    | `1`                    | 机器编号(Master 固定为 0,  Worker 依次为 1, 2...) |
| `--master_addr`    | `192.168.1.100`        | `192.168.1.100`        | 主节点的 IP 地址(两边填完全一样的 Master IP)      |
| `--master_port`    | `29500`                | `29500`                | 通信端口                                          |

---

### 4. 多机训练的注意事项(高级研究员建议)

1. 网卡环境变量(通信超时卡死的常见原因):

   如果机器有多个网卡(如同时有万兆网卡和普通网卡, 或者 Infiniband 节点), NCCL 可能会绑错网卡导致卡死. 建议在运行命令前显式指定网卡, 例如:

   ```bash
   export NCCL_SOCKET_IFNAME=eth0   # 替换为你机器实际的主网卡名称(如 bond0, enp1s0 等)
   export GLOO_SOCKET_IFNAME=eth0
   ```

2. 跨机通信带宽与性能优化(混合分片 HSDP):

   * 在跨机训练中, 节点内(NVLink 互联)带宽极大(可达 900GB/s), 而节点间网络(以太网/InfiniBand)带宽相对有限.

   * 如果跨机传输成了瓶颈, 可以使用 FSDP2 支持的 HSDP(Hybrid Sharded Data Parallel): 即节点内 8 卡做 FSDP2 参数切片, 节点间做传统 DDP 复制.

   * PyTorch 2.4+ 通过 `DeviceMesh` 只需要增加 3 行代码即可将上述纯 FSDP2 无缝提升为 HSDP(节点内切片 + 节点间复制):

     ```python
     from torch.distributed.device_mesh import init_device_mesh

     # 创建 2D 设备网格: 节点间(2) x 节点内(8)
     mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))

     # 传入 device_mesh, FSDP2 就会自动按 HSDP 策略执行!
     for layer in model.layers:
         fully_shard(layer, mesh=mesh)
     fully_shard(model, mesh=mesh)
     ```

-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`单机多卡 FSDP2`, 使用 torchrun 执行训练脚本, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本
```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

这是针对单机 8 卡 GPU 环境、使用 `torchrun` 启动的标准 PyTorch FSDP2(`fully_shard`)完整训练脚本.

脚本中包含了:

1. 分布式环境自动初始化与显卡绑定;

2. 自底向上的 FSDP2 切片机制;

3. 延迟实例化的优化器配置;

4. 随机 Batch 数据生成与训练 Loop;

5. 基于 Distributed Checkpoint (DCP) 的高效多卡并行权重保存;

6. 资源清理与进程回收.

---

### 1. 完整代码文件 (`train_fsdp2_singlenode.py`)

将以下代码保存为 `train_fsdp2_singlenode.py`:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 FSDP2 核心 API
from torch.distributed.fsdp import fully_shard

# 导入 PyTorch 原生 Distributed Checkpoint (DCP) 模块
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# ================= 1. 模型定义 =================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ================= 2. 环境初始化与资源回收 =================
def setup_distributed():
    """初始化单机多卡 NCCL 通信组"""
    dist.init_process_group(backend="nccl")

    # torchrun 会自动注入 RANK, LOCAL_RANK, WORLD_SIZE 等环境变量
    rank = int(os.environ["RANK"])  # 0 ~ 7
    local_rank = int(os.environ["LOCAL_RANK"])  # 0 ~ 7
    world_size = int(os.environ["WORLD_SIZE"])  # 8

    # 将当前子进程绑定到指定的卡上
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    """销毁进程组, 回收 GPU 通信资源"""
    dist.barrier()  # 等待所有卡完全退出训练与写盘
    dist.destroy_process_group()


# ================= 3. 主训练流程 =================
def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[INFO] 成功初始化环境: 单机 8 卡环境, WORLD_SIZE = {world_size}")

    vocab_size = 10000
    dim = 2048
    num_layers = 4

    # A. 实例化模型并推送至对应的显卡
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)

    # B. 应用 FSDP2 (`fully_shard`) -- 遵循"自底向上 (Bottom-Up)"原则
    # 步骤 1: 优先对内层 TransformerBlock 子模块分片
    for layer in model.layers:
        fully_shard(layer)
    # 步骤 2: 对外层根模型分片(处理剩余 Embed 和 Head 矩阵)
    fully_shard(model)

    # C. 构造优化器([注意]: 必须在 fully_shard 之后创建)
    # 此时 model.parameters() 中的参数已被转换为 FSDP2 专属的 DTensor 切片
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    model.train()

    # D. 模拟训练过程
    epochs = 5
    batch_size = 4
    seq_len = 128

    for epoch in range(1, epochs + 1):
        # 每张 GPU 独立生成本地的数据
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        # 前向传播: FSDP2 自动通过通信收集该层的临时全量权重
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # 反向传播: 计算梯度并 Reduce-Scatter 散播切片
        loss.backward()

        # 优化器更新各自负责的 1/8 状态
        optimizer.step()

        if rank == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

    # E. 模型 Checkpoint 保存(使用 Distributed Checkpoint - DCP)
    checkpoint_dir = "./fsdp2_single_node_ckpt"

    if rank == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"\n[INFO] 正在将模型 Checkpoint 保存至: {checkpoint_dir}")

    dist.barrier()  # 等待 Rank 0 目录整理完成

    # 提取支持 DTensor 的分布式状态字典
    model_state, optim_state = get_state_dict(model, optimizer)
    state_dict = {
        "model": model_state,
        "optimizer": optim_state,
    }

    # DCP 会让 8 张卡并行写入各自的切片到目标目录, 显存开销极小且速度极快
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    dist.barrier()
    if rank == 0:
        print(f"[SUCCESS] 8 卡分布式 Checkpoint 保存完成!")

    # F. 资源清理
    cleanup()


if __name__ == "__main__":
    main()

```

---

### 2. 如何运行脚本?

打开终端, 使用标准的 PyTorch 原生命令 `torchrun` 执行:

```bash
torchrun --nproc_per_node=8 train_fsdp2_singlenode.py
```

#### 关键参数解析:

* `torchrun`: PyTorch 官方的多进程启动器, 会自动拉起 8 个独立进程, 并自动向每个进程注入 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 `MASTER_ADDR` / `MASTER_PORT` 环境变量.

* `--nproc_per_node=8`: 告诉 `torchrun` 当前单台机器上运行 8 个 GPU 进程.

---

### 3. 代码的核心点总结(为什么这样写?)

1. 分布式绑定 (`setup_distributed`):

   使用 `torch.cuda.set_device(local_rank)` 保证每个进程独占一张 GPU.

2. 自底向上封装 (`fully_shard`):

   优先对子模块 `TransformerBlock` 封装, 再对根模型 `SimpleLLM` 封装. 这种分层结构允许 FSDP2 在计算上一个 Layer 的 Forward/Backward 时, 预取(Prefetch)下一个 Layer 的参数, 从而隐藏通信开销.

3. 延迟构造优化器:

   优化器必须在 `fully_shard(model)` 调用完成之后定义. 这样 `model.parameters()` 传给 `AdamW` 的就是切片后的 `DTensor` 参数.

4. 分布式 Checkpoint (`dcp.save`):

   避免了传统 `torch.save` 需要将所有卡参数收集回 CPU 主显存(极易引起 OOM)的问题, 8 张 GPU 会以并行 IO 的方式各自写盘, 是大模型训练的标准做法.

-----------------------------------------------------------------------------------------

Q: 假设我现在有 2 台机器, 每台机器上有 8 张 GPU, 现在我想模拟`多机多卡 FSDP2`, 使用 torchrun 执行训练脚本, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本
```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

这是针对 2 台机器 $\times$ 8 张 GPU(共 16 张 GPU) 的多机多卡 FSDP2 完整训练脚本以及使用 `torchrun` 启动训练的详细指南.

---

### 1. 完整多机多卡训练脚本 (`train_fsdp2_multinode.py`)

创建脚本文件 `train_fsdp2_multinode.py`:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 FSDP2 核心 API
from torch.distributed.fsdp import fully_shard

# 导入 PyTorch Distributed Checkpoint (DCP) 模块
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# ================= 1. 模型定义 =================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ================= 2. 多机分布式环境初始化 =================
def setup_distributed():
    """初始化跨机器 NCCL 通信组"""
    dist.init_process_group(backend="nccl")

    # 获取 torchrun 注入的环境变量
    rank = int(os.environ["RANK"])  # 全局唯一 Rank ID (0 ~ 15)
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前机器内的 GPU 编号 (0 ~ 7)
    world_size = int(os.environ["WORLD_SIZE"])  # 全局总 GPU 卡数 (16)

    # 绑定当前进程到本机的对应 GPU 卡上
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    """多机资源回收"""
    dist.barrier()  # 等待所有机器的所有 GPU 完成任务
    dist.destroy_process_group()


# ================= 3. 主训练流程 =================
def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 仅在全局主节点(Global Rank 0)打印环境信息
    if rank == 0:
        print(f"[INFO] 成功启动多机多卡训练! 全局卡数 WORLD_SIZE = {world_size}")

    vocab_size = 10000
    dim = 2048
    num_layers = 4

    # 1. 实例化模型并移动到本机的对应 GPU
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)

    # 2. 应用 FSDP2 (`fully_shard`) -- 跨两台机器(16卡)切片模型参数
    # 自底向上封装: 先封装内层 TransformerBlock 子模块
    for layer in model.layers:
        fully_shard(layer)
    # 再封装根模型 SimpleLLM
    fully_shard(model)

    # 3. 构造优化器([注意]: 必须在 fully_shard 之后创建)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    model.train()

    epochs = 5
    batch_size = 4
    seq_len = 128

    for epoch in range(1, epochs + 1):
        # 16 张 GPU 会各自独立生成随机样本(全球总 Batch Size = 4 * 16 = 64)
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        # 前向传播: 跨 16 卡收集临时全量参数
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # 反向传播: 跨 16 卡执行 Reduce-Scatter 切分梯度
        loss.backward()

        # 优化器更新各自负责的 1/16 参数切片
        optimizer.step()

        if rank == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

    # 4. 多机 Checkpoint 保存 (使用 Distributed Checkpoint - DCP)
    # 提示: 在多机环境下, 最好保存到共享存储(如 NFS / Ceph), 如果没有共享存储, 各机器会在本地路径保存各自负责的 8 张卡切片
    checkpoint_dir = "./fsdp2_multinode_ckpt"

    if rank == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"\n[INFO] 正在跨 16 张 GPU 分布式保存 Checkpoint 到: {checkpoint_dir}")

    dist.barrier()  # 等待目录初始化完成

    # 获取包含 DTensor 切片的状态字典
    model_state, optim_state = get_state_dict(model, optimizer)
    state_dict = {
        "model": model_state,
        "optimizer": optim_state,
    }

    # DCP 16 卡并行写入自己的切片文件, 写盘速度快且避免 Rank 0 显存爆满
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    dist.barrier()
    if rank == 0:
        print(f"[SUCCESS] 2 机 16 卡分布式 Checkpoint 保存完成!")

    # 5. 回收分布式资源
    cleanup()


if __name__ == "__main__":
    main()

```

---

### 2. 如何使用 `torchrun` 运行脚本?

假设两台机器的信息配置如下:

* 机器 0(主节点 / Master Node): IP 为 `192.168.1.100`, 内有 8 张 GPU.

* 机器 1(工作节点 / Worker Node): IP 为 `192.168.1.101`, 内有 8 张 GPU.

* 确保两台机器之间开放了端口 `29500`(或其他未被占用的端口).

#### 第一步: 在两台机器相同目录下均保存 `train_fsdp2_multinode.py`

#### 第二步: 在机器 0(主节点)上运行以下命令

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train_fsdp2_multinode.py
```

#### 第三步: 在机器 1(工作节点)上运行以下命令

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train_fsdp2_multinode.py
```

---

### 3. 多机参数说明与排坑指南

#### 1. `torchrun` 参数解析:
| 参数               | 机器 0 (Master) | 机器 1 (Worker) | 作用                                           |
| :----------------- | :-------------- | :-------------- | :--------------------------------------------- |
| `--nnodes`         | `2`             | `2`             | 参与训练的机器节点总数                         |
| `--nproc_per_node` | `8`             | `8`             | 每台机器上的 GPU 数量                          |
| `--node_rank`      | `0`             | `1`             | 节点的全局编号(Master 设为 0, Worker 依次递增) |
| `--master_addr`    | `192.168.1.100` | `192.168.1.100` | 主节点 IP(两台机器填同一个地址)                |
| `--master_port`    | `29500`         | `29500`         | 主节点侦听的通信端口                           |

#### 2. 多机网卡绑定的常见问题(通信卡死排查):

如果服务器有多个网卡(比如同时有局域网网卡和 InfiniBand/高速网卡), NCCL 通信可能会因为选错网卡而超时挂起(Hang 住).

在运行 `torchrun` 前, 建议显式设置指定网卡的环境变量, 例如:

```bash
export NCCL_SOCKET_IFNAME=eth0   # 将 eth0 替换为机器实际可互通的网络接口名(如 eth0 / bond0)
export GLOO_SOCKET_IFNAME=eth0
```

-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
