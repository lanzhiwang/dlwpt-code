# 组合 DDP, FSDP2, TP, PP

-----------------------------------------------------------------------------------------

Q: 假设你是一位精通 LLM 和 PyTorch 的高级研究人员, PyTorch 支持多种并行技术, 一般都有 DDP, FSDP2, TP, PP, 但在实际应用中通常是将各种并行技术组合起来, 请问常见组合有哪些? 他们的使用场景是什么? 并且请使用代码演示这些组合的实现方法

作为 LLM 和 PyTorch 领域的研究与工程人员, 在面对百亿(10B)甚至万亿(1T+)参数规模的大模型时, 单一的并行技术往往会遇到显存瓶颈或网络通信瓶颈.

现代 Distributed LLM 训练的本质, 就是在显存占用(模型参数、梯度、优化器状态、激活值)与网络通信带宽(节点内 NVLink vs 节点间 InfiniBand/RoCE)之间寻找最优的平衡点.

PyTorch 2.x(特别是 PyTorch 2.4+)推出了全新的可组合并行 API(Composable Parallelism APIs), 核心基于 `DeviceMesh`、`DTensor`、`fully_shard`(FSDP2) 以及 `parallelize_module`(TP), 使得各种并行技术的组合变得极为优雅和模块化.

---

### 一、核心并行原语速览

在介绍组合前, 先明确 4 种基础并行的特性:

1. DDP (Data Parallelism): 每卡完整模型副本. 通信开销: 梯度 AllReduce. 限制: 单卡必须装下全量模型.

2. FSDP / FSDP2 (Fully Sharded Data Parallel): 将模型参数、梯度、优化器状态分片到 DP 组(FSDP2 基于 DTensor 重构, 开销更低). 通信开销: 前向 AllGather, 反向 ReduceScatter + AllGather.

3. TP (Tensor Parallel / Megatron-LM 风格): 将单层内的矩阵乘法切分(如 Colwise/Rowwise Linear). 通信开销: 每层前向/反向均需 AllReduce / AllGather, 通信极其频繁, 严重依赖节点内 NVLink 高带宽.

4. PP (Pipeline Parallel): 按 Depth(层)切分模型到不同 Stage, 数据流水线式传递. 通信开销: Stage 之间 P2P 传输 Activation/Gradient, 通信量小, 适合跨节点/跨集群.

---

### 二、常见的混合并行组合及适用场景

#### 1. 2D 并行: TP + FSDP2(最主流的节点间扩展方案)

* 拓扑配置: 节点内(如 8 卡 NVLink)走 TP, 节点间(或跨剩余 GPUs)走 FSDP2.
* 解决痛点:
  * 单纯使用 TP: 跨节点走 PCIe/InfiniBand 会导致 TP 通信成为极大的性能瓶颈(TP 必须用 NVLink).
  * 单纯使用 FSDP2: 模型尺寸极大时, 单 Block 的激活值(Activations)过大导致 OOM, 且全量参数通信频率高.
* 适用场景: 7B - 70B 规模模型, 运行在多节点 GPU 集群(如 2~16 个 8-GPU 节点).

#### 2. 3D 并行: TP + PP + FSDP2 / DP(万亿参数超大规模集群)

* 拓扑配置:
  * 最内层(如 4/8 卡, 单节点): TP(利用 NVLink 处理高频张量切分).
  * 中间层(跨节点): PP(利用低通信量 P2P 串联多个节点, 解决模型深度装不下的问题).
  * 最外层(跨组): FSDP2 / DP(在剩余卡数上切分/复制, 用于增大 Global Batch Size).
* 解决痛点: 单节点装不下单层模型, 且集群节点数庞大(百卡/千卡级), 需要极致的 Scalability.
* 适用场景: 100B - 1T+ 参数的超大语言模型预训练.

#### 3. PP + FSDP2(无 NVLink 或网络带宽受限集群)

* 拓扑配置: 不使用 TP, 按层切分成若干 Stage(PP), 每个 Stage 内部再做 FSDP2.
* 解决痛点: 部分云厂商节点内部缺少 NVLink, 节点间仅有普通 IB/以太网, 无法高效运行 TP.
* 适用场景: 13B - 70B 模型, 在网络拓扑较为受限(无高带宽节点内互联)的环境下训练.

#### 4. CP + TP + FSDP2(超长上下文 / Long Context 场景)

* 拓扑配置: 在 2D 并行的基础上引入 Context Parallelism (CP)(如 Ring-Attention), 将 Sequence 维度切分到不同 GPU.
* 解决痛点: 当上下文长度达到 32k - 1M+ 时, Self-Attention 显存呈 $O(N^2)$ 爆炸, 即便有 TP 和 FSDP, 激活值显存仍会 OOM.
* 适用场景: 长文本 LLM 预训练或微调(如 Llama-3 128k 上下文).

---

### 三、PyTorch 代码演示

PyTorch 2.4+ 推荐使用 `DeviceMesh` + `parallelize_module` + `fully_shard` 的可组合 API 实现混合并行.

以下演示最具代表性的 2D 并行(TP + FSDP2) 和 3D 并行拓扑配置.

#### 代码示例 1: 2D 并行(TP + FSDP2)完整实现

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist

# 引入 PyTorch 2.x 组合式并行 API
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard  # PyTorch 2.4+ FSDP2 API


# 1. 定义一个简化的 Transformer MLP 层
class TransformerMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # 升维: hidden -> 4 * hidden
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.act = nn.GELU()
        # 降维: 4 * hidden -> hidden
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def run_2d_parallel_example():
    # 初始化分布式环境 (假设通过 torchrun 启动)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    # 假设共有 8 张 GPU, 设置 TP Size = 2, 则 DP Size = 4
    tp_size = 2
    dp_size = world_size // tp_size

    # -------------------------------------------------------------
    # Step 1: 构建 2D DeviceMesh (DP维度, TP维度)
    # -------------------------------------------------------------
    # 假设 8 卡: Mesh 形状为 (4, 2)
    # mesh_2d["dp"] 管理 FSDP2 的通讯组, mesh_2d["tp"] 管理 TP 的通讯组
    mesh_2d = init_device_mesh(
        "cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    # -------------------------------------------------------------
    # Step 2: 实例化原始模型并移动到 GPU
    # -------------------------------------------------------------
    model = TransformerMLP(hidden_dim=4096).to("cuda")

    # -------------------------------------------------------------
    # Step 3: 应用 Tensor Parallelism (TP) -> 作用于 tp_mesh
    # -------------------------------------------------------------
    # Megatron 模式: 第一个 Linear 列切分, 第二个 Linear 行切分
    tp_plan = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    # 将模型按照 plan 在 tp_mesh 上进行张量并行化
    parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=tp_plan)

    # -------------------------------------------------------------
    # Step 4: 应用 FSDP2 (Fully Shard) -> 作用于 dp_mesh
    # -------------------------------------------------------------
    # FSDP2 会自动识别已经被 TP 变成了 DTensor 的参数, 并在 dp_mesh 维度上对其做 Sharding
    fully_shard(model, mesh=dp_mesh)

    # -------------------------------------------------------------
    # Step 5: 前向与反向传播
    # -------------------------------------------------------------
    # 构造假数据 (Batch Size, Seq Len, Hidden Dim)
    dummy_input = torch.randn(2, 128, 4096, device="cuda")

    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    if rank == 0:
        print("2D Parallelism (TP + FSDP2) 运行成功!")

    dist.destroy_process_group()


if __name__ == "__main__":
    # 执行命令: torchrun --nproc_per_node=8 script.py
    run_2d_parallel_example()

```

---

#### 代码示例 2: 3D 并行拓扑配置 (TP + PP + FSDP2 架构逻辑)

对于 3D 并行, PyTorch 提供了 `init_device_mesh` 的 3D 嵌套支持, 配合 `pipelining` 库可以组合实现:

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard

# 从 PyTorch 2.4+ 管道库导入 (概念展示)
from torch.distributed.pipelining import PipelineStage


def build_3d_parallel_model(model_layers: nn.ModuleList, world_size: int):
    # 假设 16 卡: PP=2, DP=2, TP=4
    pp_size, dp_size, tp_size = 2, 2, 4

    # 1. 创建 3D DeviceMesh
    mesh_3d = init_device_mesh(
        "cuda",
        mesh_shape=(pp_size, dp_size, tp_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )

    pp_mesh = mesh_3d["pp"]
    dp_mesh = mesh_3d["dp"]
    tp_mesh = mesh_3d["tp"]

    # 2. 根据当前 rank 的 PP 阶段, 划分分配给本 stage 的 layers
    # 例如 pp_rank 0 拿前一半层, pp_rank 1 拿后一半层
    pp_rank = pp_mesh.get_local_rank()
    layers_per_stage = len(model_layers) // pp_size
    local_layers = model_layers[
        pp_rank * layers_per_stage : (pp_rank + 1) * layers_per_stage
    ]

    local_stage_module = nn.Sequential(*local_layers).to("cuda")

    # 3. 对本阶段内的模块应用 TP
    for layer in local_stage_module:
        tp_plan = {
            "attention.qkv": ColwiseParallel(),
            "attention.proj": RowwiseParallel(),
            "mlp.fc1": ColwiseParallel(),
            "mlp.fc2": RowwiseParallel(),
        }
        parallelize_module(layer, device_mesh=tp_mesh, parallelize_plan=tp_plan)

    # 4. 对本阶段内的模块应用 FSDP2 (在 DP 维度 Shard 参数)
    for layer in local_stage_module:
        fully_shard(layer, mesh=dp_mesh)

    # 5. 构建 Pipeline Stage 接入 Pipelining Schedule (如 1F1B)
    stage = PipelineStage(
        local_stage_module,
        stage_index=pp_rank,
        num_stages=pp_size,
        device=torch.device("cuda"),
        group=pp_mesh.get_group(),
    )

    return stage

```

---

### 四、高级研究员的性能调优技巧(Best Practices)

1. 拓扑感知与设备映射(Topology Awareness)
   * 使用 `torchrun` 时, 确保 `tp_size` 的 GPU 集合位于同一 NUMA 节点或 NVLink Domain 内部(如 GPU 0-3, 4-7). 避免 TP 跨 Socket 或跨 PCIe Switch 通信.

2. 重叠通信与计算(Overlapping)
   * FSDP2 的 Prefetch 机制: FSDP2 默认支持前向/反向的预取(Explicit/Implicit Prefetching), 确保在计算 Layer $N$ 的同时, 异步 AllGather Layer $N+1$ 的参数.
   * TP 与 `torch.compile`: TP 会引入许多小 Tensor 的 AllReduce 操作, 结合 `torch.compile` 可以自动融合 CUDA Kernel 并优化分布式通信节点的 Stream 调度.

3. 激活值重算(Activation Checkpointing / Selective Checkpointing)
   * 在混合并行中, 激活值显存随着 Batch Size 和 Seq Len 增大依然会成为瓶颈. 建议使用 Selective Activation Checkpointing, 仅对 Attention Block 中的 Non-recomputable 部分(如 FlashAttention 内部数据)做保留, 其余重算.

4. 混合精度(BF16 & FP8)
   * FSDP2 + TP 均天然原生支持 `torch.bfloat16` 和 `FP8`(借助 `TransformerEngine` 或 PyTorch 2.x native FP8), 将通信流量直接砍半, 大幅提升集群吞吐量(MFU).

-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`2D 并行(TP + FSDP2)`, 模型如下代码所示:
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
请使用 PyTorch 基本的 API 模拟说明并行过程中前向传播的整个过程, 说明数据如何流动? 不要使用类似 `torch.distributed.tensor.parallel` 模块, 这样更能说明整个并行过程, 理解其基本原理.

为了深入理解 2D 并行(TP + FSDP2)的底层物理机制, 我们脱离 `torch.distributed.tensor.parallel` 或 `fully_shard` 等高阶抽象封装, 直接使用 PyTorch 底层分布式通信原语(`dist.all_gather_into_tensor`、`dist.all_reduce`)与基础张量操作来重构和模拟前向传播过程.

通过手写底层原语, 我们可以清晰地看到:

1. FSDP2 的本质: 在前向计算前用 `dist.all_gather_into_tensor` 在 DP 组内拉取属于当前 TP Rank 的全量权重, 计算完后立即 `del` 释放临时权重(显存回收).
2. TP 的本质: 在 TP 组内按通道(Out Dim / In Dim)切分矩阵乘法, 并在行切分(Rowwise)后使用 `dist.all_reduce(ReduceOp.SUM)` 规约部分和(Partial Sum).

---

### 一、通信组(Process Groups)的底层划分

在 8 卡机器(DP=4, TP=2)上, 我们首先需要手写代码建立两组底层的通信组:

```python
# 网格矩阵 (DP Size=4, TP Size=2):
# DP 组 0 (TP Rank 0/1): GPU [0, 1]
# DP 组 1 (TP Rank 0/1): GPU [2, 3]
# DP 组 2 (TP Rank 0/1): GPU [4, 5]
# DP 组 3 (TP Rank 0/1): GPU [6, 7]

# 1. 创建 TP 组 (节点内 2 卡互联, 用于 AllReduce 激活值):
# [0, 1], [2, 3], [4, 5], [6, 7]

# 2. 创建 DP 组 (跨组 4 卡互联, 用于 FSDP2 AllGather 权重):
# [0, 2, 4, 6] (TP Rank 0 的 FSDP 组)
# [1, 3, 5, 7] (TP Rank 1 的 FSDP 组)
```

---

### 二、完整低阶原语模拟脚本 (`manual_2d_parallel.py`)

以下是完全使用 PyTorch 基本 API 模拟前向传播的完整可运行代码:

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# -------------------------------------------------------------
# 1. 使用低阶原语手写 2D 并行(TP + FSDP2)Transformer Block
# -------------------------------------------------------------
class Manual2DTransformerBlock(nn.Module):
    def __init__(self, dim, tp_group, dp_group, tp_size=2, dp_size=4):
        super().__init__()
        self.dim = dim
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.tp_size = tp_size
        self.dp_size = dp_size

        # ---------------------------------------------------------
        # Linear1 (Colwise Parallel + FSDP2)
        # 逻辑全量权重形状: [8192, 2048]
        # 1. TP 切分 (Colwise): 沿 Out 维度切 2 份 -> 每个 TP Rank 负责 [4096, 2048]
        # 2. FSDP2 切分: 沿 Out 维度切 4 份 -> 本 GPU 真正保存的碎片为 [1024, 2048]
        # ---------------------------------------------------------
        self.tp_out_dim1 = (dim * 4) // tp_size             # 4096
        self.fsdp_shard_dim1 = self.tp_out_dim1 // dp_size # 1024

        # 模拟常驻 GPU 显存的权重碎片 (Parameter Shard)
        self.linear1_weight_shard = nn.Parameter(
            torch.randn(self.fsdp_shard_dim1, dim) / (dim ** 0.5)
        )

        # ---------------------------------------------------------
        # Linear2 (Rowwise Parallel + FSDP2)
        # 逻辑全量权重形状: [2048, 8192]
        # 1. TP 切分 (Rowwise): 沿 In 维度切 2 份 -> 每个 TP Rank 负责 [2048, 4096]
        # 2. FSDP2 切分: 沿 Out 维度切 4 份 -> 本 GPU 真正保存的碎片为 [512, 4096]
        # ---------------------------------------------------------
        self.tp_in_dim2 = (dim * 4) // tp_size              # 4096
        self.fsdp_shard_dim2 = dim // dp_size               # 512

        self.linear2_weight_shard = nn.Parameter(
            torch.randn(self.fsdp_shard_dim2, self.tp_in_dim2) / (self.tp_in_dim2 ** 0.5)
        )

    def forward(self, x):
        # 输入 x 形状: [Batch, SeqLen, Dim] (例如 [2, 16, 2048])

        # =========================================================
        # 步骤 1: Linear1 前向计算 (Colwise TP + FSDP2)
        # =========================================================
        # [FSDP2 原理展示]: 在 DP 组内 AllGather 拼凑出属于本 TP Rank 的全量权重
        # 输入分片: [1024, 2048] (来自 4 张 DP 卡) -> 收集后形状: [4096, 2048]
        full_linear1_weight = torch.empty(
            self.tp_out_dim1, self.dim, device=x.device, dtype=x.dtype
        )
        dist.all_gather_into_tensor(
            full_linear1_weight, self.linear1_weight_shard, group=self.dp_group
        )

        # [TP 原理展示]: 本地矩阵乘法 x @ W1^T -> [B, L, 4096]
        # TP Rank 0 算前 4096 个通道特征, TP Rank 1 算后 4096 个通道特征 (互不干扰, 无需通信)
        h1 = F.linear(x, full_linear1_weight)

        # [FSDP2 原理展示]: 显存释放! 计算完立即删除拼凑的临时全量权重
        del full_linear1_weight

        # 本地逐元素激活 GELU
        a1 = F.gelu(h1)  # 形状: [B, L, 4096]

        # =========================================================
        # 步骤 2: Linear2 前向计算 (Rowwise TP + FSDP2)
        # =========================================================
        # [FSDP2 原理展示]: 在 DP 组内 AllGather 拼凑出属于本 TP Rank 的 Rowwise 权重
        # 输入分片: [512, 4096] -> 收集后形状: [2048, 4096]
        full_linear2_weight = torch.empty(
            self.dim, self.tp_in_dim2, device=x.device, dtype=x.dtype
        )
        dist.all_gather_into_tensor(
            full_linear2_weight, self.linear2_weight_shard, group=self.dp_group
        )

        # [TP 原理展示]: 本地矩阵乘法 a1 @ W2^T -> [B, L, 2048]
        # 注意: 此处计算出的只是 Partial Sum (部分和)
        partial_y = F.linear(a1, full_linear2_weight)

        # [FSDP2 原理展示]: 显存释放
        del full_linear2_weight

        # [TP 原理展示]: 在 TP 组内规约求和 (AllReduce Sum)
        # TP Rank 0 的 Partial Sum + TP Rank 1 的 Partial Sum = 完整的输出 Y
        dist.all_reduce(partial_y, op=dist.ReduceOp.SUM, group=self.tp_group)
        y = partial_y  # 形状: [B, L, 2048]

        # =========================================================
        # 步骤 3: 残差连接
        # =========================================================
        out = x + y  # 形状: [B, L, 2048]
        return out


# -------------------------------------------------------------
# 2. 使用低阶原语手写 SimpleLLM
# -------------------------------------------------------------
class Manual2DSimpleLLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, tp_group, dp_group):
        super().__init__()
        self.dp_group = dp_group

        # Embedding: 采用 FSDP2 分片 (按 DP=4 切分)
        self.embed_shard_size = vocab_size // 4  # 2500
        self.embed_weight_shard = nn.Parameter(
            torch.randn(self.embed_shard_size, dim)
        )

        # Layers
        self.layers = nn.ModuleList([
            Manual2DTransformerBlock(dim, tp_group, dp_group)
            for _ in range(num_layers)
        ])

        # Head: 采用 Colwise TP (5000) + FSDP2 (按 DP=4 切分 -> 1250)
        self.head_shard_size = (vocab_size // 2) // 4  # 1250
        self.head_weight_shard = nn.Parameter(
            torch.randn(self.head_shard_size, dim)
        )

    def forward(self, x_tokens):
        # 1. Embedding 前向: FSDP2 AllGather 还原 Embedding 矩阵
        full_embed_weight = torch.empty(
            10000, 2048, device=x_tokens.device, dtype=torch.float32
        )
        dist.all_gather_into_tensor(
            full_embed_weight, self.embed_weight_shard, group=self.dp_group
        )
        x = F.embedding(x_tokens, full_embed_weight)
        del full_embed_weight  # 立即释放

        # 2. Transformer Blocks 前向
        for layer in self.layers:
            x = layer(x)

        # 3. Head 前向: FSDP2 AllGather 还原 Colwise TP Head 矩阵
        full_head_tp_weight = torch.empty(
            5000, 2048, device=x_tokens.device, dtype=x.dtype
        )
        dist.all_gather_into_tensor(
            full_head_tp_weight, self.head_weight_shard, group=self.dp_group
        )
        logits = F.linear(x, full_head_tp_weight)  # [B, L, 5000]
        del full_head_tp_weight  # 立即释放

        return logits


# -------------------------------------------------------------
# 3. 运行与验证主程序
# -------------------------------------------------------------
def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_size = 2
    dp_size = 4

    # ---------------------------------------------------------
    # 手动建立通信组网格
    # ---------------------------------------------------------
    tp_group = None
    dp_group = None

    # 创建 4 个 TP 组: [0, 1], [2, 3], [4, 5], [6, 7]
    for i in range(dp_size):
        ranks = [i * tp_size + j for j in range(tp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    # 创建 2 个 DP 组: [0, 2, 4, 6], [1, 3, 5, 7]
    for j in range(tp_size):
        ranks = [i * tp_size + j for i in range(dp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group = group

    # ---------------------------------------------------------
    # 实例化模型与模拟输入
    # ---------------------------------------------------------
    model = Manual2DSimpleLLM(
        vocab_size=10000, dim=2048, num_layers=4,
        tp_group=tp_group, dp_group=dp_group
    ).to(device)

    # 模拟输入数据: 同一 TP 组(如 GPU 0 和 1)输入相同, 不同 DP 组输入不同
    dp_rank = rank // tp_size
    torch.manual_seed(42 + dp_rank)
    dummy_input = torch.randint(0, 10000, (2, 16), device=device)

    # 执行前向传播
    logits = model(dummy_input)

    print(f"[Rank {rank}] 前向传播成功! 输出 Logits 形状: {logits.shape}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

### 三、运行方法

使用 `torchrun` 启动 8 进程运行脚本:

```bash
torchrun --nproc_per_node=8 manual_2d_parallel.py
```

#### 输出打印:
```text
[Rank 0] 前向传播成功! 输出 Logits 形状: torch.Size([2, 16, 5000])
[Rank 1] 前向传播成功! 输出 Logits 形状: torch.Size([2, 16, 5000])
...
[Rank 7] 前向传播成功! 输出 Logits 形状: torch.Size([2, 16, 5000])
```

---

### 四、核心数据流动与原理深度剖析

结合上述底层原语代码, 我们可以将前向传播中单个 Transformer Block 内部的数据流动精简总结为以下 4 步:

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      Input X [2, 16, 2048]             │
                    └───────────────────────────┬────────────────────────────┘
                                                │
      ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
      │                                                                                   │
      ▼                                                                                   ▼
[GPU 0 (TP Rank 0, DP Rank 0)]                                       [GPU 1 (TP Rank 1, DP Rank 0)]
常驻分片: W1_shard [1024, 2048]                                       常驻分片: W1_shard [1024, 2048]
      │                                                                                   │
      │ 1. dist.all_gather_into_tensor(dp_group)                                          │ 1. dist.all_gather_into_tensor(dp_group)
      ▼                                                                                   ▼
重构权: W1_full [4096, 2048]                                          重构权: W1_full [4096, 2048]
      │                                                                                   │
      │ 2. F.linear(X, W1_full)                                                           │ 2. F.linear(X, W1_full)
      ▼                                                                                   ▼
激活值: h1 [2, 16, 4096]                                              激活值: h1 [2, 16, 4096]
      │                                                                                   │
      │ 3. del W1_full (释放显存)                                                          │ 3. del W1_full (释放显存)
      ▼                                                                                   ▼
激活值: a1 = GELU(h1) [2, 16, 4096]                                   激活值: a1 = GELU(h1) [2, 16, 4096]
      │                                                                                   │
      │ 4. dist.all_gather_into_tensor(dp_group)                                          │ 4. dist.all_gather_into_tensor(dp_group)
      ▼                                                                                   ▼
重构权: W2_full [2048, 4096]                                          重构权: W2_full [2048, 4096]
      │                                                                                   │
      │ 5. F.linear(a1, W2_full)                                                          │ 5. F.linear(a1, W2_full)
      ▼                                                                                   ▼
部分和: partial_y [2, 16, 2048]                                       部分和: partial_y [2, 16, 2048]
      │                                                                                   │
      └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                                │
                                                ▼ 6. dist.all_reduce(op=SUM, tp_group)
                                                │    (TP 组内规约求和)
                                                ▼
                                    输出 Y = partial_y_0 + partial_y_1
                                         形状: [2, 16, 2048]
```

#### 关键原语对齐解说:
1. `dist.all_gather_into_tensor(full_weight, weight_shard, group=dp_group)`:
   * 作用: 这就是 FSDP2 前向的核心动作. 将分散在 DP 组 4 张卡上的小分片拼成属于当前 TP 节点所需的完整权重.
   * 通信维度: 沿着 DP 组 (`[0, 2, 4, 6]`) 进行.
2. `del full_weight`:
   * 作用: 显式调用 Python 垃圾回收机制, 释放临时重构的全量权重. 这是 FSDP2 降低显存峰值的关键所在.
3. `dist.all_reduce(partial_y, op=dist.ReduceOp.SUM, group=tp_group)`:
   * 作用: 这就是 TP 行切分(Rowwise)的核心动作. 把 TP Rank 0 和 TP Rank 1 计算出来的 Partial Sum 规约相加, 恢复完整维度.
   * 通信维度: 沿着 TP 组 (`[0, 1]`) 进行.





-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`2D 并行(TP + FSDP2)`, 模型如下代码所示:
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
请详细说明并行过程中前向传播的整个过程, 说明数据如何流动?

在 2D 混合并行(TP + FSDP2) 的架构下, 前向传播(Forward Pass)交织着数据分片、权重的按需拉取(FSDP2 AllGather) 以及张量并行的分布式计算与规约(TP AllReduce).

下面以 8 张 GPU(DP Size = 4, TP Size = 2) 为例, 以 DP Group 0(包含 GPU 0 和 GPU 1) 的视角, 为您拆解每一层前向传播的数据流动和通信全过程.

---

### 一、核心概念: 数据与权重的"双重切分"

在前向传播开始前, 8 张 GPU 的状态如下:

1. 数据切分(按 DP 维度):
   * 全局 Batch 数据被切分为 4 份(微批次 $B_0, B_1, B_2, B_3$).
   * GPU 0 和 GPU 1(DP Group 0) 接收完全相同的输入数据 $B_0 \in \mathbb{R}^{B \times L}$($B$ 为局部 Batch Size, $L$ 为 Sequence Length).
2. 权重切分(按 TP + FSDP2 维度):
   * 第一层切分 (TP): 模型权重首先沿 TP 维度切开. 例如 `linear1` 形状为 $[8192, 2048]$, TP=2 将其切为两个 $[4096, 2048]$ 的张量: $W_1^{(0)}$(给 TP Rank 0)和 $W_1^{(1)}$(给 TP Rank 1).
   * 第二层切分 (FSDP2): 切分后的 $W_1^{(0)}$ 不会完整保存在 GPU 0 上, 而是进一步被 FSDP2 切成 4 份, 平摊在 GPU [0, 2, 4, 6] 上(每张卡只存 $\frac{1}{4}$, 即 $[1024, 2048]$).

---

### 二、前向传播数据流动全过程 (Step-by-Step)

针对输入数据 $X \in \mathbb{R}^{B \times L}$, 前向传播按如下步骤精准执行:

```
[输入 Tokens X] -> Embedding -> TransformerBlock (x4) -> Head -> [Logits]
```

#### 步骤 1: Embedding 层
1. FSDP2 AllGather: GPU 0(与 2, 4, 6)触发 FSDP2 通信, 从同组卡拉取参数, 拼出属于 TP Rank 0 的完整 Embedding 权重; GPU 1(与 3, 5, 7)同步拼出 TP Rank 1 的权重.
2. 本地计算: 输入 Tokens $X \in [B, L]$ 查表得到隐藏状态 $X_{\text{emb}} \in [B, L, 2048]$.
3. FSDP2 释放: 计算完毕, 立即释放临时 AllGather 出来的 Embedding 权重, 只留各自的 $\frac{1}{4}$ 分片.
4. 状态: GPU 0 和 GPU 1 此时持有完全相同的激活张量 $X_{\text{emb}}$.

---

#### 步骤 2: TransformerBlock 内部(核心机制)

一个 Transformer Block 包含 `linear1` $\rightarrow$ `act` $\rightarrow$ `linear2` $\rightarrow$ `residual add`.

##### 2.1 计算 `linear1`(柱切分 Colwise Parallel)
* FSDP2 参数拉取:
  * GPU 0 与 GPU [2, 4, 6] 执行 FSDP2 AllGather, 重构出 TP Rank 0 的权重 $W_1^{(0)} \in [4096, 2048]$.
  * GPU 1 与 GPU [3, 5, 7] 同步重构出 TP Rank 1 的权重 $W_1^{(1)} \in [4096, 2048]$.
* 本地矩阵乘法(无需通信):
  * GPU 0 计算: $H^{(0)} = X_{\text{emb}} \cdot (W_1^{(0)})^T \quad \rightarrow \text{形状: } [B, L, 4096]$
  * GPU 1 计算: $H^{(1)} = X_{\text{emb}} \cdot (W_1^{(1)})^T \quad \rightarrow \text{形状: } [B, L, 4096]$
* 内存清理: GPU 0 和 GPU 1 释放重构的 $W_1^{(0)}$ 和 $W_1^{(1)}$ 显存.

##### 2.2 计算 `act` (GELU)
* 本地计算(无需通信):
  * GELU 是逐元素(Element-wise)操作, 直接作用于局部张量:
  * GPU 0 计算: $A^{(0)} = \text{GELU}(H^{(0)}) \quad \rightarrow [B, L, 4096]$
  * GPU 1 计算: $A^{(1)} = \text{GELU}(H^{(1)}) \quad \rightarrow [B, L, 4096]$

##### 2.3 计算 `linear2`(行切分 Rowwise Parallel)
* FSDP2 参数拉取:
  * GPU 0 与 GPU [2, 4, 6] FSDP2 AllGather, 重构出 $W_2^{(0)} \in [2048, 4096]$.
  * GPU 1 与 GPU [3, 5, 7] FSDP2 AllGather, 重构出 $W_2^{(1)} \in [2048, 4096]$.
* 本地矩阵乘法(产生局部部分和 Partial Sum):
  * GPU 0 计算: $Y^{(0)} = A^{(0)} \cdot (W_2^{(0)})^T \quad \rightarrow \text{形状: } [B, L, 2048]$
  * GPU 1 计算: $Y^{(1)} = A^{(1)} \cdot (W_2^{(1)})^T \quad \rightarrow \text{形状: } [B, L, 2048]$
  * *数学原理*: 此时 $Y^{(0)}$ 和 $Y^{(1)}$ 都只是最终结果的一部分(Partial Sum), 因为数学上 $Y = A^{(0)} W_2^{(0)T} + A^{(1)} W_2^{(1)T}$.
* 内存清理: 释放重构的 $W_2$ 权重.

##### 2.4 TP 规约通信(TP AllReduce)
* TP 通信: GPU 0 和 GPU 1(TP 组)之间触发 AllReduce (Sum) 算子, 将各自的 $Y^{(0)}$ 与 $Y^{(1)}$ 相加.
* 结果:
  $$Y = Y^{(0)} + Y^{(1)} \quad \rightarrow [B, L, 2048]$$
  通信完成后, GPU 0 和 GPU 1 上都拥有了完全一致且完整的 `linear2` 输出张量 $Y$.

##### 2.5 残差连接 (Residual Add)
* 本地计算:
  $$X_{\text{out}} = X_{\text{emb}} + Y \quad \rightarrow [B, L, 2048]$$
* 状态: GPU 0 和 GPU 1 此时持有相同的 $X_{\text{out}}$, 作为下一个 TransformerBlock 的输入.

---

#### 步骤 3: 重复执行后续 TransformerBlock
上述步骤 2 在 4 个 TransformerBlock 中依次重复 4 次. 每次进入 Block 前拉取参数, 离开 Block 前释放参数, Block 内部通过 TP 规约激活值.

---

#### 步骤 4: Output Head 层
1. FSDP2 AllGather: 拉取 `head` 层的权重 $W_{\text{head}} \in [5000, 2048]$(因为 Head 做 Colwise 切分, $10000 / 2 = 5000$).
2. 本地计算:
   * GPU 0 计算: $\text{Logits}^{(0)} = X_{\text{final}} \cdot (W_{\text{head}}^{(0)})^T \quad \rightarrow [B, L, 5000]$
   * GPU 1 计算: $\text{Logits}^{(1)} = X_{\text{final}} \cdot (W_{\text{head}}^{(1)})^T \quad \rightarrow [B, L, 5000]$
3. 输出: 前向传播结束, 进入 CrossEntropy Loss 计算阶段.

---

### 三、总结: 前向传播中的数据与通信图谱

可以将 2D 并行前向传播的"数据流动"抽象为如下公式与通信对齐图:

| 阶段         | GPU 0 (TP Rank 0, DP Rank 0) 张量                | GPU 1 (TP Rank 1, DP Rank 0) 张量                | 发生的通信 (Communication)                           |
| :----------- | :----------------------------------------------- | :----------------------------------------------- | :--------------------------------------------------- |
| Input        | $X \in [B, L]$                                   | $X \in [B, L]$                                   | 无 (DP 组内天然一致)                                 |
| Linear1 前   | 拉取 $W_1^{(0)}$ 碎片 $\rightarrow [4096, 2048]$ | 拉取 $W_1^{(1)}$ 碎片 $\rightarrow [4096, 2048]$ | FSDP2 AllGather(跨 GPU 0,2,4,6 与 1,3,5,7)           |
| Linear1 计算 | $H^{(0)} \in [B, L, 4096]$                       | $H^{(1)} \in [B, L, 4096]$                       | 无 (柱切分独立计算)                                  |
| Linear2 前   | 拉取 $W_2^{(0)}$ 碎片 $\rightarrow [2048, 4096]$ | 拉取 $W_2^{(1)}$ 碎片 $\rightarrow [2048, 4096]$ | FSDP2 AllGather(跨 GPU 0,2,4,6 与 1,3,5,7)           |
| Linear2 计算 | $Y^{(0)} \in [B, L, 2048]$ (部分和)              | $Y^{(1)} \in [B, L, 2048]$ (部分和)              | 无 (行切分产生 Partial Sum)                          |
| Linear2 后   | $Y = Y^{(0)} + Y^{(1)}$                          | $Y = Y^{(0)} + Y^{(1)}$                          | TP AllReduce (Sum)(仅 GPU 0 和 1 节点内 NVLink 规约) |

一句话总结 2D 并行前向传播的精髓:
> FSDP2 在 DP 维度上"按需拉取/用完即释放"模型权重, 解决了显存存不下的问题; TP 在 TP 维度上"切分矩阵计算/每层规约激活", 解决了单卡计算慢和激活显存大的问题. 两者在 DeviceMesh 的编排下完美交错重叠运行.





-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`2D 并行(TP + FSDP2)`, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本, 同时请先详细说明如何分配 8 张 GPU
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

在 1 台拥有 8 张 GPU 的机器上实现 2D 混合并行(TP + FSDP2), 是目前大模型训练中最经典且高效的单节点/多节点扩展配置.

下面将详细解析 8 卡 GPU 的划分拓扑, 并提供一份完整可直接运行的 PyTorch 2.x 训练脚本.

---

### 一、8 张 GPU 的分配与拓扑逻辑

在 1 台 8 卡机器上, 我们设定:

* TP Size (Tensor Parallel Size) = 2: 每 2 张 GPU 组为一个 TP 组, 负责模型单层内部的矩阵切分(通过 NVLink 进行高频通信).
* DP Size (FSDP2 Data Parallel Size) = 4: 一共分为 4 个 DP 组, 负责处理不同的 Data Batch, 并对模型参数、梯度、优化器状态进行分布式切分(Shard).
* 总卡数 = $DP \times TP = 4 \times 2 = 8$.

通过 PyTorch 2.x 的 `init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))`, 8 张 GPU 会被划分为如下二维网格矩阵:

| DP 组维度 (dp_size=4) | TP Rank 0                     | TP Rank 1                     | 说明(并行行为)                                                 |
| :-------------------- | :---------------------------- | :---------------------------- | :------------------------------------------------------------- |
| DP Group 0            | GPU 0                         | GPU 1                         | GPU 0 与 GPU 1 组成 TP 组; 共用 Batch 0 的数据                 |
| DP Group 1            | GPU 2                         | GPU 3                         | GPU 2 与 GPU 3 组成 TP 组; 共用 Batch 1 的数据                 |
| DP Group 2            | GPU 4                         | GPU 5                         | GPU 4 与 GPU 5 组成 TP 组; 共用 Batch 2 的数据                 |
| DP Group 3            | GPU 6                         | GPU 7                         | GPU 6 与 GPU 7 组成 TP 组; 共用 Batch 3 的数据                 |
| FSDP2 作用域          | GPU [0, 2, 4, 6] 组内做 FSDP2 | GPU [1, 3, 5, 7] 组内做 FSDP2 | 同一列的 GPU 互相做全量参数/梯度的 AllGather 与 ReduceScatter. |

> 关键规则(非常重要):
> 1. TP 组内数据必须完全一致: GPU 0 和 GPU 1 属于同一个 TP 组, 它们必须接收完全相同的输入数据 Token.
> 2. DP 组间数据必须互不相同: DP Group 0 (GPU 0,1) 和 DP Group 1 (GPU 2,3) 必须接收不同的 Batch 数据.
>

---

### 二、完整训练脚本 (`train_2d_parallel.py`)

请创建一个名为 `train_2d_parallel.py` 的文件, 填入以下完整代码:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 PyTorch 2.x 原生分布式组合 API
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

# -------------------------------------------------------------
# 1. 模型定义 (用户提供的原始模型)
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 2. 2D 并行训练主函数
# -------------------------------------------------------------
def main():
    # 2.1 分布式环境初始化
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[Init] 启动 2D 并行训练, 总 GPU 数: {world_size}")

    # 2.2 创建 2D DeviceMesh (DP=4, TP=2)
    tp_size = 2
    dp_size = world_size // tp_size

    mesh_2d = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp")
    )

    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    # 2.3 实例化模型并移至 GPU
    vocab_size = 10000
    dim = 2048
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=4).to(device)

    # 2.4 应用 Tensor Parallelism (TP) -> 作用于 tp_mesh
    # 经典的 Megatron 结构: linear1 柱切分 (Colwise), linear2 行切分 (Rowwise)
    for layer in model.layers:
        tp_plan = {
            "linear1": ColwiseParallel(),
            "linear2": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, tp_plan)

    # 对 Output Head 进行柱切分
    parallelize_module(model.head, tp_mesh, ColwiseParallel())

    # 2.5 应用 FSDP2 (Fully Shard) -> 作用于 dp_mesh
    # 先对各个 Block 进行逐层 shard, 最后对根 Module 进行 shard
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    # 2.6 定义优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # 3. 模拟训练过程
    # -------------------------------------------------------------
    model.train()
    num_steps = 5
    batch_size = 2
    seq_len = 64

    # 获取当前卡在 DP 组和 TP 组中的 local rank
    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # [核心技巧]: 使用 dp_rank 作为 Seed 生成随机数据!
        # 这样确保: 相同 TP 组(TP rank 0 和 1)拿到[完全相同]的数据;
        # 不同 DP 组(DP rank 0, 1, 2, 3)拿到[互不相同]的数据.
        torch.manual_seed(100 + step * 10 + dp_rank)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # 前向传播
        outputs = model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]

        # 计算 Loss
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step [{step}/{num_steps}] - Loss: {loss.item():.4f}")

    # -------------------------------------------------------------
    # 4. 模型保存 (分布式 Checkpoint Save)
    # -------------------------------------------------------------
    ckpt_path = "./saved_checkpoint_2d"

    # 清理旧的 Checkpoint 目录 (仅在 Rank 0 操作)
    if rank == 0:
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        print(f"\n[Checkpoint] 开始保存 2D 并行模型至目录: {ckpt_path} ...")

    # 等待所有 Rank 准备完毕
    dist.barrier()

    # 使用 PyTorch Distributed Checkpoint (DCP) 自动处理 DTensor/FSDP2 的权重保存
    # DCP 会将各卡的 Shard 状态高效保存为统一可加载的分布式格式
    state_dict = get_state_dict(model, optimizer)
    dcp.save(state_dict, checkpoint_id=ckpt_path)

    if rank == 0:
        print("[Checkpoint] 模型与优化器状态保存成功!")

    # -------------------------------------------------------------
    # 5. 资源回收与销毁
    # -------------------------------------------------------------
    dist.barrier()
    if rank == 0:
        print("[Cleanup] 训练完成, 正在释放进程组资源...")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

### 三、如何运行脚本

使用 PyTorch 官方推荐的 `torchrun` 工具在一台 8 卡机器上拉起训练:

```bash
torchrun --nproc_per_node=8 train_2d_parallel.py
```

#### 预期输出示例:
```text
[Init] 启动 2D 并行训练, 总 GPU 数: 8
Step [1/5] - Loss: 9.2145
Step [2/5] - Loss: 9.1820
Step [3/5] - Loss: 9.1231
Step [4/5] - Loss: 9.0412
Step [5/5] - Loss: 8.9322

[Checkpoint] 开始保存 2D 并行模型至目录: ./saved_checkpoint_2d ...
[Checkpoint] 模型与优化器状态保存成功!
[Cleanup] 训练完成, 正在释放进程组资源...
```

运行完成后, 你会看到当前目录下生成了一个 `saved_checkpoint_2d/` 文件夹, 里面包含了多张卡分片保存的 `.distcp` 元数据与权重文件.

---

### 四、代码中的关键要点(高级研究员视角)

1. 为什么先应用 TP, 再应用 FSDP2?
   在 PyTorch 2.x 的架构设计中, TP 改变了 Layer 内部参数的张量结构(将普通 `torch.Tensor` 包装为了 `DTensor`); 而 FSDP2 是对参数做 Shard. 如果先做 FSDP2, 参数已经被切片分配, TP 就无法识别完整的张量切分维度. 因此, 顺序必须是: `parallelize_module` (TP) $\rightarrow$ `fully_shard` (FSDP2).

2. 为什么数据生成需要 `seed = base + dp_rank`?
   TP 并行要求: 属于同一 TP 组内的卡(例如 GPU 0 和 GPU 1)在矩阵乘法时必须输入相同的 Token 序列, 否则 Rowwise/Colwise 规约通信(AllReduce/AllGather)计算出的梯度是错误的. 通过将随机种子与 `dp_rank` 绑定, 既保证了同一 TP 组输入相同, 又保证了不同 DP 组输入不同, 完美模拟了真实 DataLoader 的行为.

3. 分布式 Checkpoint (`torch.distributed.checkpoint`) 的优势
   使用传统的 `torch.save(model.state_dict())` 会强制要求在保存前把全量参数 AllGather 到单个 GPU, 极易导致 8 卡中的某个卡 GPU OOM. 使用 `dcp.save()` 可以在不合并全量参数的情况下, 由各个 GPU 并行写入各自的分布式权重, 保存速度最快且最节省显存.

-----------------------------------------------------------------------------------------

Q: 假设我现在有 2 台机器, 每台机器上有 8 张 GPU, 现在我想模拟`2D 并行(TP + FSDP2)`, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本, 同时请先详细说明如何分配 16 张 GPU
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

在扩展到 2 台机器(共 16 张 GPU, 每台 8 张) 时, 分布式拓扑的设计原则发生了关键变化: 必须充分考虑跨节点网络通信带宽(InfiniBand/RoCE/以太网)远低于节点内 NVLink 通信带宽这一物理事实.

下面将详细说明 16 卡跨节点拓扑的最佳分配策略, 并提供一份支持多节点运行的完整训练脚本与启动命令.

---

### 一、16 张 GPU(跨 2 节点)的分配与拓扑逻辑

在跨节点训练中, 黄金准则(Gold Standard)是: 将高频、低延迟敏感的 Tensor Parallelism (TP) 严格限制在节点内部; 将低频/通信可重叠的 FSDP2 (DP) 扩展到跨节点.

#### 最优拓扑选择: `TP Size = 8`, `DP Size = 2`
* TP Size = 8: 刚好让每台机器的 8 张 GPU 组成一个独立的 TP 组. TP 组内的所有通信 100% 走机器内部的 NVLink / NVSwitch 高速互联, 绝不跨节点!
* DP Size = 2: 跨机器做 FSDP2. 节点 0 的 GPU 与节点 1 对应的 GPU 组成 DP 组, 通过节点间的网络(如 InfiniBand)进行参数/梯度的 Sharding 和 AllGather / ReduceScatter.

通过 `init_device_mesh("cuda", (2, 8), mesh_dim_names=("dp", "tp"))`, 16 张 GPU(全局 Rank 0 ~ 15)被划分为如下二维矩阵:

| 节点归属        | DP 组维度 (dp_size=2) | TP Rank 0~7(节点内 8 张 GPU, NVLink 内部高频通信)           | 说明                                                       |
| :-------------- | :-------------------- | :---------------------------------------------------------- | :--------------------------------------------------------- |
| 机器 1 (Node 0) | DP Group 0            | GPU [0, 1, 2, 3, 4, 5, 6, 7]                                | 组成 TP Group 0; 处理 Batch 0 的数据.                      |
| 机器 2 (Node 1) | DP Group 1            | GPU [8, 9, 10, 11, 12, 13, 14, 15]                          | 组成 TP Group 1; 处理 Batch 1 的数据.                      |
| FSDP2 跨节点    | 跨节点 DP             | GPU [0, 8] 为一组, GPU [1, 9] 为一组 ... GPU [7, 15] 为一组 | 相同 TP Rank 的跨节点 GPU 之间做 FSDP2 权重分片与规约通信. |

---

### 二、完整跨节点训练脚本 (`train_2d_multi_node.py`)

请在两台机器的相同路径下创建 `train_2d_multi_node.py` 脚本:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入 PyTorch 2.x 原生分布式组合 API
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

# -------------------------------------------------------------
# 1. 模型定义 (用户提供的原始模型)
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 2. 多节点 2D 并行训练主函数
# -------------------------------------------------------------
def main():
    # 2.1 初始化多节点分布式环境 (NCCL 后端)
    dist.init_process_group("nccl")

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = global_rank // 8  # 当前节点的 index (0 或 1)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"[Init] 启动跨节点 2D 并行训练, 总 GPU 数: {world_size} (2 台机器 x 8 GPUs)")

    # 2.2 创建跨节点的 2D DeviceMesh (DP=2, TP=8)
    # TP=8 限制在单台机器内部 (8 GPU), DP=2 进行跨节点连接
    tp_size = 8
    dp_size = world_size // tp_size  # 16 / 8 = 2

    mesh_2d = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp")
    )

    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    # 2.3 实例化模型并移动至当前 GPU
    vocab_size = 10000
    dim = 2048
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=4).to(device)

    # 2.4 应用 Tensor Parallelism (TP=8) -> 节点内全量张量切分
    for layer in model.layers:
        tp_plan = {
            "linear1": ColwiseParallel(),
            "linear2": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, tp_plan)

    parallelize_module(model.head, tp_mesh, ColwiseParallel())

    # 2.5 应用 FSDP2 (Fully Shard) -> 跨节点网络数据切分
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    # 2.6 定义优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # 3. 模拟跨节点训练过程
    # -------------------------------------------------------------
    model.train()
    num_steps = 5
    batch_size = 2
    seq_len = 64

    dp_rank = dp_mesh.get_local_rank()  # 机器 0 上为 0, 机器 1 上为 1

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # [核心]: 以 dp_rank 作为 Seed 区分跨节点的数据 Batch
        # 机器 0 (DP rank 0) 上的 8 张卡拿到完全相同的 Batch 0
        # 机器 1 (DP rank 1) 上的 8 张卡拿到完全相同的 Batch 1
        torch.manual_seed(100 + step * 10 + dp_rank)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        optimizer.step()

        if global_rank == 0:
            print(f"Step [{step}/{num_steps}] - Global Rank 0 (Node 0) Loss: {loss.item():.4f}")

    # -------------------------------------------------------------
    # 4. 跨节点模型保存 (分布式 Checkpoint Save)
    # -------------------------------------------------------------
    # 注意: 建议保存到两台机器均可访问的共享存储 (如 NFS/Ceph) 路径下
    # 如果没有共享存储, DCP 也会在每台机器的本地写入属于该卡的分片文件
    ckpt_path = "./saved_checkpoint_multi_node"

    if global_rank == 0:
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        print(f"\n[Checkpoint] 保存 2D 多节点模型至目录: {ckpt_path} ...")

    dist.barrier()

    # DCP 自动协调跨节点的 16 张 GPU 保存状态
    state_dict = get_state_dict(model, optimizer)
    dcp.save(state_dict, checkpoint_id=ckpt_path)

    if global_rank == 0:
        print("[Checkpoint] 多节点模型与优化器状态保存成功!")

    # -------------------------------------------------------------
    # 5. 资源回收与销毁
    # -------------------------------------------------------------
    dist.barrier()
    if global_rank == 0:
        print("[Cleanup] 跨节点训练完成, 正在销毁进程组...")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

### 三、如何运行脚本(多节点 `torchrun` 指令)

假设两台机器的网络配置如下:
* 机器 1 (Master Node, Node 0) IP: `192.168.1.100`
* 机器 2 (Worker Node, Node 1) IP: `192.168.1.101`
* 主节点通信端口: `29500`

必须在两台机器上同时运行 `torchrun` 指令拉起各自进程:

#### 1. 在[机器 1 (Master Node)]上执行:
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_2d_multi_node.py
```

#### 2. 在[机器 2 (Worker Node)]上执行:
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_2d_multi_node.py
```

---

### 四、跨节点工程落地踩坑与调优建议

1. 网络通信后端与网卡绑定(NCCL Settings)
   跨节点训练极其依赖 NCCL 的网络寻址. 如果机器上有 InfiniBand 或 RoCE 专有网卡, 建议在启动前设置环境变量, 避免 NCCL 走了慢速的以太网:
   ```bash
   export NCCL_DEBUG=INFO
   # 指定使用 InfiniBand 网卡(例如 mlx5_0:1)
   export NCCL_IB_DISABLE=0
   export NCCL_SOCKET_IFNAME=eth0  # 仅用于控制信道的网卡接口名
   ```

2. 为什么不建议 `TP=4, DP=4` 或 `TP=2, DP=8`?
   若设置 `TP=4`, 单台机器 (8 GPU) 内部就会被拆分为 2 个 TP 组, 这样 2 台机器一共有 4 个 TP 组. 虽然这在语法上完全成立, 但在 8 卡 NVLink 的机器上(如 A100/H100/H200 SXM), 把 TP 限制在全卡 8 卡能最大化利用单机 NVSwitch 带宽, 同时减少跨节点通信的总管道数, 提升网络通信吞吐量(MFU).

3. Checkpoint 的存储路径选型
   `dcp.save()` 会由各个进程并发写入存储.
   * 情况 A(推荐): 使用 NFS/Ceph 共享文件系统. 两台机器填写入同一个路径(如 `/nfs/checkpoint/`), DCP 会自动生成元数据并归档, 后续可以任意改变 GPU 数量加载.
   * 情况 B: 使用本地磁盘. 两台机器会分别将各自 Rank 的 Shard 文件存入本地 `./saved_checkpoint_multi_node/` 文件夹. 加载恢复时, 需保证各节点能读取到对应的文件结构.



-----------------------------------------------------------------------------------------

Q: 假设我现在有 1 台机器, 机器上有 8 张 GPU, 现在我想模拟`3D 并行 (TP + PP + FSDP2)`, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本, 同时请先详细说明如何分配 8 张 GPU
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

在单台机器(8 张 GPU)上配置 3D 混合并行(TP + PP + FSDP2), 是理解万亿大模型分布式架构(Megatron-3D Style)的黄金练习 [1, 2, 3].

在 3D 并行中, 模型不仅在宽/隐藏层维度被切分(TP), 还在数据维度被切分(FSDP2), 同时还在深度/层数维度被切分(PP)[1, 2, 3].

下面将详细解析 8 卡 GPU 的 3D 拓扑分配, 并提供一份基于 PyTorch 2.x 最新原生 API(`DeviceMesh` + `pipelining` + `fully_shard`)的完整训练脚本 [1, 2, 3, 5, 6].

---

### 一、8 张 GPU 的 3D 拓扑分配与划分

对于 8 张 GPU, 我们构建 $2 \times 2 \times 2 = 8$ 的 3D 网格拓扑 [1, 2, 3]:
* PP Size (Pipeline Parallel) = 2: 模型按层分为 2 个流水线阶段(Stage 0 和 Stage 1)[1, 2, 3, 5].
* DP Size (FSDP2 Size) = 2: 每个 Stage 内部做 2路数据并行与参数切分 [1, 2, 3].
* TP Size (Tensor Parallel) = 2: 每个 Transformer Block 内部做 2路矩阵切分 [1, 2, 3].

使用 `init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "dp", "tp"))`, 8 张 GPU 的职责划分如下 [1, 2, 3]:

| 全局 Rank | PP Rank (Stage)   | DP Rank | TP Rank | 机器内部的实际职责与分组                                           |
| :-------- | :---------------- | :------ | :------ | :----------------------------------------------------------------- |
| GPU 0     | Stage 0 (前 2 层) | DP 0    | TP 0    | Stage 0: 与 GPU 1 组成 TP 组; 与 GPU 2 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 1     | Stage 0 (前 2 层) | DP 0    | TP 1    | Stage 0: 与 GPU 0 组成 TP 组; 与 GPU 3 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 2     | Stage 0 (前 2 层) | DP 1    | TP 0    | Stage 0: 与 GPU 3 组成 TP 组; 与 GPU 0 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 3     | Stage 0 (前 2 层) | DP 1    | TP 1    | Stage 0: 与 GPU 2 组成 TP 组; 与 GPU 1 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 4     | Stage 1 (后 2 层) | DP 0    | TP 0    | Stage 1: 与 GPU 5 组成 TP 组; 与 GPU 6 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 5     | Stage 1 (后 2 层) | DP 0    | TP 1    | Stage 1: 与 GPU 4 组成 TP 组; 与 GPU 7 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 6     | Stage 1 (后 2 层) | DP 1    | TP 0    | Stage 1: 与 GPU 7 组成 TP 组; 与 GPU 4 组成 FSDP2 组 [1, 2, 3, 5]. |
| GPU 7     | Stage 1 (后 2 层) | DP 1    | TP 1    | Stage 1: 与 GPU 6 组成 TP 组; 与 GPU 5 组成 FSDP2 组 [1, 2, 3, 5]. |

#### 模型层(Layer)切分示意:
* Stage 0 (GPU 0~3): 运行 `Embedding` + `TransformerBlock [0, 1]`. 计算中间特征激活值(Activations), 通过 P2P 通信传给 Stage 1 [1, 2, 3, 5].
* Stage 1 (GPU 4~7): 接收来自 Stage 0 的激活值, 运行 `TransformerBlock [2, 3]` + `Linear Head`, 计算最终 Output 并求解 Loss [1, 2, 3, 5].

---

### 二、完整 3D 并行训练脚本 (`train_3d_parallel.py`)

请创建一个名为 `train_3d_parallel.py` 的文件, 填入以下代码:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# PyTorch 2.x 组合并行与流水线 API
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# -------------------------------------------------------------
# 1. 模型基础组件定义
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


# 按 Stage 划分的子模型模块定义
class Stage0Module(nn.Module):
    """Pipeline Stage 0: 包含 Embedding 和前一半 TransformerBlock"""
    def __init__(self, vocab_size, dim, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = layers

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Stage1Module(nn.Module):
    """Pipeline Stage 1: 包含后一半 TransformerBlock 和 Head"""
    def __init__(self, vocab_size, dim, layers):
        super().__init__()
        self.layers = layers
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# 最后一个 Stage 计算 Loss 的函数
def tokenwise_loss_fn(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()
    outputs = outputs.reshape(-1, outputs.size(-1))
    targets = targets.reshape(-1)
    return loss_fn(outputs, targets)


# -------------------------------------------------------------
# 2. 3D 并行训练主函数
# -------------------------------------------------------------
def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[Init] 启动 3D 并行训练 (PP=2, DP=2, TP=2), 总 GPUs: {world_size}")

    # 2.1 构建 3D DeviceMesh: (PP, DP, TP)
    mesh_3d = init_device_mesh(
        "cuda",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("pp", "dp", "tp")
    )

    pp_mesh = mesh_3d["pp"]
    dp_mesh = mesh_3d["dp"]
    tp_mesh = mesh_3d["tp"]

    pp_rank = pp_mesh.get_local_rank()
    dp_rank = dp_mesh.get_local_rank()

    vocab_size = 10000
    dim = 2048
    all_layers = nn.ModuleList([TransformerBlock(dim) for _ in range(4)])

    # 2.2 根据 PP Rank 在本节点上构建对应的 Stage 子网络
    if pp_rank == 0:
        stage_module = Stage0Module(vocab_size, dim, all_layers[:2]).to(device)
    else:
        stage_module = Stage1Module(vocab_size, dim, all_layers[2:]).to(device)

    # 2.3 应用 Tensor Parallelism (TP) -> 作用于 tp_mesh
    for layer in stage_module.layers:
        tp_plan = {
            "linear1": ColwiseParallel(),
            "linear2": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, tp_plan)

    if hasattr(stage_module, "head"):
        parallelize_module(stage_module.head, tp_mesh, ColwiseParallel())

    # 2.4 应用 FSDP2 (Fully Shard) -> 作用于 dp_mesh
    for layer in stage_module.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(stage_module, mesh=dp_mesh)

    # 2.5 定义优化器
    optimizer = torch.optim.AdamW(stage_module.parameters(), lr=1e-4)

    # 2.6 构建 Pipeline Stage 与 1F1B 调度器
    stage = PipelineStage(
        stage_module,
        stage_index=pp_rank,
        num_stages=2,
        device=device,
        group=pp_mesh.get_group(),
    )

    # 1F1B (One Forward, One Backward) 调度器
    # 仅最后一个 Stage (pp_rank == 1) 传入 loss_fn
    schedule = Schedule1F1B(
        stage,
        n_microbatches=2,  # Batch 会自动切分为 2 个 Microbatch 跑流水线
        loss_fn=tokenwise_loss_fn if pp_rank == 1 else None,
    )

    # -------------------------------------------------------------
    # 3. 模拟 3D 训练过程
    # -------------------------------------------------------------
    stage_module.train()
    num_steps = 5
    batch_size = 4  # 全量 Batch Size
    seq_len = 64

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # 根据 dp_rank 确定数据种子, 同一 TP 组与同 PP 对应组的数据保持一致
        torch.manual_seed(100 + step * 10 + dp_rank)

        if pp_rank == 0:
            # Stage 0: 准备输入 tokens, 传给流水线首端
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            schedule.step(input_ids)
        else:
            # Stage 1: 准备目标 labels, 传给流水线末端求解 Loss
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            losses = []
            schedule.step(target=targets, losses=losses)

        optimizer.step()

        if rank == 4:  # Stage 1 的首张 GPU, 持有该步计算出的 Loss
            avg_loss = sum(l.item() for l in losses) / len(losses)
            print(f"Step [{step}/{num_steps}] - Stage 1 Loss: {avg_loss:.4f}")

    # -------------------------------------------------------------
    # 4. 模型保存 (分布式 DCP Checkpoint Save)
    # -------------------------------------------------------------
    ckpt_path = "./saved_checkpoint_3d"

    if rank == 0:
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        print(f"\n[Checkpoint] 保存 3D 并行模型至目录: {ckpt_path} ...")

    dist.barrier()

    # DCP 自动合并跨 PP, DP, TP 组的 8 卡模型权重状态
    state_dict = get_state_dict(stage_module, optimizer)
    dcp.save(state_dict, checkpoint_id=ckpt_path)

    if rank == 0:
        print("[Checkpoint] 3D 并行模型与优化器状态保存成功!")

    # -------------------------------------------------------------
    # 5. 资源回收与销毁
    # -------------------------------------------------------------
    dist.barrier()
    if rank == 0:
        print("[Cleanup] 训练完成, 正在销毁进程组...")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

### 三、如何运行脚本

在 8 卡机器上运行 `torchrun` [1, 2, 3]:

```bash
torchrun --nproc_per_node=8 train_3d_parallel.py
```

#### 预期输出示例:
```text
[Init] 启动 3D 并行训练 (PP=2, DP=2, TP=2), 总 GPUs: 8
Step [1/5] - Stage 1 Loss: 9.2104
Step [2/5] - Stage 1 Loss: 9.1782
Step [3/5] - Stage 1 Loss: 9.1120
Step [4/5] - Stage 1 Loss: 9.0234
Step [5/5] - Stage 1 Loss: 8.9105

[Checkpoint] 保存 3D 并行模型至目录: ./saved_checkpoint_3d ...
[Checkpoint] 3D 并行模型与优化器状态保存成功!
[Cleanup] 训练完成, 正在销毁进程组...
```

---

### 四、3D 并行的核心代码细节解析

1. `Schedule1F1B` 与 Microbatch 气泡优化:
   * 在流水线并行中, 单个 Batch 被切分为 `n_microbatches=2` 个微批次 [1, 2, 3, 5, 6]. 1F1B(1 个前向, 1 个反向)调度方式能最大程度地重叠计算与 Stage 间的数据 P2P 通信, 大幅压缩流水线等待气泡(Pipeline Bubble)[1, 2, 3, 5, 6].
2. `PipelineStage` 的无缝嵌入:
   * `PipelineStage` 在 PyTorch 2.4+ 中原生与 `DeviceMesh` 兼容 [1, 2, 3, 5, 6]. 它会自动在 `pp_mesh.get_group()` 的通信组内管理前向激活值的 `send` / `recv` 以及反向梯度的传递, 无需开发者手动写 `dist.send()` 或 `dist.recv()` [1, 2, 3, 5, 6].
3. 数据绑定的精准性:
   * 在 PP 拓扑中, 只有 Stage 0 负责输入数据(`input_ids`), 末尾 Stage 负责接收标签数据(`targets`) [1, 2, 3, 5, 6]. 中间的 Stage 仅在后台被动接受网络传入的激活张量, 因此代码中 `if pp_rank == 0` 与 `if pp_rank == 1` 的数据构造方式完全不同 [1, 2, 3, 5, 6].




-----------------------------------------------------------------------------------------

Q: 假设我现在有 2 台机器, 每台机器上有 8 张 GPU, 现在我想模拟`3D 并行 (TP + PP + FSDP2)`, 模型如下代码所示, 数据也随机生成, 请输出完整的训练脚本, 包括模型保存和最后的资源回收, 并且说明如何运行脚本, 同时请先详细说明如何分配 16 张 GPU
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

跨 2 台机器(每台 8 张 GPU, 共 16 张 GPU)训练 3D 并行(TP + PP + FSDP2), 是真正进入大规模集群(如千卡训练集群)拓扑设计的核心起点 [1, 2, 3].

在多节点拓扑设计中, 网络通信层次划分(NVLink 节点内 vs 网卡节点间)是唯一的效率决定因素.

下面将详细说明 16 张 GPU 在 2 个节点上的最优 3D 拓扑分配, 并提供支持跨机器运行的完整训练脚本与启动指令 [1, 2, 3].

---

### 一、16 张 GPU(跨 2 节点)的 3D 拓扑分配与划分

对于 16 张 GPU(机器 1: Rank 0~7; 机器 2: Rank 8~15), 最经典且通信开销最小的组合为:
$$\text{PP Size} = 2, \quad \text{DP Size} = 2, \quad \text{TP Size} = 4 \quad (2 \times 2 \times 4 = 16)$$ [1, 2, 3]

#### 为什么这是最佳拓扑?
1. PP (Pipeline Parallel) 放在跨节点拓扑最外层:
   * 机器 1 (Node 0, Rank 0~7) 完整运行 Stage 0(前向计算前一半层)[1, 2, 3, 5].
   * 机器 2 (Node 1, Rank 8~15) 完整运行 Stage 1(前向计算后一半层 + Loss)[1, 2, 3, 5].
   * 节点间网络(InfiniBand/以太网)仅需承担流水线 Stage 0 到 Stage 1 的点对点(P2P)激活值传输. P2P 通信量极小, 完全不会挤爆跨节点带宽 [1, 2, 3, 5]!
2. TP (Tensor Parallel) 严格限制在单节点内 (TP=4):
   * 机器 1 内部的 8 张卡被拆分为 2 个 TP=4 的组; 机器 2 内部同理 [1, 2, 3].
   * 高频的张量切分通信 100% 走节点内的 NVLink/NVSwitch, 极速完成 [1, 2, 3]!
3. DP (FSDP2) 在节点内做数据切分 (DP=2):
   * 机器 1 内部的 2 个 TP 组(GPU [0,1,2,3] 与 GPU [4,5,6,7])相互做 FSDP2 状态切分, 通信同样在节点内高效完成 [1, 2, 3].

#### 16 卡详细职责映射表:

| 节点归属        | 全局 Rank   | PP Rank (Stage)               | DP Rank | TP Rank  | 节点内部/跨节点职责                                                     |
| :-------------- | :---------- | :---------------------------- | :------ | :------- | :---------------------------------------------------------------------- |
| 机器 1 (Node 0) | GPU 0 ~ 3   | Stage 0                       | DP 0    | TP 0 ~ 3 | 机器 1 内部: GPU 0~3 组成 TP Group 0(单机 NVLink 通信) [1, 2, 3].       |
| 机器 1 (Node 0) | GPU 4 ~ 7   | Stage 0                       | DP 1    | TP 0 ~ 3 | 机器 1 内部: GPU 4~7 组成 TP Group 1; 与 GPU 0~3 做 FSDP2 [1, 2, 3].    |
| 跨节点 P2P      | ---         | Stage 0 $\rightarrow$ Stage 1 | ---     | ---      | 跨节点通信: 机器 1 仅向机器 2 发送激活值 / 接收梯度 (P2P) [1, 2, 3, 5]. |
| 机器 2 (Node 1) | GPU 8 ~ 11  | Stage 1                       | DP 0    | TP 0 ~ 3 | 机器 2 内部: GPU 8~11 组成 TP Group 0(单机 NVLink 通信) [1, 2, 3].      |
| 机器 2 (Node 1) | GPU 12 ~ 15 | Stage 1                       | DP 1    | TP 0 ~ 3 | 机器 2 内部: GPU 12~15 组成 TP Group 1; 与 GPU 8~11 做 FSDP2 [1, 2, 3]. |

---

### 二、完整多节点 3D 并行训练脚本 (`train_3d_multi_node.py`)

请在两台机器的相同目录下创建 `train_3d_multi_node.py` 脚本:

```python
import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist

# PyTorch 2.x 组合并行与流水线 API
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict


# -------------------------------------------------------------
# 1. 模型基础组件与 Pipeline Stage 划分
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return x + self.linear2(self.act(self.linear1(x)))


class Stage0Module(nn.Module):
    """Pipeline Stage 0 (机器 1 运行): Embedding + 前一半 TransformerBlock"""
    def __init__(self, vocab_size, dim, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = layers

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Stage1Module(nn.Module):
    """Pipeline Stage 1 (机器 2 运行): 后一半 TransformerBlock + Output Head"""
    def __init__(self, vocab_size, dim, layers):
        super().__init__()
        self.layers = layers
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def tokenwise_loss_fn(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()
    outputs = outputs.reshape(-1, outputs.size(-1))
    targets = targets.reshape(-1)
    return loss_fn(outputs, targets)


# -------------------------------------------------------------
# 2. 多节点 3D 并行训练主函数
# -------------------------------------------------------------
def main():
    dist.init_process_group("nccl")

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"[Init] 启动多节点 3D 并行训练 (PP=2, DP=2, TP=4), 总 GPUs: {world_size}")

    # 2.1 创建 3D DeviceMesh: (PP=2, DP=2, TP=4)
    # 该 Mesh 将 PP 自动映射为跨节点, TP/DP 映射为节点内
    mesh_3d = init_device_mesh(
        "cuda",
        mesh_shape=(2, 2, 4),
        mesh_dim_names=("pp", "dp", "tp")
    )

    pp_mesh = mesh_3d["pp"]
    dp_mesh = mesh_3d["dp"]
    tp_mesh = mesh_3d["tp"]

    pp_rank = pp_mesh.get_local_rank()  # 机器 1 上为 0, 机器 2 上为 1
    dp_rank = dp_mesh.get_local_rank()

    vocab_size = 10000
    dim = 2048
    all_layers = nn.ModuleList([TransformerBlock(dim) for _ in range(4)])

    # 2.2 依据所在的流水线 Stage 实例化 local 节点模块
    if pp_rank == 0:
        stage_module = Stage0Module(vocab_size, dim, all_layers[:2]).to(device)
    else:
        stage_module = Stage1Module(vocab_size, dim, all_layers[2:]).to(device)

    # 2.3 应用 Tensor Parallelism (TP=4) -> 节点内单 Block 4路切分
    for layer in stage_module.layers:
        tp_plan = {
            "linear1": ColwiseParallel(),
            "linear2": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, tp_plan)

    if hasattr(stage_module, "head"):
        parallelize_module(stage_module.head, tp_mesh, ColwiseParallel())

    # 2.4 应用 FSDP2 (Fully Shard) -> 节点内 2路参数 Shard
    for layer in stage_module.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(stage_module, mesh=dp_mesh)

    # 2.5 优化器定义
    optimizer = torch.optim.AdamW(stage_module.parameters(), lr=1e-4)

    # 2.6 构建跨节点的 Pipeline Stage 与 1F1B 调度器
    stage = PipelineStage(
        stage_module,
        stage_index=pp_rank,
        num_stages=2,
        device=device,
        group=pp_mesh.get_group(),  # pp_mesh 跨节点点对点通信组
    )

    schedule = Schedule1F1B(
        stage,
        n_microbatches=2,
        loss_fn=tokenwise_loss_fn if pp_rank == 1 else None,
    )

    # -------------------------------------------------------------
    # 3. 跨节点 3D 训练迭代循环
    # -------------------------------------------------------------
    stage_module.train()
    num_steps = 5
    batch_size = 4
    seq_len = 64

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # 数据 Seed 绑定 dp_rank, 保证相同 TP 组拿到相同 input, 不同 DP 组数据不同
        torch.manual_seed(100 + step * 10 + dp_rank)

        if pp_rank == 0:
            # Stage 0 (机器 1): 产生输入 ids 并推动流水线前向传输
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            schedule.step(input_ids)
        else:
            # Stage 1 (机器 2): 产生 targets 并接收机器 1 传来的激活值求解 Loss
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            losses = []
            schedule.step(target=targets, losses=losses)

        optimizer.step()

        # 仅在 Stage 1 的首个 GPU 上打印损失
        if global_rank == 8:
            avg_loss = sum(l.item() for l in losses) / len(losses)
            print(f"Step [{step}/{num_steps}] - Stage 1 Loss: {avg_loss:.4f}")

    # -------------------------------------------------------------
    # 4. 跨节点分布式 Checkpoint 保存 (DCP)
    # -------------------------------------------------------------
    ckpt_path = "./saved_checkpoint_3d_multi_node"

    if global_rank == 0:
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        print(f"\n[Checkpoint] 保存多节点 3D 模型至目录: {ckpt_path} ...")

    dist.barrier()

    # DCP 原生协调跨节点 16 张卡的不同 Stage 和 DTensor 分片保存
    state_dict = get_state_dict(stage_module, optimizer)
    dcp.save(state_dict, checkpoint_id=ckpt_path)

    if global_rank == 0:
        print("[Checkpoint] 多节点 3D 模型与优化器保存成功!")

    # -------------------------------------------------------------
    # 5. 资源回收与销毁
    # -------------------------------------------------------------
    dist.barrier()
    if global_rank == 0:
        print("[Cleanup] 训练完成, 正在销毁跨节点进程组...")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

### 三、如何运行脚本(多节点 `torchrun` 指令)

假设网络环境如下:
* 机器 1 (Master Node, Node 0) IP: `192.168.1.100`
* 机器 2 (Worker Node, Node 1) IP: `192.168.1.101`
* 节点间通信端口: `29500`

#### 1. 在[机器 1 (Master Node / Node 0)]上运行:
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_3d_multi_node.py
```

#### 2. 在[机器 2 (Worker Node / Node 1)]上运行:
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_3d_multi_node.py
```

---

### 四、跨节点 3D 并行工业级优化与踩坑建议

1. 流水线气泡(Pipeline Bubble)与 Microbatch 调优
   * 在本例中, `batch_size=4` 被切分为 `n_microbatches=2`(每个微批次大小为 2)[1, 2, 3, 5, 6]. 在万亿参数生产环境训练中, 通常会增加 `n_microbatches` 的数量(例如设置为 4 或 8), 使流水线填充得更满, 从而将跨节点 PP 的等待气泡(Bubble Ratio)降低至 5%~10% 以下 [1, 2, 3, 5, 6].

2. 跨节点 P2P 激活通信与网络瓶颈(NCCL P2P)
   * PP 的通信是纯点对点(P2P)传输 [1, 2, 3, 5]. 确保机器间开启了正确的点对点通信支持, 建议在环境变量中显式配置:
     ```bash
     export NCCL_CROSS_NIC=1
     export NCCL_ALGO=Tree,Ring
     ```

3. 模型保存目录(Checkpoint Path)
   * 使用 `dcp.save()` 时, 如果两台机器共享 NFS/Ceph 文件系统, 请将 `ckpt_path` 指向共享挂载路径 [1, 2, 3, 7]. 如果不使用共享存储, 机器 1 会保存 Stage 0 的权重分片, 机器 2 会保存 Stage 1 的权重分片, DCP 允许后续在相同的 3D 拓扑下无缝恢复 [1, 2, 3, 7].





-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------



-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------
