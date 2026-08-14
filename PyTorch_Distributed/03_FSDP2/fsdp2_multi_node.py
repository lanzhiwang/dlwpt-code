import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ==============================================================================
# 1. 手动实现 FSDP2 风格的分片模块 (支持 16 卡切分)
# ==============================================================================


class FSDP2ShardedEmbedding(nn.Module):
    """
    手动实现 FSDP2 分片 Embedding 层:
    - 静态状态: 每张卡仅保存 vocab_size / 16 行权重 (常驻显存仅 1/16)
    - 前向计算: all_gather 收集全量 16 卡词表 -> 查表 -> 立即释放全量词表
    """

    def __init__(self, vocab_size: int, dim: int, world_size: int, rank: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        self.rank = rank

        assert (
            vocab_size % world_size == 0
        ), f"vocab_size ({vocab_size}) 必须能被 world_size ({world_size}) 整除"
        self.sharded_vocab_size = vocab_size // world_size

        # 静态存储: 每张卡仅分配 1/16 的显存空间
        # shape: [vocab_size / 16, dim] -> [10240 / 16, 2048] = [640, 2048]
        self.sharded_weight = nn.Parameter(
            torch.randn(self.sharded_vocab_size, dim, dtype=torch.float32) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [local_batch_size, seq_len] -> [2, 16]

        # 步骤 1: 在 GPU 上开辟接收 16 卡完整权重的临时 Buffer
        # full_weight shape: [vocab_size, dim] -> [10240, 2048]
        full_weight = torch.empty(
            self.vocab_size, self.dim, device=x.device, dtype=self.sharded_weight.dtype
        )

        # 步骤 2: 底层 NCCL 原语通信 (16 卡 All-Gather 跨机拼接)
        # 收集 16 个 [640, 2048] 的分片 -> 拼成 [10240, 2048]
        dist.all_gather_into_tensor(full_weight, self.sharded_weight.contiguous())

        # 步骤 3: 本地查表前向计算
        # 输入 [2, 16] + 完整词表 [10240, 2048] -> 输出 [2, 16, 2048]
        out = F.embedding(x, full_weight)

        # 步骤 4: FSDP2 核心机制 —— 销毁全量权重临时张量, 释放显存
        del full_weight

        # 返回张量 shape: [local_batch_size, seq_len, dim] -> [2, 16, 2048]
        return out


class FSDP2ShardedLinear(nn.Module):
    """
    手动实现 FSDP2 分片 Linear 层:
    - 静态状态: 每张卡只存储 out_features / 16 行权重和偏置
    - 前向计算: all_gather 收集全量权重 -> 矩阵乘法 -> 立即释放全量权重
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        assert (
            out_features % world_size == 0
        ), f"out_features ({out_features}) 必须能被 world_size ({world_size}) 整除"
        self.sharded_out_features = out_features // world_size

        # 静态分片权重: [out_features / 16, in_features]
        self.sharded_weight = nn.Parameter(
            torch.randn(self.sharded_out_features, in_features, dtype=torch.float32)
            * 0.02
        )
        if bias:
            # 静态分片偏置: [out_features / 16]
            self.sharded_bias = nn.Parameter(
                torch.zeros(self.sharded_out_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("sharded_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [local_batch_size, seq_len, in_features]

        # 步骤 1: 准备全量权重 Buffer
        # full_weight shape: [out_features, in_features]
        full_weight = torch.empty(
            self.out_features,
            self.in_features,
            device=x.device,
            dtype=self.sharded_weight.dtype,
        )

        # 步骤 2: 底层跨机通信拼接 16 张卡的局部参数
        dist.all_gather_into_tensor(full_weight, self.sharded_weight.contiguous())

        # 步骤 3: 处理偏置 (Bias) All-Gather
        full_bias = None
        if self.sharded_bias is not None:
            # full_bias shape: [out_features]
            full_bias = torch.empty(
                self.out_features, device=x.device, dtype=self.sharded_bias.dtype
            )
            dist.all_gather_into_tensor(full_bias, self.sharded_bias.contiguous())

        # 步骤 4: 执行前向矩阵乘法: x @ full_weight.T + full_bias
        # [local_batch_size, seq_len, in_features] x [out_features, in_features].T
        # -> shape: [local_batch_size, seq_len, out_features]
        out = F.linear(x, full_weight, full_bias)

        # 步骤 5: 立即释放通信产生的全量参数显存
        del full_weight
        del full_bias

        # 返回张量 shape: [local_batch_size, seq_len, out_features]
        return out


# ==============================================================================
# 2. 组装 TransformerBlock 与 SimpleLLM 模型
# ==============================================================================


class FSDP2TransformerBlock(nn.Module):
    def __init__(self, dim: int, world_size: int, rank: int):
        super().__init__()
        # linear1: [dim -> 4 * dim] (2048 -> 8192)
        # 每张卡分片后存储: [8192 / 16, 2048] = [512, 2048]
        self.linear1 = FSDP2ShardedLinear(dim, dim * 4, world_size, rank)
        self.act = nn.GELU()

        # linear2: [4 * dim -> dim] (8192 -> 2048)
        # 每张卡分片后存储: [2048 / 16, 8192] = [128, 8192]
        self.linear2 = FSDP2ShardedLinear(dim * 4, dim, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [2, 16, 2048]
        residual = x

        # 1. 经 linear1 前向 (触发 16 卡 all_gather)
        # 输出 h1 shape: [2, 16, 8192]
        h1 = self.linear1(x)

        # 2. GELU 激活 (本地逐元素计算, 无跨机通信)
        # 输出 h2 shape: [2, 16, 8192]
        h2 = self.act(h1)

        # 3. 经 linear2 前向 (触发 16 卡 all_gather)
        # 输出 h3 shape: [2, 16, 2048]
        h3 = self.linear2(h2)

        # 4. 残差相加
        # 输出 shape: [2, 16, 2048]
        return residual + h3


class FSDP2SimpleLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10240,
        dim: int = 2048,
        num_layers: int = 4,
        world_size: int = 16,
        rank: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

        # 1. 分片 Embedding: [10240 / 16, 2048] = [640, 2048]
        self.embed = FSDP2ShardedEmbedding(vocab_size, dim, world_size, rank)

        # 2. 堆叠 num_layers 个分片 TransformerBlock
        self.layers = nn.ModuleList(
            [FSDP2TransformerBlock(dim, world_size, rank) for _ in range(num_layers)]
        )

        # 3. 分片 LM Head: [10240 / 16, 2048] = [640, 2048]
        self.head = FSDP2ShardedLinear(dim, vocab_size, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 Token IDs shape: [2, 16]

        # [阶段 1] Embedding 聚合 -> 查表 -> 释放
        # 输出 x shape: [2, 16, 2048]
        x = self.embed(x)

        # [阶段 2] 逐层执行 TransformerBlock
        # 每一层内部依次触发 linear1 与 linear2 的通信聚合与释放
        for layer in self.layers:
            x = layer(x)
            # x 保持 shape: [2, 16, 2048]

        # [阶段 3] Head 聚合 -> 投影回词表 -> 释放
        # 输出 logits shape: [2, 16, 10240]
        logits = self.head(x)
        return logits


# ==============================================================================
# 3. 简单的纯文本分词器 (模拟纯文本转张量矩阵)
# ==============================================================================


class SimpleCharTokenizer:
    """负责将原始纯文本 (String) 转为固定形状的 Token ID 矩阵"""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, texts: list[str], max_len: int) -> torch.Tensor:
        batch_ids = []
        for text in texts:
            # 字符转为 ASCII 整数编码
            ids = [ord(c) % self.vocab_size for c in text]
            # Padding 补齐或截断到 max_len
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            batch_ids.append(ids)
        # 返回 LongTensor, shape: [batch_size, max_len]
        return torch.tensor(batch_ids, dtype=torch.long)


# ==============================================================================
# 4. 多机分布式主执行逻辑
# ==============================================================================


def main():
    # --- 1. 读取多机环境变量并初始化通信组 ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 绑定当前进程到本地对应的物理显卡
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化 16 卡 NCCL 通信组 (跨机通过 TCP 握手, 随后建立跨机网络拓扑)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    node_id = rank // 8  # 0~7 属于 Node 0, 8~15 属于 Node 1
    if rank == 0:
        print("=" * 80)
        print(
            f"[多机通信组初始化完成] 总机器数: 2 | 每台 GPU: 8 | 总 World Size: {world_size}"
        )
        print("=" * 80)

    # --- 2. 纯文本转全局矩阵与 16 卡切分 ---
    # 全局生成 32 条文本句子
    raw_texts = [
        f"Prompt {i:02d}: Testing multi-node FSDP2 low-level simulation!"
        for i in range(32)
    ]
    seq_len = 16
    vocab_size = 10240  # 10240 / 16 = 640 (整除)
    dim = 2048  # 2048 / 16 = 128 (整除)
    num_layers = 4

    tokenizer = SimpleCharTokenizer(vocab_size)
    # 全局矩阵 shape: [32, 16]
    global_input_ids = tokenizer.encode(raw_texts, max_len=seq_len)

    # 数据并行切片: 每张 GPU 分配 local_batch_size = 32 / 16 = 2 条样本
    local_batch_size = len(raw_texts) // world_size
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size

    # 局部张量 shape: [2, 16]
    local_input_ids = global_input_ids[start_idx:end_idx].to(device)

    # --- 3. 实例化 16 卡分片 FSDP2 模型 ---
    model = FSDP2SimpleLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        world_size=world_size,
        rank=rank,
    ).to(device)

    # 计算本地常驻参数量
    local_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Node {node_id} | Rank {rank:02d} | GPU {local_rank}] 静态常驻参数量: {local_params / 1e6:.2f} M "
        f"(仅占全局总参数 176.16M 的 1/{world_size})"
    )

    # 等待所有节点与显卡就绪
    dist.barrier()

    # --- 4. 模拟 FSDP2 前向传播计算 ---
    if rank == 0:
        print("\n[开始执行 16 卡跨机前向计算 (逐层 All-Gather 动态加载与释放)]...")

    # 执行前向传播
    # [2, 16] -> Embed Gather -> [2, 16, 2048] -> 4x Block Gathers -> [2, 16, 2048] -> Head Gather -> [2, 16, 10240]
    logits = model(local_input_ids)

    # 校验输出张量形状: [2, 16, 10240]
    expected_shape = (local_batch_size, seq_len, vocab_size)
    assert (
        logits.shape == expected_shape
    ), f"形状错误! 期望 {expected_shape}, 实际 {logits.shape}"

    print(
        f"[Node {node_id} | Rank {rank:02d}] 前向完成! 输出 Logits 形状: {list(logits.shape)}"
    )

    # --- 5. 全局 All-Reduce 汇聚损失 ---
    local_loss = logits.sum()
    # 底层通信原语: 跨 2 台机器共 16 张显卡对标量 Loss 求和
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    global_loss = local_loss / (world_size * local_batch_size * seq_len)

    if rank == 0:
        print("\n" + "=" * 80)
        print(
            f"[16 卡跨机 All-Reduce 成功] 全局平均损失 (Global Loss): {global_loss.item():.4f}"
        )
        print("=" * 80)

    # --- 6. 销毁通信组 ---
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
