import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ==============================================================================
# 1. 手动实现 FSDP2 风格的分片模块 (Sharded Linear & Sharded Embedding)
# ==============================================================================


class FSDP2ShardedEmbedding(nn.Module):
    """
    手动实现 FSDP2 分片 Embedding 层:
    - 静态状态: 每张卡只存储 vocab_size / 8 行权重
    - 前向传播: all_gather 收集完整词表权重 -> 查表计算 -> 释放完整权重
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

        # 静态只分配 1/8 权重的显存空间
        # shape: [vocab_size / world_size, dim] -> [1280, 2048]
        self.sharded_weight = nn.Parameter(
            torch.randn(self.sharded_vocab_size, dim, dtype=torch.float32) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [local_batch_size, seq_len] -> [2, 16]

        # 1. 分配接收完整权重的临时张量: [vocab_size, dim] -> [10240, 2048]
        full_weight = torch.empty(
            self.vocab_size, self.dim, device=x.device, dtype=self.sharded_weight.dtype
        )

        # 2. 调用底层 NCCL 原语 all_gather_into_tensor
        # 作用: 从 8 个 rank 收集各自的 sharded_weight (1280, 2048), 沿第 0 维拼接到 full_weight (10240, 2048)
        dist.all_gather_into_tensor(full_weight, self.sharded_weight.contiguous())

        # 3. 查表计算: [2, 16] -> [2, 16, 2048]
        out = F.embedding(x, full_weight)

        # 4. FSDP2 核心内存优化: 计算完毕立即丢弃全量权重指针, 释放显存
        del full_weight

        # 输出 out shape: [2, 16, 2048]
        return out


class FSDP2ShardedLinear(nn.Module):
    """
    手动实现 FSDP2 分片 Linear 层:
    - 静态状态: 每张卡只存储 out_features / 8 行权重与 bias
    - 前向传播: all_gather 拼接得到完整权重与偏置 -> 线性变换计算 -> 立即释放完整权重
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

        # 静态分片权重: [out_features / world_size, in_features]
        self.sharded_weight = nn.Parameter(
            torch.randn(self.sharded_out_features, in_features, dtype=torch.float32)
            * 0.02
        )
        if bias:
            # 静态分片偏置: [out_features / world_size]
            self.sharded_bias = nn.Parameter(
                torch.zeros(self.sharded_out_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("sharded_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [local_batch_size, seq_len, in_features]

        # 1. 准备接收全量权重的 Buffer: [out_features, in_features]
        full_weight = torch.empty(
            self.out_features,
            self.in_features,
            device=x.device,
            dtype=self.sharded_weight.dtype,
        )

        # 2. 全局通信拼接 Weight
        dist.all_gather_into_tensor(full_weight, self.sharded_weight.contiguous())

        # 3. 处理 Bias (如果有)
        full_bias = None
        if self.sharded_bias is not None:
            # 准备接收全量偏置的 Buffer: [out_features]
            full_bias = torch.empty(
                self.out_features, device=x.device, dtype=self.sharded_bias.dtype
            )
            dist.all_gather_into_tensor(full_bias, self.sharded_bias.contiguous())

        # 4. 前向矩阵乘法: [B, S, in_features] x [out_features, in_features].T -> [B, S, out_features]
        out = F.linear(x, full_weight, full_bias)

        # 5. 立即释放全量权重与全量偏置显存
        del full_weight
        del full_bias

        # 输出 out shape: [local_batch_size, seq_len, out_features]
        return out


# ==============================================================================
# 2. 基于分片模块构建 TransformerBlock 与 SimpleLLM
# ==============================================================================


class FSDP2TransformerBlock(nn.Module):
    def __init__(self, dim: int, world_size: int, rank: int):
        super().__init__()

        # linear1 将维度从 dim 放大到 dim * 4 (例如 2048 -> 8192)
        # 每张卡分片后存储大小: [8192 / 8, 2048] = [1024, 2048]
        self.linear1 = FSDP2ShardedLinear(dim, dim * 4, world_size, rank)

        self.act = nn.GELU()

        # linear2 将维度从 dim * 4 还原回 dim (例如 8192 -> 2048)
        # 每张卡分片后存储大小: [2048 / 8, 8192] = [256, 8192]
        self.linear2 = FSDP2ShardedLinear(dim * 4, dim, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [local_batch_size, seq_len, dim] -> [2, 16, 2048]
        residual = x

        # 步骤 1: linear1 前向 (触发 linear1 的 all_gather)
        # 输出 shape: [2, 16, 8192]
        h1 = self.linear1(x)

        # 步骤 2: GELU 激活函数 (纯逐元素本地计算, 无跨卡通信)
        # 输出 shape: [2, 16, 8192]
        h2 = self.act(h1)

        # 步骤 3: linear2 前向 (触发 linear2 的 all_gather)
        # 输出 shape: [2, 16, 2048]
        h3 = self.linear2(h2)

        # 步骤 4: 残差连接
        # 输出 shape: [2, 16, 2048]
        return residual + h3


class FSDP2SimpleLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10240,
        dim: int = 2048,
        num_layers: int = 4,
        world_size: int = 8,
        rank: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

        # 1. 分片 Embedding 层: [10240 / 8, 2048] = [1280, 2048]
        self.embed = FSDP2ShardedEmbedding(vocab_size, dim, world_size, rank)

        # 2. 堆叠 num_layers 个分片 TransformerBlock
        self.layers = nn.ModuleList(
            [FSDP2TransformerBlock(dim, world_size, rank) for _ in range(num_layers)]
        )

        # 3. 分片 LM Head 线性分类层: [10240 / 8, 2048] = [1280, 2048]
        self.head = FSDP2ShardedLinear(dim, vocab_size, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 Token IDs shape: [local_batch_size, seq_len] -> [2, 16]

        # --- 阶段 1: Embedding 查找 ---
        # 内部触发 Embedding 参数 All-Gather -> 查表 -> 释放权重
        # 输出 x shape: [2, 16, 2048]
        x = self.embed(x)

        # --- 阶段 2: Transformer 层依次前向 ---
        # 每经过一层, 动态加载该层完整权重 -> 计算 -> 销毁权重
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 输出 x 保持 shape: [2, 16, 2048]

        # --- 阶段 3: Head 预测 Logits ---
        # 内部触发 Head 参数 All-Gather -> 投影回词表 -> 释放权重
        # 输出 logits shape: [2, 16, 10240]
        logits = self.head(x)
        return logits


# ==============================================================================
# 3. 简单的字符级分词器 (模拟纯文本转张量矩阵过程)
# ==============================================================================


class SimpleCharTokenizer:
    """一个简单的字符级分词器, 负责将纯文本 String 转为 Integer ID 矩阵"""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, texts: list[str], max_len: int) -> torch.Tensor:
        batch_ids = []
        for text in texts:
            # 取字符的 ASCII 码作为基础 ID
            ids = [ord(c) % self.vocab_size for c in text]
            # 截断或 Padding 到指定 max_len
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))  # 0 为 padding_id
            else:
                ids = ids[:max_len]
            batch_ids.append(ids)
        # 转为 PyTorch LongTensor 矩阵
        return torch.tensor(batch_ids, dtype=torch.long)


# ==============================================================================
# 4. 主执行函数: 初始化通信组并运行前向传播
# ==============================================================================


def main():
    # --- 1. 显卡通信组初始化 ---
    # torchrun 会自动在环境中注入 RANK, WORLD_SIZE, LOCAL_RANK
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 绑定当前进程到对应的物理 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化 NCCL 通信组 (所有 8 张显卡完成 TCP 握手并构建通信拓扑)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    if rank == 0:
        print("=" * 70)
        print(f"[NCCL 初始化完成] 总显卡数 (World Size): {world_size}")
        print("=" * 70)

    # --- 2. 纯文本转矩阵 (Text to Matrix) ---
    # 假设全局有 16 句输入文本 (Global Batch Size = 16)
    raw_texts = [
        f"Sample prompt number {i}: Hello FSDP2 distributed world!" for i in range(16)
    ]
    seq_len = 16
    vocab_size = 10240  # 保证能被 8 整除 (10240 / 8 = 1280)
    dim = 2048  # 保证能被 8 整除 (2048 / 8 = 256)
    num_layers = 4

    tokenizer = SimpleCharTokenizer(vocab_size)
    # 全局 Token 矩阵 shape: [16, 16]
    global_input_ids = tokenizer.encode(raw_texts, max_len=seq_len)

    # 数据并行切分 (Data Sharding):
    # 每张卡分配 local_batch_size = 16 / 8 = 2 个样本
    local_batch_size = len(raw_texts) // world_size
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    """
    rank = 0
        start_idx = 0
        end_idx = 2
    rank = 1
        start_idx = 2
        end_idx = 4
    rank = 2
        start_idx = 4
        end_idx = 6
    rank = 3
        start_idx = 6
        end_idx = 8
    rank = 4
        start_idx = 8
        end_idx = 10
    rank = 5
        start_idx = 10
        end_idx = 12
    rank = 6
        start_idx = 12
        end_idx = 14
    rank = 7
        start_idx = 14
        end_idx = 16
    """

    # 局部 Token 矩阵 shape: [2, 16]
    local_input_ids = global_input_ids[start_idx:end_idx].to(device)

    # --- 3. 实例化手动分片的 FSDP2 模型 ---
    model = FSDP2SimpleLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        world_size=world_size,
        rank=rank,
    ).to(device)

    # --- 4. 显存占用检查 (验证参数仅占 1/8) ---
    local_param_count = sum(p.numel() for p in model.parameters())
    # 理论全量参数量 = vocab * dim + num_layers * (dim * (4 * dim) + (4 * dim) * dim) + dim * vocab
    # ≈ 10240 * 2048 + 4 * (2048 * 8192 * 2) + 2048 * 10240 ≈ 20.97M + 134.21M + 20.97M = 176.16M
    # 分片后每张卡参数量 = 176.16M / 8 ≈ 22.02M
    print(
        f"[Rank {rank}] 静态常驻分片参数量: {local_param_count / 1e6:.2f} M (仅占完整模型的 1/{world_size})"
    )

    # 进程同步
    dist.barrier()

    # --- 5. 执行 FSDP2 模拟前向传播 ---
    if rank == 0:
        print("\n[开始执行模拟 FSDP2 前向传播 (含 All-Gather 收集与释放)]...")

    # 执行 forward:
    # [2, 16] -> Embed AllGather -> [2, 16, 2048] -> Layer AllGathers -> [2, 16, 2048] -> Head AllGather -> [2, 16, 10240]
    logits = model(local_input_ids)

    # 验证输出张量形状
    expected_shape = (local_batch_size, seq_len, vocab_size)
    assert (
        logits.shape == expected_shape
    ), f"输出形状错误: 期望 {expected_shape}, 实际 {logits.shape}"

    print(
        f"[Rank {rank}] 前向传播成功! 输出 Logits 形状: {list(logits.shape)} (设备: {logits.device})"
    )

    # --- 6. 演示全局损失汇聚 (模拟 All-Reduce) ---
    # 计算局部的虚拟 Loss
    local_dummy_loss = logits.sum()

    # 使用 dist.all_reduce 原语计算 8 卡上的全局总 Loss
    dist.all_reduce(local_dummy_loss, op=dist.ReduceOp.SUM)
    global_loss = local_dummy_loss / (world_size * local_batch_size * seq_len)

    if rank == 0:
        print(
            f"\n[All-Reduce 通信成功] 全局平均损失 (Global Loss): {global_loss.item():.4f}"
        )
        print("=" * 70)

    # --- 7. 清理通信组 ---
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
