"""
文件名: tp_simulation_2nodes.py
说明: 使用底层通信原语在 2 台机器 16 张 GPU 上模拟 Megatron-style Tensor Parallelism (TP)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# =========================================================================
# 1. 原始单卡模型结构定义 (用于精度对照的 Baseline 模型)
# =========================================================================
class SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S, dim]
        # self.linear1(x) -> [B, S, dim * 4]
        # self.act(...)   -> [B, S, dim * 4]
        # self.linear2(...) -> [B, S, dim]
        # 残差连接        -> [B, S, dim]
        return x + self.linear2(self.act(self.linear1(x)))


class SingleSimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [SingleTransformerBlock(dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S] (整数 Token ID 矩阵)
        x = self.embed(x)  # [B, S] -> [B, S, dim]
        for layer in self.layers:
            x = layer(x)  # [B, S, dim] -> [B, S, dim]
        return self.head(x)  # [B, S, dim] -> [B, S, vocab_size]


# =========================================================================
# 2. 16 卡 TP 底层并行算子定义
# =========================================================================


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding:
    将大小为 [vocab_size, dim] 的词表在 16 张卡上均匀横向切片, 每张卡维护 [vocab_size / tp_size, dim].
    """

    def __init__(self, vocab_size: int, dim: int, rank: int, tp_size: int):
        super().__init__()
        self.rank = rank
        self.tp_size = tp_size
        self.dim = dim
        self.vocab_size = vocab_size

        assert (
            vocab_size % tp_size == 0
        ), f"vocab_size ({vocab_size}) 必须能被 tp_size ({tp_size}) 整除"
        self.vocab_per_rank = vocab_size // tp_size
        self.vocab_start_idx = rank * self.vocab_per_rank
        self.vocab_end_idx = (rank + 1) * self.vocab_per_rank

        # 本卡持有的局部 Embedding 权重切片: [vocab_per_rank, dim] -> [625, 2048]
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S] (全局 Token ID)

        # 步骤 1: 确定哪些 token 属于当前 GPU 负责的词表切片区间
        # mask 形状: [B, S] (布尔值)
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

        # 步骤 2: 将全局 ID 转换为当前卡的局部索引 [0, vocab_per_rank - 1]
        # local_x 形状: [B, S]
        local_x = torch.clamp(
            x - self.vocab_start_idx, min=0, max=self.vocab_per_rank - 1
        )

        # 步骤 3: 局部查表
        # local_embed 形状: [B, S, dim] -> [B, S, 2048]
        local_embed = F.embedding(local_x, self.weight)

        # 步骤 4: 非本卡负责的 Token 查表结果置 0
        # local_embed 形状: [B, S, dim]
        local_embed[~mask] = 0.0

        # 步骤 5: 跨 16 卡 All-Reduce (SUM) 累加求和
        # 通信后 local_embed 形状: [B, S, dim] -> [B, S, 2048] (每张卡均获得完整的查表结果)
        dist.all_reduce(local_embed, op=dist.ReduceOp.SUM)
        return local_embed


class ColumnParallelLinear(nn.Module):
    """
    列并行 Linear: 权重按输出特征维度 (dim * 4) 切分.
    全局权重 [dim, dim * 4] -> 本地切片 [dim, (dim * 4) / tp_size].
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        assert (
            out_features % tp_size == 0
        ), f"out_features 必须能被 tp_size ({tp_size}) 整除"
        self.in_features = in_features
        self.out_features_per_rank = out_features // tp_size

        # PyTorch 内部权重保存形式为 [Out, In] -> [512, 2048]
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        # 局部 bias 形状: [512]
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S, in_features] -> [B, S, 2048]
        # 本地矩阵乘法: [B, S, 2048] @ [2048, 512] + [512]
        # 输出 output 形状: [B, S, out_features_per_rank] -> [B, S, 512]
        # 无需跨卡通信!
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    行并行 Linear: 权重按输入特征维度 (dim * 4) 切分.
    全局权重 [dim * 4, dim] -> 本地切片 [(dim * 4) / tp_size, dim].
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        assert (
            in_features % tp_size == 0
        ), f"in_features 必须能被 tp_size ({tp_size}) 整除"
        self.in_features_per_rank = in_features // tp_size
        self.out_features = out_features

        # PyTorch 权重保存为 [Out, In] -> [2048, 512]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        # 完整的全局 Bias 形状: [2048] (在 All-Reduce 求和完成后独立加上, 防止多卡累加放大)
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S, in_features_per_rank] -> [B, S, 512]

        # 步骤 1: 本地部分内积计算 (Partial Matmul, 暂不加 bias)
        # local_output 形状: [B, S, out_features] -> [B, S, 2048]
        local_output = F.linear(x, self.weight, bias=None)

        # 步骤 2: 跨 16 卡执行 All-Reduce (SUM) 求和: Sum(X_i * W_i) = X * W
        # 通信后 local_output 形状: [B, S, out_features] -> [B, S, 2048]
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)

        # 步骤 3: 累加全局 Bias
        # 输出形状: [B, S, out_features] -> [B, S, 2048]
        return local_output + self.bias


class ColumnParallelHead(nn.Module):
    """
    输出 Head (Column Parallel + All-Gather):
    权重按词表维度切分: [dim, vocab_size / tp_size] -> [2048, 625].
    计算局部 Logits 后, 使用 dist.all_gather_into_tensor 聚合得到 [B, S, 10000].
    """

    def __init__(self, dim: int, vocab_size: int, rank: int, tp_size: int):
        super().__init__()
        self.rank = rank
        self.tp_size = tp_size
        self.dim = dim
        self.vocab_size = vocab_size
        self.vocab_per_rank = vocab_size // tp_size

        # 本地权重切片: [625, 2048]
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, dim))
        # 本地偏置切片: [625]
        self.bias = nn.Parameter(torch.empty(self.vocab_per_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S, dim] -> [B, S, 2048]

        # 步骤 1: 本地线性变换得到局部词表 Logits
        # local_logits 形状: [B, S, vocab_per_rank] -> [B, S, 625]
        local_logits = F.linear(x, self.weight, self.bias)
        B, S, _ = local_logits.shape

        # 步骤 2: 使用底层通信原语 all_gather_into_tensor
        # 接收张量需沿 dim=0 接收 16 张卡的数据, 形状为: [tp_size * B, S, vocab_per_rank] -> [16 * B, S, 625]
        gathered_tensor = torch.empty(
            self.tp_size * B,
            S,
            self.vocab_per_rank,
            device=local_logits.device,
            dtype=local_logits.dtype,
        )
        dist.all_gather_into_tensor(gathered_tensor, local_logits.contiguous())

        # 步骤 3: 还原并拼合张量维度
        # gathered_tensor: [16 * B, S, 625]
        # .view(16, B, S, 625)               -> [16, B, S, 625]
        # .permute(1, 2, 0, 3)               -> [B, S, 16, 625]
        # .contiguous().view(B, S, vocab)    -> [B, S, 10000]
        full_logits = (
            gathered_tensor.view(self.tp_size, B, S, self.vocab_per_rank)
            .permute(1, 2, 0, 3)
            .contiguous()
            .view(B, S, self.vocab_size)
        )
        return full_logits


# =========================================================================
# 3. TP Transformer 层与完整网络组装
# =========================================================================


class TPTransformerBlock(nn.Module):
    def __init__(self, dim: int, tp_size: int):
        super().__init__()
        # 列并行 Linear1: [dim] -> [dim * 4 / tp_size]
        self.linear1 = ColumnParallelLinear(dim, dim * 4, tp_size)
        self.act = nn.GELU()
        # 行并行 Linear2: [dim * 4 / tp_size] -> [dim]
        self.linear2 = RowParallelLinear(dim * 4, dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [B, S, dim] -> [B, S, 2048]
        # 1. 列并行投影 (无通信) -> [B, S, 512]
        h1 = self.linear1(x)
        # 2. 局部激活函数计算 (无通信) -> [B, S, 512]
        h2 = self.act(h1)
        # 3. 行并行投影 + 内部 All-Reduce (SUM) 跨 16 卡聚合 -> [B, S, 2048]
        h3 = self.linear2(h2)
        # 4. 残差连接 (本地独立计算) -> [B, S, 2048]
        return x + h3


class TPSimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4, rank=0, tp_size=16):
        super().__init__()
        self.embed = VocabParallelEmbedding(vocab_size, dim, rank, tp_size)
        self.layers = nn.ModuleList(
            [TPTransformerBlock(dim, tp_size) for _ in range(num_layers)]
        )
        self.head = ColumnParallelHead(dim, vocab_size, rank, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [B, S]
        # 1. 词表并行查表 + All-Reduce -> [B, S, 2048]
        x = self.embed(x)
        # 2. 4 层 Transformer 串行前向传播 -> [B, S, 2048]
        for layer in self.layers:
            x = layer(x)
        # 3. 输出 Head 局部计算 + All-Gather 收集 -> [B, S, 10000]
        return self.head(x)


# =========================================================================
# 4. 权重精确切分与对齐函数
# =========================================================================


def sync_weights_from_single_to_tp(
    single_model: SingleSimpleLLM, tp_model: TPSimpleLLM, rank: int, tp_size: int
):
    """
    将 Baseline 单卡模型的完整权重精确切分后, 拷贝至当前 rank 的 TP 局部模型中.
    """
    with torch.no_grad():
        # 1. 切分 Embedding
        v_shard = single_model.embed.num_embeddings // tp_size
        tp_model.embed.weight.copy_(
            single_model.embed.weight[rank * v_shard : (rank + 1) * v_shard, :]
        )

        # 2. 切分各层 Transformer Blocks
        for s_layer, tp_layer in zip(single_model.layers, tp_model.layers):
            # Linear1: 列切分 (切分 weight 0 维, bias 0 维)
            h1_shard = s_layer.linear1.out_features // tp_size
            tp_layer.linear1.weight.copy_(
                s_layer.linear1.weight[rank * h1_shard : (rank + 1) * h1_shard, :]
            )
            tp_layer.linear1.bias.copy_(
                s_layer.linear1.bias[rank * h1_shard : (rank + 1) * h1_shard]
            )

            # Linear2: 行切分 (切分 weight 1 维, bias 复制完整副本)
            h2_shard = s_layer.linear2.in_features // tp_size
            tp_layer.linear2.weight.copy_(
                s_layer.linear2.weight[:, rank * h2_shard : (rank + 1) * h2_shard]
            )
            tp_layer.linear2.bias.copy_(s_layer.linear2.bias)

        # 3. 切分 Head (列切分)
        head_shard = single_model.head.out_features // tp_size
        tp_model.head.weight.copy_(
            single_model.head.weight[rank * head_shard : (rank + 1) * head_shard, :]
        )
        tp_model.head.bias.copy_(
            single_model.head.bias[rank * head_shard : (rank + 1) * head_shard]
        )


# =========================================================================
# 5. 主程序与多机验证
# =========================================================================


def main():
    # 读取跨机环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 绑定本机物理 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化 NCCL 跨机 16 卡全局通信组
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 固定种子以保证 Baseline 模型在各卡创建时权重严格一致
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 超参数定义
    batch_size = 2
    seq_len = 4
    vocab_size = 10000
    dim = 2048
    num_layers = 4

    # 模拟输入文本矩阵 (2 个样本, 每个样本 4 个 Token)
    # shape: [B, S] = [2, 4]
    input_ids = torch.tensor(
        [[102, 594, 8821, 33], [4201, 78, 9012, 512]], dtype=torch.long, device=device
    )

    # 实例化单卡 Baseline 与 16 卡 TP 模型
    baseline_model = SingleSimpleLLM(
        vocab_size=vocab_size, dim=dim, num_layers=num_layers
    ).to(device)
    tp_model = TPSimpleLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        rank=rank,
        tp_size=world_size,
    ).to(device)

    # 将单卡权重切片赋值给 TP 模型
    sync_weights_from_single_to_tp(baseline_model, tp_model, rank, world_size)

    baseline_model.eval()
    tp_model.eval()

    with torch.no_grad():
        # Baseline 单卡前向
        baseline_logits = baseline_model(input_ids)
        # 16 卡 TP 跨机协同前向
        tp_logits = tp_model(input_ids)

    # 计算最大绝对误差
    max_diff = torch.max(torch.abs(baseline_logits - tp_logits)).item()
    is_close = torch.allclose(baseline_logits, tp_logits, atol=1e-5)

    # 在 Rank 0 (Node 0 的第一张卡) 上打印验证结果
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"====== 2 台机器 16 卡 Tensor Parallelism (TP={world_size}) 验证 ======")
        print("=" * 60)
        print(
            f"输入张量矩阵 Shape      : {list(input_ids.shape)} (dtype: {input_ids.dtype})"
        )
        print(f"Baseline 输出 Logits Shape : {list(baseline_logits.shape)}")
        print(f"TP-16 模型输出 Logits Shape : {list(tp_logits.shape)}")
        print(f"最大绝对误差 (Max Diff)    : {max_diff:.8e}")
        print(
            f"数值精度是否完全对齐 (atol=1e-5): {'[成功 PASSED]' if is_close else '[失败 FAILED]'}"
        )
        print("=" * 60 + "\n")

    # 销毁通信组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
