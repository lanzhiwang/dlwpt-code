"""
文件名: tp_simulation.py
说明: 使用 PyTorch 底层通信原语 (all_reduce, all_gather_into_tensor) 模拟 8 卡 Tensor Parallelism (TP)
运行方式: torchrun --nproc_per_node=8 tp_simulation.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ==========================================
# 1. 原始单卡模型结构定义 (Baseline 参考模型)
# ==========================================
class SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, dim]
        # linear1(x): [B, S, dim * 4]
        # act(...): [B, S, dim * 4]
        # linear2(...): [B, S, dim]
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
        # x shape: [B, S] (整数 Token ID)
        x = self.embed(x)  # shape: [B, S, dim]
        for layer in self.layers:
            x = layer(x)  # shape: [B, S, dim]
        return self.head(x)  # shape: [B, S, vocab_size]


# ==========================================
# 2. TP (张量并行) 底层模块实现
# ==========================================


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding: 将词表维度 [vocab_size, dim] 沿行切分为 tp_size 份.
    每张卡只负责 [vocab_start_index, vocab_end_index) 区间内的 Token.
    """

    def __init__(self, vocab_size: int, dim: int, rank: int, tp_size: int):
        super().__init__()
        self.rank = rank
        self.tp_size = tp_size
        self.dim = dim
        self.vocab_size = vocab_size  # 10000

        # 计算每张卡分到的词表大小及对应全局 Token ID 的边界
        assert vocab_size % tp_size == 0, "vocab_size 必须能被 tp_size 整除"
        self.vocab_per_rank = vocab_size // tp_size  # 10000 / 8 = 1250
        self.vocab_start_idx = rank * self.vocab_per_rank
        self.vocab_end_idx = (rank + 1) * self.vocab_per_rank
        """
        rank = 0
            self.vocab_start_idx = 0
            self.vocab_end_idx = 1250
        rank = 1
            self.vocab_start_idx = 1250
            self.vocab_end_idx = 2500
        rank = 2
            self.vocab_start_idx = 2500
            self.vocab_end_idx = 3750
        rank = 3
            self.vocab_start_idx = 3750
            self.vocab_end_idx = 5000
        rank = 4
            self.vocab_start_idx = 5000
            self.vocab_end_idx = 6250
        rank = 5
            self.vocab_start_idx = 6250
            self.vocab_end_idx = 7500
        rank = 6
            self.vocab_start_idx = 7500
            self.vocab_end_idx = 8750
        rank = 7
            self.vocab_start_idx = 8750
            self.vocab_end_idx = 10000
        """

        # 本卡持有的局部 Embedding 权重矩阵: [vocab_per_rank, dim]
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S] (全局 Token ID)
        # 步骤 1: 生成掩码, 找出当前落在本卡负责范围内的 Token
        # mask shape: [B, S] (布尔类型)
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

        # 步骤 2: 将全局 Token ID 映射为本卡切片内的局部索引 [0, vocab_per_rank - 1]
        # local_x shape: [B, S]
        local_x = torch.clamp(
            x - self.vocab_start_idx, min=0, max=self.vocab_per_rank - 1
        )

        # 步骤 3: 局部 Embedding 查表
        # local_embed shape: [B, S, dim]
        local_embed = F.embedding(local_x, self.weight)

        # 步骤 4: 不属于本卡的 Token 查表结果置 0
        # local_embed shape: [B, S, dim]
        local_embed[~mask] = 0.0

        # 步骤 5: 全局通信 All-Reduce (求和), 将各卡非 0 向量累加, 还原完整 Embedding
        # 通信后 local_embed shape: [B, S, dim]
        dist.all_reduce(local_embed, op=dist.ReduceOp.SUM)
        return local_embed


class ColumnParallelLinear(nn.Module):
    """
    列并行 Linear: 权重沿输出维度 (Out Features) 纵向切分.
    全局权重 [in_features, out_features] -> 本地切片 [in_features, out_features / tp_size].
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        assert out_features % tp_size == 0, "out_features 必须被 tp_size 整除"
        self.in_features = in_features  # 2048
        self.out_features_per_rank = out_features // tp_size  # 2048 * 4 / 8 = 1024

        # 本地权重形状: [in_features, out_features / tp_size]
        # (注: PyTorch Linear weight 内部保存为 [out_features / tp_size, in_features])
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S, in_features] (各卡输入完整副本)
        # 本地矩阵乘计算: [B, S, in_features] @ [in_features, out_features / tp_size]
        # 输出 output shape: [B, S, out_features / tp_size]
        # 无需任何卡间通信!
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    行并行 Linear: 权重沿输入维度 (In Features) 横向切分.
    全局权重 [in_features, out_features] -> 本地切片 [in_features / tp_size, out_features].
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int):
        super().__init__()
        assert in_features % tp_size == 0, "in_features 必须被 tp_size 整除"
        self.in_features_per_rank = in_features // tp_size  # 2048 * 4 / 8 = 1024
        self.out_features = out_features  # 2048

        # 本地权重形状: [in_features / tp_size, out_features]
        # (PyTorch 内部保存为 [out_features, in_features / tp_size])
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        # 全局 bias 仅在 All-Reduce 求和完成后在各卡独立加上, 防止多卡累加放大 bias
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S, in_features / tp_size] (上一层列并行的局部切片)
        # 步骤 1: 本地部分矩阵乘法 (Partial Matmul)
        # local_output shape: [B, S, out_features]
        local_output = F.linear(x, self.weight, bias=None)

        # 步骤 2: 全局通信 All-Reduce (求和)
        # 累加 8 张卡上的部分矩阵乘积: Sum(X_i * W_i) = X * W
        # 通信后 local_output shape: [B, S, out_features]
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)

        # 步骤 3: 加上 bias
        # output shape: [B, S, out_features]
        return local_output + self.bias


class ColumnParallelHead(nn.Module):
    """
    LM Head (Column Parallel + All-Gather):
    将输出层 [dim, vocab_size] 沿词表维度分片计算局部 Logits,
    然后使用 dist.all_gather_into_tensor 聚合得到全局完整 Logits.
    """

    def __init__(self, dim: int, vocab_size: int, rank: int, tp_size: int):
        super().__init__()
        self.rank = rank
        self.tp_size = tp_size  # 8
        self.dim = dim  # 2048
        self.vocab_size = vocab_size  # 10000
        self.vocab_per_rank = vocab_size // tp_size  # 10000 / 8 = 1250

        # 本地切片权重: [vocab_per_rank, dim]
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, dim))
        self.bias = nn.Parameter(torch.empty(self.vocab_per_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S, dim]
        # 步骤 1: 本地线性变换, 得到局部词表得分
        # local_logits shape: [B, S, vocab_per_rank]
        local_logits = F.linear(x, self.weight, self.bias)
        B, S, _ = local_logits.shape

        # 步骤 2: 使用 dist.all_gather_into_tensor 原语收集所有卡的分片
        # all_gather_into_tensor 默认沿着 dim=0 进行数据拼合, 要求接收 Tensor 大小为 [tp_size * B, S, vocab_per_rank]
        gathered_tensor = torch.empty(
            self.tp_size * B,
            S,
            self.vocab_per_rank,
            device=local_logits.device,
            dtype=local_logits.dtype,
        )
        dist.all_gather_into_tensor(gathered_tensor, local_logits.contiguous())

        # 步骤 3: 张量维度重组, 将多卡的 Vocab 分片拼接回完整的 vocab_size
        # gathered_tensor 原形状: [tp_size * B, S, vocab_per_rank]
        # view -> [tp_size, B, S, vocab_per_rank]
        # permute(1, 2, 0, 3) -> [B, S, tp_size, vocab_per_rank]
        # view -> [B, S, tp_size * vocab_per_rank] = [B, S, vocab_size]
        full_logits = (
            gathered_tensor.view(self.tp_size, B, S, self.vocab_per_rank)
            .permute(1, 2, 0, 3)
            .contiguous()
            .view(B, S, self.vocab_size)
        )
        # full_logits shape: [B, S, vocab_size]
        return full_logits


# ==========================================
# 3. 组合 TP Transformer 块与完整 TP 模型
# ==========================================


class TPTransformerBlock(nn.Module):
    def __init__(self, dim: int, tp_size: int):
        super().__init__()

        # MLP Layer 1: 列并行, 输出维度从 dim 扩展到 (4 * dim) / tp_size
        self.linear1 = ColumnParallelLinear(dim, dim * 4, tp_size)

        self.act = nn.GELU()

        # MLP Layer 2: 行并行, 输入维度从 (4 * dim) / tp_size 还原至 dim
        self.linear2 = RowParallelLinear(dim * 4, dim, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S, dim]
        # 列并行变换 -> shape: [B, S, 4 * dim / tp_size]
        h1 = self.linear1(x)
        # 激活函数独立在各卡计算 -> shape: [B, S, 4 * dim / tp_size]
        h2 = self.act(h1)
        # 行并行变换 + 内部 All-Reduce -> shape: [B, S, dim]
        h3 = self.linear2(h2)
        # 残差连接在各卡本地独立执行 -> shape: [B, S, dim]
        return x + h3


class TPSimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4, rank=0, tp_size=8):
        super().__init__()
        self.embed = VocabParallelEmbedding(vocab_size, dim, rank, tp_size)
        self.layers = nn.ModuleList(
            [TPTransformerBlock(dim, tp_size) for _ in range(num_layers)]
        )
        self.head = ColumnParallelHead(dim, vocab_size, rank, tp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x shape: [B, S] (整数 Token ID)
        # Vocab Parallel 查表 + All-Reduce -> shape: [B, S, dim]
        x = self.embed(x)
        for layer in self.layers:
            # Transformer 块内部计算 -> shape: [B, S, dim]
            x = layer(x)
        # Head 线性变换 + All-Gather -> shape: [B, S, vocab_size]
        return self.head(x)


# ==========================================
# 4. 权重对齐工具 (将单卡权重切分赋值给 8 卡 TP 模型)
# ==========================================


def sync_weights_from_single_to_tp(
    single_model: SingleSimpleLLM, tp_model: TPSimpleLLM, rank: int, tp_size: int
):
    """
    精确切分单卡 baseline 模型参数并赋予当前卡上的 TP 模型, 用于验证数值一致性.
    """
    with torch.no_grad():
        # 1. 切分 Embedding
        v_start = rank * (single_model.embed.num_embeddings // tp_size)
        v_end = (rank + 1) * (single_model.embed.num_embeddings // tp_size)
        tp_model.embed.weight.copy_(single_model.embed.weight[v_start:v_end, :])

        # 2. 切分各 Transformer 层的 Linear1 与 Linear2
        for s_layer, tp_layer in zip(single_model.layers, tp_model.layers):
            # Linear1: 列并行切分 (切分 weight 的 0 维, bias 的 0 维)
            hidden_4x = s_layer.linear1.out_features
            h1_shard = hidden_4x // tp_size
            tp_layer.linear1.weight.copy_(
                s_layer.linear1.weight[rank * h1_shard : (rank + 1) * h1_shard, :]
            )
            tp_layer.linear1.bias.copy_(
                s_layer.linear1.bias[rank * h1_shard : (rank + 1) * h1_shard]
            )

            # Linear2: 行并行切分 (切分 weight 的 1 维, bias 完整复制)
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


# ==========================================
# 5. 主执行逻辑与验证
# ==========================================


def main():
    # 步骤 1: 读取分布式环境变量并绑定本地 GPU
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 步骤 2: 初始化 NCCL 通信组
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 固定随机种子, 确保基准模型生成相同的初始权重
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 步骤 3: 模拟自然语言文本生成与分词张量化 (Text -> Tokens -> IDs -> Tensor Matrix)
    # 模拟纯文本输入: 2 句话, 每句话分词后长度为 4
    # "Hello world TP distributed" -> [102, 594, 8821, 33]
    # "Deep learning parallel compute" -> [4201, 78, 9012, 512]
    batch_size = 2
    seq_len = 4
    vocab_size = 10000
    dim = 2048
    num_layers = 2

    # 构建 [Batch_Size, Seq_Len] 的整数张量
    input_ids = torch.tensor(
        [[102, 594, 8821, 33], [4201, 78, 9012, 512]], dtype=torch.long, device=device
    )

    # 步骤 4: 构建单卡 Baseline 模型与 8 卡 TP 模型
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

    # 将 Baseline 的权重切分后载入 TP 模型
    sync_weights_from_single_to_tp(baseline_model, tp_model, rank, world_size)

    baseline_model.eval()
    tp_model.eval()

    # 步骤 5: 执行前向推理
    with torch.no_grad():
        # 单卡前向输出
        single_gpu_logits = baseline_model(input_ids)
        # 8 卡 TP 分布式协同前向输出
        tp_logits = tp_model(input_ids)

    # 步骤 6: 验证精度一致性
    max_diff = torch.max(torch.abs(single_gpu_logits - tp_logits)).item()
    is_correct = torch.allclose(single_gpu_logits, tp_logits, atol=1e-5)

    if rank == 0:
        print("\n" + "=" * 50)
        print("====== 8 卡 Tensor Parallelism (TP) 模拟验证 ======")
        print("=" * 50)
        print(
            f"输入矩阵 Shape       : {list(input_ids.shape)} (dtype: {input_ids.dtype})"
        )
        print(f"Baseline 输出 Shape   : {list(single_gpu_logits.shape)}")
        print(f"TP 模型输出 Shape     : {list(tp_logits.shape)}")
        print(f"两模型最大绝对误差    : {max_diff:.8e}")
        print(
            f"数值是否完全对齐 (atol=1e-5): {'[成功 PASSED]' if is_correct else '[失败 FAILED]'}"
        )
        print("=" * 50 + "\n")

    # 步骤 7: 销毁通信组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
