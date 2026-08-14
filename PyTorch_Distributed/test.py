import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# -------------------------------------------------------------
# 1. 使用低阶原语手写 2D 并行(TP + FSDP2)Transformer Block
# -------------------------------------------------------------
class Manual2DTransformerBlock(nn.Module):
    """
    手动实现的 2D 混合并行 Transformer Block (TP=2, FSDP2=4)
    架构设计:
    - TP (Tensor Parallel): Megatron 风格, Linear1 柱切分, Linear2 行切分.
    - FSDP2 (Fully Sharded Data Parallel): 在 DP 维度上将 TP 后的权重进一步做 4 等份 Shard.
    """

    def __init__(self, dim, tp_group, dp_group, tp_size=2, dp_size=4):
        super().__init__()
        self.dim = dim  # 隐藏层维度 (隐藏特征数, 如 2048)
        self.tp_group = tp_group  # 当前卡所属的 TP 通信组 (大小为 2, 如 [0, 1])
        self.dp_group = dp_group  # 当前卡所属的 DP 通信组 (大小为 4, 如 [0, 2, 4, 6])
        self.tp_size = tp_size  # TP 并行度 = 2
        self.dp_size = dp_size  # DP 并行度 = 4

        # ---------------------------------------------------------
        # [ Linear1 权重切分逻辑 (Colwise Parallel + FSDP2) ]
        # 逻辑全量权重形状: [out_features, in_features] = [8192, 2048]
        #
        # 步骤 1 (TP 柱切分): 沿 Out 维度 (8192) 切分为 tp_size(2) 份
        # -> 每个 TP Rank 对应的逻辑权重形状为: [4096, 2048]
        #
        # 步骤 2 (FSDP2 显存切分): 将 TP 逻辑权重 [4096, 2048] 在 dp_size(4) 组内进一步切分
        # -> 沿 Out 维度切 4 份: 4096 / 4 = 1024
        # -> 本 GPU 显存中真正常驻保存的参数碎片 (Shard) 形状为: [1024, 2048]
        # ---------------------------------------------------------
        self.tp_out_dim1 = (dim * 4) // tp_size
        # 8192 / 2 = 4096 (单个 TP Rank 的输出维度)

        self.fsdp_shard_dim1 = self.tp_out_dim1 // dp_size
        # 4096 / 4 = 1024 (单个 GPU 保存的 FSDP 碎片维度)

        # 定义常驻 GPU 显存的 Parameter Shard(相比全量权重, 显存直接压缩为 1/8)
        self.linear1_weight_shard = nn.Parameter(
            torch.randn(self.fsdp_shard_dim1, dim) / (dim**0.5)
        )

        # ---------------------------------------------------------
        # [ Linear2 权重切分逻辑 (Rowwise Parallel + FSDP2) ]
        # 逻辑全量权重形状: [out_features, in_features] = [2048, 8192]
        #
        # 步骤 1 (TP 行切分): 沿 In 维度 (8192) 切分为 tp_size(2) 份
        # -> 每个 TP Rank 对应的逻辑权重形状为: [2048, 4096]
        #
        # 步骤 2 (FSDP2 显存切分): 将 TP 逻辑权重 [2048, 4096] 在 dp_size(4) 组内进一步切分
        # -> 沿 Out 维度 (2048) 切 4 份: 2048 / 4 = 512
        # -> 本 GPU 显存中真正常驻保存的参数碎片 (Shard) 形状为: [512, 4096]
        # ---------------------------------------------------------
        self.tp_in_dim2 = (dim * 4) // tp_size
        # 8192 / 2 = 4096 (单个 TP Rank 的输入维度)

        self.fsdp_shard_dim2 = dim // dp_size
        # 2048 / 4 = 512 (单个 GPU 保存的 FSDP 碎片维度)

        self.linear2_weight_shard = nn.Parameter(
            torch.randn(self.fsdp_shard_dim2, self.tp_in_dim2) / (self.tp_in_dim2**0.5)
        )

    def forward(self, x):
        # 输入 x 的形状: [Batch_Size, Seq_Len, Dim] -> 例如 [2, 16, 2048]
        # 注意: 同一 TP 组内的 GPU (如 GPU 0 和 GPU 1) 接收到的 x 是完全相同的!

        # =========================================================
        # 步骤 1: Linear1 前向计算 (Colwise TP + FSDP2)
        # =========================================================

        # [1.1 准备 FSDP2 接收缓冲区]
        # [FSDP2 原理展示]: 在 DP 组内 AllGather 拼凑出属于本 TP Rank 的全量权重
        # 输入分片: [1024, 2048] (来自 4 张 DP 卡) -> 收集后形状: [4096, 2048]
        # 目标: 分配一块能够容纳当前 TP Rank 全量权重 [4096, 2048] 的空 Tensor
        full_linear1_weight = torch.empty(
            self.tp_out_dim1, self.dim, device=x.device, dtype=x.dtype
        )

        # [1.2 FSDP2 参数重构 (AllGather)]
        # 在 DP 组 (如 GPU 0, 2, 4, 6) 内部收集各自持有的 [1024, 2048] 碎片
        # 沿第 0 维拼接后得到 [4096, 2048] 并写入 full_linear1_weight
        dist.all_gather_into_tensor(
            full_linear1_weight, self.linear1_weight_shard, group=self.dp_group
        )

        # [1.3 TP 柱切分本地矩阵乘法]
        # 输入 x: [B, L, 2048], 权重: [4096, 2048] -> 计算 x @ W^T
        # TP Rank 0 算前 4096 个特征通道, TP Rank 1 算后 4096 个特征通道
        # 输出 h1 形状: [B, L, 4096]
        h1 = F.linear(x, full_linear1_weight)

        # [1.4 FSDP2 显存回收]
        # 重构出的 [4096, 2048] 全量权重已完成计算, 立即从显存中销毁!
        # 显存占用瞬间回落到常驻碎片 [1024, 2048] 的水平
        del full_linear1_weight

        # [1.5 本地 GELU 激活函数]
        # 逐元素作用于局部特征通道, 形状不变: [B, L, 4096]
        a1 = F.gelu(h1)

        # =========================================================
        # 步骤 2: Linear2 前向计算 (Rowwise TP + FSDP2)
        # =========================================================

        # [2.1 准备 FSDP2 接收缓冲区]
        # [FSDP2 原理展示]: 在 DP 组内 AllGather 拼凑出属于本 TP Rank 的 Rowwise 权重
        # 输入分片: [512, 4096] -> 收集后形状: [2048, 4096]
        # 目标: 准备接收当前 TP Rank 的 Rowwise 逻辑权重 [2048, 4096]
        full_linear2_weight = torch.empty(
            self.dim, self.tp_in_dim2, device=x.device, dtype=x.dtype
        )

        # [2.2 FSDP2 参数重构 (AllGather)]
        # 在 DP 组内收集各自持有的 [512, 4096] 碎片 -> 拼出 [2048, 4096]
        dist.all_gather_into_tensor(
            full_linear2_weight, self.linear2_weight_shard, group=self.dp_group
        )

        # [2.3 TP 行切分本地矩阵乘法]
        # 输入 a1: [B, L, 4096], 权重: [2048, 4096]
        # 计算: a1 @ W^T -> 输出 partial_y 形状: [B, L, 2048]
        # [数学注意]: 此时算出的 partial_y 只是完整输出的一部分(部分和 Partial Sum)
        partial_y = F.linear(a1, full_linear2_weight)

        # [2.4 FSDP2 显存回收]
        del full_linear2_weight

        # [2.5 TP 规约求和 (AllReduce Sum)]
        # 在 TP 组 (如 GPU 0 和 GPU 1) 内部将 Partial Sum 累加:
        # Y = Partial_Y(TP_Rank_0) + Partial_Y(TP_Rank_1)
        # 规约后, TP 组内的每张卡都得到了完整且一致的 Y, 形状: [B, L, 2048]
        dist.all_reduce(partial_y, op=dist.ReduceOp.SUM, group=self.tp_group)
        y = partial_y  # 形状: [B, L, 2048]

        # =========================================================
        # 步骤 3: 残差连接
        # =========================================================
        # 本地元素级相加, 输出形状: [B, L, 2048]
        out = x + y
        return out


# -------------------------------------------------------------
# 2. 使用低阶原语手写 SimpleLLM
# -------------------------------------------------------------
class Manual2DSimpleLLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, tp_group, dp_group):
        super().__init__()
        self.dp_group = dp_group

        # ---------------------------------------------------------
        # Embedding 层: 采用 FSDP2 显存切分 (不切分 TP, 按 DP=4 切分)
        # 全量词表权重: [10000, 2048]
        # FSDP2 4 等分 -> 本 GPU 保存的碎片形状: [2500, 2048]
        # ---------------------------------------------------------
        self.embed_shard_size = vocab_size // 4  # 10000 / 4 = 2500
        self.embed_weight_shard = nn.Parameter(torch.randn(self.embed_shard_size, dim))

        # Transformer Blocks 列表
        self.layers = nn.ModuleList(
            [
                Manual2DTransformerBlock(dim, tp_group, dp_group)
                for _ in range(num_layers)
            ]
        )

        # ---------------------------------------------------------
        # Head 分类头: 采用 Colwise TP (5000) + FSDP2 (DP=4)
        # 全量 Output 维度: 10000 -> TP=2 切分为 5000 -> FSDP2=4 切分为 1250
        # 本 GPU 保存的碎片形状: [1250, 2048]
        # ---------------------------------------------------------
        self.head_shard_size = (vocab_size // 2) // 4  # (10000 / 2) / 4 = 1250
        self.head_weight_shard = nn.Parameter(torch.randn(self.head_shard_size, dim))

    def forward(self, x_tokens):
        # 输入 Token IDs 形状: [Batch_Size, Seq_Len] -> 如 [2, 16]

        # [1. Embedding 前向]
        # 1.1 分配全量 Embedding 缓冲区 [10000, 2048]
        full_embed_weight = torch.empty(
            10000, 2048, device=x_tokens.device, dtype=torch.float32
        )

        # 1.2 在 DP 组内 AllGather 重构完整 Embedding 矩阵
        dist.all_gather_into_tensor(
            full_embed_weight, self.embed_weight_shard, group=self.dp_group
        )

        # 1.3 查表得到隐藏状态, 输出 x: [B, L, 2048]
        x = F.embedding(x_tokens, full_embed_weight)
        # 1.4 立即释放全量 Embedding 显存
        del full_embed_weight

        # [2. Transformer Blocks 顺序前向]
        for layer in self.layers:
            x = layer(x)

        # [3. Head 分类头前向]
        # 3.1 分配属于当前 TP Rank 的 Head 缓冲区 [5000, 2048]
        full_head_tp_weight = torch.empty(
            5000, 2048, device=x_tokens.device, dtype=x.dtype
        )
        # 3.2 在 DP 组内 AllGather 重构属于当前 TP Rank 的 Head 权重
        dist.all_gather_into_tensor(
            full_head_tp_weight, self.head_weight_shard, group=self.dp_group
        )
        # 3.3 矩阵乘法计算分类 Logits, 输出形状: [B, L, 5000]
        logits = F.linear(x, full_head_tp_weight)
        # 3.4 立即释放 Head 显存
        del full_head_tp_weight

        return logits


# -------------------------------------------------------------
# 3. 运行与验证主程序
# -------------------------------------------------------------
def main():
    # 初始化 NCCL 分布式后端
    dist.init_process_group("nccl")

    # 从环境变量获取当前的 Rank 状态
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_size = 2  # 张量并行度
    dp_size = 4  # 数据并行度 (FSDP)

    # ---------------------------------------------------------
    # 手动建立 2D 通信组网格 (Device Mesh Grid)
    # 网格维度: [dp_size, tp_size] = [4, 2]
    # GPU 布局:
    # 行 0: GPU 0, GPU 1  (DP 组 0 的 2 个 TP 卡)
    # 行 1: GPU 2, GPU 3  (DP 组 1 的 2 个 TP 卡)
    # 行 2: GPU 4, GPU 5  (DP 组 2 的 2 个 TP 卡)
    # 行 3: GPU 6, GPU 7  (DP 组 3 的 2 个 TP 卡)
    # ---------------------------------------------------------
    tp_group = None
    dp_group = None

    # [构建 4 个独立的 TP 通信组] -> 沿着网格的"行"切分
    # 组 0: [0, 1] | 组 1: [2, 3] | 组 2: [4, 5] | 组 3: [6, 7]
    for i in range(dp_size):
        ranks = [i * tp_size + j for j in range(tp_size)]
        group = dist.new_group(ranks)  # 创建分布式通信子组
        if rank in ranks:
            tp_group = group  # 当前 GPU 记录自己所在的 TP 组
    """
    i = 0
        j = 0
        j = 1
        ranks = [0, 1]
        group
        tp_group
    i = 1
        j = 0
        j = 1
        ranks = [2, 3]
        group
        tp_group
    i = 2
        j = 0
        j = 1
        ranks = [4, 5]
        group
        tp_group
    i = 3
        j = 0
        j = 1
        ranks = [6, 7]
        group
        tp_group
    """

    # [构建 2 个独立的 DP 通信组] -> 沿着网格的"列"切分
    # 组 0 (TP Rank 0 组): [0, 2, 4, 6]
    # 组 1 (TP Rank 1 组): [1, 3, 5, 7]
    for j in range(tp_size):
        ranks = [i * tp_size + j for i in range(dp_size)]
        group = dist.new_group(ranks)  # 创建分布式通信子组
        if rank in ranks:
            dp_group = group  # 当前 GPU 记录自己所在的 DP 组
    """
    j = 0
        i = 0
        i = 1
        i = 2
        i = 3
        ranks = [0, 2, 4, 6]
        group
        dp_group
    j = 1
        i = 0
        i = 1
        i = 2
        i = 3
        ranks = [1, 3, 5, 7]
        group
        dp_group
    """

    # ---------------------------------------------------------
    # 实例化模型与模拟输入
    # ---------------------------------------------------------
    model = Manual2DSimpleLLM(
        vocab_size=10000, dim=2048, num_layers=4, tp_group=tp_group, dp_group=dp_group
    ).to(device)

    # [数据并行采样规则]:
    # 1. 相同 TP 组内(如 GPU 0 和 GPU 1)必须输入[完全相同]的数据 Token.
    # 2. 不同 DP 组之间(如 DP Rank 0 和 DP Rank 1)必须输入[互不相同]的数据 Token.
    dp_rank = rank // tp_size  # 计算当前 GPU 属于第几个 DP 组 (0, 1, 2, 3)
    """
    >>> rank = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> [r // 2 for r in rank]
    [0, 0, 1, 1, 2, 2, 3, 3]
    """
    torch.manual_seed(42 + dp_rank)
    dummy_input = torch.randint(0, 10000, (2, 16), device=device)

    # 执行前向传播
    logits = model(dummy_input)

    # 打印前向传播结果 (Logits 形状应为 [2, 16, 5000])
    print(f"[Rank {rank}] 前向传播成功! 输出 Logits 形状: {logits.shape}")

    # 等待所有卡完成计算, 随后销毁进程组, 回收网络资源
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
