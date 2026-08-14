import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW


# ==========================================
# 1. 模型架构定义
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        # x 形状: [B_local, S, dim]
        hidden = self.linear1(x)  # [B_local, S, dim * 4]
        hidden = self.act(hidden)  # [B_local, S, dim * 4]
        out = self.linear2(hidden)  # [B_local, S, dim]
        return x + out  # 残差连接: [B_local, S, dim]


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x (Token IDs) 形状: [B_local, S]
        x = self.embed(x)  # 词嵌入查表 -> [B_local, S, dim]
        for layer in self.layers:
            x = layer(x)  # 逐层前向 -> [B_local, S, dim]
        logits = self.head(x)  # 线性投影到词表空间 -> [B_local, S, vocab_size]
        return logits


# ==========================================
# 2. 模拟简易分词器与数据集
# ==========================================
class DummyTokenizer:
    """模拟 Tokenizer: 将字符转换为固定词表范围内的整数 ID"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_seq_len: int) -> torch.Tensor:
        # 基于字符的 ascii 码作为简单的离散 id
        tokens = [ord(c) % self.vocab_size for c in text]
        if len(tokens) < max_seq_len:
            tokens = tokens + [0] * (max_seq_len - len(tokens))  # Padding
        else:
            tokens = tokens[:max_seq_len]  # Truncate
        # 返回张量形状: [S]
        return torch.tensor(tokens, dtype=torch.long)


# ==========================================
# 3. 手动 DDP 模拟主逻辑
# ==========================================
def run_manual_ddp():
    # ------------------------------------------------------------------
    # Step A: 初始化分布式通信组
    # ------------------------------------------------------------------
    # 从 torchrun 启动环境中获取当前进程的 Rank 和 World Size
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 绑定当前进程到对应的物理 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化进程组 (使用 NCCL 高性能 GPU 通信后端)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"[Init] 分布式通信组初始化成功. 总节点卡数(World Size): {world_size}")

    # ------------------------------------------------------------------
    # Step B: 固定随机种子 & 模型初始化 (确保 8 张卡初始权重严格一致)
    # ------------------------------------------------------------------
    # 关键点: 在真实 DDP 中, Rank 0 的权重会广播到全卡;
    # 此处通过对所有 Rank 设置相同的 seed 实现全卡初始权重 100% 对齐.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    vocab_size = 10000
    dim = 2048
    num_layers = 4
    seq_len = 16
    local_batch_size = 2
    global_batch_size = local_batch_size * world_size  # 2 * 8 = 16

    # 实例化模型并迁移至当前 GPU
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Step C: 模拟数据加载与切分 (Data Sharding)
    # ------------------------------------------------------------------
    tokenizer = DummyTokenizer(vocab_size=vocab_size)

    # 模拟由 16 条真实文本组成的 Global Batch
    raw_texts = [
        f"这是分布式训练文本样本数据编号_{i}" for i in range(global_batch_size)
    ]

    # 将 16 条文本转为全局矩阵: [global_batch_size, seq_len] -> [16, 16]
    global_input_ids = torch.stack([tokenizer.encode(t, seq_len) for t in raw_texts])
    # 构造假定的下一个 Token 标签 (Shifted labels): [16, 16]
    global_targets = torch.roll(global_input_ids, shifts=-1, dims=-1).to(device)
    global_input_ids = global_input_ids.to(device)

    # 模拟 DistributedSampler 切片: 每个 Rank 提取属于自己的局部 Batch
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

    # local_x 形状: [B_local, S] -> [2, 16]
    local_x = global_input_ids[start_idx:end_idx].contiguous()
    # local_y 形状: [B_local, S] -> [2, 16]
    local_y = global_targets[start_idx:end_idx].contiguous()

    # ------------------------------------------------------------------
    # Step D: 前向传播 (Forward Pass)
    # ------------------------------------------------------------------
    optimizer.zero_grad()

    # 各 GPU 独立进行局部前向计算
    # local_logits 形状: [B_local, S, vocab_size] -> [2, 16, 10000]
    local_logits = model(local_x)

    # ------------------------------------------------------------------
    # Step E: 通信原语演示 1 - dist.all_gather_into_tensor
    # 作用: 零拷贝收集所有 GPU 上的 logits, 恢复出全量全局输出 (用于分布式 Eval/监控)
    # ------------------------------------------------------------------
    # 预分配接收全局聚合结果的张量
    # global_gathered_logits 形状: [B_global, S, vocab_size] -> [16, 16, 10000]
    global_gathered_logits = torch.empty(
        (global_batch_size, seq_len, vocab_size),
        dtype=local_logits.dtype,
        device=device,
    )

    # 底层通信: 将 8 张 GPU 上的 local_logits (各 2 个样本) 沿第 0 维拼接并广播给所有 GPU
    # local_logits: [2, 16, 10000] ---> global_gathered_logits: [16, 16, 10000]
    dist.all_gather_into_tensor(global_gathered_logits, local_logits)

    # ------------------------------------------------------------------
    # Step F: 损失计算与反向传播 (Loss & Backward)
    # ------------------------------------------------------------------
    # 将局部 logits 和 labels 打平以适配 CrossEntropy 计算
    # local_logits: [2 * 16, 10000] -> [32, 10000]
    # local_y: [2 * 16] -> [32]
    loss = loss_fn(local_logits.view(-1, vocab_size), local_y.view(-1))

    # 本地反向传播: 算出当前卡上局部批次对应的参数梯度 (Local Gradients)
    loss.backward()

    # ------------------------------------------------------------------
    # Step G: 通信原语演示 2 - dist.all_reduce 同步梯度 (模拟 DDP 核心机制)
    # 作用: 所有 GPU 的梯度求和并取平均, 使得每张卡上的梯度完全相等
    # ------------------------------------------------------------------
    for param in model.parameters():
        if param.grad is not None:
            # param.grad 形状与参数矩阵形状相同 (例如 embed.weight 梯度: [10000, 2048])
            # dist.all_reduce 跨 8 张卡执行 Ring-AllReduce 求和归约:
            # grad_global = sum(grad_rank_0, grad_rank_1, ..., grad_rank_7)
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            # 除以卡数 world_size, 完成平均梯度的计算 (Avg Gradients)
            param.grad.data /= world_size

    # ------------------------------------------------------------------
    # Step H: 参数更新 (Optimizer Step)
    # ------------------------------------------------------------------
    # 因为初始参数相同, 且同步后的平均梯度完全相同, 各 GPU 独立 step 后的新权重依然保持 100% 同步
    optimizer.step()

    # ------------------------------------------------------------------
    # Step I: 全局 Loss 监控聚合
    # ------------------------------------------------------------------
    # 收集各卡 loss 做监控打印
    global_loss = loss.clone().detach()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    global_loss /= world_size

    if rank == 0:
        print("\n" + "=" * 50)
        print(f"[Rank 0 输出报告]")
        print(f"1. 局部输入 local_x 形状: {list(local_x.shape)}")
        print(f"2. 局部前向输出 local_logits 形状: {list(local_logits.shape)}")
        print(
            f"3. all_gather 收集后的 global_gathered_logits 形状: {list(global_gathered_logits.shape)}"
        )
        print(f"4. 本地 Loss (Rank 0): {loss.item():.4f}")
        print(f"5. 全局平均 Loss (8 卡平均): {global_loss.item():.4f}")
        print("6. 梯度已通过 dist.all_reduce 完成跨卡同步并成功执行 Optimizer.step()")
        print("=" * 50)

    # 销毁通信组, 释放资源
    dist.destroy_process_group()


if __name__ == "__main__":
    run_manual_ddp()
