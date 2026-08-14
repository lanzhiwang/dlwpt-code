import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW


# ==============================================================================
# 1. 模型架构定义
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 前馈网络升维: dim -> dim * 4
        self.linear1 = nn.Linear(dim, dim * 4)
        # 激活函数: GELU
        self.act = nn.GELU()
        # 前馈网络降维: dim * 4 -> dim
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        # 输入 x 形状: [B_local, S, dim] -> [2, 16, 2048]

        # 线性升维变换: [2, 16, 2048] @ [2048, 8192] -> [2, 16, 8192]
        hidden = self.linear1(x)

        # 非线性激活: 形状保持不变 -> [2, 16, 8192]
        hidden = self.act(hidden)

        # 线性降维变换: [2, 16, 8192] @ [8192, 2048] -> [2, 16, 2048]
        out = self.linear2(hidden)

        # 残差连接 (Residual Connection): [2, 16, 2048] + [2, 16, 2048] -> [2, 16, 2048]
        return x + out


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=10000, dim=2048, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # 词嵌入层: 将离散 ID 映射为 dim 维的密集向量
        self.embed = nn.Embedding(vocab_size, dim)
        # 堆叠 num_layers 层 TransformerBlock
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        # 输出头 (Language Model Head): 投影回词表空间
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # 输入 x 形状: [B_local, S] -> [2, 16] (整数 Token ID 矩阵)

        # 1. 词嵌入查表: [2, 16] -> [2, 16, 2048]
        x = self.embed(x)

        # 2. 逐层前向计算 (4层): 每一层输入输出形状均为 [2, 16, 2048]
        for layer in self.layers:
            x = layer(x)

        # 3. 输出线性分类头: [2, 16, 2048] @ [2048, 10000] -> [2, 16, 10000]
        logits = self.head(x)

        return logits  # [B_local, S, vocab_size] -> [2, 16, 10000]


# ==============================================================================
# 2. 简易分词器 (模拟纯文本转张量)
# ==============================================================================
class SimpleTokenizer:
    """简单的 Tokenizer 实现, 演示纯文本到 ID 向量的映射过程"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        # 将每个字符的 Unicode 编码按词表大小取模, 映射为 [0, vocab_size - 1] 的整数
        tokens = [ord(char) % self.vocab_size for char in text]
        # 填充 (Padding) 或截断 (Truncation) 到指定最大序列长度 max_len
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        # 输出一维张量: [S] -> [16]
        return torch.tensor(tokens, dtype=torch.long)


# ==============================================================================
# 3. 2 机 16 卡 DDP 模拟主逻辑
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # Step A: 解析环境变量并初始化分布式环境 (2 机 16 卡)
    # --------------------------------------------------------------------------
    # 由 torchrun 注入的环境变量:
    rank = int(os.environ["RANK"])  # 全局卡号: 0 ~ 15
    local_rank = int(os.environ["LOCAL_RANK"])  # 机器内部卡号: 0 ~ 7
    world_size = int(os.environ["WORLD_SIZE"])  # 全局总卡数: 16

    # 将当前进程绑定到当前物理机的对应 GPU 显卡
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化进程组: 16 个进程通过 TCPStore 完成连接, 建立跨机与机内的 NCCL 通信网络
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"[Cluster Info] 2 机 16 卡分布式通信组初始化成功!")
        print(f"[Cluster Info] 总进程数 (World Size): {world_size}")

    # --------------------------------------------------------------------------
    # Step B: 保证所有 16 张 GPU 上的模型初始权重 100% 相同
    # --------------------------------------------------------------------------
    # 设定固定的随机种子, 确保 Rank 0 ~ Rank 15 初始化出来的权重矩阵完全对齐
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    vocab_size = 10000
    dim = 2048
    num_layers = 4
    seq_len = 16
    local_batch_size = 2
    # 2 台机器 * 每台 8 卡 = 16 卡, 全局总批次大小: 2 * 16 = 32
    global_batch_size = local_batch_size * world_size

    # 实例化模型并迁移至本地 GPU 显存
    model = SimpleLLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------
    # Step C: 数据加载、文本转矩阵与数据并行切片 (Data Sharding)
    # --------------------------------------------------------------------------
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    # 模拟构建 32 条原始纯文本数据 (对应全局 Batch)
    raw_text_dataset = [
        f"这是由第{i}台机器和GPU处理的多机训练文本样本_{i}"
        for i in range(global_batch_size)
    ]

    # 文本转张量: [32, 16] 的整数张量 (Global Input Batch)
    global_input_ids = torch.stack(
        [tokenizer.encode(t, seq_len) for t in raw_text_dataset]
    )
    # 构造自回归下一个 Token 标签 (Shifted Targets): [32, 16]
    global_targets = torch.roll(global_input_ids, shifts=-1, dims=-1).to(device)
    global_input_ids = global_input_ids.to(device)

    # 模拟 DistributedSampler: 计算当前 Rank 对应的样本切片区间
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
    rank = 8
        start_idx = 16
        end_idx = 18
    rank = 9
        start_idx = 18
        end_idx = 20
    rank = 10
        start_idx = 20
        end_idx = 22
    rank = 11
        start_idx = 22
        end_idx = 24
    rank = 12
        start_idx = 24
        end_idx = 26
    rank = 13
        start_idx = 26
        end_idx = 28
    rank = 14
        start_idx = 28
        end_idx = 30
    rank = 15
        start_idx = 30
        end_idx = 32
    """

    # 提取当前 Rank 本地专属的局部批次数据:
    # local_x 形状: [B_local, S] -> [2, 16]
    local_x = global_input_ids[start_idx:end_idx].contiguous()
    # local_y 形状: [B_local, S] -> [2, 16]
    local_y = global_targets[start_idx:end_idx].contiguous()

    # --------------------------------------------------------------------------
    # Step D: 本地独立前向传播 (Local Forward Pass)
    # --------------------------------------------------------------------------
    optimizer.zero_grad()

    # 本地计算, 无通信开销:
    # local_logits 形状: [B_local, S, vocab_size] -> [2, 16, 10000]
    local_logits = model(local_x)

    # --------------------------------------------------------------------------
    # Step E: 原语模拟 1 - dist.all_gather_into_tensor (跨 2 机 16 卡收集前向结果)
    # --------------------------------------------------------------------------
    # 预先分配用于接收 16 张卡聚合结果的高维张量
    # global_gathered_logits 形状: [B_global, S, vocab_size] -> [32, 16, 10000]
    global_gathered_logits = torch.empty(
        (global_batch_size, seq_len, vocab_size),
        dtype=local_logits.dtype,
        device=device,
    )

    # 通信原语: 将 16 张卡上的 local_logits 沿第 0 维拼接到 global_gathered_logits
    # 底层自动执行: 机内 NVLink 汇总 -> 跨机网络交换 -> 广播至全部 16 张卡
    dist.all_gather_into_tensor(global_gathered_logits, local_logits)

    # --------------------------------------------------------------------------
    # Step F: 损失计算与反向传播 (Loss & Backward)
    # --------------------------------------------------------------------------
    # 将局部 logits 与 targets 展平成 2D/1D 计算交叉熵
    # local_logits.view(-1, vocab_size) 形状: [2 * 16, 10000] -> [32, 10000]
    # local_y.view(-1) 形状: [2 * 16] -> [32]
    loss = loss_fn(local_logits.view(-1, vocab_size), local_y.view(-1))

    # 局部反向传播: 计算当前卡上局部批次生成的梯度 (Local Gradients)
    # 生成的 param.grad 仅代表当前卡的局部样本梯度
    loss.backward()

    # --------------------------------------------------------------------------
    # Step G: 原语模拟 2 - dist.all_reduce 梯度全局平均同步 (DDP 的核心)
    # --------------------------------------------------------------------------
    # 遍历所有层参数, 同步梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 1. 跨 2 机 16 卡对梯度张量执行求和归约 (Ring/Tree AllReduce)
            # param.grad.data 形状保持不变 (例如 linear1.weight.grad: [8192, 2048])
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            # 2. 除以全集群总卡数 (16), 求得全局平均梯度
            param.grad.data /= world_size

    # --------------------------------------------------------------------------
    # Step H: 优化器参数更新 (Optimizer Step)
    # --------------------------------------------------------------------------
    # 关键机制: 由于初始权重一致, 且每张卡更新使用的梯度均为全集群对齐的平均梯度,
    # 因此每张 GPU 独立执行 step() 后, 权重依然保持绝对一致, 无需再次同步权重.
    optimizer.step()

    # --------------------------------------------------------------------------
    # Step I: 跨机聚合全局 Loss 并打印
    # --------------------------------------------------------------------------
    global_loss = loss.clone().detach()
    # 跨 16 卡求和 Loss
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    global_loss /= world_size

    # 仅在 Master 节点 (Node 0 的 GPU 0) 进行日志打印
    if rank == 0:
        print("\n" + "=" * 60)
        print(" [Node 0 - Rank 0 运行报告 - 2 机 16 卡手动 DDP]")
        print("=" * 60)
        print(
            f"1. 全局输入张量 (Global Input IDs) 形状 : {list(global_input_ids.shape)}"
        )
        print(f"2. 本地输入切片 (Local x) 形状          : {list(local_x.shape)}")
        print(f"3. 本地模型输出 (Local Logits) 形状     : {list(local_logits.shape)}")
        print(
            f"4. all_gather 聚合后 Logits 形状       : {list(global_gathered_logits.shape)}"
        )
        print(f"5. Rank 0 本地样本 Loss                 : {loss.item():.4f}")
        print(f"6. 全集群 16 卡全局平均 Loss            : {global_loss.item():.4f}")
        print("7. 16 张显卡的模型梯度已通过 dist.all_reduce 完成跨机同步更新!")
        print("=" * 60 + "\n")

    # 释放通信资源
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
