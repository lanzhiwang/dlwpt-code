import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn

# ==============================================================================
# 1. 基础模型组件定义
# ==============================================================================


class TransformerBlock(nn.Module):
    """
    基础 Transformer 计算块 (含两层 MLP 投影与 GELU 激活)
    """

    def __init__(self, dim: int):
        super().__init__()
        # 升维线性变换: [*, dim] -> [*, dim * 4]
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        # 降维线性变换: [*, dim * 4] -> [*, dim]
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [micro_batch_size, seq_len, dim]
        residual = x
        h = self.linear1(x)  # 形状: [micro_batch_size, seq_len, dim * 4]
        h = self.act(h)  # 形状: [micro_batch_size, seq_len, dim * 4]
        h = self.linear2(h)  # 形状: [micro_batch_size, seq_len, dim]
        out = residual + h  # 残差相加: [micro_batch_size, seq_len, dim]
        return out


class PipelineStage(nn.Module):
    """
    16 阶段流水线中的单卡模型切片:
      - Rank 0 (Node 0, GPU 0): Embedding + Block 0
      - Rank 1~14 (Node 0/Node 1): Block 1 ~ Block 14 (每卡 1 层)
      - Rank 15 (Node 1, GPU 7): Block 15 + LM Head
    """

    def __init__(
        self, stage_id: int, num_stages: int, vocab_size: int = 10000, dim: int = 2048
    ):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.dim = dim
        self.vocab_size = vocab_size

        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1

        # 1. 只有首卡 Rank 0 初始化 Embedding 词表层
        if self.is_first_stage:
            self.embed = nn.Embedding(vocab_size, dim)

        # 2. 每张卡分配 1 个 TransformerBlock
        self.block = TransformerBlock(dim)

        # 3. 只有末卡 Rank 15 初始化 LM Head (输出分类头)
        if self.is_last_stage:
            self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
          - Rank 0 输入为 Token IDs [micro_batch, seq_len] -> 输出 [micro_batch, seq_len, dim]
          - Rank 1~14 输入为激活值 [micro_batch, seq_len, dim] -> 输出 [micro_batch, seq_len, dim]
          - Rank 15 输入为激活值 [micro_batch, seq_len, dim] -> 输出 Logits [micro_batch, seq_len, vocab_size]
        """
        if self.is_first_stage:
            # 输入 x 形状: [micro_batch_size, seq_len] (整数张量)
            x = self.embed(x)  # 形状: [micro_batch_size, seq_len, dim]

        # 经过本卡持有的 Transformer 块
        # 输入 x 形状: [micro_batch_size, seq_len, dim]
        x = self.block(x)  # 形状: [micro_batch_size, seq_len, dim]

        if self.is_last_stage:
            # 经过 LM Head 线性分类
            # 输入 x 形状: [micro_batch_size, seq_len, dim]
            x = self.head(x)  # 形状: [micro_batch_size, seq_len, vocab_size]

        return x


# ==============================================================================
# 2. 文本矩阵化与微批次切分模拟 (Text Processing)
# ==============================================================================


def mock_text_tokenization_and_chunking(
    batch_size: int, seq_len: int, vocab_size: int, num_microbatches: int
):
    """
    模拟分词后的文本矩阵生成, 并切分成 micro-batches
    """
    # 模拟构建整个全局批次的 Token 矩阵: 形状 [batch_size, seq_len]
    torch.manual_seed(42)
    input_ids = torch.randint(
        low=1, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long
    )

    # 沿批次维度拆分成 num_microbatches 份
    # 每个 micro_batch 的形状: [micro_batch_size, seq_len] (其中 micro_batch_size = batch_size // num_microbatches)
    micro_batches = torch.chunk(input_ids, chunks=num_microbatches, dim=0)
    return micro_batches


# ==============================================================================
# 3. 2 机 16 卡流水线执行引擎 (底层通信原语调度)
# ==============================================================================


def run_pipeline_forward_16gpus(
    stage_model: PipelineStage,
    micro_batches: list,
    rank: int,
    world_size: int,
    dim: int,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
):
    """
    执行 16 卡跨机流水线前向传递
    """
    num_microbatches = len(micro_batches)
    micro_batch_size = micro_batches[0].size(0)

    is_first = rank == 0
    is_last = rank == world_size - 1
    prev_rank = rank - 1
    next_rank = rank + 1
    node_id = rank // 8  # 0~7 属于 Node 0, 8~15 属于 Node 1

    final_outputs = []

    for mb_idx in range(num_microbatches):
        # ----------------------------------------------------------------------
        # Step A: 接收上一阶段传来的数据 (P2P Receive)
        # ----------------------------------------------------------------------
        if is_first:
            # Rank 0 读取当前 micro-batch 的 Token ID 数据
            # 形状: [micro_batch_size, seq_len]
            input_tensor = micro_batches[mb_idx].to(device)
        else:
            # Rank 1~15 预分配用于接收激活值的缓冲区 Buffer
            # 形状: [micro_batch_size, seq_len, dim]
            recv_buffer = torch.empty(
                (micro_batch_size, seq_len, dim), dtype=torch.float32, device=device
            )

            # 使用非阻塞底层点对点接收原语 dist.irecv
            recv_req = dist.irecv(tensor=recv_buffer, src=prev_rank)
            recv_req.wait()  # 等待通信接收完成
            input_tensor = recv_buffer

            # 跨机边界高亮标识 (Rank 7 -> Rank 8 为跨机传输)
            comm_type = "[跨机通信: Node 0 -> Node 1]" if rank == 8 else "[机内通信]"
            print(
                f"  -> [Node {node_id} | Rank {rank:02d}] {comm_type} 成功接收来自 Rank {prev_rank:02d} 的 Micro-batch {mb_idx}, 形状: {list(input_tensor.shape)}"
            )

        # ----------------------------------------------------------------------
        # Step B: 本地 Stage 前向计算 (Local Compute)
        # ----------------------------------------------------------------------
        # 计算当前切片层的输出
        output_tensor = stage_model(input_tensor)
        print(
            f"  *  [Node {node_id} | Rank {rank:02d}] 完成 Micro-batch {mb_idx} 计算, 输出形状: {list(output_tensor.shape)}"
        )

        # ----------------------------------------------------------------------
        # Step C: 发送计算结果至下一阶段 (P2P Send)
        # ----------------------------------------------------------------------
        if not is_last:
            # Rank 0~14 将激活值发送给下一个 Rank
            # 待发送张量形状: [micro_batch_size, seq_len, dim]
            send_req = dist.isend(tensor=output_tensor.contiguous(), dst=next_rank)
            send_req.wait()  # 等待发送完成

            comm_type = "[跨机通信: Node 0 -> Node 1]" if rank == 7 else "[机内通信]"
            print(
                f"  <- [Node {node_id} | Rank {rank:02d}] {comm_type} 成功发送 Micro-batch {mb_idx} 给 Rank {next_rank:02d}, 形状: {list(output_tensor.shape)}"
            )
        else:
            # Rank 15 (末阶段) 收集各 micro-batch 的 Logits
            # output_tensor 形状: [micro_batch_size, seq_len, vocab_size]
            final_outputs.append(output_tensor)

    # --------------------------------------------------------------------------
    # Step D: 末阶段聚合与全局同步验证 (Concat & all_gather 原语演示)
    # --------------------------------------------------------------------------
    if is_last:
        # 将 8 个 [micro_batch_size, seq_len, vocab_size] 沿 batch 维度拼接
        # 最终形状: [batch_size, seq_len, vocab_size]
        full_logits = torch.cat(final_outputs, dim=0)
        print(
            f"\n[Node {node_id} | Rank {rank:02d} (末节点)] 所有微批次处理完毕! 拼接后的全局 Logits 矩阵形状: {list(full_logits.shape)}"
        )

        # 模拟计算一个虚拟标量损失用于全网验证
        loss = full_logits.mean().unsqueeze(0)  # 形状: [1]
    else:
        loss = torch.zeros(1, device=device)  # 形状: [1]

    # 使用底层集合通信原语 dist.all_gather_into_tensor 将结果标量同步到全部 16 张卡
    # 接收容器 total_loss_tensor 形状: [16]
    gathered_loss = torch.empty(world_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(output_tensor=gathered_loss, input_tensor=loss)

    if rank == 0:
        print(
            f"\n[Node 0 | Rank 00] 通过 dist.all_gather_into_tensor 成功验证 16 卡全局同步! 汇总数据: {gathered_loss.cpu().numpy()}"
        )


# ==============================================================================
# 4. 多机多卡环境初始化与主入口
# ==============================================================================


def main():
    # 1. 读取由 torchrun 注入的多机环境变量
    rank = int(os.environ["RANK"])  # 全局 Rank: 0 ~ 15
    world_size = int(os.environ["WORLD_SIZE"])  # 全局卡数: 16
    local_rank = int(os.environ["LOCAL_RANK"])  # 节点内 GPU 编号: 0 ~ 7
    node_id = rank // 8  # 所属物理机编号: 0 或 1

    # 2. 绑定当前进程到本机对应的显卡设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 3. 初始化跨机进程组通信 (NCCL 后端)
    # 会自动读取 MASTER_ADDR 和 MASTER_PORT 建立跨机 Socket 与 NCCL Ring/Tree
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # 4. 模型与批次超参数设置
    VOCAB_SIZE = 10000  # 词表大小
    DIM = 2048  # 隐藏层维度
    SEQ_LEN = 32  # 序列长度
    BATCH_SIZE = 16  # 全局批次大小
    NUM_MICROBATCHES = 8  # 微批次数量 (每个 micro_batch_size = 16 // 8 = 2)

    # 5. 模拟生成文本 Token 矩阵
    micro_batches = mock_text_tokenization_and_chunking(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_microbatches=NUM_MICROBATCHES,
    )

    # 6. 初始化当前卡分配到的流水线阶段模型
    stage_model = PipelineStage(
        stage_id=rank, num_stages=world_size, vocab_size=VOCAB_SIZE, dim=DIM
    ).to(device)

    # 7. 打印环境与拓扑信息
    if rank == 0:
        print(
            "=============================================================================="
        )
        print(f" 启动 2 机 16 卡流水线并行模拟 | 总卡数 (Stages): {world_size}")
        print(f" 节点拓扑: Node 0 (Rank 00~07) <==Network==> Node 1 (Rank 08~15)")
        print(
            f" 全局批次: {BATCH_SIZE} | 切分为 {NUM_MICROBATCHES} 个 Micro-batches | 单批次大小: {BATCH_SIZE // NUM_MICROBATCHES}"
        )
        print(
            f" 序列长度 Seq_Len: {SEQ_LEN} | 隐藏维度 Hidden_Dim: {DIM} | 词表 Vocab: {VOCAB_SIZE}"
        )
        print(
            "=============================================================================="
        )
    dist.barrier()  # 等待所有卡对齐

    # 8. 执行流水线前向传播
    start_time = time.time()
    run_pipeline_forward_16gpus(
        stage_model=stage_model,
        micro_batches=micro_batches,
        rank=rank,
        world_size=world_size,
        dim=DIM,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        device=device,
    )
    dist.barrier()  # 同步等待全流水线结束

    if rank == 0:
        print(
            f"\n[Success] 2 机 16 卡流水线并行前向模拟执行成功! 总耗时: {time.time() - start_time:.4f} 秒"
        )

    # 9. 销毁通信进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
