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
    基础 Transformer 块 (包含两层 MLP 和 GELU 激活)
    """

    def __init__(self, dim: int):
        super().__init__()
        # 升维映射: [*, dim] -> [*, dim * 4]
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        # 降维映射: [*, dim * 4] -> [*, dim]
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [micro_batch_size, seq_len, dim]
        residual = x
        h = self.linear1(x)  # 形状: [micro_batch_size, seq_len, dim * 4]
        h = self.act(h)  # 形状: [micro_batch_size, seq_len, dim * 4]
        h = self.linear2(h)  # 形状: [micro_batch_size, seq_len, dim]
        out = residual + h  # 残差连接, 形状: [micro_batch_size, seq_len, dim]
        return out


class PipelineStage(nn.Module):
    """
    流水线阶段 (Pipeline Stage): 根据当前 GPU 的 rank 决定分配模型的哪一部分
    总共 8 张卡 (Stage 0 ~ 7), 模型共 8 层 TransformerBlock:
      - Rank 0 (Stage 0): Embedding + Block 0
      - Rank 1~6 (Stage 1~6): Block 1 ~ Block 6 (每卡 1 层)
      - Rank 7 (Stage 7): Block 7 + LM Head (Linear)
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

        # 1. 首阶段持有 Embedding 层
        if self.is_first_stage:
            self.embed = nn.Embedding(vocab_size, dim)

        # 2. 每个阶段持有 1 个 TransformerBlock
        self.block = TransformerBlock(dim)

        # 3. 末阶段持有最后的 LM Head (输出投影层)
        if self.is_last_stage:
            self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
          - Stage 0: x 是 token_ids [micro_batch, seq_len] -> 返回 [micro_batch, seq_len, dim]
          - Stage 1~6: x 是激活值 [micro_batch, seq_len, dim] -> 返回 [micro_batch, seq_len, dim]
          - Stage 7: x 是激活值 [micro_batch, seq_len, dim] -> 返回 Logits [micro_batch, seq_len, vocab_size]
        """
        if self.is_first_stage:
            # 输入 x 形状: [micro_batch_size, seq_len] (整数张量)
            x = self.embed(x)  # 输出形状: [micro_batch_size, seq_len, dim]

        # 经过本卡持有的 TransformerBlock
        # 输入 x 形状: [micro_batch_size, seq_len, dim]
        x = self.block(x)  # 输出形状: [micro_batch_size, seq_len, dim]

        if self.is_last_stage:
            # 经过 LM Head
            # 输入 x 形状: [micro_batch_size, seq_len, dim]
            x = self.head(x)  # 输出形状: [micro_batch_size, seq_len, vocab_size]

        return x


# ==============================================================================
# 2. 文本分词与微批次切分模拟 (Text Processing)
# ==============================================================================


def mock_tokenize_and_batch(
    batch_size: int, seq_len: int, vocab_size: int, num_microbatches: int
):
    """
    模拟将纯文本转为张量矩阵并拆分成 micro-batches
    """
    # 模拟构建 8 句话的 Token 矩阵: 形状 [batch_size, seq_len]
    torch.manual_seed(42)
    input_ids = torch.randint(
        low=1, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long
    )

    # 按照 Batch 维度拆分成 num_microbatches 个小切片
    # 每个 micro_batch 的形状: [micro_batch_size, seq_len], 其中 micro_batch_size = batch_size // num_microbatches
    micro_batches = torch.chunk(input_ids, chunks=num_microbatches, dim=0)
    return micro_batches


# ==============================================================================
# 3. 流水线核心执行调度引擎 (基于底层 P2P 原语 dist.isend / dist.irecv)
# ==============================================================================


def run_pipeline_forward(
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
    执行 GPipe 风格的 1F (Forward-only) 流水线调度
    """
    num_microbatches = len(micro_batches)
    micro_batch_size = micro_batches[0].size(0)

    is_first = rank == 0
    is_last = rank == world_size - 1
    prev_rank = rank - 1
    next_rank = rank + 1

    # 用于保存最后阶段产生的输出结果
    final_outputs = []

    print(
        f"[Rank {rank}] 阶段初始化完成. 准备处理 {num_microbatches} 个 Micro-batches..."
    )

    for mb_idx in range(num_microbatches):
        # ------------------------------------------------------------------
        # Step A: 接收上一阶段传来的数据 (Receive Activation)
        # ------------------------------------------------------------------
        if is_first:
            # Rank 0 直接从 CPU 取出当前 micro-batch 的 Token ID 张量
            # 形状: [micro_batch_size, seq_len]
            input_tensor = micro_batches[mb_idx].to(device)
        else:
            # Rank 1~7 需要预先分配一个形状一致的空张量缓冲区 (Buffer) 来接收激活值
            # 形状: [micro_batch_size, seq_len, dim]
            recv_buffer = torch.empty(
                (micro_batch_size, seq_len, dim), dtype=torch.float32, device=device
            )

            # 使用底层的非阻塞点对点接收原语 dist.irecv
            recv_req = dist.irecv(tensor=recv_buffer, src=prev_rank)
            recv_req.wait()  # 等待通信完成
            input_tensor = recv_buffer
            print(
                f"  -> [Rank {rank}] 成功接收来自 [Rank {prev_rank}] 的 Micro-batch {mb_idx} 激活值, 形状: {list(input_tensor.shape)}"
            )

        # ------------------------------------------------------------------
        # Step B: 本地 Stage 前向计算 (Local Compute)
        # ------------------------------------------------------------------
        # 调用本卡对应的子模型
        output_tensor = stage_model(input_tensor)
        print(
            f"  *  [Rank {rank}] 完成 Micro-batch {mb_idx} 本地计算, 输出张量形状: {list(output_tensor.shape)}"
        )

        # ------------------------------------------------------------------
        # Step C: 将计算结果发送给下一阶段 (Send Activation)
        # ------------------------------------------------------------------
        if not is_last:
            # Rank 0~6 需要将当前阶段输出的激活值发送给下一张卡
            # 待发送张量 output_tensor 形状: [micro_batch_size, seq_len, dim]
            send_req = dist.isend(tensor=output_tensor.contiguous(), dst=next_rank)
            send_req.wait()  # 等待通信发送完成
            print(
                f"  <- [Rank {rank}] 成功发送 Micro-batch {mb_idx} 激活值给 [Rank {next_rank}], 形状: {list(output_tensor.shape)}"
            )
        else:
            # Rank 7 (末阶段) 收集最终的 Logits
            # output_tensor 形状: [micro_batch_size, seq_len, vocab_size]
            final_outputs.append(output_tensor)

    # ----------------------------------------------------------------------
    # Step D: 末阶段拼接微批次, 还原全局 Batch 输出 (Concat Micro-batches)
    # ----------------------------------------------------------------------
    if is_last:
        # 将 4 个 [micro_batch_size, seq_len, vocab_size] 沿 Batch 维度(dim=0)拼接
        # 最终形状: [batch_size, seq_len, vocab_size]
        full_logits = torch.cat(final_outputs, dim=0)
        print(
            f"\n[Rank {rank} (Output Stage)] 全流程执行完毕! 最终输出 Logits 矩阵形状: {list(full_logits.shape)}"
        )
        return full_logits

    return None


# ==============================================================================
# 4. 主程序与通信环境初始化
# ==============================================================================


def main():
    # 1. 获取分布式环境变量 (由 torchrun 自动注入)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2. 绑定当前进程到指定的本地 GPU 设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 3. 初始化默认进程组 (通信后端采用英伟达 NCCL)
    # 这会在 8 张 GPU 之间建立 P2P 通信信道
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # 4. 超参数设置
    VOCAB_SIZE = 10000  # 词表大小
    DIM = 2048  # 隐藏层维度
    SEQ_LEN = 16  # 序列长度
    BATCH_SIZE = 8  # 全局批次大小
    NUM_MICROBATCHES = 4  # 微批次数量 (每个 micro_batch_size = 8 // 4 = 2)

    # 5. 模拟文本分词并切分微批次 (所有进程生成同样的虚拟输入)
    micro_batches = mock_tokenize_and_batch(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_microbatches=NUM_MICROBATCHES,
    )

    # 6. 实例化当前 Rank 所属的流水线切片子模型, 并移动到对应 GPU
    stage_model = PipelineStage(
        stage_id=rank, num_stages=world_size, vocab_size=VOCAB_SIZE, dim=DIM
    ).to(device)

    # 7. 打印各卡角色分工
    if rank == 0:
        print("=================================================================")
        print(f"启动流水线并行模拟 | 总卡数 (Stages): {world_size}")
        print(
            f"全局批次 Batch: {BATCH_SIZE} | 拆分为 {NUM_MICROBATCHES} 个 Micro-batches | 单个大小: {BATCH_SIZE // NUM_MICROBATCHES}"
        )
        print(
            f"序列长度 Seq_Len: {SEQ_LEN} | 隐藏维度 Hidden_Dim: {DIM} | 词表 Vocab: {VOCAB_SIZE}"
        )
        print("=================================================================")
    dist.barrier()  # 等待所有卡打印完毕

    # 8. 启动流水线前向计算
    start_time = time.time()
    logits = run_pipeline_forward(
        stage_model=stage_model,
        micro_batches=micro_batches,
        rank=rank,
        world_size=world_size,
        dim=DIM,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        device=device,
    )
    dist.barrier()  # 同步所有卡

    if rank == 0:
        print(
            f"\n[Success] 8 卡流水线并行前向模拟执行成功! 总耗时: {time.time() - start_time:.4f} 秒"
        )

    # 9. 销毁分布式通信进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
