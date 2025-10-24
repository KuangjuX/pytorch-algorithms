from typing import Optional
import torch
import math

from einops import repeat, rearrange

def construct_streaming_mask(
    seqlen_q: int,
    seqlen_k: int,
    sink_size: int,
    local_size: int,
    is_causal: bool,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cpu'),
):
    """
    构建用于 Streaming Attention 的注意力掩码。

    该掩码结合了“注意力汇聚（sink）”和“局部滑动窗口（local window）”。
    一个查询 token `i` 可以关注：
    1. 序列最开始的 `sink_size` 个 token。
    2. 在其局部窗口内的 `local_size` 个 token（包括自身）。

    Args:
        seqlen_q (int): 查询序列的长度。
        seqlen_k (int): 键序列的长度。
        sink_size (int): 注意力汇聚区的大小。
        local_size (int): 局部滑动窗口的大小。
        is_causal (bool): 是否应用因果约束。
        query_padding_mask (Optional[torch.Tensor]): 查询序列的填充掩码。
        key_padding_mask (Optional[torch.Tensor]): 键序列的填充掩码。
        device (torch.device): 张量所在的设备。

    Returns:
        torch.Tensor: 一个布尔类型的注意力掩码张量，形状为 (seqlen_q, seqlen_k)。
                      `True` 表示该位置需要被掩码（忽略），`False` 表示保留。
    """
    assert sink_size >= 0, "sink_size must be greater than 0"
    assert local_size >= 1, "local_size must be greater than 0"

    # 创建行索引和列索引，用于构建注意力矩阵的坐标网格
    # row_idx 代表查询 token 的位置 (i)
    # rearrange 将其形状从 (s,) 变为 (s, 1)，以便与 col_idx 进行广播操作
    row_idx = rearrange(torch.arange(seqlen_q, device=device), "s -> s 1")

    # col_idx 代表键 token 的位置 (j)
    col_idx = torch.arange(seqlen_k, device=device)

    # 只有在需要因果约束时，才进行复杂的掩码计算
    if is_causal:
        # --- 处理带有填充（Padding）的情况 ---
        # 如果提供了填充掩码，我们需要计算每个序列的实际长度，
        # 因为因果关系和窗口位置是相对于实际 token 而言的，而不是固定的序列长度。
        if query_padding_mask is not None or key_padding_mask is not None:
            # sk 是键序列的实际长度。如果没提供掩码，就用 seqlen_k。
            # 否则，通过对掩码求和得到实际长度 (假设真实 token 为 1，padding 为 0)。
            # rearrange 用于调整维度以支持广播。
            sk = (
                seqlen_k
                if key_padding_mask is None 
                else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            )
            # sq 是查询序列的实际长度，逻辑同上。
            sq = (
                seqlen_q
                if query_padding_mask is None
                else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
            )
        # 处理没有填充的简单情况 (seqlen_q, seqlen_k)
        else:
            sk = seqlen_k
            sq = seqlen_q

        if query_padding_mask is not None or key_padding_mask is not None:
            sk_tensor = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            beyond_causal = col_idx > torch.minimum(row_idx + sk - sq, sk_tensor)
            outside_window = torch.logical_and(
                col_idx < row_idx + sk - sq - (local_size - 1),
                col_idx >= sink_size
            )
        else:
            # 没有填充的简化逻辑
            # 条件1：因果性掩码
            # 键的位置 `j` (col_idx) 不能大于查询的位置 `i` (row_idx)。
            # 这会创建一个上三角矩阵（对角线之上为 True）。
            # beyond_causal 的形状是 (seqlen_q, seqlen_k)
            # 因为 row_idx 的形状是 (seqlen_q, 1)，col_idx 的形状是 (seqlen_k,)
            # 通过广播机制，比较操作会产生 (seqlen_q, seqlen_k) 的布尔张量
            beyond_causal = col_idx > row_idx 

            # 条件2：窗口外掩码
            # `row_idx - (local_size - 1)` 是滑动窗口的左边界。
            # 例如，对于查询 i=7, local_size=3, 窗口是 [5, 6, 7]。左边界是 7-(3-1)=5。
            # 任何 col_idx < 5 的 token 都在窗口之外。
            # `col_idx >= sink_size` 同样是豁免汇聚区的 token。
            outside_window = torch.logical_and(
                col_idx < row_idx - (local_size - 1),
                col_idx >= sink_size
            )
        mask = torch.logical_or(beyond_causal, outside_window)

    # 如果不是因果模式（例如，BERT 那样的编码器），则不应用任何掩码。
    # 创建一个全为 False 的掩码，允许所有 token 相互关注。
    else:
        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    return mask



def block_streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = True,
    dropout_mask: Optional[torch.Tensor] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
):
    """
    实现一个块状的、支持混合模式（密集、流式）的注意力机制。

    Args:
        q, k, v (torch.Tensor): 输入的查询、键、值张量。它们是批处理中所有序列拼接后的结果。
                                形状为 (total_tokens, num_heads, head_dim)。
        cu_seqlens_q, cu_seqlens_k (torch.Tensor): 累积序列长度。
                                                  例如, 对于长度为 [L1, L2] 的批处理, cu_seqlens 为 [0, L1, L1+L2]。
                                                  用于从 q, k, v 中切分出每个序列。
        max_seqlen_q, max_seqlen_k (int): 批处理中最大的查询/键序列长度。
        head_mask_type (torch.Tensor): 一个形状为 (num_heads,) 的张量，决定每个头的注意力类型。
                                       0: Dense Attention
                                       <0: Streaming Attention
                                       >0: Block Sparse Attention (此处未实现)
        streaming_info (torch.Tensor): 一个形状为 (num_heads * 2,) 的扁平化张量，
                                       存储每个头的 [sink_size, local_size] 对。
        p_dropout (float): Dropout 概率。
        softmax_scale (Optional[float]): Softmax 的缩放因子。
        is_causal (bool): 是否应用因果掩码。
        ... (其他参数)
    """
    device = q.device
    total_q, num_heads, head_dim = q.shape
    _, num_heads_k, _ = k.shape
    
    # batch_size 可以从 cu_seqlens 的长度推断出来
    batch_size = cu_seqlens_q.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    attn_weight_lists = [] if return_attn_probs else None

    # --- 2. 处理 GQA/MQA (分组查询注意力/多查询注意力) ---
    # GQA/MQA 是一种优化，多个查询头共享同一组键/值头，以减少 KV 缓存大小
    if num_heads_k != num_heads:
        # 确保查询头数量是键/值头数量的整数倍
        assert num_heads % num_heads_k == 0
        # 使用 einops.repeat 将 K 和 V 的头复制，以匹配 Q 的头数，便于后续计算
        k = repeat(k, "t h d -> t (h g) d", g = num_heads // num_heads_k)
        v = repeat(v, "t h d -> t (h g) d", g = num_heads // num_heads_k)

    output = torch.zeros(total_q, num_heads, head_dim, device=device, dtype=q.dtype)

    # --- 3. 主循环：逐个处理批处理中的序列 ---
    for batch_idx in range(batch_size):
        # 使用累积长度来确定当前序列在拼接张量中的起止位置
        q_start, q_end = cu_seqlens_q[batch_idx].item(), cu_seqlens_q[batch_idx + 1].item()
        k_start, k_end = cu_seqlens_k[batch_idx].item(), cu_seqlens_k[batch_idx + 1].item()

        # 从大张量中切片出当前序列的 q, k, v
        q_batch = q[q_start:q_end] #(seqlen_q, num_heads, head_dim)
        k_batch = k[k_start:k_end]
        v_batch = v[k_start:k_end]

        seqlen_q, seqlen_k = q_batch.shape[0], k_batch.shape[0] # query sequence length, key sequence length

        # --- 4. 计算注意力分数 ---
        # 使用 einsum 高效计算 Q 和 K 的点积，得到原始注意力分数
        # "qhd,khd->hqk" 表示:
        # qhd: (seqlen_q, num_heads, head_dim)
        # khd: (seqlen_k, num_heads, head_dim)
        # hqk: 输出 (num_heads, seqlen_q, seqlen_k)
        scores = torch.einsum("qhd,khd->hqk", q_batch * softmax_scale, k_batch) # (seqlen_q, seqlen_k)

        # 如果有键填充掩码，则将填充位置的分数设置为负无穷
        # 这样在 softmax 后，这些位置的概率会变为 0
        if key_padding_mask is not None:
            key_mask = key_padding_mask[batch_idx, :seqlen_k].to(device)
            scores = scores.masked_fill(~key_mask(1, 1, -1), float('-inf'))

        for head_idx in range(num_heads):
            mask_type = head_mask_type[head_idx].item()

            if mask_type == 0: 
                # Dense Attention
                if is_causal:
                    causal_mask = torch.triu(torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device), diagonal=1)
                    scores[head_idx].masked_fill_(causal_mask, float('-inf'))

            elif mask_type < 0:
                # Streaming Attention
                # 模式 < 0: 流式注意力 (Streaming Attention)
                # 从 streaming_info 中获取该头的 sink_size 和 local_size
                sink_size = streaming_info[head_idx * 2].item()
                local_size = streaming_info[head_idx * 2 + 1].item()

                streaming_mask = construct_streaming_mask(
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    sink_size=sink_size,
                    local_size=local_size,
                    is_causal=is_causal,
                    query_padding_mask=query_padding_mask,
                    key_padding_mask=key_padding_mask,
                    device=device
                )

                scores[head_idx].masked_fill_(streaming_mask, float('-inf'))

            elif mask_type > 0:
                    # 这里表示 Block Sparse Attention，暂时不考虑
                    pass

        attn = torch.softmax(scores, dim=-1).to(v_batch.dtype)

        if query_padding_mask is not None:
            query_mask = query_padding_mask[batch_idx, :seqlen_q].to(device)
            attn = attn.masked_fill(~query_mask(1, -1, 1), 0.0)

        # 应用 dropout
        if p_dropout > 0.0:
            if dropout_mask is not None:
                # 使用预定义的 dropout mask（用于测试）
                attn_drop = attn.masked_fill(
                    ~dropout_mask[batch_idx, :, :seqlen_q, :seqlen_k], 0.0
                )
            else:
                # 随机 dropout
                drop_mask = torch.rand_like(attn) > p_dropout
                attn_drop = attn.masked_fill(~drop_mask, 0.0)
            dropout_scale = 1.0 / (1.0 - p_dropout)
        else:
            attn_drop = attn
            dropout_scale = 1.0

        out_batch = torch.einsum("hqk,khd->qhd", attn_drop, v_batch * dropout_scale)

        if query_padding_mask is not None:
            query_mask = query_padding_mask[batch_idx, :seqlen_q].to(device)
            out_batch = out_batch.masked_fill(~query_mask(-1, 1, 1), 0.0)

        output[q_start:q_end] = out_batch

        if return_attn_probs:
            attn_weight_lists.append(attn)

    if return_attn_probs:
        return output, attn_weight_lists
    else:
        return output, None