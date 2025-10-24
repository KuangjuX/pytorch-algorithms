import torch
import unittest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch.duoattention.block_streaming_attention import block_streaming_attention


class TestBlockStreamingAttention(unittest.TestCase):
    """测试 block_streaming_attention 函数"""

    def setUp(self):
        """设置测试环境"""
        self.device = torch.device('cpu')
        torch.manual_seed(42)  # 设置随机种子以保证可重复性

    def test_single_sequence_dense_attention(self):
        """测试单个序列的密集注意力（Dense Attention）"""
        # 设置参数
        batch_size = 1
        seqlen = 8
        num_heads = 4
        head_dim = 16
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度：单个序列，长度为 seqlen
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 所有头都使用密集注意力（mask_type = 0）
        head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=self.device)
        
        # streaming_info 在密集注意力模式下不使用，但仍需提供
        streaming_info = torch.zeros(num_heads * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数
        output, attn_weights = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")

    def test_single_sequence_streaming_attention(self):
        """测试单个序列的流式注意力（Streaming Attention）"""
        # 设置参数
        batch_size = 1
        seqlen = 16
        num_heads = 2
        head_dim = 32
        sink_size = 2
        local_size = 4
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 所有头都使用流式注意力（mask_type < 0）
        head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=self.device)
        
        # 设置 streaming_info：每个头的 [sink_size, local_size]
        streaming_info = torch.tensor(
            [sink_size, local_size] * num_heads, 
            dtype=torch.int32, 
            device=self.device
        )
        
        # 调用函数
        output, attn_weights = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")

    def test_mixed_attention_heads(self):
        """测试混合注意力头：部分密集，部分流式"""
        # 设置参数
        seqlen = 12
        num_heads = 4
        head_dim = 16
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 混合模式：前两个头密集注意力，后两个头流式注意力
        head_mask_type = torch.tensor([0, 0, -1, -1], dtype=torch.int32, device=self.device)
        
        # streaming_info：为所有头提供信息，但只有流式注意力头会使用
        streaming_info = torch.tensor(
            [2, 3,  # 头 0 (不使用)
             2, 3,  # 头 1 (不使用)
             2, 4,  # 头 2 (流式：sink=2, local=4)
             1, 3], # 头 3 (流式：sink=1, local=3)
            dtype=torch.int32,
            device=self.device
        )
        
        # 调用函数
        output, attn_weights = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")

    def test_batch_sequences(self):
        """测试批处理多个序列"""
        # 设置参数
        batch_size = 3
        seqlens = [5, 8, 6]  # 三个不同长度的序列
        num_heads = 2
        head_dim = 16
        
        # 将所有序列拼接成一个大张量
        total_len = sum(seqlens)
        q = torch.randn(total_len, num_heads, head_dim, device=self.device)
        k = torch.randn(total_len, num_heads, head_dim, device=self.device)
        v = torch.randn(total_len, num_heads, head_dim, device=self.device)
        
        # 构建累积序列长度
        cu_seqlens_q = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(batch_size)], 
                                     dtype=torch.int32, device=self.device)
        cu_seqlens_k = cu_seqlens_q.clone()
        
        # 使用密集注意力
        head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=self.device)
        streaming_info = torch.zeros(num_heads * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数
        output, attn_weights = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(seqlens),
            max_seqlen_k=max(seqlens),
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (total_len, num_heads, head_dim))
        
        # 检查每个序列的输出
        for i in range(batch_size):
            start = cu_seqlens_q[i].item()
            end = cu_seqlens_q[i + 1].item()
            seq_output = output[start:end]
            
            # 检查形状
            self.assertEqual(seq_output.shape[0], seqlens[i])
            
            # 检查不包含 NaN 或 Inf
            self.assertFalse(torch.isnan(seq_output).any(), 
                           f"序列 {i} 的输出包含 NaN")
            self.assertFalse(torch.isinf(seq_output).any(), 
                           f"序列 {i} 的输出包含 Inf")

    def test_return_attention_probs(self):
        """测试返回注意力概率"""
        # 设置参数
        seqlen = 6
        num_heads = 2
        head_dim = 8
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 使用密集注意力
        head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=self.device)
        streaming_info = torch.zeros(num_heads * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数并返回注意力权重
        output, attn_weights = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=True
        )
        
        # 检查返回的注意力权重
        self.assertIsNotNone(attn_weights, "应该返回注意力权重")
        self.assertEqual(len(attn_weights), 1, "应该有一个批次的注意力权重")
        
        # 检查注意力权重的形状
        attn = attn_weights[0]
        self.assertEqual(attn.shape, (num_heads, seqlen, seqlen))
        
        # 检查注意力权重的性质：每行的和应该约等于 1（softmax 的性质）
        attn_sum = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5),
                       "注意力权重每行的和应该为 1")

    def test_non_causal_mode(self):
        """测试非因果模式"""
        # 设置参数
        seqlen = 8
        num_heads = 2
        head_dim = 16
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 使用密集注意力
        head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=self.device)
        streaming_info = torch.zeros(num_heads * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数（非因果模式）
        output, _ = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=False,  # 非因果模式
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")

    def test_custom_softmax_scale(self):
        """测试自定义 softmax 缩放因子"""
        # 设置参数
        seqlen = 8
        num_heads = 2
        head_dim = 16
        custom_scale = 0.5
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 使用密集注意力
        head_mask_type = torch.zeros(num_heads, dtype=torch.int32, device=self.device)
        streaming_info = torch.zeros(num_heads * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数（使用自定义缩放因子）
        output, _ = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            softmax_scale=custom_scale,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")

    def test_gqa_mode(self):
        """测试分组查询注意力（GQA）模式"""
        # 设置参数
        seqlen = 8
        num_heads_q = 8  # 查询头数
        num_heads_kv = 2  # 键值头数（分组）
        head_dim = 16
        
        # 创建输入张量
        q = torch.randn(seqlen, num_heads_q, head_dim, device=self.device)
        k = torch.randn(seqlen, num_heads_kv, head_dim, device=self.device)
        v = torch.randn(seqlen, num_heads_kv, head_dim, device=self.device)
        
        # 累积序列长度
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=self.device)
        
        # 使用密集注意力
        head_mask_type = torch.zeros(num_heads_q, dtype=torch.int32, device=self.device)
        streaming_info = torch.zeros(num_heads_q * 2, dtype=torch.int32, device=self.device)
        
        # 调用函数
        output, _ = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            return_attn_probs=False
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (seqlen, num_heads_q, head_dim))
        
        # 检查输出不包含 NaN 或 Inf
        self.assertFalse(torch.isnan(output).any(), "输出包含 NaN")
        self.assertFalse(torch.isinf(output).any(), "输出包含 Inf")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

