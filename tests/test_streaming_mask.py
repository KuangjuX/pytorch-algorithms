import torch
import unittest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch.duoattention.block_streaming_attention import construct_streaming_mask


class TestConstructStreamingMask(unittest.TestCase):
    """测试 construct_streaming_mask 函数"""

    def setUp(self):
        """设置测试环境"""
        self.device = torch.device('cpu')

    def test_basic_causal_mask(self):
        """测试基本的因果掩码，不带 sink 和 window"""
        seqlen_q = 5
        seqlen_k = 5
        sink_size = 0
        local_size = 5  # 窗口大小等于序列长度，实际上没有窗口限制
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 检查形状
        self.assertEqual(mask.shape, (seqlen_q, seqlen_k))
        
        # 检查因果性：上三角应该被掩码（True）
        for i in range(seqlen_q):
            for j in range(seqlen_k):
                if j > i:
                    self.assertTrue(mask[i, j].item(), 
                                  f"位置 ({i}, {j}) 应该被掩码")
                else:
                    self.assertFalse(mask[i, j].item(), 
                                   f"位置 ({i}, {j}) 不应该被掩码")

    def test_sink_attention(self):
        """测试 sink attention：序列开头的 token 总是可见"""
        seqlen_q = 8
        seqlen_k = 8
        sink_size = 2  # 前 2 个 token 作为 sink
        local_size = 2  # 局部窗口大小为 2
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 对于任意查询位置 i，前 sink_size 个 key 应该总是可见（不被掩码）
        for i in range(seqlen_q):
            for j in range(sink_size):
                if j <= i:  # 同时满足因果约束
                    self.assertFalse(mask[i, j].item(), 
                                   f"Sink token ({i}, {j}) 不应该被掩码")

    def test_local_window(self):
        """测试局部滑动窗口"""
        seqlen_q = 10
        seqlen_k = 10
        sink_size = 2
        local_size = 3  # 每个 query 只能看到最近的 3 个 key
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 检查查询位置 7
        i = 7
        # 应该能看到：sink [0, 1] 和 local window [5, 6, 7]
        visible_positions = [0, 1, 5, 6, 7]
        
        for j in range(seqlen_k):
            if j in visible_positions:
                self.assertFalse(mask[i, j].item(), 
                               f"位置 ({i}, {j}) 应该可见")
            else:
                self.assertTrue(mask[i, j].item(), 
                              f"位置 ({i}, {j}) 应该被掩码")

    def test_non_causal_mode(self):
        """测试非因果模式（如 BERT）"""
        seqlen_q = 5
        seqlen_k = 5
        sink_size = 1
        local_size = 2
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=False,  # 非因果模式
            device=self.device
        )
        
        # 在非因果模式下，所有位置都应该可见（全为 False）
        self.assertTrue(torch.all(~mask), 
                       "非因果模式下，所有位置都应该可见")

    def test_edge_case_first_few_queries(self):
        """测试序列开头的查询 token"""
        seqlen_q = 6
        seqlen_k = 6
        sink_size = 2
        local_size = 3
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 查询位置 0：只能看到自己
        self.assertFalse(mask[0, 0].item())
        for j in range(1, seqlen_k):
            self.assertTrue(mask[0, j].item())
        
        # 查询位置 1：可以看到 [0, 1]
        for j in [0, 1]:
            self.assertFalse(mask[1, j].item())
        for j in range(2, seqlen_k):
            self.assertTrue(mask[1, j].item())
        
        # 查询位置 2：可以看到 sink [0, 1] 和 local [0, 1, 2]
        # 合并后是 [0, 1, 2]
        for j in [0, 1, 2]:
            self.assertFalse(mask[2, j].item())
        for j in range(3, seqlen_k):
            self.assertTrue(mask[2, j].item())

    def test_long_sequence(self):
        """测试较长序列"""
        seqlen_q = 100
        seqlen_k = 100
        sink_size = 5
        local_size = 10
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 检查中间某个位置，例如位置 50
        i = 50
        # 应该能看到：sink [0, 1, 2, 3, 4] 和 local window [41, 42, ..., 50]
        visible = set(range(sink_size)) | set(range(i - local_size + 1, i + 1))
        
        for j in range(seqlen_k):
            if j in visible:
                self.assertFalse(mask[i, j].item(), 
                               f"位置 ({i}, {j}) 应该可见")
            else:
                self.assertTrue(mask[i, j].item(), 
                              f"位置 ({i}, {j}) 应该被掩码")

    def test_different_qk_lengths(self):
        """测试不同的 query 和 key 序列长度"""
        seqlen_q = 5
        seqlen_k = 8
        sink_size = 2
        local_size = 3
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 检查形状
        self.assertEqual(mask.shape, (seqlen_q, seqlen_k))
        
        # 基本的因果性检查
        self.assertTrue(mask[0, 1].item(), "应该满足因果约束")

    def test_minimal_local_size(self):
        """测试最小的 local_size = 1"""
        seqlen_q = 5
        seqlen_k = 5
        sink_size = 1
        local_size = 1  # 每个 query 只能看到自己
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 查询位置 3：应该能看到 sink [0] 和自己 [3]
        self.assertFalse(mask[3, 0].item(), "应该能看到 sink")
        self.assertFalse(mask[3, 3].item(), "应该能看到自己")
        self.assertTrue(mask[3, 1].item(), "不应该看到窗口外的 token")
        self.assertTrue(mask[3, 2].item(), "不应该看到窗口外的 token")

    def test_zero_sink_size(self):
        """测试 sink_size = 0 的情况"""
        seqlen_q = 6
        seqlen_k = 6
        sink_size = 0
        local_size = 3
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 查询位置 5：应该只能看到 local window [3, 4, 5]
        for j in [3, 4, 5]:
            self.assertFalse(mask[5, j].item(), 
                           f"位置 (5, {j}) 应该可见")
        for j in [0, 1, 2]:
            self.assertTrue(mask[5, j].item(), 
                          f"位置 (5, {j}) 应该被掩码")

    def test_dtype_and_device(self):
        """测试返回的掩码类型和设备"""
        seqlen_q = 4
        seqlen_k = 4
        sink_size = 1
        local_size = 2
        
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            device=self.device
        )
        
        # 检查数据类型
        self.assertEqual(mask.dtype, torch.bool, 
                        "掩码应该是布尔类型")
        
        # 检查设备
        self.assertEqual(mask.device, self.device, 
                        "掩码应该在指定的设备上")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

