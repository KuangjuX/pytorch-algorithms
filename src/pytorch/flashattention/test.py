from flashattention import FlashAttention
from flashdecoding import FlashDecoding

import unittest
import torch

class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def run_flash_attention(self, Q, K, V, m, n, k, p, ChunkN):

        flash_attn = FlashAttention(Q.half().flatten(),
                                    K.half().flatten(),
                                    V.half().flatten(), m, n, k, p, ChunkN)

        torch_o = flash_attn.forward().half()

        return torch_o
        
    def run_flash_attention_lse(self, Q, K, V, m, n, k, p, ChunkN):
    
        flash_attn = FlashAttention(Q.half().flatten(),
                                    K.half().flatten(),
                                    V.half().flatten(), m, n, k, p, ChunkN)

        torch_o = flash_attn.forward_lse().half()

        return torch_o
        
    def run_flash_decoding(self, Q, K, V, m, n, k, p, ChunkN, SplitN):
        
        flash_decoding = FlashDecoding(Q.half().flatten(),
                                       K.half().flatten(),
                                       V.half().flatten(), m, n, k, p, ChunkN, SplitN)
        
        torch_o = flash_decoding.forward().half()
        
        return torch_o

    def test_flash_attention_v0(self, atol=5e-2):
        M = 64
        N = 64
        K = 128
        P = 128

        ChunkN = 64
        
        query = torch.randn(M, K, device='cpu')
        key = torch.randn(K, N, device='cpu')
        value = torch.randn(N, P, device='cpu')

        flash_attention_o = self.run_flash_attention(query, key, value, M, N, K, P, ChunkN) 
        flash_attention_lse_o = self.run_flash_attention_lse(query, key, value, M, N, K, P, ChunkN)
        
        # Compare the outputs of the forward and forward_lse.       
        err_sum = 0.0
        for i in range(0, M):
            for j in range(0, P):
                err_sum += (flash_attention_o[i][j] - flash_attention_lse_o[i][j]).abs().item()
        
        avarage_error = err_sum / (M * P)
        
        assert avarage_error < atol
        
    def test_flash_attention_v1(self, atol=5e-2):
        M = 128
        N = 128
        K = 256
        P = 256

        ChunkN = 64
        
        query = torch.randn(M, K, device='cpu')
        key = torch.randn(K, N, device='cpu')
        value = torch.randn(N, P, device='cpu')

        flash_attention_o = self.run_flash_attention(query, key, value, M, N, K, P, ChunkN) 
        flash_attention_lse_o = self.run_flash_attention_lse(query, key, value, M, N, K, P, ChunkN)
        
        err_sum = 0.0
        for i in range(0, M):
            for j in range(0, P):
                err_sum += (flash_attention_o[i][j] - flash_attention_lse_o[i][j]).abs().item()
        
        avarage_error = err_sum / (M * P)
        
        assert avarage_error < atol
        
    def test_flash_attention_v2(self, atol=5e-2):
        M = 256
        N = 256
        K = 512
        P = 512
        
        ChunkN = 128
        SplitN = 64
        
        query = torch.randn(M, K, device='cpu')
        key = torch.randn(K, N, device='cpu')
        value = torch.randn(N, P, device='cpu')

        flash_attention_o = self.run_flash_attention(query, key, value, M, N, K, P, ChunkN) 
        flash_decoding_o = self.run_flash_decoding(query, key, value, M, N, K, P, ChunkN, SplitN)
        
        # Compare the outputs of the flash_attention and flash_decoding.
        err_sum = 0.0
        for i in range(0, M):
            for j in range(0, P):
                err_sum += (flash_attention_o[i][j] - flash_decoding_o[i][j]).abs().item()
        
        avarage_error = err_sum / (M * P)
        
        assert avarage_error < atol


if __name__ == '__main__':
    unittest.main()