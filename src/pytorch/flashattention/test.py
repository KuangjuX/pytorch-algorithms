from flashattention import FlashAttention

import unittest
import torch

class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def run_flash_attention(self, Q, K, V, m, n, k, p, kTM, kTN, kTK, kTP):

        flash_attn = FlashAttention(Q.half().flatten(),
                                    K.half().flatten(),
                                    V.half().flatten(), m, n, k, p, kTM, kTN,
                                    kTK, kTP)

        torch_o = flash_attn.forward().half()

        print(torch_o)
        
    def run_flash_attention_lse(self, Q, K, V, m, n, k, p, kTM, kTN, kTK, kTP):
    
            flash_attn = FlashAttention(Q.half().flatten(),
                                        K.half().flatten(),
                                        V.half().flatten(), m, n, k, p, kTM, kTN,
                                        kTK, kTP)
    
            torch_o = flash_attn.forward_lse().half()
    
            print(torch_o)

    def test_flash_attention_v0(self):
        M = 64
        N = 64
        K = 128
        P = 128

        kTM = 64
        kTN = 64
        kTK = 128
        kTP = 128
        
        query = torch.randn(M, K, device='cpu')
        key = torch.randn(K, N, device='cpu')
        value = torch.randn(N, P, device='cpu')

        self.run_flash_attention(query, key, value, M, N, K, P, kTM, kTN, kTK, kTP) 
        self.run_flash_attention_lse(query, key, value, M, N, K, P, kTM, kTN, kTK, kTP)
        
    def test_flash_attention_v1(self):
        M = 128
        N = 128
        K = 256
        P = 256

        kTM = 64
        kTN = 64
        kTK = 128
        kTP = 128
        
        query = torch.randn(M, K, device='cpu')
        key = torch.randn(K, N, device='cpu')
        value = torch.randn(N, P, device='cpu')

        self.run_flash_attention(query, key, value, M, N, K, P, kTM, kTN, kTK, kTP) 
        self.run_flash_attention_lse(query, key, value, M, N, K, P, kTM, kTN, kTK, kTP)


if __name__ == '__main__':
    unittest.main()