from flashattention import FlashAttention

import unittest
import torch

class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    def run_flash_attention(self, m, n, k, p, kTM, kTN, kTK, kTP):

        Q = torch.randn(m, k, device='cpu')
        K = torch.randn(k, n, device='cpu')
        V = torch.randn(n, p, device='cpu')

        flash_attn = FlashAttention(Q.half().flatten(),
                                    K.half().flatten(),
                                    V.half().flatten(), m, n, k, p, kTM, kTN,
                                    kTK, kTP)

        torch_o = flash_attn.forward().half()

        print(torch_o)
        
    def run_flash_attention_lse(self, m, n, k, p, kTM, kTN, kTK, kTP):
            
            Q = torch.randn(m, k, device='cpu')
            K = torch.randn(k, n, device='cpu')
            V = torch.randn(n, p, device='cpu')
    
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

        self.run_flash_attention(M, N, K, P, kTM, kTN, kTK, kTP) 
        self.run_flash_attention_lse(M, N, K, P, kTM, kTN, kTK, kTP)


if __name__ == '__main__':
    unittest.main()