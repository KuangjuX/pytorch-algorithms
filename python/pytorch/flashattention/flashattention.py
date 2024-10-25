import torch
import unittest


class FlashAttention:

    def __init__(self, query, key, value, M, N, K, P, kTM, kTN, kTK, kTP):
        self.M = M
        self.N = N
        self.K = K
        self.P = P

        self.kTM = kTM
        self.kTN = kTN
        self.kTK = kTK
        self.kTP = kTP

        self.query = query
        self.key = key
        self.value = value
        self.output = torch.empty(M, P, device='cpu')

    def forward(self):
        iter_n = self.N // self.kTN

        prev_maxes = torch.zeros(self.M, 1, device='cpu')
        prev_sums = torch.zeros(self.M, 1, device='cpu')

        output = self.output.view(self.M, self.P)

        dK = self.key.view(self.K, self.N)
        dV = self.value.view(self.N, self.P)

        ks = torch.chunk(dK, iter_n, dim=-1)
        vs = torch.chunk(dV, iter_n, dim=-2)

        for n in range(iter_n):
            q = self.query.view(self.M, self.K)  # m * k

            k = ks[n]
            v = vs[n]

            attn_weights = q @ k  # m * ktn

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            exp_weights = torch.exp(attn_weights - cur_maxes)
            # unnormalized attention score @ values
            exp_values = exp_weights @ v
            # move the normalization step to the very end of the attention computation.
            cur_sums = torch.sum(exp_weights, dim=-1, keepdim=True)  # l(x_cur)

            # =======================    renormalization  ======================#
            new_maxes = torch.max(cur_maxes, prev_maxes)  # update m(x)
            # renormalization factor for the previous block
            renorm_prev = torch.exp(prev_maxes - new_maxes)
            # renormalization factor for the current block
            renorm_cur = torch.exp(cur_maxes - new_maxes)

            # update normalization factor l(x)
            new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

            output = (output * prev_sums * renorm_prev +
                      renorm_cur * exp_values) / new_sums

            prev_sums = new_sums
            prev_maxes = new_maxes

        self.output = output

        return self.output


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


if __name__ == '__main__':
    unittest.main()
