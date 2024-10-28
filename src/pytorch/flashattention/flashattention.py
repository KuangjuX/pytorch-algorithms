import torch
import unittest


class FlashAttention:

    def __init__(self, query, key, value, M, N, K, P, kTM, kTN, kTK, kTP, device = 'cpu'):
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
        
        self.device = device

    def forward(self):
        loop_n = self.N // self.kTN

        prev_maxes = torch.zeros(self.M, 1, device=self.device)
        prev_sums = torch.zeros(self.M, 1, device=self.device)

        output = torch.empty(self.M, self.P, device=self.device)

        dK = self.key.view(self.K, self.N)
        dV = self.value.view(self.N, self.P)

        ks = torch.chunk(dK, loop_n, dim=-1)
        vs = torch.chunk(dV, loop_n, dim=-2)

        for n in range(loop_n):
            q = self.query.view(self.M, self.K)  # m * k

            k = ks[n]
            v = vs[n]

            attn_weights = q @ k  # m * ktn

            # reduce maxes
            cur_maxes, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            # print('current block max:', cur_maxes)
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

        return output
    
    def forward_lse(self):
        loop_n = self.N // self.kTN
        
        # The LogSumExp(LSE) is a smooth maximum,
        # LSE(x1,...,xn) = log(exp(x1)+...+exp(xn))
        # = c + log(exp(x1-c)+...+exp(xn-c))
        # c = max(x1,...,xn)
        lse = torch.full((self.M, 1), float('-inf'), device=self.device)
        # prev_maxes: maximun up to the previous block.
        prev_maxes = torch.full((self.M, 1), float('-inf'), device=self.device)
        output = torch.empty(self.M, self.P, device=self.device)

        dK = self.key.view(self.K, self.N)
        dV = self.value.view(self.N, self.P)

        ks = torch.chunk(dK, loop_n, dim=-1)
        vs = torch.chunk(dV, loop_n, dim=-2)
        
        q = self.query.view(self.M, self.K)  # m * k

        for n in range(loop_n):
            k = ks[n]
            v = vs[n]

            qk = q @ k  # m * ktn
            cur_maxes, _ = torch.max(qk, dim=-1, keepdim=True)
            # cur_maxes: maximun up to the current block.
            cur_maxes = torch.max(cur_maxes, lse)
            p = torch.exp(qk - cur_maxes).half()
            l_ij = torch.sum(p, dim=-1, keepdim=True)
                
            # renormalize o
            acc_o_scale = torch.exp(prev_maxes - cur_maxes)
            output = acc_o_scale * output + p @ v
                
            # Update statistics
            prev_maxes = cur_maxes
            l_i_new = torch.exp(lse - cur_maxes) + l_ij
            lse = cur_maxes + torch.log(l_i_new)
        
        # o_scale is the denominator of the softmax function.    
        o_scale = torch.exp(prev_maxes - lse)
        output = o_scale * output

        return output
    

