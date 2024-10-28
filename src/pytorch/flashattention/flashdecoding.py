import torch


class FlashDecoding:

    def __init__(self, query, key, value, M, N, K, P, ChunkN, SplitN, device):
        self.M = M
        self.N = N
        self.K = K
        self.P = P
        self.ChunkN = ChunkN
        self.SplitN = SplitN

        self.query = query
        self.key = key
        self.value = value
        self.output = torch.empty(M, P, device)
        
        self.device = device
        
        
    def flash_attn(self, q, k, v):
        loop_n = self.ChunkN // self.SplitN
        
        prev_maxes = torch.zeros(self.M, 1, device=self.device)
        prev_sums = torch.zeros(self.M, 1, device=self.device)
        
        # LogSumExp(LSE) is a smooth maximum. LSE(x1,..., xn) = log(exp(x1) + ... + exp(xn))
        lse = torch.zeros(self.M, 1, device=self.device)
        
        # output = self.output.view(self.M, self.P)
        output = torch.empty(self.M, self.P, device=self.device)
        
        ks = torch.chunk(k, loop_n, dim=-1)
        vs = torch.chunk(v, loop_n, dim=-2)
        
        for n in range(loop_n):
            k = ks[n]
            v = vs[n]
            
            attn_weights = q @ k
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
            
            
        return output
        
    def forward(self):
        q = self.query.view(self.M, self.K)  # m * k
        k = self.key.view(self.K, self.N)
        v = self.value.view(self.N, self.P)
        
        loop_n = self.N // self.ChunkN
        
        ks = torch.chunk(k, loop_n, dim=-1)
        vs = torch.chunk(v, loop_n, dim=-2)
        
        for n in range(loop_n):
            k = ks[n]
            v = vs[n]
            
            output, lse = self.flash_attn(q, k, v)
            
            # TODO: Compute the actual output by reducing over all the splits, using the log-sum-exp
            # to scale the contribution of each split.
