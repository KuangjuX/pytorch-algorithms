import torch


class FlashDecoding:

    def __init__(self, query, key, value, M, N, K, P, ChunkN, SplitN, device = 'cpu'):
        self.M = M
        self.N = N
        self.K = K
        self.P = P
        self.ChunkN = ChunkN
        self.SplitN = SplitN

        self.query = query
        self.key = key
        self.value = value
        
        self.device = device
        
        
    def flash_attn_split_kv(self, q, k, v):
        loop_n = self.ChunkN // self.SplitN
        
        # The LogSumExp(LSE) is a smooth maximum,
        # LSE(x1,...,xn) = log(exp(x1)+...+exp(xn))
        # = c + log(exp(x1-c)+...+exp(xn-c))
        # c = max(x1,...,xn)
        lse = torch.full((self.M, 1), float('-inf'), device=self.device)
        # prev_maxes: maximun up to the previous block.
        prev_maxes = torch.full((self.M, 1), float('-inf'), device=self.device)
        output = torch.empty(self.M, self.P, device=self.device)
        
        ks = torch.chunk(k, loop_n, dim=-1)
        vs = torch.chunk(v, loop_n, dim=-2)
        
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
            
        return output, lse
        
    def forward(self):
        q = self.query.view(self.M, self.K)  # m * k
        k = self.key.view(self.K, self.N)
        v = self.value.view(self.N, self.P)
        
        loop_n = self.N // self.ChunkN
        
        ks = torch.chunk(k, loop_n, dim=-1)
        vs = torch.chunk(v, loop_n, dim=-2)
        
        max_logic = torch.full((self.M, 1), float('-inf'), device=self.device)
        acc = torch.zeros(self.M, self.P, device=self.device)
        sum_exp = 0.0
        
        # This loop should be mapped to different blocks for parallel execution.
        for n in range(loop_n):
            k = ks[n]
            v = vs[n]
            
            output, lse = self.flash_attn_split_kv(q, k, v)
            
            # Compute the actual output by reducing over all the splits, using the log-sum-exp to scale the contribution of each split.
            
            new_max_logic = torch.max(max_logic, lse)
            
            old_scale = torch.exp(max_logic - new_max_logic)
            acc *= old_scale
            
            exp_logic = torch.exp(lse - new_max_logic)
            acc += exp_logic * output
            sum_exp = sum_exp * old_scale + exp_logic
            max_logic = new_max_logic
            
        return acc / sum_exp
