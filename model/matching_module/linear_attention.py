"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from einops.einops import rearrange
    

class GeneralMaskedAttention(Module):
    def __init__(self, nhead=8, dim=256, fp32=False, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, query, key, value, hard_mask = None, bias = None):
        '''
        Args:
            query   :   (b l c)
            key     :   (b l c)
            value   :   (b l c)
            q_mask  :   (b l)
            kv_mask :   (b l)
        '''
        attn_mask = None
        if hard_mask is not None:
            assert query.shape[0] == 8
            forward_list = [self.forward_hard_mask_one_batch(query[i].unsqueeze(0), key[i].unsqueeze(0), value[i].unsqueeze(0), hard_mask[i], bias[i]) for i in range(query.shape[0])]
            return torch.stack(forward_list, dim=0)
        if bias is not None:
            attn_mask = bias
    
        q, k, v = map(lambda x: rearrange(x, 'n l (nhead d) -> n nhead l d', nhead=self.nhead, d=self.dim), [query, key, value])

        # q, k, v = [t.transpose(1, 2) for t in [query, key, value]]
        m = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        # m = self.attention(query, key, value, q_mask=q_mask, kv_mask=kv_mask, hard_decision = hard_decision)
        
        return m
    
    def forward_hard_mask_one_batch(self, query, key, value, hard_mask = None, bias = None):
        '''
        query       :   (1 l c)
        key         :   (1 l c)
        value       :   (1 l c)
        hard_mask   :   (1 l)
        bias        :   (1,l,l)
        '''
        L = query.size(1)
        C = query.size(2)
        device = query.device
        
        # 1. Early Exit: If all tokens are halted, spend 0 FLOPs
        if not hard_mask.any():
            return value

        # 2. Compute Stable Sorting Indices for the single batch
        sort_keys = (~hard_mask.bool()).long() * L + torch.arange(L, device=device).unsqueeze(0)
        sort_idx = sort_keys.argsort(dim=-1)       # (1, L)
        unsort_idx = sort_idx.argsort(dim=-1)      # (1, L)

        # 3. Exact active count (No need for max() since B=1)
        num_active = hard_mask.sum().item()
        # print(num_active)

        # Helper to gather and slice 3D tensors for B=1
        def gather_and_slice(t):
            expanded_idx = sort_idx.unsqueeze(-1).expand(-1, -1, C)
            t_sorted = torch.gather(t, 1, expanded_idx)
            return t_sorted[:, :num_active, :]

        # 4. Prune the tokens
        q_active = gather_and_slice(query)
        k_active = gather_and_slice(key)
        v_active = gather_and_slice(value)

        # 5. Handle and Prune the Bias Matrix
        attn_mask = None
        if bias is not None:

            idx = sort_idx[0] 
            
            # The ellipsis (...) tells PyTorch to leave all leading dimensions alone (Batch, Heads, etc.)
            # First we sort the second-to-last dimension (Query/Rows)
            bias_sorted = bias[..., idx, :]
            
            # Then we sort the last dimension (Key/Cols)
            bias_sorted = bias_sorted[..., :, idx]
            
            # Slice down to num_active. 
            # BIG WIN: No pad_mask required at all because B=1 guarantees 
            # we are only computing exactly the active tokens.
            attn_mask = bias_sorted[..., :num_active, :num_active]

        q, k, v = map(lambda x: rearrange(x, 'n l (nhead d) -> n nhead l d', nhead=self.nhead, d=self.dim), [q_active, k_active, v_active])


        # 7. Dense Attention
        m_active = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask, # If bias is None, this correctly passes None for unmasked SDPA
            dropout_p=self.dropout.p if self.use_dropout and self.training else 0.0,
            is_causal=False,
        )

        # Reshape back to 3D: (1, num_active, C)
        m_active = m_active.transpose(1, 2).reshape(1, num_active, C)

        # 8. Reconstruct the full sequence
        expanded_idx = sort_idx.unsqueeze(-1).expand(-1, -1, C)
        out_sorted = torch.gather(value, 1, expanded_idx)
        
        # Overwrite the front portion with the new attention results
        out_sorted[:, :num_active, :] = m_active

        # Unsort everything back to original physical locations
        unsort_expanded = unsort_idx.unsqueeze(-1).expand(-1, -1, C)
        out = torch.gather(out_sorted, 1, unsort_expanded)

        return out
    
