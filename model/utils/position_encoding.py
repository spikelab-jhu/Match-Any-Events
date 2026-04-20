import math
import torch
from torch import nn

class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1) # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1) # [W, 1]
        
        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[3] # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W
        
        div_term = torch.exp(torch.arange(0, d_model//4, 1).float() * (-math.log(10000.0) / (d_model//4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]        

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, :x.size(1), :x.size(2), :]) + (self.rotate_half(x) * self.sin[:, :x.size(1), :x.size(2), :])
    
    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
    
class RoPE1d(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 1-dimensional sequence
    """

    def __init__(self, d_model, seq_len=8, npe=[8,8], ropefp16=True):
        """
        Args:
            seq_len = 8 during training
        """
        super().__init__()

        i_position = torch.arange(seq_len).float().unsqueeze(1) # [H, 1]

        
        assert npe is not None
        train_len, test_len = npe[0], npe[1] # train_res_H, train_res_W, test_res_H, test_res_W
        i_position = i_position * train_len / test_len
        
        
        # frequencies
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # positions [seq_len, dim/2]

        angles = i_position * theta.unsqueeze(0)   # [seq_len, d/2]

        if ropefp16:
            cos, sin = angles.cos().half(), angles.sin().half()
        else:
            cos, sin = angles.cos(), angles.sin()

        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, seq_len, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, seq_len, C//2]        

    def forward(self, x):
        """
        Args:
            x: [N, L, C]
        """

        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * self.cos - x2 * self.sin, x2 * self.cos + x1 * self.sin], dim=-1).flatten(-2, -1)
