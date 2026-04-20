import numpy as np
import torch

class EventFrame():
    def __init__(self, input_size: tuple):
        self.input_size = input_size
        self.img = torch.zeros(input_size, dtype=torch.float32, requires_grad=False)

    def convert(self, events):
        x = events['x'].to(torch.int64)
        y = events['y'].to(torch.int64)
        t = events['t']
        p = events['p']

        if p.min()>=0:
            p = 2*p -1

        num_events = len(x)

        idx = (y * self.input_size[-1] + x).to(torch.int64)
        img = self.img.flatten()
        img.scatter_(0, idx, p)

        return (img.reshape(*self.img.shape)+1)/2.0 # Normalize to 0-1