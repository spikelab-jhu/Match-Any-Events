import numpy as np
import torch

def get_event_polarity(event:dict, polarity = 1):
    if polarity>0:
        index = event['p'] > 0
    else:
        index = event['p'] < 1

    if len(index) == 0:
        return None
    event_updated = {'x':event['x'][index],
                     'y':event['y'][index],
                     't':event['t'][index],
                     'p':event['p'][index]
                     }
    return event_updated

class EventVis():
    def __init__(self, input_size: tuple, classic = False, trans = False):
        self.input_size = input_size
        self.trans = trans
        self.classic = classic
        if classic:
            self.img = torch.ones(input_size, dtype=torch.float32, requires_grad=False) 
        else:
            self.img = torch.zeros(input_size, dtype=torch.float32, requires_grad=False) + 0.8


    def scatter_img(self, events, color):
        x = events['x'].to(torch.int64)
        y = events['y'].to(torch.int64)
        # t = events['t']
        # p = events['p']

        num_events = len(x)

        idx = (y * self.input_size[-1] + x).to(torch.int64)
        img_flat = self.img.view(3, -1)
        img_flat.scatter_(1, idx.unsqueeze(0).expand(3, -1), color.T.repeat(1,idx.shape[-1]))
        self.img = img_flat.reshape(*self.img.shape)


    def convert(self, events):
        e_p = get_event_polarity(events, 1)
        e_n = get_event_polarity(events, 0)
        

        # if p.min()>=0:
        #     p = 2*p -1

        # p_color = torch.tensor([
        #     [1.0, 1.0, 1.0]])
        # n_color = torch.tensor([
        #     [0.0, 0.192, 0.3254]
        # ])
        if self.classic:
            p_color = torch.tensor([
                [1.0, 0.0, 0.0]])
            n_color = torch.tensor([
                [0.0, 0.0, 1.0]
            ])
        else:
            n_color = torch.tensor([
                [0.2, 0.2, 0.2]
            ])
            p_color = torch.tensor([
                [0.2, 0.2, 0.2]])


        self.scatter_img(e_p, p_color)
        self.scatter_img(e_n, n_color)

        if self.trans:
            rgba = np.zeros((4, self.img.shape[-2], self.img.shape[-2]), dtype=np.uint8)
            rgba[:3] = self.img
            white_mask = (self.img[0] == 1) & (self.img[1] == 1) & (self.img[2] == 1)

            # Set alpha = 0 for white pixels
            rgba[3, white_mask] = 0

            return rgba

        return self.img # Normalize to 0-1