#%%
import torch
import torch.nn as nn
import numpy as np

#%%
class Transformer(nn.Module):
    def __init__(self, max_length=18):
        self.max_length = max_length

    def positional_encoding(max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        return pos_enc    

#%%
