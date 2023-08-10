#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
#%%
class NameDataset(torch.utils.data.Dataset):
    def __init__(self, name_list):
        self.name_list = name_list
        self.max_length = max(len(word) for word in self.name_list) + 1
        self.chars = sorted(list(set(''.join(self.name_list))))

    def __len__(self):
        return len(self.name_list)
    
    def tokeniseWord(self, word):
        stoi = {s:i+1 for i,s in enumerate(self.chars)}
        stoi['.'] = 0
        token = []
        for letter in word:
            token.append(stoi[letter])
        return token

    def __getitem__(self, index):
        name = self.name_list[index] + '.'
        tokenisedName = self.tokeniseWord(name)
        return torch.tensor(tokenisedName)
# %%
