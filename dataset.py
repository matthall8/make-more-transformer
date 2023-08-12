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
        self.chars = ['<SOS>']+ sorted(list(set(''.join(self.name_list))))+['<EOS>']

    def __len__(self):
        return len(self.name_list)
    
    def tokeniseWord(self, word):
        stoi = {s:i+1 for i,s in enumerate(self.chars)}
        stoi['<PAD>'] = 0
        token = [stoi['<SOS>']]
        for letter in word:
            token.append(stoi[letter])
        token.append(stoi['<EOS>'])
        return token

    def __getitem__(self, index):
        name = self.name_list[index]
        source = self.tokeniseWord(name)[:-1]
        target = self.tokeniseWord(name)[1:]
        return torch.tensor(source), torch.tensor(target)