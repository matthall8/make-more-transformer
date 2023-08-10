#%%
import torch
import torch.backends.cudnn as cudnn
from dataset import NameDataset
from torch.nn.utils.rnn import pad_sequence

#%%
names = open('names.txt', 'r').read().splitlines()
len(names)
#%%
def collate_fn(batch):
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

full_dataset = NameDataset(names)

# hyperparameters 
batch_size = 1
n_embd = 24
train_loader = torch.utils.data.DataLoader(full_dataset, collate_fn=collate_fn)
# %%
for i, batch in enumerate(train_loader):
    print(batch.shape)
# %%
