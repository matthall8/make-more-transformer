#%%
import torch
import torch.backends.cudnn as cudnn
from dataset import NameDataset
from model import MatTransformer
from torch.nn.utils.rnn import pad_sequence
#%%
names = open('names.txt', 'r').read().splitlines()
len(names)
#%%
def collate_fn(batch):
    source, target = zip(*batch)
    source_padded = pad_sequence(source, batch_first=True, padding_value=0) 
    target_padded = pad_sequence(target, batch_first=True, padding_value=0)
    return source_padded, target_padded

full_dataset = NameDataset(names)

# hyperparameters 
batch_size = 32
n_embd = 34
epochs = 10
max_len = max(len(item) for item in names) + 1
vocab_size = len(set(''.join(names))) + 3 #number of letters + 3 special tokens - <SOS>, <EOS> & <PAD>
full_training_set = torch.utils.data.DataLoader(full_dataset, batch_size = batch_size, collate_fn=collate_fn)

#%%
m = MatTransformer(max_len, n_embd,vocab_size)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
for epoch in range(epochs):
  for idx, (source, target) in enumerate(full_training_set):
    x = source
    y = target
    p = m(x)
    p_class = p.permute(0, 2, 1)
    l = torch.nn.functional.cross_entropy(p_class, y)
    if idx % 1000 == 0: print("Loss:", l.item())
    l.backward()
    opt.step()
    opt.zero_grad()
# %%
