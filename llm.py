import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

with open ("data.txt", 'r', encoding='utf-8') as f:
  text = f.read()

print("length of dataset in characters: ", len(text))

chars = sorted(list(set(text)))
vocab = "".join(chars)
vocab_size = len(vocab)


# Creating character level encoder and decoder
str_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_str = {i:ch for ch,i in str_to_int.items()}
encode = lambda s : [str_to_int[x] for x in s]
decode = lambda l : [''.join(int_to_str[x] for x in l)]

data = torch.tensor(encode(text), dtype=torch.long)


batch_size = 32
block_size = 8
max_iters = 1000
learning_rate = 1e-3
eval_iters = 200
eval_interval = 300
n_embd = 32

size = int(0.9*len(data))
train_data = data[:size]
val_data = data[size:]

def get_batch(split):
  if split == 'train':
    data = train_data
  else:
    data = val_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:block_size+i] for i in ix])
  y = torch.stack([data[i+1:block_size+i+1] for i in ix])
  return x, y

xb, yb = get_batch('train')


# Defining the components of the decoder side of the transformer 
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init()
    self.linear = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLu(), nn.Linear(4*n_embd, n_embd))

  def forward(self, x):
    return self.linear(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size= n_embd // n_heads
    self.sa = MultiHeadAttention(n_embd, head_size)
    self.ffwd = FeedForward(n_embd)

  def forward(self, x):
    x = x + self.sa(x)
    x = x + self.ffwd(x)
    return x
  


# Defining the transformer model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
                    Block(n_embd, n_heads=4),
                    Block(n_embd, n_heads=4),
                    Block(n_embd, n_heads=4),
                    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T))
    x = token_emb + pos_emb
    x = self.blocks(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for i in range(max_new_tokens):

      idx_cond = idx[:, -block_size:]

      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      prob = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(prob, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel()
log, loss = model(xb, yb)
print(log.shape, loss)



def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


# Training the model
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  x, y = get_batch('train')
  logits, loss = model(x, y)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

# Getting the predictions
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, 300)[0].tolist()))