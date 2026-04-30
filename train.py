import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size = 32 # how many sequences will we process in parallel
block_size = 8 # what is the maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

#import data set of jokes
df = pd.read_csv("shortjokes.csv")

text = '\n'.join(df['Joke'].astype(str))
text = text.lower().strip()

# All unique chars that appear
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokenize chars
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])


#train and test splits
data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(0.9 * len(data)) #get first 90% of data for training
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad() #No backprop here for efficiency
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = logits.mean()
    model.train()
    return out

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indecies in the current context
        for _ in range(max_new_tokens):
            #get predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B, C)
            #Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            #Sample from distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            #append sampled index to runnniong sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

#create a pyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    #evalate the loss on train and val sets once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')
    xb = xb
    yb = yb

    #evalate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from model
context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))
