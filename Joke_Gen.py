import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

#Hyperparameters
max_iters = 20000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1000
batch_size = 32 # how many sequences will we process in parallel
block_size = 128 # what is the maximum context length for prediction
n_embd = 256
n_head = 4
n_layers = 4
dropout = 0.1
checkpoint_path = "checkpoint.pt"

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

#import data set of jokes
df = pd.read_csv("shortjokes.csv")

text = '\n'.join(df['Joke'].astype(str))
text = text.lower().strip()


enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

vocab_size = enc.n_vocab


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
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    """One head of self-attentive language model."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores aka affinities
        weight = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        weight = F.softmax(weight, dim = -1)  #(B, T, T)
        weight = self.dropout(weight)
        #perform weighted aggregation of values
        v = self.value(x)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attentive language model."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)

        return out

class FeedFoward(nn.Module):
    """ Linear layer followed by a non-linear activation function."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_headL the number of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x= x + self.ffwd(self.ln2(x))
        return x

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(* [Block(n_embd, n_head = n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device = device))  # (T, C)
        x = tok_embd + pos_embd # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

start_iter = 0

def save_checkpoint(iter_num, model, optimizer, loss=None, path=checkpoint_path):
    torch.save({
        "iter": iter_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layers": n_layers,
            "dropout": dropout,
            "vocab_size": vocab_size,
        }
    }, path)
    print(f"Checkpoint saved at step {iter_num} -> {path}")


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iter"] + 1
    print(f"Resuming from step {start_iter}")
else:
    print("No checkpoint found. Starting fresh.")

try:
    for iter in range(start_iter, max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
            save_checkpoint(iter, m, optimizer, losses['val'].item())

        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving checkpoint...")
    save_checkpoint(iter, m, optimizer, loss.item())
    torch.save(m.state_dict(), "joke_model.pt")
    print("Checkpoint saved. You can safely exit.")


save_checkpoint(max_iters, m, optimizer, loss.item())
torch.save({
    "model_state_dict": m.state_dict(),
    "config": {
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layers": n_layers,
        "dropout": dropout,
        "vocab_size": vocab_size,
    }
}, "joke_model.pt")
print("Final model saved to joke_model.pt")