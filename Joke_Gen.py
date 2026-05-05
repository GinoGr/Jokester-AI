import os # used for file manipulation
import pandas as pd #Using to read dataset
import torch
import torch.nn as nn
from torch.nn import functional as F #using torch to build and train neural network
import tiktoken #gpt-2 tokenizer

#Hyperparameters
max_iters = 5000 #From 20000. Trains model up to 5000 steps
eval_interval = 1000 #Will be used as a check while model trains
learning_rate = 1e-3 #From 3e-4. Control how much adjustment model makes during training
device = 'cuda' if torch.cuda.is_available() else 'cpu' #Use gpu if avail
eval_iters = 1000 #Average loss over 1000 batches
batch_size = 32 # how many sequences will we process in parallel
block_size = 64 # what is the maximum context length for prediction, From 128
n_embd = 256 # Each token is represented as a vector of length n_embd
n_head = 4  # 4 attention heads
n_layers = 4 #From 2. # of stacked blocks
dropout = 0.1 #From .1 #Randomly ignores some "neurons" during training to reduce over fitting
checkpoint_path = "checkpoint.pt"

torch.manual_seed(1337) # Kept same throughout for consistency
torch.cuda.manual_seed_all(1337) # Seed chosen from source where this concept was learned

#import data set of jokes
df = pd.read_csv("question_jokes.csv")


text = '\n'.join(df['Joke'].astype(str)) #convert to one var
text = text.lower().strip()  #clean text


enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}) #turn text into tokens
decode = lambda l: enc.decode(l) #turns tokens back into text

vocab_size = enc.n_vocab #How many possible tokens there are


#train and test splits
print("Using GPU") if torch.cuda.is_available() else print("Using CPU")
data = torch.tensor(encode(text), dtype=torch.long).to(device) #Move tokenized dataset into a torch tensor
n = int(0.9 * len(data)) #get first 90% of data for training (rest will be used for validation)
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y. Will run 32 times simultaneously at a time
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #Choose random starting position
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device) #Making the input sequence (64 tokens as indetified in block_size)
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix]).to(device) #Same as x but shifted up one token. Target to be "Predicted"
    return x, y

@torch.no_grad() #No backprop here for efficiency. Static function so no learning will be used here
def estimate_loss():
    """Evaluate performance"""
    out = {} #Store results
    m.eval() #Switch model to evaluation mode
    for split in ['train', 'val']: # performance is checked on both splits
        losses = torch.zeros(eval_iters).to(device) #Store results here
        for k in range(eval_iters): #Fill losses tensor
            X, Y = get_batch(split) #Store batch
            logits, loss = m(X, Y) #Evaluate batch on loss amount based on predictions
            losses[k] = loss.item() #Store results of loss
        out[split] = losses.mean() #Average loss number
    m.train() #Revert model back to training mode
    return out

class Head(nn.Module):
    """One head of self-attentive language model. Will help model determine which earlier word matters modst when predicting next word"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False) #Convert to key vector to gather token information
        self.query = nn.Linear(n_embd, head_size, bias = False) #What is the head looking for
        self.value = nn.Linear(n_embd, head_size, bias = False) #Store token info based on information gathered
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #Create lower triangular mask
        self.dropout = nn.Dropout(dropout) #Randomly drop sometimes

    def forward(self, x): #Will  actually run the attention process
        B, T, C = x.shape #Batch size (B) = 32, Sequnce length (T) = 64 tokens, Embedding size (C) = 256
        k = self.key(x)
        q = self.query(x)
        #compute attention scores aka affinities
        weight = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) --> (B, T, T). This will compare this token with every other token in sequence to find relationships
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T) #ignore future tokens in context
        weight = F.softmax(weight, dim = -1)  #(B, T, T). Sum values to 1
        weight = self.dropout(weight) #randomly drop
        #perform weighted aggregation of values
        v = self.value(x) #Gather existing information
        out = weight @ v #Actually sets the new vector info based on its relationship with other tokens
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attentive language model. Same as single headed, but does attention head in different ways at the same time"""

    def __init__(self, num_heads, head_size): #number of heads, and size of each head output
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #Create list of attention heads (4 in this case)
        self.proj = nn.Linear(n_embd, n_embd) #Will project 4 heads into a one head output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #concatinate each head's output. Same input but different layer used per token
        out = self.proj(out) #lay out the resulting output into a 256 vector so its useable

        return out

class FeedFoward(nn.Module):
    """ Linear layer followed by a non-linear activation function. Process each token individually"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # initialize layer
            nn.Linear(n_embd,4 * n_embd), #expand layer
            nn.ReLU(), #Converts to non linear activation  to add compllexity
            nn.Linear(4 * n_embd, n_embd), #Shrinks back down to useable size
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) #sends it

class Block(nn.Module):
    """Transformer block. Will be used to use attention heads and feedforward"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_headL the number of heads we would like
        super().__init__()
        head_size = n_embd // n_head #Split embedding accross heads
        self.sa = MultiHeadAttention(n_head, head_size) #create multihead attention
        self.ffwd = FeedFoward(n_embd) # create feed forward network
        self.ln1 = nn.LayerNorm(n_embd) #Normalized to keep efficiency
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): #pass data to block init
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #turn token id into a tensor
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #Get postional information ordered
        self.blocks = nn.Sequential(* [Block(n_embd, n_head = n_head) for _ in range(n_layers)]) #Create stack of transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) #normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) #Linear neural network

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device = device))  # (T, C)
        x = tok_embd + pos_embd # (B, T, C). Token position with meaning
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #Convert final results to activation percent for predicitons

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) #How wrong was the model

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) #run current text through model

            logits = logits[:, -1, :] / temperature

            if top_k is not None: #When clustering
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # will be the one adjusting the parameters according to adamw network

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