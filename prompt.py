import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "joke_model.pt"

# GPT-2 tokenizer, same as training
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

# Load checkpoint config
checkpoint = torch.load(checkpoint_path, map_location=device)

if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
    raise ValueError("joke_model.pt must contain 'model_state_dict' and 'config'.")

config = checkpoint["config"]
block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layers = config["n_layers"]
dropout = config["dropout"]


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (C ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Simple feed-forward layer."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



# Load model

def load_model(path):
    model = BigramLanguageModel().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model



# Main generation loop

def main():
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
    print(f"Model loaded on {device}.")

    while True:
        prompt = input("\nEnter a prompt (or type 'quit' to exit): ").strip()
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not prompt:
            prompt = "why did"

        try:
            max_new_tokens = input("Max new tokens [200]: ").strip()
            max_new_tokens = int(max_new_tokens) if max_new_tokens else 200

            temperature = input("Temperature [0.8]: ").strip()
            temperature = float(temperature) if temperature else 0.8

            top_k = input("Top-k [40, blank for none]: ").strip()
            top_k = int(top_k) if top_k else None

        except ValueError:
            print("Invalid number entered. Using defaults.")
            max_new_tokens = 200
            temperature = 0.8
            top_k = 40

        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        output_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()

        generated_text = decode(output_ids)

        print("\n" + "=" * 60)
        print(generated_text)
        print("=" * 60)

        save = input("Save output to generated_jokes.txt? [y/N]: ").strip().lower()
        if save == "y":
            with open("generated_jokes.txt", "w", encoding="utf-8") as f:
                f.write(generated_text)
            print("Saved to generated_jokes.txt")


if __name__ == "__main__":
    main()