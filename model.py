import torch
from torch import nn
from torch.nn import functional as F

n_embed = 32
block_size = 8
n_heads = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_heads),
            Block(n_embed, n_heads),
            Block(n_embed, n_heads)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (batch size, time (block size), embedding size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (batch size, time, vocab size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        targets = targets.view(B*T)
        logits = logits.view(B*T, C)

        loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Block(nn.Module):

    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_heads, n_embed // n_heads)
        self.ff = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa_head(x)
        x = x + self.ff(x)
        return x


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        out = weights @ v
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):


    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

