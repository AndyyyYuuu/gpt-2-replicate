import torch
import model
from config import *


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(69420)


with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(f"Text size: {len(text)} chars")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {vocab_size}")

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(string):
    return [string_to_int[c] for c in string]


def decode(ints):
    return ''.join([int_to_string[i] for i in ints])


data = torch.tensor(encode(text), dtype=torch.long)

train_data = data[:int(0.9*len(data))]
valid_data = data[int(0.9*len(data)):]


def get_batch(split):
    split_data = train_data if split == "train" else valid_data
    ix = torch.randint(len(split_data)-block_size, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    lm.eval()

    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = lm(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    lm.train()
    return out


xb, yb = get_batch("train")

lm = model.GPT(vocab_size).to(device)
logits, loss = lm(xb, yb)
print(logits.shape)

idx = torch.zeros((1, 1), dtype=torch.long)


optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)

for steps in range(10000):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {steps}: Train loss: {losses['train']}, Valid loss: {losses['valid']}")
        torch.save([lm.state_dict(), chars], save_path)
    xb, yb = get_batch("train")

    logits, loss = lm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(lm.generate(idx, max_new_tokens=1000)[0].tolist()))
torch.save([lm.state_dict(), chars], save_path)