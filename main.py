import torch
import model

block_size = 8
batch_size = 32
learning_rate = 1e-2
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



xb, yb = get_batch("train")

lm = model.BigramLM(vocab_size).to_device()
logits, loss = lm(xb, yb)
print(logits.shape)

idx = torch.zeros((1, 1), dtype=torch.long)


optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)

for steps in range(10000):
    xb, yb = get_batch("train")

    logits, loss = lm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

print(decode(lm.generate(idx, max_new_tokens=500)[0].tolist()))