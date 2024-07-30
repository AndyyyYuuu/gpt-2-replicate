import model
import torch
from config import *


state_dict, vocab = torch.load(save_path)
lm = model.GPT(len(vocab))
lm.load_state_dict(state_dict)
int_to_string = {i: ch for i, ch in enumerate(vocab)}


def decode(ints):
    return ''.join([int_to_string[i] for i in ints])


idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(lm.generate(idx, max_new_tokens=1000)[0].tolist()))

