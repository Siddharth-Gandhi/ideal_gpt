# sourcery skip: identity-comprehension
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True # type: ignore
torch.backends.cudnn.benchmark = False # type: ignore

# hyperparameters
sequence_length = 8
batch_size = 64
lr=1e-3
num_iters=10000
eval_iters=314
eval_interval=1000



# parameters
split = 0.8
data_path = 'data/shakespeare/input.txt'
device='cuda' if torch.cuda.is_available() else 'cpu'

with open(data_path, 'r') as f:
    text = f.read()

vocab = sorted(set(text))
vocab_size = len(vocab)

stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}



def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return ''.join(itos[i] for i in t)


def get_batches(data):
    idx = torch.randint(len(data) - sequence_length, (batch_size, ))
    xb = torch.stack([data[i:i+sequence_length] for i in idx], dim=0)
    yb = torch.stack([data[i+1:i+sequence_length+1] for i in idx], dim=0)
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


data = encode(text)

train_data = data[:int(len(data)*split)]
val_data = data[int(len(data)*split):]

# model
class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, ixs, targets=None):
        # ixs: (b,t)
        # targets: (b,t)
        logits = self.embed(ixs) # (b,t,c=vocab_size)
        if targets is None:
            loss = None
        else:
            logits = logits.permute(0,2,1) # (b,c,t)
            loss = F.cross_entropy(logits, targets)
            logits = logits.permute(0,2,1) # back to (b,t,c)
        return logits, loss

    def generate(self, ixs, max_len):
        """
        ixs: (b,t) - input sequence to start generating from
        max_len: int - maximum length of the generated sequence
        """
        b, t = ixs.shape
        for _ in range(max_len):
            # generation (b, ) next tokens in parallel
            logits, loss = self.forward(ixs) # logits=(b,t,c), loss is ignored
            # get juse the final timestep
            last_logits = logits[:, -1, :] # (b,c)
            # normalize
            last_probs = F.softmax(last_logits, dim=-1) # across c
            next_tokens = torch.multinomial(last_probs, 1) # (b,c) -> (b)
            ixs = torch.cat((ixs, next_tokens), dim=1) # across t so (b,t) -> (b, t+1)
        return ixs


blm = BigramLanguageModel().to(device)
start_ix = torch.zeros((1,1), dtype=torch.long, device=device) # (newline character in a single batch)

optimzer = torch.optim.AdamW(blm.parameters(), lr=lr)

@torch.no_grad()
def estimate_losses():
    blm.eval()
    losses = {'train': -1., 'val': -1.}
    for split in ['train', 'val']:
        d = train_data if split == 'train' else val_data
        loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batches(d)
            loss += blm(xb, yb)[1].item()
        loss /= eval_iters
        if split == 'train':
            losses['train'] = loss
        else:
            losses['val'] = loss
    blm.train()
    return losses


for i in range(num_iters):
    xb, yb = get_batches(train_data)
    logits, loss = blm(xb, yb)
    if i % eval_interval == 0:
        losses = estimate_losses()
        print(f'Epoch {i}: Train Loss={losses["train"]:.4f}, Val Loss={losses["val"]:.4f}')

    # print(f'Epoch {i}: Loss={loss.item()}')

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

print(decode(blm.generate(start_ix, 1000)[0].tolist()))