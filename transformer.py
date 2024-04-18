import torch
import torch.nn as nn
import torch.nn.functional as F

# set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True # type: ignore
torch.backends.cudnn.benchmark = False # type: ignore

# parameters
split = 0.8
data_path = 'data/shakespeare/input.txt'
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')
save_path='model.pt'



# hyperparameters
sequence_length = 8
batch_size = 64
lr=1e-3
num_iters=10000
eval_iters=100
eval_interval=1000
embed_dim=32
num_heads=8

num_blocks=4
dropout=0.2


# sanity checks
assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'


# Data input and preprocessing
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




# modelling

class Head(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((sequence_length, sequence_length))))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (b,t,c) -> (b,t,h)
        q = self.query(x) # (b,t,c) -> (b,t,h)
        v = self.value(x) # (b,t,c) -> (b,t,h)
        wei = k @ q.transpose(-2, -1) * self.embed_dim**(-0.5) # (b,t,h) @ (b,h,t) -> (b,t,t)

        wei = wei.masked_fill((self.tril[:T, :T] == 0.), -torch.inf) # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        xbow = wei @ v # (b,t,t) @ (b,t,h) -> (b,t,h)
        return xbow

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.head_list = nn.ModuleList([Head(embed_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.head_list], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, embed_dim // num_heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(sequence_length, embed_dim)
        # self.attn_head = Head(embed_dim)
        # self.mha = MultiHeadAttention(num_heads, embed_dim // num_heads)
        # self.ff = FeedForward(embed_dim)
        self.block_list = nn.Sequential(*[Block(num_heads, embed_dim) for _ in range(num_blocks)])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, ixs, targets=None):
        # ixs: (b,t)
        # targets: (b,t)
        B, T = ixs.shape
        x = self.token_embeddings(ixs) # (b,t,c=embed_dim)
        pos_embeds = self.position_embeddings(torch.arange(T, device=device)) # (t,c=embed_dim)
        x += pos_embeds
        x = self.block_list(x)
        x = self.final_ln(x)
        logits = self.lm_head(x) # (b,t,c=vocab_size)
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
            ixs_cond = ixs[:, -sequence_length:] # consider only the last sequence_length tokens
            logits, loss = self.forward(ixs_cond) # logits=(b,t,c), loss is ignored
            # get juse the final timestep
            last_logits = logits[:, -1, :] # (b,c)
            # normalize
            last_probs = F.softmax(last_logits, dim=-1) # across c
            next_tokens = torch.multinomial(last_probs, 1) # (b,c) -> (b)
            ixs = torch.cat((ixs, next_tokens), dim=1) # across t so (b,t) -> (b, t+1)
        return ixs


model = Transformer(embed_dim).to(device)
start_ix = torch.zeros((1,1), dtype=torch.long, device=device) # (newline character in a single batch)

optimzer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_losses():
    model.eval()
    losses = {'train': -1., 'val': -1.}
    for split in ['train', 'val']:
        d = train_data if split == 'train' else val_data
        loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batches(d)
            loss += model(xb, yb)[1].item()
        loss /= eval_iters
        if split == 'train':
            losses['train'] = loss
        else:
            losses['val'] = loss
    model.train()
    return losses


for i in range(num_iters):
    xb, yb = get_batches(train_data)
    logits, loss = model(xb, yb)
    if i % eval_interval == 0:
        losses = estimate_losses()
        print(f'Epoch {i}: Train Loss={losses["train"]:.4f}, Val Loss={losses["val"]:.4f}')

    # print(f'Epoch {i}: Loss={loss.item()}')

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

print(decode(model.generate(start_ix, 1000)[0].tolist()))

# save model
torch.save(model.state_dict(), save_path)