import math
import os
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchsummaryX import summary
from tqdm import tqdm

import wandb

# set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
cudnn.deterministic = True
cudnn.benchmark = False

tokenizer = tiktoken.encoding_for_model('gpt-2')

@dataclass
class IDeaLGPTConfig:

    # General
    name: str = 'pretrain_v2'
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    num_iters: int = 100000
    eval_iters: int = 3
    eval_interval: int = 2000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    sequence_length: int = 256
    vocab_size: int = 50257 # gpt2 vocab
    num_blocks: int = 8
    num_heads: int = 8
    embed_dim: int = 512
    dropout: float = 0.1
    bias: bool = False

    # Data
    num_workers: int = 8
    train_test_split: float = 0.8
    SUBSET_PERCENTAGE: float = 0.01 # % of OWT to train on, between 0 and 1

    # LR scheduler
    lr: float = 2e-3
    lr_decay: bool = True
    warmup_iters: int = 314
    min_lr: float = 6e-5

    # optimizer
    weight_decay: float = 1e-1
    grad_clip: float = 1.0


config = IDeaLGPTConfig()
device = config.device
print(config)
print(f'Effective batch size = {config.batch_size * config.gradient_accumulation_steps}')

# poor man's dataloader
# but actual motivation is - im too lazy to write and deal with pad tokens in above method to read data
# since there are documents which are less than sequence length and they mess up the batch
# this method is cleaner, i get to learn something new (np.memmap!) and it's fun!

data_dir = os.path.join('data', 'owt')

def get_batch(split):
    file_path = os.path.join(data_dir, 'val' if split == 'val.bin' else 'train.bin')
    # memmap allows to read huge .bin files without loading entire thing. magic?
    data = np.memmap(file_path, mode='r', dtype=np.uint16) # fp16?
    idx = torch.randint(len(data) - config.sequence_length, (config.batch_size, ))
    xb = torch.stack([torch.from_numpy(data[i:i+config.sequence_length].astype(np.int64)) for i in idx], dim=0)
    yb = torch.stack([torch.from_numpy(data[i+1:i+config.sequence_length+1].astype(np.int64)) for i in idx], dim=0)
    if device == 'cuda':
        # pin_memory is an optimization to reserve some space in cpu mem which is used for moving to gpu
        # reduces overhead -> increases perf
        # non_blocking = True is async data transfer
        xb, yb = xb.pin_memory().to(device, non_blocking=True), yb.pin_memory().to(device, non_blocking=True)
    return xb, yb

# poor man's lr scheduler. why? because cosine with warmup isn't readily available on torch (it's warm RESTARTS)
# but idc about restarting eh?
def get_lr(it):
    "get lr at a specific iteration"
    max_lr = config.lr
    min_lr = config.min_lr
    warmup_iters = config.warmup_iters
    max_lr_decay_iters = config.num_iters # can also be made into another param
    if it <= warmup_iters:
        return max_lr * (it / warmup_iters)

    if it > max_lr_decay_iters:
        # decaying only up to a certain point, interesting
        return min_lr
    ratio = (it - warmup_iters) / (max_lr_decay_iters - warmup_iters) # how much % of decay cycle is done?
    coeff = 0.5 * (1 + math.cos(math.pi * ratio)) # [0,1]
    return min_lr + coeff * (max_lr - min_lr) # beautiful

@torch.no_grad()
def estimate_losses(config):
    model.eval()
    losses = {'train': -1., 'val': -1.}
    for split in ['train', 'val']:
        loss = 0
        for _ in range(config.eval_iters):
            # xb, yb = next(iter(val_loader))
            # xb, yb = xb.to(device), yb.to(device)
            xb, yb = get_batch('val')
            loss += model(xb, yb)[1].item()
        loss /= config.eval_iters
        if split == 'train':
            losses['train'] = loss
        else:
            losses['val'] = loss
    model.train()
    return losses

class Head(nn.Module):
    # def __init__(self, embed_dim, head_size, sequence_length, dropout):
    def __init__(self, config, interim_head_size):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.interim_head_size = interim_head_size # say embed_dim = 32 -> broken into say 4 heads, so this will be 8, to be concated back to 32
        self.key = nn.Linear(config.embed_dim, interim_head_size, bias=config.bias)
        self.query = nn.Linear(config.embed_dim, interim_head_size, bias=config.bias)
        self.value = nn.Linear(config.embed_dim, interim_head_size, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones((config.sequence_length, config.sequence_length))))

        self.dropout = nn.Dropout(config.dropout)

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
    # def __init__(self, num_heads, embed_dim, head_size, sequence_length, dropout):
    def __init__(self, config, interim_head_size):
        super().__init__()
        self.head_list = nn.ModuleList([Head(config, interim_head_size) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.head_list], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.embed_dim, 4*config.embed_dim),
            nn.GELU(),
            nn.Linear(4*config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    # def __init__(self, num_heads, embed_dim, sequence_length, dropout):
    def __init__(self, config):
        super().__init__()
        self.interim_head_size = config.embed_dim // config.num_heads
        self.sa = MultiHeadAttention(config, self.interim_head_size)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # communication
        x = x + self.ff(self.ln2(x)) # computation
        return x


class Transformer(torch.nn.Module):
    # def __init__(self, embed_dim, vocab_size, sequence_length, num_heads, num_blocks, dropout):
    def __init__(self, config):
        super().__init__()
        self.sequence_length = config.sequence_length
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.sequence_length, config.embed_dim)
        self.block_list = nn.Sequential(*[Block(config)
                                          for _ in range(config.num_blocks)])
        self.final_ln = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

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
            ixs_cond = ixs[:, -self.sequence_length:] # consider only the last sequence_length tokens
            logits, loss = self.forward(ixs_cond) # logits=(b,t,c), loss is ignored
            # get juse the final timestep
            last_logits = logits[:, -1, :] # (b,c)
            # normalize
            last_probs = F.softmax(last_logits, dim=-1) # across c
            next_tokens = torch.multinomial(last_probs, 1) # (b,c) -> (b)
            ixs = torch.cat((ixs, next_tokens), dim=1) # across t so (b,t) -> (b, t+1)
        return ixs

model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=True)


# for generation
start_ix = torch.zeros((1,1), dtype=torch.long, device=device) # (newline character in a single batch)



wandb.login(key="5508720f47b02cabd61bb6acd61dc553d313b062", force=True) #API Key is in your wandb account, under settings (wandb.ai/settings)
run = wandb.init(
        name    = config.name, ## Wandb creates random run names if you skip this field
        # reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # entity = 'thunderbuddies',
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "ideal_gpt", ### Project should be created in your wandb account
        config = config ### Wandb Config for your run
)

cur_iter = 0
best_val = 1e9
os.makedirs(os.path.join('exps', config.name), exist_ok=True)
best_path = os.path.join('exps', config.name, 'best_model.pth')
running_loss = 0.0
loss_counter=0
pbar = tqdm(total=config.num_iters, dynamic_ncols=True, leave=False, position=0, desc="Train")

while cur_iter <= config.num_iters:
    optimizer.zero_grad(set_to_none = True) # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
    # poor man's lr scheduler
    cur_lr = get_lr(cur_iter) if config.lr_decay else config.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    # xb, yb = next(iter(train_loader))
    for micro_step in range(config.gradient_accumulation_steps):
        xb, yb = get_batch('train')
        # xb, yb = xb.to(device), yb.to(device)
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb)

        running_loss += loss.item()
        train_loss = running_loss / (loss_counter + 1)
        loss_counter += 1

        scaler.scale(loss).backward()

    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    # val every eval_intervals
    if cur_iter % config.eval_interval == 0:
        losses = estimate_losses(config)
        val_loss = losses['val']
        train_loss = losses['train']
        print(f'Val @ Epoch {cur_iter}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        wandb.log({
            'val_loss': val_loss,
            'iter': cur_iter,
            'lr': optimizer.param_groups[0]['lr']
        })
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f'Saved best model to {best_path}')

        # save current model too for vibes
        torch.save(model.state_dict(), os.path.join('exps', config.name, f'model_i{cur_iter}_t{train_loss:.2f}_v{val_loss:.2f}.pth'))
        print('Sample Generation')
        print(tokenizer.decode(model.generate(start_ix, 100)[0].tolist()))



    # train logs
    wandb.log({
        'train_loss': train_loss,
        'iter': cur_iter,
        'lr': cur_lr
    })
    pbar.set_postfix(
            loss = "{:.04f}".format(train_loss),
            lr = cur_lr
        )
    pbar.update()


    cur_iter += 1

print('Final Sample')
print(tokenizer.decode(model.generate(start_ix, 1000)[0].tolist()))