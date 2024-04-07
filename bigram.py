import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers. We basically just want an integer to numbers and vice-versa
string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
# Create encoder and decoder functions
encode = lambda str:[string_to_int[char] for char in str] # encoder function
decode = lambda arr:''.join([int_to_string[int] for int in arr]) # decoder function

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data for inputs x and predictions y
    data = train_data if split == 'train' else validation_data
    random_offsets = torch.randint(len(data) - block_size, (batch_size,))
    # print(f"random_offsets:\n  {random_offsets}")
    x = torch.stack([data[i:i+block_size] for i in random_offsets])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_offsets])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # this says to not call .backward on
def estimate_loss():
    out = {}
    model.eval() # set to evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set to training phase
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)
        print(f"self.token_embedding_table:\n{self.token_embedding_table}")

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) = (Batch x Time x Channel) = (4 x 8 x 65) 65 is vocab_size
        # Ever number in the xb tensor will get its own row in the embedding table.
        # So for example, `24` in the xb tensor will refer to the 24th row in the embedding table.

        if targets is None:
            loss = None
        else:
            # We need to reshape the table to get it into the shape that Pytorch needs for the `cross_entropy` function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # the loss function

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B x T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) #
            # focus only on the last step
            logits = logits[:, -1, :] # get last element then becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, C)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx





model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# this is the training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
