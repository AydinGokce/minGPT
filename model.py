import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
import time
import os

def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

def last_20_lines(text):
    text = text.split('\n')
    text = text[-20:]
    return '\n'.join(text)

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_size, head_size, context_length, dropout):
        super().__init__()

        self.embedding_size = embedding_size
        self.val = nn.Linear(embedding_size, head_size, bias=False) 
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, X):
        # X = (B, T, C)
        k = self.key(X) # B, T, head_size
        q = self.query(X) # B, T, head_size
        weights = q @ k.transpose(-2, -1) # B, T, T

        # normalization
        weights = weights / (self.embedding_size ** 0.5)

        # masking
        weights = weights.masked_fill(self.tril == 0, float("-inf")) # TODO: karpathy added [:T, :T] to tril, dunnot the point of that so omitting
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)

        # readout
        return weights @ self.val(X) # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_size, head_size, context_length, dropout, projection_factor = 4):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(embedding_size, head_size * projection_factor, context_length, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embedding_size * projection_factor, embedding_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        out = torch.concatenate([head(X) for head in self.heads], dim = -1)
        return self.dropout(X) + self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, projection_factor = 4):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_ch, out_ch * projection_factor),
            nn.ReLU(),
            nn.Linear(out_ch * projection_factor, out_ch),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        return X + self.linear(X)


class Block(nn.Module):

    def __init__(self, n_heads, embedding_size, head_size, context_length, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            MultiHeadAttention(n_heads, embedding_size, head_size, context_length, dropout),
            nn.LayerNorm(embedding_size),
            FeedForward(embedding_size, embedding_size, dropout)
        
        )

    def forward(self, X):
        return self.block(X)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_length, device, n_heads, dropout):
        super().__init__()
        self.context_length = context_length
        self.embedding_size = embedding_size
        self.device = device

        # embeddings
        self.embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding_table = nn.Embedding(context_length, embedding_size)
        self.readout_head = nn.Linear(embedding_size, vocab_size)
    
        # multi-head attention and feed-forward
        self.blocks = nn.Sequential(
            Block(n_heads, embedding_size, embedding_size // n_heads, context_length, dropout),
            Block(n_heads, embedding_size, embedding_size // n_heads, context_length, dropout),
            Block(n_heads, embedding_size, embedding_size // n_heads, context_length, dropout),
            nn.LayerNorm(embedding_size)
        )
        
    def forward(self, X, y = None):

        token_embeddings = self.embedding_table(X) # (B, T, embedding_size)
        pos_embeddings = self.positional_embedding_table(torch.arange(self.context_length, device=self.device))
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        logits = self.readout_head(x) # (B, T, embedding_size) @ (embedding_size, vocab_size) -> (B, T, vocab_size)

        loss = None
        if y is not None:
            B, T, C = logits.size()
            y = y.view(B * T)
            logits = logits.view(B * T, C) # problematic as this will cause logits to return in two different shapes
            loss = F.cross_entropy(logits, y)

        return logits, loss # (32, 65), f32


    def generate(self, prompt, decode=None, print_=True, frequency = 10):
        # prompt = (B, T)
        with torch.no_grad():
            output = prompt
            while True:
                prompt = output[:, -self.context_length:] # crop to context window
                logits, _ = self(prompt)
                logits = logits[:, -1, :] # B, C
                logits = F.softmax(logits, dim = -1)
                preds = torch.multinomial(logits, num_samples = 1)
                if print_:
                    clear_console() 
                    print(last_20_lines(decode(output.cpu()[0].tolist())), end=" ")
                    #time.sleep(1/frequency)

                output = torch.concatenate((output, preds), dim = 1)

            return output