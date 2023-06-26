import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_size, head_size, context_length):
        super().__init__()

        self.embedding_size = embedding_size
        self.val = nn.Linear(embedding_size, head_size, bias=False) 
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))


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

        # readout
        return weights @ self.val(X) # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_size, head_size, context_length):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(embedding_size, head_size, context_length) for _ in range(n_heads)])

    
    def forward(self, X):
        return torch.concatenate([head(X) for head in self.heads], dim = -1)


class FeedForward(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU()
        )

    def forward(self, X):
        return self.linear(X)



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_length, device, n_heads):
        super().__init__()
        self.context_length = context_length
        self.embedding_size = embedding_size
        self.device = device

        # embeddings
        self.embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding_table = nn.Embedding(context_length, embedding_size)
        self.readout_head = nn.Linear(embedding_size, vocab_size)
    
        # single-head attention
        self.multi_head_attention = MultiHeadAttention(n_heads, embedding_size, embedding_size // n_heads, context_length) # TODO according to karpathy these don't have to be the same size
    
        # feed-forward
        self.feed_forward = FeedForward(embedding_size, embedding_size)
        
    def forward(self, X, y = None):

        token_embeddings = self.embedding_table(X) # (B, T, embedding_size)
        pos_embeddings = self.positional_embedding_table(torch.arange(self.context_length, device=self.device))
        x = token_embeddings + pos_embeddings
        x = self.multi_head_attention(x)
        x = self.feed_forward(x)
        logits = self.readout_head(x) # (B, T, embedding_size) @ (embedding_size, vocab_size) -> (B, T, vocab_size)

        loss = None
        if y is not None:
            B, T, C = logits.size()
            y = y.view(B * T)
            logits = logits.view(B * T, C) # problematic as this will cause logits to return in two different shapes
            loss = F.cross_entropy(logits, y)

        return logits, loss # (32, 65), f32


    def generate(self, prompt, max_num_steps):
        # prompt = (B, T)

        output = prompt
            
        for i  in range(max_num_steps):
            prompt = output[:, -self.context_length:] # crop to context window
            logits, _ = self(prompt)
            logits = logits[:, -1, :] # B, C
            logits = F.softmax(logits, dim = -1)
            preds = torch.multinomial(logits, num_samples = 1)

            output = torch.concatenate((output, preds), dim = 1)

        return output