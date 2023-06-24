import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, X, y = None):

        logits = self.embedding_table(X) # (B, T, C)
        loss = None

        if y is not None:
            B, T, C = logits.size()
            y = y.view(B * T)
            logits = logits.view(B * T, C) # problematic as this will cause logits to return in two different shapes
            loss = F.cross_entropy(logits, y)

        return logits, loss # (32, 65), f32


    def generate(self, prompt, max_num_steps):
        # prompt = (B, T)

        for i  in range(max_num_steps):
            logits, _ = self(prompt)
            logits = logits[:, -1, :] # B, C
            logits = F.softmax(logits, dim = -1)
            preds = torch.multinomial(logits, num_samples = 1)

            prompt = torch.concatenate((prompt, preds), dim = 1)

        return prompt
