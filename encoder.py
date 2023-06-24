import os

class Encoder:
    def __init__(self, data_source: str):
        self.data = open(data_source, "r").read()
        self.vocabulary = sorted(list(set(self.data)))

        self.chtoi = { ch:i for (i, ch) in enumerate(self.vocabulary) }
        self.itoch = { i:ch for (i, ch) in enumerate(self.vocabulary) }


    def decode(self, tokens: list[int]) -> str:
        return ''.join([ self.itoch[tok] for tok in tokens ])


    def encode(self, text: str) -> list[int]:
        return [ self.chtoi[ch] for ch in text ]