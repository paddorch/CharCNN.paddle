DEFAULT_ALPHABET = '''abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'/\\|_@#$%ˆ&*˜`+-=<>()[]{} \n'''


__all__ = ['Tokenizer']


class Tokenizer:
    def __init__(self, max_len: int, alphabet: str = DEFAULT_ALPHABET):
        self.max_len = max_len

        self.char2idx = dict()
        self.unk_char = '<unk>'
        self.pad_char = '<pad>'
        self.char2idx[self.unk_char] = 0
        self.char2idx[self.pad_char] = 1
        for i, c in enumerate(alphabet):
            self.char2idx[c] = i + 2
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    @property
    def alphabet_size(self):
        return len(self.char2idx.keys())

    def quantize(self, c: str):
        if c in self.char2idx:
            return self.char2idx[c]
        else:
            return self.char2idx[self.unk_char]

    def tokenize(self, text: str):
        text = text.lower()
        text = text[:self.max_len]
        ids = [self.quantize(c) for c in text]

        pad_idx = self.char2idx[self.pad_char]
        ids = ids + [pad_idx] * (self.max_len - len(text))

        return ids
