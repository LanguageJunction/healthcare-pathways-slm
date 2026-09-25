import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        text,
        tokenizer,
        block_size=512
    ):

        self.tokenizer = tokenizer
        self.block_size = block_size

        tokens = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

    def __len__(self):

        return max(
            0,
            len(self.tokens) - self.block_size
        )

    def __getitem__(self, idx):

        chunk = self.tokens[
            idx:idx + self.block_size + 1
        ]

        return chunk