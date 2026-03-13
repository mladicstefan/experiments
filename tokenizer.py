from datasets import DatasetDict, Dataset
import torch


class Tokenizer:
    train_dataset: Dataset
    val_dataset: Dataset
    vocab: set
    stoi: dict
    itos: dict

    def __init__(self, dataset: DatasetDict):
        self.train_dataset = dataset["train"].select(range(10_000))
        self.val_dataset = dataset["validation"].select(range(10_00))

        all_text = "".join(self.train_dataset["text"]) + "".join(
            self.val_dataset["text"]
        )
        self.vocab = sorted(set(all_text))

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def _encode(self, seq: str) -> list(str):
        return [self.stoi[c] for c in seq]

    def _decode(self, tok: str) -> list(str):
        return "".join([self.itos[i] for i in tok])

    def _tokenize(self, tokenize_me) -> dict(str, list(str)):
        return {"ids": self._encode(tokenize_me["text"])}

    def _flatten(self, ds: Dataset) -> torch.Tensor:
        return torch.tensor([id for row in ds["ids"] for id in row], dtype=torch.long)

    def run(self) -> (torch.Tensor, torch.Tensor):
        tokenized_train = self.train_dataset.map(
            self._tokenize, remove_columns=["text"]
        )
        tokenized_val = self.val_dataset.map(self._tokenize, remove_columns=["text"])

        return (self._flatten(tokenized_train), self._flatten(tokenized_val))
