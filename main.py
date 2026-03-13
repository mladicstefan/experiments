from datasets import load_dataset
from dataclasses import dataclass
import torch

from tokenizer import Tokenizer
from model import Model


@dataclass
class Config:
    vocab_size: int
    batch_size: int = 32
    block_size: int = 8
    max_iters: int = 30000
    lr: float = 1e-3
    eval_iters: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    model: Model
    cfg: Config
    train_data: torch.Tensor
    val_data: torch.Tensor
    optimizer: torch.optim.AdamW

    def __init__(
        self, m: Model, cfg: Config, train_data: torch.Tensor, val_data: torch.Tensor
    ):
        self.model = m
        self.cfg = cfg
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.cfg.lr
        )

    def _get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - self.cfg.block_size, (self.cfg.batch_size,))
        x = torch.stack([data[i : i + self.cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.cfg.block_size + 1] for i in ix])

        return x.to(self.cfg.device), y.to(self.cfg.device)

    def train(self, epochs: int):

        for epoch in range(epochs):
            xb, yb = self._get_batch("train")

            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Loss: {loss.item()}")

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters)

            for k in range(self.cfg.eval_iters):
                X, Y = self._get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out


def main():

    dataset: DatasetDict = load_dataset("roneneldan/TinyStories")
    tok: Tokenizer = Tokenizer(dataset=dataset)
    (train, val) = tok.run()

    torch.manual_seed(1337)
    cfg: Config = Config(vocab_size=len(tok.vocab))
    m = Model(vocab_size=cfg.vocab_size).to(cfg.device)

    trainer: Trainer = Trainer(m, cfg, train, val)
    trainer.train(3000)

    idx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    print(tok._decode(m.generate(idx, max_new_tokens=300)[0].tolist()))
    return


if __name__ == "__main__":
    main()
