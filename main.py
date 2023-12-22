from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

import wandb
from src.const import SPECIAL_TOKENS, SEGMENT_TYPES, MASKING_RATE, MAX_SEQUENCE_LENGTH
from src.data import BaiduTrainDataset
from src.model import MonoBERT


def main():
    wandb.init(project="baidu-bert-test")

    dataset = BaiduTrainDataset(
        Path("data/part-00000.gz"),
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
    )

    config = BertConfig(
        vocab_size=22_000,
        hidden_size=100,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=100,
    )

    data_loader = DataLoader(dataset, batch_size=32)
    model = MonoBERT(config)
    torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=0.0001)

    for epoch in range(1):
        for inputs in tqdm(data_loader):
            loss, query_document_embedding = model(**inputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train/loss": loss})


if __name__ == "__main__":
    main()
