from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, Trainer, TrainingArguments, EarlyStoppingCallback

import wandb
from src.const import SPECIAL_TOKENS, SEGMENT_TYPES, MASKING_RATE, MAX_SEQUENCE_LENGTH
from src.data import BaiduTrainDataset
from src.model import MonoBERT


def main():
    wandb.init(project="baidu-bert-test")

    # %FIXME: Fix iterable dataset on multiple machines...

    train_dataset = BaiduTrainDataset(
        Path("/ivi/ilps/datasets/baidu_ultr/part-00001.gz"),
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
    )

    eval_dataset = BaiduTrainDataset(
        Path("/ivi/ilps/datasets/baidu_ultr/part-00000.gz"),
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
    )

    config = BertConfig(
        vocab_size=22_000,
        num_hidden_layers=12,
        num_attention_heads=12,
    )

    model = MonoBERT(config)
    torch.compile(model)

    args = TrainingArguments(
        output_dir="output",
        report_to=["wandb"],
        run_name="baidu-bert-test",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1_000,
        max_steps=1_000_000,
        dataloader_num_workers=4,
        seed=0,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        callbacks=[EarlyStoppingCallback(3)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
