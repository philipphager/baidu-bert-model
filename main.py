import os
from pathlib import Path

import torch
from transformers import BertConfig, Trainer, TrainingArguments, EarlyStoppingCallback

import wandb
from src.const import SPECIAL_TOKENS, SEGMENT_TYPES, MASKING_RATE, MAX_SEQUENCE_LENGTH
from src.data import BaiduTrainDataset
from src.model import MonoBERT


def main():
    wandb.init(project="baidu-bert-test")

    directory = Path("/ivi/ilps/datasets/baidu_ultr")
    train_files = [f for f in directory.glob("part-*") if f.name != "part-00000.gz"]
    eval_files = [Path("part-00000.gz")]

    print("Train files:", train_files[:10])
    print("Eval file:", train_files[:10])

    train_dataset = BaiduTrainDataset(
        train_files,
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
    )

    eval_dataset = BaiduTrainDataset(
        eval_files,
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
        evaluation_strategy="steps",
        max_steps=500_000,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        dataloader_num_workers=4,
        save_total_limit=1,
        load_best_model_at_end=True,
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
