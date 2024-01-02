import torch
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments

from src.const import SPECIAL_TOKENS, SEGMENT_TYPES, MASKING_RATE, MAX_SEQUENCE_LENGTH
from src.data import BaiduTrainDataset
from src.model import MonoBERT


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    directory = Path(config.dataset_directory)
    train_files = [f for f in directory.glob("part-*") if f.name != "part-00000.gz"]
    eval_files = [directory / Path("part-00000.gz")]

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

    TrainingArguments

    bert_config = instantiate(config.bert_config)
    training_arguments = instantiate(config.training_arguments)
    model = MonoBERT(bert_config)
    torch.compile(model)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(3)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
