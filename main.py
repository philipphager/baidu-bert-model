import torch
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import Trainer

from src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MASKING_RATE,
    MAX_SEQUENCE_LENGTH,
    MISSING_TITLE,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
)
from src.data import BaiduTrainDataset


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    directory = Path(config.dataset_directory)
    train_files = [f for f in directory.glob("part-*")]
    train_dataset = BaiduTrainDataset(
        files=train_files,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        masking_rate=MASKING_RATE,
        mask_query=True,
        mask_doc=True,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
        ignored_titles=[MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE],
    )

    training_arguments = instantiate(config.training_arguments)
    model = instantiate(config.model)
    torch.compile(model)

    if config.base_model_path is not None:
        print("Initializing from pre-trained model:", config.base_model_path)
        model = model.from_pretrained(
            config.base_model_path,
            config=model.config,
        )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
    )

    trainer.train()


if __name__ == "__main__":
    main()
