from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import Trainer

from src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MAX_SEQUENCE_LENGTH,
    MISSING_TITLE,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    directory = Path(config.dataset_directory)
    train_files = [f for f in directory.glob("part-*")]

    train_dataset = instantiate(
        config.data,
        files=train_files,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
        ignored_titles=[MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE],
    )

    model = instantiate(config.model)
    training_arguments = instantiate(config.training_arguments)

    if config.base_model_path is not None:
        print("Initializing from pre-trained model:", config.base_model_path)
        model = model.from_pretrained(
            config.base_model_path,
            config=model.config,
        )

    torch.compile(model)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
    )

    trainer.train(resume_from_checkpoint = config.training_arguments.resume_from_checkpoint)


if __name__ == "__main__":
    main()
