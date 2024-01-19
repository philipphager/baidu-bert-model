from pathlib import Path

import hydra
import jax
import numpy as np
import torch
from hydra.utils import instantiate
from jax.tree_util import tree_map
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import flax
from flax.training import checkpoints
flax.config.update('flax_use_orbax_checkpointing', False)

import wandb
from src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MAX_SEQUENCE_LENGTH,
    MISSING_TITLE,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
)
from src.trainer import Trainer


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.training_arguments.seed)
    torch.manual_seed(config.training_arguments.seed)

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

    np_collate = lambda batch: tree_map(np.asarray, torch.utils.data.default_collate(batch))
    batch_size = config.training_arguments.per_device_train_batch_size * jax.device_count()
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=np_collate)

    model = instantiate(config.model)

    if config.base_model_path is not None:
        print("Initializing from pre-trained model:", config.base_model_path)
        model = model.from_pretrained(
            config.base_model_path,
            config=model.config,
        )

    trainer = Trainer(**config.training_arguments)

    if config.training_arguments.log_metrics:
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=False,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            name=config.run_name,
            save_code=True,
        )
    trained_state = trainer.train(model, train_loader)
    checkpoints.save_checkpoint(ckpt_dir=config.training_arguments.output_dir + config.run_name,
                            target=trained_state,
                            step=0,
                            overwrite=True,
                            keep=2)


if __name__ == "__main__":
    main()
