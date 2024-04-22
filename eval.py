from functools import partial
from pathlib import Path

import flax
import hydra
import numpy as np
import pyarrow
import pyarrow_hotfix
import torch
import wandb
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

flax.config.update('flax_use_orbax_checkpointing', False)

from src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MAX_SEQUENCE_LENGTH,
    CLICK_METRICS,
    REL_METRICS,
)
from src.evaluator import Evaluator
from src.data import LabelEncoder, collate_rel_fn, random_split, collate_click_fn


def load_clicks(config: DictConfig, split: str):
    encode_query = LabelEncoder()

    def preprocess(batch):
        batch["query_id"] = encode_query(batch["query_id"])
        return batch

    dataset = load_dataset(
        "philipphager/baidu-ultr_uva-mlm-ctr",
        name="clicks",
        split=split,
        cache_dir=config.cache_dir,
        trust_remote_code=True,
    )
    dataset.set_format("numpy")

    return dataset.map(preprocess)


def load_annotations(config: DictConfig, split="test"):
    encode_query = LabelEncoder()

    def preprocess(batch):
        batch["query_id"] = encode_query(batch["query_id"])
        return batch

    dataset = load_dataset(
        "philipphager/baidu-ultr_uva-mlm-ctr",
        name="annotations",
        split=split,
        cache_dir=config.cache_dir,
        trust_remote_code=True,
    )
    dataset.set_format("numpy")

    return dataset.map(preprocess)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load test clicks
    test_clicks = load_clicks(config, split="test")
    _, test_clicks = random_split(
        test_clicks,
        shuffle=True,
        random_state=config.seed,
        test_size=0.5,
    )
    collate_clicks = partial(
        collate_click_fn,
        max_tokens=MAX_SEQUENCE_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
    )
    click_loader = DataLoader(
        test_clicks,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=collate_clicks,
    )

    # Load test set of expert annotations
    test_rels = load_annotations(config)
    collate_rels = partial(
        collate_rel_fn,
        max_tokens=MAX_SEQUENCE_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
    )
    rels_loader = DataLoader(test_rels, batch_size=1, collate_fn=collate_rels)

    # Download model checkpoint from huggingface
    model = instantiate(config.model)
    model = model.from_pretrained(f"{config.hf_hub_user}/{config.hf_hub_model}")

    evaluator = Evaluator(
        click_metrics=CLICK_METRICS,
        rel_metrics=REL_METRICS,
        ckpt_dir=config.output_dir,
        seed=config.seed,
        progress_bar=config.progress_bar,
    )

    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        sync_tensorboard=False,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=config.model.name,
    )

    clicks_df = evaluator.eval_clicks(model, click_loader)
    rels_df = evaluator.eval_rels(model, rels_loader)
    clicks_df.to_parquet(Path(config.output_dir) / "test_click.parquet")
    rels_df.to_parquet(Path(config.output_dir) / "test_rel.parquet")

    print(clicks_df.mean(axis=0).to_dict() | rels_df.mean(axis=0).to_dict())


if __name__ == "__main__":
    main()
