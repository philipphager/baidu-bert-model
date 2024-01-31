import hydra
import numpy as np
import torch
import jax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset

import flax
flax.config.update('flax_use_orbax_checkpointing', False)

from src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MAX_SEQUENCE_LENGTH,
    CLICK_METRICS,
    REL_METRICS,
)
from src.evaluator import Evaluator
from src.data import LabelEncoder, collate_for_clicks, collate_for_rels, random_split

def load_clicks(config: DictConfig, split: str):
    encode_query = LabelEncoder()

    def preprocess(batch):
        batch["query_id"] = encode_query(batch["query_id"])
        return batch

    dataset = load_dataset(
        "philipphager/baidu-ultr_baidu-mlm-ctr",
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
        "philipphager/baidu-ultr_baidu-mlm-ctr",
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

    test_clicks = load_clicks(config, split="test")
    _, test_clicks = random_split(
        test_clicks,
        shuffle=True,
        random_state=config.seed,
        test_size=0.5,
    )
    collate_clicks = lambda batch: collate_for_clicks(batch, MAX_SEQUENCE_LENGTH, SPECIAL_TOKENS, SEGMENT_TYPES)
    click_loader = DataLoader(test_clicks, 
                              batch_size = config.per_device_eval_batch_size,# * jax.device_count(), 
                              collate_fn=collate_clicks)

    collate_rels = lambda batch: collate_for_rels(batch, MAX_SEQUENCE_LENGTH, SPECIAL_TOKENS, SEGMENT_TYPES)
    test_rels = load_annotations(config)
    rels_loader = DataLoader(test_rels, 
                             batch_size = 1, 
                             collate_fn=collate_rels)

    model = instantiate(config.model)
    evaluator = Evaluator(click_metrics = CLICK_METRICS, rel_metrics = REL_METRICS, 
                          ckpt_dir = config.output_dir, **OmegaConf.to_container(config))

    clicks_df = evaluator.eval_clicks(model, click_loader)
    rels_df = evaluator.eval_rels(model, rels_loader)
    clicks_df.to_parquet(config.output_dir + "test_click.parquet")
    rels_df.to_parquet(config.output_dir + "test_rel.parquet")

    print(clicks_df.mean(axis=0).to_dict() | rels_df.mean(axis=0).to_dict())

if __name__ == "__main__":
    main()
