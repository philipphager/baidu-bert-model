import hydra
import numpy as np
import torch
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
    EVAL_METRICS,
)
from src.evaluator import Evaluator
from src.data import collate_for_eval


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


    eval_dataset = load_dataset(
        "philipphager/baidu-ultr_baidu-mlm-ctr",
        name="annotations",
        split="test",
        cache_dir="/beegfs/scratch/user/rdeffaye/baidu-bert/datasets",
        trust_remote_code=True,
    )

    collate_fn = lambda batch: collate_for_eval(batch, MAX_SEQUENCE_LENGTH, SPECIAL_TOKENS, SEGMENT_TYPES)
    eval_loader = DataLoader(eval_dataset, batch_size = 1, collate_fn=collate_fn)

    model = instantiate(config.model)

    evaluator = Evaluator(metrics = EVAL_METRICS, ckpt_dir = config.output_dir, 
                          **OmegaConf.to_container(config))

    metrics = evaluator.eval(model, eval_loader)
    print(metrics)

if __name__ == "__main__":
    main()
