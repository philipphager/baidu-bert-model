from functools import partial
from typing import List, Dict

import jax
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm


def dict_to_numpy(_dict: Dict[str, jax.Array]) -> Dict[str, np.ndarray]:
    return {k: jax.device_get(v) for k, v in _dict.items()}


def collect_metrics(results: List[Dict[str, jax.Array]]) -> pd.DataFrame:
    """
    Collects batches of metrics into a single pandas DataFrame:
    [
        {"ndcg": [0.8, 0.3], "MRR": [0.9, 0.2]},
        {"ndcg": [0.2, 0.1], "MRR": [0.1, 0.02]},
        ...
    ]
    """
    # Convert Jax Arrays to numpy:
    np_results = [dict_to_numpy(r) for r in results]
    # Unroll values in batches into individual rows:
    df = pd.DataFrame(np_results, dtype=np.float32)
    return df.explode(column=list(df.columns)).reset_index(drop=True)


class Evaluator:
    def __init__(
        self,
        click_metrics: dict,
        rel_metrics: dict,
        ckpt_dir: str,
        seed: int,
        progress_bar: bool,
    ):
        self.click_metrics = click_metrics
        self.rel_metrics = rel_metrics
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.progress_bar = progress_bar

    def eval_rels(
        self,
        model,
        loader: DataLoader,
    ) -> pd.DataFrame:
        metrics = []

        for batch in tqdm(loader, disable=not self.progress_bar):
            metrics.append(self._eval_step_rels(model, batch))
            wandb.log(metrics[-1])

        return collect_metrics(metrics)

    def eval_clicks(
        self,
        model,
        loader: DataLoader,
    ) -> pd.DataFrame:
        metrics = []

        for batch in tqdm(loader, disable=not self.progress_bar):
            metrics.append(self._eval_step_clicks(model, batch))
            wandb.log(metrics[-1])

        return collect_metrics(metrics)

    @partial(jax.jit, static_argnums=(0, 1))
    def _eval_step_rels(self, model, batch):
        relevance = model.predict_relevance(batch, train=False)

        return {
            metric_name: metric(relevance.squeeze(), batch["labels"])
            for metric_name, metric in self.rel_metrics.items()
        }

    @partial(jax.jit, static_argnums=(0, 1))
    def _eval_step_clicks(self, model, batch):
        outputs = model(batch, train=False)

        return {
            metric_name: metric(
                outputs.click.squeeze(),
                batch["clicks"],
            )
            for metric_name, metric in self.click_metrics.items()
        }
