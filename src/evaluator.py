from tqdm import tqdm
import numpy as np
import jax
from functools import partial
from typing import List, Dict
import pandas as pd

from flax.training import checkpoints
from torch.utils.data import DataLoader

def dict_to_numpy(_dict: Dict[str, jax.Array]) -> Dict[str, np.ndarray]:
    return {k: float(jax.device_get(v)) for k, v in _dict.items()}

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
    df = pd.DataFrame(np_results, dtype = np.float32)
    return df.explode(column=list(df.columns)).reset_index(drop=True)

class Evaluator:
    def __init__(
        self,
        click_metrics: dict,
        rel_metrics: dict,
        ckpt_dir: str,
        seed: int = 2024,
        progress_bar: bool = True,
        **kwargs,
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
        params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=None)["params"]

        for batch in tqdm(loader, disable=not self.progress_bar):
            metrics.append(self._eval_step_rels(model, params, batch))

        return collect_metrics(metrics)

    def eval_clicks(
        self,
        model,
        loader: DataLoader,
    ) -> pd.DataFrame:
        metrics = []
        params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=None)["params"]

        for batch in tqdm(loader, disable=not self.progress_bar):
            metrics.append(self._eval_step_clicks(model, params, batch))

        return collect_metrics(metrics)


    @partial(jax.jit, static_argnums = (0, 1))
    def _eval_step_rels(self, model, params, batch):
        relevances = model.predict_relevance(batch, params = params)
        return {metric_name: metric(relevances.squeeze(), batch["label"]) 
                for metric_name, metric in self.rel_metrics.items()}

    @partial(jax.jit, static_argnums = (0, 1))
    def _eval_step_clicks(self, model, params, batch):
        click_scores = model.forward(batch, params = params, train=False).click
        return {metric_name: metric(click_scores.squeeze(), 
                                    batch["click"],) 
                for metric_name, metric in self.click_metrics.items()}


