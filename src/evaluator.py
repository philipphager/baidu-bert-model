from tqdm import tqdm
import numpy as np
import jax
from functools import partial

from flax.training import checkpoints
from torch.utils.data import DataLoader

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
    ) -> dict:
        metrics = []
        params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=None)["params"]

        for batch in tqdm(loader, total = 7008, disable=not self.progress_bar):
            metrics.append(self._eval_step_rels(model, params, batch))

        return {key: np.mean([m[key] for m in metrics]) for key in self.rel_metrics.keys()}

    @partial(jax.jit, static_argnums = (0, 1))
    def _eval_step_rels(self, model, params, batch):
        relevances = model.predict_relevance(batch, params = params)
        return {metric_name: metric(relevances.squeeze(), batch["label"]) 
                for metric_name, metric in self.rel_metrics.items()}

    def eval_clicks(
        self,
        model,
        loader: DataLoader,
    ) -> dict:
        metrics = []
        params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=None)["params"]

        for batch in tqdm(loader, disable=not self.progress_bar):
            metrics.append(self._eval_step_clicks(model, params, batch))

        return {key: np.mean([m[key] for m in metrics]) for key in self.click_metrics.keys()}

    @partial(jax.jit, static_argnums = (0, 1))
    def _eval_step_clicks(self, model, params, batch):
        relevances = model.predict_relevance(batch, params = params)
        return {metric_name: metric(relevances.squeeze(), batch["label"]) 
                for metric_name, metric in self.click_metrics.items()}


