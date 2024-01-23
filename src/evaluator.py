from tqdm import tqdm
import numpy as np
import jax
import wandb
import optax
from functools import partial

from flax.training.train_state import TrainState
from flax.training import checkpoints
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(
        self,
        metrics: dict,
        ckpt_dir: str,
        seed: int = 2024,
        progress_bar: bool = True,
        **kwargs,
    ):
        self.metrics = metrics
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.progress_bar = progress_bar
    
    def eval(
        self,
        model,
        loader: DataLoader,
    ) -> dict:
        metrics = []
        params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=None)["params"]

        for batch in tqdm(loader, total = 7008, disable=not self.progress_bar):
            if len(batch["label"]) < 2:
                continue
            metrics.append(self._eval_step(model, params, batch))
            wandb.log({key: np.mean([m[key] for m in metrics]) for key in self.metrics.keys()})

        return {key: np.mean([m[key] for m in metrics]) for key in self.metrics.keys()}

    @partial(jax.jit, static_argnums = (0, 1))
    def _eval_step(self, model, params, batch):
        relevances = model.get_relevance_score(batch, params = params)
        return {metric_name: metric(relevances.squeeze(), batch["label"]) 
                for metric_name, metric in self.metrics.items()}


