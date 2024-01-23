from tqdm import tqdm
import numpy as np
import jax
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
        key = jax.random.PRNGKey(self.seed)
        key, init_key = jax.random.split(key, 2)

        init_batch = next(iter(loader))
        init_batch["position"] = np.arange(25)
        state = TrainState.create(
            apply_fn=model.get_relevance_score,
            params=model.init(init_key, init_batch),
            tx = optax.adamw(5e-5),
        )
        checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=state)

        for batch in tqdm(loader, total = 7008, disable=not self.progress_bar):
            metrics.append(self._eval_step(state, batch))

        return {key: np.mean([m[key] for m in metrics]) for key in self.metrics.keys()}

    @partial(jax.jit, static_argnums = (0,))
    def _eval_step(self, state, batch):
        relevances = state.apply_fn(batch, params = state.params)
        return {metric_name: metric(relevances, batch["label"]) 
                for metric_name, metric in self.metrics.items()}


