import enum
import logging
from functools import partial
from typing import Tuple

import flax
import jax
from jax import Array
import optax
import wandb
from flax.training.common_utils import shard
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("rich")


class Stage(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Trainer:
    def __init__(
        self,
        seed: int,
        weight_decay: float,
        max_steps: int,
        log_metrics: bool,
        progress_bar: bool = True,
        **kwargs,
    ):
        self.seed = seed
        self.optimizer = optax.adamw(
            learning_rate=5e-5, b1=0.9, b2=0.98, eps=1e-8, weight_decay=weight_decay
        )
        self.max_steps = max_steps
        self.progress_bar = progress_bar
        self.log_metrics = log_metrics

    def train(
        self,
        model,
        train_loader: DataLoader,
    ) -> TrainState:
        key = jax.random.PRNGKey(self.seed)
        key, init_key = jax.random.split(key, 2)
        init_batch = next(iter(train_loader))
        state = TrainState.create(
            apply_fn=model.forward,
            params=model.init(init_key, init_batch),
            tx=self.optimizer,
        )
        state = flax.jax_utils.replicate(state)
        self.init_track(model)

        for step, batch in enumerate(tqdm(train_loader, disable=not self.progress_bar)):
            if step == self.max_steps:
                break
                
            state, loss, losses = self._train_step(model, state, shard(batch))
            self.track(step, loss, losses, len(batch["tokens"]))

        return flax.jax_utils.unreplicate(state)
    
    def init_track(self, model):
        self.mean_loss = jax.numpy.zeros(1)
        self.mean_losses = {l: jax.numpy.zeros(1) for l in model.losses}
        self.mean_batch_size = 0
    
    def track(self, step: int, loss: Array, losses: dict, batch_size: int):
        self.mean_loss += loss.mean()
        self.mean_losses = {k: v + losses[k].mean() for k,v in self.mean_losses.items()} 
        self.mean_batch_size += batch_size

        if self.log_metrics and step % 1000 == 0:
            wandb.log(
                {
                    **{
                        "train/loss": jax.device_get(self.mean_loss / min(step + 1, 1000)),
                        "train/batch_size": self.mean_batch_size / min(step + 1, 1000),
                        "train/global_step": step,
                    },
                    **{
                    f"train/{k}": jax.device_get(v / min(step + 1, 1000)) for k,v in self.mean_losses.items()
                    }
                }
            )
            mean_loss = jax.numpy.zeros(1)   

    @partial(
        jax.pmap,
        axis_name="batch",
        in_axes=(None, None, 0, 0),
        static_broadcasted_argnums=(0, 1),
    )
    def _train_step(
        self, model, state: TrainState, batch: dict
    ) -> Tuple[TrainState, jax.Array, dict]:
        def loss_fn(params):
            outputs, _ = state.apply_fn(batch, params=params)
            return model.get_training_loss(outputs, batch)

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
        return state, jax.lax.pmean(loss, axis_name="batch"), jax.lax.pmean(losses, axis_name="batch")
