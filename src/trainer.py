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
from chex import PRNGKey
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import BertLoss

logger = logging.getLogger("rich")


class Stage(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Trainer:
    def __init__(
        self,
        seed: int,
        learning_rate: float,
        weight_decay: float,
        max_steps: int,
        log_metrics: bool,
        progress_bar: bool = True,
        **kwargs,
    ):
        self.seed = seed
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.98,
            eps=1e-8,
            weight_decay=weight_decay,
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
        self.mean_losses = model.loss_dataclass()

        for step, batch in enumerate(tqdm(train_loader, disable=not self.progress_bar)):
            if step == self.max_steps:
                break
                
            state, losses = self._train_step(model, state, shard(batch))
            self.track(step, losses)

        return flax.jax_utils.unreplicate(state)
    
    def track(self, step: int, losses: BertLoss):
        self.mean_losses = self.mean_losses.add(losses.mean())

        if self.log_metrics and step % 1000 == 0:
            wandb.log({"train/global_step": step} | {f"train/{k}": jax.device_get(v / min(step + 1, 1000)) 
                                                        for k,v in self.mean_losses.__dict__.items()})
            self.mean_losses = losses.__class__()

    @partial(
        jax.pmap,
        axis_name="batch",
        in_axes=(None, None, 0, 0),
        static_broadcasted_argnums=(0, 1),
    )
    def _train_step(
            self, model, state: TrainState, batch: dict, step: int, key: PRNGKey
    ) -> Tuple[TrainState]:
        dropout_rng = jax.random.fold_in(key=key, data=step)

        def loss_fn(params):
            outputs = state.apply_fn(
                batch,
                params=params,
                train=True,
                rngs={"dropout": dropout_rng}
            )
            losses = model.get_loss(outputs, batch)
            return losses.loss, losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
        return state, jax.lax.pmean(losses, axis_name="batch")
