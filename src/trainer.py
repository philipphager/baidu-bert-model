import logging
from functools import partial
from typing import Tuple

import flax
import jax
import optax
import wandb
from chex import PRNGKey
from flax.training.common_utils import shard
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.struct import CrossEncoderLoss

logger = logging.getLogger("rich")


class Trainer:
    def __init__(
        self,
        seed: int,
        learning_rate: float,
        weight_decay: float,
        max_steps: int,
        log_metrics: bool,
        log_steps: int,
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
        self.log_steps = log_steps
        self.mean_losses = None

    def train(
        self,
        model,
        train_loader: DataLoader,
    ) -> TrainState:
        key = jax.random.PRNGKey(self.seed)
        key, init_key = jax.random.split(key, 2)
        state = TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=self.optimizer,
        )
        state = flax.jax_utils.replicate(state)

        for step, batch in enumerate(tqdm(train_loader, disable=not self.progress_bar)):
            if step == self.max_steps:
                break

            state, losses = self._train_step(model, state, shard(batch), step, key)
            self.track(step, losses)

        return flax.jax_utils.unreplicate(state)

    def track(self, step: int, losses: CrossEncoderLoss):
        self.mean_losses = (
            self.mean_losses + losses.mean()
            if self.mean_losses is not None
            else losses.mean()
        )

        if self.log_metrics and step % self.log_steps == 0:
            metrics = {"train/global_step": step}

            for metric, loss in self.mean_losses.__dict__.items():
                mean_loss = jax.device_get(loss / min(step + 1, self.log_steps))
                metrics[f"train/{metric}"] = mean_loss

            wandb.log(metrics)
            self.mean_losses = None

    @partial(
        jax.pmap,
        axis_name="batch",
        in_axes=(None, None, 0, 0, None, None),
        static_broadcasted_argnums=(0, 1),
    )
    def _train_step(
        self,
        model,
        state: TrainState,
        batch: dict,
        step: int,
        key: PRNGKey,
    ) -> Tuple[TrainState, CrossEncoderLoss]:
        dropout_rng = jax.random.fold_in(key=key, data=step)

        def loss_fn(params):
            outputs = state.apply_fn(
                batch=batch,
                params=params,
                train=True,
                dropout_rng=dropout_rng,
            )
            losses = model.get_loss(outputs, batch)
            return losses.loss, losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
        return state, jax.lax.pmean(losses, axis_name="batch")
