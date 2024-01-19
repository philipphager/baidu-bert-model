
import enum
import logging
from typing import Tuple
from functools import partial

import jax
import flax
import optax
import wandb
from flax.training.train_state import TrainState
from flax.training.common_utils import shard
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
        progress_bar: bool = True,
        **kwargs,
    ):
        self.seed = seed
        self.optimizer = optax.adamw(learning_rate=5e-5, b1=0.9, b2=0.98, eps=1e-8, weight_decay=weight_decay)
        self.progress_bar = progress_bar

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

        for step, batch in enumerate(tqdm(train_loader, disable=not self.progress_bar)):
            state, loss = self._train_step(model, state, shard(batch))

            if step % 1000 == 0:
                wandb.log({"train/loss": jax.device_get(loss.mean()), "train/global_step": step})

        return state        

    @partial(jax.pmap, axis_name="batch", in_axes = (None, None, 0, 0), static_broadcasted_argnums = (0,1))
    def _train_step(self, model, state: TrainState, batch: dict) -> Tuple[TrainState, jax.Array]:
        def loss_fn(params):
            outputs, _ = state.apply_fn(batch, params = params)
            return model.get_training_loss(outputs, batch)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name = "batch"))
        return state, jax.lax.pmean(loss, axis_name = "batch")
