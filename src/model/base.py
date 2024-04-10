from typing import Tuple, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import FlaxPreTrainedModel, BertConfig


class FlaxPreTrainedCrossEncoder(FlaxPreTrainedModel):
    """
    A base class to handle weight initialization and downloading pretrained models.

    We need to extend FlaxPreTrainedModel to handle custom inputs to our BERT model.
    Implementation based on FlaxBertPreTrainedModel:
    https://github.com/huggingface/transformers/blob/1773afcec338c2b1a741a86b7431ad10be4518c7/src/transformers/models/bert/modeling_flax_bert.py#L760
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    main_input_name = "input_ids"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config=config,
            module=module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self,
        rng: jax.random.PRNGKey,
        input_shape: Tuple,
        params: FrozenDict = None,
    ) -> FrozenDict:
        # Init a dummy input batch:
        tokens = jnp.zeros(input_shape, dtype="i4")
        token_types = jnp.zeros_like(tokens)
        attention_mask = jnp.ones_like(tokens)
        positions = jnp.arange(input_shape[0], dtype="i4")
        query_id = jnp.arange(input_shape[0], dtype="i4")

        batch = {
            "tokens": tokens,
            "token_types": token_types,
            "attention_mask": attention_mask,
            "positions": positions,
            "query_id": query_id,
        }

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            batch,
        )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        batch: Dict,
        train: bool,
        params: Dict = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
    ):
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        return self.module.apply(
            inputs,
            batch=batch,
            train=train,
            rngs=rngs,
        )
