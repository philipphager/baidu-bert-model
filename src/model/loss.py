from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import optax
import rax
from jax import Array
from jax._src.lax.lax import stop_gradient
from rax._src.segment_utils import (
    segment_softmax,
    first_item_segment_mask,
    same_segment_mask,
)
from rax._src.utils import normalize_probabilities

from src.model.struct import BertOutput, CrossEncoderOutput, DLACrossEncoderOutput


def masked_language_modeling_loss(outputs: BertOutput, batch: Dict) -> Array:
    logits = outputs.logits
    labels = batch["labels"]

    # Tokens with label -100 are ignored during the CE computation
    label_mask = jnp.where(labels != -100, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask

    return loss.sum() / label_mask.sum()


def pointwise_sigmoid_ips(
    outputs: CrossEncoderOutput,
    batch: Dict,
    max_weight: float,
    eps: float = 1.0e-9,
) -> Array:
    """
    Pointwise IPS loss as in Bekker et al.:
    https://arxiv.org/pdf/1809.03207.pdf
    and Saito et al.:
    https://dl.acm.org/doi/abs/10.1145/3336191.3371783
    """
    weights = 1 / outputs.examination
    weights = weights.clip(min=0, max=max_weight)

    labels = weights * batch["clicks"].reshape(-1)
    scores = jax.nn.sigmoid(outputs.relevance.reshape(-1))
    log_p = jnp.log(scores.clip(min=eps))
    log_not_p = jnp.log((1 - scores).clip(min=eps))

    return (-labels * log_p - (1.0 - labels) * log_not_p).mean()


def listwise_softmax_ips(
    outputs: CrossEncoderOutput, batch: Dict, max_weight: float
) -> Array:
    weights = 1 / outputs.examination
    weights = weights.clip(max=max_weight)
    labels = weights * batch["clicks"]

    return rax.softmax_loss(
        scores=outputs.relevance.reshape(-1),
        labels=labels.reshape(-1),
        label_fn=partial(normalize_probabilities, segments=batch["query_id"]),
        segments=batch["query_id"],
    )


def dual_learning_algorithm(
    outputs: DLACrossEncoderOutput, batch: Dict, max_weight: float
):
    examination_weights = _normalize_weights(
        outputs.examination.reshape(-1),
        max_weight=max_weight,
        segments=batch["query_id"],
        softmax=True,
    )

    relevance_weights = _normalize_weights(
        outputs.relevance.reshape(-1),
        max_weight=max_weight,
        segments=batch["query_id"],
        softmax=True,
    )

    examination_labels = examination_weights * batch["clicks"].reshape(-1)
    relevance_labels = relevance_weights * batch["clicks"].reshape(-1)

    examination_loss = rax.softmax_loss(
        scores=outputs.examination.reshape(-1),
        labels=relevance_labels,
        label_fn=partial(normalize_probabilities, segments=batch["query_id"]),
        segments=batch["query_id"],
    )
    relevance_loss = rax.softmax_loss(
        scores=outputs.relevance.reshape(-1),
        labels=examination_labels,
        label_fn=partial(normalize_probabilities, segments=batch["query_id"]),
        segments=batch["query_id"],
    )

    return examination_loss, relevance_loss


def _normalize_weights(
    scores: Array,
    max_weight: float,
    segments: Array,
    softmax: bool = False,
) -> Array:
    """
    Converts logits to normalized propensity weights by:
    1. [Optional] Apply a softmax to all scores in a ranking (ignoring masked values)
    2. Normalize the resulting probabilities by the first item in each ranking
    3. Invert propensities to obtain weights: e.g., propensity 0.5 -> weight 2
    4. [Optional] Clip the final weights to reduce variance
    """

    if softmax:
        probabilities = segment_softmax(scores, segments=segments)
    else:
        probabilities = scores

    # Normalize propensities by the item in first position in each segment and convert propensities
    # to weights by computing weights as 1 / propensities:
    fism = first_item_segment_mask(segments)
    ssm = same_segment_mask(segments)
    weights = (
        probabilities
        @ jnp.where(fism[:, jnp.newaxis], ssm, jnp.zeros_like(ssm))
        / probabilities
    )
    # Mask padding and apply clipping
    weights = weights.clip(min=0, max=max_weight)

    return stop_gradient(weights)
