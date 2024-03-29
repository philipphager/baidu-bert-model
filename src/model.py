from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import rax
from functools import partial
from jax import Array
from jax._src.lax.lax import stop_gradient
from jax.random import KeyArray
from rax._src.segment_utils import segment_softmax, first_item_segment_mask, same_segment_mask
from rax._src.utils import normalize_probabilities
from transformers import FlaxBertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig

from src.struct import BertLoss, CrossEncoderLoss, DLALoss
from src.struct import BertOutput, CrossEncoderOutput, PBMCrossEncoderOutput


class BertModel(FlaxBertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__(config)
        self.mlm_loss = optax.softmax_cross_entropy_with_integer_labels
        self.loss_dataclass = BertLoss

    def forward(
        self,
        batch: dict,
        params: dict,
        train: bool,
        **kwargs,
    ) -> BertOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=not train,
            **kwargs,
        )

        return BertOutput(
            logits=outputs.prediction_logits,
            query_document_embedding=outputs.hidden_states[-1][:, 0],
        )

    def init(self, key: KeyArray, batch: dict) -> dict:
        outputs = self.module.apply(
            {"params": self.params},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            return_dict=True,
        )

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
        }

    def get_loss(self, outputs: BertOutput, batch: Dict) -> BertLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)
        return BertLoss(
            loss=mlm_loss,
            mlm_loss=mlm_loss,
        )

    def get_mlm_loss(self, outputs: BertOutput, batch: Dict) -> Array:
        logits = outputs.logits
        labels = batch["labels"]

        # Tokens with label -100 are ignored during the CE computation
        label_mask = jax.numpy.where(labels != -100, 1.0, 0.0)
        loss = self.mlm_loss(logits, labels) * label_mask

        return loss.sum() / label_mask.sum()


class CrossEncoder(BertModel):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. To reproduce the original
    model released by Baidu, we use clicks or annotations as the relevance signal.
    """

    def __init__(self, config: BertConfig):
        super(CrossEncoder, self).__init__(config)
        self.click_head = nn.Dense(1)
        self.loss_dataclass = CrossEncoderLoss

    def forward(
        self,
        batch: Dict,
        params: Dict,
        train: bool,
        **kwargs,
    ) -> CrossEncoderOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=not train,
            **kwargs,
        )

        query_document_embedding = outputs.hidden_states[-1][:, 0]

        click_scores = self.click_head.apply(
            params["click_head"], query_document_embedding
        )

        return CrossEncoderOutput(
            click=click_scores,
            relevance=click_scores,
            logits=outputs.prediction_logits,
            query_document_embedding=query_document_embedding,
        )

    def init(self, key: KeyArray, batch: Dict) -> Dict:
        outputs = self.module.apply(
            {"params": self.params},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
        )

        key, click_key = jax.random.split(key, 2)
        query_document_embedding = outputs.hidden_states[-1][:, 0]
        click_params = self.click_head.init(click_key, query_document_embedding)

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
            "click_head": click_params,
        }

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = rax.pointwise_sigmoid_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )

    def predict_relevance(self, batch: Dict, params: Dict) -> Array:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=True,
        )
        query_document_embedding = outputs.hidden_states[-1][:, 0]
        click_scores = self.click_head.apply(
            params["click_head"], query_document_embedding
        )
        return click_scores


class ListwiseCrossEncoder(CrossEncoder):
    """
    BERT-based cross-encoder with listwise click loss
    """

    def __init__(self, config: BertConfig):
        super(ListwiseCrossEncoder, self).__init__(config)

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = rax.softmax_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
            segments=batch["query_id"],
        )

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class PBMCrossEncoder(CrossEncoder):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. We use debiased clicks as
    the relevance signal.
    """

    def __init__(self, config: BertConfig):
        super(PBMCrossEncoder, self).__init__(config)
        self.propensities = nn.Embed(50, 1)

    def forward(
        self,
        batch: Dict,
        params: Dict,
        train: bool,
        **kwargs,
    ) -> PBMCrossEncoderOutput:
        cse = super(PBMCrossEncoder, self).forward(batch, params, train, **kwargs)
        examination = self.propensities.apply(
            params["propensities"],
            batch["positions"],
        )
        click = examination + cse.relevance

        return PBMCrossEncoderOutput(
            click=click,
            relevance=cse.relevance,
            examination=examination,
            logits=cse.logits,
            query_document_embedding=cse.query_document_embedding,
        )

    def init(self, key: KeyArray, batch: Dict) -> Dict:
        ce_key, prop_key = jax.random.split(key, 2)
        params = super(PBMCrossEncoder, self).init(ce_key, batch)
        params["propensities"] = self.propensities.init(prop_key, batch["positions"])
        return params

    def get_loss(self, outputs: PBMCrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = rax.pointwise_sigmoid_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class ListwisePBMCrossEncoder(PBMCrossEncoder):
    """
    BERT-based cross-encoder with listwise click loss
    """

    def __init__(self, config: BertConfig):
        super(ListwisePBMCrossEncoder, self).__init__(config)

    def get_loss(self, outputs: PBMCrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = rax.softmax_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
            segments=batch["query_id"],
        )

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class IPSCrossEncoder(CrossEncoder):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. We use debiased clicks as
    the relevance signal.
    """

    def __init__(self, config: BertConfig, propensities_path: str):
        super(IPSCrossEncoder, self).__init__(config)
        self.propensities = self.get_propensities(propensities_path)
        self.max_weight = 10

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        examination = self.propensities[batch["positions"]].reshape(-1)

        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = self.pointwise_sigmoid_ips(
            examination=examination,
            relevance=outputs.click.reshape(-1),
            labels=batch["clicks"].reshape(-1),
            max_weight=self.max_weight,
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )

    @staticmethod
    def pointwise_sigmoid_ips(
        examination: Array,
        relevance: Array,
        labels: Array,
        max_weight: float = 10,
        eps: float = 1.0e-9,
    ) -> Array:
        """
        Pointwise IPS loss as in Bekker et al.:
        https://arxiv.org/pdf/1809.03207.pdf
        and Saito et al.:
        https://dl.acm.org/doi/abs/10.1145/3336191.3371783
        """
        weights = 1 / examination
        weights = weights.clip(min=0, max=max_weight)

        scores = jax.nn.sigmoid(relevance)
        log_p = jnp.log(scores.clip(min=eps))
        log_not_p = jnp.log((1 - scores).clip(min=eps))

        return -(weights * labels) * log_p - (1.0 - (weights * labels)) * log_not_p

    @staticmethod
    def get_propensities(path, positions=50):
        propensities = jnp.zeros(positions)
        data = jnp.load(path)
        return propensities.at[1 : len(data) + 1].set(data)


class ListwiseIPSCrossEncoder(IPSCrossEncoder):
    """
    BERT-based cross-encoder with listwise click loss
    """

    def __init__(self, config: BertConfig, propensities_path: str):
        super(ListwiseIPSCrossEncoder, self).__init__(config, propensities_path)

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        weights = 1 / self.propensities[batch["positions"]].reshape(-1)
        weights = weights.clip(max=self.max_weight)

        labels = weights * batch["clicks"].reshape(-1)

        click_loss = rax.softmax_loss(
            scores=outputs.click.reshape(-1),
            labels=labels,
            label_fn=partial(normalize_probabilities, segments=batch["query_id"]),
            segments=batch["query_id"],
        )

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class ListwiseDLACrossEncoder(ListwisePBMCrossEncoder):
    """
    Implementation of the Dual Learning Algorithm from Ai et al, 2018: https://arxiv.org/pdf/1804.05938.pdf
    """

    def __init__(self, config: BertConfig):
        super(ListwiseDLACrossEncoder, self).__init__(config)
        self.max_weight = 10

    def get_loss(self, outputs: PBMCrossEncoderOutput, batch: dict) -> DLALoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        examination_weights = self._normalize_weights(
            outputs.examination.reshape(-1),
            self.max_weight,
            segments=batch["query_id"],
            softmax=True,
        )

        relevance_weights = self._normalize_weights(
            outputs.relevance.reshape(-1),
            self.max_weight,
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

        return DLALoss(
            loss=mlm_loss + examination_loss + relevance_loss,
            mlm_loss=mlm_loss,
            click_loss=examination_loss + relevance_loss,
            examination_loss=examination_loss,
            relevance_loss=relevance_loss,
        )

    def _normalize_weights(
        self,
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
        weights =  probabilities @ jnp.where(fism[:, jnp.newaxis], ssm, jnp.zeros_like(ssm)) / probabilities
        # Mask padding and apply clipping
        weights = weights.clip(min=0, max=max_weight)

        return stop_gradient(weights)
