from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import rax
from jax import Array
from jax.random import KeyArray
from transformers import FlaxBertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig

from src.struct import BertLoss, CrossEncoderLoss
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
    ) -> BertOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
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
    ) -> CrossEncoderOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            return_dict=True,
        )
        sequence_output, query_document_embedding = outputs[:2]
        logits = self.mlm_head.apply(params["mlm_head"], sequence_output)
        click_scores = self.click_head.apply(
            params["click_head"], query_document_embedding
        )

        return CrossEncoderOutput(
            click=click_scores,
            relevance=click_scores,
            logits=logits,
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
            return_dict=True,
        )
        mlm_key, click_key = jax.random.split(key, 2)
        mlm_params = self.mlm_head.init(mlm_key, outputs[0])
        click_params = self.click_head.init(click_key, outputs[1])

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
            "mlm_head": mlm_params,
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
            return_dict=True,
        )
        _, query_document_embedding = outputs[:2]
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
    ) -> PBMCrossEncoderOutput:
        cse = super(PBMCrossEncoder, self).forward(batch, params)
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
        mlm_loss = self.get_mlm_loss(outputs, batch)

        weights = 1 / self.propensities[batch["positions"]].reshape(-1)
        weights = weights.clip(max=self.max_weight)

        click_loss = rax.pointwise_sigmoid_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
            weights=weights,
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )

    @staticmethod
    def get_propensities(path, positions=50):
        propensities = jnp.zeros(positions)
        data = jnp.load(path)
        return propensities.at[1:len(data) + 1].set(data)


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

        click_loss = rax.softmax_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
            weights=weights,
            segments=batch["query_id"],
        )

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )
