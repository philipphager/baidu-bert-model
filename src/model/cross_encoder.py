import flax.linen as nn
import jax.numpy as jnp
import rax
from transformers import BertConfig
from transformers.models.bert.modeling_flax_bert import (
    FlaxBertModule,
    FlaxBertLMPredictionHead,
)

from src.model.base import FlaxPreTrainedCrossEncoder
from src.model.loss import masked_language_modeling_loss
from src.model.struct import CrossEncoderOutput, CrossEncoderLoss


class CrossEncoderModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.bert = FlaxBertModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.mlm_head = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)
        self.click_head = nn.Dense(1)

    def __call__(
        self,
        batch,
        train=True,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            deterministic=not train,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # Masked language modeling task
        logits = self.mlm_head(outputs.last_hidden_state)

        # Click / relevance prediction based on the CLS token
        query_document_embedding = outputs.last_hidden_state[:, 0]
        click_scores = self.click_head(query_document_embedding)

        return CrossEncoderOutput(
            click=click_scores,
            relevance=click_scores,
            logits=logits,
            query_document_embedding=query_document_embedding,
        )


class CrossEncoder(FlaxPreTrainedCrossEncoder):
    """
    Naive MonoBERT cross-encoder training the CLS token to naively predict clicks
    without any bias correction.
    """

    module_class = CrossEncoderModule

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = masked_language_modeling_loss(outputs, batch)
        click_loss = rax.pointwise_sigmoid_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class ListwiseCrossEncoder(FlaxPreTrainedCrossEncoder):
    """
    Naive MonoBERT cross-encoder training the CLS token to naively trained on clicks
    with a listwise softmax cross-entropy loss.
    """

    module_class = CrossEncoderModule

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = masked_language_modeling_loss(outputs, batch)
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
