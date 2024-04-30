import flax.linen as nn
import rax

from src.model.base import FlaxPreTrainedCrossEncoder
from src.model.cross_encoder import (
    CrossEncoderModule,
    CrossEncoderLoss,
)
from src.model.loss import masked_language_modeling_loss
from src.model.struct import PBMCrossEncoderOutput


class PBMCrossEncoderModule(CrossEncoderModule):
    def setup(self):
        super().setup()
        self.propensities = nn.Embed(50, 1)

    def __call__(
        self,
        batch,
        train=True,
        **kwargs,
    ):
        cross_encoder_output = super().__call__(batch, train, **kwargs)

        examination = self.propensities(batch["positions"]).reshape(-1)
        relevance = cross_encoder_output.relevance
        click = examination + relevance

        return PBMCrossEncoderOutput(
            click=click,
            relevance=relevance,
            examination=examination,
            logits=cross_encoder_output.logits,
            query_document_embedding=cross_encoder_output.query_document_embedding,
        )


class PBMCrossEncoder(FlaxPreTrainedCrossEncoder):
    """
    MonoBERT cross-encoder training the CLS token on click prediction
    """

    module_class = PBMCrossEncoderModule

    def get_loss(self, outputs: PBMCrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
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
