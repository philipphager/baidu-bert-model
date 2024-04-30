from src.model.base import FlaxPreTrainedCrossEncoder
from src.model.cross_encoder import (
    CrossEncoderLoss,
)
from src.model.loss import masked_language_modeling_loss, dual_learning_algorithm
from src.model.pbm import PBMCrossEncoderModule
from src.model.struct import DLALoss, DLACrossEncoderOutput


class DLACrossEncoder(FlaxPreTrainedCrossEncoder):
    module_class = PBMCrossEncoderModule
    max_weight = 10.0

    def get_loss(self, outputs: DLACrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = masked_language_modeling_loss(outputs, batch)
        examination_loss, relevance_loss = dual_learning_algorithm(
            outputs, batch, max_weight=self.max_weight
        )

        return DLALoss(
            loss=mlm_loss + examination_loss + relevance_loss,
            mlm_loss=mlm_loss,
            click_loss=examination_loss + relevance_loss,
            examination_loss=examination_loss,
            relevance_loss=relevance_loss,
        )
