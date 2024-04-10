from pathlib import Path
from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
import pandas as pd
from jax import Array

from src.model.base import FlaxPreTrainedCrossEncoder
from src.model.cross_encoder import (
    CrossEncoderModule,
    CrossEncoderLoss,
)
from src.model.loss import (
    masked_language_modeling_loss,
    pointwise_sigmoid_ips,
    listwise_softmax_ips,
)
from src.model.struct import IPSCrossEncoderOutput


class PretrainedExaminationModel(nn.Module):
    file: str
    positions: int

    def setup(self):
        assert Path(self.file).exists()

        df = pd.read_csv(self.file)
        model = jnp.zeros(self.positions)
        # Load propensities, position 0 is used for padding and has propensity 0:
        positions = df["position"].values
        propensities = df.iloc[:, 1].values
        self.model = model.at[positions].set(propensities)

    def __call__(self, batch: Dict, training: bool) -> Array:
        return self.model[batch["positions"]]


class IPSCrossEncoderModule(CrossEncoderModule):
    propensity_path = "propensities/global_all_pairs.csv"
    positions = 50

    def setup(self):
        super().setup()
        self.propensities = PretrainedExaminationModel(
            file=self.propensity_path,
            positions=self.positions,
        )

    def __call__(
        self,
        batch,
        train=True,
        **kwargs,
    ):
        cross_encoder_output = super().__call__(batch, train, **kwargs)

        examination = self.propensities(batch, training=train)
        relevance = cross_encoder_output.relevance

        return IPSCrossEncoderOutput(
            relevance=relevance,
            examination=examination,
            logits=cross_encoder_output.logits,
            query_document_embedding=cross_encoder_output.query_document_embedding,
        )


class IPSCrossEncoder(FlaxPreTrainedCrossEncoder):
    module_class = IPSCrossEncoderModule
    max_weight = 10.0

    def get_loss(self, outputs: IPSCrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = masked_language_modeling_loss(outputs, batch)
        click_loss = pointwise_sigmoid_ips(outputs, batch, max_weight=self.max_weight)

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )


class ListwiseIPSCrossEncoder(FlaxPreTrainedCrossEncoder):
    module_class = IPSCrossEncoderModule
    max_weight = 10.0

    def get_loss(self, outputs: IPSCrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = masked_language_modeling_loss(outputs, batch)
        click_loss = listwise_softmax_ips(outputs, batch, max_weight=self.max_weight)

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )
