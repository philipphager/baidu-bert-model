
import flax.struct
from jax import Array
import jax.numpy as jnp


@flax.struct.dataclass
class Output:
    click: Array
    relevance: Array

@flax.struct.dataclass
class BertOutput:
    logits: Array
    query_document_embedding: Array

@flax.struct.dataclass
class BertLoss:
    loss: Array = jnp.zeros(1)
    mlm_loss: Array = jnp.zeros(1)

    def add(self, losses):
        return self.__class__(
            loss = self.loss + losses.loss,
            mlm_loss = self.mlm_loss + losses.mlm_loss
        )
    
    def mean(self):
        return self.__class__(
            loss = self.loss.mean(),
            mlm_loss = self.mlm_loss.mean()
        )


@flax.struct.dataclass
class CrossEncoderOutput(BertOutput, Output):
    click: Array
    relevance: Array
    logits: Array
    query_document_embedding: Array

@flax.struct.dataclass
class CrossEncoderLoss(BertLoss):
    loss: Array = jnp.zeros(1)
    mlm_loss: Array = jnp.zeros(1)
    click_loss: Array = jnp.zeros(1)

    def add(self, losses):
        return self.__class__(
            loss = self.loss + losses.loss,
            mlm_loss = self.mlm_loss + losses.mlm_loss,
            click_loss = self.click_loss + losses.click_loss,
        )
    
    def mean(self):
        return self.__class__(
            loss = self.loss.mean(),
            mlm_loss = self.mlm_loss.mean(),
            click_loss = self.click_loss.mean(),
        )


@flax.struct.dataclass
class PBMCrossEncoderOutput(CrossEncoderOutput):
    click: Array
    relevance: Array
    examination: Array
    logits: Array
    query_document_embedding: Array