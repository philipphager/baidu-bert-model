
import flax.struct
from jax import Array
import jax.numpy as jnp


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
class CrossEncoderOutput(BertOutput):
    logits: Array
    click_probs: Array
    query_document_embedding: Array

@flax.struct.dataclass
class CrossEncoderLoss(BertLoss):
    loss: Array = jnp.zeros(1)
    mlm_loss: Array = jnp.zeros(1)
    click_loss: Array = jnp.zeros(1)

    def add(self, loss):
        self.loss += loss.loss
        self.mlm_loss += loss.mlm_loss
        self.click_loss += loss.click_loss
    
    def mean(self):
        self.loss = self.loss.mean()
        self.click_loss = self.click_loss.mean()
    
    def reset(self):
        self.loss = jnp.zeros(1)
        self.mlm_loss = jnp.zeros(1)
        self.click_loss = jnp.zeros(1)


@flax.struct.dataclass
class PBMCrossEncoderOutput(CrossEncoderOutput):
    logits: Array
    click_probs: Array
    query_document_embedding: Array
    examination: Array