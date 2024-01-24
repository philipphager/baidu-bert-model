from typing import Tuple, Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.random import KeyArray
import rax


from transformers import FlaxBertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertLMPredictionHead


class BertModel(FlaxBertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__(config)
        self.mlm_head = FlaxBertLMPredictionHead(config=config)
        self.mlm_loss = optax.softmax_cross_entropy_with_integer_labels

    def forward(
        self,
        batch: dict,
        params: dict,
    ) -> Tuple[dict, Any]:
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

        return {"logits": logits}, query_document_embedding

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
        mlm_params = self.mlm_head.init(key, outputs[0])

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
            "mlm_head": mlm_params,
        }

    def get_training_loss(self, outputs: Dict, batch: Dict) -> Array:
        return self.get_mlm_loss(outputs, batch)

    def get_mlm_loss(self, outputs: Dict, batch: Dict) -> Array:
        logits = outputs["logits"]
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
        self.click_loss = optax.sigmoid_binary_cross_entropy

    def forward(
        self,
        batch: Dict,
        params: Dict,
    ) -> Tuple[Dict, Any]:
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
        click_probs = self.click_head.apply(
            params["click_head"], query_document_embedding
        )

        return {"logits": logits, "click_probs": click_probs}, query_document_embedding

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

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)
        click_loss = self.click_loss(
            outputs["click_probs"].reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()
        return mlm_loss + click_loss
    
    def get_relevance_score(self, batch: dict, params: dict) -> Array:
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
        click_probs = self.click_head.apply(
            params["click_head"], query_document_embedding
        )
        return click_probs


class ListwiseCrossEncoder(CrossEncoder):
    "BERT-based cross-encoder with listwise click loss"

    def __init__(self, config: BertConfig):
        super(ListwiseCrossEncoder, self).__init__(config)
        self.click_loss = rax.softmax_loss

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = self.click_loss(
                outputs["click_probs"].reshape(-1), 
                batch["clicks"].reshape(-1), 
                where = (batch["query_ids"] != -1), 
                segments = batch["query_ids"])
        return mlm_loss + click_loss

class PBMCrossEncoder(CrossEncoder):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. We use debiased clicks as 
    the relevance signal.
    """

    def __init__(self, config: BertConfig):
        super(PBMCrossEncoder, self).__init__(config)
        self.propensities = nn.Embed(25, 1)

    def forward(
        self,
        batch: Dict,
        params: Dict,
    ) -> Tuple[Dict, Any]:
        outputs, query_document_embedding = super(PBMCrossEncoder, self).forward(batch, params)
        outputs["exam"] = self.propensities.apply(params["propensities"], batch["positions"])
        return outputs, query_document_embedding

    def init(self, key: KeyArray, batch: Dict) -> Dict:
        ce_key, prop_key = jax.random.split(key, 2)
        params = super(PBMCrossEncoder, self).init(ce_key, batch)
        params["propensities"] = self.propensities.init(prop_key, batch["positions"])
        return params

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)
        click_loss = self.click_loss(
            outputs["click_probs"].reshape(-1) + outputs["exam"].reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()
        return mlm_loss + click_loss


class ListwisePBMCrossEncoder(PBMCrossEncoder):
    "BERT-based cross-encoder with listwise click loss"

    def __init__(self, config: BertConfig):
        super(ListwisePBMCrossEncoder, self).__init__(config)
        self.click_loss = rax.softmax_loss

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = self.click_loss(
                outputs["click_probs"].reshape(-1) + outputs["exam"].reshape(-1), 
                batch["clicks"].reshape(-1), 
                where = (batch["query_ids"] != -1), 
                segments = batch["query_ids"])
        return mlm_loss + click_loss

class IPSCrossEncoder(CrossEncoder):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. We use debiased clicks as 
    the relevance signal.
    """

    def __init__(self, config: BertConfig, propensities_path: str):
        super(IPSCrossEncoder, self).__init__(config)
        self.propensities = jnp.load(propensities_path)

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)
        click_loss = self.click_loss(
            outputs["click_probs"].reshape(-1) + self.propensities[batch["positions"]].reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()
        return mlm_loss + click_loss


class ListwiseIPSCrossEncoder(IPSCrossEncoder):
    "BERT-based cross-encoder with listwise click loss"

    def __init__(self, config: BertConfig, propensities_path: str):
        super(ListwiseIPSCrossEncoder, self).__init__(config, propensities_path)
        self.click_loss = rax.softmax_loss

    def get_training_loss(self, outputs: dict, batch: dict) -> Array:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = self.click_loss(
                outputs["click_probs"].reshape(-1) + self.propensities[batch["positions"]].reshape(-1), 
                batch["clicks"].reshape(-1), 
                where = (batch["query_ids"] != -1), 
                segments = batch["query_ids"])
        return mlm_loss + click_loss
    
