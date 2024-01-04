from typing import Optional

import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor, IntTensor
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, BertForPreTraining, BertLayer
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class BertModel(BertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: PretrainedConfig):
        super(BertModel, self).__init__(config)
        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_loss = CrossEntropyLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding

    def get_mlm_loss(self, sequence_output: FloatTensor, labels: LongTensor):
        token_scores = self.mlm_head(sequence_output)

        return self.mlm_loss(
            token_scores.view(-1, self.config.vocab_size),
            labels.view(-1),
        )


class CrossEncoder(BertModel):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. To reproduce the original
    model released by Baidu, we use clicks or annotations as the relevance signal.
    """

    def __init__(self, config: PretrainedConfig):
        super(CrossEncoder, self).__init__(config)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.click_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        clicks: Optional[FloatTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        if clicks is not None:
            click_scores = self.cls(query_document_embedding).squeeze()
            loss += self.click_loss(click_scores, clicks)

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding


class Condenser(BertModel):
    """
    Condenser setup for dense retrieval: https://arxiv.org/pdf/2104.08253.pdf
    Unsupervised pre-training technique to encourage BERT to leverage the CLS token.
    """

    def __init__(self, config: PretrainedConfig):
        super(Condenser, self).__init__(config)
        self.head_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.head_layers)]
        )

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            output_hidden_states=True,
            return_dict=True,
        )

        cls_late = outputs.hidden_states[-1][:, :1]
        tokens_early = outputs.hidden_states[self.config.num_early_layers][:, 1:]
        sequence_output = torch.cat([cls_late, tokens_early], dim=1)

        # Extend initial attention mask to match the number of attention heads:
        head_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device
        )

        for layer in self.head_layers:
            sequence_output = layer(sequence_output, head_attention_mask)[0]

        # Compute MLM loss after condenser head layers
        loss += self.get_mlm_loss(sequence_output, labels)

        if self.config.do_late_mlm:
            # In addition, add the original mlm
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, cls_late
