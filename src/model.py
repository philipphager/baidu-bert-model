from typing import Optional

import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor, IntTensor
from transformers import PretrainedConfig, BertForPreTraining


class BertModel(BertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

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

        if self.config.do_mlm_task:
            assert labels is not None, "Expected labels of masked tokens for MLM task"
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
    model released by Baidu, we use clicks as the relevance signal.
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

        if self.config.do_ctr_task:
            assert clicks is not None, "Expected click labels for CTR task"
            click_scores = self.cls(query_document_embedding).squeeze()
            loss += self.click_loss(click_scores, clicks)

        if self.config.do_mlm_task:
            assert labels is not None, "Expected labels of masked tokens for MLM task"
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding


class Condenser(BertModel):
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
        cls_late = outputs.pooler_output

        if self.config.do_condenser_task:
            cls_late = outputs.hidden_states[-1][:, :1]
            tokens_early = outputs.hidden_states[self.config.num_early_layers][:, 1:]
            head_sequence_output = torch.cat([cls_late, tokens_early], dim=1)

            # Extend initial attention mask to match the number of attention heads:
            head_attention_mask = self.bert.get_extended_attention_mask(
                attention_mask, attention_mask.shape, attention_mask.device
            )

            for layer in self.head_layers:
                head_sequence_output = layer(head_sequence_output, head_attention_mask)[
                    0
                ]

            # Compute MLM loss after condenser head layers
            loss += self.get_mlm_loss(head_sequence_output, labels)

        if self.config.do_mlm_task:
            # In addition, add the original mlm
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, cls_late
