from typing import Optional

from torch import nn, LongTensor, FloatTensor, BoolTensor, IntTensor
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class BERTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_mlm_task = kwargs.pop("do_mlm_task", False)
        self.do_ctr_task = kwargs.pop("do_ctr_task", False)


class BERT(BertForPreTraining):
    def __init__(self, config: BERTConfig):
        super(BERT, self).__init__(config)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_loss = CrossEntropyLoss()
        self.click_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        clicks: Optional[FloatTensor] = None,
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
            loss += self.click_loss(
                click_scores,
                clicks,
            )

        if self.config.do_mlm_task:
            assert clicks is not None, "Expected labels of masked tokens for MLM task"
            sequence_output = outputs[0]
            token_scores = self.mlm_head(sequence_output)

            loss += self.mlm_loss(
                token_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return loss, query_document_embedding
