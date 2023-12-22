from typing import Optional

from torch import nn, LongTensor, FloatTensor, BoolTensor, IntTensor
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class MonoBERT(BertForPreTraining):
    def __init__(self, config: PretrainedConfig):
        super(MonoBERT, self).__init__(config)
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

        if clicks is not None:
            click_scores = self.cls(query_document_embedding).squeeze()
            loss += self.click_loss(
                click_scores,
                clicks,
            )

        if labels is not None:
            token_scores = self.mlm_head(outputs[0])

            loss += self.mlm_loss(
                token_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return loss, query_document_embedding
