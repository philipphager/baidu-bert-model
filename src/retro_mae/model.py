from typing import Optional

import torch
from torch import LongTensor, BoolTensor
from transformers import PretrainedConfig

from src.decoder import BertLayerForDecoder
from src.model import BertModel


class RetroMAE(BertModel):
    def __init__(self, config: PretrainedConfig):
        super(RetroMAE, self).__init__(config)
        self.decoder_embeddings = self.bert.embeddings
        self.decoder = BertLayerForDecoder(config)
        self.decoder.apply(self._init_weights)

    def forward(
        self,
        encoder_tokens: LongTensor,
        encoder_attention_mask: BoolTensor,
        encoder_labels: Optional[LongTensor] = None,
        decoder_tokens: Optional[LongTensor] = None,
        decoder_attention_mask: Optional[BoolTensor] = None,
        decoder_labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0
        encoder_out = self.bert(
            input_ids=encoder_tokens,
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        encoder_cls = encoder_out.hidden_states[-1][:, :1]

        if encoder_labels is not None:
            loss += self.get_mlm_loss(encoder_out[0], encoder_labels)

        if decoder_labels is not None:
            decoder_embeddings = self.decoder_embeddings(input_ids=decoder_tokens)
            # Replace decoder CLS with encoder CLS
            decoder_embeddings = decoder_embeddings[:, 1:]
            decoder_embeddings = torch.cat([encoder_cls, decoder_embeddings], dim=1)

            # EQ. 4: https://aclanthology.org/2022.emnlp-main.35.pdf
            position_embeddings = self._get_position_embeddings(decoder_tokens)
            query = position_embeddings + encoder_cls

            matrix_attention_mask = self.bert.get_extended_attention_mask(
                decoder_attention_mask,
                decoder_attention_mask.shape,
                decoder_attention_mask.device,
            )

            decoder_out = self.decoder(
                query=query,
                key=decoder_embeddings,
                value=decoder_embeddings,
                attention_mask=matrix_attention_mask,
            )
            loss += self.get_mlm_loss(decoder_out[0], decoder_labels)

        return loss, encoder_cls

    def _get_position_embeddings(self, tokens: LongTensor):
        position_ids = self.bert.embeddings.position_ids[:, : tokens.size(1)]
        return self.bert.embeddings.position_embeddings(position_ids)
