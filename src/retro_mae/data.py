import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import LongTensor, IntTensor, BoolTensor
from torch.utils.data import IterableDataset

from src.const import (
    TrainColumns,
    QueryColumns,
    TOKEN_OFFSET,
    IGNORE_LABEL_TOKEN,
)


class RetroMAEPretrainDataset(IterableDataset):
    def __init__(
        self,
        files: List[Path],
        max_sequence_length: int,
        encoder_masking_rate: float,
        decoder_masking_rate: float,
        decoder_attention_rate: float,
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
        ignored_titles: List[bytes],
    ):
        self.files = files
        self.max_sequence_length = max_sequence_length
        self.encoder_masking_rate = encoder_masking_rate
        self.decoder_masking_rate = decoder_masking_rate
        self.decoder_attention_rate = decoder_attention_rate
        self.special_tokens = special_tokens
        self.segment_types = segment_types
        self.ignored_titles = set(ignored_titles)

    def __iter__(self) -> Tuple[LongTensor, LongTensor, LongTensor, int]:
        files = self.get_local_files()

        for file in files:
            with gzip.open(file, "rb") as f:
                query = None

                for i, line in enumerate(f):
                    columns = line.strip(b"\n").split(b"\t")
                    is_query = len(columns) <= 3

                    if is_query:
                        query = columns[QueryColumns.QUERY]
                    else:
                        title = columns[TrainColumns.TITLE]
                        abstract = columns[TrainColumns.ABSTRACT]
                        click = float(columns[TrainColumns.CLICK])

                        if title in self.ignored_titles:
                            # Skipping documents during training based on their titles.
                            # Used to ignore missing or navigational items.
                            continue

                        tokens, token_types = preprocess(
                            query=query,
                            title=title,
                            abstract=abstract,
                            max_tokens=self.max_sequence_length,
                            special_tokens=self.special_tokens,
                            segment_types=self.segment_types,
                        )

                        encoder_attention_mask = tokens > self.special_tokens["PAD"]
                        decoder_attention_mask = self.position_based_attention_mask(
                            tokens, self.decoder_attention_rate
                        )

                        encoder_tokens, encoder_labels = self.mask(
                            tokens, self.encoder_masking_rate
                        )
                        decoder_tokens, decoder_labels = self.mask(
                            tokens, self.decoder_masking_rate
                        )

                        yield {
                            "encoder_tokens": encoder_tokens,
                            "encoder_attention_mask": encoder_attention_mask,
                            "encoder_labels": encoder_labels,
                            "decoder_tokens": decoder_tokens,
                            "decoder_attention_mask": decoder_attention_mask,
                            "decoder_labels": decoder_labels,
                        }

    def mask(
        self,
        tokens: LongTensor,
        masking_rate: float,
    ) -> Tuple[LongTensor, LongTensor]:
        tokens = tokens.clone()

        # Mask title and abstract with a given probability, the query is never masked:
        masking_probability = torch.full_like(
            tokens, fill_value=masking_rate, dtype=torch.float
        )

        mask = torch.bernoulli(masking_probability).bool()
        # Ignore all special tokens in masking procedure:
        mask[tokens < TOKEN_OFFSET] = False

        # Create labels for the MLM prediction task. All non-masked tokens will be
        # marked as -100 to be ignored by the torch cross entropy loss:
        labels = tokens.clone()
        labels[~mask] = IGNORE_LABEL_TOKEN

        # Apply token mask:
        tokens[mask] = self.special_tokens["MASK"]

        return tokens, labels

    def position_based_attention_mask(
        self,
        tokens: torch.LongTensor,
        attention_rate: float,
    ) -> BoolTensor:
        """
        Sample a token x token matrix indicating which tokens can be attended during
        decoding. The idea is to give each token a slightly different context
        to enhance the decoder training:

        1. All tokens can attend the CLS
        2. No token can attend itself
        3. Padding cannot be attended.

        The % of tokens that can be attended is its own hyperparameter.
        RetroMAE defines this implicitly based on the encoder masking rate:
        https://github.com/staoxiao/RetroMAE/blob/67a06a2ade9065c7d7cd6c2934f9a745b8fcae2b/src/pretrain/data.py#L47
        """
        attention_rate = torch.full(
            size=(len(tokens), len(tokens)),
            fill_value=attention_rate,
        )
        # Sample attention mask, tokens marked as 1 can be attended:
        attention_mask = torch.bernoulli(attention_rate)

        # Tokens shall not attend to themselves:
        attention_mask.fill_diagonal_(0)
        # All tokens can access to the CLS token in first position:
        attention_mask[0, :] = 1
        attention_mask[:, 0] = 1
        # Padding shall not be attended:
        padding = tokens == self.special_tokens["PAD"]
        attention_mask[padding, :] = 0
        attention_mask[:, padding] = 0

        return attention_mask.bool()

    def get_local_files(self):
        """
        Select a subset of files to iterate, based on the current worker process.
        See: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        info = torch.utils.data.get_worker_info()

        if info is None:
            worker_num = 1
            worker_id = 0
        else:
            worker_num = info.num_workers
            worker_id = info.id

        return [f for i, f in enumerate(self.files) if i % worker_num == worker_id]


def split_idx(text: bytes, offset: int) -> List[int]:
    """
    Split tokens in Baidu dataset and convert to integer token ids.
    """
    return [int(t) + offset for t in text.split(b"\x01") if len(t.strip()) > 0]


def preprocess(
    query: bytes,
    title: bytes,
    abstract: bytes,
    max_tokens: int,
    special_tokens: Dict[str, int],
    segment_types: Dict[str, int],
) -> Tuple[LongTensor, IntTensor]:
    """
    Format BERT model input as:
    [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    query_idx = split_idx(query, TOKEN_OFFSET)
    title_idx = split_idx(title, TOKEN_OFFSET)
    abstract_idx = split_idx(abstract, TOKEN_OFFSET)

    query_tokens = [special_tokens["CLS"]] + query_idx + [special_tokens["SEP"]]
    query_token_types = [segment_types["QUERY"]] * len(query_tokens)

    text_tokens = title_idx + [special_tokens["SEP"]]
    text_tokens += abstract_idx + [special_tokens["SEP"]]
    text_token_types = [segment_types["TEXT"]] * len(text_tokens)

    tokens = query_tokens + text_tokens
    token_types = query_token_types + text_token_types

    padding = max(max_tokens - len(tokens), 0)
    tokens = tokens[:max_tokens] + padding * [special_tokens["PAD"]]
    token_types = token_types[:max_tokens] + padding * [segment_types["PAD"]]

    tokens = torch.tensor(tokens, dtype=torch.long)
    token_types = torch.tensor(token_types, dtype=torch.int)

    return tokens, token_types
