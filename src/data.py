import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import LongTensor, IntTensor
from torch.utils.data import IterableDataset

from src.const import (
    TrainColumns,
    QueryColumns,
    TOKEN_OFFSET,
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    IGNORE_LABEL_TOKEN,
)


class BaiduTrainDataset(IterableDataset):
    def __init__(
        self,
        files: List[Path],
        max_sequence_length: int,
        masking_rate: float,
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
    ):
        self.files = files
        self.max_sequence_length = max_sequence_length
        self.masking_rate = masking_rate
        self.special_tokens = special_tokens
        self.segment_types = segment_types

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

                        tokens, token_types = preprocess(
                            query=query,
                            title=title,
                            abstract=abstract,
                            max_tokens=self.max_sequence_length,
                            special_tokens=self.special_tokens,
                            segment_types=self.segment_types,
                        )

                        attention_mask = tokens > self.special_tokens["PAD"]
                        masked_tokens, labels = mask(
                            tokens,
                            token_types,
                            self.segment_types,
                            self.special_tokens,
                            self.masking_rate,
                        )

                        yield {
                            "tokens": masked_tokens,
                            "attention_mask": attention_mask,
                            "token_types": token_types,
                            "labels": labels,
                            "clicks": click,
                        }

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


def mask(
    tokens: LongTensor,
    token_types: LongTensor,
    segment_types: Dict[str, int],
    special_tokens: Dict[str, int],
    rate: float,
) -> Tuple[LongTensor, LongTensor]:
    tokens = tokens.clone()

    # Mask title and abstract with a given probability, the query is never masked:
    masking_probability = torch.full_like(tokens, fill_value=rate, dtype=torch.float)
    is_doc = token_types == segment_types["TEXT"]
    is_not_seperator = tokens != special_tokens["SEP"]
    should_mask = torch.bernoulli(masking_probability).bool()
    mask = is_doc & is_not_seperator & should_mask

    # Create labels for the MLM prediction task. All non-masked tokens will be
    # marked as -100 to be ignored by the torch cross entropy loss:
    labels = tokens.clone()
    labels[~mask] = IGNORE_LABEL_TOKEN

    # Apply token mask:
    tokens[mask] = special_tokens["MASK"]

    return tokens, labels
