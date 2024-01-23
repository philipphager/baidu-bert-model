import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import flax.struct
import numpy as np
from torch.utils.data import IterableDataset

from src.const import (
    TrainColumns,
    QueryColumns,
    TOKEN_OFFSET,
    IGNORE_LABEL_TOKEN,
)

@flax.struct.dataclass
class CrossEncoderPretrainDataset(IterableDataset):
    files: List[Path]
    max_sequence_length: int
    masking_rate: float
    mask_query: bool
    mask_doc: bool
    special_tokens: Dict[str, int]
    segment_types: Dict[str, int]
    ignored_titles: List[bytes]

    def __iter__(self):
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
                        position = int(columns[TrainColumns.POS])

                        if title in set(self.ignored_titles):
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

                        attention_mask = tokens > self.special_tokens["PAD"]
                        masked_tokens, labels = self.mask(tokens, token_types)

                        yield {
                            "tokens": masked_tokens,
                            "attention_mask": attention_mask,
                            "token_types": token_types,
                            "labels": labels,
                            "clicks": click,
                            "positions": position,
                        }

    def mask(
        self,
        tokens: np.ndarray,
        token_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokens = tokens.copy()

        # Mask title and abstract with a given probability, the query is never masked:
        masking_probability = np.full_like(
            tokens, fill_value=self.masking_rate, dtype=float
        )

        mask = np.random.binomial(1, p=masking_probability).astype(bool)
        # Ignore all special tokens in masking procedure:
        mask[tokens < TOKEN_OFFSET] = False

        if not self.mask_query:
            mask[token_types == self.segment_types["QUERY"]] = False

        if not self.mask_doc:
            mask[token_types == self.segment_types["TEXT"]] = False

        # Create labels for the MLM prediction task. All non-masked tokens will be
        # marked as -100 to be ignored by the torch cross entropy loss:
        labels = tokens.copy()
        labels[~mask] = IGNORE_LABEL_TOKEN

        # Apply token mask:
        tokens[mask] = self.special_tokens["MASK"]

        return tokens, labels

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
) -> Tuple[np.ndarray, np.ndarray]:
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

    tokens = np.array(tokens, dtype=int)
    token_types = np.array(token_types, dtype=int)

    return tokens, token_types

def collate_for_eval(
        batch: List[dict], 
        max_tokens: int, 
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
    ) -> dict:
    b = batch[0]
    collated = {"tokens": [], "attention_mask": [], "token_types": []}
    n_docs = len(b["label"])
    
    for k in range(n_docs):
        query_tokens = [special_tokens["CLS"]] + b["query"] + [special_tokens["SEP"]]
        doc_tokens = b["title"][k] + [special_tokens["SEP"]] + b["abstract"][k] + [special_tokens["SEP"]]
        tokens = query_tokens + doc_tokens
        tokens = tokens[:max_tokens] + max(max_tokens - len(tokens), 0) * [special_tokens["PAD"]]
        collated["tokens"].append(np.asarray(tokens))

        query_token_types = [segment_types["QUERY"]] * len(query_tokens)
        doc_token_types = [segment_types["TEXT"]] * len(doc_tokens)
        token_types = query_token_types + doc_token_types
        token_types = token_types[:max_tokens] + max(max_tokens - len(token_types), 0) * [segment_types["PAD"]]
        collated["token_types"].append(np.asarray(token_types))

        collated["attention_mask"].append(np.asarray(tokens) > special_tokens["PAD"])

    return {
        "tokens": np.stack(collated["tokens"], axis = 0),
        "attention_mask": np.stack(collated["attention_mask"], axis = 0),
        "token_types": np.stack(collated["token_types"], axis = 0),
        "label": np.asarray(b["label"]),
        "frequency_bucket": b["frequency_bucket"],
    }
