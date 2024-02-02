import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import flax.struct
import numpy as np
import torch
from jax.tree_util import tree_map
from torch.utils.data import Dataset, IterableDataset
from sklearn.model_selection import train_test_split

from src.const import (
    TrainColumns,
    QueryColumns,
    MISSING_TITLE,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
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
        query_no = 0

        for file in files:
            with gzip.open(file, "rb") as f:
                query = None

                for i, line in enumerate(f):
                    columns = line.strip(b"\n").split(b"\t")
                    is_query = len(columns) <= 3

                    if is_query:
                        query_no += 1
                        position = 0
                        query = columns[QueryColumns.QUERY]
                    else:
                        title = columns[TrainColumns.TITLE]
                        abstract = columns[TrainColumns.ABSTRACT]
                        click = float(columns[TrainColumns.CLICK])

                        if title == MISSING_TITLE:
                            # Drop results with "-" as content. The dropped item
                            # is reflected in the item position, e.g.: 1, [drop], 3, ...
                            position += 1
                            continue
                        if title == WHAT_OTHER_PEOPLE_SEARCHED_TITLE:
                            # Skipping item "what other people searched for". The skip
                            # is not reflected in the item position: 1, [skip], 2, ...
                            continue
                        position += 1

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
                            "query_id": query_no,
                            "tokens": masked_tokens,
                            "attention_mask": attention_mask,
                            "token_types": token_types,
                            "labels": labels,
                            "clicks": click,
                            "positions": position,
                        }

    def mask(
        self, tokens: np.ndarray, token_types: np.ndarray
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

    @staticmethod
    def collate_fn(batch):
        return tree_map(np.asarray, torch.utils.data.default_collate(batch))


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


def collate_for_rels(
        batch: List[dict], 
        max_tokens: int, 
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
    ) -> dict:
    collated = {"tokens": [], 
                "attention_mask": [], 
                "token_types": [],
                "label": [],
                "frequency_bucket":[],}
    

    for b in batch:
        query_tokens = np.concatenate([np.asarray([special_tokens["CLS"]]), 
                                        b["query"], 
                                        np.asarray([special_tokens["SEP"]])])
        query_token_types = [segment_types["QUERY"]] * len(query_tokens)

        collated["label"].append(b["label"])
        collated["frequency_bucket"].append(b["frequency_bucket"])
        
        for k in range(b["n"]):
            doc_tokens = np.concatenate([b["title"][k], 
                                        np.asarray([special_tokens["SEP"]]), 
                                        b["abstract"][k], 
                                        np.asarray([special_tokens["SEP"]])])
            tokens = np.concatenate([query_tokens, doc_tokens])
            tokens = np.concatenate([tokens[:max_tokens], 
                                    np.full(max(max_tokens - len(tokens), 0) , special_tokens["PAD"])])
            collated["tokens"].append(tokens)

            doc_token_types = [segment_types["TEXT"]] * len(doc_tokens)
            token_types = query_token_types + doc_token_types
            token_types = token_types[:max_tokens] + max(max_tokens - len(token_types), 0) * [segment_types["PAD"]]
            collated["token_types"].append(np.asarray(token_types))

            collated["attention_mask"].append(np.asarray(tokens) > special_tokens["PAD"])

    return {
        "tokens": np.stack(collated["tokens"], axis=0),
        "attention_mask": np.stack(collated["attention_mask"], axis=0),
        "token_types": np.stack(collated["token_types"], axis=0),
        "labels": np.concatenate(collated["label"]),
        "frequency_bucket": np.asarray(collated["frequency_bucket"]),
    }

def collate_for_clicks(
    batch: List[dict], 
    max_tokens: int, 
    special_tokens: Dict[str, int],
    segment_types: Dict[str, int],
) -> dict:
    collated = {"query_id": [], 
                "tokens": [], 
                "attention_mask": [], 
                "token_types": [],
                "click": [],
                "positions": [],}
    
    for b in batch:
        query_tokens = np.concatenate([np.asarray([special_tokens["CLS"]]), 
                                    b["query"], 
                                    np.asarray([special_tokens["SEP"]])])
        query_token_types = [segment_types["QUERY"]] * len(query_tokens)

        for k in range(b["n"]):
            doc_tokens = np.concatenate([b["title"][k], 
                                        np.asarray([special_tokens["SEP"]]), 
                                        b["abstract"][k], 
                                        np.asarray([special_tokens["SEP"]])])
            tokens = np.concatenate([query_tokens, doc_tokens])
            tokens = np.concatenate([tokens[:max_tokens], 
                                    np.full(max(max_tokens - len(tokens), 0) , special_tokens["PAD"])])
            collated["tokens"].append(tokens)

            doc_token_types = [segment_types["TEXT"]] * len(doc_tokens)
            token_types = query_token_types + doc_token_types
            token_types = token_types[:max_tokens] + max(max_tokens - len(token_types), 0) * [segment_types["PAD"]]
            collated["token_types"].append(np.asarray(token_types))

            collated["attention_mask"].append(tokens > special_tokens["PAD"])
            collated["query_id"].append(b["query_id"])
            collated["click"].append(b["click"][k])
            collated["position"].append(b["position"][k])

    return {
        "query_id": np.asarray(collated["query_id"]),
        "tokens": np.stack(collated["tokens"], axis=0),
        "attention_mask": np.stack(collated["attention_mask"], axis=0),
        "token_types": np.stack(collated["token_types"], axis=0),
        "clicks": np.asarray(collated["click"]),
        "positions": np.asarray(collated["position"]),
    }

class LabelEncoder:
    def __init__(self):
        self.value2id = {}
        self.max_id = 1

    def __call__(self, x):
        if x not in self.value2id:
            self.value2id[x] = self.max_id
            self.max_id += 1

        return self.value2id[x]

    def __len__(self):
        return len(self.value2id)
    
def random_split(
    dataset: Dataset,
    shuffle: bool,
    random_state: int,
    test_size: float,
    stratify: Optional[str] = None,
):
    """
    Stratify a train/test split of a Huggingface dataset.
    While huggingface implements stratification, this function enables stratification
    on all columns, not only the dataset's class label.
    """
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        idx,
        stratify=dataset[stratify] if stratify else None,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state,
    )
    return dataset.select(train_idx), dataset.select(test_idx)