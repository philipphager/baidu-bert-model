from enum import IntEnum
import rax
from functools import partial

MAX_SEQUENCE_LENGTH = 128
MASKING_RATE = 0.4
IGNORE_LABEL_TOKEN = -100
TOKEN_OFFSET = 10

SPECIAL_TOKENS = {
    "PAD": 0,
    "SEP": 1,
    "CLS": 2,
    "MASK": 3,
}

SEGMENT_TYPES = {
    "QUERY": 0,
    "TEXT": 1,
    "PAD": 1,
}

MISSING_TITLE = b"21429"
WHAT_OTHER_PEOPLE_SEARCHED_TITLE = b"3742\x0111492\x0112169\x015061\x0116905"

REL_METRICS = {
    "DCG@1": partial(rax.dcg_metric, topn=1),
    "DCG@3": partial(rax.dcg_metric, topn=3),
    "DCG@5": partial(rax.dcg_metric, topn=5),
    "DCG@10": partial(rax.dcg_metric, topn=10),
    "MRR@10": partial(rax.mrr_metric, topn=10),
    "nDCG@10": partial(rax.ndcg_metric, topn=10),
}

CLICK_METRICS = {
    "log-likelihood": rax.pointwise_sigmoid_loss,
}


class QueryColumns(IntEnum):
    QID = 0
    QUERY = 1
    QUERY_REFORMULATION = 2


class TrainColumns(IntEnum):
    POS = 0
    URL_MD5 = 1
    TITLE = 2
    ABSTRACT = 3
    MULTIMEDIA_TYPE = 4
    CLICK = 5
    SKIP = 8
    SERP_HEIGHT = 9
    DISPLAYED_TIME = 10
    DISPLAYED_TIME_MIDDLE = 11
    FIRST_CLICK = 12
    DISPLAYED_COUNT = 13
    SERO_MAX_SHOW_HEIGHT = 14
    SLIPOFF_COUNT_AFTER_CLICK = 15
    DWELLING_TIME = 16
    DISPLAYED_TIME_TOP = 17
    SERO_TO_TOP = 18
    DISPLAYED_COUNT_TOP = 19
    DISPLAYED_COUNT_BOTTOM = 20
    SLIPOFF_COUNT = 21
    FINAL_CLICK = 23
    DISPLAYED_TIME_BOTTOM = 24
    CLICK_COUNT = 25
    DISPLAYED_COUNT_2 = 26
    LAST_CLICK = 28
    REVERSE_DISPLAY_COUNT = 29
    DISPLAYED_COUNT_MIDDLE = 30


class TestColumns(IntEnum):
    QUERY = 0
    TITLE = 1
    ABSTRACT = 2
    LABEL = 3
    BUCKET = 4
