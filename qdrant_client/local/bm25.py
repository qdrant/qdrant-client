import math
from collections import Counter
from typing import Any

from qdrant_client.http import models
from qdrant_client.local.payload_filters import unindexed_text_tokens

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
# the server keeps this many leading bits of a document length, so lengths below 2^13 = 8192
# tokens are exact
DOC_LEN_BITS = 13


def has_default_tokenization(params: models.TextIndexParams) -> bool:
    """Whether the text index tokenizes like the default word tokenizer, the only one supported"""
    return (
        params.tokenizer in (None, models.TokenizerType.WORD)
        and params.lowercase in (None, True)
        and not params.ascii_folding
        and params.stopwords is None
        and (params.stemmer is None or isinstance(params.stemmer, models.DisabledStemmerParams))
        and params.min_token_len is None
        and params.max_token_len is None
        and params.nested is None
    )


def document_tokens(values: list[Any] | None) -> list[str]:
    return [
        token
        for value in values or []
        if isinstance(value, str)
        for token in unindexed_text_tokens(value)
    ]


def round_doc_len(length: int) -> int:
    # a longer length loses the bits past its leading DOC_LEN_BITS, which rounds it down to a
    # multiple of 2 from 8192 tokens, of 4 from 16384, of 8 from 32768, and so on
    lost_bits = max(length.bit_length() - DOC_LEN_BITS, 0)
    step = 2**lost_bits
    return length // step * step


def bm25_scores(
    documents: dict[int, list[str]], query: str, k1: float, b: float
) -> dict[int, float]:
    """BM25 score of every document holding at least one query term

    Args:
        documents: tokens of each document with at least one token, the corpus of the statistics
        query: query text, a repeated term counts once
        k1: term frequency saturation
        b: document length normalization
    """
    terms = set(unindexed_text_tokens(query))
    if not terms or not documents:
        return {}

    frequencies = {idx: Counter(tokens) for idx, tokens in documents.items()}
    lengths = {idx: round_doc_len(len(tokens)) for idx, tokens in documents.items()}
    num_docs = len(documents)
    avg_doc_len = sum(lengths.values()) / num_docs

    idf = {}
    for term in terms:
        df = sum(1 for frequency in frequencies.values() if term in frequency)
        idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

    scores = {}
    for idx, frequency in frequencies.items():
        norm = k1 * (1 - b + b * lengths[idx] / avg_doc_len)
        score = 0.0
        for term in terms:
            tf = frequency[term]
            if tf > 0:
                score += idf[term] * tf * (k1 + 1) / (tf + norm)
        if score > 0:
            scores[idx] = score
    return scores
