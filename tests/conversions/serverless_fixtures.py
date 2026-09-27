from google.protobuf.message import Message

from qdrant_client.serverless.grpc import serverless_collections_pb2 as pb2

# Dense vectors: every Distance and every PrecisionTier member is covered, so a
# mis-mapped enum constant cannot hide behind a self-consistent round-trip
# (_*_FROM_GRPC is derived by inverting _*_TO_GRPC).
dense_vector = pb2.DenseVectorConfig(size=1536, distance=pb2.COSINE)
dense_vector_multivector = pb2.DenseVectorConfig(
    size=128, distance=pb2.DOT, multivector=True, precision_tier=pb2.LOW
)
dense_vector_manhattan = pb2.DenseVectorConfig(
    size=4, distance=pb2.MANHATTAN, precision_tier=pb2.MEDIUM
)
dense_vector_euclid = pb2.DenseVectorConfig(size=4, distance=pb2.EUCLID, precision_tier=pb2.HIGH)

sparse_vector = pb2.SparseVectorConfig(use_idf=True)
sparse_vector_tier = pb2.SparseVectorConfig(use_idf=False, precision_tier=pb2.HIGH)

payload_index_keyword = pb2.PayloadIndexConfig(keyword=pb2.KeywordIndex())
payload_index_integer = pb2.PayloadIndexConfig(integer=pb2.IntegerIndex())
# `range=False` is set, not absent: catches `if range_:` in place of `is not None`
payload_index_integer_falsy = pb2.PayloadIndexConfig(
    integer=pb2.IntegerIndex(lookup=True, range=False)
)
payload_index_float = pb2.PayloadIndexConfig(float=pb2.FloatIndex())
payload_index_uuid = pb2.PayloadIndexConfig(uuid=pb2.UuidIndex())
payload_index_datetime = pb2.PayloadIndexConfig(datetime=pb2.DatetimeIndex())
payload_index_text = pb2.PayloadIndexConfig(text=pb2.TextIndex())
# every optional TextIndex field set, with falsy values where they are legal
payload_index_text_full = pb2.PayloadIndexConfig(
    text=pb2.TextIndex(
        tokenizer=pb2.MULTILINGUAL,
        lowercase=False,
        phrase_matching=True,
        min_token_len=0,
        max_token_len=20,
    )
)
payload_index_text_prefix = pb2.PayloadIndexConfig(text=pb2.TextIndex(tokenizer=pb2.PREFIX))
payload_index_text_whitespace = pb2.PayloadIndexConfig(
    text=pb2.TextIndex(tokenizer=pb2.WHITESPACE)
)
payload_index_text_word = pb2.PayloadIndexConfig(text=pb2.TextIndex(tokenizer=pb2.WORD))
payload_index_geo = pb2.PayloadIndexConfig(geo=pb2.GeoIndex())
payload_index_bool = pb2.PayloadIndexConfig(bool=pb2.BoolIndex())

stopwords_empty = pb2.StopwordsSet()
stopwords_languages = pb2.StopwordsSet(languages=["english", "german"])
stopwords_custom = pb2.StopwordsSet(languages=["english"], custom=["foo", "bar"])

stemmer_snowball = pb2.StemmingAlgorithm(snowball=pb2.SnowballParams(language="english"))
stemmer_disabled = pb2.StemmingAlgorithm(disabled=pb2.DisabledStemmer())

# presence on an empty submessage: `prefix` set but carrying no fields
payload_index_keyword_prefix = pb2.PayloadIndexConfig(
    keyword=pb2.KeywordIndex(prefix=pb2.KeywordPrefixParams())
)
payload_index_text_analysis = pb2.PayloadIndexConfig(
    text=pb2.TextIndex(
        ascii_folding=True,
        stopwords=stopwords_custom,
        stemmer=stemmer_snowball,
    )
)
payload_index_text_stemmer_disabled = pb2.PayloadIndexConfig(
    text=pb2.TextIndex(
        ascii_folding=False,
        stopwords=stopwords_empty,
        stemmer=stemmer_disabled,
    )
)

collection_config = pb2.CollectionConfig(
    dense_vectors={"": dense_vector},
    payload_indexes={"user_id": payload_index_keyword},
)
collection_config_named_vectors = pb2.CollectionConfig(
    dense_vectors={"dense": dense_vector, "colbert": dense_vector_multivector},
    sparse_vectors={"bm25": sparse_vector},
    payload_indexes={"age": payload_index_integer_falsy, "text": payload_index_text_full},
)
collection_config_empty = pb2.CollectionConfig()

fixtures: dict[str, list[Message]] = {
    "dense_vector": [
        dense_vector,
        dense_vector_multivector,
        dense_vector_manhattan,
        dense_vector_euclid,
    ],
    "sparse_vector": [
        sparse_vector,
        sparse_vector_tier,
    ],
    "stopwords": [
        stopwords_empty,
        stopwords_languages,
        stopwords_custom,
    ],
    "stemmer": [
        stemmer_snowball,
        stemmer_disabled,
    ],
    "payload_index": [
        payload_index_keyword,
        payload_index_keyword_prefix,
        payload_index_integer,
        payload_index_integer_falsy,
        payload_index_float,
        payload_index_uuid,
        payload_index_datetime,
        payload_index_text,
        payload_index_text_full,
        payload_index_text_prefix,
        payload_index_text_whitespace,
        payload_index_text_word,
        payload_index_text_analysis,
        payload_index_text_stemmer_disabled,
        payload_index_geo,
        payload_index_bool,
    ],
    "collection_config": [
        collection_config,
        collection_config_named_vectors,
        collection_config_empty,
    ],
}
