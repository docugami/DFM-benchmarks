import collections

from sentence_transformers import SentenceTransformer, util
from torch.types import Number

from docugami_dfm_benchmarks.utils.text import get_tokens

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def semantic_similarity(text1: str, text2: str) -> Number:
    """Compute semantic similarity (cosine) between embeddings of given texts."""
    embedding_1 = embedding_model.encode(text1, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()


def compute_f1(text1: str, text2: str) -> float:
    gold_toks = get_tokens(text1)
    pred_toks = get_tokens(text2)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
