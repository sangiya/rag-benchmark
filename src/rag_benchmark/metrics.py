from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalEvaluation:
    query: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]


@dataclass
class GenerationEvaluation:
    query: str
    answer: str
    context: str


@dataclass
class MetricResult:
    name: str
    score: float
    details: dict


def context_precision(evaluation: RetrievalEvaluation) -> MetricResult:
    """Fraction of retrieved documents that are relevant (precision@k)."""
    if not evaluation.retrieved_ids:
        return MetricResult("context_precision", 0.0, {"retrieved": 0, "relevant_retrieved": 0})

    relevant_retrieved = sum(
        1 for doc_id in evaluation.retrieved_ids if doc_id in evaluation.relevant_ids
    )
    score = relevant_retrieved / len(evaluation.retrieved_ids)
    return MetricResult(
        "context_precision",
        score,
        {"retrieved": len(evaluation.retrieved_ids), "relevant_retrieved": relevant_retrieved},
    )


def context_recall(evaluation: RetrievalEvaluation) -> MetricResult:
    """Fraction of relevant documents that were retrieved (recall@k)."""
    if not evaluation.relevant_ids:
        return MetricResult("context_recall", 1.0, {"relevant": 0, "relevant_retrieved": 0})

    relevant_retrieved = sum(
        1 for doc_id in evaluation.retrieved_ids if doc_id in evaluation.relevant_ids
    )
    score = relevant_retrieved / len(evaluation.relevant_ids)
    return MetricResult(
        "context_recall",
        score,
        {"relevant": len(evaluation.relevant_ids), "relevant_retrieved": relevant_retrieved},
    )


def answer_faithfulness(evaluation: GenerationEvaluation) -> MetricResult:
    """
    Token-level faithfulness: fraction of answer tokens grounded in the retrieved context.
    Approximates whether the answer introduces information not present in context.
    """
    answer_tokens = set(evaluation.answer.lower().split())
    context_tokens = set(evaluation.context.lower().split())

    if not answer_tokens:
        return MetricResult("answer_faithfulness", 0.0, {"answer_tokens": 0, "grounded_tokens": 0})

    grounded = answer_tokens & context_tokens
    score = len(grounded) / len(answer_tokens)
    return MetricResult(
        "answer_faithfulness",
        score,
        {"answer_tokens": len(answer_tokens), "grounded_tokens": len(grounded)},
    )


def answer_relevance(evaluation: GenerationEvaluation) -> MetricResult:
    """
    Cosine similarity between query and answer using TF-IDF vectors.
    Measures whether the answer is topically aligned with the question.
    """
    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        tfidf = vectorizer.fit_transform([evaluation.query, evaluation.answer])
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        score = 0.0

    return MetricResult(
        "answer_relevance",
        score,
        {"query_preview": evaluation.query[:60], "answer_len": len(evaluation.answer)},
    )
