from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .metrics import (
    RetrievalEvaluation,
    GenerationEvaluation,
    answer_faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)


@dataclass
class BenchmarkSample:
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    answer: str
    context: str


@dataclass
class BenchmarkReport:
    num_samples: int
    scores: dict
    per_sample: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"RAG Benchmark Report — {self.num_samples} sample(s)"]
        lines.append("-" * 50)
        for name, score in self.scores.items():
            lines.append(f"  {name:30s}: {score:.4f}")
        return "\n".join(lines)


class RagBenchmarkRunner:

    def run(self, samples: List[BenchmarkSample]) -> BenchmarkReport:
        accumulator: dict[str, list[float]] = {
            "context_precision": [],
            "context_recall": [],
            "answer_faithfulness": [],
            "answer_relevance": [],
        }
        per_sample: List[dict] = []

        for sample in samples:
            retrieval_eval = RetrievalEvaluation(
                query=sample.query,
                retrieved_ids=sample.retrieved_ids,
                relevant_ids=set(sample.relevant_ids),
            )
            gen_eval = GenerationEvaluation(
                query=sample.query,
                answer=sample.answer,
                context=sample.context,
            )

            sample_scores = {
                "context_precision": context_precision(retrieval_eval).score,
                "context_recall": context_recall(retrieval_eval).score,
                "answer_faithfulness": answer_faithfulness(gen_eval).score,
                "answer_relevance": answer_relevance(gen_eval).score,
            }

            for metric, score in sample_scores.items():
                accumulator[metric].append(score)

            per_sample.append({"query": sample.query, **sample_scores})

        avg_scores = {
            k: float(np.mean(v)) for k, v in accumulator.items() if v
        }

        return BenchmarkReport(
            num_samples=len(samples),
            scores=avg_scores,
            per_sample=per_sample,
        )
