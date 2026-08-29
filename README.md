# RAG Benchmark

RAGAS-inspired evaluation harness for Retrieval-Augmented Generation systems — context precision, context recall, answer faithfulness, and answer relevance — with no LLM dependency.

## Metrics

| Metric | Formula | Measures |
|--------|---------|---------|
| **Context Precision** | `|relevant ∩ retrieved| / |retrieved|` | Signal-to-noise of retrieval |
| **Context Recall** | `|relevant ∩ retrieved| / |relevant|` | Coverage of relevant documents |
| **Answer Faithfulness** | Token overlap: answer ∩ context / answer | Hallucination rate |
| **Answer Relevance** | Cosine similarity (TF-IDF) of query and answer | Topical alignment |

All scores are bounded `[0, 1]` where `1.0` is perfect.

## Usage

```python
from rag_benchmark import BenchmarkSample, RagBenchmarkRunner

samples = [
    BenchmarkSample(
        query="What is retrieval-augmented generation?",
        retrieved_ids=["chunk_1", "chunk_2", "chunk_5"],
        relevant_ids=["chunk_1", "chunk_2"],
        answer="RAG combines document retrieval with language model generation.",
        context="RAG stands for retrieval-augmented generation, combining retrieval with generation.",
    ),
    BenchmarkSample(
        query="How does vector similarity search work?",
        retrieved_ids=["chunk_7"],
        relevant_ids=["chunk_7", "chunk_9"],
        answer="Vector similarity search computes dot products between embeddings.",
        context="Vector similarity is computed using dot products or cosine distance on dense embeddings.",
    ),
]

runner = RagBenchmarkRunner()
report = runner.run(samples)
print(report.summary())
```

Output:
```
RAG Benchmark Report — 2 sample(s)
--------------------------------------------------
  context_precision             : 0.8333
  context_recall                : 0.7500
  answer_faithfulness           : 0.8571
  answer_relevance              : 0.6124
```

## Individual Metrics

```python
from rag_benchmark import (
    RetrievalEvaluation, GenerationEvaluation,
    context_precision, context_recall,
    answer_faithfulness, answer_relevance,
)

# Retrieval metrics
ev = RetrievalEvaluation(
    query="What is RAG?",
    retrieved_ids=["doc1", "doc2", "doc3"],
    relevant_ids={"doc1", "doc2"},
)
print(context_precision(ev).score)  # 0.667
print(context_recall(ev).score)     # 1.0

# Generation metrics
gen = GenerationEvaluation(
    query="What is RAG?",
    answer="RAG combines retrieval and generation.",
    context="RAG stands for retrieval augmented generation combining retrieval with language generation.",
)
print(answer_faithfulness(gen).score)  # high — all answer tokens grounded in context
print(answer_relevance(gen).score)     # high — answer is topically aligned with query
```

## Running Tests

```bash
pip install -e . -r requirements.txt
pytest --tb=short -v
```

## Design Notes

**Faithfulness** uses token-overlap precision (answer tokens present in context) as a fast, deterministic proxy for LLM-graded faithfulness. In production, replace with an LLM judge call.

**Answer Relevance** uses TF-IDF cosine similarity as a zero-dependency proxy for embedding-based relevance. In production, replace with a fine-tuned embedding model.

## Tech Stack

- Python 3.11+
- NumPy 1.26 — vector aggregation
- scikit-learn 1.5 — TF-IDF vectorizer, cosine similarity
- pytest — test runner
