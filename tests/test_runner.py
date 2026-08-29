import pytest
from rag_benchmark import BenchmarkSample, RagBenchmarkRunner


def _sample(query="q", retrieved=None, relevant=None, answer="answer", context="context"):
    return BenchmarkSample(
        query=query,
        retrieved_ids=retrieved or ["d1"],
        relevant_ids=relevant or ["d1"],
        answer=answer,
        context=context,
    )


class TestRagBenchmarkRunner:

    def setup_method(self):
        self.runner = RagBenchmarkRunner()

    def test_single_perfect_sample_all_scores_one(self):
        sample = _sample(
            query="What is RAG?",
            retrieved=["doc1", "doc2"],
            relevant=["doc1", "doc2"],
            answer="RAG is retrieval augmented generation.",
            context="RAG stands for retrieval augmented generation combining retrieval with generation.",
        )
        report = self.runner.run([sample])

        assert report.num_samples == 1
        assert report.scores["context_precision"] == pytest.approx(1.0)
        assert report.scores["context_recall"] == pytest.approx(1.0)

    def test_multiple_samples_averages_precision(self):
        # Sample 1: precision = 1.0 (1/1 relevant)
        # Sample 2: precision = 0.5 (1/2 relevant)
        samples = [
            _sample(retrieved=["d1"], relevant=["d1"]),
            _sample(retrieved=["d1", "d2"], relevant=["d1"]),
        ]
        report = self.runner.run(samples)

        assert report.num_samples == 2
        assert report.scores["context_precision"] == pytest.approx(0.75)

    def test_report_contains_all_four_metrics(self):
        report = self.runner.run([_sample()])
        assert set(report.scores.keys()) == {
            "context_precision",
            "context_recall",
            "answer_faithfulness",
            "answer_relevance",
        }

    def test_per_sample_data_preserved(self):
        samples = [_sample(query="query-1"), _sample(query="query-2")]
        report = self.runner.run(samples)

        assert len(report.per_sample) == 2
        queries = [s["query"] for s in report.per_sample]
        assert "query-1" in queries
        assert "query-2" in queries

    def test_summary_contains_all_metric_names(self):
        report = self.runner.run([_sample()])
        summary = report.summary()

        assert "context_precision" in summary
        assert "context_recall" in summary
        assert "answer_faithfulness" in summary
        assert "answer_relevance" in summary

    def test_summary_contains_sample_count(self):
        samples = [_sample(), _sample(), _sample()]
        report = self.runner.run(samples)
        assert "3" in report.summary()

    def test_all_scores_bounded_between_zero_and_one(self):
        samples = [
            _sample(retrieved=["d1", "d2", "d3"], relevant=["d1"], answer="some answer text", context="other text"),
            _sample(retrieved=[], relevant=["d1"], answer="", context="ctx"),
        ]
        report = self.runner.run(samples)
        for metric, score in report.scores.items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of bounds"
