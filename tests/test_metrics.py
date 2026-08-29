import pytest
from rag_benchmark import (
    RetrievalEvaluation,
    GenerationEvaluation,
    context_precision,
    context_recall,
    answer_faithfulness,
    answer_relevance,
)


class TestContextPrecision:

    def test_perfect_precision_all_retrieved_are_relevant(self):
        ev = RetrievalEvaluation("q", ["doc1", "doc2"], {"doc1", "doc2"})
        assert context_precision(ev).score == pytest.approx(1.0)

    def test_zero_precision_no_relevant_in_retrieved(self):
        ev = RetrievalEvaluation("q", ["doc3", "doc4"], {"doc1", "doc2"})
        assert context_precision(ev).score == pytest.approx(0.0)

    def test_partial_precision_one_of_three_relevant(self):
        ev = RetrievalEvaluation("q", ["doc1", "doc2", "doc3"], {"doc1"})
        assert context_precision(ev).score == pytest.approx(1 / 3)

    def test_empty_retrieved_returns_zero(self):
        ev = RetrievalEvaluation("q", [], {"doc1"})
        assert context_precision(ev).score == pytest.approx(0.0)

    def test_result_carries_metric_name(self):
        ev = RetrievalEvaluation("q", ["doc1"], {"doc1"})
        assert context_precision(ev).name == "context_precision"

    def test_details_expose_counts(self):
        ev = RetrievalEvaluation("q", ["d1", "d2", "d3"], {"d1"})
        result = context_precision(ev)
        assert result.details["retrieved"] == 3
        assert result.details["relevant_retrieved"] == 1


class TestContextRecall:

    def test_perfect_recall_all_relevant_retrieved(self):
        ev = RetrievalEvaluation("q", ["doc1", "doc2"], {"doc1", "doc2"})
        assert context_recall(ev).score == pytest.approx(1.0)

    def test_zero_recall_no_relevant_retrieved(self):
        ev = RetrievalEvaluation("q", ["doc3"], {"doc1", "doc2"})
        assert context_recall(ev).score == pytest.approx(0.0)

    def test_partial_recall_half_of_relevant_retrieved(self):
        ev = RetrievalEvaluation("q", ["doc1"], {"doc1", "doc2"})
        assert context_recall(ev).score == pytest.approx(0.5)

    def test_no_relevant_documents_returns_one(self):
        # When there are no relevant docs, recall is vacuously 1.0
        ev = RetrievalEvaluation("q", ["doc1"], set())
        assert context_recall(ev).score == pytest.approx(1.0)

    def test_result_carries_metric_name(self):
        ev = RetrievalEvaluation("q", ["doc1"], {"doc1"})
        assert context_recall(ev).name == "context_recall"


class TestAnswerFaithfulness:

    def test_fully_grounded_answer(self):
        context = "The capital of France is Paris and it has the Eiffel Tower."
        answer = "Paris is the capital of France."
        ev = GenerationEvaluation("q", answer, context)
        assert answer_faithfulness(ev).score == pytest.approx(1.0)

    def test_hallucinated_tokens_lower_score(self):
        context = "France is in Europe."
        answer = "Python is a programming language invented by Guido van Rossum."
        ev = GenerationEvaluation("q", answer, context)
        assert answer_faithfulness(ev).score < 0.2

    def test_partial_faithfulness(self):
        context = "The sky is blue and the sun is yellow."
        answer = "The sky is blue but the stars are invisible in daylight."
        ev = GenerationEvaluation("q", answer, context)
        result = answer_faithfulness(ev)
        assert 0.0 < result.score < 1.0

    def test_empty_answer_returns_zero(self):
        ev = GenerationEvaluation("q", "", "some context here")
        assert answer_faithfulness(ev).score == pytest.approx(0.0)

    def test_result_carries_metric_name(self):
        ev = GenerationEvaluation("q", "answer text", "context text")
        assert answer_faithfulness(ev).name == "answer_faithfulness"


class TestAnswerRelevance:

    def test_on_topic_answer_scores_higher_than_off_topic(self):
        query = "What is machine learning?"
        on_topic = GenerationEvaluation(
            query,
            "Machine learning is a subset of artificial intelligence that learns from data.",
            "ctx",
        )
        off_topic = GenerationEvaluation(
            query,
            "The Roman Empire fell in 476 AD due to economic and military pressures.",
            "ctx",
        )
        assert answer_relevance(on_topic).score > answer_relevance(off_topic).score

    def test_identical_query_and_answer_returns_one(self):
        text = "What is retrieval augmented generation?"
        ev = GenerationEvaluation(text, text, "ctx")
        assert answer_relevance(ev).score == pytest.approx(1.0)

    def test_score_bounded_between_zero_and_one(self):
        ev = GenerationEvaluation("query about dogs", "cats are mammals", "ctx")
        result = answer_relevance(ev)
        assert 0.0 <= result.score <= 1.0

    def test_result_carries_metric_name(self):
        ev = GenerationEvaluation("q", "a", "c")
        assert answer_relevance(ev).name == "answer_relevance"
