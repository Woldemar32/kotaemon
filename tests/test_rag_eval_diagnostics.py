import json
from types import SimpleNamespace

import pandas as pd

from kotaemon.indices.vectorindex import VectorRetrieval
from ktem.evaluation.ragas_eval import (
    _build_failure_report,
    _candidate_relevance,
    _retrieval_metric_row,
    load_eval_dataset,
)


class _CapacityLimitedVectorStore:
    def __init__(self, maximum_results):
        self.maximum_results = maximum_results
        self.requests = []

    def query(self, embedding, top_k, doc_ids=None, **kwargs):
        self.requests.append(top_k)
        if top_k > self.maximum_results:
            raise RuntimeError(
                "Cannot return the results in a contigious 2D array. "
                "Probably ef or M is too small"
            )
        return [], [0.9] * top_k, [f"doc-{index}" for index in range(top_k)]


def test_dataset_annotations_are_optional(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "plain",
                    "question": "Which degree?",
                    "ground_truth": "Bachelor of Science",
                    "source_file": "program.pdf",
                },
                {
                    "id": "annotated",
                    "question": "Which degree?",
                    "ground_truth": "Bachelor of Science",
                    "source_file": "program.pdf",
                    "expected_pages": 2,
                    "required_phrases": "Bachelor of Science",
                },
            ]
        ),
        encoding="utf-8",
    )

    samples = load_eval_dataset(dataset)

    assert samples[0]["expected_pages"] == []
    assert samples[0]["required_phrases"] == []
    assert samples[0]["has_manual_relevance_annotations"] is False
    assert samples[1]["expected_pages"] == ["2"]
    assert samples[1]["required_phrases"] == ["Bachelor of Science"]
    assert samples[1]["has_manual_relevance_annotations"] is True


def test_manual_relevance_accepts_expected_page_or_phrase():
    sample = {
        "source_file": "program.pdf",
        "reference": "Bachelor of Science",
        "expected_pages": ["2"],
        "required_phrases": ["Bachelor of Science"],
        "has_manual_relevance_annotations": True,
    }

    relevant, method, score, phrases = _candidate_relevance(
        sample, "program.pdf", 8, "The degree is Bachelor of Science."
    )

    assert relevant is True
    assert method == "manual"
    assert score == 1.0
    assert phrases == ["Bachelor of Science"]


def test_metrics_distinguish_retrieval_from_context_packing_failure():
    sample = {
        "id": "degree",
        "source_file": "program.pdf",
        "reference": "Bachelor of Science",
        "expected_pages": ["2"],
        "required_phrases": [],
        "has_manual_relevance_annotations": True,
    }
    irrelevant = SimpleNamespace(
        doc_id="wrong",
        text="General information",
        metadata={"file_name": "program.pdf", "page_label": 1},
    )
    relevant = SimpleNamespace(
        doc_id="right",
        text="Bachelor of Science",
        metadata={"file_name": "program.pdf", "page_label": 2},
    )
    context_by_id = {
        "wrong": {"included_chars": 20},
        "right": {"included_chars": 0, "exclusion_reason": "context_budget"},
    }

    metrics = _retrieval_metric_row(
        sample,
        [irrelevant, relevant],
        [
            {"stage": "initial", "rank": 1, "doc_id": "wrong", "relevant": False},
            {"stage": "initial", "rank": 2, "doc_id": "right", "relevant": True},
        ],
        context_by_id,
        {"max_context_tokens": 5000, "trimmed": True},
        "expected-source",
    )

    assert metrics["first_relevant_rank"] == 2
    assert metrics["answer_recall_at_3"] is True
    assert metrics["answer_chunk_candidate_found"] is True
    assert metrics["answer_chunk_retrieved"] is True
    assert metrics["answer_chunk_included"] is False
    assert metrics["answer_chunk_dropped"] is True


def test_failure_report_names_packing_failures():
    samples = pd.DataFrame(
        [{"id": "degree", "question": "Which degree?", "status": "ok"}]
    )
    retrieval = pd.DataFrame(
        [
            {
                "id": "degree",
                "source_file": "program.pdf",
                "relevance_method": "manual",
                "answer_chunk_candidate_found": True,
                "answer_chunk_retrieved": True,
                "answer_chunk_included": False,
                "answer_chunk_dropped": True,
                "first_relevant_rank": 5,
                "context_trimmed": True,
            }
        ]
    )

    report = _build_failure_report(samples, retrieval)

    assert "PACKING FAILURE" in report
    assert "First relevant rank: 5" in report


def test_vector_query_caps_scope_and_retries_hnsw_capacity_error():
    store = _CapacityLimitedVectorStore(maximum_results=3)
    retrieval = SimpleNamespace(
        vector_store=store,
        _last_retrieval_trace={},
    )

    _, scores, ids = VectorRetrieval._query_vectorstore_with_backoff(
        retrieval,
        embedding=[0.1, 0.2],
        requested_top_k=50,
        doc_ids=[f"scope-{index}" for index in range(12)],
    )

    assert store.requests == [12, 6, 3]
    assert len(scores) == 3
    assert len(ids) == 3
    assert retrieval._last_retrieval_trace["vector_query_attempts"] == [12, 6, 3]


def test_vector_query_does_not_hide_unrelated_runtime_errors():
    class BrokenVectorStore:
        def query(self, **kwargs):
            raise RuntimeError("embedding dimensions do not match")

    retrieval = SimpleNamespace(
        vector_store=BrokenVectorStore(),
        _last_retrieval_trace={},
    )

    try:
        VectorRetrieval._query_vectorstore_with_backoff(
            retrieval,
            embedding=[0.1, 0.2],
            requested_top_k=50,
            doc_ids=["scope-1"],
        )
    except RuntimeError as exc:
        assert "embedding dimensions" in str(exc)
    else:
        raise AssertionError("Unrelated vector-store error was unexpectedly hidden")


def test_vector_query_skips_an_empty_document_scope():
    store = _CapacityLimitedVectorStore(maximum_results=3)
    retrieval = SimpleNamespace(
        vector_store=store,
        _last_retrieval_trace={},
    )

    result = VectorRetrieval._query_vectorstore_with_backoff(
        retrieval,
        embedding=[0.1, 0.2],
        requested_top_k=50,
        doc_ids=[],
    )

    assert result == ([], [], [])
    assert store.requests == []
