import json
from types import SimpleNamespace

import pandas as pd

from kotaemon.indices.vectorindex import VectorRetrieval
from ktem.evaluation.ragas_eval import (
    PreparedEvalSample,
    _answer_output_limit,
    _build_failure_report,
    _candidate_relevance,
    _retrieval_metric_row,
    _record_answer_timing,
    _token_set,
    load_eval_dataset,
    run_evaluation,
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


def test_required_phrase_prevents_unrelated_same_page_false_positive():
    sample = {
        "source_file": "amendment.pdf",
        "reference": "Examiner decides",
        "expected_pages": ["3"],
        "required_phrases": ["examiner decides"],
        "has_manual_relevance_annotations": True,
    }

    relevant, method, score, phrases = _candidate_relevance(
        sample, "amendment.pdf", 3, "Unrelated withdrawal rules"
    )

    assert relevant is False
    assert method == "manual"
    assert score == 0.0
    assert phrases == []


def test_list_questions_receive_configurable_output_headroom(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_LIST_MAX_OUTPUT_TOKENS", "512")
    assert (
        _answer_output_limit(
            {
                "question": "Welche Module sind aufgeführt?",
                "reference": "A; B; C",
            },
            256,
        )
        == 512
    )
    assert (
        _answer_output_limit(
            {"question": "Wie viele ECTS?", "reference": "10 ECTS"}, 256
        )
        == 256
    )


def test_active_latency_excludes_phased_queue_wait():
    prepared = PreparedEvalSample(
        sample={"id": "q"},
        evidence="",
        evidence_mode=0,
        images=[],
        row={"retrieval_latency_sec": 2.5},
        candidates=[],
        retrieval_metrics={},
        started_at=10.0,
        retrieval_finished_at=12.5,
    )

    _record_answer_timing(prepared, answer_started=100.0, finished=103.0)

    assert prepared.row["retrieval_latency_sec"] == 2.5
    assert prepared.row["generation_latency_sec"] == 3.0
    assert prepared.row["answer_queue_wait_sec"] == 87.5
    assert prepared.row["latency_sec"] == 5.5


def test_lightweight_metric_tokens_normalize_equivalent_fact_forms():
    assert _token_set("höchstens 25 % Fehlzeit") & _token_set(
        "maximal 25 Prozent versäumen"
    ) == {"höchstens", "25", "percent"}
    assert "bachelor_science" in _token_set("Bachelor of Science (B.Sc.)")
    assert "6" in _token_set("im sechsten Semester")


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


def test_metrics_report_exact_phrase_coverage_not_only_expected_page():
    sample = {
        "id": "aids",
        "source_file": "amendment.pdf",
        "reference": "Examiner decides which aids are permitted",
        "expected_pages": ["3"],
        "required_phrases": ["examiner decides", "permitted aids"],
        "has_manual_relevance_annotations": True,
    }
    partial = SimpleNamespace(
        doc_id="partial",
        text="The examiner decides and informs students.",
        metadata={"file_name": "amendment.pdf", "page_label": 3},
    )
    unrelated = SimpleNamespace(
        doc_id="same-page",
        text="Rules for examination withdrawal.",
        metadata={"file_name": "amendment.pdf", "page_label": 3},
    )

    metrics = _retrieval_metric_row(
        sample,
        [partial, unrelated],
        [
            {
                "stage": "initial",
                "rank": 1,
                "doc_id": "partial",
                "relevant": True,
            },
            {
                "stage": "initial",
                "rank": 2,
                "doc_id": "same-page",
                "relevant": False,
            },
        ],
        {"partial": {"included_chars": 40}},
        {"max_context_tokens": 1000, "trimmed": False},
        "expected-source",
    )

    assert metrics["required_phrases_retrieved"] == 1
    assert metrics["required_phrases_included"] == 1
    assert metrics["required_phrase_recall_included"] == 0.5
    assert metrics["exact_evidence_included"] is False


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


def test_evaluation_runs_retrieval_phase_before_answer_phase(monkeypatch):
    from ktem.evaluation import ragas_eval

    samples = [
        {
            "id": f"q{index}",
            "question": f"Question {index}",
            "reference": f"Answer {index}",
            "source_file": "program.pdf",
        }
        for index in (1, 2)
    ]
    events = []
    llm = SimpleNamespace(model="answer-model", max_tokens=512)
    embedding = SimpleNamespace(model="embedding-model", ollama_keep_alive="5m")

    def retrieve(_app, _settings, _user_id, sample, retrieval_scope):
        events.append(f"retrieve:{sample['id']}")
        return PreparedEvalSample(
            sample=sample,
            evidence=f"Evidence {sample['id']}",
            evidence_mode=0,
            images=[],
            row={
                **sample,
                "indexed_source": "program.pdf",
                "answer": "",
                "contexts": [f"Evidence {sample['id']}"],
                "context_count": 1,
                "top_context_preview": "Evidence",
                "top_source": "program.pdf",
                "top_score": 0.9,
                "latency_sec": 0,
                "status": "retrieved",
                "error": "",
            },
            candidates=[],
            retrieval_metrics={
                "id": sample["id"],
                "source_file": sample["source_file"],
                "answer_chunk_included": True,
            },
            started_at=0,
        )

    def answer(_settings, prepared):
        events.append(f"answer:{prepared.sample['id']}")
        prepared.row["answer"] = prepared.sample["reference"]
        prepared.row["status"] = "ok"
        return prepared.row

    monkeypatch.setattr(ragas_eval, "load_eval_dataset", lambda _path: samples)
    monkeypatch.setattr(
        ragas_eval, "_resolve_evaluation_models", lambda _settings: (llm, embedding)
    )
    monkeypatch.setattr(ragas_eval, "_retrieve_with_pipeline", retrieve)
    monkeypatch.setattr(ragas_eval, "_answer_prepared_sample", answer)
    monkeypatch.setattr(ragas_eval, "_check_memory_guard", lambda _stage: None)
    monkeypatch.setattr(
        ragas_eval,
        "_unload_ollama_resource",
        lambda resource, _warnings, purpose: events.append(
            f"unload:{purpose}:{resource.model}"
        ),
    )
    monkeypatch.setattr(ragas_eval, "_effective_runtime_config", lambda *_args: {})
    monkeypatch.setenv("RAG_EVAL_PHASED_EXECUTION", "true")
    monkeypatch.setenv("RAG_EVAL_OLLAMA_RECYCLE_QUESTIONS", "2")
    monkeypatch.setenv("RAG_EVAL_OLLAMA_UNLOAD_AT_END", "true")
    monkeypatch.setenv("RAG_EVAL_MIN_AVAILABLE_GB", "0")

    result = run_evaluation(
        app=SimpleNamespace(),
        settings={"reasoning.max_context_length": 3000},
        user_id="default",
        dataset_path="unused.json",
        question_limit=2,
        run_ragas_metrics=False,
    )

    assert events[:2] == ["retrieve:q1", "retrieve:q2"]
    assert events.index("answer:q1") > events.index("retrieve:q2")
    assert list(result.samples["answer"]) == ["Answer 1", "Answer 2"]
    assert embedding.ollama_keep_alive == "5m"
    assert llm.max_tokens == 512
