from __future__ import annotations

import gc
import json
import math
import os
import re
import time
import traceback
import unicodedata
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from decouple import config as env_config
from ktem.components import reasonings
from ktem.db.engine import engine
from kotaemon.base import Document
from kotaemon.indices.qa.utils import strip_think_tag
from sqlalchemy import select
from sqlalchemy.orm import Session

ProgressFn = Callable[[int, int, str], None]


@dataclass
class EvalRunResult:
    """Result bundle returned by the local Kotaemon + RAGAS evaluation run."""

    samples: pd.DataFrame
    ragas_scores: pd.DataFrame
    retrieval_metrics: pd.DataFrame
    retrieval_candidates: pd.DataFrame
    summary: dict[str, float | int | str]
    runtime_config: dict[str, Any]
    failure_report: str
    warnings: list[str]


@dataclass
class RagasEvaluatorModels:
    """Local evaluator models passed into RAGAS to avoid OpenAI defaults."""

    llm: Any
    embeddings: Any
    raw_embeddings: Any
    llm_name: str
    embeddings_name: str
    run_config: Any | None
    notes: list[str]


@dataclass
class PreparedEvalSample:
    """Retrieved and packed evidence waiting for answer generation."""

    sample: dict[str, Any]
    evidence: str
    evidence_mode: int
    images: list[str]
    row: dict[str, Any]
    candidates: list[dict[str, Any]]
    retrieval_metrics: dict[str, Any]
    started_at: float
    retrieval_finished_at: float | None = None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(env_config(name, default=default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return bool(env_config(name, default=default, cast=bool))


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(env_config(name, default=default)))
    except (TypeError, ValueError):
        return default


def _env_nonnegative_int(name: str, default: int) -> int:
    return _env_int(name, default, minimum=0)


def _env_optional_int(name: str, minimum: int = 1) -> int | None:
    """Return an integer override only when the deployment configured one."""

    raw = env_config(name, default=None)
    if raw in (None, ""):
        return None
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return None


def _answer_output_limit(sample: dict[str, Any], default: int) -> int:
    """Give enumeration questions headroom without inflating every KV cache."""

    question = str(sample.get("question") or "").lower()
    reference = str(sample.get("reference") or "")
    enumeration = reference.count(";") >= 2 or bool(
        re.search(
            r"\b(?:welche|nenne|liste|what|which)\b.*\b"
            r"(?:module|modules|bereiche|areas|profile|profiles|fächer|courses)\b",
            question,
        )
    )
    if not enumeration:
        return default
    return max(
        default,
        _env_nonnegative_int("RAG_EVAL_LIST_MAX_OUTPUT_TOKENS", 0),
    )


def _record_answer_timing(
    prepared: PreparedEvalSample, answer_started: float, finished: float
) -> None:
    retrieval_latency = float(prepared.row.get("retrieval_latency_sec") or 0.0)
    prepared.row["generation_latency_sec"] = round(finished - answer_started, 2)
    prepared.row["answer_queue_wait_sec"] = round(
        max(0.0, answer_started - (prepared.retrieval_finished_at or answer_started)),
        2,
    )
    # Active latency deliberately excludes phased queue waiting.
    prepared.row["latency_sec"] = round(
        retrieval_latency + prepared.row["generation_latency_sec"], 2
    )


@contextmanager
def _temporary_model_attribute(model: Any, name: str, value: Any):
    """Temporarily tune a shared model object for the serialized evaluation job."""

    if model is None or not hasattr(model, name):
        yield
        return
    original = getattr(model, name)
    setattr(model, name, value)
    try:
        yield
    finally:
        setattr(model, name, original)


def _check_memory_guard(stage: str) -> None:
    """Stop before swap pressure becomes dangerous when a guard is configured."""

    minimum_gb = _env_float("RAG_EVAL_MIN_AVAILABLE_GB", 0.0)
    if minimum_gb <= 0:
        return
    try:
        import psutil
    except ImportError:
        return
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < minimum_gb:
        raise RuntimeError(
            f"Evaluation stopped before {stage}: only {available_gb:.2f} GiB "
            f"memory is available (configured minimum: {minimum_gb:.2f} GiB)."
        )


def _ragas_run_config() -> Any | None:
    """Runtime settings for local RAGAS scoring.

    Local LLM endpoints are usually slower than hosted evaluators, but a strong
    single-GPU server can still handle bounded parallel judge calls. RAGAS defaults
    to 16 concurrent jobs with a 180s per-job timeout, which can overload local
    endpoints and produce `Exception raised in Job[...]` timeout logs. Keep the
    metric set unchanged, but use moderate parallelism plus a longer per-operation
    timeout. Environment variables allow users to tune this without code changes.
    """

    configured = {
        "timeout": _env_optional_int("RAGAS_EVAL_TIMEOUT_SEC"),
        "max_workers": _env_optional_int("RAGAS_EVAL_MAX_WORKERS"),
        "max_retries": _env_optional_int("RAGAS_EVAL_MAX_RETRIES", minimum=0),
        "max_wait": _env_optional_int("RAGAS_EVAL_MAX_WAIT_SEC", minimum=0),
    }
    if not any(value is not None for value in configured.values()):
        return None

    try:
        from ragas.run_config import RunConfig  # type: ignore
    except Exception:
        return None

    return RunConfig(
        **{key: value for key, value in configured.items() if value is not None}
    )


def _run_config_note(run_config: Any | None) -> str:
    if run_config is None:
        return "RAGAS runtime uses installed-version defaults."
    return (
        "RAGAS runtime: "
        f"timeout={getattr(run_config, 'timeout', 'default')}s, "
        f"max_workers={getattr(run_config, 'max_workers', 'default')}, "
        f"batch_size={_env_optional_int('RAGAS_EVAL_BATCH_SIZE') or 'default'}, "
        f"max_retries={getattr(run_config, 'max_retries', 'default')}."
    )


def _apply_model_timeout(model: Any, run_config: Any | None) -> None:
    """Best-effort propagation of RAGAS timeout into LangChain clients."""

    timeout = getattr(run_config, "timeout", None)
    if timeout is None:
        return

    for attr in ("request_timeout", "timeout"):
        try:
            setattr(model, attr, timeout)
        except Exception:
            pass

    # Some LangChain wrappers keep the actual model one level down.
    for nested_attr in ("langchain_llm", "bound", "model"):
        nested = getattr(model, nested_attr, None)
        if nested is not None and nested is not model:
            for attr in ("request_timeout", "timeout"):
                try:
                    setattr(nested, attr, timeout)
                except Exception:
                    pass


def find_default_dataset_path(start: Path | None = None) -> Path | None:
    """Find the user's rag_eval_dataset file near the app/repository root."""

    candidates: list[Path] = []
    env_path = os.environ.get("RAG_EVAL_DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = (start or Path.cwd()).resolve()
    names = ["rag_eval_dataset", "rag_eval_dataset.json", "rag_eval_dataset.jsonl"]
    for parent in [root, *root.parents]:
        for name in names:
            candidates.append(parent / name)

    for path in candidates:
        if path.is_file():
            return path
    return None


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON/JSONL dataset and normalize fields used by RAGAS."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        raw = [
            json.loads(line) for line in dataset_path.read_text().splitlines() if line
        ]
    else:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("samples") or raw.get("questions") or []
    if not isinstance(raw, list):
        raise ValueError(
            "Dataset must be a list or a dict with data/samples/questions."
        )

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{idx} is not an object.")

        question = item.get("question") or item.get("user_input") or item.get("query")
        reference = (
            item.get("ground_truth")
            or item.get("reference")
            or item.get("answer")
            or item.get("expected_answer")
        )
        source_file = (
            item.get("source_file") or item.get("file") or item.get("document")
        )
        expected_pages = item.get("expected_pages") or []
        required_phrases = item.get("required_phrases") or []

        if not question or not reference:
            raise ValueError(
                f"Dataset item #{idx} must contain question and ground_truth/reference."
            )

        if not isinstance(expected_pages, list):
            expected_pages = [expected_pages]
        if not isinstance(required_phrases, list):
            required_phrases = [required_phrases]

        normalized.append(
            {
                "id": str(item.get("id") or idx),
                "question": str(question),
                "reference": str(reference),
                "source_file": str(source_file or ""),
                "expected_pages": [
                    str(page).strip() for page in expected_pages if str(page).strip()
                ],
                "required_phrases": [
                    str(phrase).strip()
                    for phrase in required_phrases
                    if str(phrase).strip()
                ],
                "has_manual_relevance_annotations": bool(
                    expected_pages or required_phrases
                ),
            }
        )

    return normalized


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        score = float(value)
        if not math.isfinite(score):
            return None
        return score
    except (TypeError, ValueError):
        return None


def _doc_source_name(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    for key in ("file_name", "filename", "source", "Source", "document_name"):
        if metadata.get(key):
            return str(metadata[key])
    return ""


def _doc_score(doc: Any) -> float | None:
    metadata = getattr(doc, "metadata", {}) or {}
    retrieval_metadata = getattr(doc, "retrieval_metadata", {}) or {}
    score_keys = (
        "llm_trulens_score",
        "llm_reranking_score",
        "reranking_score",
        "retrieval_score",
        "vector_score",
        "vectorstore_score",
        "similarity_score",
        "similarity",
        "_score",
        "score",
    )
    for key in score_keys:
        score = _safe_float(metadata.get(key))
        if score is not None:
            return round(score, 4)
    for key in score_keys:
        score = _safe_float(retrieval_metadata.get(key))
        if score is not None:
            return round(score, 4)
    score = _safe_float(getattr(doc, "score", None))
    if score == -1.0:
        return None
    return round(score, 4) if score is not None else None


def _normalize_match_text(value: Any) -> str:
    return " ".join(re.findall(r"[\wÄÖÜäöüß]+", str(value).lower()))


def _source_matches(actual: str, expected: str) -> bool:
    if not expected:
        return True
    return Path(actual).name.lower() == Path(expected).name.lower()


def _candidate_relevance(
    sample: dict[str, Any], source: str, page: Any, text: str
) -> tuple[bool, str, float, list[str]]:
    """Classify a candidate using optional labels or a conservative inference."""

    expected_source = sample.get("source_file", "")
    source_match = _source_matches(source, expected_source)
    normalized_text = _normalize_match_text(text)
    expected_pages = {str(value).strip() for value in sample.get("expected_pages", [])}
    required_phrases = sample.get("required_phrases", [])
    matched_phrases = [
        phrase
        for phrase in required_phrases
        if _normalize_match_text(phrase) in normalized_text
    ]

    if sample.get("has_manual_relevance_annotations"):
        page_match = bool(expected_pages) and str(page).strip() in expected_pages
        phrase_match = bool(matched_phrases)
        # Required phrases are stronger annotations than a page number. A page
        # can contain several unrelated chunks, especially in amendment statutes.
        relevant = source_match and (
            phrase_match if required_phrases else page_match
        )
        phrase_recall = (
            len(matched_phrases) / len(required_phrases)
            if required_phrases
            else float(page_match)
        )
        return (
            relevant,
            "manual",
            round(phrase_recall, 4) if source_match else 0.0,
            matched_phrases,
        )

    reference_tokens = _token_set(sample.get("reference", ""))
    candidate_tokens = _token_set(text)
    keyword_recall = (
        len(reference_tokens & candidate_tokens) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    # This is intentionally conservative and explicitly labelled as inferred.
    relevant = source_match and keyword_recall >= 0.2
    return relevant, "inferred_keyword", round(keyword_recall, 4), []


def _context_diagnostics_by_id(pipeline: Any) -> tuple[dict[str, dict], dict]:
    diagnostics = getattr(pipeline.evidence_pipeline, "_last_run_diagnostics", {}) or {}
    by_id: dict[str, dict] = {}
    for item in diagnostics.get("contributions", []):
        if not item.get("doc_id"):
            continue
        doc_id = str(item["doc_id"])
        existing = by_id.get(doc_id)
        if existing is None or item.get("included_chars", 0) > existing.get(
            "included_chars", 0
        ):
            by_id[doc_id] = item
    return by_id, diagnostics


def _retrieval_traces(retrievers: list[Any]) -> list[dict[str, Any]]:
    traces = []
    for index, retriever in enumerate(retrievers):
        trace = getattr(retriever, "_last_retrieval_trace", {}) or {}
        traces.append({"retriever_index": index, **trace})
    return traces


def _candidate_rows(
    sample: dict[str, Any],
    traces: list[dict[str, Any]],
    final_docs: list[Any],
    context_by_id: dict[str, dict],
) -> list[dict[str, Any]]:
    final_rank = {str(doc.doc_id): rank for rank, doc in enumerate(final_docs, start=1)}
    rows: list[dict[str, Any]] = []

    for trace in traces:
        retriever_index = trace.get("retriever_index", 0)
        stages = trace.get("stages") or []
        if not stages:
            fallback_documents = [
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "source": _doc_source_name(doc),
                    "page": (doc.metadata or {}).get("page_label"),
                    "document_type": (doc.metadata or {}).get("type", "text"),
                    "score": _doc_score(doc),
                    "reranking_score": (doc.metadata or {}).get("reranking_score"),
                    "text": doc.text or "",
                }
                for rank, doc in enumerate(final_docs, start=1)
            ]
            stages = [
                {"name": "initial", "documents": fallback_documents},
                {"name": "final", "documents": fallback_documents},
            ]

        for stage in stages:
            stage_name = stage.get("name", "unknown")
            for candidate in stage.get("documents", []):
                doc_id = str(candidate.get("doc_id", ""))
                source = str(candidate.get("source", ""))
                text = str(candidate.get("text", ""))
                relevant, method, relevance_score, matched_phrases = (
                    _candidate_relevance(sample, source, candidate.get("page"), text)
                )
                context_item = context_by_id.get(doc_id, {})
                candidate_row = {
                    "id": sample["id"],
                    "question": sample["question"],
                    "retriever_index": retriever_index,
                    "retrieval_mode": trace.get("mode", ""),
                    "stage": stage_name,
                    "rank": candidate.get("rank"),
                    "final_rank": final_rank.get(doc_id),
                    "doc_id": doc_id,
                    "source": source,
                    "page": candidate.get("page"),
                    "document_type": candidate.get("document_type", "text"),
                    "vector_score": candidate.get("score"),
                    "reranker_score": candidate.get("reranking_score"),
                    "llm_relevance_score": candidate.get("llm_relevance_score"),
                    "relevant": relevant,
                    "relevance_method": method,
                    "relevance_score": relevance_score,
                    "matched_required_phrases": matched_phrases,
                    "included_in_context": bool(context_item.get("included_chars", 0)),
                    "fully_included_in_context": bool(
                        context_item.get("fully_included", False)
                    ),
                    "context_exclusion_reason": context_item.get(
                        "exclusion_reason", ""
                    ),
                    "text_chars": len(text),
                    "preview": text[:500],
                }
                if _env_bool("RAG_EVAL_RETAIN_CANDIDATE_TEXT", False):
                    candidate_row["text"] = text
                rows.append(candidate_row)
    return rows


def _retrieval_metric_row(
    sample: dict[str, Any],
    final_docs: list[Any],
    candidate_rows: list[dict[str, Any]],
    context_by_id: dict[str, dict],
    context_diagnostics: dict[str, Any],
    retrieval_scope: str,
) -> dict[str, Any]:
    candidates = []
    for rank, doc in enumerate(final_docs, start=1):
        relevant, method, score, _ = _candidate_relevance(
            sample,
            _doc_source_name(doc),
            (doc.metadata or {}).get("page_label"),
            doc.text or "",
        )
        candidates.append((rank, doc, relevant, method, score))

    relevant_ranks = [rank for rank, _, relevant, _, _ in candidates if relevant]
    final_first_relevant_rank = min(relevant_ranks) if relevant_ranks else None
    relevant_doc_ids = {
        str(doc.doc_id) for _, doc, relevant, _, _ in candidates if relevant
    }
    included_relevant = [
        doc_id
        for doc_id in relevant_doc_ids
        if context_by_id.get(doc_id, {}).get("included_chars", 0)
    ]
    required_phrases = sample.get("required_phrases", [])
    retrieved_phrase_matches = {
        phrase
        for phrase in required_phrases
        if any(
            _normalize_match_text(phrase)
            in _normalize_match_text(doc.text or "")
            for _, doc, _, _, _ in candidates
        )
    }
    included_documents = [
        doc
        for _, doc, _, _, _ in candidates
        if context_by_id.get(str(doc.doc_id), {}).get("included_chars", 0)
    ]
    included_phrase_matches = {
        phrase
        for phrase in required_phrases
        if any(
            _normalize_match_text(phrase)
            in _normalize_match_text(doc.text or "")
            for doc in included_documents
        )
    }
    phrase_count = len(required_phrases)
    source_ranks = [
        rank
        for rank, doc, _, _, _ in candidates
        if _source_matches(_doc_source_name(doc), sample.get("source_file", ""))
    ]
    initial_rows = [row for row in candidate_rows if row["stage"] == "initial"]
    initial_doc_ids = [row["doc_id"] for row in initial_rows]
    initial_relevant_ranks = [
        int(row["rank"])
        for row in initial_rows
        if row.get("relevant") and row.get("rank") is not None
    ]
    first_relevant_rank = (
        min(initial_relevant_ranks) if initial_relevant_ranks else None
    )
    duplicate_ratio = (
        1 - len(set(initial_doc_ids)) / len(initial_doc_ids) if initial_doc_ids else 0.0
    )

    return {
        "id": sample["id"],
        "source_file": sample.get("source_file", ""),
        "retrieval_scope": retrieval_scope,
        "relevance_method": (
            "manual"
            if sample.get("has_manual_relevance_annotations")
            else "inferred_keyword"
        ),
        "retrieved_count": len(final_docs),
        "source_hit_at_1": bool(source_ranks and min(source_ranks) <= 1),
        "source_hit_at_3": bool(source_ranks and min(source_ranks) <= 3),
        "source_hit_at_5": bool(source_ranks and min(source_ranks) <= 5),
        "answer_recall_at_1": bool(first_relevant_rank and first_relevant_rank <= 1),
        "answer_recall_at_3": bool(first_relevant_rank and first_relevant_rank <= 3),
        "answer_recall_at_5": bool(first_relevant_rank and first_relevant_rank <= 5),
        "answer_recall_at_10": bool(first_relevant_rank and first_relevant_rank <= 10),
        "first_relevant_rank": first_relevant_rank,
        "final_first_relevant_rank": final_first_relevant_rank,
        "reciprocal_rank": (
            round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0
        ),
        "answer_chunk_candidate_found": bool(initial_relevant_ranks),
        "answer_chunk_retrieved": bool(relevant_doc_ids),
        "answer_chunk_included": bool(included_relevant),
        "answer_chunk_dropped": bool(relevant_doc_ids and not included_relevant),
        "required_phrase_count": phrase_count,
        "required_phrases_retrieved": len(retrieved_phrase_matches),
        "required_phrases_included": len(included_phrase_matches),
        "required_phrase_recall_retrieved": round(
            len(retrieved_phrase_matches) / phrase_count, 4
        )
        if phrase_count
        else None,
        "required_phrase_recall_included": round(
            len(included_phrase_matches) / phrase_count, 4
        )
        if phrase_count
        else None,
        "exact_evidence_retrieved": bool(
            phrase_count and len(retrieved_phrase_matches) == phrase_count
        ),
        "exact_evidence_included": bool(
            phrase_count and len(included_phrase_matches) == phrase_count
        ),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "context_limit_tokens": context_diagnostics.get("max_context_tokens"),
        "context_original_chars": context_diagnostics.get("original_evidence_chars", 0),
        "context_final_chars": context_diagnostics.get("final_evidence_chars", 0),
        "context_trimmed": bool(context_diagnostics.get("trimmed", False)),
        "chunks_included": sum(
            bool(item.get("included_chars", 0)) for item in context_by_id.values()
        ),
        "chunks_dropped": sum(
            not bool(item.get("included_chars", 0)) for item in context_by_id.values()
        ),
    }


def _ensure_simple_reasoning_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Use Simple QA for deterministic single-turn RAG evaluation."""

    eval_settings = deepcopy(settings)
    if "simple" in reasonings:
        eval_settings["reasoning.use"] = "simple"

    configured_context = int(eval_settings.get("reasoning.max_context_length", 5000))
    context_cap = _env_optional_int("RAG_EVAL_MAX_CONTEXT_TOKENS")
    eval_settings["reasoning.max_context_length"] = (
        min(configured_context, context_cap) if context_cap else configured_context
    )

    # Disable expensive UI-only artifacts; RAGAS evaluates answer + contexts instead.
    for key, value in {
        "reasoning.options.simple.highlight_citation": "off",
        "reasoning.options.simple.create_mindmap": False,
        "reasoning.options.simple.create_citation_viz": False,
    }.items():
        if key in eval_settings:
            eval_settings[key] = value
    return eval_settings


def _resolve_evaluation_models(settings: dict[str, Any]) -> tuple[Any, Any]:
    """Return the raw answer and retrieval models used by this evaluation."""

    from ktem.embeddings.manager import embedding_models_manager
    from ktem.llms.manager import llms

    reasoning_id = settings.get("reasoning.use") or "simple"
    llm_name = (
        settings.get(f"reasoning.options.{reasoning_id}.llm")
        or settings.get("reasoning.options.simple.llm")
        or llms.get_default_name()
    )
    llm = llms.get(str(llm_name), None) or llms.get_default()

    embedding_name = ""
    for key, value in settings.items():
        if key.startswith("index.options.") and key.endswith(".embedding") and value:
            embedding_name = str(value)
            break
    if not embedding_name or embedding_name == "default":
        embedding_name = embedding_models_manager.get_default_name()
    embedding = (
        embedding_models_manager.get(embedding_name, None)
        or embedding_models_manager.get("default", None)
        or embedding_models_manager.get_default()
    )
    return llm, embedding


def _unload_ollama_resource(
    resource: Any, warnings: list[str], purpose: str
) -> bool:
    """Best-effort unload that is a no-op for non-Ollama providers."""

    if resource is None:
        return False
    model_name = str(getattr(resource, "model", "") or "")
    base_url = str(getattr(resource, "base_url", "") or "")
    resource_type = type(resource).__name__.lower()
    if not model_name or (
        "ollama" not in resource_type and "11434" not in base_url
    ):
        return False
    try:
        endpoint = base_url.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        if not endpoint:
            endpoint = "http://localhost:11434"
        payload = json.dumps(
            {"model": model_name, "prompt": "", "keep_alive": 0}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{endpoint}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15):
            return True
    except Exception as exc:
        warnings.append(f"Could not release Ollama {purpose} model: {exc}")
        return False


def _find_source_ids(
    app: Any, source_file: str, user_id: Any
) -> list[tuple[Any, str, str]]:
    """Return (index, source_id, source_name) tuples matching the dataset file name."""

    if not source_file:
        return []

    requested = Path(source_file).name.lower()
    matches: list[tuple[Any, str, str]] = []

    with Session(engine) as session:
        for index in app.index_manager.indices:
            Source = getattr(index, "_resources", {}).get("Source")
            if Source is None:
                continue

            statement = select(Source)
            if index.config.get("private", False):
                statement = statement.where(Source.user == user_id)

            # Prefer exact normalized basename match; fallback to contains match for
            # uploads where Kotaemon preserved a prefix/suffix around the PDF name.
            rows = session.execute(statement).scalars().all()
            exact: list[Any] = []
            fuzzy: list[Any] = []
            for row in rows:
                row_name = str(getattr(row, "name", ""))
                row_base = Path(row_name).name.lower()
                if row_base == requested:
                    exact.append(row)
                elif requested in row_base or row_base in requested:
                    fuzzy.append(row)

            for row in exact or fuzzy:
                matches.append((index, str(row.id), str(row.name)))

    return matches


def _build_retrievers(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    source_file: str,
    retrieval_scope: str = "expected-source",
):
    if retrieval_scope == "all":
        retrievers = []
        used_sources: list[str] = []
        for index in app.index_manager.indices:
            if getattr(index, "_selector_ui", None) is None:
                index.get_selector_component_ui()
            retrievers.extend(
                index.get_retriever_pipelines(settings, user_id, ["all", [], user_id])
            )
            used_sources.append(f"{index.name}: all visible documents")
        if not retrievers:
            raise ValueError("No retriever pipelines available for all-document scope")
        return retrievers, used_sources

    source_matches = _find_source_ids(app, source_file, user_id)
    if not source_matches:
        raise ValueError(f"Source file is not indexed or not visible: {source_file}")

    retrievers = []
    used_sources: list[str] = []
    for index, source_id, source_name in source_matches:
        if getattr(index, "_selector_ui", None) is None:
            index.get_selector_component_ui()
        retrievers.extend(
            index.get_retriever_pipelines(
                settings, user_id, ["select", [source_id], user_id]
            )
        )
        used_sources.append(f"{index.name}: {source_name}")

    if not retrievers:
        raise ValueError(f"No retriever pipelines available for {source_file}")

    return retrievers, used_sources


def _retrieve_with_pipeline(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    sample: dict[str, Any],
    retrieval_scope: str = "expected-source",
) -> PreparedEvalSample:
    question = sample["question"]
    source_file = sample.get("source_file", "")
    started = time.time()

    retrievers, used_sources = _build_retrievers(
        app, settings, user_id, source_file, retrieval_scope
    )
    reasoning_id = settings.get("reasoning.use")
    if reasoning_id not in reasonings:
        reasoning_id = "simple" if "simple" in reasonings else next(iter(reasonings))
    pipeline = reasonings[reasoning_id].get_pipeline(
        settings,
        {"app": {}, "pipeline": {}},
        retrievers,
    )

    docs, _ = pipeline.retrieve(question, [])
    evidence_mode, evidence, images = pipeline.evidence_pipeline.run(docs).content
    context_by_id, context_diagnostics = _context_diagnostics_by_id(pipeline)
    traces = _retrieval_traces(retrievers)
    candidates = _candidate_rows(sample, traces, docs, context_by_id)
    retrieval_metrics = _retrieval_metric_row(
        sample,
        docs,
        candidates,
        context_by_id,
        context_diagnostics,
        retrieval_scope,
    )

    contexts = [
        getattr(doc, "text", "") or getattr(doc, "content", "") or "" for doc in docs
    ]
    top_doc = docs[0] if docs else None
    top_score = None
    for doc in docs:
        top_score = _doc_score(doc)
        if top_score is not None:
            break
    if top_score is None and docs:
        top_score = 0.0

    row = {
        "id": sample["id"],
        "question": question,
        "reference": sample["reference"],
        "source_file": source_file,
        "indexed_source": "; ".join(used_sources),
        "answer": "",
        "contexts": contexts,
        "context_count": len(contexts),
        "top_context_preview": (contexts[0][:500] if contexts else ""),
        "top_source": _doc_source_name(top_doc) if top_doc is not None else "",
        "top_score": top_score,
        "latency_sec": 0,
        "retrieval_latency_sec": round(time.time() - started, 2),
        "answer_queue_wait_sec": 0,
        "generation_latency_sec": 0,
        "answer_output_token_limit": 0,
        "status": "retrieved",
        "error": "",
    }
    finished = time.time()
    row["retrieval_latency_sec"] = round(finished - started, 2)
    return PreparedEvalSample(
        sample=sample,
        evidence=evidence,
        evidence_mode=evidence_mode,
        images=images,
        row=row,
        candidates=candidates,
        retrieval_metrics=retrieval_metrics,
        started_at=started,
        retrieval_finished_at=finished,
    )


def _answer_prepared_sample(
    settings: dict[str, Any], prepared: PreparedEvalSample
) -> dict[str, Any]:
    """Generate an answer without invoking the embedding/retrieval model again."""

    answer_started = time.time()
    reasoning_id = settings.get("reasoning.use")
    if reasoning_id not in reasonings:
        reasoning_id = "simple" if "simple" in reasonings else next(iter(reasonings))
    pipeline = reasonings[reasoning_id].get_pipeline(
        settings,
        {"app": {}, "pipeline": {}},
        [],
    )

    def collect_answer() -> str:
        answer_chunks: list[str] = []
        for response in pipeline.answering_pipeline.stream(
            question=prepared.sample["question"],
            history=[],
            evidence=prepared.evidence,
            evidence_mode=prepared.evidence_mode,
            images=prepared.images,
            conv_id=f"rag-eval-{prepared.sample['id']}",
        ):
            if isinstance(response, Document) and response.channel == "chat":
                if response.content is None:
                    answer_chunks = []
                else:
                    answer_chunks.append(str(response.content))
        return strip_think_tag("".join(answer_chunks)).strip()

    answer = collect_answer()
    prepared.row["answer_retry_count"] = 0
    if not answer:
        # Reasoning models can consume a small completion allowance entirely with
        # thinking tokens. Keep thinking enabled and retry once with bounded
        # headroom; this is evaluation-only and does not alter the user's model.
        answer_llm = getattr(pipeline.answering_pipeline, "llm", None)
        retry_tokens = _env_nonnegative_int(
            "RAG_EVAL_EMPTY_ANSWER_RETRY_TOKENS", 2048
        )
        current_tokens = int(getattr(answer_llm, "max_tokens", 0) or 0)
        if answer_llm is not None and retry_tokens > current_tokens:
            prepared.row["answer_retry_count"] = 1
            with _temporary_model_attribute(answer_llm, "max_tokens", retry_tokens):
                answer = collect_answer()

    if not answer:
        raise RuntimeError(
            "The model completed without final answer content. It may have used "
            "the output allowance for reasoning tokens."
        )

    finished = time.time()
    prepared.row["answer"] = answer
    _record_answer_timing(prepared, answer_started, finished)
    prepared.row["status"] = "ok"
    return prepared.row


def _answer_with_pipeline(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    sample: dict[str, Any],
    retrieval_scope: str = "expected-source",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible one-sample path used when phased execution is disabled."""

    prepared = _retrieve_with_pipeline(
        app, settings, user_id, sample, retrieval_scope=retrieval_scope
    )
    row = _answer_prepared_sample(settings, prepared)
    return row, prepared.candidates, prepared.retrieval_metrics


def _ragas_metrics() -> list[Any]:
    """Return a metric set compatible with recent and older RAGAS releases."""

    try:
        import ragas.metrics as metrics_module  # type: ignore

        class_names = [
            "LLMContextPrecisionWithReference",
            "LLMContextRecall",
            "Faithfulness",
            "ResponseRelevancy",
            "FactualCorrectness",
        ]
        metrics = [
            getattr(metrics_module, class_name)()
            for class_name in class_names
            if hasattr(metrics_module, class_name)
        ]
        if metrics:
            return metrics
    except Exception:
        pass

    from ragas.metrics import (  # type: ignore
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    return [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]


def _to_langchain_llm(settings: dict[str, Any]) -> tuple[Any, str]:
    """Resolve the active Kotaemon LLM as a LangChain model for RAGAS."""

    from ktem.llms.manager import llms

    reasoning_id = settings.get("reasoning.use") or "simple"
    configured_name = (
        settings.get(f"reasoning.options.{reasoning_id}.llm")
        or settings.get("reasoning.options.simple.llm")
        or ""
    )
    llm_name = configured_name or llms.get_default_name()
    llm = llms.get(llm_name, None) or llms.get_default()
    if not configured_name:
        llm_name = llms.get_default_name()

    if not hasattr(llm, "to_langchain_format"):
        langchain_llm = _openai_compatible_chat_to_langchain(llm)
        if langchain_llm is not None:
            return langchain_llm, llm_name
        raise RuntimeError(
            f"Kotaemon LLM `{llm_name}` cannot be passed to RAGAS. "
            "Use a local LangChain-compatible LLM such as Ollama or LlamaCpp."
        )

    try:
        return llm.to_langchain_format(), llm_name
    except NotImplementedError:
        langchain_llm = _openai_compatible_chat_to_langchain(llm)
        if langchain_llm is not None:
            return langchain_llm, llm_name
        raise RuntimeError(
            f"Kotaemon LLM `{llm_name}` exposes no LangChain adapter for RAGAS. "
            "Use a LangChain-compatible LLM or an OpenAI-compatible local endpoint."
        )


def _openai_compatible_chat_to_langchain(llm: Any) -> Any | None:
    """Adapt Kotaemon's OpenAI-compatible chat client to LangChain for RAGAS.

    Kotaemon's own ``ChatOpenAI`` works for the app and for local Ollama
    OpenAI-compatible endpoints, but it inherits ``to_langchain_format`` from the
    abstract base where it raises ``NotImplementedError``. RAGAS requires a
    LangChain model, so rebuild an equivalent ``langchain_openai.ChatOpenAI``
    when the active Kotaemon LLM has the OpenAI-compatible chat shape.
    """

    if not all(
        hasattr(llm, attr)
        for attr in ("api_key", "model", "prepare_client", "openai_response")
    ):
        return None

    from langchain_openai import ChatOpenAI as LangChainChatOpenAI

    params: dict[str, Any] = {
        "api_key": getattr(llm, "api_key"),
        "model": getattr(llm, "model"),
    }
    optional_attrs = {
        "base_url": "base_url",
        "organization": "organization",
        "timeout": "timeout",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "n": "n",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "logprobs": "logprobs",
        "top_logprobs": "top_logprobs",
        "logit_bias": "logit_bias",
        "top_p": "top_p",
    }
    for source_attr, target_attr in optional_attrs.items():
        value = getattr(llm, source_attr, None)
        if value is not None:
            params[target_attr] = value

    max_retries = getattr(llm, "max_retries", None)
    if max_retries is not None:
        params["max_retries"] = max_retries

    # Avoid token-counting failures for local model names such as ``qwen3:8b``.
    if params.get("base_url"):
        params["tiktoken_model_name"] = "gpt-3.5-turbo"

    return LangChainChatOpenAI(**params)


def _kotaemon_embedding_adapter(embedding_model: Any) -> Any:
    """Wrap any Kotaemon embedding model in LangChain's Embeddings interface."""

    try:
        from langchain_core.embeddings import Embeddings
    except Exception:
        from langchain.embeddings.base import Embeddings  # type: ignore

    class KotaemonEmbeddingsAdapter(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embedded_docs = embedding_model.run(texts)
            return [list(doc.embedding) for doc in embedded_docs]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return KotaemonEmbeddingsAdapter()


def _to_langchain_embeddings(settings: dict[str, Any]) -> tuple[Any, str]:
    """Resolve the active Kotaemon embedding model for RAGAS."""

    from ktem.embeddings.manager import embedding_models_manager

    embedding_name = ""
    for key, value in settings.items():
        if key.startswith("index.options.") and key.endswith(".embedding") and value:
            embedding_name = str(value)
            break

    if not embedding_name or embedding_name == "default":
        embedding_name = embedding_models_manager.get_default_name()

    embedding_model = (
        embedding_models_manager.get(embedding_name, None)
        or embedding_models_manager.get("default", None)
        or embedding_models_manager.get_default()
    )

    if hasattr(embedding_model, "to_langchain_format"):
        return embedding_model.to_langchain_format(), embedding_name

    # LangChain-based Kotaemon embeddings keep the underlying object in `_obj`.
    raw_obj = getattr(embedding_model, "_obj", None)
    if raw_obj is not None and all(
        hasattr(raw_obj, method) for method in ("embed_documents", "embed_query")
    ):
        return raw_obj, embedding_name

    return _kotaemon_embedding_adapter(embedding_model), embedding_name


def _wrap_for_ragas(
    llm: Any, embeddings: Any, run_config: Any | None = None
) -> tuple[Any, Any]:
    """Use RAGAS' official wrappers when the installed version exposes them."""

    try:
        from ragas.llms import LangchainLLMWrapper  # type: ignore

        try:
            llm = LangchainLLMWrapper(llm, run_config=run_config)
        except TypeError:
            llm = LangchainLLMWrapper(llm)
    except Exception:
        # Recent RAGAS versions can also auto-wrap LangChain models in evaluate().
        pass

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

        try:
            embeddings = LangchainEmbeddingsWrapper(embeddings, run_config=run_config)
        except TypeError:
            embeddings = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        pass

    return llm, embeddings


def _local_ragas_evaluator_models(
    settings: dict[str, Any], run_config: Any | None = None
) -> RagasEvaluatorModels:
    """Build the local evaluator LLM/embeddings explicitly for RAGAS.

    If these are omitted, RAGAS creates its default evaluator stack, which is
    OpenAI-backed and requires OPENAI_API_KEY. The app must stay local, so every
    RAGAS evaluate() call receives Kotaemon's configured local models.
    """

    llm, llm_name = _to_langchain_llm(settings)
    _apply_model_timeout(llm, run_config)
    raw_embeddings, embeddings_name = _to_langchain_embeddings(settings)
    llm, embeddings = _wrap_for_ragas(llm, raw_embeddings, run_config)
    _apply_model_timeout(llm, run_config)
    return RagasEvaluatorModels(
        llm=llm,
        embeddings=embeddings,
        raw_embeddings=raw_embeddings,
        llm_name=llm_name,
        embeddings_name=embeddings_name,
        run_config=run_config,
        notes=[
            "RAGAS evaluator uses Kotaemon local models: "
            f"llm={llm_name}, embeddings={embeddings_name}.",
            _run_config_note(run_config),
        ],
    )


def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"id", "source_file", "latency_sec", "context_count", "top_score"}
    return [
        column
        for column in df.columns
        if column not in ignored
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]


def _looks_all_nan(df: pd.DataFrame) -> bool:
    if df.empty:
        return False

    ignored = {"id", "source_file", "question", "answer", "contexts", "ground_truth"}
    candidate_columns = [column for column in df.columns if column not in ignored]
    if not candidate_columns:
        return True

    for column in candidate_columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.notna().any():
            return False
    return True


def _is_finite_score(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_metric_text(text: str) -> str:
    """Canonicalize equivalent numeric, degree, semester, and percent forms."""

    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"\bb\s*\.?\s*sc\s*\.?\b", " bachelor_science ", value)
    value = value.replace("bachelor of science", "bachelor_science")
    value = re.sub(r"\b(?:prozent|percent)\b|%", " percent ", value)
    value = re.sub(r"\b(?:wise|ws|wintersemester)\b", " wintersemester ", value)
    value = re.sub(r"\b(?:sose|ss|sommersemester)\b", " sommersemester ", value)
    replacements = {
        "eins": "1",
        "erste": "1",
        "ersten": "1",
        "erstes": "1",
        "zwei": "2",
        "zweite": "2",
        "zweiten": "2",
        "zweimal": "2",
        "drei": "3",
        "dritte": "3",
        "vier": "4",
        "vierte": "4",
        "fünf": "5",
        "fünfte": "5",
        "sechs": "6",
        "sechste": "6",
        "sechsten": "6",
        "sechstes": "6",
        "sieben": "7",
        "siebte": "7",
        "acht": "8",
        "achte": "8",
        "neun": "9",
        "neunte": "9",
        "zehn": "10",
        "zehnte": "10",
        "one": "1",
        "first": "1",
        "two": "2",
        "second": "2",
        "three": "3",
        "third": "3",
        "four": "4",
        "fourth": "4",
        "five": "5",
        "fifth": "5",
        "seven": "7",
        "six": "6",
        "sixth": "6",
        "ten": "10",
        "maximal": "höchstens",
    }
    for source, target in replacements.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    return " ".join(re.findall(r"[\wäöüß]+", value))


def _token_set(text: str) -> set[str]:

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in _normalize_metric_text(text).split()
        if (len(token) > 2 or token.isdigit()) and token not in stopwords
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _embed_documents(embeddings: Any, texts: list[str]) -> list[list[float]]:
    # LangChain embeddings
    if hasattr(embeddings, "embed_documents"):
        return [list(vector) for vector in embeddings.embed_documents(texts)]

    # RAGAS LangchainEmbeddingsWrapper commonly exposes the wrapped object.
    for attr in ("embeddings", "langchain_embeddings"):
        wrapped = getattr(embeddings, attr, None)
        if wrapped is not None and hasattr(wrapped, "embed_documents"):
            return [list(vector) for vector in wrapped.embed_documents(texts)]

    raise RuntimeError("Could not embed texts with the configured local embeddings.")


def _local_defined_scores(
    rows: list[dict[str, Any]],
    embeddings: Any,
    note: str = (
        "LLM-judge RAGAS metrics were undefined, so the table shows local "
        "always-defined RAGAS-style fallback metrics instead."
    ),
) -> tuple[pd.DataFrame, list[str]]:
    """Always-defined local metrics used when LLM-judge RAGAS scores are NaN.

    These metrics keep the evaluation useful in fully-local mode. Semantic similarity
    and non-LLM string similarity mirror official RAGAS metric families, while keyword
    recalls make retrieval/answer coverage visible without an LLM judge.
    """

    records: list[dict[str, Any]] = []
    notes: list[str] = []
    for row in rows:
        reference = row.get("reference", "")
        answer = row.get("answer", "")
        contexts = " ".join(row.get("contexts") or [])

        try:
            ref_vec, answer_vec = _embed_documents(embeddings, [reference, answer])
            semantic_similarity = round(_cosine_similarity(ref_vec, answer_vec), 4)
        except Exception as exc:
            semantic_similarity = None
            notes.append(f"{row.get('id')}: semantic fallback failed: {exc}")

        string_similarity = round(
            SequenceMatcher(
                None,
                _normalize_metric_text(reference),
                _normalize_metric_text(answer),
            ).ratio(),
            4,
        )
        reference_tokens = _token_set(reference)
        answer_tokens = _token_set(answer)
        context_tokens = _token_set(contexts)
        if reference_tokens:
            answer_keyword_recall = round(
                len(reference_tokens & answer_tokens) / len(reference_tokens), 4
            )
            context_keyword_recall = round(
                len(reference_tokens & context_tokens) / len(reference_tokens), 4
            )
        else:
            answer_keyword_recall = 0.0
            context_keyword_recall = 0.0

        score_values = [
            value
            for value in (
                semantic_similarity,
                string_similarity,
                answer_keyword_recall,
                context_keyword_recall,
            )
            if _is_finite_score(value)
        ]

        records.append(
            {
                "id": row["id"],
                "source_file": row["source_file"],
                "semantic_similarity": semantic_similarity,
                "non_llm_string_similarity": string_similarity,
                "answer_keyword_recall": answer_keyword_recall,
                "context_keyword_recall": context_keyword_recall,
                "ragas_local_score": (
                    round(sum(score_values) / len(score_values), 4)
                    if score_values
                    else None
                ),
            }
        )

    notes.append(note)
    return pd.DataFrame(records), notes


def _evaluate_with_local_models(
    dataset: Any, metrics: list[Any], evaluator: RagasEvaluatorModels
) -> Any:
    """Call ragas.evaluate across old/new RAGAS signatures without losing locality."""

    from ragas import evaluate  # type: ignore

    base_kwargs = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": evaluator.llm,
        "embeddings": evaluator.embeddings,
    }
    optional_kwargs: dict[str, Any] = {
        "raise_exceptions": False,
        "show_progress": False,
    }
    batch_size = _env_optional_int("RAGAS_EVAL_BATCH_SIZE")
    if batch_size is not None:
        optional_kwargs["batch_size"] = batch_size
    if evaluator.run_config is not None:
        optional_kwargs["run_config"] = evaluator.run_config

    while True:
        try:
            return evaluate(**base_kwargs, **optional_kwargs)
        except TypeError as exc:
            message = str(exc)
            unsupported = [
                key
                for key in (
                    "raise_exceptions",
                    "show_progress",
                    "batch_size",
                    "run_config",
                )
                if key in message and key in optional_kwargs
            ]
            if not unsupported:
                raise
            for key in unsupported:
                optional_kwargs.pop(key, None)


def _run_ragas(
    rows: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Execute bounded local RAGAS metric batches without retrying a failed run."""

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return pd.DataFrame(), []

    run_config = _ragas_run_config()
    evaluator = _local_ragas_evaluator_models(settings, run_config)
    metrics = _ragas_metrics()
    notes = list(evaluator.notes)

    try:
        from ragas import EvaluationDataset  # type: ignore

        evaluation_dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["contexts"],
                    "reference": row["reference"],
                }
                for row in valid_rows
            ]
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        from datasets import Dataset  # type: ignore

        evaluation_dataset = Dataset.from_dict(
            {
                "question": [row["question"] for row in valid_rows],
                "answer": [row["answer"] for row in valid_rows],
                "contexts": [row["contexts"] for row in valid_rows],
                "ground_truth": [row["reference"] for row in valid_rows],
            }
        )

    metrics_per_batch = _env_optional_int("RAGAS_EVAL_METRICS_PER_BATCH") or len(
        metrics
    )
    recycle_batches = _env_bool(
        "RAGAS_EVAL_OLLAMA_RECYCLE_METRIC_BATCHES", False
    )
    llm_resource, embedding_resource = _resolve_evaluation_models(settings)
    ragas_df = pd.DataFrame(index=range(len(valid_rows)))
    try:
        for start in range(0, len(metrics), metrics_per_batch):
            metric_batch = metrics[start : start + metrics_per_batch]
            _check_memory_guard(
                f"RAGAS metric batch {start // metrics_per_batch + 1}"
            )
            result = _evaluate_with_local_models(
                evaluation_dataset, metric_batch, evaluator
            )
            batch_df = (
                result.to_pandas()
                if hasattr(result, "to_pandas")
                else pd.DataFrame(result)
            ).reset_index(drop=True)
            for column in batch_df.columns:
                if column not in ragas_df.columns:
                    ragas_df[column] = batch_df[column]
            if recycle_batches:
                _unload_ollama_resource(llm_resource, notes, "RAGAS answer")
                _unload_ollama_resource(
                    embedding_resource, notes, "RAGAS embedding"
                )
                gc.collect()

        for column in ("id", "source_file"):
            if column not in ragas_df.columns:
                ragas_df.insert(0, column, [row[column] for row in valid_rows])

        if _looks_all_nan(ragas_df):
            ragas_df, fallback_notes = _local_defined_scores(
                valid_rows, evaluator.raw_embeddings
            )
            notes.extend(fallback_notes)
        return ragas_df, notes
    finally:
        if _env_bool("RAG_EVAL_OLLAMA_UNLOAD_AT_END", True):
            _unload_ollama_resource(llm_resource, notes, "RAGAS answer")
            _unload_ollama_resource(
                embedding_resource, notes, "RAGAS embedding"
            )


def _effective_runtime_config(
    settings: dict[str, Any], retrieval_scope: str
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "retrieval_scope": retrieval_scope,
        "retrieval_mode": settings.get("index.options.1.retrieval_mode"),
        "retrieval_count": settings.get("index.options.1.num_retrieval"),
        "reranking_enabled": settings.get("index.options.1.use_reranking"),
        "llm_reranking_enabled": settings.get("index.options.1.use_llm_reranking"),
        "table_priority": settings.get("index.options.1.prioritize_table"),
        "context_limit": settings.get("reasoning.max_context_length"),
        "reader_mode": settings.get("index.options.1.reader_mode"),
        "phased_execution": _env_bool("RAG_EVAL_PHASED_EXECUTION", True),
        "max_output_tokens": _env_nonnegative_int(
            "RAG_EVAL_MAX_OUTPUT_TOKENS", 0
        ),
        "list_max_output_tokens": _env_nonnegative_int(
            "RAG_EVAL_LIST_MAX_OUTPUT_TOKENS", 0
        ),
        "ollama_recycle_questions": _env_nonnegative_int(
            "RAG_EVAL_OLLAMA_RECYCLE_QUESTIONS", 0
        ),
        "ollama_embed_keep_alive": str(
            env_config("RAG_EVAL_OLLAMA_EMBED_KEEP_ALIVE", default="-1")
        ),
        "ragas_metrics_per_batch": _env_optional_int(
            "RAGAS_EVAL_METRICS_PER_BATCH"
        )
        or "all",
        "min_available_gb": _env_float("RAG_EVAL_MIN_AVAILABLE_GB", 0.0),
    }
    try:
        from ktem.embeddings.manager import embedding_models_manager
        from ktem.llms.manager import llms

        config["llm"] = llms.get_default_name()
        config["embedding"] = embedding_models_manager.get_default_name()
    except Exception as exc:
        config["model_resolution_warning"] = str(exc)
    return config


def _build_failure_report(samples_df: pd.DataFrame, retrieval_df: pd.DataFrame) -> str:
    lines = ["# RAG retrieval failures", ""]
    if retrieval_df.empty:
        return "\n".join(lines + ["No retrieval diagnostics were produced.", ""])

    sample_by_id = {str(row["id"]): row for _, row in samples_df.iterrows()}
    failure_count = 0
    for _, metric in retrieval_df.iterrows():
        phrase_annotated = int(metric.get("required_phrase_count", 0) or 0) > 0
        exact_included = bool(metric.get("exact_evidence_included", False))
        evidence_succeeded = (
            exact_included
            if phrase_annotated
            else bool(metric.get("answer_chunk_included", False))
        )
        if evidence_succeeded:
            continue
        failure_count += 1
        sample = sample_by_id.get(str(metric["id"]), {})
        if phrase_annotated and not exact_included:
            failure_type = "EXACT EVIDENCE GAP"
            detail = (
                "Not all annotated evidence phrases reached the final context "
                f"({metric.get('required_phrases_included', 0)}/"
                f"{metric.get('required_phrase_count', 0)})."
            )
        elif not bool(metric.get("answer_chunk_candidate_found", False)):
            failure_type = "RETRIEVAL FAILURE"
            detail = "No answer-bearing chunk was detected in initial candidates."
        elif not bool(metric.get("answer_chunk_retrieved", False)):
            failure_type = "RANKING/FILTERING FAILURE"
            detail = (
                "An answer-bearing candidate was found but not returned to the "
                "evidence stage."
            )
        elif bool(metric.get("answer_chunk_dropped", False)):
            failure_type = "PACKING FAILURE"
            detail = "An answer-bearing chunk was retrieved but omitted from evidence."
        else:
            failure_type = "UNCLASSIFIED CONTEXT FAILURE"
            detail = "No answer-bearing chunk reached the final evidence."

        lines.extend(
            [
                f"## {metric['id']}: {failure_type}",
                "",
                f"- Question: {sample.get('question', '')}",
                f"- Expected source: {metric.get('source_file', '')}",
                f"- Relevance labels: {metric.get('relevance_method', '')}",
                f"- First relevant rank: {metric.get('first_relevant_rank', '')}",
                f"- Context trimmed: {bool(metric.get('context_trimmed', False))}",
                f"- Detail: {detail}",
                "",
            ]
        )

    if not failure_count:
        lines.append("No retrieval or packing failures were detected.")
        lines.append("")
    return "\n".join(lines)


def _summarize(
    samples_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples_total": int(len(samples_df)),
        "samples_ok": (
            int((samples_df["status"] == "ok").sum()) if len(samples_df) else 0
        ),
        "samples_failed": (
            int((samples_df["status"] != "ok").sum()) if len(samples_df) else 0
        ),
        "avg_latency_sec": (
            round(float(samples_df["latency_sec"].mean()), 2)
            if "latency_sec" in samples_df and len(samples_df)
            else 0
        ),
    }
    for column in (
        "retrieval_latency_sec",
        "answer_queue_wait_sec",
        "generation_latency_sec",
    ):
        if column in samples_df and len(samples_df):
            summary[f"avg_{column}"] = round(
                float(pd.to_numeric(samples_df[column], errors="coerce").mean()), 2
            )

    for column in ragas_df.columns:
        if column in {"id", "source_file"}:
            continue
        numeric_values = pd.to_numeric(ragas_df[column], errors="coerce")
        if numeric_values.notna().any():
            value = numeric_values.mean(skipna=True)
            if pd.notna(value):
                summary[column] = round(float(value), 4)

    retrieval_summary_columns = (
        "source_hit_at_5",
        "answer_recall_at_1",
        "answer_recall_at_3",
        "answer_recall_at_5",
        "answer_recall_at_10",
        "reciprocal_rank",
        "answer_chunk_retrieved",
        "answer_chunk_candidate_found",
        "answer_chunk_included",
        "answer_chunk_dropped",
        "duplicate_ratio",
        "context_trimmed",
        "required_phrase_recall_retrieved",
        "required_phrase_recall_included",
        "exact_evidence_retrieved",
        "exact_evidence_included",
    )
    for column in retrieval_summary_columns:
        if column not in retrieval_df or retrieval_df.empty:
            continue
        numeric_values = pd.to_numeric(retrieval_df[column], errors="coerce")
        if numeric_values.notna().any():
            summary[f"retrieval_{column}"] = round(
                float(numeric_values.mean(skipna=True)), 4
            )

    return summary


def _evaluation_error_row(sample: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "question": sample["question"],
        "reference": sample["reference"],
        "source_file": sample.get("source_file", ""),
        "indexed_source": "",
        "answer": "",
        "contexts": [],
        "context_count": 0,
        "top_context_preview": "",
        "top_source": "",
        "top_score": None,
        "latency_sec": 0,
        "retrieval_latency_sec": 0,
        "answer_queue_wait_sec": 0,
        "generation_latency_sec": 0,
        "answer_output_token_limit": 0,
        "status": "error",
        "error": f"{exc}\n{traceback.format_exc(limit=2)}",
    }


def _retrieval_error_row(
    sample: dict[str, Any], retrieval_scope: str, exc: Exception
) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "source_file": sample.get("source_file", ""),
        "retrieval_scope": retrieval_scope,
        "relevance_method": (
            "manual"
            if sample.get("has_manual_relevance_annotations")
            else "inferred_keyword"
        ),
        "retrieved_count": 0,
        "answer_chunk_candidate_found": False,
        "answer_chunk_retrieved": False,
        "answer_chunk_included": False,
        "answer_chunk_dropped": False,
        "required_phrase_count": len(sample.get("required_phrases", [])),
        "required_phrases_retrieved": 0,
        "required_phrases_included": 0,
        "required_phrase_recall_retrieved": 0.0,
        "required_phrase_recall_included": 0.0,
        "exact_evidence_retrieved": False,
        "exact_evidence_included": False,
        "error": str(exc),
    }


def run_evaluation(
    app: Any,
    settings: dict[str, Any],
    user_id: Any,
    dataset_path: str | Path,
    question_limit: int,
    run_ragas_metrics: bool = True,
    retrieval_scope: str = "expected-source",
    progress: ProgressFn | None = None,
) -> EvalRunResult:
    """Run Kotaemon RAG over a dataset subset and optionally score it with RAGAS."""

    if retrieval_scope not in {"expected-source", "all"}:
        raise ValueError("retrieval_scope must be 'expected-source' or 'all'")

    samples = load_eval_dataset(dataset_path)
    limit = max(1, min(int(question_limit), len(samples)))
    samples = samples[:limit]

    eval_settings = _ensure_simple_reasoning_settings(settings)
    rows_by_index: list[dict[str, Any] | None] = [None] * limit
    candidate_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    llm, embedding = _resolve_evaluation_models(eval_settings)
    phased = _env_bool("RAG_EVAL_PHASED_EXECUTION", True)
    recycle_every = _env_nonnegative_int(
        "RAG_EVAL_OLLAMA_RECYCLE_QUESTIONS", 0
    )
    unload_at_end = _env_bool("RAG_EVAL_OLLAMA_UNLOAD_AT_END", True)
    embedding_keep_alive = str(
        env_config("RAG_EVAL_OLLAMA_EMBED_KEEP_ALIVE", default="-1")
    )
    max_output_tokens = _env_nonnegative_int("RAG_EVAL_MAX_OUTPUT_TOKENS", 0)

    if phased:
        prepared_samples: list[tuple[int, PreparedEvalSample]] = []
        with _temporary_model_attribute(
            embedding, "ollama_keep_alive", embedding_keep_alive
        ):
            for idx, sample in enumerate(samples, start=1):
                if progress:
                    progress(
                        idx - 1,
                        limit * 2,
                        f"Retrieval {idx}/{limit}: {sample['id']}",
                    )
                try:
                    _check_memory_guard(f"retrieval question {idx}")
                    prepared = _retrieve_with_pipeline(
                        app,
                        eval_settings,
                        user_id,
                        sample,
                        retrieval_scope=retrieval_scope,
                    )
                    prepared_samples.append((idx - 1, prepared))
                    candidate_rows.extend(prepared.candidates)
                    retrieval_rows.append(prepared.retrieval_metrics)
                except Exception as exc:
                    warnings.append(f"{sample['id']}: {exc}")
                    rows_by_index[idx - 1] = _evaluation_error_row(sample, exc)
                    retrieval_rows.append(
                        _retrieval_error_row(sample, retrieval_scope, exc)
                    )
                finally:
                    if idx % _env_int("RAG_EVAL_GC_EVERY", 10) == 0:
                        gc.collect()

        _unload_ollama_resource(embedding, warnings, "embedding")

        for answer_idx, (row_index, prepared) in enumerate(
            prepared_samples, start=1
        ):
            if progress:
                progress(
                    limit + answer_idx - 1,
                    limit * 2,
                    f"Answer {answer_idx}/{len(prepared_samples)}: "
                    f"{prepared.sample['id']}",
                )
            try:
                _check_memory_guard(f"answer question {answer_idx}")
                output_limit = _answer_output_limit(
                    prepared.sample,
                    max_output_tokens or getattr(llm, "max_tokens", 256),
                )
                prepared.row["answer_output_token_limit"] = output_limit
                with _temporary_model_attribute(llm, "max_tokens", output_limit):
                    rows_by_index[row_index] = _answer_prepared_sample(
                        eval_settings, prepared
                    )
            except Exception as exc:
                warnings.append(f"{prepared.sample['id']}: {exc}")
                prepared.row.update(
                    {
                        "status": "error",
                        "error": f"{exc}\n{traceback.format_exc(limit=2)}",
                        "latency_sec": prepared.row.get(
                            "retrieval_latency_sec", 0
                        ),
                    }
                )
                rows_by_index[row_index] = prepared.row
            finally:
                prepared.evidence = ""
                prepared.images = []
                if recycle_every and answer_idx % recycle_every == 0:
                    _unload_ollama_resource(llm, warnings, "answer")
                if answer_idx % _env_int("RAG_EVAL_GC_EVERY", 10) == 0:
                    gc.collect()
    else:
        with _temporary_model_attribute(
            embedding, "ollama_keep_alive", embedding_keep_alive
        ):
            for idx, sample in enumerate(samples, start=1):
                if progress:
                    progress(
                        idx - 1, limit, f"Question {idx}/{limit}: {sample['id']}"
                    )
                try:
                    _check_memory_guard(f"question {idx}")
                    output_limit = _answer_output_limit(
                        sample,
                        max_output_tokens or getattr(llm, "max_tokens", 256),
                    )
                    with _temporary_model_attribute(
                        llm, "max_tokens", output_limit
                    ):
                        row, candidates, retrieval_metrics = _answer_with_pipeline(
                            app,
                            eval_settings,
                            user_id,
                            sample,
                            retrieval_scope=retrieval_scope,
                        )
                    row["answer_output_token_limit"] = output_limit
                    rows_by_index[idx - 1] = row
                    candidate_rows.extend(candidates)
                    retrieval_rows.append(retrieval_metrics)
                except Exception as exc:
                    warnings.append(f"{sample['id']}: {exc}")
                    rows_by_index[idx - 1] = _evaluation_error_row(sample, exc)
                    retrieval_rows.append(
                        _retrieval_error_row(sample, retrieval_scope, exc)
                    )
                finally:
                    if recycle_every and idx % recycle_every == 0:
                        _unload_ollama_resource(llm, warnings, "answer")
                    if idx % _env_int("RAG_EVAL_GC_EVERY", 10) == 0:
                        gc.collect()

    if unload_at_end:
        _unload_ollama_resource(embedding, warnings, "embedding")
        _unload_ollama_resource(llm, warnings, "answer")

    rows = [row for row in rows_by_index if row is not None]

    if progress:
        progress(limit, limit, "RAG answers collected")

    ragas_df = pd.DataFrame()
    if run_ragas_metrics:
        try:
            ragas_df, ragas_notes = _run_ragas(rows, eval_settings)
            warnings.extend(ragas_notes)
        except ModuleNotFoundError as exc:
            warnings.append(
                f"RAGAS dependencies are not installed ({exc.name}). "
                "Install/update dependencies with `pip install -r requirements.txt`."
            )
        except Exception as exc:
            warnings.append(f"RAGAS scoring failed: {exc}")
    elif _env_bool("RAG_EVAL_LIGHTWEIGHT_METRICS", True):
        valid_rows = [row for row in rows if row.get("status") == "ok"]
        if valid_rows:
            try:
                _check_memory_guard("lightweight answer-quality metrics")
                scoring_embeddings, _ = _to_langchain_embeddings(eval_settings)
                with _temporary_model_attribute(
                    embedding, "ollama_keep_alive", embedding_keep_alive
                ):
                    ragas_df, local_notes = _local_defined_scores(
                        valid_rows,
                        scoring_embeddings,
                        note=(
                            "Showing lightweight local answer-quality metrics. "
                            "Enable RAGAS in the UI for LLM-judge metrics."
                        ),
                    )
                warnings.extend(local_notes)
            except Exception as exc:
                warnings.append(f"Lightweight quality scoring failed: {exc}")
            finally:
                if unload_at_end:
                    _unload_ollama_resource(
                        embedding, warnings, "quality-metric embedding"
                    )

    samples_df = pd.DataFrame(rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    summary = _summarize(samples_df, ragas_df, retrieval_df)
    runtime_config = _effective_runtime_config(eval_settings, retrieval_scope)
    failure_report = _build_failure_report(samples_df, retrieval_df)
    return EvalRunResult(
        samples=samples_df,
        ragas_scores=ragas_df,
        retrieval_metrics=retrieval_df,
        retrieval_candidates=candidates_df,
        summary=summary,
        runtime_config=runtime_config,
        failure_report=failure_report,
        warnings=warnings,
    )
