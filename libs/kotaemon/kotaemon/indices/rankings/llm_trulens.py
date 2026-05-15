from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Lock

import tiktoken

from kotaemon.base import Document, HumanMessage, SystemMessage
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.llms import BaseLLM, PromptTemplate

from .llm import LLMReranking

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    """Rate how relevant the document is to the query on a scale from 0 to 10.
Return only one number from 0 to 10.
Do not include explanations, words, JSON, or punctuation.
Examples:
0
5
10"""  # noqa: E501
)

USER_PROMPT_TEMPLATE = PromptTemplate(
    """QUESTION: {question}

        CONTEXT: {context}

        RELEVANCE: """
)  # noqa

PATTERN_INTEGER: re.Pattern = re.compile(r"([+-]?[1-9][0-9]*|0)")
"""Regex that matches integers."""
PATTERN_NUMBER: re.Pattern = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?![\w.])")
"""Regex that matches standalone integer and decimal numbers."""
PATTERN_FRACTION_10: re.Pattern = re.compile(
    r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*/\s*10(?:\.0+)?(?![\w.])",
    re.IGNORECASE,
)
PATTERN_OUT_OF_10: re.Pattern = re.compile(
    r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:out\s+of|/)\s*10(?:\.0+)?(?![\w.])",
    re.IGNORECASE,
)
PATTERN_LABELED_NUMBER: re.Pattern = re.compile(
    r"\b(?:score|rating|relevance|relevancy|relevant)\b\s*[:=\-]?\s*"
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*/\s*10(?:\.0+)?)?",
    re.IGNORECASE,
)

MAX_CONTEXT_LEN = 7500
FALLBACK_RELEVANCE_SCORE = 0.0

_ERROR_COUNT_LOCK = Lock()
_LLM_RELEVANCE_SCORER_ERRORS_COUNT = 0


def validate_rating(rating) -> int:
    """Validate a rating is between 0 and 10."""

    if not 0 <= rating <= 10:
        raise ValueError("Rating must be between 0 and 10")

    return rating


def _coerce_rating(value: str | int | float) -> float | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, rating))


def _rating_from_json_like(text: str) -> float | None:
    """Best-effort extraction for JSON/JSON-like scorer outputs."""
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("score", "rating", "relevance", "relevancy"):
            if key in parsed:
                return _coerce_rating(parsed[key])
    elif isinstance(parsed, (int, float)):
        return _coerce_rating(parsed)

    # Handle non-strict JSON-ish fragments embedded in explanations, e.g.
    # ``Here is my answer: {"score": 8, "reason": ...}``.
    match = re.search(
        r"""["']?(?:score|rating|relevance|relevancy)["']?\s*[:=]\s*
            ["']?([+-]?(?:\d+(?:\.\d+)?|\.\d+))["']?""",
        text,
        re.IGNORECASE | re.VERBOSE,
    )
    if match:
        return _coerce_rating(match.group(1))

    return None


def _first_valid_rating(matches: list[str], *, clamp: bool = False) -> float | None:
    for match in matches:
        try:
            value = float(match)
        except (TypeError, ValueError):
            continue
        if clamp:
            return max(0.0, min(10.0, value))
        if 0.0 <= value <= 10.0:
            return value
    return None


def re_0_10_rating(text: str, default: float | None = None) -> float:
    """Extract a 0-10 rating from an LLM output string.

    This parser intentionally accepts common non-compliant model outputs such as
    ``"Score: 8"``, ``"8/10"``, ``"I would rate this 8 out of 10"``,
    ``"Rating: 7.5"``, JSON-like ``{"score": 8}``, and answers with extra
    explanation. It returns the first parseable rating in the 0-10 range and
    supports decimals.

    Args:
        text: String to extract rating from.
        default: Fallback value to return if no rating is parseable. If omitted,
            a ValueError is raised.

    Returns:
        float: Extracted rating.

    Raises:
        ValueError: If no 0-10 rating is found and no default is provided.
    """
    raw = "" if text is None else str(text)
    rating = _rating_from_json_like(raw)
    if rating is not None:
        return rating

    # Prefer explicit ``8/10`` or ``8 out of 10`` ratings over unrelated numbers.
    for pattern in (PATTERN_OUT_OF_10, PATTERN_FRACTION_10, PATTERN_LABELED_NUMBER):
        rating = _first_valid_rating(pattern.findall(raw), clamp=True)
        if rating is not None:
            return rating

    numeric_matches = PATTERN_NUMBER.findall(raw)
    rating = _first_valid_rating(numeric_matches)
    if rating is not None:
        return rating
    if len(numeric_matches) == 1:
        rating = _first_valid_rating(numeric_matches, clamp=True)
        if rating is not None:
            return rating

    if default is not None:
        return max(0.0, min(10.0, float(default)))

    raise ValueError(f"Could not parse 0-10 rating from: {raw!r}")


def _repo_root() -> Path:
    # libs/kotaemon/kotaemon/indices/rankings/llm_trulens.py -> repository root
    return Path(__file__).resolve().parents[5]


def _relevance_error_log_path() -> Path:
    return _repo_root() / "ku_d3b_eval" / "debug" / "llm_relevance_scorer_errors.jsonl"


def get_llm_relevance_scorer_errors_count() -> int:
    with _ERROR_COUNT_LOCK:
        return _LLM_RELEVANCE_SCORER_ERRORS_COUNT


def _doc_source(doc: Document) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    for key in (
        "file_name",
        "file_path",
        "source",
        "path",
        "url",
        "page_label",
        "doc_id",
    ):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(getattr(doc, "doc_id", "") or "")


def log_llm_relevance_scorer_parse_failure(
    *,
    query: str,
    document_index: int,
    document: Document | None,
    raw_llm_output: str,
    error: str,
    fallback_score: float = FALLBACK_RELEVANCE_SCORE,
) -> None:
    """Append one malformed scorer output to the KU D3B debug JSONL log."""
    global _LLM_RELEVANCE_SCORER_ERRORS_COUNT

    payload = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "document_index": document_index,
        "document_source": _doc_source(document) if document is not None else "",
        "document_metadata": getattr(document, "metadata", {}) if document is not None else {},
        "raw_llm_output": raw_llm_output,
        "error": error,
        "fallback_score": fallback_score,
    }

    try:
        path = _relevance_error_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("Could not write LLM relevance scorer parse failure log: %s", exc)
    finally:
        with _ERROR_COUNT_LOCK:
            _LLM_RELEVANCE_SCORER_ERRORS_COUNT += 1


class LLMTrulensScoring(LLMReranking):
    llm: BaseLLM
    system_prompt_template: PromptTemplate = SYSTEM_PROMPT_TEMPLATE
    user_prompt_template: PromptTemplate = USER_PROMPT_TEMPLATE
    concurrent: bool = True
    normalize: float = 10
    trim_func: TokenSplitter = TokenSplitter.withx(
        chunk_size=MAX_CONTEXT_LEN,
        chunk_overlap=0,
        separator=" ",
        tokenizer=partial(
            tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
            allowed_special=set(),
            disallowed_special="all",
        ),
    )

    def run(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []

        documents = sorted(documents, key=lambda doc: doc.get_content())
        if self.concurrent:
            with ThreadPoolExecutor() as executor:
                futures = []
                for doc in documents:
                    chunked_doc_content = self.trim_func(
                        [
                            Document(content=doc.get_content())
                            # skip metadata which cause troubles
                        ]
                    )[0].text

                    messages = []
                    messages.append(
                        SystemMessage(self.system_prompt_template.populate())
                    )
                    messages.append(
                        HumanMessage(
                            self.user_prompt_template.populate(
                                question=query, context=chunked_doc_content
                            )
                        )
                    )

                    def llm_call(messages=messages):
                        return self.llm(messages).text

                    futures.append(executor.submit(llm_call))

                results = [future.result() for future in futures]
        else:
            results = []
            for doc in documents:
                messages = []
                messages.append(SystemMessage(self.system_prompt_template.populate()))
                messages.append(
                    SystemMessage(
                        self.user_prompt_template.populate(
                            question=query, context=doc.get_content()
                        )
                    )
                )
                results.append(self.llm(messages).text)

        parsed_results = []
        parse_errors_count = 0
        for r_idx, result in enumerate(results):
            try:
                rating = re_0_10_rating(result)
            except Exception as exc:
                parse_errors_count += 1
                fallback = FALLBACK_RELEVANCE_SCORE
                logger.warning(
                    "Malformed LLM relevance score for document %s; using fallback %s: %s",
                    r_idx,
                    fallback,
                    exc,
                )
                log_llm_relevance_scorer_parse_failure(
                    query=query,
                    document_index=r_idx,
                    document=documents[r_idx] if r_idx < len(documents) else None,
                    raw_llm_output=str(result),
                    error=f"{exc.__class__.__name__}: {exc}",
                    fallback_score=fallback,
                )
                rating = re_0_10_rating(result, default=fallback)

            parsed_results.append((r_idx, float(rating) / self.normalize))

        try:
            self.parse_errors_count = parse_errors_count
        except Exception:
            pass

        parsed_results.sort(key=lambda x: x[1], reverse=True)

        for r_idx, score in parsed_results:
            doc = documents[r_idx]
            doc.metadata["llm_trulens_score"] = score
            filtered_docs.append(doc)

        print(
            "LLM rerank scores",
            [doc.metadata["llm_trulens_score"] for doc in filtered_docs],
        )

        return filtered_docs
