"""Deterministic contextual enrichment for structured document retrieval.

The helpers in this module deliberately avoid model calls.  They enrich chunks with
document structure at indexing time and use that metadata for inexpensive reranking
and bounded neighbor expansion at retrieval time.
"""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from kotaemon.base import Document, RetrievedDocument

logger = logging.getLogger(__name__)

_MARKDOWN_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
_LEGAL_SECTION = re.compile(
    r"^\s*(?:#{1,6}\s*)?(§{1,2}\s*\d+[a-zA-Z]?(?:\s+[^\n]+)?)\s*$",
    re.IGNORECASE,
)
_PROFILE_SECTION = re.compile(
    r"^\s*(?:#{1,6}\s*)?(Studienprofil|Wahlpflichtbereich|Kompetenzbereich|"
    r"Schwerpunkt|Vertiefung)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_PLAIN_SECTION = re.compile(
    r"^\s*(?:studienprofil|profil|section|subsection|abschnitt|kapitel|chapter|"
    r"bereich|kompetenzbereich|schwerpunkt|vertiefung)\s*:\s*.+$",
    re.IGNORECASE,
)
_PAGE_QUERY = re.compile(r"\b(?:page|seite|s\.)\s*(\d+)\b", re.IGNORECASE)
_TOKEN = re.compile(r"[\wÄÖÜäöüß]+", re.UNICODE)
_MARKDOWN_NOISE = re.compile(r"^[\s|:\-#*_`]+$")


def _stable_id(*parts: Any) -> str:
    value = "\x1f".join(str(part or "").strip().lower() for part in parts)
    return sha1(value.encode("utf-8")).hexdigest()[:20]


def _structural_boundary(line: str) -> str:
    legal = _LEGAL_SECTION.match(line)
    if legal:
        return legal.group(1).strip()
    profile = _PROFILE_SECTION.match(line)
    if profile:
        return f"{profile.group(1).strip()}: {profile.group(2).strip()}"
    return ""


def _is_hard_boundary(line: str) -> bool:
    if _structural_boundary(line):
        return True
    markdown = _MARKDOWN_HEADING.match(line)
    return bool(markdown and len(markdown.group(1)) >= 2)


def _split_hard_sections(chunks: list[Document]) -> list[Document]:
    """Split chunks that contain multiple legal/profile sections.

    The token splitter is intentionally generic and can otherwise put the end of
    one APO paragraph beside the start of another. These boundaries are semantic,
    so preserving them is more important than preserving the split chunk id.
    """

    output: list[Document] = []
    for chunk in chunks:
        lines = (chunk.text or "").splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines) if _is_hard_boundary(line.strip())]
        if len(starts) <= 1:
            output.append(chunk)
            continue
        cuts = ([0] if starts[0] else []) + starts
        cuts = sorted(set(cuts))
        cuts.append(len(lines))
        for part_index, (start, end) in enumerate(zip(cuts, cuts[1:])):
            text = "".join(lines[start:end]).strip()
            if not text:
                continue
            metadata = dict(chunk.metadata or {})
            metadata["structural_split_parent_id"] = chunk.doc_id
            metadata["structural_split_index"] = part_index
            doc_id = (
                chunk.doc_id
                if part_index == 0
                else f"{chunk.doc_id}-section-{_stable_id(chunk.doc_id, part_index, text[:120])}"
            )
            output.append(Document(id_=doc_id, text=text, metadata=metadata))
    return output


def _source_name(metadata: dict[str, Any], fallback: str = "") -> str:
    return str(
        metadata.get("source_file")
        or metadata.get("file_name")
        or metadata.get("filename")
        or fallback
    )


def _page_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return 10**9, str(value or "")


def _heading_lines(text: str, document_title: str = "") -> list[tuple[int, str]]:
    """Extract explicit Markdown headings plus conservative plain-text headings."""

    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        match = _MARKDOWN_HEADING.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip()))
            continue

        # Docling may preserve the wording but not the Markdown marker. Restrict
        # plain-text inference to the document's first line and clear section labels
        # to avoid treating module/table rows as headings.
        if (
            not headings
            and not document_title
            and index == 0
            and len(line) <= 160
            and not _structural_boundary(line)
        ):
            headings.append((1, line))
        elif _PLAIN_SECTION.match(line) and len(line) <= 180:
            headings.append((2, line))

        # Markdown underline-style headings.
        if index + 1 < len(lines) and re.fullmatch(r"\s*(?:=+|-+)\s*", lines[index + 1]):
            headings.append((1 if "=" in lines[index + 1] else 2, line))
    return headings


def _contains_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_rows = [line for line in lines if line.count("|") >= 2]
    tab_rows = [line for line in lines if line.count("\t") >= 2]
    return len(pipe_rows) >= 2 or len(tab_rows) >= 2


def _parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Parse simple Markdown-pipe or tab-separated tables."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum(line.count("|") >= 2 for line in lines) >= 2:
        raw_rows = [line for line in lines if line.count("|") >= 2]
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in raw_rows]
    elif sum(line.count("\t") >= 2 for line in lines) >= 2:
        raw_rows = [line for line in lines if line.count("\t") >= 2]
        rows = [[cell.strip() for cell in line.split("\t")] for line in raw_rows]
    else:
        return [], []

    rows = [row for row in rows if any(row)]
    if not rows:
        return [], []
    headers = rows[0]
    body = [
        row
        for row in rows[1:]
        if not all(re.fullmatch(r"\s*:?-{2,}:?\s*", cell or "") for cell in row)
    ]
    return headers, body


def _header_index(headers: list[str], *needles: str) -> int | None:
    normalized = [" ".join(_TOKEN.findall(header.lower())) for header in headers]
    for index, header in enumerate(normalized):
        if any(needle in header for needle in needles):
            return index
    return None


def _table_row_fact_records(
    text: str,
    source_file: str,
    page: Any,
    section: str,
) -> list[dict[str, str]]:
    """Return normalized table rows with a self-contained German fact."""

    headers, rows = _parse_table(text)
    if not headers or not rows:
        return []

    module_i = _header_index(headers, "modulbezeichnung", "modul", "module")
    exam_i = _header_index(headers, "prüfungsform", "pruefungsform", "exam")
    ects_i = _header_index(headers, "ects")
    semester_i = _header_index(headers, "semesterlage", "semester")
    prerequisite_i = _header_index(
        headers, "zulassungsvoraussetzungen", "voraussetzung", "prerequisite"
    )

    def value(row: list[str], index: int | None) -> str:
        return row[index].strip() if index is not None and index < len(row) else ""

    facts: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        module = value(row, module_i)
        if not module:
            continue
        prefix = f'Im Dokument „{source_file}“'
        if page not in (None, ""):
            prefix += f", Seite {page}"
        if section:
            prefix += f', im Abschnitt „{section}“'

        clauses = [f'hat das Modul „{module}“']
        exam = value(row, exam_i)
        ects = value(row, ects_i)
        semester = value(row, semester_i)
        prerequisite = value(row, prerequisite_i)
        if exam:
            clauses.append(f'die Prüfungsform „{exam}“')
        if ects:
            clauses.append(f"umfasst {ects} ECTS")
        if semester:
            clauses.append(f"liegt im {semester}")
        if prerequisite:
            if prerequisite in {"-", "–", "—", "keine", "none"}:
                clauses.append("hat keine Zulassungsvoraussetzungen")
            else:
                clauses.append(f'hat die Zulassungsvoraussetzung „{prerequisite}“')
        facts.append(
            {
                "row_index": str(row_index),
                "module": module,
                "exam_form": exam,
                "ects": ects,
                "semester": semester,
                "prerequisites": prerequisite,
                "fact": prefix + ", " + ", ".join(clauses) + ".",
            }
        )
    return facts


def table_row_facts(text: str, source_file: str, page: Any, section: str) -> str:
    """Render table rows as self-contained German facts (legacy helper)."""

    return "\n".join(
        item["fact"]
        for item in _table_row_fact_records(text, source_file, page, section)
    )


def enrich_chunks(
    chunks: list[Document],
    file_name: str,
    enable_table_row_facts: bool = False,
    enable_table_row_fact_chunks: bool | None = None,
) -> list[Document]:
    """Attach stable section/table metadata and optionally create row chunks."""

    if not chunks:
        return chunks

    if enable_table_row_fact_chunks is None:
        enable_table_row_fact_chunks = enable_table_row_facts

    chunks = _split_hard_sections(chunks)

    # Page order is the best deterministic approximation available after Docling
    # emits page text and tables as separate records. Python's stable sort preserves
    # the splitter order within each page.
    ordered = [
        chunk
        for _, chunk in sorted(
            enumerate(chunks),
            key=lambda item: (
                _page_sort_key((item[1].metadata or {}).get("page_label")),
                item[0],
            ),
        )
    ]

    source = file_name
    document_title = ""
    hierarchy: dict[int, str] = {}
    enriched: list[Document] = []
    row_chunks: list[Document] = []
    for index, chunk in enumerate(ordered):
        metadata = dict(chunk.metadata or {})
        original_text = chunk.text or ""
        source = _source_name(metadata, file_name)
        boundary = next(
            (
                _structural_boundary(line.strip())
                for line in original_text.splitlines()
                if _structural_boundary(line.strip())
            ),
            "",
        )
        headings = _heading_lines(original_text, document_title)
        if boundary and not any(heading == boundary for _, heading in headings):
            headings.append((2, boundary))
        for level, heading in headings:
            hierarchy = {key: value for key, value in hierarchy.items() if key < level}
            hierarchy[level] = heading
            if level == 1 and not document_title:
                document_title = heading

        if not document_title:
            document_title = Path(source).stem.replace("_", " ")
        nearest_heading = hierarchy[max(hierarchy)] if hierarchy else ""
        section_heading = next(
            (hierarchy[level] for level in sorted(hierarchy, reverse=True) if level >= 2),
            nearest_heading,
        )
        heading_path = " > ".join(hierarchy[level] for level in sorted(hierarchy))
        parent_heading = ""
        levels = sorted(hierarchy)
        if len(levels) > 1:
            parent_heading = hierarchy[levels[-2]]
        section_key = heading_path or section_heading
        section_id = (
            f"section-{_stable_id(source, section_key)}" if section_key else ""
        )
        page = metadata.get("page_label", metadata.get("page_number"))
        is_table = metadata.get("type") == "table" or _contains_markdown_table(
            original_text
        )
        table_context = ""
        row_facts = ""
        table_heading = ""
        table_id = ""
        if is_table:
            section_label = section_heading or nearest_heading or document_title
            table_heading = section_label
            table_id = f"table-{_stable_id(source, page, section_id, original_text[:500])}"
            table_context = (
                f"The following table belongs to section {section_label} and lists "
                "structured document fields such as modules, examination form, ECTS, "
                "semester and admission requirements."
            )
            records = _table_row_fact_records(original_text, source, page, section_label)
            row_facts = "\n".join(record["fact"] for record in records)
            if enable_table_row_fact_chunks:
                for record in records:
                    fact = record["fact"]
                    row_metadata = dict(metadata)
                    row_metadata.update(
                        {
                            "source_file": source,
                            "page_number": page if page is not None else "",
                            "document_title": document_title,
                            "document_type": "table_row_fact",
                            "section_id": section_id,
                            "section_heading": section_heading,
                            "section_path": heading_path,
                            "parent_heading": parent_heading,
                            "table_id": table_id,
                            "table_heading": table_heading,
                            "table_row_index": record["row_index"],
                            "module": record["module"],
                            "exam_form": record["exam_form"],
                            "ects": record["ects"],
                            "semester": record["semester"],
                            "prerequisites": record["prerequisites"],
                            "original_text": fact,
                            "enriched_text": "\n".join(
                                part
                                for part in (
                                    f"Document: {source}",
                                    f"Page: {page}" if page not in (None, "") else "",
                                    f"Section: {heading_path}" if heading_path else "",
                                    f"Table: {table_heading}" if table_heading else "",
                                    fact,
                                )
                                if part
                            ),
                            "previous_chunk_id": "",
                            "next_chunk_id": "",
                        }
                    )
                    row_chunks.append(
                        Document(
                            id_=f"{chunk.doc_id}-row-{_stable_id(table_id, record['row_index'], fact)}",
                            text=fact,
                            metadata=row_metadata,
                        )
                    )

        prefix = [f"Document: {source}"]
        if page not in (None, ""):
            prefix.append(f"Page: {page}")
        if heading_path:
            prefix.append(f"Section: {heading_path}")
        if table_heading:
            prefix.append(f"Table: {table_heading}")
        enriched_parts = ["\n".join(prefix), original_text]

        metadata.update(
            {
                "source_file": source,
                "page_number": page if page is not None else "",
                "chunk_index": index,
                "previous_chunk_id": ordered[index - 1].doc_id if index else "",
                "next_chunk_id": (
                    ordered[index + 1].doc_id if index + 1 < len(ordered) else ""
                ),
                "parent_section_heading": section_heading,
                "nearest_markdown_heading": nearest_heading,
                "heading_path": heading_path,
                "section_id": section_id,
                "section_heading": section_heading,
                "section_path": heading_path,
                "parent_heading": parent_heading,
                "document_title": document_title,
                "document_type": metadata.get("type", "text"),
                "table_id": table_id,
                "table_heading": table_heading,
                "table_context": table_context,
                "table_row_facts": row_facts,
                "original_text": original_text,
                "enriched_text": "\n\n".join(part for part in enriched_parts if part),
            }
        )
        chunk.metadata = metadata
        enriched.append(chunk)

    # Row facts are independent retrieval units. They deliberately do not join the
    # physical previous/next chain, so expanding one cannot leak into another row.
    for offset, row_chunk in enumerate(row_chunks, start=len(enriched)):
        row_chunk.metadata["chunk_index"] = offset
    return enriched + row_chunks


def use_contextual_text(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Use stored enriched text for LLM context while retaining original metadata."""

    for document in documents:
        enriched = (document.metadata or {}).get("enriched_text")
        if enriched:
            document.text = str(enriched)
            document.content = str(enriched)
    return documents


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(str(value or "").lower())
        if len(token) >= 3
    }


def _document_family(query: str) -> str:
    value = query.lower()
    if re.search(r"\bapo\s*4\b|prüfungsordnung.*(?:2024|version\s*4)", value):
        return "apo_4"
    if re.search(r"\bapo\s*3\b|prüfungsordnung.*(?:2020|version\s*3)", value):
        return "apo_3"
    if any(term in value for term in ("wahlpflicht", "elective", "studienprofil")):
        return "wahlpflicht"
    if any(term in value for term in ("studienverlaufsplan", "study plan")):
        return "studienverlaufsplan"
    if re.search(r"\bsemester\b|\bsemesters\b", value):
        return "studienverlaufsplan"
    if any(term in value for term in ("studiengangsbeschreibung", "program description")):
        return "studiengangsbeschreibung"
    return ""


def metadata_rerank(
    documents: list[RetrievedDocument],
    query: str,
    enable_document_routing: bool = False,
) -> list[RetrievedDocument]:
    """Apply deterministic, inspectable section and document-family boosts."""

    query_tokens = _tokens(query)
    if not documents or not query_tokens:
        return documents
    requested_pages = set(_PAGE_QUERY.findall(query))
    filename_matches = [
        bool(query_tokens & _tokens(_source_name(doc.metadata or {})))
        for doc in documents
    ]
    query_names_a_source = any(filename_matches)

    requested_family = _document_family(query) if enable_document_routing else ""
    for doc, filename_match in zip(documents, filename_matches):
        metadata = doc.metadata or {}
        base_score = float(doc.score) if doc.score is not None and doc.score >= 0 else 0.0
        original_score = doc.score
        heading_tokens = _tokens(
            metadata.get("section_path")
            or metadata.get("heading_path")
            or metadata.get("parent_section_heading")
            or metadata.get("nearest_markdown_heading")
        )
        table_tokens = _tokens(metadata.get("table_context"))
        text = str(
            metadata.get("original_text")
            or metadata.get("enriched_text")
            or doc.text
            or ""
        )
        text_tokens = _tokens(text)
        page = str(metadata.get("page_label", metadata.get("page_number", "")))

        metadata_boost = 0.0
        routing_boost = 0.0
        metadata_boost += 0.08 if filename_match else 0.0
        metadata_boost += 0.18 * len(query_tokens & heading_tokens) / len(query_tokens)
        metadata_boost += 0.06 * len(query_tokens & table_tokens) / len(query_tokens)
        metadata_boost += 0.14 * len(query_tokens & text_tokens) / len(query_tokens)
        metadata_boost += 0.10 if requested_pages and page in requested_pages else 0.0
        if query_names_a_source and not filename_match:
            metadata_boost -= 0.10
        useful = re.sub(r"[\s|:\-#*_`]", "", text)
        if len(useful) < 24 or _MARKDOWN_NOISE.fullmatch(text.strip() or "-"):
            metadata_boost -= 0.25

        if requested_family:
            source_value = _source_name(metadata).lower().replace("-", "_")
            document_type = str(metadata.get("document_type") or "").lower()
            family_text = f"{source_value} {document_type}"
            family_terms = {
                "apo_4": ("apo_4", "apo4"),
                "apo_3": ("apo_3", "apo3"),
                "wahlpflicht": ("wahlpflicht", "elective"),
                "studienverlaufsplan": ("studienverlaufsplan", "study_plan"),
                "studiengangsbeschreibung": ("studiengangsbeschreibung", "program_description"),
            }[requested_family]
            if any(term in family_text for term in family_terms):
                routing_boost += 0.16
            if requested_family == "apo_4" and ("apo_3" in family_text or "apo3" in family_text):
                routing_boost -= 0.18
            if requested_family == "apo_3" and ("apo_4" in family_text or "apo4" in family_text):
                routing_boost -= 0.18

        boost = metadata_boost + routing_boost
        rerank_score = base_score + boost
        metadata["original_retrieval_score"] = original_score
        metadata["metadata_rerank_boost"] = round(metadata_boost, 4)
        metadata["document_routing_family"] = requested_family
        metadata["document_routing_boost"] = round(routing_boost, 4)
        metadata["lightweight_metadata_boost"] = round(boost, 4)
        metadata["lightweight_rerank_score"] = round(rerank_score, 4)
        doc.metadata = metadata
        doc.score = rerank_score

    return sorted(documents, key=lambda document: document.score or 0.0, reverse=True)


def lightweight_metadata_rerank(
    documents: list[RetrievedDocument], query: str
) -> list[RetrievedDocument]:
    """Backward-compatible name for the deterministic metadata reranker."""

    return metadata_rerank(documents, query)


def expand_neighbor_chunks(
    documents: list[RetrievedDocument],
    doc_store: Any,
    previous: int = 1,
    next_: int = 1,
    max_chars: int = 5000,
    allow_cross_section: bool = False,
    use_enriched_text: bool = False,
    skipped_reasons: list[dict[str, str]] | None = None,
) -> list[RetrievedDocument]:
    """Expand only within a known section/table and a hard character budget.

    ``allow_cross_section`` is retained only for API compatibility. It no longer
    permits crossing a known legal/profile/table boundary; blind expansion has a
    separate explicit pipeline switch.
    """

    if not documents or doc_store is None:
        return documents

    cache: dict[str, Document | None] = {doc.doc_id: doc for doc in documents}

    def fetch(doc_id: str | None) -> Document | None:
        if not doc_id:
            return None
        if doc_id not in cache:
            found = doc_store.get([doc_id])
            cache[doc_id] = found[0] if found else None
        return cache[doc_id]

    skipped_reasons = skipped_reasons if skipped_reasons is not None else []

    def compatible(seed: Document, candidate: Document) -> bool:
        seed_meta = seed.metadata or {}
        candidate_meta = candidate.metadata or {}
        if _source_name(seed_meta) != _source_name(candidate_meta):
            skipped_reasons.append(
                {"chunk_id": candidate.doc_id, "reason": "different_source"}
            )
            return False
        seed_section = str(seed_meta.get("section_id") or "")
        candidate_section = str(candidate_meta.get("section_id") or "")
        if not seed_section or not candidate_section:
            skipped_reasons.append(
                {"chunk_id": candidate.doc_id, "reason": "missing_section_id"}
            )
            return False
        seed_table = str(seed_meta.get("table_id") or "")
        candidate_table = str(candidate_meta.get("table_id") or "")
        if seed_table or candidate_table:
            compatible_table = bool(seed_table and seed_table == candidate_table)
            if not compatible_table:
                skipped_reasons.append(
                    {"chunk_id": candidate.doc_id, "reason": "different_table_id"}
                )
            return compatible_table
        if seed_section != candidate_section:
            skipped_reasons.append(
                {"chunk_id": candidate.doc_id, "reason": "different_section_id"}
            )
            return False
        return True

    expanded: list[RetrievedDocument] = []
    seen: set[str] = set()
    used_chars = 0
    max_chars = max(1, int(max_chars))

    # With conservative zero-neighbor defaults, do not reduce the vector top-k to
    # a 5,000-character subset. The normal QA context packer already applies its
    # token budget; this budget exists specifically for added expansion context.
    if max(0, int(previous)) == 0 and max(0, int(next_)) == 0:
        return use_contextual_text(documents) if use_enriched_text else documents

    for seed in documents:
        before: list[Document] = []
        cursor: Document = seed
        for _ in range(max(0, int(previous))):
            candidate = fetch((cursor.metadata or {}).get("previous_chunk_id"))
            if candidate is None or not compatible(seed, candidate):
                break
            before.append(candidate)
            cursor = candidate
        before.reverse()

        after: list[Document] = []
        cursor = seed
        for _ in range(max(0, int(next_))):
            candidate = fetch((cursor.metadata or {}).get("next_chunk_id"))
            if candidate is None or not compatible(seed, candidate):
                break
            after.append(candidate)
            cursor = candidate

        # Keep vector/rerank order authoritative: the retrieved seed is always
        # admitted before its optional local context.
        for candidate in [seed, *before, *after]:
            if candidate.doc_id in seen:
                continue
            remaining = max_chars - used_chars
            if remaining <= 0:
                skipped_reasons.append(
                    {"chunk_id": candidate.doc_id, "reason": "context_budget_exhausted"}
                )
                return expanded
            text = str(
                (candidate.metadata or {}).get("enriched_text")
                if use_enriched_text
                and (candidate.metadata or {}).get("enriched_text")
                else candidate.text or ""
            )
            if len(text) > remaining:
                text = text[:remaining]
            score = seed.score if candidate.doc_id == seed.doc_id else seed.score
            candidate_data = candidate.to_dict()
            candidate_data["text"] = text
            candidate_data["content"] = text
            candidate_data.pop("score", None)
            retrieved = RetrievedDocument(**candidate_data, score=score)
            retrieved.metadata["neighbor_expansion_role"] = (
                "retrieved" if candidate.doc_id == seed.doc_id else "neighbor"
            )
            retrieved.metadata["neighbor_of"] = seed.doc_id
            expanded.append(retrieved)
            seen.add(candidate.doc_id)
            used_chars += len(text)
            if len(text) >= remaining:
                return expanded
    return expanded


def expand_blind_neighbor_chunks(
    documents: list[RetrievedDocument],
    doc_store: Any,
    previous: int = 1,
    next_: int = 1,
    max_chars: int = 5000,
    use_enriched_text: bool = False,
) -> list[RetrievedDocument]:
    """Legacy same-source expansion, available only behind an explicit switch."""

    if not documents or doc_store is None:
        return documents
    cache: dict[str, Document | None] = {doc.doc_id: doc for doc in documents}

    def fetch(doc_id: str | None) -> Document | None:
        if not doc_id:
            return None
        if doc_id not in cache:
            found = doc_store.get([doc_id])
            cache[doc_id] = found[0] if found else None
        return cache[doc_id]

    result: list[RetrievedDocument] = []
    seen: set[str] = set()
    used = 0
    for seed in documents:
        neighbors: list[Document] = []
        for key, count in (
            ("previous_chunk_id", max(0, int(previous))),
            ("next_chunk_id", max(0, int(next_))),
        ):
            cursor: Document = seed
            for _ in range(count):
                candidate = fetch((cursor.metadata or {}).get(key))
                if candidate is None or _source_name(candidate.metadata or {}) != _source_name(seed.metadata or {}):
                    break
                neighbors.append(candidate)
                cursor = candidate
        for candidate in [seed, *neighbors]:
            if candidate.doc_id in seen:
                continue
            text = str(
                (candidate.metadata or {}).get("enriched_text")
                if use_enriched_text and (candidate.metadata or {}).get("enriched_text")
                else candidate.text or ""
            )
            remaining = max(0, int(max_chars) - used)
            if not remaining:
                return result
            text = text[:remaining]
            data = candidate.to_dict()
            data.update({"text": text, "content": text})
            data.pop("score", None)
            retrieved = RetrievedDocument(**data, score=seed.score)
            retrieved.metadata["neighbor_expansion_role"] = (
                "retrieved" if candidate.doc_id == seed.doc_id else "blind_neighbor"
            )
            result.append(retrieved)
            seen.add(candidate.doc_id)
            used += len(text)
    return result


def retrieval_debug_payload(
    query: str,
    before: Iterable[RetrievedDocument],
    after: Iterable[RetrievedDocument],
    used_enriched_text: bool,
    skipped_reasons: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    before = list(before)
    after = list(after)
    return {
        "query": query,
        "chunks_before_metadata_rerank": [
            {
                "chunk_id": doc.doc_id,
                "source_file": _source_name(doc.metadata or {}),
                "page": (doc.metadata or {}).get(
                    "page_label", (doc.metadata or {}).get("page_number")
                ),
                "section_id": (doc.metadata or {}).get("section_id"),
                "section_heading": (doc.metadata or {}).get("section_heading"),
                "table_id": (doc.metadata or {}).get("table_id"),
                "vector_score": doc.score,
            }
            for doc in before
        ],
        "final_chunks": [
            {
                "chunk_id": doc.doc_id,
                "source_file": _source_name(doc.metadata or {}),
                "page": (doc.metadata or {}).get(
                    "page_label", (doc.metadata or {}).get("page_number")
                ),
                "section_id": (doc.metadata or {}).get("section_id"),
                "section_heading": (doc.metadata or {}).get("section_heading"),
                "table_id": (doc.metadata or {}).get("table_id"),
                "original_score": (doc.metadata or {}).get(
                    "original_retrieval_score", doc.score
                ),
                "metadata_boost": (doc.metadata or {}).get("metadata_rerank_boost", 0),
                "routing_boost": (doc.metadata or {}).get("document_routing_boost", 0),
                "final_score": doc.score,
                "expansion_role": (doc.metadata or {}).get(
                    "neighbor_expansion_role", "retrieved"
                ),
                "original_text_chars": len(
                    str((doc.metadata or {}).get("original_text") or doc.text or "")
                ),
                "enriched_text_chars": len(
                    str((doc.metadata or {}).get("enriched_text") or "")
                ),
            }
            for doc in after
        ],
        "skipped_expansion": skipped_reasons or [],
        "text_mode": "enriched_text" if used_enriched_text else "original_text",
        "final_context_chars": sum(len(doc.text or "") for doc in after),
    }


def log_retrieval_debug(payload: dict[str, Any]) -> None:
    message = "Contextual retrieval debug: " + json.dumps(
        payload, ensure_ascii=False, default=str
    )
    logger.info(message)
    print(f"[contextual-retrieval] {message}", flush=True)
