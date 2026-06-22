"""Evidence-preserving document ingestion for heterogeneous RAG corpora.

Version 2 intentionally does not call an LLM.  It turns Docling page elements into
hierarchical source chunks and creates table records only when deterministic
validation succeeds.  Uncertain content is retained as a searchable source chunk.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from kotaemon.base import Document


INGESTION_VERSION = "v2"
SCHEMA_VERSION = "ingestion-v2.1"

_LEGAL_SECTION = re.compile(r"^§{1,2}\s*\d+[a-zA-Z]?(?:\s+.*)?$")
_NUMBERED_HEADING = re.compile(
    r"^(\d+(?:\.\d+){0,4})[.)]\s+([A-ZÄÖÜ][^.!?]{2,160})$"
)
_AMENDMENT_ITEM = re.compile(
    r"^(\d{1,3})\.\s+(?:(?:In|Der|Die|Das|Nach|Vor)\s+)?"
    r"§\s*(\d+[a-zA-Z]?)\b.*$",
    re.IGNORECASE,
)
_PROFILE_HEADING = re.compile(
    r"^(?:Exemplarisches\s+)?(?:Studienprofil|Wahlpflichtbereich|"
    r"Kompetenzbereich|Schwerpunkt|Anlage)\b.*$",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9§])")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
_DOT_LEADER = re.compile(r"\.{4,}")

_MODULE_SUBHEADINGS = {
    "kompetenzen",
    "inhalte und themen",
    "formale voraussetzungen für die teilnahme",
    "empfohlene voraussetzungen für die teilnahme",
    "lehr- und prüfungssprache",
    "lehr- und lernformen/ lehrveranstaltungstypen",
    "voraussetzungen für die vergabe von ects-punkten",
    "zeitaufwand/ berechnung der ects-punkte innerhalb des moduls",
    "modulnote",
    "polyvalenz mit anderen studiengängen",
    "bemerkungen",
}


@dataclass
class SourceElement:
    element_id: str
    page: int
    order: int
    label: str
    text: str
    bbox: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    records: list[Document]
    report: dict[str, Any]


def _stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(part or "").strip() for part in parts)
    return sha256(raw.encode("utf-8")).hexdigest()[:24]


def _ascii_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    return re.sub(r"[^a-z0-9]+", "", normalized.encode("ascii", "ignore").decode())


def _clean_text(value: Any, *, single_line: bool = False) -> str:
    """Conservatively normalize whitespace without guessing word boundaries."""

    text = str(value or "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", " " if single_line else "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_cell(value: Any) -> str:
    text = _clean_text(value, single_line=True)
    # A hyphen followed by whitespace inside one Docling cell is a line-wrap.
    # Hyphens without whitespace are retained because they may be semantic.
    text = re.sub(
        r"([A-Za-zÄÖÜäöüß])-\s+([A-Za-zÄÖÜäöüß])", r"\1\2", text
    )
    return text


def classify_document(source_file: str, text: str) -> tuple[str, float]:
    """Classify only broad families needed for safe structural handling."""

    name = _ascii_key(Path(source_file).stem)
    sample = _ascii_key(text[:12000])
    if "modulkatalog" in name:
        return "module_catalog", 0.99
    if "studiengangsbeschreibung" in name:
        return "program_description", 0.99
    if "studienverlaufsplan" in name:
        return "study_plan", 0.99
    if "wahlpflichtkatalog" in name or name.startswith("module"):
        return "elective_catalog", 0.96
    if any(token in name for token in ("anmeldung", "zeugnisantrag", "umfrage")):
        return "form", 0.96
    if "flyer" in name:
        return "brochure", 0.98
    amendment_name = any(
        token in name for token in ("aenderungssatzung", "satzungzuraenderung")
    )
    if amendment_name and "konsolidiert" not in name:
        return "legal_amendment", 0.98
    legal_name = (
        "apo" in name
        or name.startswith("po")
        or "pruefungsordnung" in sample
        or "prufungsordnung" in sample
    )
    if legal_name:
        return "legal_consolidated", 0.94
    return "generic", 0.50


def _parse_page_elements(doc: Document, page: int) -> list[SourceElement]:
    metadata = doc.metadata or {}
    payload = metadata.get("docling_page_elements")
    parsed: list[dict[str, Any]] = []
    if payload:
        try:
            value = json.loads(payload) if isinstance(payload, str) else payload
            parsed = value if isinstance(value, list) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = []
    elements: list[SourceElement] = []
    if parsed:
        for order, item in enumerate(parsed):
            text = _clean_text(item.get("text"))
            if not text:
                continue
            elements.append(
                SourceElement(
                    element_id=str(item.get("id") or f"p{page}-e{order}"),
                    page=page,
                    order=order,
                    label=str(item.get("label") or "text"),
                    text=text,
                    bbox=item.get("bbox") if isinstance(item.get("bbox"), dict) else {},
                )
            )
        return elements

    # Compatibility fallback for readers without canonical element metadata.
    for order, line in enumerate((doc.text or "").splitlines()):
        text = _clean_text(line)
        if text:
            elements.append(
                SourceElement(
                    element_id=f"p{page}-l{order}",
                    page=page,
                    order=order,
                    label="text",
                    text=text,
                )
            )
    return elements


def _repeated_artifacts(elements: list[SourceElement]) -> set[str]:
    pages = {element.page for element in elements}
    if len(pages) < 3:
        return set()
    occurrences: dict[str, set[int]] = defaultdict(set)
    explicit: set[str] = set()
    by_page: dict[int, list[SourceElement]] = defaultdict(list)
    for element in elements:
        key = " ".join(element.text.lower().split())
        by_page[element.page].append(element)
        if element.label in {"page_header", "page_footer"}:
            explicit.add(key)
    for page, page_elements in by_page.items():
        candidates = page_elements[:2] + page_elements[-2:]
        for element in candidates:
            key = " ".join(element.text.lower().split())
            if 3 <= len(key) <= 120:
                occurrences[key].add(page)
    threshold = max(3, int(len(pages) * 0.45))
    repeated = {key for key, seen in occurrences.items() if len(seen) >= threshold}
    return explicit | repeated


def _is_heading(element: SourceElement, family: str) -> bool:
    text = element.text.strip().rstrip(":")
    if not text or len(text) > 220:
        return False
    if element.label == "title":
        return True
    if element.label == "section_header":
        if family in {"legal_consolidated", "legal_amendment", "module_catalog"}:
            return True
        if family != "program_description":
            return True
        return bool(
            _NUMBERED_HEADING.fullmatch(text)
            or _PROFILE_HEADING.fullmatch(text)
            or re.fullmatch(r"[A-ZÄÖÜ][.)]\s+.+", text)
            or text.lower().startswith("anlage ")
        )
    if family.startswith("legal") and _LEGAL_SECTION.fullmatch(text):
        return True
    if family == "legal_amendment" and _AMENDMENT_ITEM.fullmatch(text):
        return True
    if family in {"program_description", "module_catalog"}:
        if _NUMBERED_HEADING.fullmatch(text) or _PROFILE_HEADING.fullmatch(text):
            return True
    if _PROFILE_HEADING.fullmatch(text):
        return True
    if 3 <= len(text) <= 100 and text.isupper() and any(char.isalpha() for char in text):
        return True
    return False


def _heading_level(text: str, family: str) -> int:
    stripped = text.strip().rstrip(":")
    if family.startswith("legal"):
        return 2 if stripped.startswith("§") or _AMENDMENT_ITEM.match(stripped) else 1
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        base = 2 if family == "program_description" else 1
        return min(5, numbered.group(1).count(".") + base)
    if family == "module_catalog":
        return 2 if _ascii_key(stripped) in {_ascii_key(v) for v in _MODULE_SUBHEADINGS} else 1
    if _PROFILE_HEADING.match(stripped):
        return 2
    return 1


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidates = [paragraph]
        if len(paragraph) > max_chars:
            candidates = _SENTENCE_BOUNDARY.split(paragraph)
        for candidate in candidates:
            proposed = f"{current}\n\n{candidate}".strip() if current else candidate
            if current and len(proposed) > max_chars:
                pieces.append(current)
                current = candidate
            elif len(candidate) > max_chars:
                # Last-resort word boundary; never cut in the middle of a word.
                words = candidate.split()
                for word in words:
                    proposed_word = f"{current} {word}".strip()
                    if current and len(proposed_word) > max_chars:
                        pieces.append(current)
                        current = word
                    else:
                        current = proposed_word
            else:
                current = proposed
    if current:
        pieces.append(current)
    return pieces


def _deterministic_context(
    source_file: str, family: str, pages: tuple[int, int], path: str
) -> str:
    page_label = str(pages[0]) if pages[0] == pages[1] else f"{pages[0]}-{pages[1]}"
    parts = [f"Dokument: {source_file}", f"Dokumenttyp: {family}", f"Seite: {page_label}"]
    if path:
        parts.append(f"Abschnitt: {path}")
    return "\n".join(parts)


def _source_metadata(
    base: dict[str, Any],
    source_file: str,
    family: str,
    pages: tuple[int, int],
    path: str,
    kind: str,
    element_ids: Iterable[str],
) -> dict[str, Any]:
    metadata = dict(base)
    metadata.update(
        {
            "type": "text",
            "document_type": "text",
            "source_file": source_file,
            "document_family": family,
            "ingestion_version": INGESTION_VERSION,
            "ingestion_schema": SCHEMA_VERSION,
            "chunk_kind": kind,
            "retrieval_role": "primary",
            "page_label": pages[0],
            "end_page_label": pages[1],
            "section_path": path,
            "section_heading": path.split(" > ")[-1] if path else "",
            "source_element_ids": json.dumps(list(element_ids), ensure_ascii=False),
            "context_enrichment_mode": "deterministic-v2",
        }
    )
    metadata.pop("docling_page_elements", None)
    return metadata


def _build_text_chunks(
    text_docs: list[Document], source_file: str, family: str, max_chars: int
) -> tuple[list[Document], dict[str, Any]]:
    base = dict(text_docs[0].metadata or {}) if text_docs else {}
    elements: list[SourceElement] = []
    for doc in sorted(text_docs, key=lambda value: int((value.metadata or {}).get("page_label", 0))):
        page = int((doc.metadata or {}).get("page_label", 1) or 1)
        elements.extend(_parse_page_elements(doc, page))

    artifacts = _repeated_artifacts(elements)
    retained = [
        element
        for element in elements
        if " ".join(element.text.lower().split()) not in artifacts
        and not (_PAGE_NUMBER.fullmatch(element.text) and element.label in {"page_footer", "text"})
    ]
    has_toc = any(_DOT_LEADER.search(element.text) for element in retained)
    if has_toc:
        retained = [
            element
            for element in retained
            if not _DOT_LEADER.search(element.text)
            and _ascii_key(element.text) not in {"inhalt", "inhaltsverzeichnis"}
        ]

    # A legal amendment boundary is meaningful only for a document already
    # classified as an amendment. Ordinary numbered lists are never promoted.
    if family == "legal_amendment":
        for element in retained:
            if _AMENDMENT_ITEM.match(element.text):
                element.label = "section_header"

    groups: list[tuple[list[str], list[SourceElement]]] = []
    heading_stack: list[str] = []
    current: list[SourceElement] = []
    current_path: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append((list(current_path), current))
            current = []

    for element in retained:
        if _is_heading(element, family):
            if (
                family.startswith("legal")
                and len(current) == 1
                and re.fullmatch(r"§{1,2}\s*\d+[a-zA-Z]?", current[0].text.strip())
                and not element.text.strip().startswith("§")
                and element.label == "section_header"
            ):
                combined = f"{current[0].text.strip()} {element.text.strip()}"
                heading_stack[-1] = combined
                current_path = list(heading_stack)
                current.append(element)
                continue
            flush()
            heading = element.text.strip().rstrip(":")
            level = _heading_level(heading, family)
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(heading)
            current_path = list(heading_stack)
            current.append(element)
        else:
            current.append(element)
    flush()

    chunks: list[Document] = []
    mapped_ids: set[str] = set()
    for path_parts, group in groups:
        text = "\n\n".join(element.text for element in group).strip()
        if not text:
            continue
        # Structural parent headings remain in section_path but do not become
        # low-value standalone vectors when they have no body of their own.
        if all(_is_heading(element, family) for element in group):
            mapped_ids.update(element.element_id for element in group)
            continue
        path = " > ".join(path_parts)
        page_span = (min(element.page for element in group), max(element.page for element in group))
        ids = [element.element_id for element in group]
        mapped_ids.update(ids)
        for split_index, piece in enumerate(_split_long_text(text, max_chars)):
            metadata = _source_metadata(
                base, source_file, family, page_span, path, "source", ids
            )
            context = _deterministic_context(source_file, family, page_span, path)
            visible_text = piece
            if path and len(path_parts) > 1:
                visible_text = f"Abschnitt: {path}\n\n{piece}"
            metadata.update(
                {
                    "split_index": split_index,
                    "original_text": piece,
                    "rule_based_context": context,
                    "enriched_text": f"{context}\n\nOriginaltext:\n{visible_text}",
                }
            )
            chunks.append(
                Document(
                    id_=f"v2-{_stable_id(source_file, path, page_span, split_index, piece)}",
                    text=visible_text,
                    metadata=metadata,
                )
            )

    report = {
        "source_elements": len(elements),
        "retained_elements": len(retained),
        "removed_page_artifacts": len(elements) - len(retained),
        "mapped_elements": len(mapped_ids),
        "element_coverage": round(len(mapped_ids) / max(1, len(retained)), 4),
        "source_text_chunks": len(chunks),
    }
    return chunks, report


def _table_grid(metadata: dict[str, Any], text: str) -> list[list[str]]:
    structure = metadata.get("docling_table_structure")
    if structure:
        try:
            payload = json.loads(structure) if isinstance(structure, str) else structure
            rows = [
                [_clean_cell(cell.get("text", "")) for cell in row]
                for row in payload.get("rows", [])
            ]
            return [row for row in rows if any(row)]
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            pass
    rows: list[list[str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if cells and not all(re.fullmatch(r"[-: ]*", cell) for cell in cells):
            rows.append(cells)
    return rows


def _header_position(headers: list[str], aliases: Iterable[str]) -> int | None:
    alias_keys = [_ascii_key(alias) for alias in aliases]
    for index, header in enumerate(headers):
        key = _ascii_key(header)
        if any(alias in key or key in alias for alias in alias_keys if alias and key):
            return index
        if any(SequenceMatcher(None, key, alias).ratio() >= 0.84 for alias in alias_keys):
            return index
    return None


def _classify_table(grid: list[list[str]]) -> tuple[str, float]:
    if len(grid) < 2:
        return "unknown", 0.0
    headers = grid[0]
    module = _header_position(headers, ("Modulbezeichnung", "Modultitel", "Module"))
    signals = sum(
        position is not None
        for position in (
            _header_position(headers, ("Prüfungsform", "Assessment")),
            _header_position(headers, ("ECTS", "ECTS-Anzahl", "Leistungspunkte")),
            _header_position(headers, ("Semesterlage", "Smesterlage", "Offering")),
            _header_position(headers, ("Zulassungsvoraussetzungen", "Prerequisites")),
        )
    )
    if module is not None and signals >= 2:
        return "elective_catalog", min(0.99, 0.75 + signals * 0.05)
    normalized = [_ascii_key(header) for header in headers]
    if sum("semester" in value for value in normalized) >= 2 and sum(
        "ects" in value for value in normalized
    ) >= 2:
        semester_row = grid[1]
        if sum(bool(re.fullmatch(r"\d{1,2}", value)) for value in semester_row[::2]) >= 2:
            return "study_plan", 0.98
    return "unknown", 0.0


def _is_toc_table(grid: list[list[str]]) -> bool:
    if len(grid) < 5:
        return False
    flattened = [cell for row in grid for cell in row if cell]
    dotted = sum(bool(_DOT_LEADER.search(cell)) for cell in flattened)
    page_like = sum(bool(re.fullmatch(r".*\.{4,}\s*\d{1,4}", cell)) for cell in flattened)
    return dotted >= 4 or page_like >= 4


def _valid_module_name(value: str) -> bool:
    value = value.strip()
    return bool(
        len(value) >= 3
        and not value.endswith("-")
        and not value[:1].islower()
        and not re.search(r"\b\d+(?:[.,]\d+)?\s+\d+(?:[.,]\d+)?\b", value)
    )


def _study_plan_records(grid: list[list[str]]) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if len(grid) < 3:
        return [], ["table has fewer than three rows"]
    semester_row = grid[1]
    pairs: list[tuple[int, int, str, float]] = []
    for column in range(0, min(len(grid[0]), len(semester_row)) - 1, 2):
        semester = semester_row[column].strip()
        declared = semester_row[column + 1].strip()
        if not re.fullmatch(r"\d{1,2}", semester) or not _NUMBER.fullmatch(declared):
            errors.append(f"invalid semester header at column {column}")
            continue
        pairs.append((column, column + 1, semester, float(declared.replace(",", "."))))
    if not pairs:
        return [], errors or ["no semester columns found"]

    records: list[dict[str, str]] = []
    for module_col, ects_col, semester, declared_total in pairs:
        semester_records: list[dict[str, str]] = []
        pending = ""
        for row_index, row in enumerate(grid[2:], start=2):
            module = row[module_col].strip() if module_col < len(row) else ""
            ects = row[ects_col].strip() if ects_col < len(row) else ""
            if module and not ects:
                if semester_records and semester_records[-1]["module"].rstrip().endswith(
                    ("-", "&", "/", ",")
                ):
                    semester_records[-1]["module"] = _clean_cell(
                        f'{semester_records[-1]["module"]} {module}'
                    )
                else:
                    pending = f"{pending} {module}".strip()
                continue
            if not module and not ects:
                continue
            if not module or not _NUMBER.fullmatch(ects):
                errors.append(
                    f"semester {semester}, row {row_index}: ambiguous module/ECTS cell"
                )
                continue
            module = _clean_cell(f"{pending} {module}" if pending else module)
            pending = ""
            if not _valid_module_name(module):
                errors.append(
                    f"semester {semester}, row {row_index}: suspicious module name {module!r}"
                )
            record = {
                "module": module,
                "ects": ects.replace(",", "."),
                "semester": semester,
                "source_row": str(row_index),
                "source_column": str(module_col),
            }
            if not (
                semester_records
                and semester_records[-1]["module"] == record["module"]
                and semester_records[-1]["ects"] == record["ects"]
            ):
                semester_records.append(record)
        if pending:
            errors.append(f"semester {semester}: unmatched trailing cell {pending!r}")
        total = sum(float(record["ects"]) for record in semester_records)
        if abs(total - declared_total) > 0.01:
            errors.append(
                f"semester {semester}: parsed {total:g} ECTS, declared {declared_total:g}"
            )
        records.extend(semester_records)
    return (records, errors) if not errors else ([], errors)


def _catalog_records(grid: list[list[str]]) -> tuple[list[dict[str, str]], list[str]]:
    headers = grid[0] if grid else []
    positions = {
        "module": _header_position(headers, ("Modulbezeichnung", "Modultitel", "Module")),
        "exam_form": _header_position(headers, ("Prüfungsform", "Assessment")),
        "ects": _header_position(headers, ("ECTS", "ECTS-Anzahl", "Leistungspunkte")),
        "semester": _header_position(headers, ("Semesterlage", "Smesterlage", "Offering")),
        "prerequisites": _header_position(
            headers, ("Zulassungsvoraussetzungen", "Voraussetzungen", "Prerequisites")
        ),
    }
    required = ("module", "ects")
    if any(positions[key] is None for key in required):
        return [], ["required module or ECTS header is missing"]
    output: list[dict[str, str]] = []
    errors: list[str] = []
    for row_index, row in enumerate(grid[1:], start=1):
        values = {
            key: (
                row[position].strip()
                if position is not None and position < len(row)
                else ""
            )
            for key, position in positions.items()
        }
        if not any(values.values()):
            continue
        if not _valid_module_name(values["module"]):
            errors.append(f"row {row_index}: invalid module name {values['module']!r}")
        if not _NUMBER.fullmatch(values["ects"]):
            errors.append(f"row {row_index}: invalid ECTS value {values['ects']!r}")
        values["source_row"] = str(row_index)
        output.append(values)
    return (output, errors) if not errors else ([], errors)


def _table_source(
    table: Document,
    source_file: str,
    family: str,
    table_index: int,
    semantic_type: str,
    confidence: float,
    errors: list[str],
) -> Document:
    metadata = dict(table.metadata or {})
    page = int(metadata.get("page_label", 1) or 1)
    heading = str(metadata.get("table_heading") or metadata.get("section_heading") or "")
    table_id = f"table-{_stable_id(source_file, page, table_index, table.text)}"
    context = _deterministic_context(source_file, family, (page, page), heading)
    metadata.update(
        {
            "type": "table",
            "document_type": "table",
            "source_file": source_file,
            "document_family": family,
            "ingestion_version": INGESTION_VERSION,
            "ingestion_schema": SCHEMA_VERSION,
            "chunk_kind": "source_table",
            "retrieval_role": "primary",
            "table_id": table_id,
            "semantic_table_type": semantic_type,
            "table_classification_confidence": confidence,
            "table_validation_status": "valid" if not errors else "rejected",
            "table_validation_errors": json.dumps(errors, ensure_ascii=False),
            "original_text": table.text,
            "rule_based_context": context,
            "enriched_text": f"{context}\n\nOriginaltabelle:\n{table.text}",
        }
    )
    visible_text = table.text
    if heading and heading.lower() not in (table.text or "")[:300].lower():
        visible_text = f"Tabellenkontext: {heading}\n\n{table.text}"
    return Document(id_=table_id, text=visible_text, metadata=metadata)


def _fact_document(
    parent: Document,
    fact_type: str,
    fact_index: str,
    fact_text: str,
    fields: dict[str, str],
) -> Document:
    parent_meta = parent.metadata or {}
    metadata = {
        key: value
        for key, value in parent_meta.items()
        if key not in {"docling_table_structure", "table_origin", "enriched_text"}
    }
    metadata.update(
        {
            "type": "text",
            "document_type": "structured_record",
            "chunk_kind": "structured_record",
            "retrieval_role": "supporting",
            "fact_type": fact_type,
            "fact_schema_version": SCHEMA_VERSION,
            "table_parent_id": parent.doc_id,
            "fact_index": fact_index,
            "original_text": fact_text,
            "source_evidence": json.dumps(fields, ensure_ascii=False),
            **{key: value for key, value in fields.items() if isinstance(value, str)},
        }
    )
    context = str(parent_meta.get("rule_based_context") or "")
    metadata["enriched_text"] = f"{context}\n\nVerifizierter Fakt:\n{fact_text}"
    return Document(
        id_=f"fact-{_stable_id(parent.doc_id, fact_type, fact_index, fact_text)}",
        text=fact_text,
        metadata=metadata,
    )


def _process_tables(
    table_docs: list[Document], source_file: str, family: str
) -> tuple[list[Document], dict[str, Any]]:
    output: list[Document] = []
    rejected = 0
    facts = 0
    types: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []
    for table_index, table in enumerate(table_docs):
        grid = _table_grid(table.metadata or {}, table.text or "")
        if _is_toc_table(grid):
            types["table_of_contents"] += 1
            continue
        table_type, confidence = _classify_table(grid)
        records: list[dict[str, str]] = []
        errors: list[str] = []
        if table_type == "study_plan":
            records, errors = _study_plan_records(grid)
        elif table_type == "elective_catalog":
            records, errors = _catalog_records(grid)
        source = _table_source(
            table, source_file, family, table_index, table_type, confidence, errors
        )
        output.append(source)
        types[table_type] += 1
        if errors:
            rejected += 1
            warnings.append(
                {
                    "table_id": source.doc_id,
                    "page": (source.metadata or {}).get("page_label"),
                    "errors": errors,
                }
            )
            continue
        if table_type == "study_plan":
            per_semester: dict[str, list[dict[str, str]]] = defaultdict(list)
            for index, record in enumerate(records):
                text = (
                    f'Dokument: "{source_file}"; Abschnitt: '
                    f'{(source.metadata or {}).get("table_heading", "")}; '
                    f'Semester: {record["semester"]}; Modul: "{record["module"]}"; '
                    f'ECTS: {record["ects"]}.'
                )
                output.append(
                    _fact_document(source, "study_plan_module", str(index), text, record)
                )
                facts += 1
                per_semester[record["semester"]].append(record)
            for semester, semester_records in per_semester.items():
                modules = "; ".join(record["module"] for record in semester_records)
                total = sum(float(record["ects"]) for record in semester_records)
                fields = {
                    "semester": semester,
                    "ects": f"{total:g}",
                    "module_count": str(len(semester_records)),
                }
                text = (
                    f'Dokument: "{source_file}"; Abschnitt: '
                    f'{(source.metadata or {}).get("table_heading", "")}; '
                    f"Semester {semester} umfasst {total:g} ECTS; Module: {modules}."
                )
                output.append(
                    _fact_document(
                        source, "study_plan_semester", f"semester-{semester}", text, fields
                    )
                )
                facts += 1
        elif table_type == "elective_catalog":
            for index, record in enumerate(records):
                semester = record.get("semester") or "nicht angegeben"
                prerequisites = record.get("prerequisites") or "nicht angegeben"
                text = (
                    f'Dokument: "{source_file}"; Abschnitt: '
                    f'{(source.metadata or {}).get("table_heading", "")}; '
                    f'Modul: "{record["module"]}"; Prüfungsform: '
                    f'"{record.get("exam_form") or "nicht angegeben"}"; '
                    f'ECTS: {record["ects"]}; Semester: {semester}; '
                    f'Zulassungsvoraussetzungen: "{prerequisites}".'
                )
                output.append(_fact_document(source, "module_row", str(index), text, record))
                facts += 1
            if records:
                modules = "; ".join(record["module"] for record in records)
                text = (
                    f'Dokument: "{source_file}"; Abschnitt: '
                    f'{(source.metadata or {}).get("table_heading", "")}; Module: {modules}.'
                )
                output.append(
                    _fact_document(
                        source,
                        "catalog_section",
                        "aggregate",
                        text,
                        {"module_count": str(len(records))},
                    )
                )
                facts += 1
    return output, {
        "source_tables": len(table_docs),
        "table_types": dict(types),
        "rejected_tables": rejected,
        "structured_records": facts,
        "table_warnings": warnings,
    }


def build_ingestion_v2(
    text_docs: list[Document],
    non_text_docs: list[Document],
    source_file: str,
    *,
    max_chunk_chars: int = 4200,
) -> IngestionResult:
    """Create source chunks and strictly validated structured records."""

    all_text = "\n".join(doc.text or "" for doc in text_docs)
    family, confidence = classify_document(source_file, all_text)
    text_chunks, text_report = _build_text_chunks(
        text_docs, source_file, family, max(800, max_chunk_chars)
    )
    tables = [doc for doc in non_text_docs if (doc.metadata or {}).get("type") == "table"]
    other = [doc for doc in non_text_docs if (doc.metadata or {}).get("type") != "table"]
    table_records, table_report = _process_tables(tables, source_file, family)
    for item in other:
        metadata = dict(item.metadata or {})
        metadata.update(
            {
                "document_family": family,
                "ingestion_version": INGESTION_VERSION,
                "ingestion_schema": SCHEMA_VERSION,
                "chunk_kind": "source_media",
                "retrieval_role": "primary",
            }
        )
        item.metadata = metadata

    records = text_chunks + table_records + other
    for index, record in enumerate(records):
        metadata = dict(record.metadata or {})
        metadata["ingestion_index"] = index
        record.metadata = metadata

    report = {
        "ingestion_version": INGESTION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "document_family": family,
        "classification_confidence": confidence,
        **text_report,
        **table_report,
        "media_records": len(other),
        "total_records": len(records),
    }
    return IngestionResult(records=records, report=report)
