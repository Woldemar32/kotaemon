"""Consistent, idempotent cleanup for file-index sources."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from theflow.settings import settings as flowsettings

from ktem.db.engine import engine

logger = logging.getLogger(__name__)


@dataclass
class DeletionResult:
    source_ids: list[str] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)
    vector_chunks: int = 0
    document_chunks: int = 0
    stored_files: int = 0
    cache_files: int = 0


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _remove_cached_files(source_names: list[str], live_source_names: set[str]) -> int:
    """Remove generated artifacts only when no live source shares the same stem."""

    live_stems = {Path(name).stem for name in live_source_names}
    deleted_stems = {
        Path(name).stem for name in source_names if Path(name).stem not in live_stems
    }
    if not deleted_stems:
        return 0

    directories = [
        getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None),
        getattr(flowsettings, "KH_MARKDOWN_OUTPUT_DIR", None),
        getattr(flowsettings, "KH_ZIP_OUTPUT_DIR", None),
        getattr(flowsettings, "KH_ZIP_INPUT_DIR", None),
    ]
    removed = 0
    for directory in directories:
        if not directory:
            continue
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in root.iterdir():
            matches_source = any(
                path.stem == stem
                or path.name.startswith(f"{stem}_")
                or path.name.startswith(f"{stem}.")
                for stem in deleted_stems
            )
            # The aggregate download archive can contain every deleted source and
            # must be invalidated whenever any source is removed.
            is_aggregate_archive = path.is_file() and path.name == "all.zip"
            if path.is_file() and (matches_source or is_aggregate_archive):
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove cached file %s: %s", path, exc)
            elif path.is_dir() and matches_source:
                try:
                    file_count = sum(1 for item in path.rglob("*") if item.is_file())
                    shutil.rmtree(path)
                    removed += file_count
                except OSError as exc:
                    logger.warning("Could not remove cache directory %s: %s", path, exc)
    return removed


def delete_file_sources(
    *,
    source_model: Any,
    index_model: Any,
    vector_store: Any,
    doc_store: Any,
    file_storage_path: str | Path,
    file_ids: Iterable[str],
    group_model: Any | None = None,
) -> DeletionResult:
    """Delete sources and every associated artifact.

    External stores are cleaned before SQL rows are committed. This makes a failed
    deletion safely retryable instead of losing the mapping needed to find orphaned
    chunks. Store deletion itself is idempotent.
    """

    requested_ids = _unique(file_ids)
    result = DeletionResult(source_ids=requested_ids)
    if not requested_ids:
        return result

    with Session(engine) as session:
        sources = list(
            session.scalars(select(source_model).where(source_model.id.in_(requested_ids)))
        )
        mappings = list(
            session.scalars(
                select(index_model).where(index_model.source_id.in_(requested_ids))
            )
        )

    source_ids = {str(source.id) for source in sources}
    # Include requested IDs so stale mapping rows can still be cleaned on a retry.
    cleanup_ids = set(requested_ids) | source_ids
    vector_ids = _unique(
        mapping.target_id
        for mapping in mappings
        if mapping.relation_type == "vector"
    )
    document_ids = _unique(
        mapping.target_id
        for mapping in mappings
        if mapping.relation_type == "document"
    )
    result.source_names = [str(source.name or "") for source in sources]
    result.vector_chunks = len(vector_ids)
    result.document_chunks = len(document_ids)

    # These calls happen in batches. In particular, LanceDB must not rebuild its
    # complete FTS index once per chunk or once per selected source.
    if vector_ids and vector_store is not None:
        vector_store.delete(vector_ids)
    if document_ids and doc_store is not None:
        try:
            doc_store.delete(document_ids, refresh_indices=False)
        except TypeError:
            # Backward compatibility for document stores without this optimization.
            doc_store.delete(document_ids)

    source_paths = [str(source.path or "") for source in sources if source.path]
    with Session(engine) as session:
        if group_model is not None:
            groups = list(session.scalars(select(group_model)))
            for group in groups:
                data = dict(group.data or {})
                files = list(data.get("files") or [])
                filtered = [file_id for file_id in files if str(file_id) not in cleanup_ids]
                if filtered != files:
                    data["files"] = filtered
                    group.data = data
                    session.add(group)

        session.execute(delete(index_model).where(index_model.source_id.in_(cleanup_ids)))
        session.execute(delete(source_model).where(source_model.id.in_(cleanup_ids)))
        session.commit()

    # Remove stored originals only when no remaining source references their hash.
    storage_root = Path(file_storage_path)
    with Session(engine) as session:
        remaining_paths = {
            str(value)
            for value in session.scalars(select(source_model.path))
            if value
        }
        live_source_names = {
            str(value)
            for value in session.scalars(select(source_model.name))
            if value
        }
    for source_path in source_paths:
        if source_path in remaining_paths:
            continue
        stored_path = storage_root / Path(source_path).name
        try:
            if stored_path.is_file():
                stored_path.unlink()
                result.stored_files += 1
        except OSError as exc:
            logger.warning("Could not remove stored source %s: %s", stored_path, exc)

    result.cache_files = _remove_cached_files(
        result.source_names, live_source_names
    )
    return result
