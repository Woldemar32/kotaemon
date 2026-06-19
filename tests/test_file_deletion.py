from pathlib import Path

from sqlalchemy import JSON, Column, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session

import ktem.index.file.deletion as deletion


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "test_source"
    id = Column(String, primary_key=True)
    name = Column(String)
    path = Column(String)


class IndexRecord(Base):
    __tablename__ = "test_index"
    id = Column(String, primary_key=True)
    source_id = Column(String)
    target_id = Column(String)
    relation_type = Column(String)


class Group(Base):
    __tablename__ = "test_group"
    id = Column(String, primary_key=True)
    data = Column(JSON)


class MemoryStore:
    def __init__(self, ids=(), fail=False):
        self.ids = set(ids)
        self.fail = fail
        self.refresh_indices = None

    def delete(self, ids, refresh_indices=None):
        if self.fail:
            raise RuntimeError("store unavailable")
        self.refresh_indices = refresh_indices
        self.ids.difference_update(ids)


def _configure_test_data(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    storage = tmp_path / "files"
    storage.mkdir()
    caches = []
    for name in ("chunks", "markdown", "zip", "zip-in"):
        path = tmp_path / name
        path.mkdir()
        caches.append(path)

    with Session(engine) as session:
        session.add_all(
            [
                Source(id="source-1", name="deleted.pdf", path="hash-1"),
                Source(id="source-2", name="remaining.pdf", path="hash-2"),
                IndexRecord(
                    id="i1",
                    source_id="source-1",
                    target_id="doc-1",
                    relation_type="document",
                ),
                IndexRecord(
                    id="i2",
                    source_id="source-1",
                    target_id="doc-1",
                    relation_type="vector",
                ),
                IndexRecord(
                    id="i3",
                    source_id="source-2",
                    target_id="doc-2",
                    relation_type="document",
                ),
                Group(id="group", data={"files": ["source-1", "source-2"]}),
            ]
        )
        session.commit()

    (storage / "hash-1").write_text("deleted", encoding="utf-8")
    (storage / "hash-2").write_text("remaining", encoding="utf-8")
    (caches[0] / "deleted_0.md").write_text("chunk", encoding="utf-8")
    (caches[0] / "remaining_0.md").write_text("chunk", encoding="utf-8")
    (caches[2] / "all.zip").write_text("cached archive", encoding="utf-8")
    extracted = caches[3] / "deleted"
    extracted.mkdir()
    (extracted / "inside.txt").write_text("deleted", encoding="utf-8")
    return engine, storage, caches


def test_delete_removes_all_artifacts_without_rebuilding_fts(tmp_path):
    engine, storage, caches = _configure_test_data(tmp_path)
    old_engine = deletion.engine
    setting_names = (
        "KH_CHUNKS_OUTPUT_DIR",
        "KH_MARKDOWN_OUTPUT_DIR",
        "KH_ZIP_OUTPUT_DIR",
        "KH_ZIP_INPUT_DIR",
    )
    old_settings = {
        name: getattr(deletion.flowsettings, name, None) for name in setting_names
    }
    try:
        deletion.engine = engine
        for name, path in zip(setting_names, caches):
            setattr(deletion.flowsettings, name, path)
        vector_store = MemoryStore(["doc-1"])
        doc_store = MemoryStore(["doc-1", "doc-2"])

        result = deletion.delete_file_sources(
            source_model=Source,
            index_model=IndexRecord,
            group_model=Group,
            vector_store=vector_store,
            doc_store=doc_store,
            file_storage_path=storage,
            file_ids=["source-1"],
        )

        assert result.document_chunks == 1
        assert result.vector_chunks == 1
        assert vector_store.ids == set()
        assert doc_store.ids == {"doc-2"}
        assert doc_store.refresh_indices is False
        assert not (storage / "hash-1").exists()
        assert (storage / "hash-2").exists()
        assert not (caches[0] / "deleted_0.md").exists()
        assert (caches[0] / "remaining_0.md").exists()
        assert not (caches[2] / "all.zip").exists()
        assert not (caches[3] / "deleted").exists()
        with Session(engine) as session:
            assert session.get(Source, "source-1") is None
            assert session.get(Source, "source-2") is not None
            assert not list(
                session.scalars(
                    select(IndexRecord).where(IndexRecord.source_id == "source-1")
                )
            )
            assert session.get(Group, "group").data["files"] == ["source-2"]

        # Repeated UI events or retries are harmless.
        deletion.delete_file_sources(
            source_model=Source,
            index_model=IndexRecord,
            group_model=Group,
            vector_store=vector_store,
            doc_store=doc_store,
            file_storage_path=storage,
            file_ids=["source-1"],
        )
    finally:
        deletion.engine = old_engine
        for name, value in old_settings.items():
            setattr(deletion.flowsettings, name, value)


def test_store_failure_preserves_sql_mappings_for_retry(tmp_path):
    engine, storage, _ = _configure_test_data(tmp_path)
    old_engine = deletion.engine
    try:
        deletion.engine = engine
        try:
            deletion.delete_file_sources(
                source_model=Source,
                index_model=IndexRecord,
                vector_store=MemoryStore(["doc-1"], fail=True),
                doc_store=MemoryStore(["doc-1"]),
                file_storage_path=storage,
                file_ids=["source-1"],
            )
        except RuntimeError as exc:
            assert "store unavailable" in str(exc)
        else:
            raise AssertionError("Expected vector-store deletion failure")

        with Session(engine) as session:
            assert session.get(Source, "source-1") is not None
            assert list(
                session.scalars(
                    select(IndexRecord).where(IndexRecord.source_id == "source-1")
                )
            )
    finally:
        deletion.engine = old_engine
