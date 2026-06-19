from types import SimpleNamespace

from kotaemon.base import Document, DocumentWithEmbedding, RetrievedDocument
from kotaemon.indices.vectorindex import VectorIndexing
from ktem.index.file.contextual import (
    enrich_chunks,
    expand_neighbor_chunks,
    lightweight_metadata_rerank,
)


SAMPLE_TABLE = """| Modulbezeichnung | Prüfungsform | ECTS-Anzahl | Semesterlage | Zulassungsvoraussetzungen |
| --- | --- | --- | --- | --- |
| Digital Seminar in Data Science & Quantitative Applications | Projektarbeit und Präsentation | 10 | WiSe | - |
| Retail Management Fundamentals | Klausur | 5 | SoSe | - |"""


class MemoryDocStore:
    def __init__(self, documents):
        self.documents = {document.doc_id: document for document in documents}

    def get(self, ids):
        return [self.documents[doc_id] for doc_id in ids if doc_id in self.documents]


def _sample_chunks():
    source = "Wahlpflichtkatalalog_BA_D3B_SoSe_2026.pdf"
    heading = Document(
        id_="heading",
        text=(
            "# Wahlpflichtkatalog BA D3B SoSe 2026\n\n"
            "## Studienprofil: Supply Chain Management and Logistics"
        ),
        metadata={"file_name": source, "page_label": 7, "type": "text"},
    )
    table = Document(
        id_="table",
        text=SAMPLE_TABLE,
        metadata={"file_name": source, "page_label": 7, "type": "table"},
    )
    following = Document(
        id_="following",
        text="Weitere Hinweise zum Studienprofil.",
        metadata={"file_name": source, "page_label": 7, "type": "text"},
    )
    return enrich_chunks(
        [heading, table, following], source, enable_table_row_facts=True
    )


def test_table_chunk_receives_heading_enrichment_and_row_facts():
    chunks = _sample_chunks()
    table = next(chunk for chunk in chunks if chunk.doc_id == "table")

    assert (
        table.metadata["parent_section_heading"]
        == "Studienprofil: Supply Chain Management and Logistics"
    )
    assert "Supply Chain Management and Logistics" in table.metadata["enriched_text"]
    facts = table.metadata["table_row_facts"]
    assert "Digital Seminar in Data Science & Quantitative Applications" in facts
    assert "Projektarbeit und Präsentation" in facts
    assert "10 ECTS" in facts
    assert "WiSe" in facts
    assert "keine Zulassungsvoraussetzungen" in facts
    assert table.metadata["original_text"] == SAMPLE_TABLE
    row_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("document_type") == "table_row_fact"
    ]
    assert len(row_chunks) == 2
    assert row_chunks[0].metadata["table_id"] == table.metadata["table_id"]
    assert row_chunks[0].metadata["section_id"] == table.metadata["section_id"]


def test_neighbor_expansion_is_ordered_and_does_not_cross_sources():
    chunks = [
        chunk
        for chunk in _sample_chunks()
        if chunk.metadata.get("document_type") != "table_row_fact"
    ]
    heading, table, following = chunks
    foreign = Document(
        id_="foreign",
        text="Unrelated document content",
        metadata={
            "file_name": "another-document.pdf",
            "source_file": "another-document.pdf",
            "parent_section_heading": table.metadata["parent_section_heading"],
        },
    )
    following.metadata["next_chunk_id"] = foreign.doc_id
    store = MemoryDocStore([*chunks, foreign])
    retrieved_table = RetrievedDocument(table, score=0.9)

    expanded = expand_neighbor_chunks(
        [retrieved_table],
        store,
        previous=1,
        next_=2,
        max_chars=20_000,
        use_enriched_text=True,
    )

    assert [document.doc_id for document in expanded] == ["table"]
    assert all(document.doc_id != "foreign" for document in expanded)
    assert "Document:" in expanded[0].text


def test_neighbor_expansion_stops_at_section_boundary():
    chunks = [
        chunk
        for chunk in _sample_chunks()
        if chunk.metadata.get("document_type") != "table_row_fact"
    ]
    heading, table, following = chunks
    following.metadata["section_id"] = "section-data-competence"
    store = MemoryDocStore(chunks)

    expanded = expand_neighbor_chunks(
        [RetrievedDocument(table, score=0.9)],
        store,
        previous=1,
        next_=1,
        max_chars=20_000,
    )

    assert [document.doc_id for document in expanded] == ["table"]


def test_neighbor_expansion_deduplicates_and_respects_character_budget():
    chunks = [
        chunk
        for chunk in _sample_chunks()
        if chunk.metadata.get("document_type") != "table_row_fact"
    ]
    store = MemoryDocStore(chunks)
    seeds = [
        RetrievedDocument(chunks[1], score=0.9),
        RetrievedDocument(chunks[2], score=0.8),
    ]

    expanded = expand_neighbor_chunks(
        seeds, store, previous=1, next_=1, max_chars=120
    )

    assert len({document.doc_id for document in expanded}) == len(expanded)
    assert sum(len(document.text or "") for document in expanded) <= 120


def test_lightweight_rerank_boosts_matching_section():
    matching = RetrievedDocument(
        id_="matching",
        text="Digital Seminar with 10 ECTS",
        metadata={
            "file_name": "catalog.pdf",
            "parent_section_heading": "Supply Chain Management and Logistics",
            "original_text": "Digital Seminar with 10 ECTS",
        },
        score=0.5,
    )
    unrelated = RetrievedDocument(
        id_="unrelated",
        text="A module in another profile",
        metadata={
            "file_name": "catalog.pdf",
            "parent_section_heading": "Data Competence",
            "original_text": "A module in another profile",
        },
        score=0.5,
    )

    reranked = lightweight_metadata_rerank(
        [unrelated, matching], "Supply Chain Management Digital Seminar"
    )

    assert reranked[0].doc_id == "matching"
    assert reranked[0].metadata["lightweight_metadata_boost"] > 0


def test_enriched_text_is_used_only_for_embedding_when_enabled():
    class RecordingEmbedding:
        def __init__(self):
            self.texts = []

        def run(self, documents):
            self.texts = [document.text for document in documents]
            return [
                DocumentWithEmbedding(embedding=[0.1, 0.2], content=document)
                for document in documents
            ]

    class RecordingVectorStore:
        def add(self, embeddings, ids):
            self.ids = ids

    document = Document(
        id_="chunk",
        text="original",
        metadata={"enriched_text": "Document: program.pdf\n\nenriched"},
    )
    embedding = RecordingEmbedding()
    indexing = SimpleNamespace(
        vector_store=RecordingVectorStore(),
        embedding=embedding,
        use_enriched_text_for_embedding=True,
    )

    VectorIndexing.add_to_vectorstore(indexing, [document])

    assert embedding.texts == ["Document: program.pdf\n\nenriched"]
    assert document.text == "original"


def test_apo_sections_are_hard_boundaries():
    source = "APO_4.pdf"
    document = Document(
        id_="apo",
        text=(
            "APO\n\n"
            "§ 7 Akteneinsicht\n\n"
            "Der Antrag auf Akteneinsicht ist binnen eines Monats zu stellen.\n\n"
            "§ 12 Bachelorarbeit\n\n"
            "Die Bachelorarbeit umfasst 10 ECTS-Punkte."
        ),
        metadata={"file_name": source, "page_label": 3, "type": "text"},
    )
    chunks = enrich_chunks([document], source)
    section_7 = next(chunk for chunk in chunks if "§ 7 Akteneinsicht" in chunk.text)
    section_12 = next(chunk for chunk in chunks if "§ 12 Bachelorarbeit" in chunk.text)

    assert section_7.metadata["section_id"] != section_12.metadata["section_id"]
    assert section_7.metadata["section_heading"] == "§ 7 Akteneinsicht"
    assert section_12.metadata["section_heading"] == "§ 12 Bachelorarbeit"

    store = MemoryDocStore(chunks)
    expanded_7 = expand_neighbor_chunks(
        [RetrievedDocument(section_7, score=0.9)], store, previous=1, next_=1
    )
    expanded_12 = expand_neighbor_chunks(
        [RetrievedDocument(section_12, score=0.9)], store, previous=1, next_=1
    )
    assert all("§ 12" not in chunk.text for chunk in expanded_7)
    assert all("§ 7" not in chunk.text for chunk in expanded_12)


def test_wahlpflicht_profiles_and_row_facts_do_not_mix():
    source = "Wahlpflichtkatalog_BA_D3B_SoSe_2026.pdf"
    document = Document(
        id_="catalog",
        text=(
            "Wahlpflichtkatalog BA D3B SoSe 2026\n\n"
            "Studienprofil: Supply Chain Management and Logistics\n\n"
            "Modulbezeichnung\tPrüfungsform\tECTS-Anzahl\tSemesterlage\tZulassungsvoraussetzungen\n"
            "Digital Seminar in Data Science & Quantitative Applications\tProjektarbeit und Präsentation\t10\tWiSe\t-\n"
            "Retail Management Fundamentals\tKlausur\t5\tSoSe\t-\n\n"
            "Studienprofil: Marketing, Organization, Innovation\n\n"
            "Modulbezeichnung\tPrüfungsform\tECTS-Anzahl\tSemesterlage\tZulassungsvoraussetzungen\n"
            "Global Marketing Management\tKlausur\t5\tSoSe\t-"
        ),
        metadata={"file_name": source, "page_label": 7, "type": "table"},
    )
    chunks = enrich_chunks([document], source, enable_table_row_fact_chunks=True)
    supply = next(chunk for chunk in chunks if "Digital Seminar" in chunk.text)
    marketing = next(chunk for chunk in chunks if "Global Marketing" in chunk.text)
    assert supply.metadata["section_id"] != marketing.metadata["section_id"]
    assert "Supply Chain" in supply.metadata["section_heading"]
    assert "Marketing, Organization, Innovation" in marketing.metadata["section_heading"]

    supply_facts = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("document_type") == "table_row_fact"
        and "Digital Seminar" in chunk.text
    ]
    assert len(supply_facts) == 1
    fact = supply_facts[0]
    assert fact.metadata["exam_form"] == "Projektarbeit und Präsentation"
    assert fact.metadata["ects"] == "10"
    assert fact.metadata["semester"] == "WiSe"
    assert fact.metadata["prerequisites"] == "-"
    assert fact.metadata["source_file"] == source
    assert fact.metadata["page_number"] == 7
    assert "Supply Chain" in fact.metadata["section_heading"]

    store = MemoryDocStore(chunks)
    expanded = expand_neighbor_chunks(
        [RetrievedDocument(supply, score=0.9)], store, previous=1, next_=1
    )
    assert all("Global Marketing" not in chunk.text for chunk in expanded)
