import json

from kotaemon.base import Document, RetrievedDocument
from kotaemon.indices.vectorindex import VectorIndexing, VectorRetrieval
from ktem.index.file.ingestion_v2 import build_ingestion_v2, classify_document


def _page(page, elements, name="Studiengangsbeschreibung_BA_D3B.pdf"):
    return Document(
        text="\n".join(item[1] for item in elements),
        metadata={
            "type": "text",
            "page_label": page,
            "file_name": name,
            "docling_page_elements": json.dumps(
                [
                    {"id": f"p{page}-{index}", "label": label, "text": text}
                    for index, (label, text) in enumerate(elements)
                ]
            ),
        },
    )


def _table(name, heading, rows, page=1):
    structure = {
        "version": 1,
        "rows": [
            [{"text": value, "row": r, "column": c} for c, value in enumerate(row)]
            for r, row in enumerate(rows)
        ],
    }
    text = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return Document(
        text=text,
        metadata={
            "type": "table",
            "page_label": page,
            "file_name": name,
            "table_heading": heading,
            "section_heading": heading,
            "docling_table_structure": json.dumps(structure),
        },
    )


def test_program_description_preserves_all_source_elements():
    pages = [
        _page(
            1,
            [
                ("page_header", "Digital & Data-Driven Business Status v03"),
                ("section_header", "1 Der Studiengang in drei Sätzen"),
                ("text", "Der Studiengang verbindet Wirtschaft und Datenanalyse."),
            ],
        ),
        _page(
            2,
            [
                ("page_header", "Digital & Data-Driven Business Status v03"),
                ("section_header", "2 Zielgruppe"),
                ("text", "Das Angebot richtet sich an Studieninteressierte."),
            ],
        ),
        _page(
            3,
            [
                ("page_header", "Digital & Data-Driven Business Status v03"),
                ("section_header", "3 Grundsätzliche Ausrichtung"),
                ("text", "Die Regelstudienzeit beträgt sechs Semester."),
            ],
        ),
    ]
    result = build_ingestion_v2(pages, [], pages[0].metadata["file_name"])
    source = [doc for doc in result.records if doc.metadata["chunk_kind"] == "source"]
    joined = "\n".join(doc.text for doc in source)
    assert "Wirtschaft und Datenanalyse" in joined
    assert "Studieninteressierte" in joined
    assert "sechs Semester" in joined
    assert "Status v03" not in joined
    assert result.report["element_coverage"] == 1.0


def test_numbered_legal_list_is_not_an_amendment_in_consolidated_document():
    name = "a_APO_ab_WS_14_15.pdf"
    page = _page(
        1,
        [
            ("section_header", "§ 11 Bestehen der Prüfung"),
            ("text", "Die Frist verlängert sich, wenn"),
            ("list_item", "1. mehr als 15 ECTS gemäß § 23 erworben wurden,"),
            ("list_item", "2. diese angerechnet wurden."),
        ],
        name=name,
    )
    result = build_ingestion_v2([page], [], name)
    assert result.report["document_family"] == "legal_consolidated"
    assert all(doc.metadata["chunk_kind"] == "source" for doc in result.records)
    assert any("1. mehr als 15 ECTS" in doc.text for doc in result.records)


def test_catalog_header_typo_keeps_real_semester_values():
    name = "Wahlpflichtkatalog_BA_D3B.pdf"
    table = _table(
        name,
        "Supply Chain Management & Logistics",
        [
            [
                "Modulbezeichnung",
                "Prüfungsform",
                "ECTSAnzahl",
                "Smesterlage",
                "Zulassungsvoraussetzungen",
            ],
            ["Supply Chain Analytics", "Klausur", "5", "WS", "-"],
            ["Retail Operations", "Klausur", "5", "SS", "-"],
        ],
    )
    result = build_ingestion_v2([], [table], name)
    facts = [doc for doc in result.records if doc.metadata.get("fact_type") == "module_row"]
    assert [doc.metadata["semester"] for doc in facts] == ["WS", "SS"]
    assert result.report["rejected_tables"] == 0


def test_valid_study_plan_creates_verified_semester_records():
    name = "Studienverlaufsplan_BA_D3B.pdf"
    table = _table(
        name,
        "Studienprofil",
        [
            ["Semester", "ECTS", "Semester", "ECTS"],
            ["1", "10", "2", "10"],
            ["Mathematik", "5", "Statistik", "5"],
            ["Business English I", "5", "Business English II", "5"],
        ],
    )
    result = build_ingestion_v2([], [table], name)
    modules = [
        doc for doc in result.records if doc.metadata.get("fact_type") == "study_plan_module"
    ]
    semesters = [
        doc for doc in result.records if doc.metadata.get("fact_type") == "study_plan_semester"
    ]
    assert len(modules) == 4
    assert len(semesters) == 2
    assert result.report["rejected_tables"] == 0


def test_ambiguous_study_plan_is_kept_but_produces_no_facts():
    name = "Studienverlaufsplan_BA_D3B.pdf"
    table = _table(
        name,
        "Marketing",
        [
            ["Semester", "ECTS", "Semester", "ECTS"],
            ["1", "10", "2", "10"],
            ["Mathematik Business English I", "5 5", "Statistik", "5"],
            ["", "", "Business English II", "5"],
        ],
    )
    result = build_ingestion_v2([], [table], name)
    assert len(result.records) == 1
    assert result.records[0].metadata["chunk_kind"] == "source_table"
    assert result.records[0].metadata["table_validation_status"] == "rejected"
    assert result.report["rejected_tables"] == 1


def test_chunk_exports_do_not_overwrite_previous_embedding_batches(tmp_path):
    indexer = VectorIndexing(cache_dir=str(tmp_path))
    name = "document.pdf"
    indexer.prepare_chunk_export(name)
    first = [
        Document(text=f"chunk {index}", metadata={"file_name": name, "ingestion_index": index})
        for index in range(3)
    ]
    second = [
        Document(text=f"chunk {index}", metadata={"file_name": name, "ingestion_index": index})
        for index in range(3, 5)
    ]
    indexer.write_chunk_to_file(first)
    indexer.write_chunk_to_file(second)
    assert sorted(path.name for path in tmp_path.glob("document_*.md")) == [
        "document_0.md",
        "document_1.md",
        "document_2.md",
        "document_3.md",
        "document_4.md",
    ]


def test_document_family_classification_covers_dataset_families():
    assert classify_document("Modulkatalog_Bachelor_D3B_DE.pdf", "")[0] == "module_catalog"
    assert classify_document("APO_4.Aenderungssatzung.pdf", "")[0] == "legal_amendment"
    assert classify_document("Zeugnisantrag_D3B_DE.pdf", "")[0] == "form"


def test_retrieval_diversifies_children_from_the_same_table():
    docs = [
        RetrievedDocument(
            text=f"fact {index}",
            id_=f"fact-{index}",
            score=float(index),
            metadata={"table_parent_id": "table-a"},
        )
        for index in range(4)
    ]
    docs += [
        RetrievedDocument(
            text="another section",
            id_="section-b",
            score=5.0,
            metadata={"chunk_kind": "source"},
        )
    ]
    diversified = VectorRetrieval._diversify_parent_chunks(docs, top_k=3)
    assert [doc.doc_id for doc in diversified] == ["fact-0", "fact-1", "section-b"]
