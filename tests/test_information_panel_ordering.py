import pytest

pytest.importorskip("theflow")

from kotaemon.base import RetrievedDocument
from kotaemon.indices.qa.citation_qa import AnswerWithContextPipeline


class _Answer:
    metadata = {"citation": None}


def test_information_panel_preserves_vector_order_when_llm_scores_are_zero():
    docs = [
        RetrievedDocument(
            text="top vector hit",
            id_="top",
            score=0.91,
            metadata={"file_name": "top.pdf", "llm_trulens_score": 0.0},
        ),
        RetrievedDocument(
            text="second vector hit",
            id_="second",
            score=0.72,
            metadata={"file_name": "second.pdf", "llm_trulens_score": 0.0},
        ),
        RetrievedDocument(
            text="third vector hit",
            id_="third",
            score=0.11,
            metadata={"file_name": "third.pdf", "llm_trulens_score": 0.0},
        ),
    ]

    pipeline = AnswerWithContextPipeline.__new__(AnswerWithContextPipeline)
    with_citation, without_citation = pipeline.prepare_citations(_Answer(), docs)

    assert with_citation == []
    rendered = [doc.content for doc in without_citation]
    assert "top.pdf" in rendered[0]
    assert "second.pdf" in rendered[1]
    assert "third.pdf" in rendered[2]
    assert "[score: 0.91]" in rendered[0]
