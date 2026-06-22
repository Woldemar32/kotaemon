import html
from functools import partial

import tiktoken

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.indices.splitters import TokenSplitter

EVIDENCE_MODE_TEXT = 0
EVIDENCE_MODE_TABLE = 1
EVIDENCE_MODE_CHATBOT = 2
EVIDENCE_MODE_FIGURE = 3


class PrepareEvidencePipeline(BaseComponent):
    """Prepare the evidence text from the list of retrieved documents

    This step usually happens after `DocumentRetrievalPipeline`.

    Args:
        trim_func: a callback function or a BaseComponent, that splits a large
            chunk of text into smaller ones. The first one will be retained.
    """

    max_context_length: int = 32000
    trim_func: TokenSplitter | None = None

    def run(self, docs: list[RetrievedDocument]) -> Document:
        evidence = ""
        images = []
        table_found = 0
        evidence_modes = []
        contributions: list[dict] = []

        def add_contribution(
            retrieved_item: RetrievedDocument,
            content: str,
            formatted_content: str,
            exclusion_reason: str = "",
        ) -> None:
            """Append evidence while recording passive context-packing diagnostics."""

            start = len(evidence)
            if not exclusion_reason:
                evidence_parts.append(formatted_content)
            end = start + (len(formatted_content) if not exclusion_reason else 0)
            contributions.append(
                {
                    "rank": len(contributions) + 1,
                    "doc_id": retrieved_item.doc_id,
                    "source": retrieved_item.metadata.get(
                        "file_name", retrieved_item.metadata.get("filename", "")
                    ),
                    "page": retrieved_item.metadata.get("page_label"),
                    "document_type": retrieved_item.metadata.get("type", "text"),
                    "content_chars": len(content),
                    "evidence_start": start,
                    "evidence_end": end,
                    "appended": not exclusion_reason,
                    "exclusion_reason": exclusion_reason,
                }
            )

        # Keeping additions in a list makes character boundaries deterministic for
        # diagnostics while producing the same concatenated evidence string.
        evidence_parts: list[str] = []

        evidence_trim_func = (
            self.trim_func
            if self.trim_func
            else TokenSplitter(
                chunk_size=self.max_context_length,
                chunk_overlap=0,
                separator=" ",
                tokenizer=partial(
                    tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
                    allowed_special=set(),
                    disallowed_special="all",
                ),
            )
        )

        for _, retrieved_item in enumerate(docs):
            evidence = "".join(evidence_parts)
            retrieved_content = ""
            page = retrieved_item.metadata.get("page_label", None)
            source = filename = retrieved_item.metadata.get("file_name", "-")
            if page:
                source += f" (Page {page})"
            if retrieved_item.metadata.get("type", "") == "table":
                evidence_modes.append(EVIDENCE_MODE_TABLE)
                if table_found < 5:
                    retrieved_content = retrieved_item.metadata.get(
                        "table_origin", retrieved_item.text
                    )
                    if retrieved_content not in evidence:
                        table_found += 1
                        formatted_content = (
                            f"<br><b>Table from {source}</b>\n"
                            + retrieved_content
                            + "\n<br>"
                        )
                        add_contribution(
                            retrieved_item, retrieved_content, formatted_content
                        )
                    else:
                        add_contribution(
                            retrieved_item, retrieved_content, "", "duplicate"
                        )
                else:
                    add_contribution(
                        retrieved_item, retrieved_item.text, "", "table_limit"
                    )
            elif retrieved_item.metadata.get("type", "") == "chatbot":
                evidence_modes.append(EVIDENCE_MODE_CHATBOT)
                retrieved_content = retrieved_item.metadata["window"]
                formatted_content = (
                    f"<br><b>Chatbot scenario from {filename} (Row {page})</b>\n"
                    + retrieved_content
                    + "\n<br>"
                )
                add_contribution(retrieved_item, retrieved_content, formatted_content)
            elif retrieved_item.metadata.get("type", "") == "image":
                evidence_modes.append(EVIDENCE_MODE_FIGURE)
                retrieved_content = retrieved_item.metadata.get("image_origin", "")
                retrieved_caption = html.escape(retrieved_item.get_content())
                formatted_content = (
                    f"<br><b>Figure from {source}</b>\n"
                    + "<img width='85%' src='<src>' "
                    + f"alt='{retrieved_caption}'/>"
                    + "\n<br>"
                )
                add_contribution(retrieved_item, retrieved_caption, formatted_content)
                images.append(retrieved_content)
            else:
                if "window" in retrieved_item.metadata:
                    retrieved_content = retrieved_item.metadata["window"]
                else:
                    retrieved_content = retrieved_item.text
                retrieved_content = retrieved_content.replace("\n", " ")
                if retrieved_content not in evidence:
                    formatted_content = (
                        f"<br><b>Content from {source}: </b> "
                        + retrieved_content
                        + " \n<br>"
                    )
                    add_contribution(
                        retrieved_item, retrieved_content, formatted_content
                    )
                else:
                    add_contribution(
                        retrieved_item, retrieved_content, "", "duplicate"
                    )

        evidence = "".join(evidence_parts)

        # resolve evidence mode
        evidence_mode = EVIDENCE_MODE_TEXT
        if EVIDENCE_MODE_FIGURE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_FIGURE
        elif EVIDENCE_MODE_TABLE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_TABLE

        # trim context by trim_len
        original_evidence = evidence
        print("len (original)", len(evidence))
        if evidence:
            texts = evidence_trim_func.run([Document(text=evidence)])
            evidence = texts[0].text
            print("len (trimmed)", len(evidence))

        exact_prefix = original_evidence.startswith(evidence)
        for contribution in contributions:
            if not contribution["appended"]:
                contribution["included_chars"] = 0
                contribution["fully_included"] = False
                contribution["truncated"] = False
                continue

            start = contribution["evidence_start"]
            end = contribution["evidence_end"]
            if exact_prefix:
                included_chars = max(0, min(end, len(evidence)) - start)
            else:
                # Token splitters normally retain an exact prefix. If an installed
                # version normalizes it, use a conservative presence check.
                segment = original_evidence[start:end]
                included_chars = len(segment) if segment and segment in evidence else 0
            contribution["included_chars"] = included_chars
            contribution["fully_included"] = included_chars == end - start
            contribution["truncated"] = 0 < included_chars < end - start
            if included_chars == 0:
                contribution["exclusion_reason"] = "context_budget"
            elif contribution["truncated"]:
                contribution["exclusion_reason"] = "context_budget_partial"

        self._last_run_diagnostics = {
            "max_context_tokens": self.max_context_length,
            "original_evidence_chars": len(original_evidence),
            "final_evidence_chars": len(evidence),
            "trimmed": len(evidence) < len(original_evidence),
            "contributions": contributions,
        }

        return Document(content=(evidence_mode, evidence, images))
