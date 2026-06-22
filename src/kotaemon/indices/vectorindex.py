from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence, cast

from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore

from .base import BaseIndexing, BaseRetrieval
from .rankings import BaseReranking, LLMReranking

VECTOR_STORE_FNAME = "vectorstore"
DOC_STORE_FNAME = "docstore"
logger = logging.getLogger(__name__)


def _vector_log(message: str, level: int = logging.INFO) -> None:
    """Log vector indexing/retrieval progress to logger and terminal."""

    logger.log(level, message)
    print(f"[vector-index] {message}", flush=True)


class VectorIndexing(BaseIndexing):
    """Ingest the document, run through the embedding, and store the embedding in a
    vector store.

    This pipeline supports the following set of inputs:
        - List of documents
        - List of texts
    """

    cache_dir: Optional[str] = getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None)
    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    use_enriched_text_for_embedding: bool = False
    count_: int = 0

    def to_retrieval_pipeline(self, *args, **kwargs):
        """Convert the indexing pipeline to a retrieval pipeline"""
        return VectorRetrieval(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            embedding=self.embedding,
            **kwargs,
        )

    def prepare_chunk_export(self, file_name: str) -> None:
        """Reset cached chunk files for one newly indexed source."""

        if not self.cache_dir:
            return
        stem = Path(file_name).stem
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for path in cache_dir.glob(f"{stem}_*.md"):
            path.unlink()
        self.count_ = 0

    def write_chunk_to_file(self, docs: list[Document]):
        # save the chunks content into markdown format
        if self.cache_dir:
            file_name = docs[0].metadata.get("file_name")
            if not file_name:
                return

            file_name = Path(file_name)
            for i in range(len(docs)):
                markdown_content = ""
                if "page_label" in docs[i].metadata:
                    page_label = str(docs[i].metadata["page_label"])
                    markdown_content += f"Page label: {page_label}"
                if "file_name" in docs[i].metadata:
                    filename = docs[i].metadata["file_name"]
                    markdown_content += f"\nFile name: {filename}"
                if "section" in docs[i].metadata:
                    section = docs[i].metadata["section"]
                    markdown_content += f"\nSection: {section}"
                if "type" in docs[i].metadata:
                    if docs[i].metadata["type"] == "image":
                        image_origin = docs[i].metadata["image_origin"]
                        image_origin = f'<p><img src="{image_origin}"></p>'
                        markdown_content += f"\nImage origin: {image_origin}"
                if docs[i].text:
                    markdown_content += f"\ntext:\n{docs[i].text}"

                export_index = docs[i].metadata.get(
                    "ingestion_index", self.count_ + i
                )
                with open(
                    Path(self.cache_dir) / f"{file_name.stem}_{export_index}.md",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(markdown_content)

    def add_to_docstore(self, docs: list[Document]):
        if self.doc_store:
            _vector_log(f"Adding {len(docs)} documents to doc store")
            self.doc_store.add(docs)
            _vector_log(f"Added {len(docs)} documents to doc store")

    def add_to_vectorstore(self, docs: list[Document]):
        # in case we want to skip embedding
        if self.vector_store:
            start_time = time.time()
            _vector_log(
                f"Getting embeddings for {len(docs)} nodes with {self.embedding}"
            )

            # Call the embedding implementation directly instead of the inherited
            # theflow Function.__call__ wrapper. The wrapper adds diskcache-based
            # result caching and can block on stale/inter-process cache locks for
            # large transient document batches. run() preserves the embedding
            # algorithm/configuration and avoids caching upload-time payloads.
            embedding_docs = docs
            if self.use_enriched_text_for_embedding:
                embedding_docs = []
                for doc in docs:
                    enriched_text = (doc.metadata or {}).get("enriched_text")
                    if enriched_text:
                        embedding_docs.append(Document(doc, text=str(enriched_text)))
                    else:
                        embedding_docs.append(doc)
            embeddings = self.embedding.run(embedding_docs)

            _vector_log(
                f"Created {len(embeddings)} embeddings "
                f"in {time.time() - start_time:.2f}s"
            )
            _vector_log("Adding embeddings to vector store")
            self.vector_store.add(
                embeddings=embeddings,
                ids=[t.doc_id for t in docs],
            )
            _vector_log(f"Added {len(embeddings)} embeddings to vector store")

    def run(self, text: str | list[str] | Document | list[Document]):
        input_: list[Document] = []
        if not isinstance(text, list):
            text = [text]

        for item in cast(list, text):
            if isinstance(item, str):
                input_.append(Document(text=item, id_=str(uuid.uuid4())))
            elif isinstance(item, Document):
                input_.append(item)
            else:
                raise ValueError(
                    f"Invalid input type {type(item)}, should be str or Document"
                )

        self.add_to_vectorstore(input_)
        self.add_to_docstore(input_)
        self.write_chunk_to_file(input_)
        self.count_ += len(input_)


class VectorRetrieval(BaseRetrieval):
    """Retrieve list of documents from vector store"""

    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    rerankers: Sequence[BaseReranking] = []
    top_k: int = 5
    first_round_top_k_mult: int = 10
    retrieval_mode: str = "hybrid"  # vector, text, hybrid

    @staticmethod
    def _diversify_parent_chunks(
        documents: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """Avoid letting facts from one table consume the complete result set."""

        selected: list[RetrievedDocument] = []
        deferred: list[RetrievedDocument] = []
        per_parent: dict[str, int] = {}
        for document in documents:
            parent = str((document.metadata or {}).get("table_parent_id") or "")
            if parent and per_parent.get(parent, 0) >= 2:
                deferred.append(document)
                continue
            selected.append(document)
            if parent:
                per_parent[parent] = per_parent.get(parent, 0) + 1
        return (selected + deferred)[:top_k]

    @staticmethod
    def _diagnostic_snapshot(
        documents: list[RetrievedDocument], stage: str
    ) -> list[dict]:
        """Return a serializable snapshot without changing retrieval results."""

        snapshot = []
        for rank, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            snapshot.append(
                {
                    "stage": stage,
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "source": metadata.get(
                        "file_name", metadata.get("filename", "")
                    ),
                    "page": metadata.get("page_label"),
                    "document_type": metadata.get("type", "text"),
                    "score": doc.score,
                    "reranking_score": metadata.get("reranking_score"),
                    "llm_relevance_score": metadata.get(
                        "llm_trulens_score",
                        metadata.get("llm_reranking_score"),
                    ),
                    "text": doc.text or "",
                }
            )
        return snapshot

    def _filter_docs(
        self, documents: list[RetrievedDocument], top_k: int | None = None
    ):
        if top_k:
            documents = documents[:top_k]
        return documents

    def _query_vectorstore_with_backoff(
        self,
        embedding: list[float],
        requested_top_k: int,
        doc_ids: list[str] | None,
        **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Query Chroma-compatible stores without over-requesting filtered HNSW hits.

        Chroma's HNSW backend can raise ``Cannot return the results in a contiguous
        2D array`` when ``n_results`` is close to, or greater than, the number of
        vectors allowed by an ID/metadata filter. Cap the request to the known scope
        and retry with progressively smaller candidate counts for that specific
        backend error. Other runtime errors still propagate unchanged.
        """

        if doc_ids is not None and not doc_ids:
            return [], [], []

        query_top_k = requested_top_k
        if doc_ids is not None:
            query_top_k = min(query_top_k, len(doc_ids))
        query_top_k = max(1, query_top_k)
        attempts: list[int] = []

        while True:
            attempts.append(query_top_k)
            try:
                result = self.vector_store.query(
                    embedding=embedding,
                    top_k=query_top_k,
                    doc_ids=doc_ids,
                    **kwargs,
                )
                if hasattr(self, "_last_retrieval_trace"):
                    self._last_retrieval_trace["vector_query_attempts"] = attempts
                    self._last_retrieval_trace["first_round_top_k_used"] = query_top_k
                return result
            except RuntimeError as exc:
                message = str(exc).lower()
                # Chroma 0.5.x misspells "contiguous" as "contigious" in
                # some releases. Match the stable parts of its HNSW message.
                is_hnsw_capacity_error = (
                    "return the results" in message
                    and "2d array" in message
                    and "too small" in message
                    and ("ef" in message or " m " in f" {message} ")
                )
                if not is_hnsw_capacity_error or query_top_k <= 1:
                    raise

                next_top_k = max(1, query_top_k // 2)
                _vector_log(
                    "Vector store could not satisfy filtered top_k="
                    f"{query_top_k}; retrying with top_k={next_top_k}",
                    logging.WARNING,
                )
                query_top_k = next_top_k

    def run(
        self, text: str | Document, top_k: Optional[int] = None, **kwargs
    ) -> list[RetrievedDocument]:
        """Retrieve a list of documents from vector store

        Args:
            text: the text to retrieve similar documents
            top_k: number of top similar documents to return

        Returns:
            list[RetrievedDocument]: list of retrieved documents
        """
        if top_k is None:
            top_k = self.top_k

        # Evaluation tooling reads this after a call. It is deliberately passive:
        # no retrieval inputs or outputs are changed by collecting the trace.
        self._last_retrieval_trace = {
            "mode": self.retrieval_mode,
            "requested_top_k": top_k,
            "first_round_top_k_requested": None,
            "stages": [],
        }

        do_extend = kwargs.pop("do_extend", False)
        thumbnail_count = kwargs.pop("thumbnail_count", 3)

        if do_extend:
            top_k_first_round = top_k * self.first_round_top_k_mult
        else:
            top_k_first_round = top_k

        scope = kwargs.pop("scope", None)
        if scope is not None:
            top_k_first_round = min(top_k_first_round, len(scope))
        self._last_retrieval_trace["first_round_top_k_requested"] = top_k_first_round

        if self.doc_store is None:
            raise ValueError(
                "doc_store is not provided. Please provide a doc_store to "
                "retrieve the documents"
            )

        result: list[RetrievedDocument] = []
        # TODO: should declare scope directly in the run params
        emb: list[float]

        _vector_log(
            f"Retrieval started: mode={self.retrieval_mode}, top_k={top_k}, "
            f"first_round_top_k={top_k_first_round}, scope={len(scope) if scope else 0}"
        )

        if self.retrieval_mode == "vector":
            start_time = time.time()
            _vector_log("Getting query embedding")
            emb = self.embedding.run(text)[0].embedding
            _vector_log(f"Query embedding ready in {time.time() - start_time:.2f}s")
            _, scores, ids = self._query_vectorstore_with_backoff(
                embedding=emb,
                requested_top_k=top_k_first_round,
                doc_ids=scope,
                **kwargs,
            )
            docs = self.doc_store.get(ids)
            result = [
                RetrievedDocument(**doc.to_dict(), score=score)
                for doc, score in zip(docs, scores)
            ]
        elif self.retrieval_mode == "text":
            query = text.text if isinstance(text, Document) else text
            docs = []
            if scope:
                docs = self.doc_store.query(
                    query, top_k=top_k_first_round, doc_ids=scope
                )
            result = [RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs]
        elif self.retrieval_mode == "hybrid":
            # similarity search section
            start_time = time.time()
            _vector_log("Getting query embedding")
            emb = self.embedding.run(text)[0].embedding
            _vector_log(f"Query embedding ready in {time.time() - start_time:.2f}s")
            vs_docs: list[RetrievedDocument] = []
            vs_ids: list[str] = []
            vs_scores: list[float] = []

            def query_vectorstore():
                nonlocal vs_docs
                nonlocal vs_scores
                nonlocal vs_ids

                assert self.doc_store is not None
                _, vs_scores, vs_ids = self._query_vectorstore_with_backoff(
                    embedding=emb,
                    requested_top_k=top_k_first_round,
                    doc_ids=scope,
                    **kwargs,
                )
                if vs_ids:
                    vs_docs = self.doc_store.get(vs_ids)

            # full-text search section
            ds_docs: list[RetrievedDocument] = []

            def query_docstore():
                nonlocal ds_docs

                assert self.doc_store is not None
                query = text.text if isinstance(text, Document) else text
                if scope:
                    ds_docs = self.doc_store.query(
                        query, top_k=top_k_first_round, doc_ids=scope
                    )

            vs_query_thread = threading.Thread(target=query_vectorstore)
            ds_query_thread = threading.Thread(target=query_docstore)

            _vector_log("Starting hybrid vector/docstore queries")
            query_start = time.time()
            vs_query_thread.start()
            ds_query_thread.start()

            vs_query_thread.join()
            ds_query_thread.join()
            _vector_log(
                f"Hybrid vector/docstore queries finished "
                f"in {time.time() - query_start:.2f}s"
            )

            result = [
                RetrievedDocument(**doc.to_dict(), score=-1.0)
                for doc in ds_docs
                if doc not in vs_ids
            ]
            result += [
                RetrievedDocument(**doc.to_dict(), score=score)
                for doc, score in zip(vs_docs, vs_scores)
            ]
            _vector_log(f"Got {len(vs_docs)} from vectorstore")
            _vector_log(f"Got {len(ds_docs)} from docstore")

        self._last_retrieval_trace["stages"].append(
            {
                "name": "initial",
                "documents": self._diagnostic_snapshot(result, "initial"),
            }
        )

        # use additional reranker to re-order the document list
        if self.rerankers and text:
            for reranker_idx, reranker in enumerate(self.rerankers, start=1):
                # if reranker is LLMReranking, limit the document with top_k items only
                if isinstance(reranker, LLMReranking):
                    result = self._filter_docs(result, top_k=top_k)
                rerank_start = time.time()
                _vector_log(f"Running reranker {reranker}")
                result = reranker.run(documents=result, query=text)
                _vector_log(
                    f"Reranker returned {len(result)} docs "
                    f"in {time.time() - rerank_start:.2f}s"
                )
                stage_name = f"reranker_{reranker_idx}_{type(reranker).__name__}"
                self._last_retrieval_trace["stages"].append(
                    {
                        "name": stage_name,
                        "documents": self._diagnostic_snapshot(result, stage_name),
                    }
                )

        result = self._diversify_parent_chunks(result, top_k=top_k)
        _vector_log(f"Got raw {len(result)} retrieved documents")

        # add page thumbnails to the result if exists
        thumbnail_doc_ids: set[str] = set()
        # we should copy the text from retrieved text chunk
        # to the thumbnail to get relevant LLM score correctly
        text_thumbnail_docs: dict[str, RetrievedDocument] = {}

        non_thumbnail_docs = []
        raw_thumbnail_docs = []
        for doc in result:
            if doc.metadata.get("type") == "thumbnail":
                # change type to image to display on UI
                doc.metadata["type"] = "image"
                raw_thumbnail_docs.append(doc)
                continue
            if (
                "thumbnail_doc_id" in doc.metadata
                and len(thumbnail_doc_ids) < thumbnail_count
            ):
                thumbnail_id = doc.metadata["thumbnail_doc_id"]
                thumbnail_doc_ids.add(thumbnail_id)
                text_thumbnail_docs[thumbnail_id] = doc
            else:
                non_thumbnail_docs.append(doc)

        linked_thumbnail_docs = self.doc_store.get(list(thumbnail_doc_ids))
        _vector_log(
            f"thumbnail docs {len(linked_thumbnail_docs)}; "
            f"non-thumbnail docs {len(non_thumbnail_docs)}; "
            f"raw-thumbnail docs {len(raw_thumbnail_docs)}"
        )
        additional_docs = []

        for thumbnail_doc in linked_thumbnail_docs:
            text_doc = text_thumbnail_docs[thumbnail_doc.doc_id]
            doc_dict = thumbnail_doc.to_dict()
            doc_dict["_id"] = text_doc.doc_id
            doc_dict["content"] = text_doc.content
            doc_dict["metadata"]["type"] = "image"
            for key in text_doc.metadata:
                if key not in doc_dict["metadata"]:
                    doc_dict["metadata"][key] = text_doc.metadata[key]

            additional_docs.append(RetrievedDocument(**doc_dict, score=text_doc.score))

        result = additional_docs + non_thumbnail_docs

        if not result:
            # return output from raw retrieved thumbnails
            result = self._filter_docs(raw_thumbnail_docs, top_k=thumbnail_count)

        self._last_retrieval_trace["stages"].append(
            {
                "name": "final",
                "documents": self._diagnostic_snapshot(result, "final"),
            }
        )

        return result


class TextVectorQA(BaseComponent):
    retrieving_pipeline: BaseRetrieval
    qa_pipeline: BaseComponent

    def run(self, question, **kwargs):
        retrieved_documents = self.retrieving_pipeline.run(question, **kwargs)
        return self.qa_pipeline.run(question, retrieved_documents, **kwargs)
